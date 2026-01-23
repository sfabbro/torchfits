#define _USE_MATH_DEFINES
#include <cmath>
#include <torch/torch.h>
#ifdef HAS_WCSLIB
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#endif
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef TORCH_CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <torch/extension.h>
#endif

namespace torchfits {

// GPU kernel for batch WCS transformations (when CUDA is available)
#ifdef TORCH_CUDA_AVAILABLE
__device__ void gpu_linear_transform(
    const double* input, double* output,
    const double* cd_matrix, const double* crval, const double* crpix,
    int ncoord) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncoord) return;
    
    // Simple linear transformation for TAN projection
    double x = input[idx * 2] - crpix[0];
    double y = input[idx * 2 + 1] - crpix[1];
    
    output[idx * 2] = crval[0] + cd_matrix[0] * x + cd_matrix[1] * y;
    output[idx * 2 + 1] = crval[1] + cd_matrix[2] * x + cd_matrix[3] * y;
}

__global__ void pixel_to_world_kernel(
    const double* pixels, double* world,
    const double* cd_matrix, const double* crval, const double* crpix,
    int ncoord) {
    gpu_linear_transform(pixels, world, cd_matrix, crval, crpix, ncoord);
}

__global__ void world_to_pixel_kernel(
    const double* world, double* pixels,
    const double* cd_matrix_inv, const double* crval, const double* crpix,
    int ncoord) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncoord) return;
    
    // Inverse transformation
    double ra = world[idx * 2] - crval[0];
    double dec = world[idx * 2 + 1] - crval[1];
    
    pixels[idx * 2] = crpix[0] + cd_matrix_inv[0] * ra + cd_matrix_inv[1] * dec;
    pixels[idx * 2 + 1] = crpix[1] + cd_matrix_inv[2] * ra + cd_matrix_inv[3] * dec;
}
#endif

#ifdef HAS_WCSLIB
class WCS {
private:
    struct wcsprm* wcs_;
    int nwcs_;  // Store the actual number of WCS structs returned by wcspih

public:
    WCS(const std::unordered_map<std::string, std::string>& header) : wcs_(nullptr), nwcs_(0) {
        int nkey = header.size();
        char* h = (char*)malloc(nkey * 81); // Allocate one extra byte for safety
        if (!h) {
            throw std::runtime_error("Failed to allocate memory for header");
        }
        
        char* p = h;
        for (const auto& [key, value] : header) {
            snprintf(p, 81, "%-8s= %-70s", key.c_str(), value.c_str());
            p += 80;
        }

        int nreject;
        int ctrl = 0;
        struct wcsprm* wcs_array = nullptr;
        int status = wcspih(h, nkey, WCSHDR_all, ctrl, &nreject, &nwcs_, &wcs_array);
        free(h); // Free the header buffer immediately after use
        
        if (status != 0 || nwcs_ <= 0) {
            // Clean up and throw error
            if (wcs_array) {
                wcsvfree(&nwcs_, &wcs_array);
            }
            if (status != 0) {
                throw std::runtime_error("Failed to parse WCS from header");
            } else {
                throw std::runtime_error("No WCS found in header");
            }
        }
        
        // Use the first wcs struct
        wcs_ = wcs_array;

        // Shift CRPIX by -1.0 to support 0-based indexing (PyTorch/Python convention)
        // FITS uses 1-based indexing, so CRPIX is 1-based.
        // We want: world = f(pixel_0based)
        // wcslib computes: world = f(pixel_input - (crpix - 1) + 1 - 1) ... ?
        // wcslib computes: intermediate = (pixel_input - crpix)
        // We want intermediate = (pixel_0based - (crpix_1based - 1))
        // So we set wcs->crpix = crpix_1based - 1.
        for (int i = 0; i < wcs_->naxis; i++) {
            wcs_->crpix[i] -= 1.0;
        }

        status = wcsset(wcs_);
        if (status != 0) {
            // Clean up and throw error
            // Note: we've already transferred ownership to wcs_, so we need to free it properly
            if (wcs_) {
                wcsvfree(&nwcs_, &wcs_);
                wcs_ = nullptr;
                nwcs_ = 0;
            }
            throw std::runtime_error("wcsset failed");
        }
        
        // Pre-compute matrices for GPU optimization
        precompute_matrices();
    }
    
    ~WCS() {
        if (wcs_) {
            // Use the actual number of WCS structs that were allocated
            wcsvfree(&nwcs_, &wcs_);
            wcs_ = nullptr;
            nwcs_ = 0;
        }
    }
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels) {
        auto device = pixels.device();
        auto dtype = pixels.dtype();
        int naxis = wcs_->naxis;
        
        // GPU-optimized path for CUDA tensors (only for 2D for now)
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection() && naxis == 2) {
            return pixel_to_world_gpu(pixels);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_pixels = pixels.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_pixels.sizes();
        int ncoord = shape[0];
        
        // Ensure input has correct shape (N, naxis)
        if (shape.size() < 2 || shape[1] != naxis) {
            throw std::runtime_error("Input pixels must have shape (N, " + std::to_string(naxis) + ")");
        }
        
        auto world = torch::empty_like(cpu_pixels);
        
        double* pixcrd = cpu_pixels.data_ptr<double>();
        double* worldcrd = world.data_ptr<double>();
        
        // Temporarily disable OpenMP path to debug
        // #ifdef HAS_OPENMP
        // if (ncoord > 1000) {
        //     return pixel_to_world_parallel(cpu_pixels, world, ncoord);
        // }
        // #endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * naxis * sizeof(double));
        if (!imgcrd) throw std::runtime_error("Failed to allocate memory for imgcrd");
        
        double* phi = (double*)malloc(ncoord * sizeof(double));
        if (!phi) { free(imgcrd); throw std::runtime_error("Failed to allocate memory for phi"); }
        
        double* theta = (double*)malloc(ncoord * sizeof(double));
        if (!theta) { free(imgcrd); free(phi); throw std::runtime_error("Failed to allocate memory for theta"); }
        
        int* stat = (int*)malloc(ncoord * sizeof(int));
        if (!stat) { free(imgcrd); free(phi); free(theta); throw std::runtime_error("Failed to allocate memory for stat"); }
        
        int wcs_status = wcsp2s(wcs_, ncoord, naxis, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        if (wcs_status != 0) {
            free(imgcrd); free(phi); free(theta); free(stat);
            throw std::runtime_error("WCS transformation failed");
        }
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return world.to(dtype).to(device);
    }
    
    torch::Tensor world_to_pixel(const torch::Tensor& world) {
        auto device = world.device();
        auto dtype = world.dtype();
        int naxis = wcs_->naxis;
        
        // GPU-optimized path for CUDA tensors (only for 2D for now)
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection() && naxis == 2) {
            return world_to_pixel_gpu(world);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_world = world.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_world.sizes();
        int ncoord = shape[0];
        
        // Ensure input has correct shape (N, naxis)
        if (shape.size() < 2 || shape[1] != naxis) {
            throw std::runtime_error("Input world coordinates must have shape (N, " + std::to_string(naxis) + ")");
        }
        
        auto pixels = torch::empty_like(cpu_world);
        
        double* worldcrd = cpu_world.data_ptr<double>();
        double* pixcrd = pixels.data_ptr<double>();
        
        // Temporarily disable OpenMP path to debug
        // #ifdef HAS_OPENMP
        // if (ncoord > 1000) {
        //     return world_to_pixel_parallel(cpu_world, pixels, ncoord);
        // }
        // #endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * naxis * sizeof(double));
        if (!imgcrd) throw std::runtime_error("Failed to allocate memory for imgcrd");
        
        double* phi = (double*)malloc(ncoord * sizeof(double));
        if (!phi) { free(imgcrd); throw std::runtime_error("Failed to allocate memory for phi"); }
        
        double* theta = (double*)malloc(ncoord * sizeof(double));
        if (!theta) { free(imgcrd); free(phi); throw std::runtime_error("Failed to allocate memory for theta"); }
        
        int* stat = (int*)malloc(ncoord * sizeof(int));
        if (!stat) { free(imgcrd); free(phi); free(theta); throw std::runtime_error("Failed to allocate memory for stat"); }
        
        int wcs_status = wcss2p(wcs_, ncoord, naxis, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        if (wcs_status != 0) {
            free(imgcrd); free(phi); free(theta); free(stat);
            throw std::runtime_error("WCS transformation failed");
        }
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return pixels.to(dtype).to(device);
    }
    
    torch::Tensor get_footprint() {
        int naxis = wcs_->naxis;
        if (naxis != 2) {
            throw std::runtime_error("get_footprint only supported for 2D WCS");
        }
        auto corners = torch::empty({4, 2}, torch::kFloat64);
        double* data = corners.data_ptr<double>();
        
        data[0] = 0.5; data[1] = 0.5;
        data[2] = wcs_->crpix[0] * 2 - 0.5; data[3] = 0.5;
        data[4] = wcs_->crpix[0] * 2 - 0.5; data[5] = wcs_->crpix[1] * 2 - 0.5;
        data[6] = 0.5; data[7] = wcs_->crpix[1] * 2 - 0.5;
        
        return pixel_to_world(corners);
    }

    int naxis() const { return wcs_->naxis; }
    int test_method() const { return 42; }
    torch::Tensor crpix() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->crpix[i];
        return tensor;
    }
    torch::Tensor crval() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->crval[i];
        return tensor;
    }
    torch::Tensor cdelt() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->cdelt[i];
        return tensor;
    }

private:
    // Cached matrices for GPU optimization
    double cd_matrix_[4];
    double cd_matrix_inv_[4];
    bool is_linear_wcs_ = false;
    
    void precompute_matrices() {
        // Only for 2D
        if (wcs_->naxis != 2) return;
        
        // Check if this is a simple linear WCS (TAN projection)
        // FIXME: TAN projection is NOT linear in RA/DEC. 
        // The current implementation treats it as Cartesian which is wrong.
        // Disabling optimization for now to ensure correctness via wcslib.
        if (false && wcs_->ctype[0][4] == '-' && wcs_->ctype[0][5] == 'T' && 
            wcs_->ctype[0][6] == 'A' && wcs_->ctype[0][7] == 'N') {
            
            // Extract CD matrix or compute from CDELT/CROTA
            if (wcs_->cd) {
                cd_matrix_[0] = wcs_->cd[0];
                cd_matrix_[1] = wcs_->cd[1]; 
                cd_matrix_[2] = wcs_->cd[2];
                cd_matrix_[3] = wcs_->cd[3];
            } else {
                double cdelt1 = wcs_->cdelt[0];
                double cdelt2 = wcs_->cdelt[1];
                double crota = wcs_->crota[1] * M_PI / 180.0; // Convert to radians
                
                cd_matrix_[0] = cdelt1 * cos(crota);
                cd_matrix_[1] = -cdelt2 * sin(crota);
                cd_matrix_[2] = cdelt1 * sin(crota);
                cd_matrix_[3] = cdelt2 * cos(crota);
            }
            
            // Compute inverse matrix
            double det = cd_matrix_[0] * cd_matrix_[3] - cd_matrix_[1] * cd_matrix_[2];
            if (abs(det) > 1e-12) {
                cd_matrix_inv_[0] = cd_matrix_[3] / det;
                cd_matrix_inv_[1] = -cd_matrix_[1] / det;
                cd_matrix_inv_[2] = -cd_matrix_[2] / det;
                cd_matrix_inv_[3] = cd_matrix_[0] / det;
                is_linear_wcs_ = true;
            }
        }
    }
    
    bool is_simple_projection() const {
        return is_linear_wcs_;
    }
    
#ifdef TORCH_CUDA_AVAILABLE
    torch::Tensor pixel_to_world_gpu(const torch::Tensor& pixels) {
        auto shape = pixels.sizes();
        int ncoord = shape[0];
        auto world = torch::empty_like(pixels);
        
        auto pixels_double = pixels.to(torch::kFloat64);
        auto world_double = world.to(torch::kFloat64);
        
        // Transfer WCS parameters to GPU
        auto cd_matrix_gpu = torch::from_blob(cd_matrix_, {4}, torch::kFloat64).to(pixels.device());
        auto crval_gpu = torch::tensor({wcs_->crval[0], wcs_->crval[1]}, torch::kFloat64).to(pixels.device());
        auto crpix_gpu = torch::tensor({wcs_->crpix[0], wcs_->crpix[1]}, torch::kFloat64).to(pixels.device());
        
        // Launch CUDA kernel
        int threads = 256;
        int blocks = (ncoord + threads - 1) / threads;
        
        pixel_to_world_kernel<<<blocks, threads>>>(
            pixels_double.data_ptr<double>(),
            world_double.data_ptr<double>(),
            cd_matrix_gpu.data_ptr<double>(),
            crval_gpu.data_ptr<double>(),
            crpix_gpu.data_ptr<double>(),
            ncoord
        );
        
        cudaDeviceSynchronize();
        return world_double.to(pixels.dtype());
    }
    
    torch::Tensor world_to_pixel_gpu(const torch::Tensor& world) {
        auto shape = world.sizes();
        int ncoord = shape[0];
        auto pixels = torch::empty_like(world);
        
        auto world_double = world.to(torch::kFloat64);
        auto pixels_double = pixels.to(torch::kFloat64);
        
        // Transfer WCS parameters to GPU
        auto cd_matrix_inv_gpu = torch::from_blob(cd_matrix_inv_, {4}, torch::kFloat64).to(world.device());
        auto crval_gpu = torch::tensor({wcs_->crval[0], wcs_->crval[1]}, torch::kFloat64).to(world.device());
        auto crpix_gpu = torch::tensor({wcs_->crpix[0], wcs_->crpix[1]}, torch::kFloat64).to(world.device());
        
        // Launch CUDA kernel
        int threads = 256;
        int blocks = (ncoord + threads - 1) / threads;
        
        world_to_pixel_kernel<<<blocks, threads>>>(
            world_double.data_ptr<double>(),
            pixels_double.data_ptr<double>(),
            cd_matrix_inv_gpu.data_ptr<double>(),
            crval_gpu.data_ptr<double>(),
            crpix_gpu.data_ptr<double>(),
            ncoord
        );
        
        cudaDeviceSynchronize();
        return pixels_double.to(world.dtype());
    }
#endif
    
#ifdef HAS_OPENMP
    torch::Tensor pixel_to_world_parallel(const torch::Tensor& cpu_pixels, torch::Tensor& world, int ncoord) {
        double* pixcrd = cpu_pixels.data_ptr<double>();
        double* worldcrd = world.data_ptr<double>();
        int naxis = wcs_->naxis;
        
        // Parallel processing using OpenMP
        #pragma omp parallel
        {
            // Thread-local storage for wcslib workspace
            // Each thread processes a subset of coordinates
            double* imgcrd = (double*)malloc(naxis * sizeof(double));
            double* phi = (double*)malloc(sizeof(double));
            double* theta = (double*)malloc(sizeof(double));
            int* stat = (int*)malloc(sizeof(int));
            
            #pragma omp for
            for (int i = 0; i < ncoord; i++) {
                wcsp2s(wcs_, 1, naxis, &pixcrd[i*naxis], imgcrd, phi, theta, &worldcrd[i*naxis], stat);
            }
            
            free(imgcrd);
            free(phi);
            free(theta);
            free(stat);
        }
        
        return world;
    }
    
    torch::Tensor world_to_pixel_parallel(const torch::Tensor& cpu_world, torch::Tensor& pixels, int ncoord) {
        double* worldcrd = cpu_world.data_ptr<double>();
        double* pixcrd = pixels.data_ptr<double>();
        int naxis = wcs_->naxis;
        
        // Parallel processing using OpenMP
        #pragma omp parallel
        {
            // Thread-local storage for wcslib workspace
            // Each thread processes a subset of coordinates
            double* imgcrd = (double*)malloc(naxis * sizeof(double));
            double* phi = (double*)malloc(sizeof(double));
            double* theta = (double*)malloc(sizeof(double));
            int* stat = (int*)malloc(sizeof(int));
            
            #pragma omp for
            for (int i = 0; i < ncoord; i++) {
                wcss2p(wcs_, 1, naxis, &worldcrd[i*naxis], phi, theta, imgcrd, &pixcrd[i*naxis], stat);
            }
            
            free(imgcrd);
            free(phi);
            free(theta);
            free(stat);
        }
        
        return pixels;
    }
#endif
};
#endif

}