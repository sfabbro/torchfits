#define _USE_MATH_DEFINES
#include <cmath>
#include <torch/torch.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
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

class WCS {
public:
    WCS(const std::unordered_map<std::string, std::string>& header) {
        int nkey = header.size();
        char* h = (char*)malloc(nkey * 80);
        char* p = h;
        for (const auto& [key, value] : header) {
            snprintf(p, 80, "%-8s= %-70s", key.c_str(), value.c_str());
            p += 80;
        }

        int nreject;
        int ctrl = 0;
        int nwcs;
        struct wcsprm* wcs_array;
        int status = wcspih(h, nkey, WCSHDR_all, ctrl, &nreject, &nwcs, &wcs_array);
        if (nwcs > 0) {
            wcs_ = wcs_array;
        } else {
            throw std::runtime_error("No WCS found in header");
        }
        free(h);

        if (status) {
            throw std::runtime_error("Failed to parse WCS from header");
        }

        status = wcsset(wcs_);
        if (status) {
            throw std::runtime_error("wcsset failed");
        }
        
        // Pre-compute matrices for GPU optimization
        precompute_matrices();
    }
    
    ~WCS() {
        if (wcs_) {
            wcsfree(wcs_);
            free(wcs_);
        }
    }
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels) {
        auto device = pixels.device();
        auto dtype = pixels.dtype();
        
        // GPU-optimized path for CUDA tensors
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection()) {
            return pixel_to_world_gpu(pixels);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_pixels = pixels.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_pixels.sizes();
        int ncoord = shape[0];
        auto world = torch::empty_like(cpu_pixels);
        
        double* pixcrd = cpu_pixels.data_ptr<double>();
        double* worldcrd = world.data_ptr<double>();
        
        // Use OpenMP for parallel processing on CPU
#ifdef HAS_OPENMP
        if (ncoord > 1000) {
            return pixel_to_world_parallel(cpu_pixels, world, ncoord);
        }
#endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
        double* phi = (double*)malloc(ncoord * sizeof(double));
        double* theta = (double*)malloc(ncoord * sizeof(double));
        int* stat = (int*)malloc(ncoord * sizeof(int));
        
        wcsp2s(wcs_, ncoord, 2, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return world.to(dtype).to(device);
    }
    
    torch::Tensor world_to_pixel(const torch::Tensor& world) {
        auto device = world.device();
        auto dtype = world.dtype();
        
        // GPU-optimized path for CUDA tensors
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection()) {
            return world_to_pixel_gpu(world);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_world = world.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_world.sizes();
        int ncoord = shape[0];
        auto pixels = torch::empty_like(cpu_world);
        
        double* worldcrd = cpu_world.data_ptr<double>();
        double* pixcrd = pixels.data_ptr<double>();
        
        // Use OpenMP for parallel processing on CPU
#ifdef HAS_OPENMP
        if (ncoord > 1000) {
            return world_to_pixel_parallel(cpu_world, pixels, ncoord);
        }
#endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
        double* phi = (double*)malloc(ncoord * sizeof(double));
        double* theta = (double*)malloc(ncoord * sizeof(double));
        int* stat = (int*)malloc(ncoord * sizeof(int));
        
        wcss2p(wcs_, ncoord, 2, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return pixels.to(dtype).to(device);
    }
    
    torch::Tensor get_footprint() {
        double corners[4][2] = {
            {0.5, 0.5},
            {wcs_->crpix[0] * 2 - 0.5, 0.5},
            {wcs_->crpix[0] * 2 - 0.5, wcs_->crpix[1] * 2 - 0.5},
            {0.5, wcs_->crpix[1] * 2 - 0.5}
        };
        
        auto corner_pixels = torch::from_blob(corners, {4, 2}, torch::kFloat64);
        return pixel_to_world(corner_pixels);
    }

    int naxis() const { return wcs_->naxis; }
    int test_method() const { return 42; }
    torch::Tensor crpix() const { 
        auto tensor = torch::empty({2}, torch::kFloat64);
        tensor[0] = wcs_->crpix[0];
        tensor[1] = wcs_->crpix[1];
        return tensor;
    }
    torch::Tensor crval() const { 
        auto tensor = torch::empty({2}, torch::kFloat64);
        tensor[0] = wcs_->crval[0];
        tensor[1] = wcs_->crval[1];
        return tensor;
    }
    torch::Tensor cdelt() const { 
        auto tensor = torch::empty({2}, torch::kFloat64);
        tensor[0] = wcs_->cdelt[0];
        tensor[1] = wcs_->cdelt[1];
        return tensor;
    }

private:
    struct wcsprm* wcs_;
    
    // Cached matrices for GPU optimization
    double cd_matrix_[4];
    double cd_matrix_inv_[4];
    bool is_linear_wcs_ = false;
    
    void precompute_matrices() {
        // Check if this is a simple linear WCS (TAN projection)
        if (wcs_->ctype[0][4] == '-' && wcs_->ctype[0][5] == 'T' && 
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
        
        // Parallel processing using OpenMP
        #pragma omp parallel
        {
            // Thread-local storage for wcslib workspace
            double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
            double* phi = (double*)malloc(ncoord * sizeof(double));
            double* theta = (double*)malloc(ncoord * sizeof(double));
            int* stat = (int*)malloc(ncoord * sizeof(int));
            
            #pragma omp for
            for (int i = 0; i < ncoord; i++) {
                wcsp2s(wcs_, 1, 2, &pixcrd[i*2], &imgcrd[i*2], &phi[i], &theta[i], &worldcrd[i*2], &stat[i]);
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
        
        // Parallel processing using OpenMP
        #pragma omp parallel
        {
            // Thread-local storage for wcslib workspace
            double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
            double* phi = (double*)malloc(ncoord * sizeof(double));
            double* theta = (double*)malloc(ncoord * sizeof(double));
            int* stat = (int*)malloc(ncoord * sizeof(int));
            
            #pragma omp for
            for (int i = 0; i < ncoord; i++) {
                wcss2p(wcs_, 1, 2, &worldcrd[i*2], &phi[i], &theta[i], &imgcrd[i*2], &pixcrd[i*2], &stat[i]);
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

}