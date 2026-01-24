#define _USE_MATH_DEFINES
#include <cmath>
#include <torch/torch.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <wcslib/wcsfix.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// GPU kernel for batch WCS transformations (when CUDA is available)
// REMOVED in Refactor

namespace torchfits {

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

        // Apply standard corrections (e.g. date formats, TPV aliases, etc.)
        int stat_fix[WCSFIX_NWCS]; // Status return for wcsfix
        // ctrl=1 (CDCXX), naxis=0 (default)
        // Note: checking return status might be noisy for minor fixes, but usually safe to ignore usually unless critical
        wcsfix(1, 0, wcs_, stat_fix);

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
        // precompute_matrices(); // Removed
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
    // double cd_matrix_[4];
    // double cd_matrix_inv_[4];
    // bool is_linear_wcs_ = false;
    
    // void precompute_matrices() { ... } // Removed

        // Only for 2D
    
public:
    // Getters for WCS properties
    
    // CTYPE (Coordinate types)
    std::vector<std::string> ctype() const {
        std::vector<std::string> result;
        for (int i = 0; i < wcs_->naxis; i++) {
            result.push_back(std::string(wcs_->ctype[i]));
        }
        return result;
    }
    
    // CUNIT (Coordinate units)
    std::vector<std::string> cunit() const {
        std::vector<std::string> result;
        for (int i = 0; i < wcs_->naxis; i++) {
            result.push_back(std::string(wcs_->cunit[i]));
        }
        return result;
    }

    // PC Matrix (Linear transformation) - returned as 1D tensor for simplicity
    torch::Tensor pc() const {
        // wcs->pc is naxis * naxis
        int n = wcs_->naxis;
        auto tensor = torch::empty({n, n}, torch::kFloat64);
        double* data = tensor.data_ptr<double>();
        for (int i = 0; i < n * n; i++) {
            data[i] = wcs_->pc[i];
        }
        return tensor;
    }

    // CD Matrix (if present, otherwise derived/unused by wcslib normalized PC)
    // Note: wcslib often normalizes CD into PC + CDELT.
    // We expose PC as the primary linear matrix.

    double lonpole() const { return wcs_->lonpole; }
    double latpole() const { return wcs_->latpole; }

};
}