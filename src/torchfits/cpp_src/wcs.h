/**
 * WCS (World Coordinate System) engine header
 * 
 * High-performance coordinate transformations using wcslib with batch processing
 */

#pragma once

#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#ifdef HAS_WCSLIB

// Forward declarations
namespace torch {
    class Tensor;
}

struct wcsprm;

namespace torchfits {

// WCS class declaration
class WCS {
public:
    WCS(const std::unordered_map<std::string, std::string>& header);
    ~WCS();
    
    // Disable copy constructor and assignment operator to prevent double free
    WCS(const WCS&) = delete;
    WCS& operator=(const WCS&) = delete;
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels);
    torch::Tensor world_to_pixel(const torch::Tensor& coords);
    torch::Tensor get_footprint();
    int test_method() const;
    
    int naxis() const;
    torch::Tensor crpix() const;
    torch::Tensor crval() const;
    torch::Tensor cdelt() const;
    
private:
    void precompute_matrices();
    bool is_simple_projection() const;
    
    struct wcsprm* wcs_;
    int nwcs_;
    
#ifdef TORCH_CUDA_AVAILABLE
    torch::Tensor pixel_to_world_gpu(const torch::Tensor& pixels);
    torch::Tensor world_to_pixel_gpu(const torch::Tensor& world);
    double cd_matrix_[4];
    double cd_matrix_inv_[4];
#endif

#ifdef HAS_OPENMP
    torch::Tensor pixel_to_world_parallel(const torch::Tensor& cpu_pixels, torch::Tensor& world, int ncoord);
    torch::Tensor world_to_pixel_parallel(const torch::Tensor& cpu_world, torch::Tensor& pixels, int ncoord);
#endif
    
    bool is_linear_wcs_;
    std::unordered_map<std::string, std::string> header_;
};

} // namespace torchfits

// Forward declaration
struct WcsHandle;

// Legacy C-style WCS functions for backward compatibility
WcsHandle* init_wcs_from_header(const std::map<std::string, std::string>& header);
void free_wcs_handle(WcsHandle* handle);

// Batch coordinate transformations
torch::Tensor pixel_to_world_batch(WcsHandle* handle, const torch::Tensor& pixels);
torch::Tensor world_to_pixel_batch(WcsHandle* handle, const torch::Tensor& coords);

// Utility functions
torch::Tensor get_footprint(WcsHandle* handle);

// Internal structure
struct WcsHandle {
    void* wcsprm;  // wcsprm struct from wcslib
    int naxis;
    bool is_initialized;
    
    WcsHandle() : wcsprm(nullptr), naxis(0), is_initialized(false) {}
};
#endif