/**
 * WCS (World Coordinate System) engine header
 * 
 * High-performance coordinate transformations using wcslib with batch processing
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <map>
#include <unordered_map>

namespace torchfits {

// WCS class declaration
class WCS {
public:
    WCS(const std::unordered_map<std::string, std::string>& header);
    ~WCS();
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels);
    torch::Tensor world_to_pixel(const torch::Tensor& coords);
    torch::Tensor get_footprint();
    std::string test_method();
    
    int naxis();
    torch::Tensor crpix();
    torch::Tensor crval();
    torch::Tensor cdelt();
    
private:
    void* wcsprm_;  // wcsprm struct from wcslib
    int naxis_;
    bool is_initialized_;
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