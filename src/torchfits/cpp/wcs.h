/**
 * WCS (World Coordinate System) engine header
 * 
 * High-performance coordinate transformations using wcslib with batch processing
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <map>

// Forward declaration
struct WcsHandle;

// WCS initialization
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