#ifndef TORCHFITS_WCS_UTILS_H
#define TORCHFITS_WCS_UTILS_H

#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <functional>

// --- WCS Function Prototypes ---

// Function to create WCS from FITS file
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);

// Function to create WCS from header map
std::unique_ptr<wcsprm, std::function<void(wcsprm*)>> read_wcs_from_header_map(
    const std::map<std::string, std::string>& header);

// Function to create WCS from header (alias with different return type)
std::unique_ptr<wcsprm> create_wcs_from_header(
    const std::map<std::string, std::string>& header, 
    bool throw_on_error = false);

// World to pixel coordinate conversion
std::tuple<torch::Tensor, torch::Tensor> world_to_pixel(
    torch::Tensor world_coords,
    std::map<std::string, std::string> header
);

// Pixel to world coordinate conversion
std::tuple<torch::Tensor, torch::Tensor> pixel_to_world(
    torch::Tensor pixel_coords,
    std::map<std::string, std::string> header
);

#endif // TORCHFITS_WCS_UTILS_H