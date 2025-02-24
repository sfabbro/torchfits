#ifndef WCS_UTILS_H
#define WCS_UTILS_H

#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <vector>
#include <map>
#include <string>
#include <memory>

// --- WCS Function Prototypes ---
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);
std::pair<torch::Tensor, torch::Tensor> world_to_pixel(const torch::Tensor& world_coords, const std::map<std::string, std::string>& header);
std::pair<torch::Tensor, torch::Tensor> pixel_to_world(const torch::Tensor& pixel_coords, const std::map<std::string, std::string>& header);

#endif //WCS_UTILS_H