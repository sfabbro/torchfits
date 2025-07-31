#ifndef TORCHFITS_WCS_UTILS_H
#define TORCHFITS_WCS_UTILS_H

#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <vector>
#include <map>
#include <string>
#include <memory>

// Forward declaration of the wcsprm struct.
struct wcsprm;

// Type alias for a unique_ptr to a wcsprm struct with a custom deleter.
// This ensures that the memory allocated by wcslib is properly freed.
using WcsUniquePtr = std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>;

/**
 * @brief Read WCS information from an open FITS file's header.
 *
 * @param fptr A pointer to an open fitsfile.
 * @return A WcsUniquePtr managing the wcsprm struct.
 * @throws std::runtime_error on failure to read or parse the WCS info.
 */
WcsUniquePtr read_wcs_from_header(fitsfile* fptr);

/**
 * @brief Create a WCS structure from a map of header key-value pairs.
 *
 * @param header A map representing the FITS header.
 * @param throw_on_error If true, throw an exception on failure. If false, return nullptr.
 * @return A WcsUniquePtr managing the wcsprm struct, or nullptr on failure if throw_on_error is false.
 */
WcsUniquePtr create_wcs_from_header(const std::map<std::string, std::string>& header, bool throw_on_error = true);

/**
 * @brief Convert world coordinates to pixel coordinates.
 *
 * @param world_coords A 2D tensor of world coordinates [N, Dims].
 * @param header A map representing the FITS header with WCS info.
 * @return A tuple containing a tensor of pixel coordinates and a tensor of status flags.
 */
std::tuple<torch::Tensor, torch::Tensor> world_to_pixel(
    torch::Tensor world_coords,
    const std::map<std::string, std::string>& header
);

/**
 * @brief Convert pixel coordinates to world coordinates.
 *
 * @param pixel_coords A 2D tensor of pixel coordinates [N, Dims].
 * @param header A map representing the FITS header with WCS info.
 * @return A tuple containing a tensor of world coordinates and a tensor of status flags.
 */
std::tuple<torch::Tensor, torch::Tensor> pixel_to_world(
    torch::Tensor pixel_coords,
    const std::map<std::string, std::string>& header
);

#endif // TORCHFITS_WCS_UTILS_H
