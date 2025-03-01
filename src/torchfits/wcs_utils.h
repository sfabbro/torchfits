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

/**
 * @brief Read WCS information from a FITS file header
 * 
 * This function extracts WCS keywords from the current HDU of the FITS file,
 * constructs a wcsprm structure, and returns it wrapped in a unique_ptr.
 * 
 * Memory ownership: The returned unique_ptr owns the wcsprm structure and
 * will free it when the unique_ptr is destroyed.
 * 
 * @param fptr Pointer to an open FITS file
 * @return A unique_ptr to a wcsprm structure containing the WCS information
 * @throws std::runtime_error if WCS parsing or initialization fails
 */
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);

/**
 * @brief Create a WCS structure from a map of header key-value pairs
 * 
 * This function builds a FITS header from the provided map and creates
 * a wcsprm structure from it.
 * 
 * Memory ownership: The returned unique_ptr owns the wcsprm structure and
 * will free it using the provided custom deleter when destroyed.
 * 
 * @param header Map of header keywords to values
 * @return A unique_ptr with custom deleter containing the WCS structure
 * @throws std::runtime_error if header parsing or WCS initialization fails
 */
std::unique_ptr<wcsprm, std::function<void(wcsprm*)>> read_wcs_from_header_map(
    const std::map<std::string, std::string>& header);

/**
 * @brief Create a WCS structure from a map of header key-value pairs
 * 
 * Alternative interface to read_wcs_from_header_map that allows controlling
 * whether errors are thrown or returned as nullptr.
 * 
 * Memory ownership: The returned unique_ptr owns the wcsprm structure and
 * will free it when the unique_ptr is destroyed.
 * 
 * @param header Map of header keywords to values
 * @param throw_on_error If true, throw exceptions on error; if false, return nullptr
 * @return A unique_ptr to a wcsprm structure, or nullptr on error if throw_on_error is false
 * @throws std::runtime_error if throw_on_error is true and parsing or initialization fails
 */
std::unique_ptr<wcsprm> create_wcs_from_header(
    const std::map<std::string, std::string>& header, 
    bool throw_on_error = false);

/**
 * @brief Convert world coordinates to pixel coordinates
 * 
 * Takes world coordinates (RA/Dec or other coordinate system) and converts
 * them to pixel coordinates using the WCS information in the header.
 * 
 * @param world_coords Tensor of shape [N, 2+] with world coordinates
 * @param header Map of FITS header key-value pairs containing WCS information
 * @return Tuple of (pixel_coords, status_tensor) where pixel_coords is a tensor
 *         of shape [N, naxis] and status_tensor indicates success/failure per point
 * @throws std::runtime_error if the header does not contain valid WCS information
 */
std::tuple<torch::Tensor, torch::Tensor> world_to_pixel(
    torch::Tensor world_coords,
    std::map<std::string, std::string> header
);

/**
 * @brief Convert pixel coordinates to world coordinates
 * 
 * Takes pixel coordinates and converts them to world coordinates (RA/Dec or
 * other coordinate system) using the WCS information in the header.
 * 
 * @param pixel_coords Tensor of shape [N, naxis] with pixel coordinates
 * @param header Map of FITS header key-value pairs containing WCS information
 * @return Tuple of (world_coords, status_tensor) where world_coords is a tensor
 *         of shape [N, 2+] and status_tensor indicates success/failure per point
 * @throws std::runtime_error if the header does not contain valid WCS information
 */
std::tuple<torch::Tensor, torch::Tensor> pixel_to_world(
    torch::Tensor pixel_coords,
    std::map<std::string, std::string> header
);

#endif // TORCHFITS_WCS_UTILS_H