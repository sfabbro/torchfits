#ifndef FITS_READER_H
#define FITS_READER_H

#include <torch/extension.h>  // For PyTorch C++ API
#include <fitsio.h>          // For CFITSIO
#include <wcslib/wcs.h>     // For WCSLIB
#include <vector>
#include <map>
#include <string>
#include <memory>         // For std::unique_ptr
#include <pybind11/pybind11.h> //For pybind

// --- Function Prototypes ---

// --- Helper Functions (defined in fits_utils.cpp) ---
std::string fits_status_to_string(int status);
void throw_fits_error(int status, const std::string& message = "");
std::pair<std::string, std::string> parse_header_card(const char* card);
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);
std::vector<long long> get_image_dims(const std::string& filename, int hdu_num);
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key);
std::string get_hdu_type(const std::string& filename, int hdu_num);
int get_num_hdus(const std::string& filename);

// --- WCS Functions (defined in wcs_utils.cpp) ---
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);
std::pair<torch::Tensor, torch::Tensor> world_to_pixel(const torch::Tensor& world_coords, const std::map<std::string, std::string>& header); //Not exposed to Python
std::pair<torch::Tensor, torch::Tensor> pixel_to_world(const torch::Tensor& pixel_coords, const std::map<std::string, std::string>& header); //Not exposed to Python

// --- Core Data Reading Functions (defined in fits_reader.cpp) ---
torch::Tensor read_image_data(fitsfile* fptr, std::unique_ptr<wcsprm>& wcs);  // Internal use
std::map<std::string, torch::Tensor> read_table_data(fitsfile* fptr);          // Internal use
pybind11::object read(const std::string& filename_with_cutout, pybind11::object hdu = pybind11::none(), pybind11::object start = pybind11::none(), pybind11::object shape = pybind11::none());

#endif // FITS_READER_H
