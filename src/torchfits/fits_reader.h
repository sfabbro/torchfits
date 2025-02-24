#ifndef FITS_READER_H
#define FITS_READER_H

#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <pybind11/pybind11.h> // Required for pybind11::object

// --- Function Prototypes ---

// --- Helper Functions (defined in fits_utils.cpp) ---
std::string fits_status_to_string(int status);
void throw_fits_error(int status, const std::string& message = "");
std::pair<std::string, std::string> parse_header_card(const char* card);
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num); // Internal
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key);
std::string get_hdu_type(const std::string& filename, int hdu_num);
int get_num_hdus(const std::string& filename);
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num); // Wrapper
std::vector<long long> get_dims(const std::string& filename, int hdu_num); //Wrapper

// --- WCS Functions (defined in wcs_utils.cpp) ---
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);
std::pair<torch::Tensor, torch::Tensor> world_to_pixel(const torch::Tensor& world_coords, const std::map<std::string, std::string>& header); //Not exposed to Python
std::pair<torch::Tensor, torch::Tensor> pixel_to_world(const torch::Tensor& pixel_coords, const std::map<std::string, std::string>& header); //Not exposed to Python

// --- Core Data Reading Functions (defined in fits_reader.cpp) ---
torch::Tensor read_data(fitsfile* fptr, std::unique_ptr<wcsprm>& wcs, torch::Device device);  // Internal use
std::map<std::string, torch::Tensor> read_table_data(fitsfile* fptr, pybind11::object columns, int start_row, pybind11::object num_rows); // Internal use
pybind11::object read(pybind11::object filename_or_url, pybind11::object hdu = pybind11::none(), pybind11::object start = pybind11::none(), pybind11::object shape = pybind11::none(), pybind11::object columns = pybind11::none(), int start_row = 0, pybind11::object num_rows = pybind11::none(), size_t cache_capacity = 0, torch::Device device = torch::kCPU);


#endif // FITS_READER_H