#ifndef TORCHFITS_FITS_READER_H
#define TORCHFITS_FITS_READER_H

#include <torch/extension.h>
#include <fitsio.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "cache.h"

// Forward declarations
class FITSFile;
struct wcsprm;

// Read image data from FITS file
torch::Tensor read_data(fitsfile* fptr, torch::Device device, const std::vector<long>& start, const std::vector<long>& shape);

// Read table data from FITS file
std::map<std::string, torch::Tensor> read_table_data(
    fitsfile* fptr,
    pybind11::object columns,
    int start_row,
    pybind11::object num_rows_obj,
    torch::Device device,
    std::shared_ptr<CacheEntry> entry
);

// Generate cache key
std::string generate_cache_key(
    const std::string& filename,
    const pybind11::object& hdu,
    const pybind11::object& start,
    const pybind11::object& shape,
    const pybind11::object& columns,
    int start_row,
    const pybind11::object& num_rows
);

// Main read implementation (exposed to Python)
pybind11::object read_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    int start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    torch::Device device
);

// Helper functions
std::map<std::string, std::string> get_header_by_name(const std::string& filename, const std::string& hdu_name);
std::map<std::string, std::string> get_header_by_number(const std::string& filename, int hdu_num);
int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name);

#endif // TORCHFITS_FITS_READER_H
