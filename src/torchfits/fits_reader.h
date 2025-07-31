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

// Read image data from a FITS HDU
torch::Tensor read_image_data(fitsfile* fptr, torch::Device device, 
                            const std::vector<long>& start, const std::vector<long>& shape);

// Read table data from a FITS HDU
pybind11::dict read_table_data(fitsfile* fptr, torch::Device device,
                             const pybind11::object& columns_obj,
                             long start_row, const pybind11::object& num_rows_obj);

// Generate a cache key for the given read parameters
std::string generate_cache_key(
    const std::string& filename,
    const pybind11::object& hdu,
    const pybind11::object& start,
    const pybind11::object& shape,
    const pybind11::object& columns,
    long start_row,
    const pybind11::object& num_rows
);

// Main read implementation exposed to Python
pybind11::object read_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    long start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    pybind11::str device_str
);

#endif // TORCHFITS_FITS_READER_H