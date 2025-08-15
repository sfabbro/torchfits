#ifndef TORCHFITS_FITS_READER_H
#define TORCHFITS_FITS_READER_H

#include <torch/extension.h>
#include <fitsio.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

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
    pybind11::object enable_mmap,
    pybind11::object enable_buffered,
    pybind11::str device_str
);

// Read table data along with per-column null masks where supported; returns
// a Python tuple (data_dict, header_dict, masks_dict) from bindings layer.
pybind11::object read_table_with_null_masks_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object columns,
    long start_row,
    pybind11::object num_rows,
    pybind11::str device_str
);

// Optimized path: read many small image cutouts from the same HDU in a single open session.
// Returns a Python list of tensors in the same order as the provided starts.
pybind11::object read_many_cutouts(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    const std::vector<std::vector<long>>& starts,
    const std::vector<long>& shape,
    pybind11::str device_str
);

// Batched small-cutout reader across multiple HDUs (MEF optimization)
pybind11::object read_many_cutouts_multi_hdu(
    pybind11::object filename_or_url,
    const std::vector<int>& hdus,
    const std::vector<std::vector<long>>& starts,
    const std::vector<long>& shape,
    pybind11::str device_str
);

// Expose last read path/flags for diagnostics/benchmarks
pybind11::dict get_last_read_info();

#endif // TORCHFITS_FITS_READER_H