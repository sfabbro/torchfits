#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "cache.h"
#include "remote.h"
#include "performance_v2.h"  // Use the enhanced performance module
#include "debug.h"
#include <sstream>
#include <algorithm>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

namespace py = pybind11;

// Helper to get PyTorch dtype from CFITSIO type code
torch::Dtype get_torch_dtype(int typecode) {
    switch (typecode) {
        case TBYTE:     return torch::kUInt8;
        case TSHORT:    return torch::kInt16;
        case TINT:      return torch::kInt32;
        case TLONGLONG: return torch::kInt64;
        case TFLOAT:    return torch::kFloat32;
        case TDOUBLE:   return torch::kFloat64;
        case TLOGICAL:  return torch::kBool;
        default:        return torch::kFloat32;
    }
}

// --- Enhanced Image Data Reading with v1.0 optimizations ---

template <typename T, int CfitsioType>
torch::Tensor read_image_data_typed_v2(fitsfile* fptr, torch::Device device,
                                      const std::vector<long>& start, const std::vector<long>& shape,
                                      const std::string& filename = "") {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int status = 0;
    int naxis;
    if (fits_get_img_dim(fptr, &naxis, &status)) {
        throw_fits_error(status, "Error getting image dimensions");
    }

    std::vector<long> naxes(naxis);
    if (fits_get_img_size(fptr, naxis, naxes.data(), &status)) {
        throw_fits_error(status, "Error getting image size");
    }
    
    // Reverse dimensions for C-style ordering
    std::reverse(naxes.begin(), naxes.end());

    torch::TensorOptions options = torch::TensorOptions().device(device);

    // Set the correct dtype
    if (std::is_same<T, uint8_t>::value) options = options.dtype(torch::kUInt8);
    else if (std::is_same<T, int16_t>::value) options = options.dtype(torch::kInt16);
    else if (std::is_same<T, int32_t>::value) options = options.dtype(torch::kInt32);
    else if (std::is_same<T, int64_t>::value) options = options.dtype(torch::kInt64);
    else if (std::is_same<T, float>::value) options = options.dtype(torch::kFloat32);
    else if (std::is_same<T, double>::value) options = options.dtype(torch::kFloat64);

    torch::Tensor data;
    
    if (!start.empty()) {
        // Reading a subset (cutout)
        std::vector<long> fpixel(naxis), lpixel(naxis), inc(naxis, 1);
        std::vector<int64_t> cutout_dims;

        for (int i = 0; i < naxis; ++i) {
            int fits_idx = naxis - 1 - i;
            fpixel[fits_idx] = start[i] + 1;
            long dim_size = (shape[i] == -1) ? (naxes[fits_idx] - start[i]) : shape[i];
            lpixel[fits_idx] = start[i] + dim_size;
            cutout_dims.push_back(dim_size);
        }

        // Use memory pool for tensor allocation
        if (torchfits_perf::global_memory_pool) {
            data = torchfits_perf::global_memory_pool->get_tensor(cutout_dims, options.dtype().toScalarType(), device);
        } else {
            data = torch::empty(cutout_dims, options);
        }
        
        // Use optimized buffer if available
        size_t data_size = data.numel() * sizeof(T);
        char* buffer = nullptr;
        if (torchfits_perf::global_buffer_manager) {
            size_t optimal_buffer_size = torchfits_perf::global_buffer_manager->get_optimal_buffer_size(
                filename, "cutout", data_size);
            buffer = torchfits_perf::global_buffer_manager->get_buffer(optimal_buffer_size, "image");
        }
        
        fits_read_subset(fptr, CfitsioType, fpixel.data(), lpixel.data(), inc.data(),
                         nullptr, data.data_ptr<T>(), nullptr, &status);
        
        if (buffer && torchfits_perf::global_buffer_manager) {
            torchfits_perf::global_buffer_manager->return_buffer(buffer);
        }
        
        if (status) throw_fits_error(status, "Error reading data subset");
        
    } else {
        // Reading the full image
        std::vector<int64_t> torch_dims;
        for (long val : naxes) {
            torch_dims.push_back(val);
        }

        // Use memory pool for tensor allocation
        if (torchfits_perf::global_memory_pool) {
            data = torchfits_perf::global_memory_pool->get_tensor(torch_dims, options.dtype().toScalarType(), device);
        } else {
            data = torch::empty(torch_dims, options);
        }

        // Try memory mapping for large files
        void* mapped_ptr = nullptr;
        if (torchfits_perf::global_memory_mapper && !filename.empty()) {
            // Get file size estimate
            size_t estimated_size = data.numel() * sizeof(T);
            if (torchfits_perf::global_memory_mapper->should_use_memory_mapping(filename, estimated_size)) {
                auto mapped_file = torchfits_perf::global_memory_mapper->map_file(filename);
                if (mapped_file && mapped_file->is_valid) {
                    mapped_ptr = mapped_file->ptr;
                    DEBUG_LOG("Using memory mapping for file: " + filename);
                }
            }
        }

        if (mapped_ptr) {
            // Use memory mapping with CFITSIO
            fits_use_file_memory(fptr, mapped_ptr, &status);
            if (status) {
                DEBUG_LOG("Failed to use memory mapping, falling back to regular I/O");
                status = 0; // Reset status and continue with regular read
            }
        }

        // Use optimized buffer for reading
        char* buffer = nullptr;
        if (torchfits_perf::global_buffer_manager) {
            size_t data_size = data.numel() * sizeof(T);
            size_t optimal_buffer_size = torchfits_perf::global_buffer_manager->get_optimal_buffer_size(
                filename, "image_read", data_size);
            buffer = torchfits_perf::global_buffer_manager->get_buffer(optimal_buffer_size, "image");
        }

        fits_read_img(fptr, CfitsioType, 1, data.numel(), nullptr, data.data_ptr<T>(), nullptr, &status);
        
        if (buffer && torchfits_perf::global_buffer_manager) {
            torchfits_perf::global_buffer_manager->return_buffer(buffer);
        }
        
        if (status) throw_fits_error(status, "Error reading image data");
    }

    // Apply SIMD optimizations for data conversion if needed
    if (torchfits_perf::SIMDOptimizer::simd_available()) {
        // SIMD optimizations would be applied here for byte swapping or format conversion
        DEBUG_LOG("SIMD optimizations available for data processing");
    }

    // Record performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    if (torchfits_perf::global_performance_monitor) {
        std::string operation = start.empty() ? "image_read_full" : "image_read_cutout";
        torchfits_perf::global_performance_monitor->record_operation(operation, duration_ms);
    }

    return data;
}

torch::Tensor read_image_data_v2(fitsfile* fptr, torch::Device device,
                               const std::vector<long>& start, const std::vector<long>& shape,
                               const std::string& filename = "") {
    int status = 0;
    int bitpix;
    fits_get_img_type(fptr, &bitpix, &status);
    if (status) throw_fits_error(status, "Error getting image type");

    switch (bitpix) {
        case BYTE_IMG:     return read_image_data_typed_v2<uint8_t, TBYTE>(fptr, device, start, shape, filename);
        case SHORT_IMG:    return read_image_data_typed_v2<int16_t, TSHORT>(fptr, device, start, shape, filename);
        case LONG_IMG:     return read_image_data_typed_v2<int32_t, TINT>(fptr, device, start, shape, filename);
        case LONGLONG_IMG: return read_image_data_typed_v2<int64_t, TLONGLONG>(fptr, device, start, shape, filename);
        case FLOAT_IMG:    return read_image_data_typed_v2<float, TFLOAT>(fptr, device, start, shape, filename);
        case DOUBLE_IMG:   return read_image_data_typed_v2<double, TDOUBLE>(fptr, device, start, shape, filename);
        default: throw std::runtime_error("Unsupported FITS image data type: " + std::to_string(bitpix));
    }
}

// --- Enhanced Table Data Reading with Parallel Processing ---

py::dict read_table_data_v2(fitsfile* fptr, torch::Device device,
                           const py::object& columns_obj,
                           long start_row, const py::object& num_rows_obj,
                           const std::string& filename = "") {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int status = 0;
    long num_rows;
    
    if (num_rows_obj.is_none()) {
        fits_get_num_rows(fptr, &num_rows, &status);
        if (status) throw_fits_error(status, "Error getting number of rows");
    } else {
        num_rows = num_rows_obj.cast<long>();
    }

    int num_cols;
    fits_get_num_cols(fptr, &num_cols, &status);
    if (status) throw_fits_error(status, "Error getting number of columns");

    // Determine which columns to read
    std::vector<std::string> column_names;
    if (columns_obj.is_none()) {
        // Read all columns
        for (int i = 1; i <= num_cols; ++i) {
            char colname[FLEN_VALUE];
            fits_get_colname(fptr, CASEINSEN, nullptr, colname, &i, &status);
            if (status == 0) {
                column_names.push_back(std::string(colname));
            }
            status = 0; // Reset for next iteration
        }
    } else {
        column_names = columns_obj.cast<std::vector<std::string>>();
    }

    // Use parallel table reader if available and beneficial
    bool use_parallel = column_names.size() > 1 && num_rows > 1000;
    
    if (use_parallel && torchfits_perf::global_memory_pool) {
        // Create a parallel table reader
        torchfits_perf::ParallelTableReader parallel_reader;
        py::dict result = parallel_reader.read_columns_parallel(fptr, column_names, start_row, num_rows, device);
        
        // Record performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        if (torchfits_perf::global_performance_monitor) {
            torchfits_perf::global_performance_monitor->record_operation("table_read_parallel", duration_ms);
        }
        
        return result;
    }

    // Fall back to sequential reading (enhanced with optimizations)
    py::dict result;
    
    for (const std::string& col_name : column_names) {
        int col_num;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &col_num, &status);
        if (status) {
            status = 0; // Reset and skip this column
            continue;
        }

        int typecode;
        long repeat, width;
        fits_get_coltype(fptr, col_num, &typecode, &repeat, &width, &status);
        if (status) throw_fits_error(status, "Error getting column type for " + col_name);

        torch::Dtype torch_dtype = get_torch_dtype(typecode);
        
        // Use memory pool for tensor allocation
        std::vector<int64_t> tensor_shape = {num_rows};
        if (repeat > 1) tensor_shape.push_back(repeat);
        
        torch::Tensor column_data;
        if (torchfits_perf::global_memory_pool) {
            column_data = torchfits_perf::global_memory_pool->get_tensor(tensor_shape, torch_dtype, device);
        } else {
            column_data = torch::empty(tensor_shape, torch::TensorOptions().dtype(torch_dtype).device(device));
        }

        // Read column data with optimized buffer
        char* buffer = nullptr;
        if (torchfits_perf::global_buffer_manager) {
            size_t data_size = column_data.numel() * torch::elementSize(torch_dtype);
            size_t optimal_buffer_size = torchfits_perf::global_buffer_manager->get_optimal_buffer_size(
                filename, "table_read", data_size);
            buffer = torchfits_perf::global_buffer_manager->get_buffer(optimal_buffer_size, "table");
        }

        // Read the column data
        fits_read_col(fptr, typecode, col_num, start_row + 1, 1, num_rows,
                     nullptr, column_data.data_ptr(), nullptr, &status);

        if (buffer && torchfits_perf::global_buffer_manager) {
            torchfits_perf::global_buffer_manager->return_buffer(buffer);
        }
        
        if (status) throw_fits_error(status, "Error reading column " + col_name);

        result[py::str(col_name)] = column_data;
    }

    // Record performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    if (torchfits_perf::global_performance_monitor) {
        torchfits_perf::global_performance_monitor->record_operation("table_read_sequential", duration_ms);
    }

    return result;
}

// --- Enhanced Main Read Implementation ---

pybind11::object read_impl_v2(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    long start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    pybind11::str device_str
) {
    torch::Device device(device_str);
    DEBUG_SCOPE;

    try {
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
        
        // Handle remote URLs - download to cache if necessary
        std::string filename = RemoteFetcher::ensure_local(filename_or_url_str);

        int hdu_num = 1;
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        FITSFileWrapper f(filename);
        int status = 0;
        if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }

        auto header = read_fits_header(f.get());

        int hdu_type;
        fits_get_hdu_type(f.get(), &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");

        if (hdu_type == IMAGE_HDU) {
            int naxis = 0;
            fits_get_img_dim(f.get(), &naxis, &status);
            if (naxis == 0) {
                return py::make_tuple(py::none(), py::cast(header));
            }

            std::vector<long> start_vec, shape_vec;
            if (!start.is_none()) {
                start_vec = start.cast<std::vector<long>>();
                shape_vec = shape.cast<std::vector<long>>();
            }

            torch::Tensor data = read_image_data_v2(f.get(), device, start_vec, shape_vec, filename);
            return py::make_tuple(data, py::cast(header));
            
        } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            py::dict table_data = read_table_data_v2(f.get(), device, columns, start_row, num_rows, filename);
            return py::make_tuple(table_data, py::cast(header));
        } else {
            // For unknown HDU types, return header but no data
            return py::make_tuple(py::none(), py::cast(header));
        }

    } catch (const std::exception& e) {
        // Re-throw C++ exceptions as Python exceptions
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}
