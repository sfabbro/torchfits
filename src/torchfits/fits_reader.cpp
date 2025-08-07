#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "cache.h"
#include "remote.h"
#include "performance.h"
#include "cfitsio_enhanced.h"
#include "real_cache.h"
#include "debug.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

namespace py = pybind11;

// --- Templated Image Data Reading ---

template <typename T, int CfitsioType>
torch::Tensor read_image_data_typed(fitsfile* fptr, torch::Device device,
                                  const std::vector<long>& start, const std::vector<long>& shape) {
    DEBUG_SCOPE;
    int status = 0;
    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) throw_fits_error(status, "Error getting image dimension");

    // Handle empty HDU (NAXIS=0) case - no image data
    if (naxis == 0) {
        DEBUG_LOG("Empty HDU detected (NAXIS=0), returning empty tensor");
        torch::TensorOptions options = torch::TensorOptions().device(device);
        if      (std::is_same<T, uint8_t>::value) options = options.dtype(torch::kUInt8);
        else if (std::is_same<T, int16_t>::value) options = options.dtype(torch::kInt16);
        else if (std::is_same<T, int32_t>::value) options = options.dtype(torch::kInt32);
        else if (std::is_same<T, int64_t>::value) options = options.dtype(torch::kInt64);
        else if (std::is_same<T, float>::value)   options = options.dtype(torch::kFloat32);
        else if (std::is_same<T, double>::value)  options = options.dtype(torch::kFloat64);
        
        return torch::empty({0}, options);  // Return empty 1D tensor with 0 elements
    }

    std::vector<long> naxes(naxis);
    fits_get_img_size(fptr, naxis, naxes.data(), &status);
    if (status) throw_fits_error(status, "Error getting image size");

    torch::TensorOptions options = torch::TensorOptions().device(device);
    if      (std::is_same<T, uint8_t>::value) options = options.dtype(torch::kUInt8);
    else if (std::is_same<T, int16_t>::value) options = options.dtype(torch::kInt16);
    else if (std::is_same<T, int32_t>::value) options = options.dtype(torch::kInt32);
    else if (std::is_same<T, int64_t>::value) options = options.dtype(torch::kInt64);
    else if (std::is_same<T, float>::value)   options = options.dtype(torch::kFloat32);
    else if (std::is_same<T, double>::value)  options = options.dtype(torch::kFloat64);

    if (!start.empty()) {
        // Reading a subset
        std::vector<long> fpixel(naxis), lpixel(naxis), inc(naxis, 1);
        std::vector<int64_t> cutout_dims;

        for (int i = 0; i < naxis; ++i) {
            int fits_idx = naxis - 1 - i;
            fpixel[fits_idx] = start[i] + 1;
            long dim_size = (shape[i] == -1) ? (naxes[fits_idx] - start[i]) : shape[i];
            lpixel[fits_idx] = start[i] + dim_size;
            cutout_dims.push_back(dim_size);
        }

        torch::Tensor data = torch::empty(cutout_dims, options);
        fits_read_subset(fptr, CfitsioType, fpixel.data(), lpixel.data(), inc.data(),
                         nullptr, data.data_ptr<T>(), nullptr, &status);
        if (status) throw_fits_error(status, "Error reading data subset");
        return data;
    } else {
        // Reading the full image - CFITSIO API OPTIMIZATION
        std::vector<int64_t> torch_dims;
        for(long val : naxes) {
            torch_dims.push_back(val);
        }
        std::reverse(torch_dims.begin(), torch_dims.end());

        torch::Tensor data = torch::empty(torch_dims, options);
        long n_elements = data.numel();
        
        // OPTIMIZATION: Use fits_read_pix for better performance on full images
        // fits_read_pix is often faster than fits_read_img for full image reads
        long fpixel[MAX_COMPRESS_DIM] = {1,1,1,1,1,1};  // Start at pixel 1,1,1... in FITS convention
        
        fits_read_pix(fptr, CfitsioType, fpixel, n_elements, nullptr,
                      data.data_ptr<T>(), nullptr, &status);
        if (status) throw_fits_error(status, "Error reading full image with fits_read_pix");
        
        return data;
    }
}

torch::Tensor read_image_data(fitsfile* fptr, torch::Device device,
                           const std::vector<long>& start, const std::vector<long>& shape) {
    int status = 0;
    int bitpix;
    fits_get_img_type(fptr, &bitpix, &status);
    if (status) throw_fits_error(status, "Error getting image type");

    switch (bitpix) {
        case BYTE_IMG:     return read_image_data_typed<uint8_t, TBYTE>(fptr, device, start, shape);
        case SHORT_IMG:    return read_image_data_typed<int16_t, TSHORT>(fptr, device, start, shape);
        case LONG_IMG:     return read_image_data_typed<int32_t, TINT>(fptr, device, start, shape);
        case LONGLONG_IMG: return read_image_data_typed<int64_t, TLONGLONG>(fptr, device, start, shape);
        case FLOAT_IMG:    return read_image_data_typed<float, TFLOAT>(fptr, device, start, shape);
        case DOUBLE_IMG:   return read_image_data_typed<double, TDOUBLE>(fptr, device, start, shape);
        default: throw std::runtime_error("Unsupported FITS image data type: " + std::to_string(bitpix));
    }
}


// --- Table Data Reading ---

// Helper to get PyTorch dtype from CFITSIO type code
torch::Dtype get_torch_dtype(int typecode) {
    switch (typecode) {
        case TBYTE:     return torch::kUInt8;
        case 12:        return torch::kInt8;     // TSBYTE - 8-bit signed byte
        case 20:        return torch::kUInt16;   // TUSHORT - 16-bit unsigned integer  
        case TSHORT:    return torch::kInt16;
        case 30:        return torch::kUInt32;   // TUINT - 32-bit unsigned integer
        case TINT:      return torch::kInt32;
        case 40:        return torch::kInt32;    // TULONG - 32-bit unsigned long (same as TUINT)
        case 41:        return torch::kInt32;    // TLONG - 32-bit signed long (same as TINT)
        case TLONGLONG: return torch::kInt64;
        case TFLOAT:    return torch::kFloat32;
        case TDOUBLE:   return torch::kFloat64;
        case TLOGICAL:  return torch::kBool;
        default:        return torch::kFloat64; // Default for unsupported
    }
}

py::dict read_table_data(fitsfile* fptr, torch::Device device,
                         const py::object& columns_obj,
                         long start_row, const py::object& num_rows_obj) {
    DEBUG_SCOPE;
    int status = 0;
    long total_rows;
    int total_cols;

    fits_get_num_rows(fptr, &total_rows, &status);
    if (status) throw_fits_error(status, "Error getting number of rows");

    fits_get_num_cols(fptr, &total_cols, &status);
    if (status) throw_fits_error(status, "Error getting number of columns");

    long rows_to_read = total_rows - start_row;
    if (!num_rows_obj.is_none()) {
        rows_to_read = std::min(rows_to_read, num_rows_obj.cast<long>());
    }
    
    if (rows_to_read <= 0) {
        return py::dict();
    }

    // Determine columns to read
    std::vector<std::string> selected_columns;
    if (!columns_obj.is_none()) {
        selected_columns = columns_obj.cast<std::vector<std::string>>();
    } else {
        // Get all column names
        selected_columns.reserve(total_cols);
        for (int i = 1; i <= total_cols; ++i) {
            char colname[FLEN_VALUE];
            status = 0;
            fits_get_bcolparms(fptr, i, colname, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &status);
            if (status) {
                throw_fits_error(status, "Error getting column info for col " + std::to_string(i));
            }
            selected_columns.emplace_back(colname);
        }
    }

    // PERFORMANCE OPTIMIZATION: Use parallel reading for multiple columns
    if (selected_columns.size() >= 3 && rows_to_read >= 1000) {
        // Use parallel table reader for large tables with multiple columns
        DEBUG_LOG("Using parallel table reader for " + std::to_string(selected_columns.size()) + 
                  " columns and " + std::to_string(rows_to_read) + " rows");
        
        // torchfits_perf::ParallelTableReader parallel_reader;  // Temporarily disabled
        // return parallel_reader.read_columns_parallel(fptr, selected_columns, start_row, rows_to_read, device);
        DEBUG_LOG("Parallel reader temporarily disabled, using sequential reader");
    }

    // FALLBACK: Use optimized sequential reading for smaller datasets
    DEBUG_LOG("Using sequential optimized reader");
    
    // Separate string and numeric columns for optimized processing
    struct ColumnInfo {
        std::string name;
        int number;
        int typecode;
        long repeat;
        long width;
        bool is_string;
    };
    
    std::vector<ColumnInfo> column_info;
    column_info.reserve(selected_columns.size());
    
    for (const auto& col_name : selected_columns) {
        ColumnInfo info;
        info.name = col_name;
        
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &info.number, &status);
        if (status) throw_fits_error(status, "Error finding column: " + col_name);
        
        fits_get_coltype(fptr, info.number, &info.typecode, &info.repeat, &info.width, &status);
        if (status) throw_fits_error(status, "Error getting column type for: " + col_name);
        
        info.is_string = (info.typecode == TSTRING);
        column_info.push_back(info);
    }

    py::dict result_dict;

    // Process string columns sequentially (required due to CFITSIO limitations)
    for (const auto& info : column_info) {
        if (!info.is_string) continue;
        
        std::vector<char*> string_array(rows_to_read);
        std::vector<char> string_buffer(rows_to_read * (info.repeat + 1));
        
        for (long i = 0; i < rows_to_read; i++) {
            string_array[i] = &string_buffer[i * (info.repeat + 1)];
        }
        
        fits_read_col_str(fptr, info.number, start_row + 1, 1, rows_to_read,
                          nullptr, string_array.data(), nullptr, &status);
        if (status) throw_fits_error(status, "Error reading string column: " + info.name);
        
        py::list string_list;
        for (long i = 0; i < rows_to_read; i++) {
            string_list.append(py::str(string_array[i]));
        }
        result_dict[py::str(info.name)] = std::move(string_list);
    }

    // Process numeric columns with memory pool optimization
    for (const auto& info : column_info) {
        if (info.is_string) continue;
        
        torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
        
        // Use memory pool for tensor allocation
        torch::Tensor col_data;
        // Temporarily disabled memory pool optimization
        // if (torchfits_perf::global_memory_pool) {
        //     std::vector<int64_t> shape = (info.repeat == 1) ? 
        //         std::vector<int64_t>{rows_to_read} : 
        //         std::vector<int64_t>{rows_to_read, info.repeat};
        //     
        //     col_data = torchfits_perf::global_memory_pool->get_tensor(shape, torch_dtype, torch::kCPU);
        // } else {
            // Fallback to direct allocation
            if (info.repeat == 1) {
                col_data = torch::empty({rows_to_read}, torch::TensorOptions().dtype(torch_dtype));
            } else {
                col_data = torch::empty({rows_to_read, info.repeat}, torch::TensorOptions().dtype(torch_dtype));
            }
        // }  // End of commented memory pool section
        
        // CFITSIO API OPTIMIZATION: Use most efficient column reading method
        // For large datasets, bulk read operations are more efficient
        status = 0;
        
        if (rows_to_read * info.repeat > 1000) {
            // Large dataset: Use fits_read_colnull for better null value handling and performance
            char null_flag = 0;  // We'll ignore null flags for now for performance
            fits_read_colnull(fptr, info.typecode, info.number, start_row + 1, 1,
                             rows_to_read * info.repeat,
                             nullptr, col_data.data_ptr(), &null_flag, &status);
        } else {
            // Small dataset: Use standard fits_read_col
            fits_read_col(fptr, info.typecode, info.number, start_row + 1, 1,
                         rows_to_read * info.repeat,
                         nullptr, col_data.data_ptr(), nullptr, &status);
        }
        
        if (status) throw_fits_error(status, "Error reading column: " + info.name);
        
        // Transfer to target device
        if (device != torch::kCPU) {
            col_data = col_data.to(device);
        }
        
        result_dict[py::str(info.name)] = col_data.squeeze();
    }

    return result_dict;
}


// --- Main read implementation with CFITSIO enhancements ---
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
) {
    torch::Device device(device_str);
    DEBUG_SCOPE;
    DEBUG_LOG("=== Starting read_impl ===");
    std::cerr << "[TORCHFITS:FORCE] Starting read_impl function" << std::endl;

    try {
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
        DEBUG_LOG("Filename: " + filename_or_url_str);
        std::cerr << "[TORCHFITS:FORCE] Filename: " << filename_or_url_str << std::endl;
        
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

        // === REAL CACHE CHECK: Try to get from cache first ===
        auto& real_cache = torchfits_real_cache::RealSmartCache::get_instance();
        std::ostringstream cache_key_stream;
        cache_key_stream << filename << "_hdu" << hdu_num;
        
        if (!start.is_none()) {
            auto start_vec = start.cast<std::vector<long>>();
            cache_key_stream << "_start";
            for (size_t i = 0; i < start_vec.size(); ++i) {
                cache_key_stream << "_" << start_vec[i];
            }
        }
        if (!shape.is_none()) {
            auto shape_vec = shape.cast<std::vector<long>>();
            cache_key_stream << "_shape";
            for (size_t i = 0; i < shape_vec.size(); ++i) {
                cache_key_stream << "_" << shape_vec[i];
            }
        }
        cache_key_stream << "_device_" << device_str.cast<std::string>();
        
        std::string cache_key = cache_key_stream.str();
        auto cached_result = real_cache.try_get(cache_key);
        if (cached_result.has_value()) {
            DEBUG_LOG("Cache hit for key: " + cache_key);
            // For cache hits, we need to read the header separately
            FITSFileWrapper f(filename);
            int status = 0;
            if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
                throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
            }
            auto header = read_fits_header(f.get());
            return py::make_tuple(py::cast(cached_result.value()), py::cast(header));
        }

        // === PERFORMANCE OPTIMIZATION: Try CFITSIO enhanced reading ===
        // Temporarily disabled to fix segfault - will re-enable after debugging
        /*
        // Check if we should use memory mapping
        try {
            torchfits_cfitsio_enhanced::CFITSIOMemoryMapper mapper(filename);
            DEBUG_LOG("CFITSIOMemoryMapper created successfully");
            bool use_mapping = mapper.should_use_memory_mapping();
            DEBUG_LOG("should_use_memory_mapping returned: " + std::to_string(use_mapping));
            if (use_mapping && start.is_none() && shape.is_none()) {
                DEBUG_LOG("Using CFITSIO memory mapping for large file");
                std::vector<long> start_vec, shape_vec;
                torch::Tensor result = mapper.read_with_memory_mapping(hdu_num, start_vec, shape_vec);
                torch::Tensor device_result = result.to(device);
                
                // Cache the result
                real_cache.put(cache_key, device_result);
                
                // Read header for complete return
                FITSFileWrapper f(filename);
                int status = 0;
                if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
                    throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
                }
                auto header = read_fits_header(f.get());
                
                return py::make_tuple(py::cast(device_result), py::cast(header));
            }
        } catch (const std::exception& e) {
            DEBUG_LOG("Memory mapping failed, falling back: " + std::string(e.what()));
        }
        */
        
        // Try buffered reading with CFITSIO optimizations
        // Temporarily disabled to fix segfault - will re-enable after debugging  
        /*
        try {
            torchfits_cfitsio_enhanced::CFITSIOBufferedReader buffered_reader(filename);
            
            // For image data, use optimized scaling read
            FITSFileWrapper f(filename);
            int status = 0;
            if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
                throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
            }
            
            int hdu_type;
            fits_get_hdu_type(f.get(), &hdu_type, &status);
            if (status) throw_fits_error(status, "Error getting HDU type");
            
            if (hdu_type == IMAGE_HDU && start.is_none() && shape.is_none()) {
                DEBUG_LOG("Using CFITSIO buffered reader with scaling");
                torch::Tensor result = buffered_reader.read_with_scaling(hdu_num);
                torch::Tensor device_result = result.to(device);
                
                // Cache the result
                real_cache.put(cache_key, device_result);
                
                // Read header for complete return
                auto header = read_fits_header(f.get());
                
                return py::make_tuple(py::cast(device_result), py::cast(header));
            }
            
        } catch (const std::exception& e) {
            DEBUG_LOG("Buffered reading failed, falling back: " + std::string(e.what()));
        }
        */

        // === FALLBACK: Standard reading with performance optimization ===
        FITSFileWrapper f(filename);
        int status = 0;
        if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }

        // Apply CFITSIO performance optimizations to standard reading
        // OPTIMIZATION: Get actual file size for better buffer sizing
        std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
        size_t file_size = 1000000; // Default fallback
        if (file.is_open()) {
            file_size = static_cast<size_t>(file.tellg());
            file.close();
            DEBUG_LOG("Detected file size: " + std::to_string(file_size) + " bytes");
        }
        
        torchfits_cfitsio_enhanced::CFITSIOPerformanceOptimizer::optimize_file_access(f.get(), file_size);

        auto header = read_fits_header(f.get());

        int hdu_type;
        fits_get_hdu_type(f.get(), &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");

        if (hdu_type == IMAGE_HDU) {
            DEBUG_LOG("Processing IMAGE_HDU");
            int naxis = 0;
            fits_get_img_dim(f.get(), &naxis, &status);
            DEBUG_LOG("Got naxis = " + std::to_string(naxis));
            if (naxis == 0) {
                DEBUG_LOG("Empty HDU detected, returning py::none()");
                return py::make_tuple(py::none(), py::cast(header));
            }
            DEBUG_LOG("Non-empty HDU, calling read_image_data()");

            std::vector<long> start_vec, shape_vec;
            if (!start.is_none()) {
                start_vec = start.cast<std::vector<long>>();
                shape_vec = shape.cast<std::vector<long>>();
            }

            torch::Tensor data = read_image_data(f.get(), device, start_vec, shape_vec);
            
            // Cache the result if it's a tensor
            if (data.defined()) {
                real_cache.put(cache_key, data);
            }
            
            return py::make_tuple(data, py::cast(header));
        } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            py::dict table_data = read_table_data(f.get(), device, columns, start_row, num_rows);
            return py::make_tuple(table_data, py::cast(header));
        } else {
            // For unknown HDU types, return header but no data.
            return py::make_tuple(py::none(), py::cast(header));
        }

    } catch (const std::exception& e) {
        // Re-throw C++ exceptions as Python exceptions
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}