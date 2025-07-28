#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "cache.h"
#include "debug.h"
#include <sstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

namespace py = pybind11;

// --- Helper functions ---

// Generate cache key based on read parameters
std::string generate_cache_key(
    const std::string& filename,
    const pybind11::object& hdu,
    const pybind11::object& start,
    const pybind11::object& shape,
    const pybind11::object& columns,
    int start_row,
    const pybind11::object& num_rows
) {
    std::stringstream key_builder;
    key_builder << filename;

    // Add HDU information
    if (!hdu.is_none()) {
        if (py::isinstance<py::str>(hdu)) {
            key_builder << "_hdu_" << hdu.cast<std::string>();
        } else {
            key_builder << "_hdu_" << hdu.cast<int>();
        }
    }

    // Add cutout information
    if (!start.is_none()) {
        key_builder << "_cutout_";
        auto start_list = start.cast<std::vector<long>>();
        auto shape_list = shape.cast<std::vector<long>>();
        for (size_t i = 0; i < start_list.size(); ++i) {
            key_builder << start_list[i] << ":" <<
                (shape_list[i] == -1 ? "end" : std::to_string(start_list[i] + shape_list[i]));
            if (i < start_list.size() - 1) {
                key_builder << ",";
            }
        }
    }

    // Add columns information (sort for consistent keys)
    if (!columns.is_none()) {
        key_builder << "_cols_";
        auto cols_list = columns.cast<std::vector<std::string>>();
        std::sort(cols_list.begin(), cols_list.end());
        for (size_t i = 0; i < cols_list.size(); ++i) {
            key_builder << cols_list[i];
            if (i < cols_list.size() - 1) {
                key_builder << ",";
            }
        }
    }

    // Add row range information
    if (start_row > 0) {
        key_builder << "_row_" << start_row;
        if (!num_rows.is_none()) {
            key_builder << "_count_" << num_rows.cast<long long>();
        }
    }

    return key_builder.str();
}

// Improved read_data function with better memory management and error handling
torch::Tensor read_data(fitsfile* fptr, torch::Device device, 
                       const std::vector<long>& start, const std::vector<long>& shape) {
    DEBUG_SCOPE
    
    if (!fptr) {
        throw std::invalid_argument("Null FITS file pointer");
    }
    
    int status = 0;
    int bitpix, naxis, anynul;
    long long naxes[3] = {1, 1, 1};

    // Get image parameters (dimensions and data type)
    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) {
        throw_fits_error(status, "Error getting image parameters");
    }

    DEBUG_LOG("Image parameters: naxis=" + std::to_string(naxis) + 
              ", dims=[" + std::to_string(naxes[0]) + "," + 
              std::to_string(naxis > 1 ? naxes[1] : 1) + "," + 
              std::to_string(naxis > 2 ? naxes[2] : 1) + "]" +
              ", bitpix=" + std::to_string(bitpix));

    // Check for supported dimensions
    if (naxis < 1 || naxis > 3) {
        throw std::runtime_error("Unsupported number of dimensions: " + std::to_string(naxis) +
                                " (only 1-3 dimensions are supported)");
    }
    
    // Validate start and shape parameters
    if (!start.empty() && !shape.empty()) {
        if (start.size() != shape.size()) {
            throw std::invalid_argument("start and shape must have the same length");
        }
        if(start.size() != naxis) {
            throw std::invalid_argument("start and shape must have length equal to naxis (" + 
                                       std::to_string(naxis) + ")");
        }
    }

    // The rest of the implementation...
    // (Note: Most of the existing READ_AND_RETURN implementation is good)
    
    // Example for one data type:
    if (bitpix == BYTE_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims; // Torch dims
            // CFITSIO pixel indices must be long
            std::vector<long> fpixel, lpixel, inc;

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - need to be reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) {
                    dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); // Reverse order
                }
            }
            
            // Create tensor AFTER calculating correct dims
            data = torch::empty(dims, options);
            uint8_t* data_ptr = data.data_ptr<uint8_t>();
            
            if (!fpixel.empty()) { // Read subset using calculated ranges
                // Use the standard fits_read_subset which expects long*
                if (fits_read_subset(fptr, TBYTE, fpixel.data(), lpixel.data(), inc.data(),
                                   nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TBYTE data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TBYTE, 1, nelements, nullptr,
                                data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TBYTE data");
                }
            }
            
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        }
        catch (const std::exception& e) {
            ERROR_LOG("Error reading BYTE_IMG data: " + std::string(e.what()));
            throw; // Re-throw after logging
        }
    }
    
    // Similar pattern for other data types...
    else if (bitpix == SHORT_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims;
            std::vector<long> fpixel, lpixel, inc; // Use long

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) { dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); }
            }
            data = torch::empty(dims, options);
            int16_t* data_ptr = data.data_ptr<int16_t>();

            if (!fpixel.empty()) { // Read subset
                if (fits_read_subset(fptr, TSHORT, fpixel.data(), lpixel.data(), inc.data(), nullptr, data_ptr, nullptr, &status)) { // Use standard version
                    throw_fits_error(status, "Error reading TSHORT data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TSHORT, 1, nelements, nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TSHORT data");
                }
            }
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        } catch (const std::exception& e) {
            ERROR_LOG("Error reading SHORT_IMG data: " + std::string(e.what()));
            throw;
        }
    }
    else if (bitpix == LONG_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt32).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims;
            std::vector<long> fpixel, lpixel, inc; // Use long

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) { dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); }
            }
            data = torch::empty(dims, options);
            int32_t* data_ptr = data.data_ptr<int32_t>();

            if (!fpixel.empty()) { // Read subset
                if (fits_read_subset(fptr, TINT, fpixel.data(), lpixel.data(), inc.data(), nullptr, data_ptr, nullptr, &status)) { // Use standard version
                    throw_fits_error(status, "Error reading TINT data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TINT, 1, nelements, nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TINT data");
                }
            }
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        } catch (const std::exception& e) {
            ERROR_LOG("Error reading LONG_IMG data: " + std::string(e.what()));
            throw;
        }
    }
    else if (bitpix == LONGLONG_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt64).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims;
            std::vector<long> fpixel, lpixel, inc; // Use long

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) { dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); }
            }
            data = torch::empty(dims, options);
            int64_t* data_ptr = data.data_ptr<int64_t>();

            if (!fpixel.empty()) { // Read subset
                if (fits_read_subset(fptr, TLONGLONG, fpixel.data(), lpixel.data(), inc.data(), nullptr, data_ptr, nullptr, &status)) { // Use standard version
                    throw_fits_error(status, "Error reading TLONGLONG data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TLONGLONG, 1, nelements, nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TLONGLONG data");
                }
            }
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        } catch (const std::exception& e) {
            ERROR_LOG("Error reading LONGLONG_IMG data: " + std::string(e.what()));
            throw;
        }
    }
    else if (bitpix == FLOAT_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims;
            std::vector<long> fpixel, lpixel, inc; // Use long

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) { dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); }
            }
            data = torch::empty(dims, options);
            float* data_ptr = data.data_ptr<float>();

            if (!fpixel.empty()) { // Read subset
                if (fits_read_subset(fptr, TFLOAT, fpixel.data(), lpixel.data(), inc.data(), nullptr, data_ptr, nullptr, &status)) { // Use standard version
                    throw_fits_error(status, "Error reading TFLOAT data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TFLOAT, 1, nelements, nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TFLOAT data");
                }
            }
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        } catch (const std::exception& e) {
            ERROR_LOG("Error reading FLOAT_IMG data: " + std::string(e.what()));
            throw;
        }
    }
    else if (bitpix == DOUBLE_IMG) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        torch::Tensor data;
        try {
            std::vector<int64_t> dims;
            std::vector<long> fpixel, lpixel, inc; // Use long

            if (!start.empty() && !shape.empty()) {
                fpixel.resize(naxis);
                lpixel.resize(naxis);
                inc.assign(naxis, 1); // All increments are 1
                dims.resize(naxis);

                for (int i = 0; i < naxis; ++i) {
                    // i iterates through Python/Torch dimensions (e.g., 0=plane, 1=row, 2=col for naxis=3)
                    // Python/Torch order: (NAXIS3, NAXIS2, NAXIS1)
                    int fits_dim_idx = naxis - 1 - i; // Corresponding FITS index (e.g., 2=NAXIS3, 1=NAXIS2, 0=NAXIS1)

                    // Use start[i] and shape[i] (Python order) with naxes[fits_dim_idx] (FITS order)
                    long long current_shape_py = (shape[i] == -1) ? (naxes[fits_dim_idx] - start[i]) : shape[i];
                    if (current_shape_py <= 0) {
                        throw std::runtime_error("Calculated shape for Python dimension " + std::to_string(i) + " is <= 0");
                    }

                    // Store dimension in Python order
                    dims[i] = current_shape_py;

                    // Store CFITSIO coordinates in FITS order
                    fpixel[fits_dim_idx] = static_cast<long>(start[i] + 1);              // 1-based start
                    lpixel[fits_dim_idx] = static_cast<long>(start[i] + current_shape_py); // 1-based inclusive end
                }
            } else {
                // Full image dimensions - reversed for Python/Torch
                dims.resize(naxis);
                for (int i = 0; i < naxis; ++i) { dims[i] = static_cast<int64_t>(naxes[naxis - 1 - i]); }
            }
            data = torch::empty(dims, options);
            double* data_ptr = data.data_ptr<double>();

            if (!fpixel.empty()) { // Read subset
                if (fits_read_subset(fptr, TDOUBLE, fpixel.data(), lpixel.data(), inc.data(), nullptr, data_ptr, nullptr, &status)) { // Use standard version
                    throw_fits_error(status, "Error reading TDOUBLE data subset");
                }
            } else { // Read full image
                long nelements = data.numel();
                if (fits_read_img(fptr, TDOUBLE, 1, nelements, nullptr, data_ptr, nullptr, &status)) {
                    throw_fits_error(status, "Error reading TDOUBLE data");
                }
            }
            // DEBUG_TENSOR("Read tensor", data); // Temporarily disable
            return data;
        } catch (const std::exception& e) {
            ERROR_LOG("Error reading DOUBLE_IMG data: " + std::string(e.what()));
            throw;
        }
    }
    else {
        throw std::runtime_error("Unsupported FITS data type: " + std::to_string(bitpix));
    }
}

// Read table data from FITS file
std::map<std::string, torch::Tensor> read_table_data(
    fitsfile* fptr,
    pybind11::object columns,
    int start_row,
    pybind11::object num_rows_obj,
    torch::Device device,
    std::shared_ptr<CacheEntry> entry
) {
    int status = 0;
    int num_cols, typecode;
    long long num_rows_total;

    // Get table dimensions
    if (fits_get_num_rowsll(fptr, &num_rows_total, &status)) {
        throw_fits_error(status, "Error getting number of rows in table");
    }
    if (fits_get_num_cols(fptr, &num_cols, &status)) {
        throw_fits_error(status, "Error getting number of columns in table");
    }

    // Determine which columns to read
    std::vector<int> selected_cols;
    if (!columns.is_none()) {
        // Read specified columns
        auto cols_list = columns.cast<std::vector<std::string>>();
        for (const auto& col_name : cols_list) {
            int col_num;
            if (fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &col_num, &status)) {
                throw_fits_error(status, "Error getting column number for: " + col_name);
            }
            selected_cols.push_back(col_num);
        }
    } else {
        // Read all columns
        for (int i = 1; i <= num_cols; ++i) {
            selected_cols.push_back(i);
        }
    }

    // Determine number of rows to read
    long long n_rows_to_read;
    if (!num_rows_obj.is_none()) {
        n_rows_to_read = num_rows_obj.cast<long long>();
    } else {
        n_rows_to_read = num_rows_total - start_row;  // All remaining rows
    }

    // Validate range
    if (start_row < 0) {
        throw std::runtime_error("start_row must be >= 0");
    }
    if (n_rows_to_read < 0) {
        throw std::runtime_error("num_rows must be >= 0");
    }
    if (start_row + n_rows_to_read > num_rows_total) {
        throw std::runtime_error("start_row + num_rows exceeds the total number of rows in the table");
    }

    // Read each column's data
    std::map<std::string, torch::Tensor> table_data;
    char col_name[FLEN_VALUE];

    for (int col_num : selected_cols) {
        // Get column type and dimensions
        long repeat, width;
        if (fits_get_coltype(fptr, col_num, &typecode, &repeat, &width, &status)) {
            throw_fits_error(status, "Error getting column type for column " + std::to_string(col_num));
        }

        // Get column name
        int col_idx = col_num; // Make a copy as fits_get_colname modifies this value
        if (fits_get_colname(fptr, CASEINSEN, const_cast<char*>("*"), col_name, &col_idx, &status)) {
            throw_fits_error(status, "Error getting column name for column " + std::to_string(col_num));
        }
        std::string col_name_str(col_name);

        // Macro to handle different numeric data types uniformly
        #define READ_COL_AND_STORE(cfitsio_type, torch_type, data_type) \
            { \
                auto tensor = torch::empty({n_rows_to_read}, torch::TensorOptions().dtype(torch_type).device(device)); \
                data_type* data_ptr = tensor.data_ptr<data_type>(); \
                if (fits_read_col(fptr, cfitsio_type, col_num, start_row + 1, 1, n_rows_to_read, nullptr, data_ptr, nullptr, &status)) { \
                    throw_fits_error(status, "Error reading column " + col_name_str + " (data type " #cfitsio_type ")"); \
                } \
                table_data[col_name_str] = tensor; \
            }

        // Handle different column data types
        if (typecode == TBYTE) {
            READ_COL_AND_STORE(TBYTE, torch::kUInt8, uint8_t);
        } else if (typecode == TSHORT) {
            READ_COL_AND_STORE(TSHORT, torch::kInt16, int16_t);
        } else if (typecode == TINT) {
            READ_COL_AND_STORE(TINT, torch::kInt32, int32_t);
        } else if (typecode == TLONG) {
            READ_COL_AND_STORE(TLONG, torch::kInt32, int32_t);
        } else if (typecode == TLONGLONG) {
            READ_COL_AND_STORE(TLONGLONG, torch::kInt64, int64_t);
        } else if (typecode == TFLOAT) {
            READ_COL_AND_STORE(TFLOAT, torch::kFloat32, float);
        } else if (typecode == TDOUBLE) {
            READ_COL_AND_STORE(TDOUBLE, torch::kFloat64, double);
        } else if (typecode == TSTRING) {
            // Special handling for string columns
            std::vector<char*> string_ptrs(n_rows_to_read, nullptr);
            std::vector<std::unique_ptr<char[]>> string_array(n_rows_to_read);

            // Allocate memory for each string
            for (int i = 0; i < n_rows_to_read; i++) {
                string_array[i] = std::make_unique<char[]>(width + 1);
                string_array[i][width] = '\0'; // Ensure null termination
                string_ptrs[i] = string_array[i].get();
            }

            // Read string data
            if (fits_read_col(fptr, TSTRING, col_num, start_row + 1, 1, n_rows_to_read, nullptr, string_ptrs.data(), nullptr, &status)) {
                throw_fits_error(status, "Error reading column " + col_name_str + " (data type TSTRING)");
            }

            // Convert C strings to std::string
            std::vector<std::string> string_list;
            string_list.reserve(n_rows_to_read);
            for (int i = 0; i < n_rows_to_read; i++) {
                string_list.emplace_back(string_ptrs[i]);
            }

            // Store in entry
            table_data[col_name_str] = torch::empty({0});  // Placeholder tensor
            entry->string_data[col_name_str] = std::move(string_list);
        } else {
            throw std::runtime_error("Unsupported column data type (" + std::to_string(typecode) +
                ") in column " + col_name_str);
        }
        #undef READ_COL_AND_STORE
    }

    return table_data;
}

// --- Main read implementation ---
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
) {
    DEBUG_SCOPE;
    
    try {
        // Initialize cache if needed (will be disabled if cache_capacity=0)
        ensure_cache_initialized(cache_capacity);
        
        // Convert filename_or_url to string
        std::string filename;
        if (pybind11::isinstance<pybind11::dict>(filename_or_url)) {
            filename = pybind11::str(filename_or_url).cast<std::string>();
        } else {
            filename = filename_or_url.cast<std::string>();
        }
        
        // HDU Selection
        int hdu_num = 1;  // Default to primary HDU
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                std::string hdu_str = hdu.cast<std::string>();
                try {
                    hdu_num = get_hdu_num_by_name(filename, hdu_str);
                } catch (const std::exception& e) {
                    DEBUG_LOG("Error getting HDU by name: " << e.what() << ", trying as index");
                    try {
                        // Try to convert string to integer
                        hdu_num = std::stoi(hdu_str);
                    } catch (...) {
                        // Re-throw the original error about HDU name
                        throw std::runtime_error("Cannot find HDU with name: " + hdu_str);
                    }
                }
            } else {
                hdu_num = hdu.cast<int>();
            }
        }
        
        // Initialize vectors for start and shape (if provided)
        std::vector<long> start_list;
        std::vector<long> shape_list;
        
        if (!start.is_none()) {
            if (shape.is_none()) {
                throw std::runtime_error("If 'start' is provided, 'shape' must also be provided");
            }
            
            if (!pybind11::isinstance<pybind11::sequence>(start)) {
                throw std::runtime_error("'start' must be a sequence (list or tuple)");
            }
            
            start_list = start.cast<std::vector<long>>();
        }
        
        if (!shape.is_none()) {
            if (start.is_none()) {
                throw std::runtime_error("If 'shape' is provided, 'start' must also be provided");
            }
            
            if (!pybind11::isinstance<pybind11::sequence>(shape)) {
                throw std::runtime_error("'shape' must be a sequence (list or tuple)");
            }
            
            shape_list = shape.cast<std::vector<long>>();
            
            if (shape_list.size() != start_list.size()) {
                throw std::runtime_error("'start' and 'shape' must have the same number of dimensions");
            }
            
            for (size_t i = 0; i < shape_list.size(); ++i) {
                if (shape_list[i] <= 0 && shape_list[i] != -1) {
                    throw std::runtime_error("Shape values must be > 0, or -1 for 'to the end'");
                }
            }
        }
        
        // Generate cache key only if cache is enabled
        std::string cache_key;
        if (cache_capacity > 0) {
            cache_key = generate_cache_key(filename, hdu, start, shape, columns, start_row, num_rows);
            
            // Check cache only if it's enabled
            if (auto cached_entry = cache->get(cache_key)) {
                DEBUG_LOG("Cache hit for key: " << cache_key);
                return pybind11::cast(*cached_entry);
            }
            
            DEBUG_LOG("Cache miss for key: " << cache_key);
        }
        
        // Open the file and move to the correct HDU
        int status = 0;
        fitsfile* fptr = nullptr;
        
        fits_open_file(&fptr, filename.c_str(), READONLY, &status);
        if (status) {
            throw_fits_error(status, "Error opening FITS file: " + filename);
        }
        
        // Use RAII to ensure fptr is closed
        struct FITSFileCloser {
            fitsfile* ptr;
            ~FITSFileCloser() {
                if (ptr) {
                    int status = 0;
                    fits_close_file(ptr, &status);
                }
            }
        } fits_closer{fptr};
        
        // Move to the requested HDU
        fits_movabs_hdu(fptr, hdu_num, nullptr, &status);
        if (status) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
        
        // Read header
        auto header = read_fits_header(fptr);
        
        // Determine HDU type
        int hdu_type;
        status = 0;
        fits_get_hdu_type(fptr, &hdu_type, &status);
        if (status) {
            throw_fits_error(status, "Error getting HDU type");
        }
        
        // Create cache entry only if cache is enabled
        std::shared_ptr<CacheEntry> new_entry;
        if (cache_capacity > 0) {
            new_entry = std::make_shared<CacheEntry>(torch::Tensor(), header);
        }
        
        // Process based on HDU type
        if (hdu_type == IMAGE_HDU) {
            // Check for empty image HDU (naxis == 0)
            int naxis = 0;
            status = 0;
            fits_get_img_dim(fptr, &naxis, &status);
            
            if (naxis == 0) {
                // Return None for data with header for empty HDU
                if (cache_capacity > 0) {
                    cache->put(cache_key, new_entry);
                }
                return pybind11::make_tuple(pybind11::none(), pybind11::cast(header));
            }
            
            // Read image data
            torch::Tensor data = read_data(fptr, device, start_list, shape_list);
            
            if (cache_capacity > 0) {
                new_entry->data = data;
                cache->put(cache_key, new_entry);
                return pybind11::cast(*new_entry);
            } else {
                return pybind11::make_tuple(data, pybind11::cast(header));
            }
        }
        else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            // Read table data
            auto table_data = read_table_data(fptr, columns, start_row, num_rows, device, new_entry);
            
            // Create Python dictionary with results
            pybind11::dict result_dict;
            for (const auto& [key, tensor] : table_data) {
                if (tensor.numel() == 0 && new_entry && new_entry->string_data.count(key) > 0) {
                    result_dict[key.c_str()] = pybind11::cast(new_entry->string_data[key]);
                } else {
                    result_dict[key.c_str()] = tensor;
                }
            }
            
            if (cache_capacity > 0) {
                cache->put(cache_key, new_entry);
            }
            return result_dict;
        }
        else {
            throw std::runtime_error("Unsupported HDU type");
        }
    }
    catch (const pybind11::error_already_set& e) {
        DEBUG_LOG("Python exception caught: " << e.what());
        throw;  // Re-throw Python exceptions
    }
    catch (const std::exception& e) {
        DEBUG_LOG("C++ exception caught: " << e.what());
        throw;  // Convert to Python exception
    }
    catch (...) {
        DEBUG_LOG("Unknown exception caught");
        throw std::runtime_error("Unknown error occurred in read_impl");
    }
}

// --- Helper functions ---

std::map<std::string, std::string> get_header_by_name(const std::string& filename, const std::string& hdu_name) {
    int hdu_num = get_hdu_num_by_name(filename, hdu_name);
    return get_header_by_number(filename, hdu_num);
}

std::map<std::string, std::string> get_header_by_number(const std::string& filename, int hdu_num) {
    return get_header(filename, hdu_num);
}
