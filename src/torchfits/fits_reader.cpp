#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "cache.h"
#include "debug.h"
#include <sstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

// Read  data from FITS file
torch::Tensor read_data(fitsfile* fptr, torch::Device device, const std::vector<long>& start, const std::vector<long>& shape) {
    int status = 0;
    int bitpix, naxis, anynul;
    long long naxes[3] = {1, 1, 1};

    // Get image parameters (dimensions and data type)
    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) {
        throw_fits_error(status, "Error getting image parameters");
    }

    // Check for supported dimensions
    if (naxis < 1 || naxis > 3) {
        throw std::runtime_error("Unsupported number of dimensions: " + std::to_string(naxis) +
            ". Only 1D, 2D, and 3D images are supported.");
    }

    // Calculate total number of elements
    long long nelements = 1;
    for (int i = 0; i < naxis; ++i) {
        // Check for multiplication overflow
        if (naxes[i] > 0 && nelements > std::numeric_limits<long long>::max() / naxes[i]) {
            throw std::runtime_error("Image dimensions too large, would cause integer overflow");
        }
        nelements *= naxes[i];
    }

    // Add cutout specification if requested
    if (!start.empty() && !shape.empty()) {

        if (start.size() != shape.size()) {
            throw std::runtime_error("'start' and 'shape' must have the same number of dimensions.");
        }
        if(start.size() != naxis){
            throw std::runtime_error("'start' and 'shape' must have the same number of dimensions as the image");
        }

        for (size_t i = 0; i < start.size(); ++i) {
            if(shape[i] <= 0 && shape[i] != -1)
                throw std::runtime_error("Shape values must be > 0, or -1 (None)");

            // Convert from 0-based indexing (Python) to 1-based (FITS)
            long long start_val = start[i] + 1;
            long long end_val;
            if (shape[i] == -1) {
                end_val = -1;  // CFITSIO convention for reading to end
            }
            long long naxes_tmp[3];
             // Get image parameters (dimensions and data type)
            if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes_tmp, &status)) {
                throw_fits_error(status, "Error getting image parameters");
            }
            if(end_val == -1){
                end_val = naxes_tmp[i];
            }
            
            // Instead of manipulating file pointers, store the start and end positions
            // for later use with fits_read_subset
            for (size_t j = 0; j < start.size(); ++j) {
                // Convert from 0-based indexing (Python) to 1-based (FITS)
                long start_val = start[j] + 1;
                long end_val;
                if (shape[j] == -1) {
                    end_val = naxes[j];  // Read to the end
                } else {
                    end_val = start[j] + shape[j];  // Specific shape
                }
                
                // Update the dimensions for the output tensor
                naxes[j] = end_val - start_val + 1;
            }
            
            // We'll use these values later in fits_read_subset

        }
    }

    // Create PyTorch tensor with appropriate data type and read data
    torch::TensorOptions options;

    #define READ_AND_RETURN(cfitsio_type, torch_type, data_type) \
        options = torch::TensorOptions().dtype(torch_type).device(device); \
        /* Create tensor with dimensions [z,y,x] for PyTorch compatibility */ \
        long dims[3] = {(naxis > 2) ? naxes[2] : 1, \
                       (naxis > 1) ? naxes[1] : 1, \
                       naxes[0]}; \
        if (!start.empty() && !shape.empty()) { \
            /* Apply cutout dimensions */ \
            for (size_t i = 0; i < start.size(); ++i) { \
                int axis = naxis - i - 1; /* Reverse axes for PyTorch */ \
                dims[axis] = (shape[i] == -1) ? naxes[i] - start[i] : shape[i]; \
            } \
        } \
        auto data = torch::empty({dims[0], dims[1], dims[2]}, options); \
        data_type* data_ptr = data.data_ptr<data_type>(); \
        \
        if (!start.empty() && !shape.empty()) { \
            /* Convert from 0-based (Python) to 1-based (FITS) indexing */ \
            long fpixel[3] = {1, 1, 1}; \
            long lpixel[3] = {naxes[0], naxes[1], naxes[2]}; \
            for (size_t i = 0; i < start.size(); ++i) { \
                fpixel[i] = start[i] + 1; \
                lpixel[i] = (shape[i] == -1) ? naxes[i] : start[i] + shape[i]; \
            } \
            long inc[3] = {1, 1, 1}; \
            if (fits_read_subset(fptr, cfitsio_type, fpixel, lpixel, inc, \
                               nullptr, data_ptr, nullptr, &status)) { \
                throw_fits_error(status, "Error reading " #cfitsio_type " data subset"); \
            } \
        } else { \
            if (fits_read_img(fptr, cfitsio_type, 1, nelements, nullptr, \
                            data_ptr, nullptr, &status)) { \
                throw_fits_error(status, "Error reading " #cfitsio_type " data"); \
            } \
        } \
        return data;

    // Select correct data type based on BITPIX
    if (bitpix == BYTE_IMG) {
        READ_AND_RETURN(TBYTE, torch::kUInt8, uint8_t);
    } else if (bitpix == SHORT_IMG) {
        READ_AND_RETURN(TSHORT, torch::kInt16, int16_t);
    } else if (bitpix == LONG_IMG) {
        READ_AND_RETURN(TINT, torch::kInt32, int32_t);
    } else if (bitpix == LONGLONG_IMG) {
        READ_AND_RETURN(TLONGLONG, torch::kInt64, int64_t);
    } else if (bitpix == FLOAT_IMG) {
        READ_AND_RETURN(TFLOAT, torch::kFloat32, float);
    } else if (bitpix == DOUBLE_IMG) {
        READ_AND_RETURN(TDOUBLE, torch::kFloat64, double);
    } else {
        throw std::runtime_error("Unsupported data type (BITPIX = " + std::to_string(bitpix) + ")");
    }
    #undef READ_AND_RETURN
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
    ensure_cache_initialized(cache_capacity);

    try {
        // Convert filename_or_url to string
        std::string filename;
        if (pybind11::isinstance<pybind11::dict>(filename_or_url)) {
            // Here is the change, we cast the dict to a string to be handled by CFITSIO
            filename = pybind11::str(filename_or_url).cast<std::string>();
        } else {
            filename = filename_or_url.cast<std::string>();
        }

        // Generate cache key for this request
        std::string cache_key = generate_cache_key(filename, hdu, start, shape, columns, start_row, num_rows);
        DEBUG_LOG("Cache key: " << cache_key);

        // Check if result is in cache
        if (auto cached_entry = cache->get(cache_key)) {
            DEBUG_LOG("Cache hit");
            return pybind11::cast(*cached_entry);
        }

        DEBUG_LOG("Cache miss, reading from file");

        // --- HDU Selection ---
        int hdu_num = 1;  // Default to primary HDU
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        // --- Construct effective filename with cutout specification ---
        std::string effective_filename = filename;

        // Add HDU selector if needed with cutout
        if (!start.is_none() && pybind11::isinstance<pybind11::int_>(hdu)) {
            effective_filename = filename + "[" + std::to_string(hdu_num) + "]";
        }

        // Add cutout specification if requested
        if (!start.is_none() || !shape.is_none()) {
            // Validate cutout parameters
            if (start.is_none() || shape.is_none()) {
                throw std::runtime_error("If 'start' is provided, 'shape' must also be provided, and vice-versa.");
            }
            if (!pybind11::isinstance<pybind11::sequence>(start) ||
                !pybind11::isinstance<pybind11::sequence>(shape)) {
                throw std::runtime_error("'start' and 'shape' must be sequences (e.g., lists or tuples).");
            }

            auto start_list = start.cast<std::vector<long>>();
            auto shape_list = shape.cast<std::vector<long>>();

            if (start_list.size() != shape_list.size()) {
                throw std::runtime_error("'start' and 'shape' must have the same number of dimensions.");
            }

            // Construct CFITSIO cutout string
            std::stringstream cutout_builder;
            cutout_builder << "[";
            for (size_t i = 0; i < start_list.size(); ++i) {
                if (shape_list[i] <= 0 && shape_list[i] != -1)
                    throw std::runtime_error("Shape values must be > 0, or -1 (None)");

                // Convert from 0-based indexing (Python) to 1-based (FITS)
                long long start_val = start_list[i] + 1;
                long long end_val;
                if (shape_list[i] == -1) {
                    end_val = -1;  // CFITSIO convention for reading to end
                }
                else {
                    end_val = start_list[i] + shape_list[i]; // end is inclusive
                }
                cutout_builder << start_val << ":" << end_val;
                if (i < start_list.size() - 1) {
                    cutout_builder << ",";
                }
            }
            cutout_builder << "]";
            effective_filename += cutout_builder.str();
        }

        // --- Open FITS file and read data ---
        DEBUG_LOG("Opening FITS file: " << effective_filename);
        FITSFile fits_file(effective_filename);
        fitsfile* fptr = fits_file.get();

        // Determine HDU type
        int hdu_type;
        int status = 0;
        if (fits_get_hdu_type(fptr, &hdu_type, &status)) {
            throw_fits_error(status, "Error getting HDU type");
        }

        // Create cache entry with header
        auto new_entry = std::make_shared<CacheEntry>(torch::empty({ 0 }), read_fits_header(fptr));
        std::unique_ptr<wcsprm> wcs = nullptr;

        // Process based on HDU type
        if (hdu_type == IMAGE_HDU) {
            // Read image data
            new_entry->data = read_data(fptr, device, pybind11::cast<std::vector<long>>(start), pybind11::cast<std::vector<long>>(shape));

            // Store in cache and return
            cache->put(cache_key, new_entry);
            return pybind11::cast(*new_entry);

        }
        else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            // Read table data
            auto table_data = read_table_data(fptr, columns, start_row, num_rows, device, new_entry);

            // Create Python dictionary with results
            pybind11::dict result_dict;
            for (const auto& [key, tensor] : table_data) {
                if (tensor.numel() == 0 && new_entry->string_data.count(key) > 0) {
                    // This was a string column
                    result_dict[key.c_str()] = pybind11::cast(new_entry->string_data[key]);
                }
                else {
                    result_dict[key.c_str()] = tensor;
                }
            }

            // Store in cache and return
            cache->put(cache_key, new_entry);
            return result_dict;
        }
        else {
            throw std::runtime_error("Unsupported HDU type.");
        }
    }
    catch (const std::exception& e) {
        DEBUG_LOG("Exception caught: " << e.what());
        throw;  // Re-throw the exception
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
