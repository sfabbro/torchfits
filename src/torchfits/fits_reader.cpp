#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include <sstream>
#include <algorithm>
#include <list>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// --- Cross-Platform Memory Check ---
#ifdef _WIN32
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <mach/mach.h>    
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/sysinfo.h>
#endif

size_t get_available_memory() {
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return static_cast<size_t>(status.ullAvailPhys);
    #elif defined(__APPLE__)
        mach_port_t host_port = mach_host_self();
        mach_msg_type_number_t host_size = sizeof(vm_statistics_data_t) / sizeof(integer_t);
        vm_size_t page_size;
        vm_statistics_data_t vm_stat;
        
        host_page_size(host_port, &page_size);
        if (host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &host_size) != KERN_SUCCESS) {
            return 0;  // Return 0 on error
        }
        
        return static_cast<size_t>(vm_stat.free_count) * static_cast<size_t>(page_size);
    #else
        struct sysinfo info;
        sysinfo(&info);
        return static_cast<size_t>(info.freeram * info.mem_unit);
    #endif
}

// --- In-Memory LRU Cache ---

// LRUCache implementation
LRUCache::LRUCache(size_t capacity) : capacity_(capacity) {}

void LRUCache::clear() {
    DEBUG_LOG("Clearing cache");
    std::lock_guard<std::mutex> lock(mutex_);
    cache_list_.clear();
    cache_map_.clear();
}

void LRUCache::put(const std::string& key, const std::shared_ptr<CacheEntry>& entry) {
    DEBUG_LOG("Putting entry with key: " << key);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        DEBUG_LOG("Updating existing entry");
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        it->second->second = entry;
        return;
    }

    cache_list_.push_front({key, entry});
    cache_map_[key] = cache_list_.begin();

    if (cache_list_.size() > capacity_) {
        DEBUG_LOG("Cache full, removing oldest entry");
        auto last = cache_list_.end();
        --last;
        cache_map_.erase(last->first);
        cache_list_.pop_back();
    }
}

std::shared_ptr<CacheEntry> LRUCache::get(const std::string& key) {
    DEBUG_LOG("Getting entry with key: " << key);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        DEBUG_LOG("Cache miss");
        return nullptr;
    }
    DEBUG_LOG("Cache hit");
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
}

// --- Core Data Reading Logic  ---

torch::Tensor read_data(fitsfile* fptr, std::unique_ptr<wcsprm>& wcs, torch::Device device) {
    int status = 0;
    int bitpix, naxis, anynul;
    long long naxes[3] = {1, 1, 1};  // CFITSIO supports up to 999 dimensions
    long long nelements;

    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) {
        throw_fits_error(status, "Error getting image parameters");
    }

     // Check for supported dimensions (1D, 2D, or 3D)
    if (naxis < 1 || naxis > 3) {
        throw std::runtime_error("Unsupported number of dimensions: " + std::to_string(naxis) + ". Only 1D, 2D, and 3D images are supported.");
    }

    nelements = 1;
    for (int i = 0; i < naxis; ++i) {
        nelements *= naxes[i];
    }

    // --- WCS Handling ---
    auto updated_wcs = read_wcs_from_header(fptr); // wcs_utils.cpp
    if (updated_wcs) {
        wcs = std::move(updated_wcs);  // Take ownership if WCS is valid
    }

    torch::TensorOptions options;
    // Use a macro for code reuse.
    #define READ_AND_RETURN(cfitsio_type, torch_type, data_type) \
        options = torch::TensorOptions().dtype(torch_type).device(device); \
        /* Create tensor with correct dimensions and order (z,y,x) for Pytorch compatibility*/  \
        auto data = torch::empty({(naxis > 2) ? naxes[2] : 1,  \
                                 (naxis > 1) ? naxes[1] : 1,  \
                                 naxes[0]}, options); \
        data_type* data_ptr = data.data_ptr<data_type>(); \
        if (fits_read_img(fptr, cfitsio_type, 1, nelements, nullptr, data_ptr, &anynul, &status)) { \
            throw_fits_error(status, "Error reading " #cfitsio_type " data"); \
        } \
        return data;

    // Select the appropriate read function based on BITPIX.
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

// --- Core Data Reading Logic (Binary and ASCII Tables) ---
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

    if (fits_get_num_rowsll(fptr, &num_rows_total, &status)) {
        throw_fits_error(status, "Error getting number of rows in table");
    }
    if (fits_get_num_cols(fptr, &num_cols, &status)) {
        throw_fits_error(status, "Error getting number of columns in table");
    }

    // --- Column Selection ---
    std::vector<int> selected_cols;  // Store selected column *numbers*
    if (!columns.is_none()) {
        auto cols_list = columns.cast<std::vector<std::string>>();
        for (const auto& col_name : cols_list) {
            int col_num;
            // fits_get_colnum is case-insensitive.
            if (fits_get_colnum(fptr, CASEINSEN, (char*)col_name.c_str(), &col_num, &status)) {
                 if (fits_close_file(fptr, &status)) { // Close file
                    throw_fits_error(status, "Error closing file");
                }
                throw_fits_error(status, "Error getting column number for: " + col_name);
            }
            selected_cols.push_back(col_num);
        }
    } else {
        // No columns specified: read all columns.
        for (int i = 1; i <= num_cols; ++i) {
            selected_cols.push_back(i);
        }
    }
     // --- Row Handling ---
    long long n_rows_to_read;
    if (!num_rows_obj.is_none()) {
        n_rows_to_read = num_rows_obj.cast<long long>();
    } else {
        n_rows_to_read = num_rows_total - start_row;  // Read all remaining rows
    }
     // Check for valid start_row and num_rows
    if (start_row < 0) {
        throw std::runtime_error("start_row must be >= 0");
    }
    if (n_rows_to_read < 0) {
        throw std::runtime_error("num_rows must be >= 0");
    }
    if (start_row + n_rows_to_read > num_rows_total) {
        throw std::runtime_error("start_row + num_rows exceeds the total number of rows in the table.");
    }

    std::map<std::string, torch::Tensor> table_data;
    char col_name[FLEN_VALUE];
    //Iterate only on selected columns
    for (int col_num : selected_cols) {
        long repeat_l, width_l;  // Use long instead of long long
        if (fits_get_coltype(fptr, col_num, &typecode, &repeat_l, &width_l, &status)) {
            throw_fits_error(status, "Error getting column type for column " + std::to_string(col_num));
        }
        //Convert to long long after the call if needed
        long long repeat = repeat_l;
        long long width = width_l;
        //Get column name
        char template_str[] = "*";  // Create a modifiable char array
        if (fits_get_colname(fptr, CASEINSEN, template_str, col_name, &col_num, &status)) {
            throw_fits_error(status, "Error getting column name for column " + std::to_string(col_num));
        }
        std::string col_name_str(col_name);

        #define READ_COL_AND_STORE(cfitsio_type, torch_type, data_type) \
            { \
                auto tensor = torch::empty({n_rows_to_read}, torch::TensorOptions().dtype(torch_type).device(device)); \
                data_type* data_ptr = tensor.data_ptr<data_type>(); \
                if (fits_read_col(fptr, cfitsio_type, col_num, start_row + 1, 1, n_rows_to_read, nullptr, data_ptr, nullptr, &status)) { \
                    throw_fits_error(status, "Error reading column " + col_name_str + " (data type " #cfitsio_type ")"); \
                } \
                table_data[col_name_str] = tensor; \
            }

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
            std::vector<char*> string_array(n_rows_to_read);
            for (int i = 0; i < n_rows_to_read; i++) {
                string_array[i] = new char[width + 1];
                string_array[i][width] = '\0';
            }

            char template_str[] = "*";
            if (fits_get_colname(fptr, CASEINSEN, template_str, col_name, &col_num, &status)) {
                for (int i = 0; i < n_rows_to_read; i++) {
                    delete[] string_array[i];
                }
                throw_fits_error(status, "Error getting column name for column " + std::to_string(col_num));
            }

            if (fits_read_col(fptr, TSTRING, col_num, start_row + 1, 1, n_rows_to_read, nullptr, string_array.data(), nullptr, &status)) {
                for (int i = 0; i < n_rows_to_read; i++) {
                    delete[] string_array[i];
                }
                throw_fits_error(status, "Error reading column " + col_name_str + " (data type TSTRING)");
            }

            std::vector<std::string> string_list;
            for (int i = 0; i < n_rows_to_read; i++) {
                string_list.emplace_back(string_array[i]);
                delete[] string_array[i];
            }
            
            table_data[col_name_str] = torch::empty({0});  // Placeholder tensor
            entry->string_data[col_name_str] = string_list;  // Store strings in the entry
        } else {
            if (fits_close_file(fptr, &status)) {
                throw_fits_error(status, "Error closing file");
            }
            throw std::runtime_error("Unsupported column data type (" + std::to_string(typecode) + ") in column " + col_name_str);
        }
        #undef READ_COL_AND_STORE
    }

    return table_data;
}

// --- Main `read` function (Handles Images, Tables, Cutouts, and HDU Selection) ---

// Define the global cache variable
std::unique_ptr<LRUCache> cache;
static std::mutex init_mutex;

// Keep this implementation:
std::string construct_url_from_params(const pybind11::dict& params) {
    std::string protocol = params["protocol"].cast<std::string>();
    std::string host = params["host"].cast<std::string>();
    std::string path = params["path"].cast<std::string>();
    return protocol + "://" + host + "/" + path;
}

void ensure_cache_initialized(size_t cache_capacity) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!cache) {
        DEBUG_LOG("Initializing cache with capacity: " << cache_capacity);
        size_t capacity = (cache_capacity > 0) ? 
            cache_capacity : 
            static_cast<size_t>(0.25 * get_available_memory() / (1024 * 1024));
        capacity = std::min(capacity, static_cast<size_t>(2048));  // Limit to 2GB
        DEBUG_LOG("Final cache capacity: " << capacity << " MB");
        cache = std::make_unique<LRUCache>(capacity);
    }
}

pybind11::object read(pybind11::object filename_or_url, pybind11::object hdu,
                      pybind11::object start, pybind11::object shape,
                      pybind11::object columns, int start_row, pybind11::object num_rows,
                      size_t cache_capacity, torch::Device device) {
    
    DEBUG_LOG("Starting read operation");
    ensure_cache_initialized(cache_capacity);
    
    fitsfile* fptr = nullptr;
    int status = 0;

    try {
        std::string filename;
        if (pybind11::isinstance<pybind11::dict>(filename_or_url)) {
            auto params = filename_or_url.cast<pybind11::dict>();
            filename = construct_url_from_params(params);
        } else {
            filename = filename_or_url.cast<std::string>();
        }

        // Construct cache key
        std::string cache_key = filename;
        if (!hdu.is_none()) {
            cache_key += "_hdu_" + std::to_string(hdu.cast<int>());
        }
        if (!start.is_none()) {
            auto start_list = start.cast<std::vector<long>>();
            for (const auto& s : start_list) {
                cache_key += "_" + std::to_string(s);
            }
        }
        
        DEBUG_LOG("Cache key: " << cache_key);

        // Check cache
        if (auto cached_entry = cache->get(cache_key)) {
            DEBUG_LOG("Cache hit");
            return pybind11::cast(*cached_entry);
        }
        
        DEBUG_LOG("Cache miss, reading from file");

        // --- Remote File Handling ---
        if (pybind11::isinstance<pybind11::dict>(filename_or_url)) {
            auto fsspec_params = filename_or_url.cast<std::map<std::string, std::string>>();
            std::string protocol = fsspec_params["protocol"];
            std::string host = fsspec_params["host"];
            std::string path = fsspec_params["path"];
            filename = protocol + "://" + host + "/" + path;
        }

        // --- HDU Handling ---
        int hdu_num = 1;  // Default to primary HDU
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        // --- Cutout Handling (start and shape) ---
        if (!start.is_none() || !shape.is_none()) {
             //If hdu is a number, reconstruct the full filename
            if (pybind11::isinstance<pybind11::int_>(hdu)) {
                filename = filename + "[" + std::to_string(hdu_num) + "]";
            }
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

            // Construct CFITSIO cutout string.
            std::stringstream cutout_builder;
            cutout_builder << "[";
            for (size_t i = 0; i < start_list.size(); ++i) {
                 if(shape_list[i] <= 0 && shape_list[i] != -1 ) // Use -1 as None
                    throw std::runtime_error("Shape values must be > 0, or -1 (None)");
                // Special case: None in shape means read to the end.
                long long start_val = start_list[i] + 1;  // FITS indexing
                long long end_val;
                if (shape_list[i] == -1) {
                    end_val = -1;  // CFITSIO convention for reading to end
                } else {
                    end_val = start_list[i] + shape_list[i]; // end is inclusive
                }
                cutout_builder << start_val << ":" << end_val;
                if (i < start_list.size() - 1) {
                    cutout_builder << ",";
                }
            }
            cutout_builder << "]";
            filename = filename + cutout_builder.str(); // Append to filename
        }

        // --- File Opening (using constructed filename) ---
        DEBUG_LOG("Opening FITS file");
        if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
            throw_fits_error(status, "Error opening FITS file: " + filename);
        }

        int hdu_type;
        if (fits_get_hdu_type(fptr, &hdu_type, &status)) {
            if (fits_close_file(fptr, &status)) { // Close file
               throw_fits_error(status, "Error closing file");
            }
            throw_fits_error(status, "Error getting HDU type");
        }

        // Create cache entry
        auto new_entry = std::make_shared<CacheEntry>(torch::empty({0}), read_fits_header(fptr));

        if (hdu_type == IMAGE_HDU) {
            auto wcs = read_wcs_from_header(fptr);  // From wcs_utils.cpp
            new_entry->data = read_data(fptr, wcs, device);
            if (fits_close_file(fptr, &status)) {
                throw_fits_error(status, "Error closing file");
            }
            // Check for empty primary HDU
            if (new_entry->data.numel() == 0) {
                new_entry->data = torch::empty({0});  // Use empty tensor instead of None
            }
            cache->put(cache_key, new_entry);
            return pybind11::cast(*new_entry);

        } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            auto table_data = read_table_data(fptr, columns, start_row, num_rows, device, new_entry);
            if (fits_close_file(fptr, &status)) {
                throw_fits_error(status, "Error closing file");
            }
            
            // Create a Python dictionary to return both tensor and string data
            pybind11::dict result_dict;
            for (const auto& [key, tensor] : table_data) {
                if (tensor.numel() == 0 && new_entry->string_data.count(key) > 0) {
                    // This was a string column
                    result_dict[key.c_str()] = pybind11::cast(new_entry->string_data[key]);
                } else {
                    result_dict[key.c_str()] = tensor;
                }
            }
            
            cache->put(cache_key, new_entry);
            return result_dict;

        } else {
             if (fits_close_file(fptr, &status)) { // Close file
               throw_fits_error(status, "Error closing file");
            }
            throw std::runtime_error("Unsupported HDU type.");
        }
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception caught: " << e.what());
        if (fptr) {
            status = 0;
            DEBUG_LOG("Closing FITS file");
            fits_close_file(fptr, &status);
        }
        throw;  // Re-throw the exception
    }
}

// Change this function signature
pybind11::object read_impl(
    const std::string& filename,  // This is const, can't be modified
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    int start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    torch::Device device
) {
    int status = 0;
    fitsfile* fptr;
    
    // Create a modifiable copy of filename
    std::string file_path = filename;
    
    // Open the FITS file
    fits_open_file(&fptr, file_path.c_str(), READONLY, &status);
    if (status) {
        throw_fits_error(status, "Error opening file: " + file_path);
    }

    // Move to the specified HDU
    int hdu_num = 1;  // Default to primary HDU
    if (!hdu.is_none()) {
        if (py::isinstance<py::str>(hdu)) {
            hdu_num = get_hdu_num_by_name(file_path, hdu.cast<std::string>());
        } else {
            hdu_num = hdu.cast<int>();
        }
    }

    // Modify the copy instead of the const reference
    file_path = file_path + "[" + std::to_string(hdu_num) + "]";

    // Create cache entry
    auto entry = std::make_shared<CacheEntry>(torch::empty({0}), read_fits_header(fptr));
    
    try {
        int hdu_type;
        fits_get_hdu_type(fptr, &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");

        if (hdu_type == IMAGE_HDU) {
            // Handle image data
            std::unique_ptr<wcsprm> wcs;
            entry->data = read_data(fptr, wcs, device);
            entry->header = read_fits_header(fptr);
            
            fits_close_file(fptr, &status);
            return pybind11::cast(*entry);
        } 
        else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            // Handle table data
            auto table_data = read_table_data(
                fptr, columns, start_row,
                num_rows, device, entry
            );
            entry->header = read_fits_header(fptr);
            
            fits_close_file(fptr, &status);
            return pybind11::cast(*entry);
        }
        else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Unsupported HDU type");
        }
    }
    catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

pybind11::dict get_header_by_name(const std::string& filename, const std::string& hdu_name) {
    int status = 0;
    fitsfile* fptr;
    
    // Open the FITS file
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        throw_fits_error(status, "Error opening file: " + filename);
    }

    try {
        // Find HDU by name
        int hdu_num = get_hdu_num_by_name(filename, hdu_name);
        
        // Move to the HDU
        int hdu_type;
        fits_movabs_hdu(fptr, hdu_num, &hdu_type, &status);
        if (status) throw_fits_error(status, "Error moving to HDU: " + hdu_name);

        // Read the header
        auto header = read_fits_header(fptr);
        
        fits_close_file(fptr, &status);
        return pybind11::cast(header);
    }
    catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

pybind11::dict get_header_by_number(const std::string& filename, int hdu_num) {
    int status = 0;
    fitsfile* fptr;
    
    // Open the FITS file
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        throw_fits_error(status, "Error opening file: " + filename);
    }

    try {
        // Move to the HDU
        int hdu_type;
        fits_movabs_hdu(fptr, hdu_num, &hdu_type, &status);
        if (status) throw_fits_error(status, "Error moving to HDU number: " + std::to_string(hdu_num));

        // Read the header
        auto header = read_fits_header(fptr);
        
        fits_close_file(fptr, &status);
        return pybind11::cast(header);
    }
    catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}

int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name) {
    int status = 0;
    fitsfile* fptr;
    
    // Open the FITS file
    fits_open_file(&fptr, filename.c_str(), READONLY, &status);
    if (status) {
        throw_fits_error(status, "Error opening file: " + filename);
    }

    try {
        int hdu_num = 1;
        int hdu_type;
        char extname[FLEN_VALUE];
        
        // Loop through HDUs to find the named one
        while (status == 0) {
            fits_movabs_hdu(fptr, hdu_num, &hdu_type, &status);
            if (status) break;
            
            // Try to read EXTNAME keyword
            status = 0; // Reset status before reading keyword
            if (fits_read_key_str(fptr, "EXTNAME", extname, nullptr, &status) == 0) {
                if (hdu_name == extname) {
                    fits_close_file(fptr, &status);
                    return hdu_num;
                }
            }
            status = 0; // Reset status for next iteration
            hdu_num++;
        }
        
        fits_close_file(fptr, &status);
        throw std::runtime_error("HDU with name '" + hdu_name + "' not found");
    }
    catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
}
