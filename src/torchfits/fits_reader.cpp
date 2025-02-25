#include "fits_reader.h"
#include "wcs_utils.h"
#include <sstream>
#include <algorithm>
#include <list>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <pybind11/stl.h> // Make sure this is included!
#include <fsspec.h> // Include fsspec for remote file support

// --- Cross-Platform Memory Check ---
#ifdef _WIN32
#include <windows.h>
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
    #else
        struct sysinfo info;
        sysinfo(&info);
        return static_cast<size_t>(info.freeram * info.mem_unit);
    #endif
}

// --- In-Memory LRU Cache ---
// We'll use a simple struct to hold the cached data and header.
struct CacheEntry {
    torch::Tensor data;
    std::map<std::string, std::string> header;
};

class LRUCache {
public:
    LRUCache(size_t capacity) : capacity_(capacity) {}

    // Get an item from the cache.
    std::shared_ptr<CacheEntry> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return nullptr;  // Not found
        }
        // Move the accessed item to the front of the list (most recently used).
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return it->second->second; // Return the CacheEntry
    }

    // Put an item into the cache.
    void put(const std::string& key, const std::shared_ptr<CacheEntry>& entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Already in cache; move to front.
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            it->second->second = entry; // Update the value (in case it changed)
            return;
        }

        // Not in cache.  Add it.
        cache_list_.push_front({key, entry});
        cache_map_[key] = cache_list_.begin();

        // Evict LRU item if necessary.
        if (cache_list_.size() > capacity_) {
            auto last = cache_list_.end();
            --last;  // Get an iterator to the last element
            cache_map_.erase(last->first);
            cache_list_.pop_back();
        }
    }
     // Method to clear the cache (useful for testing or resource management).
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_list_.clear();
        cache_map_.clear();
    }


private:
    size_t capacity_;
    std::list<std::pair<std::string, std::shared_ptr<CacheEntry>>> cache_list_; //List for recency
    std::unordered_map<std::string, typename std::list<std::pair<std::string, std::shared_ptr<CacheEntry>>>::iterator> cache_map_; //Map to list elements
    std::mutex mutex_; // Mutex for thread safety
};

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
        throw_fits_error(0, "Unsupported data type (BITPIX = " + std::to_string(bitpix) + ")");
    }
    #undef READ_AND_RETURN
}

// --- Core Data Reading Logic (Binary and ASCII Tables) ---
std::map<std::string, torch::Tensor> read_table_data(fitsfile* fptr, pybind11::object columns, int start_row, pybind11::object num_rows_obj) {
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
        long long repeat, width;
        if (fits_get_coltype(fptr, col_num, &typecode, &repeat, &width, &status)) {
            throw_fits_error(status, "Error getting column type for column " + std::to_string(col_num));
        }
        //Get column name
        if (fits_get_colname(fptr, CASEINSEN, "*", col_name, &col_num, &status)) {
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
            table_data[col_name_str] = pybind11::cast(string_list);
        } else {
            if (fits_close_file(fptr, &status)) {
                throw_fits_error(status, "Error closing file");
            }
            throw_fits_error(0, "Unsupported column data type (" + std::to_string(typecode) + ") in column " + col_name_str);
        }
        #undef READ_COL_AND_STORE
    }

    return table_data;
}
// --- Main `read` function (Handles Images, Tables, Cutouts, and HDU Selection) ---

pybind11::object read(pybind11::object filename_or_url, pybind11::object hdu,
                      pybind11::object start, pybind11::object shape,
                      pybind11::object columns, int start_row, pybind11::object num_rows,
                      size_t cache_capacity, torch::Device device) {

    fitsfile* fptr = nullptr;
    int status = 0;
    int hdu_type;
    int hdu_num = 0; // Default to primary HDU (0 for CFITSIO).

    std::string filename;
    std::string cutout_str;

    //Check if filename_with_cutout contains brackets
    size_t first_bracket = filename_or_url.cast<std::string>().find('[');
    if(first_bracket !=  std::string::npos) {
        //Cutout
        filename = filename_or_url.cast<std::string>().substr(0, first_bracket);
        cutout_str = filename_or_url.cast<std::string>().substr(first_bracket);
    }
    else{
        //No cutout
        filename = filename_or_url.cast<std::string>();
        cutout_str = "";
    }

    // --- Remote File Handling ---
    if (pybind11::isinstance<pybind11::dict>(filename_or_url)) {
        auto fsspec_params = filename_or_url.cast<std::map<std::string, std::string>>();
        std::string protocol = fsspec_params["protocol"];
        std::string host = fsspec_params["host"];
        std::string path = fsspec_params["path"];
        std::string url = protocol + "://" + host + "/" + path;
        filename = url;
    }

    // --- HDU Handling ---
    if (!hdu.is_none()) {
        if (pybind11::isinstance<pybind11::int_>(hdu)) {
            hdu_num = hdu.cast<int>();
             if(hdu_num < 0){
                throw std::runtime_error("HDU number must be > 0");
            }
        } else if (pybind11::isinstance<pybind11::str>(hdu)) {
            filename = filename + "[" + hdu.cast<std::string>() + "]" + cutout_str;
        } else {
            throw std::runtime_runtime_error("Invalid 'hdu' argument.  Must be int or str.");
        }
    }  else { //If not, append cutout if exists
        filename = filename + cutout_str;
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


    // --- Caching (Lookup) ---
    // Initialize the cache (if it hasn't been initialized yet).
    static std::unique_ptr<LRUCache> cache = nullptr;
    static std::mutex init_mutex; //Mutex for thread safety
    {   //limit scope of lock guard
        std::lock_guard<std::mutex> init_lock(init_mutex); //Lock for initialization
        if (!cache) {
            size_t capacity = (cache_capacity > 0) ? cache_capacity : static_cast<size_t>(0.25 * get_available_memory() / (1024 * 1024)); // Use 25% of available RAM (in MB) if 0.
            //Limit to 2GB (avoid excesive memory usage)
            capacity = std::min(capacity, static_cast<size_t>(2048));
            std::cout<<"Initializing cache with capacity: "<< capacity << " MB" << std::endl; //Debug
            cache = std::make_unique<LRUCache>(capacity);
        }
    }

    std::string cache_key;
    //Create the cache key
    std::stringstream key_builder;
    key_builder << filename;  // Filename is already cutout and hdu aware.
     // Add column selection to the cache key, if applicable
    if (!columns.is_none()) {
        auto cols_list = columns.cast<std::vector<std::string>>();
        for (const auto& col : cols_list) {
            key_builder << "_col_" << col;
        }
    }
    // Add row selection to the cache key.
    key_builder << "_row_" << start_row;
    if (!num_rows.is_none()) {
         key_builder << "_" << num_rows.cast<long long>();
    }
    cache_key = key_builder.str();

    std::shared_ptr<CacheEntry> cached_entry = cache->get(cache_key);
    if (cached_entry) {
        // Cache hit! Return the cached data.
        if (hdu_type == IMAGE_HDU){ //Cannot determine hdu_type before opening. So, check it here.
            return pybind11::make_tuple(cached_entry->data, cached_entry->header);
        }
        else{
            return pybind11::cast(cached_entry->data); //Return the tensor
        }
    }

    // --- Cache Miss: Read Data ---

    // --- File Opening (using constructed filename) ---
    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file/cutout: " + filename);
    }

    if (fits_get_hdu_type(fptr, &hdu_type, &status)) {
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error getting HDU type");
    }

     //Create cache entry
    auto new_entry = std::make_shared<CacheEntry>();

    if (hdu_type == IMAGE_HDU) {
        auto wcs = read_wcs_from_header(fptr);  // From wcs_utils.cpp
        new_entry->data = read_data(fptr, wcs, device); // Store the tensor, pass device
        new_entry->header = read_fits_header(fptr);  // From fits_utils.cpp. Store the header.
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        // Check for empty primary HDU
        if (new_entry->data.numel() == 0) {
            new_entry->data = pybind11::none();
        }
        cache->put(cache_key, new_entry); //Cache it. Use ->
        return pybind11::make_tuple(new_entry->data, new_entry->header);  // Return tuple for images

    } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
        auto table_data = read_table_data(fptr, columns, start_row, num_rows);
        new_entry->header = read_fits_header(fptr); // From fits_utils.cpp
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        //Combine the map into a single tensor (required by the cache)
        if (table_data.size() > 0){
            int i = 0;
            for (auto const& [key, val] : table_data){ //find first element to get the shape
                if (i==0){ //First iteration, create tensor
                    //Check if tensor is string
                    if(pybind11::isinstance<pybind11::list>(val))
                        break; //For strings we do nothing.
                    new_entry->data = torch::empty({int(table_data.size()), val.size(0)},val.options().device(device)); // Pass device
                }
                new_entry->data[i] = val.to(device); //Copy to the tensor, to device
                i++;
            }
        }
        cache->put(cache_key, new_entry); //Cache the result
        return pybind11::cast(table_data);  // Return dict for tables

    } else {
         if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw std::runtime_error("Unsupported HDU type.");
    }
}
