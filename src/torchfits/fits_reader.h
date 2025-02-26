#ifndef FITS_READER_H
#define FITS_READER_H

#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <list>
#include <unordered_map>
#include <mutex>

namespace py = pybind11;

// Debug logging macro
#ifdef DEBUG
    #include <iostream>
    #define DEBUG_LOG(x) std::cerr << "[DEBUG] " << __FILE__ << ":" << __LINE__ << " - " << x << std::endl
#else
    #define DEBUG_LOG(x)
#endif

// Forward declarations
std::string construct_url_from_params(const pybind11::dict& params);

// CacheEntry definition
struct CacheEntry {
    torch::Tensor data;
    std::map<std::string, std::string> header;
    std::map<std::string, std::vector<std::string>> string_data;  // For table string columns
    
    CacheEntry(torch::Tensor d, std::map<std::string, std::string> h)
        : data(std::move(d)), header(std::move(h)) {}
};

// LRUCache class declaration
class LRUCache {
public:
    explicit LRUCache(size_t capacity);
    void clear();
    void put(const std::string& key, const std::shared_ptr<CacheEntry>& entry);
    std::shared_ptr<CacheEntry> get(const std::string& key);
private:
    size_t capacity_;
    std::list<std::pair<std::string, std::shared_ptr<CacheEntry>>> cache_list_;
    std::unordered_map<std::string, typename std::list<std::pair<std::string, std::shared_ptr<CacheEntry>>>::iterator> cache_map_;
    std::mutex mutex_;
};

// Global cache declaration
extern std::unique_ptr<LRUCache> cache;

// Constants
static const std::string cutout_str = "";

// FITS utility functions
std::string fits_status_to_string(int status);
void throw_fits_error(int status, const std::string& message = "");
std::pair<std::string, std::string> parse_header_card(const char* card);
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num);
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key);
std::string get_hdu_type(const std::string& filename, int hdu_num);
int get_num_hdus(const std::string& filename);
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num);
std::vector<long long> get_dims(const std::string& filename, int hdu_num);

// HDU handling functions
int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name);
pybind11::dict get_header_by_name(const std::string& filename, const std::string& hdu_name);
pybind11::dict get_header_by_number(const std::string& filename, int hdu_num);

// Main read implementation
pybind11::object read_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    int start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    torch::Device device);

// WCS utility functions
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr);
std::pair<torch::Tensor, torch::Tensor> world_to_pixel(const torch::Tensor& world_coords, 
                                                      const std::map<std::string, std::string>& header);
std::pair<torch::Tensor, torch::Tensor> pixel_to_world(const torch::Tensor& pixel_coords, 
                                                      const std::map<std::string, std::string>& header);

#endif // FITS_READER_H