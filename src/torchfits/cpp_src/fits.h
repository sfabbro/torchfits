/**
 * FITS I/O engine header
 * 
 * High-performance FITS reading/writing using cfitsio with PyTorch integration
 */

#pragma once

#include <torch/torch.h>
#include <fitsio.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>

namespace torchfits {

// Forward declarations
struct FitsHandle;

// FITSFile class declaration
class FITSFile {
public:
    FITSFile(const std::string& filename, int mode = 0);
    ~FITSFile();
    
    torch::Tensor read_image(int hdu_num = 0, bool use_mmap = false);
    torch::Tensor read_image_with_device(int hdu_num, bool use_mmap, torch::Device device);
    bool write_image(const torch::Tensor& tensor, int hdu_num = 0, double bscale = 1.0, double bzero = 0.0);
    
    std::map<std::string, std::string> get_header(int hdu_num = 0);
    std::string read_header_to_string(int hdu_num = 0);
    std::vector<int64_t> get_shape(int hdu_num = 0);
    torch::ScalarType get_dtype(int hdu_num = 0);
    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2);
    std::map<std::string, double> compute_stats(int hdu_num = 0);
    
    int get_num_hdus();
    std::string get_hdu_type(int hdu_num = 0);
    bool write_hdus(const std::vector<void*>& hdus, bool overwrite);
    fitsfile* get_fptr() { return fptr_; }
    
private:
    std::string filename_;
    fitsfile* fptr_;
    int mode_;
    bool cached_;
};

} // namespace torchfits

// Legacy C-style function declarations for backward compatibility
struct FitsHandle {
    void* fitsfile;  // fitsfile pointer from cfitsio
    std::string path;
    std::string mode;
    bool is_open;
    
    FitsHandle() : fitsfile(nullptr), is_open(false) {}
};

// File operations
FitsHandle* open_fits_file(const std::string& path, const std::string& mode = "r");
void close_fits_file(FitsHandle* handle);

// HDU information
int get_num_hdus(FitsHandle* handle);
std::string get_hdu_type(FitsHandle* handle, int hdu_index);
std::map<std::string, std::string> read_header(FitsHandle* handle, int hdu_index);

// Data access
std::vector<int64_t> get_shape(FitsHandle* handle, int hdu_index);
torch::ScalarType get_dtype(FitsHandle* handle, int hdu_index);
torch::Tensor read_full(FitsHandle* handle, int hdu_index);
torch::Tensor read_subset(FitsHandle* handle, int hdu_index, const std::vector<std::tuple<int64_t, int64_t>>& slice_spec);

// Compression support
std::vector<int64_t> get_tile_dims(FitsHandle* handle, int hdu_index);

// Statistics
std::map<std::string, double> compute_stats(FitsHandle* handle, int hdu_index);

// Table operations
int64_t get_num_rows(FitsHandle* handle, int hdu_index);
std::vector<std::string> get_col_names(FitsHandle* handle, int hdu_index);
std::map<std::string, int> infer_feat_types(FitsHandle* handle, int hdu_index);

// Table materialization (placeholder - would return TensorFrame in full implementation)
void* materialize_table(FitsHandle* handle, int hdu_index, 
                       const std::vector<std::string>& selections,
                       const std::string& filters,
                       int64_t limit);

// Write operations
void write_fits_file(const std::string& path, const std::vector<void*>& hdus, bool overwrite);