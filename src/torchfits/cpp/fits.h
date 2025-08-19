/**
 * FITS I/O engine header
 * 
 * High-performance FITS reading/writing using cfitsio with PyTorch integration
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <tuple>

// Forward declarations
struct FitsHandle;

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

// Internal structures
struct FitsHandle {
    void* fitsfile;  // fitsfile pointer from cfitsio
    std::string path;
    std::string mode;
    bool is_open;
    
    FitsHandle() : fitsfile(nullptr), is_open(false) {}
};