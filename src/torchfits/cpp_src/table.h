/**
 * Table I/O engine header
 * 
 * High-performance table reading/writing using cfitsio with PyTorch integration
 */

#pragma once

#include "torchfits_torch.h"
#include <fitsio.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace torchfits {

// TableReader class declaration
class TableReader {
public:
    TableReader(const std::string& filename, int hdu_num);
    TableReader(fitsfile* fptr, int hdu_num);
    ~TableReader();
    
    std::vector<std::string> get_column_names();
    std::unordered_map<std::string, torch::Tensor> read_columns(
        const std::vector<std::string>& column_names, 
        int start_row = 1, 
        int num_rows = -1
    );
    
    int64_t get_num_rows() const { return nrows_; }
    
private:
    fitsfile* fptr_;
    int hdu_num_;
    bool owns_fptr_;
    int64_t nrows_ = 0;
    int ncols_ = 0;
    bool is_ascii_ = false;
};

} // namespace torchfits

// C-style functions for table operations
void write_fits_table(const char* filename, const std::unordered_map<std::string, torch::Tensor>& tensor_dict, 
                     const std::map<std::string, std::string>& header, bool overwrite);
void append_rows(const char* filename, int hdu_num, const std::unordered_map<std::string, torch::Tensor>& tensor_dict);