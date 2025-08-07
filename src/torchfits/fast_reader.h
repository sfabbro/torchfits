#pragma once
#include <torch/torch.h>
#include "fitsio.h"

namespace torchfits_fast {

/**
 * Fast FITS reading optimizations inspired by fitsio
 * Key insights from fitsio analysis:
 * 1. Use fits_read_pixll for large data reads (better than fits_read_pix for big arrays)
 * 2. Minimize type conversions and memory allocations
 * 3. Read directly into pre-allocated buffers
 * 4. Use CFITSIO's native data types when possible
 */

// Forward declarations
class FastImageReader {
public:
    /**
     * Read image directly into PyTorch tensor using fitsio-style optimizations
     * This mimics fitsio's PyFITSObject_read_image approach
     */
    static torch::Tensor read_image_fast(fitsfile* fptr, torch::Device device = torch::kCPU);
    
private:
    template<typename T, int CfitsioType>
    static torch::Tensor read_image_typed_fast(fitsfile* fptr, torch::Device device);
};

class FastTableReader {
public:
    /**
     * Read table columns using fitsio-inspired bulk operations
     */
    static pybind11::dict read_table_fast(
        fitsfile* fptr,
        const std::vector<std::string>& columns,
        long start_row,
        long num_rows,
        torch::Device device = torch::kCPU
    );

private:
    struct ColumnInfo {
        std::string name;
        int fits_type;
        long repeat;
        long width;
        int col_num;
        torch::Dtype torch_dtype;
    };
    
    static std::vector<ColumnInfo> analyze_columns(fitsfile* fptr, const std::vector<std::string>& requested_columns);
    
    template<typename T, int CfitsioType>
    static torch::Tensor read_column_fast(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
};

} // namespace torchfits_fast
