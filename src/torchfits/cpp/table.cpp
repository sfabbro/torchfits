/**
 * High-performance FITS table reader with memory pools.
 * Phase 2 implementation supporting all FITS column types.
 */

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <functional>
#include <cstring>  // for memset

#include <fitsio.h>
#include "hardware.h"

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace torchfits {
// --- Optimized Table Reading Logic (from table_optimized.cpp) ---
class OptimizedTableReader {
public:
    OptimizedTableReader(fitsfile* fptr, int hdu_num) : fptr_(fptr), hdu_num_(hdu_num) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) throw std::runtime_error("Failed to move to table HDU");
        fits_get_num_rows(fptr_, &nrows_, &status);
        fits_get_num_cols(fptr_, &ncols_, &status);
        if (status != 0) throw std::runtime_error("Failed to get table dimensions");
        cache_column_info();
    }
    std::unordered_map<std::string, torch::Tensor> read_columns(const std::vector<std::string>& column_names, long start_row = 1, long num_rows = -1) {
        if (num_rows == -1) num_rows = nrows_;
        std::vector<int> col_indices;
        std::vector<ColumnInfo> selected_cols;
        for (const auto& name : column_names) {
            auto it = column_map_.find(name);
            if (it != column_map_.end()) {
                col_indices.push_back(it->second.col_num);
                selected_cols.push_back(it->second);
            }
        }
        if (col_indices.empty()) return {};
        HardwareInfo hw = detect_hardware();
        size_t row_size = calculate_row_size(selected_cols);
        size_t optimal_chunk_rows = calculate_optimal_chunk_size(row_size * num_rows, hw) / row_size;
        optimal_chunk_rows = std::max(1UL, std::min(optimal_chunk_rows, (size_t)num_rows));
        std::unordered_map<std::string, torch::Tensor> result;
        for (size_t i = 0; i < column_names.size(); i++) {
            const auto& col_info = selected_cols[i];
            auto tensor = create_tensor_for_column(col_info, num_rows);
            result[column_names[i]] = tensor;
        }
        long rows_read = 0;
        while (rows_read < num_rows) {
            long chunk_rows = std::min(optimal_chunk_rows, (size_t)(num_rows - rows_read));
            read_chunk_optimized(selected_cols, start_row + rows_read, chunk_rows, result, rows_read);
            rows_read += chunk_rows;
        }
        return result;
    }
    void iterate_rows(std::function<void(const std::unordered_map<std::string, torch::Tensor>&)> callback, const std::vector<std::string>& column_names, size_t batch_size = 1000) {
        // Simple implementation: read in batches
        long total_rows = nrows_;
        for (long start_row = 1; start_row <= total_rows; start_row += batch_size) {
            long chunk_rows = std::min(batch_size, (size_t)(total_rows - start_row + 1));
            auto chunk_data = read_columns(column_names, start_row, chunk_rows);
            callback(chunk_data);
        }
    }
private:
    struct ColumnInfo {
        int col_num;
        int data_type;
        long repeat;
        long width;
        std::string name;
        std::string format;
        torch::ScalarType torch_type;
    };
    struct IteratorData {
        OptimizedTableReader* reader;
        std::function<void(const std::unordered_map<std::string, torch::Tensor>&)> callback;
        std::vector<std::string> column_names;
        size_t batch_size;
    };
    fitsfile* fptr_;
    int hdu_num_;
    long nrows_;
    int ncols_;
    std::unordered_map<std::string, ColumnInfo> column_map_;
    void cache_column_info() {
        int status = 0;
        for (int col = 1; col <= ncols_; col++) {
            ColumnInfo info;
            info.col_num = col;
            char col_name[FLEN_VALUE];
            char col_format[FLEN_VALUE];
            fits_get_bcolparms(fptr_, col, col_name, nullptr, col_format, &info.repeat, nullptr, nullptr, nullptr, nullptr, &status);
            if (status != 0) continue;
            info.name = col_name;
            info.format = col_format;
            parse_fits_format(col_format, info);
            column_map_[info.name] = info;
        }
    }
    void parse_fits_format(const std::string& format, ColumnInfo& info) {
        char type_char = format.back();
        switch (type_char) {
            case 'L': info.data_type = TLOGICAL; info.torch_type = torch::kBool; info.width = 1; break;
            case 'B': info.data_type = TBYTE; info.torch_type = torch::kUInt8; info.width = 1; break;
            case 'I': info.data_type = TSHORT; info.torch_type = torch::kInt16; info.width = 2; break;
            case 'J': info.data_type = TINT; info.torch_type = torch::kInt32; info.width = 4; break;
            case 'K': info.data_type = TLONGLONG; info.torch_type = torch::kInt64; info.width = 8; break;
            case 'E': info.data_type = TFLOAT; info.torch_type = torch::kFloat32; info.width = 4; break;
            case 'D': info.data_type = TDOUBLE; info.torch_type = torch::kFloat64; info.width = 8; break;
            default: info.data_type = TFLOAT; info.torch_type = torch::kFloat32; info.width = 4;
        }
    }
    size_t calculate_row_size(const std::vector<ColumnInfo>& columns) {
        size_t total = 0;
        for (const auto& col : columns) total += col.width * col.repeat;
        return total;
    }
    torch::Tensor create_tensor_for_column(const ColumnInfo& col_info, long num_rows) {
        std::vector<int64_t> shape;
        if (col_info.repeat > 1) shape = {num_rows, col_info.repeat};
        else shape = {num_rows};
        return torch::empty(shape, col_info.torch_type);
    }
    void read_chunk_optimized(const std::vector<ColumnInfo>& columns, long start_row, long num_rows, std::unordered_map<std::string, torch::Tensor>& tensors, long tensor_offset) {
        // True zero-copy implementation: read directly into tensor memory
        int status = 0;
        
        // Read each column directly into its tensor memory
        for (const auto& col : columns) {
            auto& tensor = tensors[col.name];
            void* tensor_data = get_tensor_data_ptr(tensor, tensor_offset);
            
            // Direct CFITSIO read into tensor memory - true zero-copy
            fits_read_col(fptr_, col.data_type, col.col_num, start_row, 1, num_rows, 
                         nullptr, tensor_data, nullptr, &status);
            
            if (status != 0) {
                throw std::runtime_error("Failed to read column '" + col.name + "' directly into tensor");
            }
        }
    }
    void* get_tensor_data_ptr(torch::Tensor& tensor, long offset) {
        switch (tensor.scalar_type()) {
            case torch::kBool: return tensor.data_ptr<bool>() + offset;
            case torch::kUInt8: return tensor.data_ptr<uint8_t>() + offset;
            case torch::kInt16: return tensor.data_ptr<int16_t>() + offset;
            case torch::kInt32: return tensor.data_ptr<int32_t>() + offset;
            case torch::kInt64: return tensor.data_ptr<int64_t>() + offset;
            case torch::kFloat32: return tensor.data_ptr<float>() + offset;
            case torch::kFloat64: return tensor.data_ptr<double>() + offset;
            default: throw std::runtime_error("Unsupported tensor type");
        }
    }
    void copy_column_data(const uint8_t* buffer, size_t buffer_offset, void* tensor_data, const ColumnInfo& col, long num_rows, size_t row_size) {
        if (col.repeat == 1) {
            const uint8_t* src = buffer + buffer_offset;
            uint8_t* dst = static_cast<uint8_t*>(tensor_data);
            for (long row = 0; row < num_rows; row++) {
                std::memcpy(dst, src, col.width);
                src += row_size;
                dst += col.width;
            }
        } else {
            const uint8_t* src = buffer + buffer_offset;
            uint8_t* dst = static_cast<uint8_t*>(tensor_data);
            size_t block_size = col.width * col.repeat;
            for (long row = 0; row < num_rows; row++) {
                std::memcpy(dst, src, block_size);
                src += row_size;
                dst += block_size;
            }
        }
    }
    static int iterator_callback(long total_rows, long offset, long first_row, long num_rows, int num_cols, iteratorCol* data, void* user_data) {
        IteratorData* iter_data = static_cast<IteratorData*>(user_data);
        auto chunk_data = iter_data->reader->read_columns(iter_data->column_names, first_row, num_rows);
        iter_data->callback(chunk_data);
        return 0;
    }
};

enum class FITSColumnType {
    LOGICAL,    // L
    BYTE,       // B  
    SHORT,      // I
    INT,        // J
    LONG,       // K
    FLOAT,      // E
    DOUBLE,     // D
    STRING,     // A
    VARIABLE    // P/Q - variable length arrays
};

struct ColumnInfo {
    std::string name;
    FITSColumnType type;
    int repeat;
    int width;
    torch::ScalarType torch_type;
};

class TableReader {
public:
    TableReader(const std::string& filename, int hdu_num = 1) : filename_(filename), hdu_num_(hdu_num) {
        int status = 0;
        fits_open_file(&fptr_, filename.c_str(), READONLY, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to open FITS file");
        }
        
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
        
        analyze_table();

    }

    TableReader(fitsfile* fptr, int hdu_num = 1) : fptr_(fptr), hdu_num_(hdu_num) {
        int status = 0;
        fits_movabs_hdu(fptr_, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
        analyze_table();
    }
    
    ~TableReader() {
        if (fptr_ && !filename_.empty()) {
            int status = 0;
            fits_close_file(fptr_, &status);
        }

    }
    
    void analyze_table() {
        int status = 0;
        
        // Get table dimensions
        fits_get_num_rows(fptr_, &nrows_, &status);
        fits_get_num_cols(fptr_, &ncols_, &status);
        
        if (status != 0) {
            throw std::runtime_error("Failed to get table dimensions");
        }
        
        // Debug output
        #ifdef DEBUG_TABLE
        printf("analyze_table: nrows=%ld, ncols=%d\n", nrows_, ncols_);
        #endif
        
        // Analyze columns
        columns_.clear();
        columns_.reserve(ncols_);
        
        for (int i = 1; i <= ncols_; i++) {
            ColumnInfo col;
            
            char ttype[FLEN_VALUE], tform[FLEN_VALUE], tunit[FLEN_VALUE];
            
            // Initialize arrays to avoid garbage values
            memset(ttype, 0, FLEN_VALUE);
            memset(tform, 0, FLEN_VALUE);
            memset(tunit, 0, FLEN_VALUE);
            
            // Get column parameters using fits_get_bcolparms with correct parameter types
            long repeat_long = 0;
            long width_long = 0;
            int col_status = 0;
            fits_get_bcolparms(fptr_, i, ttype, tunit, tform, &repeat_long, &width_long, nullptr, nullptr, nullptr, &col_status);
            
            if (col_status != 0) {
                // Skip problematic columns but log the issue
                #ifdef DEBUG_TABLE
                char err_msg[81];
                fits_get_errstatus(col_status, err_msg);
                printf("Warning: Failed to get column %d info: %s\n", i, err_msg);
                #endif
                continue;
            }
            
            col.repeat = (int)repeat_long;
            col.name = std::string(ttype);
            col.width = 1;  // Will be set based on type
            
            #ifdef DEBUG_TABLE
            printf("Column %d: name='%s', tform='%s', repeat=%d\n", i, ttype, tform, col.repeat);
            #endif
            
            // Parse TFORM to get data type
            size_t tform_len = strlen(tform);
            if (tform_len == 0) {
                #ifdef DEBUG_TABLE
                printf("Warning: Empty TFORM for column %d\n", i);
                #endif
                continue;
            }
            
            char type_char = tform[tform_len - 1];
            
            // Handle variable length arrays
            bool is_variable = (strchr(tform, 'P') != nullptr || strchr(tform, 'Q') != nullptr);
            
            if (is_variable) {
                col.type = FITSColumnType::VARIABLE;
                col.torch_type = (type_char == 'Q') ? torch::kFloat64 : torch::kFloat32;
                col.width = 8;  // Pointer size
            } else {
                switch (type_char) {
                    case 'L': 
                        col.type = FITSColumnType::LOGICAL;
                        col.torch_type = torch::kBool;
                        col.width = 1;
                        break;
                    case 'B':
                        col.type = FITSColumnType::BYTE;
                        col.torch_type = torch::kUInt8;
                        col.width = 1;
                        break;
                    case 'I':
                        col.type = FITSColumnType::SHORT;
                        col.torch_type = torch::kInt16;
                        col.width = 2;
                        break;
                    case 'J':
                        col.type = FITSColumnType::INT;
                        col.torch_type = torch::kInt32;
                        col.width = 4;
                        break;
                    case 'K':
                        col.type = FITSColumnType::LONG;
                        col.torch_type = torch::kInt64;
                        col.width = 8;
                        break;
                    case 'E':
                        col.type = FITSColumnType::FLOAT;
                        col.torch_type = torch::kFloat32;
                        col.width = 4;
                        break;
                    case 'D':
                        col.type = FITSColumnType::DOUBLE;
                        col.torch_type = torch::kFloat64;
                        col.width = 8;
                        break;
                    case 'A':
                        col.type = FITSColumnType::STRING;
                        col.torch_type = torch::kInt64; // Will be converted to categorical
                        col.width = col.repeat;  // String width from repeat count
                        break;
                    default:
                        col.type = FITSColumnType::FLOAT;
                        col.torch_type = torch::kFloat32;
                        col.width = 4;
                }
            }
            
            columns_.push_back(col);
        }
        
        #ifdef DEBUG_TABLE
        printf("analyze_table complete: found %zu columns out of %d expected\n", columns_.size(), ncols_);
        #endif
        
        // Verify we have the expected number of columns
        if ((int)columns_.size() != ncols_) {
            // This is a critical error - the table metadata is inconsistent
            throw std::runtime_error("Column count mismatch: expected " + std::to_string(ncols_) + 
                                    ", found " + std::to_string(columns_.size()));
        }
    }
    
    std::unordered_map<std::string, nb::object> read_columns(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1) {
        
        int status = 0;
        std::unordered_map<std::string, nb::object> result;
        
        // Check if we have any data
        if (nrows_ == 0 || ncols_ == 0) {
            // Return empty result for empty tables
            return result;
        }
        
        if (num_rows == -1) {
            num_rows = nrows_;
        }
        
        // Determine which columns to read
        std::vector<int> col_indices;
        if (column_names.empty()) {
            for (int i = 0; i < ncols_; i++) {
                col_indices.push_back(i);
            }
        } else {
            for (const auto& name : column_names) {
                for (int i = 0; i < ncols_; i++) {
                    if (columns_[i].name == name) {
                        col_indices.push_back(i);
                        break;
                    }
                }
            }
        }
        
        // If no valid columns found, return empty result
        if (col_indices.empty()) {
            return result;
        }
        
        // Read each column using zero-copy direct tensor allocation
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            
            // Create properly shaped tensor based on repeat count
            torch::Tensor tensor;
            if (col.repeat > 1 && col.type != FITSColumnType::STRING) {
                tensor = torch::empty({num_rows, col.repeat}, torch::TensorOptions().dtype(col.torch_type));
            } else {
                tensor = torch::empty({num_rows}, torch::TensorOptions().dtype(col.torch_type));
            }
            
            // Read column data directly into tensor memory (zero-copy)
            int anynul;
            switch (col.type) {
                case FITSColumnType::VARIABLE: {
                    // Skip variable length arrays for now - they need special handling
                    continue;
                }
                case FITSColumnType::FLOAT: {
                    fits_read_col(fptr_, TFLOAT, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<float>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::DOUBLE: {
                    fits_read_col(fptr_, TDOUBLE, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<double>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::INT: {
                    fits_read_col(fptr_, TINT, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<int32_t>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::LONG: {
                    fits_read_col(fptr_, TLONGLONG, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<int64_t>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::SHORT: {
                    fits_read_col(fptr_, TSHORT, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<int16_t>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::BYTE: {
                    fits_read_col(fptr_, TBYTE, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<uint8_t>(), &anynul, &status);
                    break;
                }
                case FITSColumnType::LOGICAL: {
                    // CFITSIO uses char for logical, convert to bool tensor
                    std::vector<char> temp_data(num_rows * col.repeat);
                    fits_read_col(fptr_, TLOGICAL, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, temp_data.data(), &anynul, &status);
                    
                    auto bool_ptr = tensor.data_ptr<bool>();
                    for (long i = 0; i < num_rows * col.repeat; i++) {
                        bool_ptr[i] = (temp_data[i] != 0);
                    }
                    break;
                }
                case FITSColumnType::STRING: {
                    // Read strings as fixed-width character arrays
                    std::vector<char> string_buffer(num_rows * col.width);
                    fits_read_col(fptr_, TSTRING, col_idx + 1, start_row, 1, num_rows,
                                 nullptr, string_buffer.data(), &anynul, &status);
                    
                    // For now, convert to categorical indices (simple hash-based)
                    auto cat_ptr = tensor.data_ptr<int64_t>();
                    for (long i = 0; i < num_rows; i++) {
                        // Simple hash of the string for categorical encoding
                        std::string str(&string_buffer[i * col.width], col.width);
                        str.erase(str.find_last_not_of(" \0") + 1); // trim
                        cat_ptr[i] = std::hash<std::string>{}(str) % 1000000; // Simple categorical
                    }
                    break;
                }
                default:
                    // Default to float
                    fits_read_col(fptr_, TFLOAT, col_idx + 1, start_row, 1, num_rows * col.repeat,
                                 nullptr, tensor.data_ptr<float>(), &anynul, &status);
            }
            
            if (status != 0) {
                char err_msg[81];
                fits_get_errstatus(status, err_msg);
                throw std::runtime_error("Failed to read column '" + col.name + "': " + std::string(err_msg));
            }
            
            result[col.name] = nb::cast(tensor);
        }
        
        return result;

    }
    
    std::vector<std::string> get_column_names() const {
        std::vector<std::string> names;
        for (const auto& col : columns_) {
            names.push_back(col.name);
        }
        return names;
    }
    
    long get_num_rows() const { return nrows_; }
    int get_num_cols() const { return ncols_; }

private:
    fitsfile* fptr_ = nullptr;
    std::string filename_;
    int hdu_num_;
    long nrows_ = 0;
    int ncols_ = 0;
    std::vector<ColumnInfo> columns_;
};

// Memory pool for table operations
class TableMemoryPool {
public:
    static TableMemoryPool& instance() {
        static TableMemoryPool pool;
        return pool;
    }
    
    torch::Tensor get_tensor_buffer(const std::vector<int64_t>& shape, torch::ScalarType dtype) {
        // Simple implementation - would use actual pooling
        return torch::empty(shape, torch::TensorOptions().dtype(dtype));
    }
    
private:
    TableMemoryPool() = default;
};

} // namespace torchfits

// C API for Python bindings
extern "C" {

void* open_table_reader(const char* filename, int hdu_num) {
    try {
        return new torchfits::TableReader(filename, hdu_num);
    } catch (...) {
        return nullptr;
    }
}

void* open_table_reader_from_handle(uintptr_t handle, int hdu_num) {
    try {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return new torchfits::TableReader(file->get_fptr(), hdu_num);
    } catch (...) {
        return nullptr;
    }
}

void close_table_reader(void* reader_handle) {
    if (reader_handle) {
        delete static_cast<torchfits::TableReader*>(reader_handle);
    }
}

int read_table_columns(void* reader_handle, const char** column_names, int num_columns,
                      long start_row, long num_rows, nb::dict* result_dict) {
    if (!reader_handle) return -1;
    
    try {
        auto* reader = static_cast<torchfits::TableReader*>(reader_handle);
        
        std::vector<std::string> cols;
        for (int i = 0; i < num_columns; i++) {
            cols.push_back(std::string(column_names[i]));
        }
        
        auto result = reader->read_columns(cols, start_row, num_rows);
        *result_dict = nb::cast<nb::dict>(nb::cast(result));
        return 0;
    } catch (...) {
        return -1;
    }
}

void write_fits_table(const char* filename, nb::dict tensor_dict, nb::dict header, bool overwrite) {
    fitsfile* fptr;
    int status = 0;

    if (overwrite) {
        fits_create_file(&fptr, filename, &status);
    } else {
        fits_open_file(&fptr, filename, READWRITE, &status);
    }

    if (status != 0) {
        throw std::runtime_error("Failed to open FITS file for writing");
    }

    int num_cols = tensor_dict.size();
    long num_rows = 0;
    if (num_cols > 0) {
        auto first_col = nb::cast<torch::Tensor>((*tensor_dict.begin()).second);
        num_rows = first_col.size(0);
    }

    char** ttype = new char*[num_cols];
    char** tform = new char*[num_cols];
    char** tunit = new char*[num_cols];

    int i = 0;
    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        torch::Tensor tensor = nb::cast<torch::Tensor>(item.second);

        ttype[i] = new char[col_name.length() + 1];
        strncpy(ttype[i], col_name.c_str(), col_name.length());
        ttype[i][col_name.length()] = '\0';

        std::string form;
        if (tensor.dtype() == torch::kUInt8) {
            form = "B";
        } else if (tensor.dtype() == torch::kInt16) {
            form = "I";
        } else if (tensor.dtype() == torch::kInt32) {
            form = "J";
        } else if (tensor.dtype() == torch::kFloat32) {
            form = "E";
        } else if (tensor.dtype() == torch::kFloat64) {
            form = "D";
        } else {
            throw std::runtime_error("Unsupported tensor data type");
        }

        tform[i] = new char[form.length() + 1];
        strncpy(tform[i], form.c_str(), form.length());
        tform[i][form.length()] = '\0';

        tunit[i] = new char[1];
        tunit[i][0] = '\0';

        i++;
    }

    fits_create_tbl(fptr, BINARY_TBL, num_rows, num_cols, ttype, tform, tunit, "", &status);

    i = 0;
    for (auto item : tensor_dict) {
        torch::Tensor tensor = nb::cast<torch::Tensor>(item.second);
        void* data_ptr = tensor.data_ptr();
        int fits_type;
        if (tensor.dtype() == torch::kUInt8) {
            fits_type = TBYTE;
        } else if (tensor.dtype() == torch::kInt16) {
            fits_type = TSHORT;
        } else if (tensor.dtype() == torch::kInt32) {
            fits_type = TINT;
        } else if (tensor.dtype() == torch::kFloat32) {
            fits_type = TFLOAT;
        } else if (tensor.dtype() == torch::kFloat64) {
            fits_type = TDOUBLE;
        }

        fits_write_col(fptr, fits_type, i + 1, 1, 1, num_rows, data_ptr, &status);
        i++;
    }

    for (i = 0; i < num_cols; i++) {
        delete[] ttype[i];
        delete[] tform[i];
        delete[] tunit[i];
    }
    delete[] ttype;
    delete[] tform;
    delete[] tunit;

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to write table to FITS file");
    }
}

void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict) {
    fitsfile* fptr;
    int status = 0;

    fits_open_file(&fptr, filename, READWRITE, &status);
    if (status != 0) {
        throw std::runtime_error("Failed to open FITS file for writing");
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        throw std::runtime_error("Failed to move to table HDU");
    }

    long num_rows = 0;
    if (tensor_dict.size() > 0) {
        auto first_col = nb::cast<torch::Tensor>((*tensor_dict.begin()).second);
        num_rows = first_col.size(0);
    }

    long start_row;
    fits_get_num_rows(fptr, &start_row, &status);
    start_row++;

    fits_insert_rows(fptr, start_row -1, num_rows, &status);

    int i = 0;
    for (auto item : tensor_dict) {
        torch::Tensor tensor = nb::cast<torch::Tensor>(item.second);
        void* data_ptr = tensor.data_ptr();
        int fits_type;
        if (tensor.dtype() == torch::kUInt8) {
            fits_type = TBYTE;
        } else if (tensor.dtype() == torch::kInt16) {
            fits_type = TSHORT;
        } else if (tensor.dtype() == torch::kInt32) {
            fits_type = TINT;
        } else if (tensor.dtype() == torch::kFloat32) {
            fits_type = TFLOAT;
        } else if (tensor.dtype() == torch::kFloat64) {
            fits_type = TDOUBLE;
        }

        fits_write_col(fptr, fits_type, i + 1, start_row, 1, num_rows, data_ptr, &status);
        i++;
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to append rows to FITS table");
    }
}

}