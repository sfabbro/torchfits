/**
 * High-performance FITS table reader with memory pools.
 * Phase 2 implementation supporting all FITS column types.
 */

#include "torchfits_torch.h"
#include <vector>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <ATen/Parallel.h>
#include <functional>
#include <cstring>  // for memset

// #define DEBUG_TABLE 1

#include <fitsio.h>
#include "hardware.h"
#include "torch_compat.h"

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace torchfits {



// Enhanced error handling utilities
class FITSError {
public:
    static std::string get_error_message(int status) {
        char error_text[FLEN_ERRMSG];
        fits_get_errstatus(status, error_text);
        return std::string(error_text);
    }
    
    static void check_status(int status, const std::string& operation) {
        if (status != 0) {
            std::string error_msg = operation + ": " + get_error_message(status);
            throw std::runtime_error(error_msg);
        }
    }
};

// Performance monitoring for chunk optimization
struct PerformanceMetrics {
    double total_read_time = 0.0;
    size_t total_bytes_read = 0;
    size_t chunk_count = 0;
    double avg_chunk_time = 0.0;
    double throughput_mbps = 0.0;
    
    void update(double chunk_time, size_t bytes_read) {
        total_read_time += chunk_time;
        total_bytes_read += bytes_read;
        chunk_count++;
        avg_chunk_time = total_read_time / chunk_count;
        throughput_mbps = (total_bytes_read / (1024.0 * 1024.0)) / total_read_time;
    }
    
    void reset() {
        total_read_time = 0.0;
        total_bytes_read = 0;
        chunk_count = 0;
        avg_chunk_time = 0.0;
        throughput_mbps = 0.0;
    }
};

// --- Enhanced Optimized Table Reading Logic ---


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
    long byte_offset; // Offset in bytes from start of row
    double tscale = 1.0;
    double tzero = 0.0;
    bool scaled = false;
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
        
        // Check HDU type
        int hdutype;
        fits_get_hdu_type(fptr_, &hdutype, &status);
        is_ascii_ = (hdutype == ASCII_TBL);
        
        // Get table dimensions
        fits_get_num_rows(fptr_, &nrows_, &status);
        fits_get_num_cols(fptr_, &ncols_, &status);
        
        if (status != 0) {
            throw std::runtime_error("Failed to get table dimensions");
        }
        
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
            
            // Get column name
            int col_status = 0;
            
            // We use fits_read_key to read TTYPEn.
            char keyname[FLEN_KEYWORD];
            snprintf(keyname, FLEN_KEYWORD, "TTYPE%d", i);
            fits_read_key(fptr_, TSTRING, keyname, ttype, nullptr, &col_status);
            
            if (col_status != 0) {
                // TTYPE is optional? If missing, use default name?
                col_status = 0; // Reset status
                snprintf(ttype, FLEN_VALUE, "COL%d", i);
            }
            
            int typecode;
            long repeat_long, width_long;
            fits_get_coltype(fptr_, i, &typecode, &repeat_long, &width_long, &col_status);
            
            if (col_status != 0) {
                 #ifdef DEBUG_TABLE
                 char err_msg[81];
                 fits_get_errstatus(col_status, err_msg);
                 fprintf(stderr, "Warning: Failed to get column %d info: %s\n", i, err_msg);
                 #endif
                 continue;
            }
            
            // Map typecode to FITSColumnType
            // ... (rest of logic needs to be updated to use typecode)
            
            // Wait, I need to replace the existing logic block.
            // The existing logic uses tform from fits_get_bcolparms.
            // I should rewrite the loop body to use fits_get_coltype.

            
            if (col_status != 0) {
                // Skip problematic columns but log the issue
                #ifdef DEBUG_TABLE
                char err_msg[81];
                fits_get_errstatus(col_status, err_msg);
                fprintf(stderr, "Warning: Failed to get column %d info: %s\n", i, err_msg);
                #endif
                continue;
            }
            
            col.repeat = (int)repeat_long;
            col.name = std::string(ttype);
            col.width = 1;  // Will be set based on type

            // Read scaling keywords if present
            col.tscale = 1.0;
            col.tzero = 0.0;
            col.scaled = false;
            int scale_status = 0;
            char scale_key[FLEN_KEYWORD];
            snprintf(scale_key, FLEN_KEYWORD, "TSCAL%d", i);
            fits_read_key(fptr_, TDOUBLE, scale_key, &col.tscale, nullptr, &scale_status);
            if (scale_status != 0) { scale_status = 0; col.tscale = 1.0; }
            snprintf(scale_key, FLEN_KEYWORD, "TZERO%d", i);
            fits_read_key(fptr_, TDOUBLE, scale_key, &col.tzero, nullptr, &scale_status);
            if (scale_status != 0) { scale_status = 0; col.tzero = 0.0; }
            col.scaled = (col.tscale != 1.0 || col.tzero != 0.0);
            
            #ifdef DEBUG_TABLE
            fprintf(stderr, "Column %d: name='%s', typecode=%d, repeat=%d\n", i, ttype, typecode, col.repeat);
            #endif
            
            if (typecode < 0) {
                // Variable length array
                col.type = FITSColumnType::VARIABLE;
                int abs_type = -typecode;
                switch (abs_type) {
                    case TLOGICAL: col.torch_type = torch::kBool; break;
                    case TBYTE: col.torch_type = torch::kUInt8; break;
                    case TSHORT: col.torch_type = torch::kInt16; break;
                    case TINT: col.torch_type = torch::kInt32; break;
                    case TLONG: col.torch_type = (sizeof(long) == 8) ? torch::kInt64 : torch::kInt32; break;
                    case TLONGLONG: col.torch_type = torch::kInt64; break;
                    case TFLOAT: col.torch_type = torch::kFloat32; break;
                    case TDOUBLE: col.torch_type = torch::kFloat64; break;
                    default: col.torch_type = torch::kFloat32;
                }
                col.width = 8;
            } else {
                switch (typecode) {
                    case TLOGICAL: 
                        col.type = FITSColumnType::LOGICAL;
                        col.torch_type = torch::kBool;
                        col.width = 1;
                        break;
                    case TBYTE:
                    case TSBYTE:
                        col.type = FITSColumnType::BYTE;
                        col.torch_type = torch::kUInt8;
                        col.width = 1;
                        break;
                    case TSHORT:
                    case TUSHORT:
                        col.type = FITSColumnType::SHORT;
                        col.torch_type = torch::kInt16;
                        col.width = 2;
                        break;
                    case TINT:
                    case TUINT:
                        col.type = FITSColumnType::INT;
                        col.torch_type = torch::kInt32;
                        col.width = 4;
                        break;
                    case TSTRING:
                        col.type = FITSColumnType::STRING;
                        col.torch_type = torch::kUInt8;
                        if (is_ascii_) {
                             col.repeat = 1; // One string per row
                             col.width = (int)width_long;
                        } else {
                             // Binary table
                             col.width = 1;
                             // For binary tables, repeat_long is often the string length,
                             // but some FITS writers may populate width_long instead.
                             if (repeat_long > 1) {
                                 col.repeat = (int)repeat_long;
                             } else if (width_long > 0) {
                                 col.repeat = (int)width_long;
                             }
                        }
                        break;
                    case TLONG:
                        if (is_ascii_) {
                             // For ASCII, TLONG usually means it fits in standard integer.
                             // Prefer Int32 to match typical usage and avoid issues with TLONGLONG on ASCII?
                             col.type = FITSColumnType::INT;
                             col.torch_type = torch::kInt32;
                             col.width = 4;
                        } else {
                            if (sizeof(long) == 8) {
                                col.type = FITSColumnType::LONG;
                                col.torch_type = torch::kInt64;
                                col.width = 8;
                            } else {
                                col.type = FITSColumnType::INT;
                                col.torch_type = torch::kInt32;
                                col.width = 4;
                            }
                        }

                        break;
                    case TULONG:
                        if (sizeof(long) == 8) {
                            col.type = FITSColumnType::LONG;
                            col.torch_type = torch::kInt64;
                            col.width = 8;
                        } else {
                            col.type = FITSColumnType::INT;
                            col.torch_type = torch::kInt32;
                            col.width = 4;
                        }
                        break;
                    case TLONGLONG:
                        col.type = FITSColumnType::LONG;
                        col.torch_type = torch::kInt64;
                        col.width = 8;
                        break;
                    case TFLOAT:
                        col.type = FITSColumnType::FLOAT;
                        col.torch_type = torch::kFloat32;
                        col.width = 4;
                        break;
                    case TDOUBLE:
                        col.type = FITSColumnType::DOUBLE;
                        col.torch_type = torch::kFloat64;
                        col.width = 8;
                        break;
                    default:
                        #ifdef DEBUG_TABLE
                        fprintf(stderr, "Warning: Unknown typecode %d for column %d\n", typecode, i);
                        #endif
                        continue;
                }
            }

            
            columns_.push_back(col);
        }
        
        #ifdef DEBUG_TABLE
        fprintf(stderr, "analyze_table complete: found %zu columns out of %d expected\n", columns_.size(), ncols_);
        #endif
        
        // Verify we have the expected number of columns
        if ((int)columns_.size() != ncols_) {
            // This is a critical error - the table metadata is inconsistent
            throw std::runtime_error("Column count mismatch: expected " + std::to_string(ncols_) + 
                                    ", found " + std::to_string(columns_.size()));
        }
        
        // Calculate offsets
        long current_offset = 0;
        for (auto& col : columns_) {
            col.byte_offset = current_offset;
            current_offset += col.width * col.repeat;
        }
        row_width_bytes_ = current_offset;
    }
    
    // Helper struct to hold column data (either fixed or VLA)
    struct ColumnData {
        bool is_vla;
        torch::Tensor fixed_data;
        std::vector<torch::Tensor> vla_data;
        
        ColumnData() : is_vla(false) {}
        ColumnData(torch::Tensor t) : is_vla(false), fixed_data(t) {}
        ColumnData(std::vector<torch::Tensor> v) : is_vla(true), vla_data(v) {}
    };

    // Read columns from the table
    // Returns a map of column name to ColumnData
    std::unordered_map<std::string, ColumnData> read_columns(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1) {
        
        if (num_rows == -1) {
            num_rows = nrows_;
        }
        
        // Handle empty table
        if (nrows_ == 0) {
            return {};
        }

        // Validate rows
        if (start_row < 1 || start_row > nrows_) {
            std::cerr << "Invalid start row: " << start_row << ", nrows: " << nrows_ << std::endl;
            throw std::runtime_error("Invalid start row");
        }
        if (start_row + num_rows - 1 > nrows_) {
            num_rows = nrows_ - start_row + 1;
        }
        
        std::vector<int> col_indices;
        if (column_names.empty()) {
            // Read all columns
            for (int i = 0; i < ncols_; i++) {
                col_indices.push_back(i);
            }
        } else {
            // Read specified columns
            for (const auto& name : column_names) {
                bool found = false;
                for (int i = 0; i < ncols_; i++) {
                    if (columns_[i].name == name) {
                        col_indices.push_back(i);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "Column not found: " << name << ". Available columns: ";
                    for(int k=0; k<ncols_; k++) std::cerr << columns_[k].name << ", ";
                    std::cerr << std::endl;
                    throw std::runtime_error("Column not found: " + name);
                }
            }
        }
        
        int status = 0;
        std::unordered_map<std::string, ColumnData> result;
        
        
        // Check if we have any data
        if (nrows_ == 0 || ncols_ == 0) {
             return result;
        }
        
        // Allocate tensors for all requested columns (except VLA)
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            
            
            if (col.type == FITSColumnType::VARIABLE) {
                // VLA columns will be handled separately
                continue;
            }
            
            std::vector<int64_t> shape;
            shape.push_back(num_rows);
            // Handle multi-dimensional columns AND strings
            if (col.type == FITSColumnType::STRING) {
                if (is_ascii_) {
                    shape.push_back(col.width);
                } else {
                    shape.push_back(col.repeat);
                }
            } else if (col.repeat > 1) {
                 shape.push_back(col.repeat);
            }
            
            // Create tensor
            torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(col.torch_type));
            
            // Store in result map as ColumnData
            
            result[col.name] = ColumnData(tensor);
        }
        
        // Heuristic: Use buffered reading if we are reading a significant portion of the row
        // or if we are reading many columns.
        // Threshold: > 25% of row bytes OR > 50% of columns
        // Note: Buffered reading currently does NOT support VLA.
        long requested_bytes = 0;
        bool has_vla = false;
        bool has_scaled = false;
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type == FITSColumnType::VARIABLE) {
                has_vla = true;
            } else {
                requested_bytes += col.width * col.repeat;
            }
            if (col.scaled) {
                has_scaled = true;
            }
        }
        
        bool use_buffered = false;
        if (!is_ascii_ && !has_vla && !has_scaled && row_width_bytes_ > 0) {
            double byte_fraction = (double)requested_bytes / row_width_bytes_;
            double col_fraction = (double)col_indices.size() / ncols_;
            
            if (byte_fraction > 0.25 || col_fraction > 0.5) {
                use_buffered = true;
            }
        }
        
        auto read_column_by_column = [&]() {
            // Read column by column
            for (int col_idx : col_indices) {
                const auto& col = columns_[col_idx];

                if (col.type == FITSColumnType::VARIABLE) {
                    // Read VLA column
                    result[col.name] = ColumnData(read_vla_column(col_idx, start_row, num_rows, col));
                } else {
                    // Read fixed width column
                    torch::Tensor tensor = result[col.name].fixed_data;

                    int status = 0;
                    // Use fits_read_col to read directly into tensor memory
                    // Note: fits_read_col handles byte swapping automatically!

                    int datatype;
                    switch (col.type) {
                        case FITSColumnType::LOGICAL: datatype = TLOGICAL; break;
                        case FITSColumnType::BYTE: datatype = TBYTE; break;
                        case FITSColumnType::SHORT: datatype = TSHORT; break;
                        case FITSColumnType::INT: datatype = TINT; break;
                        case FITSColumnType::LONG: datatype = TLONGLONG; break;
                        case FITSColumnType::FLOAT: datatype = TFLOAT; break;
                        case FITSColumnType::DOUBLE: datatype = TDOUBLE; break;
                        case FITSColumnType::STRING: datatype = TBYTE; break; // Read strings as bytes
                        default: datatype = TFLOAT;
                    }

                    long firstelem = 1;
                    long nelements = num_rows * col.repeat;

                    if (col.type == FITSColumnType::STRING) {
                         if (is_ascii_) {
                             // ASCII table: read as strings
                             std::vector<char*> pointers(nelements);
                             std::vector<char> buffer(nelements * (col.width + 1));

                             for (long i = 0; i < nelements; i++) {
                                 pointers[i] = &buffer[i * (col.width + 1)];
                             }

                             fits_read_col(fptr_, TSTRING, col_idx + 1, start_row, firstelem, nelements,
                                          nullptr, pointers.data(), nullptr, &status);

                             // Copy to tensor (row-major)
                             uint8_t* tensor_data = (uint8_t*)tensor.data_ptr();
                             for (long i = 0; i < nelements; i++) {
                                 // Copy string to tensor, padding with spaces or nulls?
                                 // TorchFits convention: raw bytes.
                                 // cfitsio returns null-terminated string.
                                 // We copy up to col.width.
                                 const char* src = pointers[i];
                                 size_t len = strlen(src);
                                 for (int j = 0; j < col.width; j++) {
                                     if (j < len) {
                                         tensor_data[i * col.width + j] = (uint8_t)src[j];
                                     } else {
                                         tensor_data[i * col.width + j] = ' '; // Pad with spaces for ASCII?
                                     }
                                 }
                             }
                         } else {
                             // Binary table: read as raw bytes
                             fits_read_col(fptr_, datatype, col_idx + 1, start_row, firstelem, nelements,
                                          nullptr, tensor.data_ptr(), nullptr, &status);
                         }
                    } else if (col.type == FITSColumnType::LOGICAL) {
                         fits_read_col(fptr_, TBYTE, col_idx + 1, start_row, firstelem, nelements,
                                      nullptr, tensor.data_ptr(), nullptr, &status);

                         // Convert 'T'/'F' to 1/0
                         // The tensor is kBool, so its data_ptr is bool*.
                         // We read into it as uint8_t* (char), then convert.
                         uint8_t* data = (uint8_t*)tensor.data_ptr();
                         for (long i = 0; i < nelements; i++) {
                             data[i] = (data[i] == 'T') ? 1 : 0;
                         }
                    } else {
                        #ifdef DEBUG_TABLE
                        fprintf(stderr, "Reading col %d (%s), type %d, datatype %d, rows %ld\n", 
                                col_idx+1, col.name.c_str(), (int)col.type, datatype, num_rows);
                        #endif
                        fits_read_col(fptr_, datatype, col_idx + 1, start_row, firstelem, nelements,
                                      nullptr, tensor.data_ptr(), nullptr, &status);
                    }

                    if (status != 0) {
                         char err_msg[81];
                         fits_get_errstatus(status, err_msg);
                         throw std::runtime_error("Failed to read column " + col.name + ": " + std::string(err_msg));
                    }
                }
            }
        };

        if (use_buffered) {
            try {
                read_columns_buffered(col_indices, start_row, num_rows, result);
            } catch (const std::exception&) {
                // Fallback for CFITSIO edge cases where tblbytes reads fail.
                read_column_by_column();
            }
        } else {
            read_column_by_column();
        }
        
        return result;

    }

    // Memory-mapped column reading
    // Returns a dict of column name to torch::Tensor (or numpy array for strings)
    nb::dict read_columns_mmap(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1) {
        
        if (num_rows == -1) {
            num_rows = nrows_;
        }

        if (nrows_ == 0) {
            return nb::dict();
        }
        
        // Validate rows
        if (start_row < 1 || start_row > nrows_) {
            throw std::runtime_error("Invalid start row");
        }
        if (start_row + num_rows - 1 > nrows_) {
            num_rows = nrows_ - start_row + 1;
        }
        
        std::vector<int> col_indices;
        if (column_names.empty()) {
            for (int i = 0; i < ncols_; i++) col_indices.push_back(i);
        } else {
            for (const auto& name : column_names) {
                bool found = false;
                for (int i = 0; i < ncols_; i++) {
                    if (columns_[i].name == name) {
                        col_indices.push_back(i);
                        found = true;
                        break;
                    }
                }
                if (!found) throw std::runtime_error("Column not found: " + name);
            }
        }

        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type == FITSColumnType::VARIABLE) {
                throw std::runtime_error("VLA columns not supported for mmap");
            }
            if (col.scaled) {
                throw std::runtime_error("Scaled columns not supported for mmap");
            }
        }
        
        // Get offset to the start of the table data
        LONGLONG headstart, data_offset, dataend;
        int status = 0;
        fits_get_hduaddrll(fptr_, &headstart, &data_offset, &dataend, &status);
        if (status != 0) {
             char err_msg[81];
             fits_get_errstatus(status, err_msg);
             throw std::runtime_error("Failed to get HDU data offset: " + std::string(err_msg));
        }
        
        // Open file with mmap
        int fd = open(filename_.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for mmap");
        }
        
        // Get file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to stat file");
        }
        
        // Map the whole file
        void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file");
        }
        
        // Create MMapHandle to manage lifetime
        auto handle = new MMapHandle();
        handle->ptr = map_ptr;
        handle->size = sb.st_size;
        handle->fd = fd;
        handle->owner = true;
        
        // Create a capsule to manage the handle lifetime
        nb::capsule handle_capsule(handle, [](void* p) noexcept {
            auto* h = static_cast<MMapHandle*>(p);
            delete h; // Destructor calls cleanup()
        });
        
        nb::dict result;
        const uint8_t* base_ptr = static_cast<const uint8_t*>(map_ptr) + data_offset;
        
        // Calculate start offset based on start_row (0-based offset)
        size_t row_start_offset = (start_row - 1) * row_width_bytes_;

#if defined(POSIX_MADV_SEQUENTIAL)
        size_t byte_len = static_cast<size_t>(num_rows) * row_width_bytes_;
        posix_madvise(const_cast<uint8_t*>(base_ptr + row_start_offset), byte_len, POSIX_MADV_SEQUENTIAL);
#endif
        
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            
            if (col.type == FITSColumnType::VARIABLE) {
                continue; 
            }
            
            // Pointer to start of column data for the first requested row
            const uint8_t* col_ptr = base_ptr + row_start_offset + col.byte_offset;
            
            // Determine shape
            std::vector<int64_t> shape;
            shape.push_back(num_rows);
            
            if (col.type == FITSColumnType::STRING) {
                // For strings, we return a ByteTensor of shape (num_rows, width)
                // width is the string length (col.width)
                shape.push_back(is_ascii_ ? col.width : col.repeat);
            } else if (col.repeat > 1) {
                shape.push_back(col.repeat);
            }
            
            try {
                // Create Torch Tensor and copy/swap
                torch::ScalarType dtype;
                switch (col.type) {
                    case FITSColumnType::FLOAT: dtype = torch::kFloat32; break;
                    case FITSColumnType::DOUBLE: dtype = torch::kFloat64; break;
                    case FITSColumnType::INT: dtype = torch::kInt32; break;
                    case FITSColumnType::SHORT: dtype = torch::kInt16; break;
                    case FITSColumnType::LONG: dtype = torch::kInt64; break;
                    case FITSColumnType::BYTE: dtype = torch::kUInt8; break;
                    case FITSColumnType::LOGICAL: dtype = torch::kBool; break;
                    case FITSColumnType::STRING: dtype = torch::kUInt8; break;
                    default: dtype = torch::kFloat32;
                }
                
                auto options = torch::TensorOptions().dtype(dtype);
                torch::Tensor tensor = torch::empty(shape, options);
                
                // Parallel copy and swap
                long repeat = (col.repeat > 1) ? col.repeat : 1;
                if (col.type == FITSColumnType::STRING) {
                    repeat = is_ascii_ ? col.width : col.repeat; // String length
                }
                
                if (col.type == FITSColumnType::FLOAT) {
                    float* out = tensor.data_ptr<float>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const int32_t* in = (const int32_t*)(col_ptr + i * row_width_bytes_);
                            float* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                int32_t val = bswap_32(in[j]);
                                memcpy(&row_out[j], &val, sizeof(float));
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::DOUBLE) {
                    double* out = tensor.data_ptr<double>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const int64_t* in = (const int64_t*)(col_ptr + i * row_width_bytes_);
                            double* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                int64_t val = bswap_64(in[j]);
                                memcpy(&row_out[j], &val, sizeof(double));
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::INT) {
                    int32_t* out = tensor.data_ptr<int32_t>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const int32_t* in = (const int32_t*)(col_ptr + i * row_width_bytes_);
                            int32_t* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                row_out[j] = bswap_32(in[j]);
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::SHORT) {
                    int16_t* out = tensor.data_ptr<int16_t>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const int16_t* in = (const int16_t*)(col_ptr + i * row_width_bytes_);
                            int16_t* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                row_out[j] = bswap_16(in[j]);
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::LONG) {
                    int64_t* out = tensor.data_ptr<int64_t>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const int64_t* in = (const int64_t*)(col_ptr + i * row_width_bytes_);
                            int64_t* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                row_out[j] = bswap_64(in[j]);
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::BYTE || col.type == FITSColumnType::STRING) {
                    uint8_t* out = tensor.data_ptr<uint8_t>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* in = (const uint8_t*)(col_ptr + i * row_width_bytes_);
                            uint8_t* row_out = out + i * repeat;
                            memcpy(row_out, in, repeat);
                        }
                    });
                } else if (col.type == FITSColumnType::LOGICAL) {
                    bool* out = tensor.data_ptr<bool>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const char* in = (const char*)(col_ptr + i * row_width_bytes_);
                            bool* row_out = out + i * repeat; // repeat is usually 1 for logical
                            for (long j = 0; j < repeat; j++) {
                                row_out[j] = (in[j] == 'T');
                            }
                        }
                    });
                }
                
                result[col.name.c_str()] = tensor_to_python(tensor);
                
            } catch (const std::exception& e) {
                 #ifdef DEBUG_TABLE
                 fprintf(stderr, "Failed to mmap column %s: %s\n", col.name.c_str(), e.what());
                 #endif
            }
        }
        
        // No need to store handle anymore as we copy everything
        // result["__mmap_handle__"] = handle_capsule;
        
        return result;
    }
    
    std::vector<std::string> get_column_names() const {
        std::vector<std::string> names;
        for (const auto& col : columns_) {
            names.push_back(col.name);
        }
        return names;
    }
    
    // Read a Variable Length Array column
    // Returns a list of tensors (one per row)
    std::vector<torch::Tensor> read_vla_column(int col_idx, long start_row, long num_rows, const ColumnInfo& col) {
        std::vector<torch::Tensor> column_data;
        column_data.reserve(num_rows);
        
        for (long i = 0; i < num_rows; i++) {
            long row = start_row + i;
            
            // Get descriptor (repeat count and offset)
            long repeat = 0;
            long offset = 0;
            int status = 0;
            
            // Note: cfitsio uses 1-based column indices
            fits_read_descript(fptr_, col_idx + 1, row, &repeat, &offset, &status);
            
            if (status != 0) {
                 char err_msg[81];
                 fits_get_errstatus(status, err_msg);
                 throw std::runtime_error("Failed to read VLA descriptor: " + std::string(err_msg));
            }
            
            // Allocate tensor for this row
            std::vector<int64_t> shape;
            if (col.width > 1 && col.type != FITSColumnType::STRING) {
                 // If elements themselves are multi-dimensional? 
                 // FITS VLA is usually 1D array of elements.
                 // But if TFORM has repeat count?
                 // For VLA, repeat count from descriptor is the number of elements.
                 shape.push_back(repeat);
            } else {
                 shape.push_back(repeat);
            }
            
            torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(col.torch_type));
            
            // Read data from heap
            int anynul;
            status = 0;
            
            // Determine type code for cfitsio
            int type_code = 0;
            switch (col.torch_type) {
                case torch::kFloat32: type_code = TFLOAT; break;
                case torch::kFloat64: type_code = TDOUBLE; break;
                case torch::kInt32: type_code = TINT; break;
                case torch::kInt16: type_code = TSHORT; break;
                case torch::kInt64: type_code = TLONGLONG; break;
                case torch::kUInt8: type_code = TBYTE; break;
                case torch::kBool: type_code = TLOGICAL; break;
                default: type_code = TFLOAT;
            }
            
            fits_read_col(fptr_, type_code, col_idx + 1, row, 1, repeat, nullptr, 
                          tensor.data_ptr(), &anynul, &status);
                          
            if (status != 0) {
                 char err_msg[81];
                 fits_get_errstatus(status, err_msg);
                 throw std::runtime_error("Failed to read VLA data: " + std::string(err_msg));
            }
            
            column_data.push_back(tensor);
        }
        
        return column_data;
    }

    // Forward declaration of helper function
    // nb::object tensor_to_python(const torch::Tensor& tensor); // Cannot declare inside class
    // torch::Tensor python_to_tensor(nb::object obj);
    
    // We need to call tensor_to_python which is defined in bindings.cpp
    // But we are in a header-only file included by bindings.cpp
    // So we can just declare it extern?
    // No, bindings.cpp includes table.cpp.
    // So if we declare it at top of table.cpp, it should work.
    
    // Let's remove the duplicate get_column_names first.
    
    // Buffered reading implementation
    void read_columns_buffered(
        const std::vector<int>& col_indices,
        long start_row, long num_rows,
        std::unordered_map<std::string, ColumnData>& result) {
        
        // 16MB chunk size
        const size_t TARGET_CHUNK_SIZE = 16 * 1024 * 1024; 
        long rows_per_chunk = std::max(1L, (long)(TARGET_CHUNK_SIZE / row_width_bytes_));
        
        std::vector<uint8_t> buffer(rows_per_chunk * row_width_bytes_);
        
        long rows_read = 0;
        while (rows_read < num_rows) {
            long current_chunk_rows = std::min(rows_per_chunk, num_rows - rows_read);
            
            int status = 0;
            // Read raw bytes for the chunk of rows
            fits_read_tblbytes(fptr_, start_row + rows_read, 1, current_chunk_rows * row_width_bytes_,
                             buffer.data(), &status);
            
            if (status != 0) {
                 char err_msg[81];
                 fits_get_errstatus(status, err_msg);
                 throw std::runtime_error("Failed to read table bytes: " + std::string(err_msg));
            }
            
            // De-interleave data for each column
            for (int col_idx : col_indices) {
                const auto& col = columns_[col_idx];
                // Get tensor from ColumnData
                torch::Tensor tensor = result[col.name].fixed_data;
                
                // Get pointer to tensor data at current offset
                uint8_t* dest_ptr = (uint8_t*)get_tensor_data_ptr(tensor, rows_read * col.repeat);
                
                // Extract and swap bytes
                extract_column_data(buffer.data(), current_chunk_rows, col, dest_ptr);
            }
            
            rows_read += current_chunk_rows;
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

    void extract_column_data(const uint8_t* buffer, long num_rows, const ColumnInfo& col, uint8_t* dest) {
        size_t col_width = col.width; // bytes per element
        size_t total_width = col.width * col.repeat; // bytes per cell
        size_t row_stride = row_width_bytes_;
        size_t col_offset = col.byte_offset;
        
        // Optimized loops for common types
        // Note: FITS is Big Endian, we need to swap if host is Little Endian
        // Assuming Little Endian host (x86/ARM)
        
        if (col.type == FITSColumnType::LOGICAL) {
             // Convert 'T'/'F' (or '1'/'0') to bool
             bool* out = reinterpret_cast<bool*>(dest);
             for (long i = 0; i < num_rows; i++) {
                 const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                 for (int j = 0; j < col.repeat; j++) {
                     const uint8_t v = src_cell[j];
                     out[i * col.repeat + j] = (v == 'T' || v == '1');
                 }
             }
        } else if (col.type == FITSColumnType::STRING || col.type == FITSColumnType::BYTE) {
             // No swapping needed for bytes/strings
             for (long i = 0; i < num_rows; i++) {
                 std::memcpy(dest + i * total_width, buffer + i * row_stride + col_offset, total_width);
             }
        } else if (col_width == 2) {
            // Int16
            uint16_t* d = (uint16_t*)dest;
            for (long i = 0; i < num_rows * col.repeat; i++) {
                // Need to handle repeat stride if repeat > 1
                // Actually, buffer layout is: [Row1][Row2]...
                // Row1: ... [ColData] ...
                // ColData: [Elem1][Elem2]...
                // So we can just copy the block if we handle stride correctly.
                // But wait, we need to iterate rows.
            }
            
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint16_t* dest_cell = (uint16_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint16_t val;
                    std::memcpy(&val, src_cell + j * 2, 2);
                    dest_cell[j] = __builtin_bswap16(val);
                }
            }
        } else if (col_width == 4) {
            // Int32, Float32
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint32_t* dest_cell = (uint32_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint32_t val;
                    std::memcpy(&val, src_cell + j * 4, 4);
                    dest_cell[j] = __builtin_bswap32(val);
                }
            }
        } else if (col_width == 8) {
            // Int64, Double
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint64_t* dest_cell = (uint64_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint64_t val;
                    std::memcpy(&val, src_cell + j * 8, 8);
                    dest_cell[j] = __builtin_bswap64(val);
                }
            }
        } else {
            // Fallback memcpy (should not happen for standard types needing swap)
             for (long i = 0; i < num_rows; i++) {
                 std::memcpy(dest + i * total_width, buffer + i * row_stride + col_offset, total_width);
             }
        }
    }
    
    long get_num_rows() const { return nrows_; }
    int get_num_cols() const { return ncols_; }

private:
    fitsfile* fptr_ = nullptr;
    std::string filename_;
    int hdu_num_;
    long nrows_;
    int ncols_;
    long row_width_bytes_ = 0;
    std::vector<ColumnInfo> columns_;
    bool is_ascii_ = false;
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
        // Cast to fitsfile pointer directly since we can't include FITSFile here
        auto* fptr = reinterpret_cast<fitsfile*>(handle);
        return new torchfits::TableReader(fptr, hdu_num);
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
        std::string path = filename ? filename : "";
        if (!path.empty() && path[0] != '!') {
            path = "!" + path;
        }
        fits_create_file(&fptr, path.c_str(), &status);
    } else {
        fits_create_file(&fptr, filename, &status);
    }

    if (status != 0) {
        throw std::runtime_error("Failed to open FITS file for writing");
    }
    
    try {
        torchfits::write_table_hdu(fptr, tensor_dict, header);
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }
    
    fits_close_file(fptr, &status);
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
