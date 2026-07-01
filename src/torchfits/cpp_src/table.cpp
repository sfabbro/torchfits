/**
 * High-performance FITS table reader with memory pools.
 * Phase 2 implementation supporting all FITS column types.
 */

#include "torchfits_torch.h"
#include <vector>
#include <memory>
#include <algorithm>
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
#include <new>
#include <cstring>  // for memset
#include <cstdlib>  // for getenv

#undef READONLY
#include <fitsio.h>
#include "hardware.h"
#include "security.h"
#include "torch_compat.h"
#include "fits_helpers.h"
#include "cache.h"

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace torchfits {
void write_table_hdu(fitsfile* fptr, nb::dict tensor_dict, nb::dict header, nb::object schema_obj, bool is_ascii);


namespace {

bool table_buffered_read_enabled() {
    // Keep buffered row-path enabled by default; allow fast local bisects.
    static const bool enabled = []() {
        const char* env = std::getenv("TORCHFITS_TABLE_BUFFERED");
        if (!env || env[0] == '\0') {
            return true;
        }
        return !(env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F');
    }();
    return enabled;
}

} // namespace



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
    BIT,        // X (bit array)
    BYTE,       // B
    SHORT,      // I
    INT,        // J
    LONG,       // K
    FLOAT,      // E
    DOUBLE,     // D
    COMPLEX_FLOAT,   // C
    COMPLEX_DOUBLE,  // M
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
    bool is_unsigned_int = false;  // uint16/uint32 FITS convention (TZERO offset)
    int64_t unsigned_offset = 0;   // 32768 or 2147483648
    long storage_bytes = 0;        // Physical bytes occupied by one table row
};


    // Filter operations
    enum class FilterOp {
        EQ, NE, GT, LT, GE, LE
    };

    struct TableFilter {
        std::string col_name;
        FilterOp op;
        double val_d = 0.0;
        int64_t val_i = 0;
        std::string val_s;
        // 0=double, 1=int, 2=string
        int type_idx = 0;
    };

class TableReader {
public:
    TableReader(const std::string& filename, int hdu_num = 1) : filename_(filename), hdu_num_(hdu_num), use_cache_(true) {
        torchfits::check_fits_filename_security(filename);
        target_hdu_ = hdu_num + 1;  // CFITSIO 1-based absolute HDU
        fptr_ = torchfits::get_or_open_cached(filename);
        if (!fptr_) {
            throw std::runtime_error("Failed to open FITS file");
        }

        int status = 0;
        fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }

        analyze_table();
    }

    TableReader(fitsfile* fptr, int hdu_num = 1) : fptr_(fptr), hdu_num_(hdu_num), use_cache_(false) {
        int status = 0;
        target_hdu_ = hdu_num + 1;
        fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
        analyze_table();
    }

    ~TableReader() {
        if (fptr_) {
            if (use_cache_) {
                torchfits::release_cached(filename_);
            }
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

            snprintf(keyname, FLEN_KEYWORD, "TFORM%d", i);
            fits_read_key(fptr_, TSTRING, keyname, tform, nullptr, &col_status);
            if (col_status != 0) {
                col_status = 0;
                tform[0] = '\0';
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

            // Detect unsigned integer FITS convention:
            //   uint16 stored as int16 with TZERO=32768, TSCAL=1.0
            //   uint32 stored as int32 with TZERO=2147483648, TSCAL=1.0
            if (col.tscale == 1.0) {
                if ((typecode == TSHORT || typecode == TUSHORT) &&
                    col.tzero == 32768.0) {
                    col.is_unsigned_int = true;
                    col.unsigned_offset = 32768;
                    col.scaled = false;  // skip float64 scaling; apply offset instead
                } else if ((typecode == TINT || typecode == TUINT) &&
                           col.tzero == 2147483648.0) {
                    col.is_unsigned_int = true;
                    col.unsigned_offset = 2147483648;
                    col.scaled = false;
                }
            }

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
                    case TLONG: col.torch_type = torch::kInt32; break;
                    case TLONGLONG: col.torch_type = torch::kInt64; break;
                    case TFLOAT: col.torch_type = torch::kFloat32; break;
                    case TDOUBLE: col.torch_type = torch::kFloat64; break;
                    default: col.torch_type = torch::kFloat32;
                }
                col.width = 8;
                // P descriptors occupy 8 bytes and Q descriptors occupy 16 bytes,
                // regardless of the maximum element repeat reported by CFITSIO.
                const std::string format(tform);
                col.storage_bytes =
                    (format.find('Q') != std::string::npos ||
                     format.find('q') != std::string::npos)
                    ? 16
                    : 8;
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
                    case TBIT:
                        col.type = FITSColumnType::BIT;
                        col.torch_type = torch::kUInt8;
                        // Expose bit arrays as uint8[repeat] values (0/1).
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
                        // CFITSIO reports TLONG as FITS 32-bit integer (same code as TINT32BIT).
                        col.type = FITSColumnType::INT;
                        col.torch_type = torch::kInt32;
                        col.width = 4;
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
#ifdef TCOMPLEX
                    case TCOMPLEX:
                        col.type = FITSColumnType::COMPLEX_FLOAT;
                        col.torch_type = at::kComplexFloat;
                        col.width = 8; // two float32 values
                        break;
#endif
#ifdef TDBLCOMPLEX
                    case TDBLCOMPLEX:
                        col.type = FITSColumnType::COMPLEX_DOUBLE;
                        col.torch_type = at::kComplexDouble;
                        col.width = 16; // two float64 values
                        break;
#endif
                    default:
                        throw std::runtime_error(
                            "Unsupported FITS column typecode " + std::to_string(typecode) +
                            " for column " + std::string(ttype)
                        );
                }
                if (col.type == FITSColumnType::BIT) {
                    col.storage_bytes = (col.repeat + 7L) / 8L;
                } else {
                    col.storage_bytes =
                        static_cast<long>(col.width) * static_cast<long>(col.repeat);
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
            current_offset += col.storage_bytes;
        }
        long declared_row_width = 0;
        int row_width_status = 0;
        fits_read_key(
            fptr_,
            TLONG,
            const_cast<char*>("NAXIS1"),
            &declared_row_width,
            nullptr,
            &row_width_status
        );
        row_width_bytes_ =
            (row_width_status == 0 && declared_row_width > 0)
            ? declared_row_width
            : current_offset;
    }

    // Helper struct to hold column data (either fixed or VLA)
    struct ColumnData {
        bool is_vla;
        torch::Tensor fixed_data;
        std::vector<torch::Tensor> vla_data;
        torch::Tensor vla_offsets;

        ColumnData() : is_vla(false) {}
        ColumnData(torch::Tensor t) : is_vla(false), fixed_data(t) {}
        ColumnData(std::vector<torch::Tensor> v) : is_vla(true), vla_data(v) {}
        ColumnData(torch::Tensor values, torch::Tensor offsets, bool /*flat_vla*/)
            : is_vla(true), fixed_data(values), vla_offsets(offsets) {}
    };

    // Read columns from the table
    // Returns a map of column name to ColumnData
    std::unordered_map<std::string, ColumnData> read_columns(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1, bool vla_flat = false) {

        ensure_table_hdu();
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

        auto cfitsio_read_datatype = [](const ColumnInfo& col) -> int {
            switch (col.type) {
                case FITSColumnType::LOGICAL: return TLOGICAL;
                case FITSColumnType::BIT: return TBIT;
                case FITSColumnType::BYTE: return TBYTE;
                case FITSColumnType::SHORT: return TSHORT;
                case FITSColumnType::INT: return TINT;
                case FITSColumnType::LONG: return TLONGLONG;
                case FITSColumnType::FLOAT: return TFLOAT;
                case FITSColumnType::DOUBLE: return TDOUBLE;
#ifdef TCOMPLEX
                case FITSColumnType::COMPLEX_FLOAT: return TCOMPLEX;
#endif
#ifdef TDBLCOMPLEX
                case FITSColumnType::COMPLEX_DOUBLE: return TDBLCOMPLEX;
#endif
                case FITSColumnType::STRING: return TBYTE; // Read strings as bytes
                default: return TFLOAT;
            }
        };


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

        // Row-buffered reads via fits_read_tblbytes when the selected fixed-width
        // columns cover the full binary-table row payload (no unused bytes).
        // Partial column projections use per-column fits_read_col to avoid I/O waste.
        long requested_bytes = 0;
        bool has_vla = false;
        bool has_bit = false;
        bool has_complex = false;
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type == FITSColumnType::VARIABLE) {
                has_vla = true;
            } else if (col.type == FITSColumnType::BIT) {
                has_bit = true;
            } else if (col.type == FITSColumnType::COMPLEX_FLOAT || col.type == FITSColumnType::COMPLEX_DOUBLE) {
                has_complex = true;
            } else {
                requested_bytes += col.width * col.repeat;
            }
        }

        bool use_buffered = false;
        if (table_buffered_read_enabled() &&
            !is_ascii_ && !has_vla && !has_bit && !has_complex && row_width_bytes_ > 0) {
            use_buffered = (requested_bytes == row_width_bytes_);
        }

        auto read_column_by_column = [&]() {
            // Read column by column
            for (int col_idx : col_indices) {
                const auto& col = columns_[col_idx];

                if (col.type == FITSColumnType::VARIABLE) {
                    // Read VLA column
                    if (vla_flat) {
                        auto flat = read_vla_column_flat(col_idx, start_row, num_rows, col);
                        result[col.name] = ColumnData(std::move(flat.first), std::move(flat.second), true);
                    } else {
                        result[col.name] = ColumnData(read_vla_column(col_idx, start_row, num_rows, col));
                    }
                } else {
                    // Read fixed width column
                    torch::Tensor tensor = result[col.name].fixed_data;

                    int status = 0;
                    // Use fits_read_col to read directly into tensor memory
                    // Note: fits_read_col handles byte swapping automatically!

                    int datatype = cfitsio_read_datatype(col);

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

        // Apply FITS TSCAL/TZERO in-memory for integer-like columns.
        // This preserves physical values while keeping the read path raw and fast.
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (!col.scaled ||
                col.type == FITSColumnType::FLOAT ||
                col.type == FITSColumnType::DOUBLE ||
                col.type == FITSColumnType::COMPLEX_FLOAT ||
                col.type == FITSColumnType::COMPLEX_DOUBLE ||
                col.type == FITSColumnType::STRING ||
                col.type == FITSColumnType::LOGICAL ||
                col.type == FITSColumnType::VARIABLE) {
                continue;
            }
            auto it = result.find(col.name);
            if (it == result.end() || !it->second.fixed_data.defined()) {
                continue;
            }
            torch::Tensor scaled = it->second.fixed_data.to(torch::kFloat64);
            if (col.tscale != 1.0) {
                scaled.mul_(col.tscale);
            }
            if (col.tzero != 0.0) {
                scaled.add_(col.tzero);
            }
            it->second.fixed_data = scaled;
        }

        // Apply BIT→bool coercion directly in C++.
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type != FITSColumnType::BIT) continue;
            auto it = result.find(col.name);
            if (it == result.end() || !it->second.fixed_data.defined()) continue;
            it->second.fixed_data = it->second.fixed_data.to(torch::kBool);
        }

        // Apply unsigned integer offset for uint16/uint32 FITS convention.
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (!col.is_unsigned_int) continue;
            auto it = result.find(col.name);
            if (it == result.end() || !it->second.fixed_data.defined()) continue;
            torch::Tensor converted = it->second.fixed_data.to(torch::kInt64);
            converted.add_(col.unsigned_offset);
            it->second.fixed_data = converted;
        }

        return result;

    }

    // Memory-mapped column reading
    // Returns a dict of column name to torch::Tensor (or numpy array for strings)
    nb::dict read_columns_mmap(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1) {

        ensure_table_hdu();
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
            if (col.type == FITSColumnType::BIT) {
                // BIT columns are read as uint8 from mmap; converted to bool below.
            }
            if (col.scaled && !col.is_unsigned_int) {
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

        // RAII mmap guard — tensors are copies (not views) so the mmap only
        // needs to survive for the duration of this function.
        // size must match the original mmap() length (full file) passed to munmap().
        MMapHandle mmap_guard(map_ptr, sb.st_size, fd);

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
            // Complex columns are supported for mmap reads below (byte-swap path).

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
                    case FITSColumnType::BIT: dtype = torch::kBool; break;
                    case FITSColumnType::LOGICAL: dtype = torch::kBool; break;
                    case FITSColumnType::STRING: dtype = torch::kUInt8; break;
                    case FITSColumnType::COMPLEX_FLOAT: dtype = at::kComplexFloat; break;
                    case FITSColumnType::COMPLEX_DOUBLE: dtype = at::kComplexDouble; break;
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
                    if (repeat == 1) {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int32_t raw_val;
                                memcpy(&raw_val, col_ptr + i * row_width_bytes_, sizeof(int32_t));
                                int32_t val = bswap_32(raw_val);
                                memcpy(&out[i], &val, sizeof(float));
                            }
                        });
                    } else {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                float* row_out = out + i * repeat;
                                for (long j = 0; j < repeat; j++) {
                                    int32_t raw_val;
                                    memcpy(&raw_val, col_ptr + i * row_width_bytes_ + j * sizeof(int32_t), sizeof(int32_t));
                                    int32_t val = bswap_32(raw_val);
                                    memcpy(&row_out[j], &val, sizeof(float));
                                }
                            }
                        });
                    }
                } else if (col.type == FITSColumnType::DOUBLE) {
                    double* out = tensor.data_ptr<double>();
                    if (repeat == 1) {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int64_t raw_val;
                                memcpy(&raw_val, col_ptr + i * row_width_bytes_, sizeof(int64_t));
                                int64_t val = bswap_64(raw_val);
                                memcpy(&out[i], &val, sizeof(double));
                            }
                        });
                    } else {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                double* row_out = out + i * repeat;
                                for (long j = 0; j < repeat; j++) {
                                    int64_t raw_val;
                                    memcpy(&raw_val, col_ptr + i * row_width_bytes_ + j * sizeof(int64_t), sizeof(int64_t));
                                    int64_t val = bswap_64(raw_val);
                                    memcpy(&row_out[j], &val, sizeof(double));
                                }
                            }
                        });
                    }
                } else if (col.type == FITSColumnType::INT) {
                    int32_t* out = tensor.data_ptr<int32_t>();
                    if (repeat == 1) {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int32_t raw_val;
                                memcpy(&raw_val, col_ptr + i * row_width_bytes_, sizeof(int32_t));
                                out[i] = bswap_32(raw_val);
                            }
                        });
                    } else {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int32_t* row_out = out + i * repeat;
                                for (long j = 0; j < repeat; j++) {
                                    int32_t raw_val;
                                    memcpy(&raw_val, col_ptr + i * row_width_bytes_ + j * sizeof(int32_t), sizeof(int32_t));
                                    row_out[j] = bswap_32(raw_val);
                                }
                            }
                        });
                    }
                } else if (col.type == FITSColumnType::SHORT) {
                    int16_t* out = tensor.data_ptr<int16_t>();
                    if (repeat == 1) {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int16_t raw_val;
                                memcpy(&raw_val, col_ptr + i * row_width_bytes_, sizeof(int16_t));
                                out[i] = bswap_16(raw_val);
                            }
                        });
                    } else {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int16_t* row_out = out + i * repeat;
                                for (long j = 0; j < repeat; j++) {
                                    int16_t raw_val;
                                    memcpy(&raw_val, col_ptr + i * row_width_bytes_ + j * sizeof(int16_t), sizeof(int16_t));
                                    row_out[j] = bswap_16(raw_val);
                                }
                            }
                        });
                    }
                } else if (col.type == FITSColumnType::LONG) {
                    int64_t* out = tensor.data_ptr<int64_t>();
                    if (repeat == 1) {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int64_t raw_val;
                                memcpy(&raw_val, col_ptr + i * row_width_bytes_, sizeof(int64_t));
                                out[i] = bswap_64(raw_val);
                            }
                        });
                    } else {
                        at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                            for (long i = start; i < end; i++) {
                                int64_t* row_out = out + i * repeat;
                                for (long j = 0; j < repeat; j++) {
                                    int64_t raw_val;
                                    memcpy(&raw_val, col_ptr + i * row_width_bytes_ + j * sizeof(int64_t), sizeof(int64_t));
                                    row_out[j] = bswap_64(raw_val);
                                }
                            }
                        });
                    }
                } else if (col.type == FITSColumnType::BYTE || col.type == FITSColumnType::STRING) {
                    uint8_t* out = tensor.data_ptr<uint8_t>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* in = (const uint8_t*)(col_ptr + i * row_width_bytes_);
                            uint8_t* row_out = out + i * repeat;
                            memcpy(row_out, in, repeat);
                        }
                    });
                } else if (col.type == FITSColumnType::BIT) {
                    bool* out = tensor.data_ptr<bool>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* row_in =
                                col_ptr + i * row_width_bytes_;
                            bool* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                const uint8_t packed = row_in[j / 8];
                                row_out[j] =
                                    ((packed >> (7 - (j % 8))) & 0x1U) != 0;
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::COMPLEX_FLOAT) {
                    c10::complex<float>* out = tensor.data_ptr<c10::complex<float>>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* row_in =
                                col_ptr + i * row_width_bytes_;
                            c10::complex<float>* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                int32_t re_bits, im_bits;
                                std::memcpy(
                                    &re_bits,
                                    row_in + j * 2 * sizeof(int32_t),
                                    sizeof(int32_t)
                                );
                                std::memcpy(
                                    &im_bits,
                                    row_in + (j * 2 + 1) * sizeof(int32_t),
                                    sizeof(int32_t)
                                );
                                float re, im;
                                re_bits = bswap_32(re_bits);
                                im_bits = bswap_32(im_bits);
                                std::memcpy(&re, &re_bits, sizeof(float));
                                std::memcpy(&im, &im_bits, sizeof(float));
                                row_out[j] = c10::complex<float>(re, im);
                            }
                        }
                    });
                } else if (col.type == FITSColumnType::COMPLEX_DOUBLE) {
                    c10::complex<double>* out = tensor.data_ptr<c10::complex<double>>();
                    at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* row_in =
                                col_ptr + i * row_width_bytes_;
                            c10::complex<double>* row_out = out + i * repeat;
                            for (long j = 0; j < repeat; j++) {
                                int64_t re_bits, im_bits;
                                std::memcpy(
                                    &re_bits,
                                    row_in + j * 2 * sizeof(int64_t),
                                    sizeof(int64_t)
                                );
                                std::memcpy(
                                    &im_bits,
                                    row_in + (j * 2 + 1) * sizeof(int64_t),
                                    sizeof(int64_t)
                                );
                                double re, im;
                                re_bits = bswap_64(re_bits);
                                im_bits = bswap_64(im_bits);
                                std::memcpy(&re, &re_bits, sizeof(double));
                                std::memcpy(&im, &im_bits, sizeof(double));
                                row_out[j] = c10::complex<double>(re, im);
                            }
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

                // Apply unsigned-int coercion before storing.
                if (col.is_unsigned_int) {
                    tensor = tensor.to(torch::kInt64);
                    tensor.add_(col.unsigned_offset);
                }
                result[col.name.c_str()] = tensor_to_python(tensor);

            } catch (const std::exception& e) {
                 #ifdef DEBUG_TABLE
                 fprintf(stderr, "Failed to mmap column %s: %s\n", col.name.c_str(), e.what());
                 #endif
            }
        }

        return result;
    }

    // Filtered reading
    std::unordered_map<std::string, torch::Tensor> read_columns_mmap_filtered(
        const std::vector<std::string>& column_names,
        const std::vector<TableFilter>& filters) {

        ensure_table_hdu();
        if (filters.empty()) {
            // Need to convert read_columns_mmap result (nb::dict) to map<string, Tensor>?
            // Or just throw error -> filters shouldn't be empty here if called correctly.
            // But for robustness:
             throw std::runtime_error("Filters cannot be empty in read_columns_mmap_filtered");
        }

        // Map file
        int fd = open(filename_.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Failed to open file");

        struct stat sb;
        if (fstat(fd, &sb) == -1) { close(fd); throw std::runtime_error("Failed to stat file"); }

        void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map_ptr == MAP_FAILED) { close(fd); throw std::runtime_error("Failed to mmap"); }

        MMapHandle guard(map_ptr, (size_t)sb.st_size, fd);

        uint8_t* base_ptr = static_cast<uint8_t*>(map_ptr);

        // fast-forward to data
        int status = 0;
        LONGLONG headstart, data_start, data_end;
        fits_get_hduaddrll(fptr_, &headstart, &data_start, &data_end, &status);
        if (status) throw std::runtime_error("Failed to get HDU address");

        uint8_t* data_ptr = base_ptr + data_start;

        // Resolve filter columns
        struct FilterContext {
            const TableFilter* filter;
            const ColumnInfo* col_info;
            size_t offset;
            // Cached types for fast switch
            bool is_float = false;
            bool is_double = false;
            bool is_int = false;
            bool is_long = false;
            bool is_short = false;
            bool is_byte = false;
        };

        std::vector<FilterContext> ctxs;
        for (const auto& f : filters) {
            bool found = false;
            for (const auto& c : columns_) {
                if (c.name == f.col_name) {
                    FilterContext ctx;
                    ctx.filter = &f;
                    ctx.col_info = &c;
                    ctx.offset = c.byte_offset;

                    if (c.type == FITSColumnType::FLOAT) ctx.is_float = true;
                    else if (c.type == FITSColumnType::DOUBLE) ctx.is_double = true;
                    else if (c.type == FITSColumnType::INT) ctx.is_int = true;
                    else if (c.type == FITSColumnType::LONG) ctx.is_long = true;
                    else if (c.type == FITSColumnType::SHORT) ctx.is_short = true;
                    else if (c.type == FITSColumnType::BYTE) ctx.is_byte = true;

                    ctxs.push_back(ctx);
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("Filter column not found: " + f.col_name);
        }

        // Scan rows
        std::vector<long> valid_indices;
        valid_indices.reserve(nrows_); // Avoid reallocations for high selectivity

        if (ctxs.size() == 1) {
            const auto& ctx = ctxs[0];
            const size_t offset = ctx.offset;
            const FilterOp op = ctx.filter->op;

            if (ctx.is_int) {
                const int32_t target = (int32_t)ctx.filter->val_i;
                for (long i = 0; i < nrows_; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint32_t tmp;
                    std::memcpy(&tmp, val_ptr, 4);
                    int32_t val = (int32_t)bswap_32(tmp);

                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        case FilterOp::NE: match = (val != target); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) valid_indices.push_back(i);
                }
            } else if (ctx.is_float) {
                const float target = (float)ctx.filter->val_d;
                for (long i = 0; i < nrows_; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint32_t tmp;
                    std::memcpy(&tmp, val_ptr, 4);
                    tmp = bswap_32(tmp);
                    float val;
                    std::memcpy(&val, &tmp, 4);

                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        case FilterOp::NE: match = (val != target); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) valid_indices.push_back(i);
                }
            } else if (ctx.is_double) {
                const double target = ctx.filter->val_d;
                for (long i = 0; i < nrows_; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint64_t tmp;
                    std::memcpy(&tmp, val_ptr, 8);
                    tmp = bswap_64(tmp);
                    double val;
                    std::memcpy(&val, &tmp, 8);

                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        case FilterOp::NE: match = (val != target); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) valid_indices.push_back(i);
                }
            } else if (ctx.is_long) {
                const int64_t target = ctx.filter->val_i;
                for (long i = 0; i < nrows_; i++) {
                    uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                    uint64_t tmp;
                    std::memcpy(&tmp, val_ptr, 8);
                    int64_t val = (int64_t)bswap_64(tmp);

                    bool match = false;
                    switch (op) {
                        case FilterOp::EQ: match = (val == target); break;
                        case FilterOp::NE: match = (val != target); break;
                        case FilterOp::GT: match = (val > target); break;
                        case FilterOp::LT: match = (val < target); break;
                        case FilterOp::GE: match = (val >= target); break;
                        case FilterOp::LE: match = (val <= target); break;
                    }
                    if (match) valid_indices.push_back(i);
                }
            } else {
                // Fallback for other single-filter types
                for (long i = 0; i < nrows_; i++) {
                    uint8_t* row_ptr = data_ptr + i * row_width_bytes_;
                    uint8_t* val_ptr = row_ptr + ctx.offset;
                    bool match = false;
                    if (ctx.is_short) {
                        uint16_t tmp; std::memcpy(&tmp, val_ptr, 2);
                        int16_t val = (int16_t)bswap_16(tmp);
                        int64_t target = ctx.filter->val_i;
                        switch (op) {
                            case FilterOp::EQ: match = (val == target); break;
                            case FilterOp::NE: match = (val != target); break;
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                        }
                    } else if (ctx.is_byte) {
                        uint8_t val = *val_ptr;
                        int64_t target = ctx.filter->val_i;
                        switch (op) {
                            case FilterOp::EQ: match = (val == target); break;
                            case FilterOp::NE: match = (val != target); break;
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                        }
                    }
                    if (match) valid_indices.push_back(i);
                }
            }
        } else {
            // Original multi-filter fallback
            for (long i = 0; i < nrows_; i++) {
                uint8_t* row_ptr = data_ptr + i * row_width_bytes_;
                bool row_match = true;
                for (const auto& ctx : ctxs) {
                    uint8_t* val_ptr = row_ptr + ctx.offset;
                    bool match = false;
                    if (ctx.is_double) {
                        uint64_t tmp; std::memcpy(&tmp, val_ptr, 8); tmp = bswap_64(tmp);
                        double val; std::memcpy(&val, &tmp, 8);
                        double target = ctx.filter->val_d;
                        switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                        }
                    } else if (ctx.is_float) {
                        uint32_t tmp; std::memcpy(&tmp, val_ptr, 4); tmp = bswap_32(tmp);
                        float val; std::memcpy(&val, &tmp, 4);
                        float target = (float)ctx.filter->val_d;
                        switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                        }
                    } else if (ctx.is_long) {
                       uint64_t tmp; memcpy(&tmp, val_ptr, 8);
                       int64_t val = (int64_t)bswap_64(tmp);
                       int64_t target = ctx.filter->val_i;
                       switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                       }
                    } else if (ctx.is_int) {
                       uint32_t tmp; memcpy(&tmp, val_ptr, 4);
                       int32_t val = (int32_t)bswap_32(tmp);
                       int64_t target = ctx.filter->val_i;
                       switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                       }
                    } else if (ctx.is_short) {
                       uint16_t tmp; memcpy(&tmp, val_ptr, 2);
                       int16_t val = (int16_t)bswap_16(tmp);
                       int64_t target = ctx.filter->val_i;
                       switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                       }
                    } else if (ctx.is_byte) {
                       uint8_t val = *val_ptr;
                       int64_t target = ctx.filter->val_i;
                       switch (ctx.filter->op) {
                           case FilterOp::EQ: match = (val == target); break;
                           case FilterOp::NE: match = (val != target); break;
                           case FilterOp::GT: match = (val > target); break;
                           case FilterOp::LT: match = (val < target); break;
                           case FilterOp::GE: match = (val >= target); break;
                           case FilterOp::LE: match = (val <= target); break;
                       }
                    }
                    if (!match) { row_match = false; break; }
                }
                if (row_match) valid_indices.push_back(i);
            }
        }

        // Gather results
        std::unordered_map<std::string, torch::Tensor> result;
        long num_valid = valid_indices.size();
        if (num_valid == 0) return result;

        std::vector<int> out_col_indices;
        if (column_names.empty()) {
            for(int i=0; i<ncols_; ++i) out_col_indices.push_back(i);
        } else {
             for(const auto& name : column_names) {
                 for(int i=0; i<ncols_; ++i) {
                     if(columns_[i].name == name) {
                         out_col_indices.push_back(i);
                         break;
                     }
                 }
             }
        }

        for (int col_idx : out_col_indices) {
            const auto& col = columns_[col_idx];
            if (col.type == FITSColumnType::VARIABLE) continue; // Skip VLA for now

            // Allocate output tensor
            std::vector<int64_t> shape;
            shape.push_back(num_valid);
             if (col.type == FITSColumnType::STRING) {
                shape.push_back(is_ascii_ ? col.width : col.repeat);
            } else if (col.repeat > 1) {
                shape.push_back(col.repeat);
            }

            auto options = torch::TensorOptions().dtype(col.torch_type);
            torch::Tensor out_tensor = torch::empty(shape, options);

            int item_size = 0;
            if (col.type == FITSColumnType::DOUBLE || col.type == FITSColumnType::LONG) item_size = 8;
            else if (col.type == FITSColumnType::FLOAT || col.type == FITSColumnType::INT) item_size = 4;
            else if (col.type == FITSColumnType::SHORT) item_size = 2;
            else item_size = 1;

            size_t cell_size = item_size * ( (col.type == FITSColumnType::STRING) ? (is_ascii_ ? col.width : col.repeat) : std::max(1, col.repeat));
            uint8_t* out_ptr = (uint8_t*)out_tensor.data_ptr();

            // Optimized gathering with contiguous block detection
            long k = 0;
            while (k < num_valid) {
                long run_start_k = k;
                long run_start_row = valid_indices[k];
                while (k + 1 < num_valid && valid_indices[k+1] == valid_indices[k] + 1) {
                    k++;
                }
                long run_len = k - run_start_k + 1;

                const uint8_t* src_base = data_ptr + run_start_row * row_width_bytes_ + col.byte_offset;
                uint8_t* dst_base = out_ptr + run_start_k * cell_size;

                if (item_size == 1 && row_width_bytes_ == cell_size) {
                    // Fully contiguous case: single memcpy
                    std::memcpy(dst_base, src_base, run_len * cell_size);
                } else {
                    // Contiguous rows but scattered in file due to other columns
                    for (long r = 0; r < run_len; ++r) {
                        const uint8_t* src = src_base + r * row_width_bytes_;
                        uint8_t* dst = dst_base + r * cell_size;

                        if (item_size == 1) {
                            std::memcpy(dst, src, cell_size);
                        } else if (item_size == 2) {
                            int n_items = cell_size / 2;
                            uint16_t* d = (uint16_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint16_t val; std::memcpy(&val, src + j*2, 2);
                                d[j] = bswap_16(val);
                            }
                        } else if (item_size == 4) {
                            int n_items = cell_size / 4;
                            uint32_t* d = (uint32_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint32_t val; std::memcpy(&val, src + j*4, 4);
                                d[j] = bswap_32(val);
                            }
                        } else if (item_size == 8) {
                            int n_items = cell_size / 8;
                            uint64_t* d = (uint64_t*)dst;
                            for(int j=0; j<n_items; ++j) {
                                uint64_t val; std::memcpy(&val, src + j*8, 8);
                                d[j] = bswap_64(val);
                            }
                        }
                    }
                }
                k++; // Move to next after the run
            }

            result[col.name] = out_tensor;
        }

        return result;
    }


    void update_rows_mmap(nb::dict tensor_dict, long start_row, long num_rows) {
        if (num_rows == -1) {
            num_rows = nrows_ - start_row + 1;
        }
        if (num_rows <= 0) {
            return;
        }
        ensure_table_hdu();
        if (start_row < 1 || start_row > nrows_) {
            throw std::runtime_error("Invalid start row");
        }
        if (start_row + num_rows - 1 > nrows_) {
            throw std::runtime_error("Row range exceeds table length");
        }

        // Build column index map
        std::unordered_map<std::string, const ColumnInfo*> column_map;
        column_map.reserve(columns_.size());
        for (const auto& col : columns_) {
            column_map[col.name] = &col;
        }

        // Validate columns and types
        for (auto item : tensor_dict) {
            std::string name = nb::cast<std::string>(item.first);
            auto it = column_map.find(name);
            if (it == column_map.end()) {
                throw std::runtime_error("Column not found: " + name);
            }
            const ColumnInfo* col = it->second;
            if (col->type == FITSColumnType::VARIABLE) {
                throw std::runtime_error("VLA columns not supported for mmap updates");
            }
            if (col->scaled) {
                throw std::runtime_error("Scaled columns not supported for mmap updates");
            }
            // BIT, STRING, COMPLEX_FLOAT, COMPLEX_DOUBLE, and LOGICAL are
            // accepted and handled by the per-element dispatches below. Other
            // column types fall through to the `default` branch in the write loop.
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

        int fd = open(filename_.c_str(), O_RDWR);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for mmap update");
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Failed to stat file for mmap update");
        }

        void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file for update");
        }

        uint8_t* base_ptr = static_cast<uint8_t*>(map_ptr) + data_offset;
        size_t row_start_offset = static_cast<size_t>(start_row - 1) * row_width_bytes_;

        for (auto item : tensor_dict) {
            std::string name = nb::cast<std::string>(item.first);
            const ColumnInfo* col = column_map.at(name);

            nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
            int ndim = tensor.ndim();
            long rows = 1;
            long repeat = 1;
            if (ndim == 0) {
                rows = 1;
                repeat = 1;
            } else if (ndim == 1) {
                rows = static_cast<long>(tensor.shape(0));
                repeat = 1;
            } else if (ndim == 2) {
                rows = static_cast<long>(tensor.shape(0));
                repeat = static_cast<long>(tensor.shape(1));
            } else {
                munmap(map_ptr, sb.st_size);
                close(fd);
                throw std::runtime_error("update_rows mmap only supports 1D/2D columns for " + name);
            }

            long expected_repeat = (col->repeat > 0) ? col->repeat : 1;
            // STRING columns allow shorter user-provided widths; trailing bytes
            // are padded with ASCII spaces to match the FITS CHAR contract.
            // BIT and COMPLEX columns require an exact repeat match.
            long user_repeat = repeat;
            if (col->type == FITSColumnType::STRING) {
                if (user_repeat == 0 || user_repeat > expected_repeat) {
                    munmap(map_ptr, sb.st_size);
                    close(fd);
                    throw std::runtime_error(
                        "update_rows mmap string width must be 1.." + std::to_string(expected_repeat) +
                        " for " + name
                    );
                }
                repeat = expected_repeat;
            } else if (repeat != expected_repeat) {
                munmap(map_ptr, sb.st_size);
                close(fd);
                throw std::runtime_error("update_rows mmap repeat mismatch for " + name);
            }
            if (rows != num_rows) {
                munmap(map_ptr, sb.st_size);
                close(fd);
                throw std::runtime_error("update_rows mmap row count mismatch for " + name);
            }

            nb::dlpack::dtype dt = tensor.dtype();

            const uint8_t* src_u8 = static_cast<const uint8_t*>(tensor.data());
            const bool* src_bool = static_cast<const bool*>(tensor.data());
            const int16_t* src_i16 = static_cast<const int16_t*>(tensor.data());
            const int32_t* src_i32 = static_cast<const int32_t*>(tensor.data());
            const int64_t* src_i64 = static_cast<const int64_t*>(tensor.data());
            const float* src_f32 = static_cast<const float*>(tensor.data());
            const double* src_f64 = static_cast<const double*>(tensor.data());

            for (long i = 0; i < num_rows; i++) {
                uint8_t* dest_row = base_ptr + row_start_offset + i * row_width_bytes_ + col->byte_offset;
                for (long j = 0; j < repeat; j++) {
                    uint8_t* dest = dest_row + j * col->width;
                    long idx = i * repeat + j;

                    switch (col->type) {
                        case FITSColumnType::BYTE: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            *dest = src_u8[idx];
                            break;
                        }
                        case FITSColumnType::LOGICAL: {
                            bool val = false;
                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                                val = src_bool[idx];
                            } else                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
                                // Read via tensor.stride() to handle DLPack strided views.
                                long byte_offset = (ndim == 2)
                                    ? i * tensor.stride(0) + j * tensor.stride(1)
                                    : i * tensor.stride(0) + j;
                                val = src_u8[byte_offset] != 0;
                            } else {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            *dest = val ? 'T' : 'F';
                            break;
                        }
                        case FITSColumnType::SHORT: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            uint16_t v;
                            std::memcpy(&v, &src_i16[idx], sizeof(uint16_t));
                            v = bswap_16(v);
                            std::memcpy(dest, &v, sizeof(uint16_t));
                            break;
                        }
                        case FITSColumnType::INT: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            uint32_t v;
                            std::memcpy(&v, &src_i32[idx], sizeof(uint32_t));
                            v = bswap_32(v);
                            std::memcpy(dest, &v, sizeof(uint32_t));
                            break;
                        }
                        case FITSColumnType::LONG: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            uint64_t v;
                            std::memcpy(&v, &src_i64[idx], sizeof(uint64_t));
                            v = bswap_64(v);
                            std::memcpy(dest, &v, sizeof(uint64_t));
                            break;
                        }
                        case FITSColumnType::FLOAT: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            uint32_t v;
                            std::memcpy(&v, &src_f32[idx], sizeof(uint32_t));
                            v = bswap_32(v);
                            std::memcpy(dest, &v, sizeof(uint32_t));
                            break;
                        }
                        case FITSColumnType::DOUBLE: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            uint64_t v;
                            std::memcpy(&v, &src_f64[idx], sizeof(uint64_t));
                            v = bswap_64(v);
                            std::memcpy(dest, &v, sizeof(uint64_t));
                            break;
                        }
                        case FITSColumnType::BIT: {
                            // Extract a bool from a packed BIT column (MSB-first).
                            bool val = false;
                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                                val = src_bool[idx];
                            } else                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
                                // Read via tensor.stride() to handle DLPack strided views.
                                long byte_offset = (ndim == 2)
                                    ? i * tensor.stride(0) + j * tensor.stride(1)
                                    : i * tensor.stride(0) + j;
                                val = src_u8[byte_offset] != 0;
                            } else {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            // FITS BIT columns are MSB-first within each byte.
                            // The default dest stride (j * col->width) is meaningless
                            // for BIT (col->width == 1 but storage is bit-packed), so
                            // we operate directly on dest_row instead.
                            uint8_t* target_byte = dest_row + (j / 8);
                            uint8_t bit_position = static_cast<uint8_t>(7 - (j % 8));
                            if (j % 8 == 0) {
                                *target_byte = 0;
                            }
                            if (val) {
                                *target_byte |= static_cast<uint8_t>(1U << bit_position);
                            }
                            break;
                        }
                        case FITSColumnType::STRING: {
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            // STRING: dest stride is 1 byte. Pad trailing bytes with
                            // ASCII spaces when the user-provided width is shorter
                            // than the FITS column width.
                            if (j < user_repeat) {
                                // Read src via tensor.stride() so nanobind DLPack
                                // strided views (e.g. S8 over UCS-4 with stride(1)==4)
                                // land the right byte instead of sweeping NUL padding.
                                long byte_offset = (ndim == 2)
                                    ? i * tensor.stride(0) + j * tensor.stride(1)
                                    : i * tensor.stride(0) + j;
                                dest[0] = src_u8[byte_offset];
                            } else {
                                dest[0] = 0x20; // ASCII space; matches FITS CHAR convention
                            }
                            break;
                        }
                        case FITSColumnType::COMPLEX_FLOAT: {
                            // DLPack `Complex` with bits == 64 corresponds to complex64
                            // (re/im float32). Total bit width is 64 (2 * 32-bit floats).
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            const auto* src_complex32 =
                                static_cast<const c10::complex<float>*>(tensor.data());
                            int32_t re_bits = 0;
                            int32_t im_bits = 0;
                            float re = src_complex32[idx].real();
                            float im = src_complex32[idx].imag();
                            std::memcpy(&re_bits, &re, sizeof(int32_t));
                            std::memcpy(&im_bits, &im, sizeof(int32_t));
                            re_bits = bswap_32(re_bits);
                            im_bits = bswap_32(im_bits);
                            std::memcpy(dest, &re_bits, sizeof(int32_t));
                            std::memcpy(dest + sizeof(int32_t), &im_bits, sizeof(int32_t));
                            break;
                        }
                        case FITSColumnType::COMPLEX_DOUBLE: {
                            // DLPack `Complex` with bits == 128 corresponds to complex128
                            // (re/im float64). Total bit width is 128 (2 * 64-bit floats).
                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            const auto* src_complex64 =
                                static_cast<const c10::complex<double>*>(tensor.data());
                            int64_t re_bits = 0;
                            int64_t im_bits = 0;
                            double re = src_complex64[idx].real();
                            double im = src_complex64[idx].imag();
                            std::memcpy(&re_bits, &re, sizeof(int64_t));
                            std::memcpy(&im_bits, &im, sizeof(int64_t));
                            re_bits = bswap_64(re_bits);
                            im_bits = bswap_64(im_bits);
                            std::memcpy(dest, &re_bits, sizeof(int64_t));
                            std::memcpy(dest + sizeof(int64_t), &im_bits, sizeof(int64_t));
                            break;
                        }
                        default:
                            munmap(map_ptr, sb.st_size);
                            close(fd);
                            throw std::runtime_error("update_rows mmap unsupported column type");
                    }
                }
            }
        }

        msync(map_ptr, sb.st_size, MS_SYNC);
        munmap(map_ptr, sb.st_size);
        close(fd);
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

        // Determine type code for cfitsio once.
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

        // Bulk-read all VLA descriptors first to reduce per-row CFITSIO overhead.
        std::vector<long> repeats(num_rows, 0);
        std::vector<long> heap_offsets(num_rows, 0);
        int status = 0;
        fits_read_descripts(
            fptr_, col_idx + 1, start_row, num_rows, repeats.data(), heap_offsets.data(), &status
        );

        if (status != 0) {
            // Fallback to per-row descriptors for older/edge-case CFITSIO behavior.
            status = 0;
            for (long i = 0; i < num_rows; i++) {
                long row = start_row + i;
                fits_read_descript(fptr_, col_idx + 1, row, &repeats[i], &heap_offsets[i], &status);
                if (status != 0) {
                    char err_msg[81];
                    fits_get_errstatus(status, err_msg);
                    throw std::runtime_error(
                        "Failed to read VLA descriptor: " + std::string(err_msg)
                    );
                }
            }
        }

        for (long i = 0; i < num_rows; i++) {
            long repeat = repeats[i];
            if (repeat < 0) {
                repeat = 0;
            }

            std::vector<int64_t> shape;
            shape.push_back(repeat);
            torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(col.torch_type));

            if (repeat > 0) {
                long row = start_row + i;
                int anynul = 0;
                status = 0;
                fits_read_col(
                    fptr_, type_code, col_idx + 1, row, 1, repeat, nullptr, tensor.data_ptr(), &anynul, &status
                );
                if (status != 0) {
                    char err_msg[81];
                    fits_get_errstatus(status, err_msg);
                    throw std::runtime_error("Failed to read VLA data: " + std::string(err_msg));
                }
            }

            column_data.push_back(tensor);
        }

        return column_data;
    }

    // Read a VLA column as flat values + row offsets for fast Arrow ListArray construction.
    std::pair<torch::Tensor, torch::Tensor> read_vla_column_flat(
        int col_idx, long start_row, long num_rows, const ColumnInfo& col
    ) {
        std::vector<long> repeats(num_rows, 0);
        std::vector<long> heap_offsets(num_rows, 0);
        int status = 0;
        fits_read_descripts(
            fptr_, col_idx + 1, start_row, num_rows, repeats.data(), heap_offsets.data(), &status
        );
        if (status != 0) {
            status = 0;
            for (long i = 0; i < num_rows; i++) {
                long row = start_row + i;
                fits_read_descript(fptr_, col_idx + 1, row, &repeats[i], &heap_offsets[i], &status);
                if (status != 0) {
                    char err_msg[81];
                    fits_get_errstatus(status, err_msg);
                    throw std::runtime_error("Failed to read VLA descriptor: " + std::string(err_msg));
                }
            }
        }

        std::vector<int64_t> offsets(num_rows + 1, 0);
        int64_t total = 0;
        for (long i = 0; i < num_rows; i++) {
            long rep = repeats[i];
            if (rep < 0) {
                rep = 0;
            }
            total += static_cast<int64_t>(rep);
            offsets[i + 1] = total;
        }

        torch::Tensor values = torch::empty(
            {total}, torch::TensorOptions().dtype(col.torch_type)
        );
        torch::Tensor offs = torch::from_blob(
            offsets.data(),
            {static_cast<long long>(offsets.size())},
            torch::TensorOptions().dtype(torch::kInt64)
        ).clone();

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

        int64_t cursor = 0;
        for (long i = 0; i < num_rows; i++) {
            long rep = repeats[i];
            if (rep <= 0) {
                continue;
            }

            long row = start_row + i;
            int anynul = 0;
            status = 0;

            void* dst = nullptr;
            switch (values.scalar_type()) {
                case torch::kBool:
                    dst = static_cast<void*>(values.data_ptr<bool>() + cursor);
                    break;
                case torch::kUInt8:
                    dst = static_cast<void*>(values.data_ptr<uint8_t>() + cursor);
                    break;
                case torch::kInt16:
                    dst = static_cast<void*>(values.data_ptr<int16_t>() + cursor);
                    break;
                case torch::kInt32:
                    dst = static_cast<void*>(values.data_ptr<int32_t>() + cursor);
                    break;
                case torch::kInt64:
                    dst = static_cast<void*>(values.data_ptr<int64_t>() + cursor);
                    break;
                case torch::kFloat32:
                    dst = static_cast<void*>(values.data_ptr<float>() + cursor);
                    break;
                case torch::kFloat64:
                    dst = static_cast<void*>(values.data_ptr<double>() + cursor);
                    break;
                default:
                    throw std::runtime_error("Unsupported VLA scalar type");
            }

            fits_read_col(
                fptr_, type_code, col_idx + 1, row, 1, rep, nullptr, dst, &anynul, &status
            );
            if (status != 0) {
                char err_msg[81];
                fits_get_errstatus(status, err_msg);
                throw std::runtime_error("Failed to read VLA data: " + std::string(err_msg));
            }
            cursor += rep;
        }

        return std::make_pair(values, offs);
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

        // 16MB chunk target, then align with CFITSIO's suggested table row buffer.
        const size_t TARGET_CHUNK_SIZE = 16 * 1024 * 1024;
        long rows_per_chunk = std::max(1L, (long)(TARGET_CHUNK_SIZE / row_width_bytes_));
        {
            int status = 0;
            long cfitsio_rows_per_buf = 0;
            fits_get_rowsize(fptr_, &cfitsio_rows_per_buf, &status);
            if (status == 0 && cfitsio_rows_per_buf > 0) {
                if (rows_per_chunk < cfitsio_rows_per_buf) {
                    rows_per_chunk = cfitsio_rows_per_buf;
                } else {
                    rows_per_chunk =
                        std::max(1L, (rows_per_chunk / cfitsio_rows_per_buf) * cfitsio_rows_per_buf);
                }
            }
        }

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

        // Optimized loops for common types.
        // FITS binary tables are big-endian; swap on little-endian hosts only.
        const bool swap_endian = host_is_little_endian();

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
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint16_t* dest_cell = (uint16_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint16_t val;
                    std::memcpy(&val, src_cell + j * 2, 2);
                    dest_cell[j] = swap_endian ? bswap_16(val) : val;
                }
            }
        } else if (col_width == 4) {
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint32_t* dest_cell = (uint32_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint32_t val;
                    std::memcpy(&val, src_cell + j * 4, 4);
                    dest_cell[j] = swap_endian ? bswap_32(val) : val;
                }
            }
        } else if (col_width == 8) {
            for (long i = 0; i < num_rows; i++) {
                const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                uint64_t* dest_cell = (uint64_t*)(dest + i * total_width);
                for (int j = 0; j < col.repeat; j++) {
                    uint64_t val;
                    std::memcpy(&val, src_cell + j * 8, 8);
                    dest_cell[j] = swap_endian ? bswap_64(val) : val;
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
    // Move to the target table HDU when the handle is shared via the cache.
    void ensure_table_hdu() {
        if (!fptr_) return;
        int cur = 0;
        fits_get_hdu_num(fptr_, &cur);
        if (cur != target_hdu_) {
            int status = 0;
            fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
            if (status != 0) {
                throw std::runtime_error("Failed to move to table HDU");
            }
        }
    }

    fitsfile* fptr_ = nullptr;
    std::string filename_;
    int hdu_num_;
    int target_hdu_ = 1;
    bool use_cache_ = false;
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



void write_fits_table(const char* filename, nb::dict tensor_dict, nb::dict header, bool overwrite, nb::object schema_obj, const std::string& table_type) {
    torchfits::check_fits_filename_security(filename ? filename : "");
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
        bool is_ascii = false;
        std::string kind = table_type;
        for (auto& c : kind) {
            c = std::tolower(static_cast<unsigned char>(c));
        }
        if (kind == "ascii") {
            is_ascii = true;
        }
        torchfits::write_table_hdu(fptr, tensor_dict, header, schema_obj, is_ascii);
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
    }

    fits_close_file(fptr, &status);
}

long infer_num_rows_from_payload(nb::dict tensor_dict) {
    long num_rows = 0;
    if (tensor_dict.size() <= 0) {
        return 0;
    }

    nb::handle first_obj = (*tensor_dict.begin()).second;
    if (nb::isinstance<nb::list>(first_obj)) {
        nb::list lst = nb::cast<nb::list>(first_obj);
        return static_cast<long>(lst.size());
    }
    if (nb::isinstance<nb::tuple>(first_obj)) {
        nb::tuple tup = nb::cast<nb::tuple>(first_obj);
        return static_cast<long>(tup.size());
    }
    if (nb::isinstance<nb::str>(first_obj) || nb::isinstance<nb::bytes>(first_obj)) {
        return 1;
    }

    nb::ndarray<> first_col = nb::cast<nb::ndarray<>>(first_obj);
    int ndim = first_col.ndim();
    if (ndim == 0) {
        return 1;
    }
    return static_cast<long>(first_col.shape(0));
}

void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows);

void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict) {
    fitsfile* fptr;
    int status = 0;

    // Use explicit cfitsio mode value to avoid macro collisions with Python headers.
    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long num_rows = infer_num_rows_from_payload(tensor_dict);

    long start_row;
    fits_get_num_rows(fptr, &start_row, &status);
    start_row++;

    fits_insert_rows(fptr, start_row -1, num_rows, &status);

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for append_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to get column type for append_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error("append_rows VLA rows must be 1D for " + col_name);
                }
                long nelements = static_cast<long>(arr.size());
                void* data_ptr = arr.size() ? arr.data() : nullptr;
                std::vector<unsigned char> logical;

                if (base_type == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    }
                    data_ptr = logical.data();
                }

                fits_write_col(fptr, base_type, colnum, start_row + row, 1, nelements, data_ptr, &status);
            }
            continue;
        }

        if (typecode == TSTRING) {
            std::vector<std::string> values;
            nb::handle obj = item.second;
            if (nb::isinstance<nb::list>(obj)) {
                nb::list lst = nb::cast<nb::list>(obj);
                values.reserve(lst.size());
                for (auto v : lst) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::tuple>(obj)) {
                nb::tuple tup = nb::cast<nb::tuple>(obj);
                values.reserve(tup.size());
                for (auto v : tup) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
                values.push_back(nb::cast<std::string>(obj));
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            long width_chars = repeat > 0 ? repeat : 1;
            std::vector<std::string> padded;
            padded.reserve(values.size());
            for (const auto& v : values) {
                std::string s = v;
                if (static_cast<long>(s.size()) > width_chars) {
                    s = s.substr(0, static_cast<size_t>(width_chars));
                } else if (static_cast<long>(s.size()) < width_chars) {
                    s.append(static_cast<size_t>(width_chars - s.size()), ' ');
                }
                padded.push_back(std::move(s));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(padded.size());
            for (const auto& s : padded) {
                ptrs.push_back(s.c_str());
            }

            fits_write_col(fptr, TSTRING, colnum, start_row, 1, num_rows,
                           const_cast<char**>(ptrs.data()), &status);
            continue;
        }

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        int ndim = tensor.ndim();
        long rows = 1;
        long repeat_vals = 1;
        if (ndim == 0) {
            rows = 1;
            repeat_vals = 1;
        } else if (ndim == 1) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = 1;
        } else if (ndim == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = static_cast<long>(tensor.shape(1));
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("append_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("append_rows column length mismatch for " + col_name);
        }

        void* data_ptr = tensor.data();
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            long nelements = rows * repeat_vals;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(tensor.data());
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) {
            fits_type = TSHORT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) {
            fits_type = TINT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) {
            fits_type = TFLOAT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) {
            fits_type = TDOUBLE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) {
            fits_type = TLONGLONG;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64) {
            fits_type = TCOMPLEX;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128) {
            fits_type = TDBLCOMPLEX;
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Unsupported dtype for append_rows");
        }

        long nelements = num_rows * repeat_vals;
        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to append rows to FITS table");
    }
}

void insert_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row) {
    long num_rows = infer_num_rows_from_payload(tensor_dict);
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long total_rows = 0;
    fits_get_num_rows(fptr, &total_rows, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to get table row count");
    }

    if (start_row < 1 || start_row > (total_rows + 1)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("insert_rows start_row out of range");
    }

    fits_insert_rows(fptr, start_row - 1, num_rows, &status);
    fits_close_file(fptr, &status);
    if (status != 0) {
        throw std::runtime_error("Failed to insert rows into FITS table");
    }

    // Reuse the existing typed write path to populate inserted rows.
    update_rows(filename, hdu_num, tensor_dict, start_row, num_rows);
}

void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    long total_rows = 0;
    fits_get_num_rows(fptr, &total_rows, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to get table row count");
    }

    if (start_row < 1 || start_row > total_rows) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("delete_rows start_row out of range");
    }

    long max_rows = total_rows - start_row + 1;
    long ndelete = std::min(num_rows, max_rows);
    fits_delete_rows(fptr, start_row, ndelete, &status);
    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to delete rows from FITS table");
    }
}

void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for update_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to get column type for update_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error("update_rows VLA rows must be 1D for " + col_name);
                }
                long nelements = static_cast<long>(arr.size());
                void* data_ptr = arr.size() ? arr.data() : nullptr;
                std::vector<unsigned char> logical;

                if (base_type == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(arr.data());
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    }
                    data_ptr = logical.data();
                }

                fits_write_col(fptr, base_type, colnum, start_row + row, 1, nelements, data_ptr, &status);
            }
            continue;
        }

        if (typecode == TSTRING) {
            std::vector<std::string> values;
            nb::handle obj = item.second;
            if (nb::isinstance<nb::list>(obj)) {
                nb::list lst = nb::cast<nb::list>(obj);
                values.reserve(lst.size());
                for (auto v : lst) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::tuple>(obj)) {
                nb::tuple tup = nb::cast<nb::tuple>(obj);
                values.reserve(tup.size());
                for (auto v : tup) {
                    values.push_back(nb::cast<std::string>(v));
                }
            } else if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
                values.push_back(nb::cast<std::string>(obj));
            } else if (nb::isinstance<nb::ndarray<>>(obj)) {
                // Python's update_rows materialises fixed-width CHAR columns
                // as a (num_rows, width) uint8 ndarray (see the
                // has_string / dtype / string_widths branch in
                // torchfits.table.update_rows). Mirror the mmap-path's
                // STRING case: copy bytes left-to-right per row and
                // right-pad with ASCII spaces (0x20) so short user
                // payloads land the same bytes as the mmap writer.
                nb::ndarray<> t_str = nb::cast<nb::ndarray<>>(obj);
                nb::dlpack::dtype dt_str = t_str.dtype();
                if (
                    !(dt_str.code == (uint8_t)nb::dlpack::dtype_code::UInt &&
                      dt_str.bits == 8)
                ) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string ndarray must be uint8 for " + col_name
                    );
                }
                int ndim_str = t_str.ndim();
                long user_repeat_str = 1;
                long rows_str = 1;
                if (ndim_str == 0) {
                    rows_str = 1;
                    user_repeat_str = 1;
                } else if (ndim_str == 1) {
                    rows_str = static_cast<long>(t_str.shape(0));
                    user_repeat_str = 1;
                } else if (ndim_str == 2) {
                    rows_str = static_cast<long>(t_str.shape(0));
                    user_repeat_str = static_cast<long>(t_str.shape(1));
                } else {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string ndarray must be 1D/2D for " + col_name
                    );
                }
                if (rows_str != num_rows) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows column length mismatch for " + col_name
                    );
                }
                long width_chars_str = repeat > 0 ? repeat : 1;
                if (user_repeat_str > width_chars_str) {
                    fits_close_file(fptr, &status);
                    throw std::runtime_error(
                        "update_rows string width " +
                        std::to_string(user_repeat_str) + " exceeds column " +
                        std::to_string(width_chars_str) + " for " + col_name
                    );
                }
                const uint8_t* src_str = static_cast<const uint8_t*>(t_str.data());
                std::vector<std::string> padded_str;
                padded_str.reserve(static_cast<size_t>(num_rows));
                for (long i = 0; i < num_rows; ++i) {
                    std::string row(static_cast<size_t>(width_chars_str), ' ');
                    for (long j = 0; j < user_repeat_str; ++j) {
                        long byte_off_str = (ndim_str == 2)
                            ? i * t_str.stride(0) + j * t_str.stride(1)
                            : i * t_str.stride(0) + j;
                        row[static_cast<size_t>(j)] =
                            static_cast<char>(src_str[byte_off_str]);
                    }
                    padded_str.push_back(std::move(row));
                }
                std::vector<const char*> ptrs_str;
                ptrs_str.reserve(padded_str.size());
                for (const auto& s : padded_str) {
                    ptrs_str.push_back(s.c_str());
                }
                fits_write_col(
                    fptr, TSTRING, colnum, start_row, 1, num_rows,
                    const_cast<char**>(ptrs_str.data()), &status
                );
                continue;
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            long width_chars = repeat > 0 ? repeat : 1;
            std::vector<std::string> padded;
            padded.reserve(values.size());
            for (const auto& v : values) {
                std::string s = v;
                if (static_cast<long>(s.size()) > width_chars) {
                    s = s.substr(0, static_cast<size_t>(width_chars));
                } else if (static_cast<long>(s.size()) < width_chars) {
                    s.append(static_cast<size_t>(width_chars - s.size()), ' ');
                }
                padded.push_back(std::move(s));
            }
            std::vector<const char*> ptrs;
            ptrs.reserve(padded.size());
            for (const auto& s : padded) {
                ptrs.push_back(s.c_str());
            }

            fits_write_col(fptr, TSTRING, colnum, start_row, 1, num_rows,
                           const_cast<char**>(ptrs.data()), &status);
            continue;
        }

        if (typecode == TBIT) {
            // BIT columns: pack booleans MSB-first into (ceil(repeat/8))
            // bytes per row and write via fits_write_col(FT, TBIT, ...).
            // CFITSIO's TBIT descriptor expects bytes containing 8 packed
            // bits each, mirroring the mmap-path's per-byte packing so the
            // two writers stay byte-equivalent.
            nb::ndarray<> t = nb::cast<nb::ndarray<>>(item.second);
            int ndim_t = t.ndim();
            long rows_t = 1;
            long user_repeat = 1;
            if (ndim_t == 0) {
                rows_t = 1;
                user_repeat = 1;
            } else if (ndim_t == 1) {
                rows_t = static_cast<long>(t.shape(0));
                user_repeat = 1;
            } else if (ndim_t == 2) {
                rows_t = static_cast<long>(t.shape(0));
                user_repeat = static_cast<long>(t.shape(1));
            } else {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows BIT only supports 1D/2D columns for " + col_name
                );
            }
            if (rows_t != num_rows) {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows column length mismatch for " + col_name
                );
            }
            if (user_repeat <= 0 || user_repeat > repeat) {
                fits_close_file(fptr, &status);
                throw std::runtime_error(
                    "update_rows BIT repeat must be 1.." + std::to_string(repeat) +
                    " for " + col_name
                );
            }

            long packed_bytes_per_row = (repeat + 7) / 8;
            std::vector<unsigned char> packed(
                static_cast<size_t>(num_rows * packed_bytes_per_row), 0
            );

            nb::dlpack::dtype dt_b = t.dtype();
            const bool* src_bool_b = static_cast<const bool*>(t.data());
            const uint8_t* src_u8_b = static_cast<const uint8_t*>(t.data());

            for (long i = 0; i < num_rows; ++i) {
                for (long j = 0; j < user_repeat; ++j) {
                    bool val = false;
                    long byte_off = (ndim_t == 2)
                        ? i * t.stride(0) + j * t.stride(1)
                        : i * t.stride(0) + j;
                    if (
                        dt_b.code == (uint8_t)nb::dlpack::dtype_code::Bool &&
                        dt_b.bits == 8
                    ) {
                        val = src_bool_b[byte_off];
                    } else if (
                        dt_b.code == (uint8_t)nb::dlpack::dtype_code::UInt &&
                        dt_b.bits == 8
                    ) {
                        val = src_u8_b[byte_off] != 0;
                    } else {
                        fits_close_file(fptr, &status);
                        throw std::runtime_error(
                            "update_rows BIT dtype must be bool or uint8 for " +
                            col_name
                        );
                    }
                    if (val) {
                        packed[
                            static_cast<size_t>(i * packed_bytes_per_row + j / 8)
                        ] |= static_cast<unsigned char>(1U << (7 - (j % 8)));
                    }
                }
            }

            fits_write_col(
                fptr, TBIT, colnum, start_row, 1,
                num_rows * packed_bytes_per_row, packed.data(), &status
            );
            continue;
        }

        nb::ndarray<> tensor = nb::cast<nb::ndarray<>>(item.second);
        int ndim = tensor.ndim();
        long rows = 1;
        long repeat_vals = 1;
        if (ndim == 0) {
            rows = 1;
            repeat_vals = 1;
        } else if (ndim == 1) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = 1;
        } else if (ndim == 2) {
            rows = static_cast<long>(tensor.shape(0));
            repeat_vals = static_cast<long>(tensor.shape(1));
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("update_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("update_rows column length mismatch for " + col_name);
        }

        void* data_ptr = tensor.data();
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            long nelements = rows * repeat_vals;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(tensor.data());
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 16) {
            fits_type = TSHORT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 32) {
            fits_type = TINT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 32) {
            fits_type = TFLOAT;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Float && dt.bits == 64) {
            fits_type = TDOUBLE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 64) {
            fits_type = TLONGLONG;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 64) {
            fits_type = TCOMPLEX;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Complex && dt.bits == 128) {
            fits_type = TDBLCOMPLEX;
        } else {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Unsupported dtype for update_rows");
        }

        long nelements = num_rows * repeat_vals;
        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to update rows in FITS table");
    }
}

void update_rows_mmap(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows) {
    torchfits::TableReader reader(filename ? filename : "", hdu_num);
    reader.update_rows_mmap(tensor_dict, start_row, num_rows);
}

void rename_columns(const char* filename, int hdu_num, nb::dict mapping) {
    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    for (auto item : mapping) {
        std::string old_name = nb::cast<std::string>(item.first);
        std::string new_name = nb::cast<std::string>(item.second);
        if (old_name == new_name) {
            continue;
        }

        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(old_name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for rename_columns: " + old_name);
        }

        int check_status = 0;
        int existing = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(new_name.c_str()), &existing, &check_status);
        if (check_status == 0 && existing > 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Target column already exists: " + new_name);
        }

        char keyname[FLEN_KEYWORD];
        fits_make_keyn("TTYPE", colnum, keyname, &status);
        fits_update_key(fptr, TSTRING, keyname, (void*)new_name.c_str(), nullptr, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to update column name for " + old_name);
        }
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to rename FITS table columns");
    }
}

void drop_columns(const char* filename, int hdu_num, nb::list columns) {
    fitsfile* fptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    fits_open_file(&fptr, filename, kFitsReadWrite, &status);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to move to table HDU");
    }

    std::vector<int> colnums;
    colnums.reserve(static_cast<size_t>(columns.size()));
    for (auto name_obj : columns) {
        std::string name = nb::cast<std::string>(name_obj);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(name.c_str()), &colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Column not found for drop_columns: " + name);
        }
        colnums.push_back(colnum);
    }

    std::sort(colnums.begin(), colnums.end(), std::greater<int>());
    colnums.erase(std::unique(colnums.begin(), colnums.end()), colnums.end());

    for (int colnum : colnums) {
        fits_delete_col(fptr, colnum, &status);
        if (status != 0) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Failed to delete column");
        }
    }

    fits_close_file(fptr, &status);

    if (status != 0) {
        throw std::runtime_error("Failed to drop FITS table columns");
    }
}

}



namespace {
nb::dict table_result_to_python(
    const std::unordered_map<std::string, torchfits::TableReader::ColumnData>& result_map,
    bool as_numpy
) {
    nb::dict result_dict;
    for (auto& [key, col_data] : result_map) {
        if (col_data.is_vla) {
            if (as_numpy && col_data.vla_offsets.defined()) {
                result_dict[key.c_str()] = nb::make_tuple(
                    tensor_to_numpy_object(col_data.fixed_data),
                    tensor_to_numpy_object(col_data.vla_offsets)
                );
                continue;
            }
            nb::list vla_list;
            for (const auto& tensor : col_data.vla_data) {
                vla_list.append(as_numpy ? tensor_to_numpy_object(tensor) : tensor_to_python(tensor));
            }
            result_dict[key.c_str()] = vla_list;
        } else {
            result_dict[key.c_str()] = as_numpy ? tensor_to_numpy_object(col_data.fixed_data)
                                                : tensor_to_python(col_data.fixed_data);
        }
    }
    return result_dict;
}
}

// Forward declare invalidation functions (they are defined in fits.cpp/cache.cpp)
namespace torchfits {
void invalidate_cached(const std::string& filepath);
void invalidate_shared_meta(const std::string& filepath);
}

void bind_table(nb::module_& m) {
    nb::class_<torchfits::TableReader>(m, "TableReader")
        .def("__init__", [](torchfits::TableReader* self, const std::string& filename, int hdu_num) {
            new (self) torchfits::TableReader(filename, hdu_num);
        }, nb::arg("filename"), nb::arg("hdu_num") = 1)
        .def("__init__", [](torchfits::TableReader* self, nb::object file_obj, int hdu_num) {
            fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
            new (self) torchfits::TableReader(fptr, hdu_num);
        }, nb::arg("file_obj"), nb::arg("hdu_num") = 1)
        .def_prop_ro("num_rows", &torchfits::TableReader::get_num_rows)
        .def("read_rows", [](torchfits::TableReader& self,
                             const std::vector<std::string>& column_names,
                             long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            auto result_map = self.read_columns(column_names, start_row, num_rows);
            nb::gil_scoped_acquire acquire;
            return table_result_to_python(result_map, false);
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def("read_rows_numpy", [](torchfits::TableReader& self,
                                  const std::vector<std::string>& column_names,
                                  long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            auto result_map = self.read_columns(column_names, start_row, num_rows, true);
            nb::gil_scoped_acquire acquire;
            return table_result_to_python(result_map, true);
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def_prop_ro("num_rows", &torchfits::TableReader::get_num_rows)
        .def_prop_ro("num_cols", &torchfits::TableReader::get_num_cols);

    m.def("write_fits_table", [](const std::string& filename, nb::dict tensor_dict, nb::dict header, bool overwrite,
                                 nb::object schema, const std::string& table_type) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite, schema, table_type);
    }, nb::arg("filename"), nb::arg("tensor_dict"), nb::arg("header"), nb::arg("overwrite"),
       nb::arg("schema") = nb::none(), nb::arg("table_type") = "binary");

    m.def("append_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        append_rows(filename.c_str(), hdu_num, tensor_dict);
    });

    m.def("insert_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        insert_rows(filename.c_str(), hdu_num, tensor_dict, start_row);
    });

    m.def("update_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        update_rows(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });

    m.def("update_fits_table_rows_mmap", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                           long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        update_rows_mmap(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });

    m.def("rename_fits_table_columns", [](const std::string& filename, int hdu_num, nb::dict mapping) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        rename_columns(filename.c_str(), hdu_num, mapping);
    });

    m.def("drop_fits_table_columns", [](const std::string& filename, int hdu_num, nb::list columns) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        drop_columns(filename.c_str(), hdu_num, columns);
    });

    m.def("delete_fits_table_rows", [](const std::string& filename, int hdu_num, long start_row,
                                       long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        delete_rows(filename.c_str(), hdu_num, start_row, num_rows);
    });

    m.def("read_fits_table", [](const std::string& filename, int hdu_num) -> nb::object {
        nb::gil_scoped_release release;
        torchfits::TableReader reader(filename, hdu_num);
        auto result_map = reader.read_columns({}, 1, -1);
        nb::gil_scoped_acquire acquire;
        nb::dict result_dict;
        for (auto& [key, col_data] : result_map) {
            if (col_data.is_vla) {
                nb::list vla_list;
                for (const auto& tensor : col_data.vla_data) {
                    vla_list.append(tensor_to_python(tensor));
                }
                result_dict[key.c_str()] = vla_list;
            } else {
                result_dict[key.c_str()] = tensor_to_python(col_data.fixed_data);
            }
        }
        return result_dict;
    });

    m.def("read_fits_table_from_handle", [](nb::object file_obj, int hdu_num) -> nb::object {
        nb::gil_scoped_release release;
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns({}, 1, -1);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, false);
    });

    m.def("read_fits_table_rows_from_handle", [](nb::object file_obj, int hdu_num,
                                                 const std::vector<std::string>& column_names,
                                                 long start_row, long num_rows) -> nb::object {
        nb::gil_scoped_release release;
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, false);
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table", [](const std::string& filename, int hdu_num, const std::vector<std::string>& column_names, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        if (mmap) {
            torchfits::TableReader reader(filename, hdu_num);
            nb::gil_scoped_acquire acquire;
            return reader.read_columns_mmap(column_names);
        } else {
            torchfits::check_fits_filename_security(filename);
            fitsfile* fptr = nullptr;
            int status = 0;
            fits_open_file(&fptr, filename.c_str(), 0 /* READONLY */, &status);
            if (status != 0 || !fptr) {
                throw std::runtime_error("Could not open FITS file");
            }
            torchfits::TableReader reader(fptr, hdu_num);
            auto result_map = reader.read_columns(column_names);
            nb::gil_scoped_acquire acquire;
            nb::object out = nb::object(table_result_to_python(result_map, false));
            int close_status = 0;
            fits_close_file(fptr, &close_status);
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1, nb::arg("column_names") = std::vector<std::string>(), nb::arg("mmap") = false);

    m.def("read_fits_table_rows", [](const std::string& filename, int hdu_num,
                                     const std::vector<std::string>& column_names,
                                     long start_row, long num_rows, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        if (mmap) {
            torchfits::TableReader reader(filename, hdu_num);
            nb::gil_scoped_acquire acquire;
            return reader.read_columns_mmap(column_names, start_row, num_rows);
        } else {
            torchfits::check_fits_filename_security(filename);
            fitsfile* fptr = nullptr;
            int status = 0;
            fits_open_file(&fptr, filename.c_str(), 0 /* READONLY */, &status);
            if (status != 0 || !fptr) {
                throw std::runtime_error("Could not open FITS file");
            }
            torchfits::TableReader reader(fptr, hdu_num);
            auto result_map = reader.read_columns(column_names, start_row, num_rows);
            nb::gil_scoped_acquire acquire;
            nb::object out = nb::object(table_result_to_python(result_map, false));
            int close_status = 0;
            fits_close_file(fptr, &close_status);
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    m.def("read_fits_table_rows_numpy_from_handle", [](nb::object file_obj, int hdu_num,
                                                       const std::vector<std::string>& column_names,
                                                       long start_row, long num_rows) -> nb::object {
        nb::gil_scoped_release release;
        fitsfile* fptr = reinterpret_cast<fitsfile*>(torchfits::get_fptr_from_python_object(file_obj));
        torchfits::TableReader reader(fptr, hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, true);
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table_rows_numpy", [](const std::string& filename, int hdu_num,
                                           const std::vector<std::string>& column_names,
                                           long start_row, long num_rows, bool mmap) -> nb::object {
        if (mmap) {
            torchfits::TableReader reader(filename, hdu_num);
            nb::dict mapped = reader.read_columns_mmap(column_names, start_row, num_rows);
            nb::dict numpy_result;
            for (auto item : mapped) {
                nb::handle key = item.first;
                nb::handle value = item.second;
                if (PyObject_HasAttrString(value.ptr(), "numpy")) {
                    PyObject* np_obj = PyObject_CallMethod(value.ptr(), "numpy", nullptr);
                    if (!np_obj) {
                        throw nb::python_error();
                    }
                    numpy_result[key] = nb::steal(np_obj);
                } else {
                    numpy_result[key] = nb::borrow(value);
                }
            }
            return nb::object(numpy_result);
        } else {
            nb::gil_scoped_release release;
            torchfits::check_fits_filename_security(filename);
            fitsfile* fptr = nullptr;
            int status = 0;
            fits_open_file(&fptr, filename.c_str(), 0 /* READONLY */, &status);
            if (status != 0 || !fptr) {
                throw std::runtime_error("Could not open FITS file");
            }
            torchfits::TableReader reader(fptr, hdu_num);
            auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
            nb::gil_scoped_acquire acquire;
            nb::object out = table_result_to_python(result_map, true);
            int close_status = 0;
            fits_close_file(fptr, &close_status);
            return out;
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    m.def("read_fits_table_filtered", [](const std::string& filename, int hdu_num,
                                         const std::vector<std::string>& column_names,
                                         nb::list filters_py) -> nb::object {
        std::vector<torchfits::TableFilter> filters;
        for (auto handle : filters_py) {
            nb::tuple item = nb::cast<nb::tuple>(handle);
            if (item.size() != 3) throw std::runtime_error("Filter must be (col, op, val)");

            torchfits::TableFilter f;
            f.col_name = nb::cast<std::string>(item[0]);
            std::string op = nb::cast<std::string>(item[1]);

            if (op == "==" || op == "eq") f.op = torchfits::FilterOp::EQ;
            else if (op == "!=" || op == "ne") f.op = torchfits::FilterOp::NE;
            else if (op == ">" || op == "gt") f.op = torchfits::FilterOp::GT;
            else if (op == "<" || op == "lt") f.op = torchfits::FilterOp::LT;
            else if (op == ">=" || op == "ge") f.op = torchfits::FilterOp::GE;
            else if (op == "<=" || op == "le") f.op = torchfits::FilterOp::LE;
            else throw std::runtime_error("Unknown operator: " + op);

            nb::handle val = item[2];
            if (nb::isinstance<float>(val)) {
                 f.val_d = nb::cast<double>(val);
                 f.type_idx = 0;
            } else if (nb::isinstance<int>(val)) {
                 f.val_i = nb::cast<int64_t>(val);
                 f.type_idx = 1;
            } else {
                 throw std::runtime_error("Unsupported filter value type (only float/int)");
            }
            filters.push_back(f);
        }

        nb::gil_scoped_release release;
        torchfits::TableReader reader(filename, hdu_num);
        auto result_map = reader.read_columns_mmap_filtered(column_names, filters);
        nb::gil_scoped_acquire acquire;

        nb::dict result;
        for (auto& [key, val] : result_map) {
             result[key.c_str()] = tensor_to_python(val);
        }
        return result;
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("filters"));
}
