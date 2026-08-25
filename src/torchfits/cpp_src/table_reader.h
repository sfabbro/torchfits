#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <ATen/Parallel.h>
#include <functional>
#include <new>
#include <mutex>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <cassert>
#include <fitsio.h>

#include "torchfits_torch.h"
#include "torch_compat.h"
#include "hardware.h"
#include "fits_detail.h"
#include "table_types.h"
#include "security.h"

namespace nb = nanobind;

namespace torchfits {

class TableReader {
public:
    TableReader(const std::string& filename, int hdu_num = 1) : filename_(filename), hdu_num_(hdu_num), owns_fptr_(true) {
        torchfits::check_fits_filename_security(filename);
        target_hdu_ = hdu_num + 1;  // CFITSIO 1-based absolute HDU
        // Private per-instance handle (CFITSIO §4 Option A): never share one
        // fitsfile* across threads for reads.
        int status = torchfits::detail::open_fits_readonly(&fptr_, filename);
        if (status != 0 || !fptr_) {
            throw std::runtime_error("Failed to open FITS file");
        }
        // Capture the file identity for stale-reader detection: a cached
        // reader may outlive a writer on another thread replacing the file
        // (thread-local caches can only be evicted by their own thread).
        stat_valid_ = (::stat(filename_.c_str(), &file_stat_) == 0);

        // If HDU move or analysis fails, the destructor will not run (the object
        // is not fully constructed), so close the handle we opened before
        // rethrowing to avoid leaking the fitsfile*.
        try {
            status = 0;
            fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
            if (status != 0) {
                throw std::runtime_error("Failed to move to table HDU");
            }
            analyze_table();
        } catch (...) {
            int close_status = 0;
            fits_close_file(fptr_, &close_status);
            fptr_ = nullptr;
            throw;
        }
    }

    TableReader(fitsfile* fptr, int hdu_num = 1) : fptr_(fptr), hdu_num_(hdu_num), owns_fptr_(false) {
        int status = 0;
        target_hdu_ = hdu_num + 1;
        fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
        // Needed for VLA heap pread coalesce (diskfile path).
        char fname[FLEN_FILENAME] = {0};
        int name_status = 0;
        fits_file_name(fptr_, fname, &name_status);
        if (name_status == 0 && fname[0] != '\0') {
            filename_ = fname;
        }
        analyze_table();
    }

    ~TableReader() {
        if (data_fd_cached_ >= 0) {
            ::close(data_fd_cached_);
            data_fd_cached_ = -1;
        }
        if (fptr_ && owns_fptr_) {
            int status = 0;
            fits_close_file(fptr_, &status);
        }
    }

    // True if the file at filename_ still matches the identity captured at
    // construction. Path-based readers cached in the thread-local LRU call
    // this on acquire; a file replaced by another thread's writer (new inode,
    // size or mtime) makes the cached handle and pread fd stale, so the cache
    // must drop it and open fresh.
    bool file_unchanged() const {
        if (!stat_valid_) {
            return false;
        }
        struct stat st;
        if (::stat(filename_.c_str(), &st) != 0) {
            return false;
        }
        return st.st_dev == file_stat_.st_dev &&
               st.st_ino == file_stat_.st_ino &&
               st.st_size == file_stat_.st_size &&
               st.st_mtime == file_stat_.st_mtime;
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

            col.repeat = (int)repeat_long;
            col.name = std::string(ttype);
            col.width = 1;  // Will be set based on type
            col.fits_typecode = typecode;

            // Defer TSCAL/TZERO until a column is first read (ensure_column_scale).
            col.tscale = 1.0;
            col.tzero = 0.0;
            col.scaled = false;
            col.scale_resolved = false;
            col.is_unsigned_int = false;
            col.unsigned_offset = 0;

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
                    case TSBYTE: col.torch_type = torch::kInt8; break;
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
                        col.type = FITSColumnType::BYTE;
                        col.torch_type = torch::kUInt8;
                        col.width = 1;
                        break;
                    case TSBYTE:
                        // Signed bytes must stay signed: kUInt8 would turn
                        // 128..255 into -128..-1 on every read path.
                        col.type = FITSColumnType::BYTE;
                        col.torch_type = torch::kInt8;
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
            // repeat_long comes from TFORMn and must fit int: truncation would
            // corrupt cell sizes/offsets for absurd TFORM repeats.
            if (repeat_long > 0x7fffffffL) {
                throw std::runtime_error(
                    "Column repeat exceeds supported range for column " + std::string(ttype));
            }
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

    void ensure_column_scale(int col_idx) {
        auto& col = columns_[col_idx];
        if (col.scale_resolved) {
            return;
        }
        col.scale_resolved = true;
        int scale_status = 0;
        char scale_key[FLEN_KEYWORD];
        const int i = col_idx + 1;
        snprintf(scale_key, FLEN_KEYWORD, "TSCAL%d", i);
        fits_read_key(fptr_, TDOUBLE, scale_key, &col.tscale, nullptr, &scale_status);
        if (scale_status != 0) {
            scale_status = 0;
            col.tscale = 1.0;
        }
        snprintf(scale_key, FLEN_KEYWORD, "TZERO%d", i);
        fits_read_key(fptr_, TDOUBLE, scale_key, &col.tzero, nullptr, &scale_status);
        if (scale_status != 0) {
            scale_status = 0;
            col.tzero = 0.0;
        }
        col.scaled = (col.tscale != 1.0 || col.tzero != 0.0);
        const int typecode = col.fits_typecode;
        if (col.tscale == 1.0) {
            if ((typecode == TSHORT || typecode == TUSHORT) && detail::is_unsigned_short_offset(col.tzero)) {
                col.is_unsigned_int = true;
                col.unsigned_offset = 32768;
                col.unsigned_target_type = torch::kUInt16;
                col.scaled = false;
            } else if (
                // CFITSIO reports FITS `J` as TLONG/TINT32BIT (41), not TINT (31).
                (typecode == TINT || typecode == TUINT || typecode == TLONG ||
                 typecode == TINT32BIT) &&
                detail::is_unsigned_long_offset(col.tzero)) {
                col.is_unsigned_int = true;
                col.unsigned_offset = 2147483648;
                col.unsigned_target_type = torch::kUInt32;
                col.scaled = false;
            }
        }
    }

    // Read columns from the table
    // Returns column data in file order (request order when column_names is
    // given), so callers that pair columns against TFORM/TTYPE cards (e.g. the
    // table rewrite path) never see an arbitrary hash iteration order.
    std::vector<std::pair<std::string, ColumnData>> read_columns(
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
        // Reordered comparison: start_row + num_rows - 1 can overflow long for
        // adversarial inputs; nrows_ - start_row + 1 cannot (start_row in range).
        if (num_rows > nrows_ - start_row + 1) {
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
        for (int col_idx : col_indices) {
            ensure_column_scale(col_idx);
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
             return {};
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

        // Row-buffered reads via fits_read_tblbytes when selecting fixed-width
        // binary columns (≥2 cols or full row). Deinterleave beats N× fits_read_col
        // on wide projections; single-column / VLA / BIT / complex stay on fits_read_col.
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
            // Full-row OR multi-column projection (≥2 fixed cols): one
            // fits_read_tblbytes + deinterleave beats N× fits_read_col on wide tables.
            const long nsel = static_cast<long>(col_indices.size());
            use_buffered = (requested_bytes == row_width_bytes_) ||
                           (nsel >= 2 && requested_bytes > 0);
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

                         // Convert logical byte to 1/0. CFITSIO converts logical
                         // columns to the integer values 1/0 when read as TBYTE;
                         // raw bytes may also be 'T'/'F' or '1'/'0'.
                         uint8_t* data = (uint8_t*)tensor.data_ptr();
                         for (long i = 0; i < nelements; i++) {
                             data[i] = (data[i] == 'T' || data[i] == '1' || data[i] == 1) ? 1 : 0;
                         }
                    } else {
                        #ifdef DEBUG_TABLE
                        fprintf(stderr, "Reading col %d (%s), type %d, datatype %d, rows %ld\n",
                                col_idx+1, col.name.c_str(), (int)col.type, datatype, num_rows);
                        #endif
                        // Raw storage codes: apply TSCAL/TZERO in-memory below.
                        // Leaving CFITSIO auto-scale on overflows int destinations
                        // when physical values exceed the storage type range.
                        {
                            int scale_status = 0;
                            fits_set_tscale(fptr_, col_idx + 1, 1.0, 0.0, &scale_status);
                        }
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

        // Apply FITS TSCAL/TZERO in-memory. This preserves physical values
        // while keeping the read path raw and fast. FLOAT/DOUBLE columns are
        // scaled too (legal FITS; CFITSIO's fits_read_col applies TSCAL/TZERO
        // to any numeric type) — only complex/string/logical/VLA are exempt.
        for (int col_idx : col_indices) {
            const auto& col = columns_[col_idx];
            if (!col.scaled ||
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
            it->second.fixed_data = converted.to(col.unsigned_target_type);
        }

        // Order by file/request column index (see signature comment).
        std::vector<std::pair<std::string, ColumnData>> ordered;
        ordered.reserve(col_indices.size());
        for (int col_idx : col_indices) {
            auto it = result.find(columns_[col_idx].name);
            if (it != result.end()) {
                ordered.emplace_back(it->first, std::move(it->second));
            }
        }
        return ordered;

    }

    // Template helper for reading typed columns from mmap with byte-swapping.
    // OutT is the output tensor element type; RawT is the integer type used for
    // raw reads and byte-swapping (same size as OutT, verified by static_assert).
    // bswap_fn is the byte-swap function (bswap_16 / bswap_32 / bswap_64).
    template <typename OutT, typename RawT, RawT (*bswap_fn)(RawT)>
    void read_typed_mmap_column(
        const uint8_t* col_ptr,
        OutT* out,
        long num_rows,
        long repeat,
        long row_width_bytes)
    {
        constexpr size_t elem_size = sizeof(RawT);
        static_assert(sizeof(OutT) == sizeof(RawT));

        if (repeat == 1) {
            at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                for (long i = start; i < end; i++) {
                    RawT raw_val;
                    memcpy(&raw_val, col_ptr + i * row_width_bytes, elem_size);
                    RawT val = bswap_fn(raw_val);
                    memcpy(&out[i], &val, elem_size);
                }
            });
        } else {
            at::parallel_for(0, num_rows, 2048, [&](long start, long end) {
                for (long i = start; i < end; i++) {
                    OutT* row_out = out + i * repeat;
                    for (long j = 0; j < repeat; j++) {
                        RawT raw_val;
                        memcpy(&raw_val, col_ptr + i * row_width_bytes + j * elem_size, elem_size);
                        RawT val = bswap_fn(raw_val);
                        memcpy(&row_out[j], &val, elem_size);
                    }
                }
            });
        }
    }

    // Memory-mapped column reading
    // Returns a dict of column name to torch::Tensor (or numpy array for strings)
    // in file/request column order (see read_columns signature comment).
    // A truncated file must raise a clean error instead of SIGBUS on first
    // touch: header-derived extents are validated against the real size.
    void ensure_extent_within_file(off_t file_size, LONGLONG data_offset,
                                   long long rows_before, long long nrows) const {
        const LONGLONG row_bytes = static_cast<LONGLONG>(row_width_bytes_);
        const LONGLONG needed = data_offset
            + static_cast<LONGLONG>(rows_before) * row_bytes
            + static_cast<LONGLONG>(nrows) * row_bytes;
        if (static_cast<LONGLONG>(file_size) < needed) {
            throw std::runtime_error(
                "FITS table is truncated: header claims table bytes through "
                "offset " + std::to_string(static_cast<long long>(needed))
                + " but the file holds only "
                + std::to_string(static_cast<long long>(file_size))
                + "; the copy is incomplete or corrupt");
        }
    }

    std::vector<std::pair<std::string, torch::Tensor>> read_columns_mmap(
        const std::vector<std::string>& column_names = {},
        long start_row = 1, long num_rows = -1) {

        ensure_table_hdu();
        if (num_rows == -1) {
            num_rows = nrows_;
        }

        if (nrows_ == 0) {
            return {};
        }

        // Validate rows
        if (start_row < 1 || start_row > nrows_) {
            throw std::runtime_error("Invalid start row");
        }
        // Reordered comparison: start_row + num_rows - 1 can overflow long for
        // adversarial inputs; nrows_ - start_row + 1 cannot (start_row in range).
        if (num_rows > nrows_ - start_row + 1) {
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
            ensure_column_scale(col_idx);
            const auto& col = columns_[col_idx];
            if (is_ascii_) {
                // ASCII tables store numeric fields as text; the mmap layout
                // (binary cells, byte-swap) does not apply. Route to the
                // buffered path, which uses fits_read_col and handles ASCII.
                throw std::runtime_error("ASCII tables are not supported for mmap reads");
            }
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
        ensure_extent_within_file(sb.st_size, data_offset, start_row - 1, num_rows);

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

        std::unordered_map<std::string, torch::Tensor> result;
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
                    case FITSColumnType::BYTE: dtype = col.torch_type; break;  // kUInt8, or kInt8 for TSBYTE
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
                    read_typed_mmap_column<float, int32_t, bswap_32>(
                        col_ptr, tensor.data_ptr<float>(), num_rows, repeat, row_width_bytes_);
                } else if (col.type == FITSColumnType::DOUBLE) {
                    read_typed_mmap_column<double, int64_t, bswap_64>(
                        col_ptr, tensor.data_ptr<double>(), num_rows, repeat, row_width_bytes_);
                } else if (col.type == FITSColumnType::INT) {
                    read_typed_mmap_column<int32_t, int32_t, bswap_32>(
                        col_ptr, tensor.data_ptr<int32_t>(), num_rows, repeat, row_width_bytes_);
                } else if (col.type == FITSColumnType::SHORT) {
                    read_typed_mmap_column<int16_t, int16_t, bswap_16>(
                        col_ptr, tensor.data_ptr<int16_t>(), num_rows, repeat, row_width_bytes_);
                } else if (col.type == FITSColumnType::LONG) {
                    read_typed_mmap_column<int64_t, int64_t, bswap_64>(
                        col_ptr, tensor.data_ptr<int64_t>(), num_rows, repeat, row_width_bytes_);
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
                                row_out[j] = (in[j] == 'T' || in[j] == '1' || in[j] == 1);
                            }
                        }
                    });
                }

                // Apply unsigned-int coercion before storing.
                if (col.is_unsigned_int) {
                    tensor = tensor.to(torch::kInt64);
                    tensor.add_(col.unsigned_offset);
                    tensor = tensor.to(col.unsigned_target_type);
                }
                result[col.name] = std::move(tensor);

            } catch (const std::exception& e) {
                // Never swallow: a failed column must surface as an error, not
                // as a silently missing column in the result dict.
                throw std::runtime_error("Failed to mmap column " + col.name + ": " + e.what());
            }
        }

        // Order by file/request column index (see signature comment).
        std::vector<std::pair<std::string, torch::Tensor>> ordered;
        ordered.reserve(col_indices.size());
        for (int col_idx : col_indices) {
            auto it = result.find(columns_[col_idx].name);
            if (it != result.end()) {
                ordered.emplace_back(it->first, std::move(it->second));
            }
        }
        return ordered;
    }

    // Filtered reading
    std::vector<std::pair<std::string, torch::Tensor>> read_columns_mmap_filtered(
        const std::vector<std::string>& column_names,
        const std::vector<TableFilter>& filters) {

        ensure_table_hdu();
        if (filters.empty()) {
            // Need to convert read_columns_mmap result (nb::dict) to map<string, Tensor>?
            // Or just throw error -> filters shouldn't be empty here if called correctly.
            // But for robustness:
             throw std::runtime_error("Filters cannot be empty in read_columns_mmap_filtered");
        }
        if (is_ascii_) {
            // ASCII tables store numeric fields as text; the mmap layout does
            // not apply. Route to the buffered/mask paths instead.
            throw std::runtime_error("ASCII tables are not supported for mmap filtered reads");
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
        ensure_extent_within_file(sb.st_size, data_start, 0, nrows_);

        // Resolve filter columns
        struct FilterContext {
            const TableFilter* filter;
            const ColumnInfo* col_info;
            size_t offset;
            int col_idx = -1;
            // Cached types for fast switch
            bool is_float = false;
            bool is_double = false;
            bool is_int = false;
            bool is_long = false;
            bool is_short = false;
            bool is_byte = false;
            // Effective integer comparison target: the raw literal, offset by
            // TZERO for the uint16/uint32 FITS convention. Comparing against
            // this (in int64) yields the same predicate as the decoded-physical
            // value comparison without truncating the literal to the column's
            // storage width.
            int64_t target_i = 0;
            // True when target_i is representable in the column's integer
            // storage width; EQ/NE raw-byte compare is exact only then.
            bool target_fits_storage = true;
        };

        std::vector<FilterContext> ctxs;
        for (const auto& f : filters) {
            bool found = false;
            for (int ci = 0; ci < static_cast<int>(columns_.size()); ++ci) {
                const auto& c = columns_[ci];
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

                    // Resolve TSCAL/TZERO so the scaled/unsigned state is
                    // accurate, then reject what the raw-byte scan cannot do.
                    ensure_column_scale(ci);
                    if (c.scaled) {
                        throw std::runtime_error(
                            "Filtering on scaled columns is not supported for mmap filtered reads: " +
                            c.name);
                    }
                    if (!ctx.is_float && !ctx.is_double && !ctx.is_int &&
                        !ctx.is_long && !ctx.is_short && !ctx.is_byte) {
                        // LOGICAL/STRING/BIT/VLA predicates would silently match
                        // nothing — fail loudly instead of returning empty data.
                        throw std::runtime_error(
                            "Filtering on column type is not supported: " + c.name);
                    }

                    // Apply the uint16/uint32 FITS TZERO convention to the
                    // literal so the raw signed storage bytes compare against
                    // the physical value (see the scan below). EQ/NE raw-byte
                    // compare is exact only when the effective target fits the
                    // column's integer storage width; ordering comparisons are
                    // promoted to int64 and never truncate.
                    ctx.target_i = f.val_i;
                    if (c.is_unsigned_int) {
                        ctx.target_i -= c.unsigned_offset;
                    }
                    if (ctx.is_short) {
                        ctx.target_fits_storage =
                            ctx.target_i >= -32768 && ctx.target_i <= 32767;
                    } else if (ctx.is_int) {
                        ctx.target_fits_storage =
                            ctx.target_i >= -2147483648LL &&
                            ctx.target_i <= 2147483647LL;
                    }

                    ctx.col_idx = ci;
                    ctxs.push_back(ctx);
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("Filter column not found: " + f.col_name);
        }

        // Hints: sequential access pattern for kernel prefetch
#if defined(POSIX_MADV_SEQUENTIAL)
        {
            size_t byte_len = static_cast<size_t>(nrows_) * row_width_bytes_;
            posix_madvise(const_cast<uint8_t*>(data_ptr), byte_len, POSIX_MADV_SEQUENTIAL);
        }
#endif

        // Scan rows
        std::vector<long> valid_indices;

        if (ctxs.size() == 1) {
            const auto& ctx = ctxs[0];
            const size_t offset = ctx.offset;
            const FilterOp op = ctx.filter->op;

            // Pre-byte-swap the target so we can compare raw FITS bytes directly,
            // eliminating per-row bswap instructions for EQ/NE on integers.
            // Uses the offset-adjusted target (ctx.target_i), which for unsigned
            // columns is the physical literal minus TZERO — the value the raw
            // signed storage bytes actually encode.
            auto pre_swapped_target_int = [&ctx]() -> int64_t {
                int64_t t = ctx.target_i;
                // The raw FITS bytes are big-endian; pre-swap the target to match.
                // For 2-byte values, we need the 16-bit-swapped value in the low 16 bits.
                if (ctx.is_short) {
                    // FITS short is big-endian int16; pre-swap to host endianness once.
                    return (int64_t)(int16_t)bswap_16((uint16_t)(int16_t)t);
                }
                if (ctx.is_int) {
                    return (int64_t)(int32_t)bswap_32((uint32_t)(int32_t)t);
                }
                if (ctx.is_long) {
                    return (int64_t)bswap_64((uint64_t)t);
                }
                // byte/float/double — not pre-swapped (compared as decoded values below)
                return t;
            }();

            valid_indices.reserve(nrows_);

            // Scan body shared between sequential and parallel dispatch.
            auto scan_chunk = [&](long start, long end, std::vector<long>& out) {
                if (ctx.is_int) {
                    if (op == FilterOp::EQ || op == FilterOp::NE) {
                        // Raw-byte EQ/NE is exact only when the literal fits
                        // int32 storage; an out-of-range literal can never equal
                        // (EQ) / always differs from (NE) any int32 cell.
                        if (!ctx.target_fits_storage) {
                            if (op == FilterOp::NE) {
                                for (long i = start; i < end; i++) out.push_back(i);
                            }
                        } else {
                            const int32_t pre_swapped = (int32_t)pre_swapped_target_int;
                            if (op == FilterOp::EQ) {
                                for (long i = start; i < end; i++) {
                                    const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                    uint32_t raw; std::memcpy(&raw, val_ptr, 4);
                                    if (raw == (uint32_t)pre_swapped) out.push_back(i);
                                }
                            } else {
                                for (long i = start; i < end; i++) {
                                    const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                    uint32_t raw; std::memcpy(&raw, val_ptr, 4);
                                    if (raw != (uint32_t)pre_swapped) out.push_back(i);
                                }
                            }
                        }
                    } else {
                        // Ordering: promote both sides to int64 so an out-of-range
                        // literal and unsigned offsets compare exactly (no int32
                        // truncation of the filter literal).
                        const int64_t target = ctx.target_i;
                        for (long i = start; i < end; i++) {
                            const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                            uint32_t tmp; std::memcpy(&tmp, val_ptr, 4);
                            const int64_t val = (int64_t)(int32_t)bswap_32(tmp);
                            bool match = false;
                            switch (op) {
                                case FilterOp::GT: match = (val > target); break;
                                case FilterOp::LT: match = (val < target); break;
                                case FilterOp::GE: match = (val >= target); break;
                                case FilterOp::LE: match = (val <= target); break;
                                default: break;
                            }
                            if (match) out.push_back(i);
                        }
                    }
                } else if (ctx.is_short) {
                    if (op == FilterOp::EQ || op == FilterOp::NE) {
                        // See the is_int branch: out-of-range literals make EQ
                        // vacuously false and NE vacuously true.
                        if (!ctx.target_fits_storage) {
                            if (op == FilterOp::NE) {
                                for (long i = start; i < end; i++) out.push_back(i);
                            }
                        } else {
                            const int16_t pre_swapped = (int16_t)pre_swapped_target_int;
                            if (op == FilterOp::EQ) {
                                for (long i = start; i < end; i++) {
                                    const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                    uint16_t raw; std::memcpy(&raw, val_ptr, 2);
                                    if (raw == (uint16_t)pre_swapped) out.push_back(i);
                                }
                            } else {
                                for (long i = start; i < end; i++) {
                                    const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                                    uint16_t raw; std::memcpy(&raw, val_ptr, 2);
                                    if (raw != (uint16_t)pre_swapped) out.push_back(i);
                                }
                            }
                        }
                    } else {
                        const int64_t target = ctx.target_i;
                        for (long i = start; i < end; i++) {
                            const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                            uint16_t tmp; std::memcpy(&tmp, val_ptr, 2);
                            const int64_t val = (int64_t)(int16_t)bswap_16(tmp);
                            bool match = false;
                            switch (op) {
                                case FilterOp::GT: match = (val > target); break;
                                case FilterOp::LT: match = (val < target); break;
                                case FilterOp::GE: match = (val >= target); break;
                                case FilterOp::LE: match = (val <= target); break;
                                default: break;
                            }
                            if (match) out.push_back(i);
                        }
                    }
                } else if (ctx.is_long) {
                    // int64 storage always fits the literal (val_i is int64); the
                    // uint16/uint32 TZERO convention never applies to LONG.
                    const int64_t pre_swapped = pre_swapped_target_int;
                    if (op == FilterOp::EQ) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                            uint64_t raw; std::memcpy(&raw, val_ptr, 8);
                            if (raw == (uint64_t)pre_swapped) out.push_back(i);
                        }
                    } else if (op == FilterOp::NE) {
                        for (long i = start; i < end; i++) {
                            const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                            uint64_t raw; std::memcpy(&raw, val_ptr, 8);
                            if (raw != (uint64_t)pre_swapped) out.push_back(i);
                        }
                    } else {
                        const int64_t target = ctx.target_i;
                        for (long i = start; i < end; i++) {
                            const uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                            uint64_t tmp; std::memcpy(&tmp, val_ptr, 8);
                            const int64_t val = (int64_t)bswap_64(tmp);
                            bool match = false;
                            switch (op) {
                                case FilterOp::GT: match = (val > target); break;
                                case FilterOp::LT: match = (val < target); break;
                                case FilterOp::GE: match = (val >= target); break;
                                case FilterOp::LE: match = (val <= target); break;
                                default: break;
                            }
                            if (match) out.push_back(i);
                        }
                    }
                } else if (ctx.is_float) {
                    const float target = (float)ctx.filter->val_d;
                    for (long i = start; i < end; i++) {
                        uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint32_t tmp; std::memcpy(&tmp, val_ptr, 4);
                        tmp = bswap_32(tmp);
                        float val; std::memcpy(&val, &tmp, 4);
                        bool match = false;
                        switch (op) {
                            case FilterOp::EQ: match = (val == target); break;
                            case FilterOp::NE: match = (val != target); break;
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                        }
                        if (match) out.push_back(i);
                    }
                } else if (ctx.is_double) {
                    const double target = ctx.filter->val_d;
                    for (long i = start; i < end; i++) {
                        uint8_t* val_ptr = data_ptr + i * row_width_bytes_ + offset;
                        uint64_t tmp; std::memcpy(&tmp, val_ptr, 8);
                        tmp = bswap_64(tmp);
                        double val; std::memcpy(&val, &tmp, 8);
                        bool match = false;
                        switch (op) {
                            case FilterOp::EQ: match = (val == target); break;
                            case FilterOp::NE: match = (val != target); break;
                            case FilterOp::GT: match = (val > target); break;
                            case FilterOp::LT: match = (val < target); break;
                            case FilterOp::GE: match = (val >= target); break;
                            case FilterOp::LE: match = (val <= target); break;
                        }
                        if (match) out.push_back(i);
                    }
                } else {
                    // Fallback for byte type
                    for (long i = start; i < end; i++) {
                        uint8_t* row_ptr = data_ptr + i * row_width_bytes_;
                        uint8_t* val_ptr = row_ptr + ctx.offset;
                        bool match = false;
                        if (ctx.is_byte) {
                            uint8_t val = *val_ptr;
                            int64_t target = ctx.target_i;
                            switch (op) {
                                case FilterOp::EQ: match = (val == target); break;
                                case FilterOp::NE: match = (val != target); break;
                                case FilterOp::GT: match = (val > target); break;
                                case FilterOp::LT: match = (val < target); break;
                                case FilterOp::GE: match = (val >= target); break;
                                case FilterOp::LE: match = (val <= target); break;
                            }
                        }
                        if (match) out.push_back(i);
                    }
                }
            };

            // Sequential scan: parallel chunk + mutex merge + sort loses on
            // typical predicate widths vs a single pass (no N threshold).
            scan_chunk(0, nrows_, valid_indices);
        } else {
            // Multi-filter fallback (single-threaded, uncommon case)
            std::vector<long> local;
            local.reserve(nrows_);
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
                       int64_t target = ctx.target_i;
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
                       int64_t target = ctx.target_i;
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
                       int64_t target = ctx.target_i;
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
                       int64_t target = ctx.target_i;
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
                if (row_match) local.push_back(i);
            }
            valid_indices = std::move(local);
        }

        // Gather results (num_valid==0 → empty tensors, keyed schema preserved)
        std::unordered_map<std::string, torch::Tensor> result;
        long num_valid = valid_indices.size();

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

        // Resolve TSCAL/TZERO for gathered columns: the unsigned-int post-pass
        // below relies on unsigned_offset being populated, and scaled (non-
        // unsigned) columns cannot be gathered from raw storage bytes.
        for (int col_idx : out_col_indices) {
            ensure_column_scale(col_idx);
            const auto& col = columns_[col_idx];
            if (col.scaled && !col.is_unsigned_int) {
                throw std::runtime_error(
                    "Scaled columns not supported for mmap filtered reads: " + col.name);
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

            auto options = torch::TensorOptions().dtype(
                col.type == FITSColumnType::BIT ? torch::kBool : col.torch_type);
            torch::Tensor out_tensor = torch::empty(shape, options);

            if (num_valid == 0) {
                result[col.name] = out_tensor;
                continue;
            }

            int item_size = 0;
            if (col.type == FITSColumnType::DOUBLE || col.type == FITSColumnType::LONG) item_size = 8;
            else if (col.type == FITSColumnType::FLOAT || col.type == FITSColumnType::INT) item_size = 4;
            else if (col.type == FITSColumnType::SHORT) item_size = 2;
            else item_size = 1;

            // Cell size in storage bytes. BIT columns are bit-packed, so the
            // repeat (bit count) must not be used as a byte count — that would
            // over-read into adjacent columns / past EOF.
            size_t cell_size = 0;
            if (col.type == FITSColumnType::STRING) {
                cell_size = static_cast<size_t>(is_ascii_ ? col.width : col.repeat);
            } else if (col.type == FITSColumnType::BIT) {
                cell_size = (static_cast<size_t>(col.repeat) + 7) / 8;
            } else {
                cell_size = static_cast<size_t>(item_size) * static_cast<size_t>(std::max(1, col.repeat));
            }
            uint8_t* out_ptr = (uint8_t*)out_tensor.data_ptr();

            // LOGICAL: storage is ASCII 'T'/'F' — raw memcpy into a bool tensor
            // would decode both as true (both nonzero).
            if (col.type == FITSColumnType::LOGICAL) {
                bool* out_bool = static_cast<bool*>(out_tensor.data_ptr());
                at::parallel_for(0, num_valid, 2048, [&](long start, long end) {
                    for (long k = start; k < end; k++) {
                        const uint8_t* src =
                            data_ptr + valid_indices[k] * row_width_bytes_ + col.byte_offset;
                        out_bool[k] = (src[0] == 'T' || src[0] == '1' || src[0] == 1);
                    }
                });
                result[col.name] = out_tensor;
                continue;
            }

            // BIT: unpack MSB-first bits into bools (mirrors read_columns_mmap).
            if (col.type == FITSColumnType::BIT) {
                bool* out_bool = static_cast<bool*>(out_tensor.data_ptr());
                const long repeat = std::max(1L, (long)col.repeat);
                at::parallel_for(0, num_valid, 2048, [&](long start, long end) {
                    for (long k = start; k < end; k++) {
                        const uint8_t* row_in =
                            data_ptr + valid_indices[k] * row_width_bytes_ + col.byte_offset;
                        bool* row_out = out_bool + k * repeat;
                        for (long j = 0; j < repeat; j++) {
                            const uint8_t packed = row_in[j / 8];
                            row_out[j] = ((packed >> (7 - (j % 8))) & 0x1U) != 0;
                        }
                    }
                });
                result[col.name] = out_tensor;
                continue;
            }

            // Parallel gathering with contiguous block detection.
            // Each thread processes a disjoint range of valid_indices and writes
            // to output at deterministic offsets, so no synchronization needed.
            at::parallel_for(0, num_valid, 2048, [&](long start, long end) {
                long k = start;
                while (k < end) {
                    long run_start_k = k;
                    long run_start_row = valid_indices[k];
                    // Extend run while rows are consecutive, but stop at the
                    // thread-chunk boundary to avoid overlapping with another thread.
                    while (k + 1 < end && valid_indices[k+1] == valid_indices[k] + 1) {
                        k++;
                    }
                    long run_len = k - run_start_k + 1;

                    const uint8_t* src_base = data_ptr + run_start_row * row_width_bytes_ + col.byte_offset;
                    uint8_t* dst_base = out_ptr + run_start_k * cell_size;

                    if (item_size == 1 && row_width_bytes_ == cell_size) {
                        std::memcpy(dst_base, src_base, run_len * cell_size);
                    } else {
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
                    k++;
                }
            });

            // Apply unsigned integer offset for uint16/uint32 FITS convention.
            if (col.is_unsigned_int) {
                out_tensor = out_tensor.to(torch::kInt64);
                out_tensor.add_(col.unsigned_offset);
                out_tensor = out_tensor.to(col.unsigned_target_type);
            }
            result[col.name] = out_tensor;
        }

        // Order by file/request column index (see read_columns signature comment).
        std::vector<std::pair<std::string, torch::Tensor>> ordered;
        ordered.reserve(out_col_indices.size());
        for (int col_idx : out_col_indices) {
            auto it = result.find(columns_[col_idx].name);
            if (it != result.end()) {
                ordered.emplace_back(it->first, std::move(it->second));
            }
        }
        return ordered;
    }


    void update_rows_mmap(nb::dict tensor_dict, long start_row, long num_rows) {
        if (num_rows == -1) {
            num_rows = nrows_ - start_row + 1;
        }
        if (num_rows <= 0) {
            return;
        }
        ensure_table_hdu();
        if (is_ascii_) {
            // Writing binary cells into ASCII text fields would corrupt the
            // table; update_rows (fits_write_col) handles ASCII tables.
            throw std::runtime_error("ASCII tables are not supported for mmap updates");
        }
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
        ensure_extent_within_file(sb.st_size, data_offset, start_row - 1, num_rows);

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
                    // Element offset honoring DLPack strides (nanobind reports
                    // strides in elements), so non-contiguous / negative-stride
                    // views write the same values as the buffered
                    // (fits_write_col) path instead of reading out of order.
                    long idx = (ndim == 2)
                        ? i * tensor.stride(0) + j * tensor.stride(1)
                        : i * tensor.stride(0) + j;

                    switch (col->type) {
                        case FITSColumnType::BYTE: {
                            const bool is_sbyte = col->fits_typecode == TSBYTE;
                            if (is_sbyte) {
                                if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8)) {
                                    munmap(map_ptr, sb.st_size);
                                    close(fd);
                                    throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                                }
                            } else if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {
                                munmap(map_ptr, sb.st_size);
                                close(fd);
                                throw std::runtime_error("update_rows mmap dtype mismatch for " + name);
                            }
                            // Byte-for-byte identical bit pattern for signed/unsigned 8-bit.
                            *dest = src_u8[idx];
                            break;
                        }
                        case FITSColumnType::LOGICAL: {
                            bool val = false;
                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                                // Read via tensor.stride() to handle DLPack strided views.
                                long byte_offset = (ndim == 2)
                                    ? i * tensor.stride(0) + j * tensor.stride(1)
                                    : i * tensor.stride(0) + j;
                                val = src_bool[byte_offset];
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
                                // Read via tensor.stride() to handle DLPack strided views.
                                long byte_offset = (ndim == 2)
                                    ? i * tensor.stride(0) + j * tensor.stride(1)
                                    : i * tensor.stride(0) + j;
                                val = src_bool[byte_offset];
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
    // Returns a list of tensors (one per row) as views into one flat buffer.
    std::vector<torch::Tensor> read_vla_column(int col_idx, long start_row, long num_rows, const ColumnInfo& col) {
        auto flat = read_vla_column_flat(col_idx, start_row, num_rows, col);
        torch::Tensor& values = flat.first;
        torch::Tensor& offs = flat.second;
        std::vector<torch::Tensor> column_data;
        column_data.reserve(static_cast<size_t>(num_rows));
        const int64_t* op = offs.data_ptr<int64_t>();
        for (long i = 0; i < num_rows; i++) {
            column_data.push_back(values.slice(0, op[i], op[i + 1]));
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

        size_t elem_bytes = 0;
        switch (values.scalar_type()) {
            case torch::kUInt8: elem_bytes = 1; break;
            case torch::kInt16: elem_bytes = 2; break;
            case torch::kInt32:
            case torch::kFloat32: elem_bytes = 4; break;
            case torch::kInt64:
            case torch::kFloat64: elem_bytes = 8; break;
            default: break;  // BOOL/LOGICAL: CFITSIO conversion required
        }

        // Contiguous heap → one pread + endian convert (skips N× fits_read_col).
        // Default off until THEAP/offset edge cases are fully proven vs CFITSIO.
        static const bool heap_pread_enabled = []() {
            const char* env = std::getenv("TORCHFITS_VLA_HEAP_PREAD");
            if (!env || env[0] == '\0') return false;
            return (env[0] == '1' || env[0] == 'y' || env[0] == 'Y' || env[0] == 't' || env[0] == 'T');
        }();
        bool heap_contiguous =
            heap_pread_enabled && (elem_bytes > 0 && total > 0 && !filename_.empty());
        long expect_off = -1;
        long first_heap = -1;
        for (long i = 0; i < num_rows && heap_contiguous; i++) {
            if (repeats[i] <= 0) {
                continue;
            }
            if (first_heap < 0) {
                first_heap = heap_offsets[i];
                expect_off = heap_offsets[i] + repeats[i] * static_cast<long>(elem_bytes);
            } else if (heap_offsets[i] != expect_off) {
                heap_contiguous = false;
            } else {
                expect_off = heap_offsets[i] + repeats[i] * static_cast<long>(elem_bytes);
            }
        }

        if (heap_contiguous && first_heap >= 0) {
            LONGLONG headstart = 0, datastart = 0, dataend = 0;
            status = 0;
            fits_get_hduaddrll(fptr_, &headstart, &datastart, &dataend, &status);
            long theap = 0;
            int ks = 0;
            fits_read_key_lng(fptr_, "THEAP", &theap, nullptr, &ks);
            if (ks != 0 || theap <= 0) {
                theap = row_width_bytes_ * nrows_;
            }
            if (status == 0) {
                const off_t file_off =
                    static_cast<off_t>(datastart + static_cast<LONGLONG>(theap) + first_heap);
                const size_t nbytes = static_cast<size_t>(total) * elem_bytes;
                int fd = ::open(filename_.c_str(), O_RDONLY);
                if (fd >= 0) {
                    std::vector<uint8_t> raw(nbytes);
                    size_t got = 0;
                    bool ok = true;
                    while (got < nbytes) {
                        const ssize_t n = ::pread(
                            fd, raw.data() + got, nbytes - got,
                            file_off + static_cast<off_t>(got));
                        if (n <= 0) {
                            ok = false;
                            break;
                        }
                        got += static_cast<size_t>(n);
                    }
                    ::close(fd);
                    if (ok) {
                        uint8_t* dst = static_cast<uint8_t*>(values.data_ptr());
                        if (elem_bytes == 1 || !host_is_little_endian()) {
                            std::memcpy(dst, raw.data(), nbytes);
                        } else if (elem_bytes == 2) {
                            for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                                uint16_t v;
                                std::memcpy(&v, raw.data() + i * 2, 2);
                                v = bswap_16(v);
                                std::memcpy(dst + i * 2, &v, 2);
                            }
                        } else if (elem_bytes == 4) {
                            for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                                uint32_t v;
                                std::memcpy(&v, raw.data() + i * 4, 4);
                                v = bswap_32(v);
                                std::memcpy(dst + i * 4, &v, 4);
                            }
                        } else {
                            for (size_t i = 0; i < static_cast<size_t>(total); i++) {
                                uint64_t v;
                                std::memcpy(&v, raw.data() + i * 8, 8);
                                v = bswap_64(v);
                                std::memcpy(dst + i * 8, &v, 8);
                            }
                        }
                        return std::make_pair(values, offs);
                    }
                }
            }
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

    // Buffered reading implementation
    void read_columns_buffered(
        const std::vector<int>& col_indices,
        long start_row, long num_rows,
        std::unordered_map<std::string, ColumnData>& result) {

        // 16MB chunk target, then align with CFITSIO's suggested table row buffer.
        // Clamp to the requested row window so small tables do not allocate a full
        // 16 MiB scratch (visible as rss when mmap/cache are off in scorecard).
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
        rows_per_chunk = std::max(1L, std::min(rows_per_chunk, num_rows));

        // Reuse the persistent scratch buffer so its pages stay resident
        // across calls (a fresh per-call allocation would refault ~6k pages
        // per 12 MiB read, ~5-10 ms on a busy host).
        const size_t need_bytes =
            static_cast<size_t>(rows_per_chunk) * row_width_bytes_;
        if (scratch_buffer_.size() < need_bytes) {
            scratch_buffer_.resize(need_bytes);
        }
        std::vector<uint8_t>& buffer = scratch_buffer_;

        // Prefer direct pread of the table heap over fits_read_tblbytes when we
        // have a local path (same bytes, less CFITSIO overhead on cold opens).
        // The data offset is cached: fits_get_hduaddrll touches ~12 MiB of
        // anonymous memory on every call (kernel rusage shows ~2930 minor
        // faults per read), which dominates the pread cost on a busy host.
        int data_fd = -1;
        if (!filename_.empty() && !has_cfitsio_extended_filename_syntax(filename_)) {
            if (data_offset_ < 0) {
                LONGLONG headstart = 0, dataend = 0;
                int addr_status = 0;
                fits_get_hduaddrll(fptr_, &headstart, &data_offset_, &dataend, &addr_status);
                if (addr_status != 0 || data_offset_ <= 0) {
                    data_offset_ = -1;
                }
            }
            if (data_offset_ > 0) {
                if (data_fd_cached_ < 0) {
                    data_fd_cached_ = ::open(filename_.c_str(), O_RDONLY);
                }
                data_fd = data_fd_cached_;
            }
        }

        // Double-buffered chunks: issue chunk N+1's pread on a helper thread
        // while chunk N de-interleaves/swaps, overlapping the buffered path's
        // two full-duplex passes. Skipped for tiny tables where thread
        // handoff costs more than the copy.
        // Overlap only pays when a single chunk's pread is large enough that
        // hiding it beats two thread handoffs (~0.1 ms): measured regression
        // on 13 MB tables with warm page cache, gain on multi-hundred-MB
        // payloads / cold storage.
        const bool prefetch =
            data_fd >= 0 && num_rows > 1 &&
            static_cast<size_t>(num_rows) * row_width_bytes_ >= (64u << 20);
        std::vector<uint8_t> second_buffer;
        if (prefetch) {
            second_buffer.resize(
                static_cast<size_t>(rows_per_chunk) * row_width_bytes_);
        }

        long rows_done = 0;
        int cur = 0;                 // scratch holding the current chunk
        bool cur_filled = false;     // ...already prefetched by the previous pass
        // SAFETY: `cur` must only rotate while prefetch is live; otherwise
        // chunk_buf alternates onto the unsized second buffer.
        while (rows_done < num_rows) {
            const long rows = std::min(rows_per_chunk, num_rows - rows_done);
            uint8_t* chunk_buf =
                (prefetch && cur == 1) ? second_buffer.data() : buffer.data();
            assert(!prefetch || cur == 0 ||
                   second_buffer.size() >=
                       static_cast<size_t>(rows) * row_width_bytes_);

            // Join the prefetch (if any): this chunk's bytes are already in
            // place, or the read failed and we surface it exactly like the
            // serial path would.
            std::atomic<bool> pf_ok{true};
            auto pread_range = [&](uint8_t* dst, long row_index, long nrows) -> bool {
                const off_t off = static_cast<off_t>(
                    data_offset_ + static_cast<LONGLONG>(row_index) * row_width_bytes_);
                const size_t nbytes = static_cast<size_t>(nrows) * row_width_bytes_;
                size_t got = 0;
                while (got < nbytes) {
                    const ssize_t n = ::pread(data_fd, dst + got, nbytes - got,
                                              off + static_cast<off_t>(got));
                    if (n <= 0) {
                        return false;
                    }
                    got += static_cast<size_t>(n);
                }
                return true;
            };

            if (!cur_filled) {
                int status = 0;
                if (data_fd >= 0) {
                    if (!pread_range(chunk_buf, start_row - 1 + rows_done, rows)) {
                        ::close(data_fd);
                        data_fd = -1;
                        data_fd_cached_ = -1;
                        throw std::runtime_error("Failed to pread table bytes");
                    }
                } else {
                    fits_read_tblbytes(
                        fptr_, start_row + rows_done, 1,
                        rows * row_width_bytes_, chunk_buf, &status);
                }
                if (status != 0) {
                    if (data_fd >= 0) {
                        ::close(data_fd);
                        data_fd = -1;
                        data_fd_cached_ = -1;
                    }
                    char err_msg[81];
                    fits_get_errstatus(status, err_msg);
                    throw std::runtime_error(
                        "Failed to read table bytes: " + std::string(err_msg));
                }
            }

            // Prefetch the next chunk into the other scratch while the
            // current one de-interleaves below.
            const long next_rows = std::min(rows_per_chunk, num_rows - rows_done - rows);
            std::thread pf_thread;
            if (prefetch && next_rows > 0 && data_fd >= 0) {
                uint8_t* dst2 = cur == 0 ? second_buffer.data() : buffer.data();
                const long next_first = start_row - 1 + rows_done + rows;
                pf_thread = std::thread([&]() {
                    if (!pread_range(dst2, next_first, next_rows)) {
                        pf_ok.store(false);
                    }
                });
            }

            try {
                for (int col_idx : col_indices) {
                    const auto& col = columns_[col_idx];
                    torch::Tensor tensor = result[col.name].fixed_data;
                    uint8_t* dest_ptr =
                        (uint8_t*)get_tensor_data_ptr(tensor, rows_done * col.repeat);
                    extract_column_data(chunk_buf, rows, col, dest_ptr);
                }
            } catch (...) {
                // The prefetch thread reads `data_fd`; join before unwinding
                // so nothing touches a closed descriptor.
                if (pf_thread.joinable()) {
                    pf_thread.join();
                }
                throw;
            }

            if (pf_thread.joinable()) {
                pf_thread.join();
                if (!pf_ok.load()) {
                    ::close(data_fd);
                    data_fd = -1;
                    data_fd_cached_ = -1;
                    throw std::runtime_error("Failed to pread table bytes");
                }
                rows_done += rows;
                cur ^= 1;
                cur_filled = true;
                continue;
            }

            rows_done += rows;
            cur_filled = false;
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
        //
        // Each row writes to a non-overlapping region of dest
        // (dest + i * total_width), so parallelizing across rows is
        // thread-safe.  This matches the mmap path's at::parallel_for
        // with grain size 2048, bringing the buffered read path to
        // parity with the mmap path for multi-core systems.
        const bool swap_endian = host_is_little_endian();
        // Tiny windows: avoid parallel_for scheduling overhead (scorecard small-N).
        const long grain = (num_rows < 512) ? num_rows : 2048;
        auto for_rows = [&](auto&& body) {
            if (num_rows <= 0) {
                return;
            }
            if (num_rows < 512) {
                body(0L, num_rows);
            } else {
                at::parallel_for(0, num_rows, grain, body);
            }
        };

        if (col.type == FITSColumnType::LOGICAL) {
             // Convert 'T'/'F' (or '1'/'0') to bool
             bool* out = reinterpret_cast<bool*>(dest);
             const int repeat = col.repeat;
             for_rows([&](long start, long end) {
                 for (long i = start; i < end; i++) {
                     const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                     for (int j = 0; j < repeat; j++) {
                         const uint8_t v = src_cell[j];
                         out[i * repeat + j] = (v == 'T' || v == '1' || v == 1);
                     }
                 }
             });
        } else if (col.type == FITSColumnType::STRING || col.type == FITSColumnType::BYTE) {
             // No swapping needed for bytes/strings
             for_rows([&](long start, long end) {
                 for (long i = start; i < end; i++) {
                     std::memcpy(dest + i * total_width, buffer + i * row_stride + col_offset, total_width);
                 }
             });
        } else if (col_width == 2) {
            const int repeat = col.repeat;
            for_rows([&](long start, long end) {
                for (long i = start; i < end; i++) {
                    const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                    uint16_t* dest_cell = (uint16_t*)(dest + i * total_width);
                    for (int j = 0; j < repeat; j++) {
                        uint16_t val;
                        std::memcpy(&val, src_cell + j * 2, 2);
                        dest_cell[j] = swap_endian ? bswap_16(val) : val;
                    }
                }
            });
        } else if (col_width == 4) {
            const int repeat = col.repeat;
            for_rows([&](long start, long end) {
                for (long i = start; i < end; i++) {
                    const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                    uint32_t* dest_cell = (uint32_t*)(dest + i * total_width);
                    for (int j = 0; j < repeat; j++) {
                        uint32_t val;
                        std::memcpy(&val, src_cell + j * 4, 4);
                        dest_cell[j] = swap_endian ? bswap_32(val) : val;
                    }
                }
            });
        } else if (col_width == 8) {
            const int repeat = col.repeat;
            for_rows([&](long start, long end) {
                for (long i = start; i < end; i++) {
                    const uint8_t* src_cell = buffer + i * row_stride + col_offset;
                    uint64_t* dest_cell = (uint64_t*)(dest + i * total_width);
                    for (int j = 0; j < repeat; j++) {
                        uint64_t val;
                        std::memcpy(&val, src_cell + j * 8, 8);
                        dest_cell[j] = swap_endian ? bswap_64(val) : val;
                    }
                }
            });
        } else {
            // Fallback memcpy (should not happen for standard types needing swap)
             for_rows([&](long start, long end) {
                 for (long i = start; i < end; i++) {
                     std::memcpy(dest + i * total_width, buffer + i * row_stride + col_offset, total_width);
                 }
             });
        }
    }

    long get_num_rows() const { return nrows_; }
    int get_num_cols() const { return ncols_; }
    bool is_ascii_table() const { return is_ascii_; }

    // Serializes read/update calls when one reader instance is shared across
    // Python threads (persistent open_fits_mmap_reader capsule, exposed
    // TableReader objects). CFITSIO cursor state and the scratch buffer are
    // not safe for unsynchronized concurrent use.
    mutable std::mutex io_mutex_;

    // Total in-row bytes for the requested columns (fixed-width columns only;
    // returns -1 when any requested column is variable-width, which makes the
    // parallel fan-out gate reject).
    long long projected_bytes(const std::vector<std::string>& names) const {
        long long total = 0;
        for (const auto& n : names) {
            bool found = false;
            for (const auto& c : columns_) {
                if (c.name == n) {
                    if (c.type == FITSColumnType::VARIABLE) {
                        return -1;
                    }
                    total += static_cast<long long>(c.width) *
                             static_cast<long long>(std::max(1, c.repeat));
                    found = true;
                    break;
                }
            }
            if (!found) {
                return -1;
            }
        }
        return total;
    }

    // True when every requested column resolves to a fixed numeric type that
    // fits_read_col can write directly into a tensor (no strings, logical
    // conversion, BIT packing, complex pairing, or VLA heaps).
    bool all_fixed_numeric(const std::vector<std::string>& names) const {
        for (const auto& n : names) {
            bool found = false;
            for (const auto& c : columns_) {
                if (c.name == n) {
                    switch (c.type) {
                        case FITSColumnType::BYTE:
                        case FITSColumnType::SHORT:
                        case FITSColumnType::INT:
                        case FITSColumnType::LONG:
                        case FITSColumnType::FLOAT:
                        case FITSColumnType::DOUBLE:
                            found = true;
                            break;
                        default:
                            return false;
                    }
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

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
    bool owns_fptr_ = false;
    long nrows_;
    int ncols_;
    long row_width_bytes_ = 0;
    std::vector<ColumnInfo> columns_;
    bool is_ascii_ = false;
    // Reused row-buffer for buffered table reads. A fresh per-call
    // std::vector would be mmap'd and munmap'd by glibc (large sizes),
    // refaulting every page on every read (~6k minor faults/12 MiB); keeping
    // it on the reader keeps the pages resident across calls.
    std::vector<uint8_t> scratch_buffer_;
    // Cached byte offset of the table data section (fits_get_hduaddrll
    // refaults ~12 MiB of anonymous memory per call on this host).
    LONGLONG data_offset_ = -1;
    // Persistent raw fd for pread of the table heap (avoids re-opening the
    // file on every buffered read).
    int data_fd_cached_ = -1;
    // File identity captured at construction (stat of filename_) for the
    // stale-reader check in file_unchanged().
    struct stat file_stat_ {};
    bool stat_valid_ = false;
};

} // namespace torchfits
