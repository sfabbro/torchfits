#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#undef READONLY
#include <fitsio.h>

#include "torchfits_torch.h"
#include "cache.h"
#include "table_types.h"
#include "table_reader.h"
#include "security.h"
#include "fits_rw.h"

namespace nb = nanobind;

namespace {

bool ndarray_is_c_contiguous(const nb::ndarray<>& t) {
    // Signed math: a negative stride must compare unequal to a positive
    // expected stride, never wrap to a huge size_t that looks contiguous.
    std::ptrdiff_t expect = 1;
    for (size_t d = t.ndim(); d-- > 0;) {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(t.shape(d));
        if (n <= 1) {
            continue;
        }
        if (static_cast<std::ptrdiff_t>(t.stride(d)) != expect) {
            return false;
        }
        expect *= n;
    }
    return true;
}

void* ensure_c_contiguous_ndarray(
    nb::ndarray<>& t, long nelements, std::vector<uint8_t>& buf
) {
    const size_t item = (static_cast<size_t>(t.dtype().bits) + 7) / 8;
    if (ndarray_is_c_contiguous(t)) {
        return t.data();
    }
    buf.resize(static_cast<size_t>(nelements) * item);
    auto* dst = buf.data();
    const auto* base = static_cast<const uint8_t*>(t.data());
    // Strides are signed byte offsets: negative strides address earlier bytes,
    // so use ptrdiff_t (never size_t) to avoid reading out of bounds.
    if (t.ndim() == 1) {
        const std::ptrdiff_t s0 =
            static_cast<std::ptrdiff_t>(t.stride(0)) * static_cast<std::ptrdiff_t>(item);
        for (long i = 0; i < nelements; ++i) {
            std::memcpy(
                dst + static_cast<size_t>(i) * item,
                base + static_cast<std::ptrdiff_t>(i) * s0,
                item
            );
        }
        return dst;
    }
    if (t.ndim() == 2) {
        const size_t n0 = static_cast<size_t>(t.shape(0));
        const size_t n1 = static_cast<size_t>(t.shape(1));
        const std::ptrdiff_t s0 =
            static_cast<std::ptrdiff_t>(t.stride(0)) * static_cast<std::ptrdiff_t>(item);
        const std::ptrdiff_t s1 =
            static_cast<std::ptrdiff_t>(t.stride(1)) * static_cast<std::ptrdiff_t>(item);
        size_t out = 0;
        for (size_t i0 = 0; i0 < n0; ++i0) {
            for (size_t i1 = 0; i1 < n1; ++i1) {
                std::memcpy(
                    dst + out * item,
                    base + static_cast<std::ptrdiff_t>(i0) * s0
                         + static_cast<std::ptrdiff_t>(i1) * s1,
                    item
                );
                ++out;
            }
        }
        return dst;
    }
    throw std::runtime_error(
        "non-contiguous table column with ndim>2; call contiguous() before write"
    );
}

void rollback_inserted_rows(fitsfile* fptr, long start_row, long num_rows) {
    int st = 0;
    fits_delete_rows(fptr, start_row, num_rows, &st);
    fits_close_file(fptr, &st);
}

}  // namespace



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

// forward decl: used by insert_rows below
void populate_rows(fitsfile* fptr, nb::dict tensor_dict, long start_row, long num_rows);
void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows);
void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows);

void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict) {
    fitsfile* fptr;
    int status = 0;

    // Use explicit cfitsio mode value to avoid macro collisions with Python headers.
    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    status = torchfits::open_fits_for_write(&fptr, filename);
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
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to insert rows for append_rows");
    }

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error("Column not found for append_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error("Failed to get column type for append_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error("append_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
                    rollback_inserted_rows(fptr, start_row, num_rows);
                    throw std::runtime_error("append_rows VLA rows must be 1D for " + col_name);
                }
                long nelements = static_cast<long>(arr.size());
                std::vector<uint8_t> contig_buf;
                void* data_ptr = nelements
                    ? ensure_c_contiguous_ndarray(arr, nelements, contig_buf)
                    : nullptr;
                std::vector<unsigned char> logical;

                if (base_type == TLOGICAL && nelements > 0) {
                    nb::dlpack::dtype dt = arr.dtype();
                    logical.resize(static_cast<size_t>(nelements));
                    if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
                        const bool* src = static_cast<const bool*>(data_ptr);
                        for (long idx = 0; idx < nelements; ++idx) {
                            logical[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
                        }
                    } else {
                        const uint8_t* src = static_cast<const uint8_t*>(data_ptr);
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
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error("append_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error("append_rows column length mismatch for " + col_name);
            }

            // ASCII tables report repeat=1 with the field width in `width`
            // (binary tables report repeat=string-length, width=1), so a
            // `repeat > 0 ? repeat : 1` fallback truncates ASCII strings to
            // a single character.
            long width_chars = repeat > 1 ? repeat : width;
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
            // BIT columns: fits_write_col(TBIT, ...) routes to CFITSIO's
            // ffpclx, which expects ONE logical (0/1) value PER BIT — not
            // pre-packed bytes. Build a per-bit buffer of length
            // num_rows * repeat and pass that many bits as nelem. Without this
            // branch a BIT payload falls through to the generic ndarray path,
            // maps uint8 -> TBYTE and writes num_rows * repeat bytes, overflowing
            // into the following rows.
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
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error(
                    "append_rows BIT only supports 1D/2D columns for " + col_name
                );
            }
            if (rows_t != num_rows) {
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error(
                    "append_rows column length mismatch for " + col_name
                );
            }
            if (user_repeat <= 0 || user_repeat > repeat) {
                rollback_inserted_rows(fptr, start_row, num_rows);
                throw std::runtime_error(
                    "append_rows BIT repeat must be 1.." + std::to_string(repeat) +
                    " for " + col_name
                );
            }

            std::vector<unsigned char> bits(
                static_cast<size_t>(num_rows * repeat), 0
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
                        rollback_inserted_rows(fptr, start_row, num_rows);
                        throw std::runtime_error(
                            "append_rows BIT dtype must be bool or uint8 for " +
                            col_name
                        );
                    }
                    bits[static_cast<size_t>(i * repeat + j)] = val ? 1 : 0;
                }
            }

            fits_write_col(
                fptr, TBIT, colnum, start_row, 1,
                num_rows * repeat, bits.data(), &status
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
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error("append_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error("append_rows column length mismatch for " + col_name);
        }

        // The 2D payload width must match the column repeat: writing a
        // different element count per row interleaves cells and corrupts
        // every following row.
        if (repeat > 0 && repeat_vals != repeat) {
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error(
                "append_rows repeat mismatch for " + col_name + ": column repeat=" +
                std::to_string(repeat) + " payload width=" + std::to_string(repeat_vals));
        }

        long nelements = num_rows * repeat_vals;
        std::vector<uint8_t> contig_buf;
        void* data_ptr = ensure_c_contiguous_ndarray(tensor, nelements, contig_buf);
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(data_ptr);
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) {
            // FITS TBYTE storage is signed bytes (bit-identical with unsigned
            // interpretation), so Int8 payloads map to TBYTE.
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
            rollback_inserted_rows(fptr, start_row, num_rows);
            throw std::runtime_error("Unsupported dtype for append_rows");
        }

        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    if (status != 0) {
        rollback_inserted_rows(fptr, start_row, num_rows);
        throw std::runtime_error("Failed to append rows to FITS table");
    }
    fits_close_file(fptr, &status);
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
    status = torchfits::open_fits_for_write(&fptr, filename);
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
    if (status != 0) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Failed to insert rows into FITS table");
    }

    // Populate the inserted rows on the same open handle (single CFITSIO
    // open for insert+populate). Best-effort rollback if column writes fail
    // after the structural insert.
    try {
        populate_rows(fptr, tensor_dict, start_row, num_rows);
        fits_close_file(fptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to insert rows into FITS table");
        }
    } catch (...) {
        fits_close_file(fptr, &status);
        try {
            delete_rows(filename, hdu_num, start_row, num_rows);
        } catch (...) {
            // NOTE: CFITSIO status may already be poisoned; prefer original error
        }
        throw;
    }
}

void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    status = torchfits::open_fits_for_write(&fptr, filename);
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

void populate_rows(fitsfile* fptr, nb::dict tensor_dict, long start_row, long num_rows) {
    // Write every column of tensor_dict into rows [start_row, start_row+num_rows)
    // of an already-open table HDU. The caller owns the handle and its
    // open/close; errors propagate as exceptions (the caller closes).
    int status = 0;

    for (auto item : tensor_dict) {
        std::string col_name = nb::cast<std::string>(item.first);
        int colnum = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &colnum, &status);
        if (status != 0) {
            throw std::runtime_error("Column not found for update_rows: " + col_name);
        }

        int col_status = 0;
        int typecode = 0;
        long repeat = 0;
        long width = 0;
        fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &col_status);
        if (col_status != 0) {
            throw std::runtime_error("Failed to get column type for update_rows: " + col_name);
        }

        if (typecode < 0) {
            int base_type = -typecode;
            nb::handle obj = item.second;
            if (!(nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj))) {
                throw std::runtime_error("update_rows VLA column expects list/tuple for " + col_name);
            }

            nb::sequence seq = nb::cast<nb::sequence>(obj);
            long seq_len = static_cast<long>(nb::len(seq));
            if (seq_len != num_rows) {
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            for (long row = 0; row < num_rows; ++row) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(seq[row]);
                if (arr.ndim() > 1) {
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
                    throw std::runtime_error(
                        "update_rows string ndarray must be 1D/2D for " + col_name
                    );
                }
                if (rows_str != num_rows) {
                    throw std::runtime_error(
                        "update_rows column length mismatch for " + col_name
                    );
                }
                long width_chars_str = repeat > 1 ? repeat : width;
                if (user_repeat_str > width_chars_str) {
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
                throw std::runtime_error("update_rows string column expects list/tuple/str for " + col_name);
            }

            if (static_cast<long>(values.size()) != num_rows) {
                throw std::runtime_error("update_rows column length mismatch for " + col_name);
            }

            // ASCII tables report repeat=1 with the field width in `width`
            // (binary tables report repeat=string-length, width=1), so a
            // `repeat > 0 ? repeat : 1` fallback truncates ASCII strings to
            // a single character.
            long width_chars = repeat > 1 ? repeat : width;
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
            // BIT columns: fits_write_col(TBIT, ...) routes to CFITSIO's
            // ffpclx, which expects ONE logical (0/1) value PER BIT — not
            // pre-packed bytes. Build a per-bit buffer of length
            // num_rows * repeat and pass that many bits as nelem.
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
                throw std::runtime_error(
                    "update_rows BIT only supports 1D/2D columns for " + col_name
                );
            }
            if (rows_t != num_rows) {
                throw std::runtime_error(
                    "update_rows column length mismatch for " + col_name
                );
            }
            if (user_repeat <= 0 || user_repeat > repeat) {
                throw std::runtime_error(
                    "update_rows BIT repeat must be 1.." + std::to_string(repeat) +
                    " for " + col_name
                );
            }

            std::vector<unsigned char> bits(
                static_cast<size_t>(num_rows * repeat), 0
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
                        throw std::runtime_error(
                            "update_rows BIT dtype must be bool or uint8 for " +
                            col_name
                        );
                    }
                    bits[static_cast<size_t>(i * repeat + j)] = val ? 1 : 0;
                }
            }

            fits_write_col(
                fptr, TBIT, colnum, start_row, 1,
                num_rows * repeat, bits.data(), &status
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
            throw std::runtime_error("update_rows only supports 1D/2D columns for " + col_name);
        }

        if (rows != num_rows) {
            throw std::runtime_error("update_rows column length mismatch for " + col_name);
        }

        // The 2D payload width must match the column repeat: writing a
        // different element count per row interleaves cells and corrupts
        // every following row.
        if (repeat > 0 && repeat_vals != repeat) {
            throw std::runtime_error(
                "update_rows repeat mismatch for " + col_name + ": column repeat=" +
                std::to_string(repeat) + " payload width=" + std::to_string(repeat_vals));
        }

        long nelements = num_rows * repeat_vals;
        std::vector<uint8_t> contig_buf;
        void* data_ptr = ensure_c_contiguous_ndarray(tensor, nelements, contig_buf);
        int fits_type = 0;
        std::vector<unsigned char> logical_buffer;

        nb::dlpack::dtype dt = tensor.dtype();
        if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {
            fits_type = TLOGICAL;
            logical_buffer.resize(static_cast<size_t>(nelements));
            const bool* src = static_cast<const bool*>(data_ptr);
            for (long idx = 0; idx < nelements; ++idx) {
                logical_buffer[static_cast<size_t>(idx)] = src[idx] ? 1 : 0;
            }
            data_ptr = logical_buffer.data();
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {
            fits_type = TBYTE;
        } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::Int && dt.bits == 8) {
            // FITS TBYTE storage is signed bytes (bit-identical with unsigned
            // interpretation), so Int8 payloads map to TBYTE.
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
            throw std::runtime_error("Unsupported dtype for update_rows");
        }

        fits_write_col(fptr, fits_type, colnum, start_row, 1, nelements, data_ptr, &status);
    }

    if (status != 0) {
        throw std::runtime_error("Failed to update rows in FITS table");
    }
}

void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row, long num_rows) {
    if (num_rows <= 0) {
        return;
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    constexpr int kFitsReadWrite = 1;
    torchfits::check_fits_filename_security(filename ? filename : "");
    status = torchfits::open_fits_for_write(&fptr, filename);
    if (status != 0) {
        char err_msg[FLEN_STATUS];
        fits_get_errstatus(status, err_msg);
        throw std::runtime_error(
            std::string("Failed to open FITS file for writing: ") + err_msg
        );
    }

    try {
        fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
        if (status != 0) {
            throw std::runtime_error("Failed to move to table HDU");
        }
        populate_rows(fptr, tensor_dict, start_row, num_rows);
    } catch (...) {
        fits_close_file(fptr, &status);
        throw;
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
    status = torchfits::open_fits_for_write(&fptr, filename);
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
    status = torchfits::open_fits_for_write(&fptr, filename);
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
