#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "remote.h"
#include <vector>
#include "cfitsio_enhanced.h"
#include "real_cache.h"
#include "debug.h" // retained for other translation units; unused here after cleanup
#include <sstream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <future>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <limits>

namespace py = pybind11;

// Track last read info for diagnostics/benchmarks (thread-local)
static thread_local struct {
    std::string filename;
    int hdu = 0;
    bool used_cache = false;
    bool used_mmap = false;
    bool used_buffered = false;
    bool used_parallel_columns = false;
    std::string path_used; // "cache" | "mmap" | "buffered" | "standard"
} g_last_read_info;

py::dict get_last_read_info() {
    py::dict d;
    d["filename"] = g_last_read_info.filename;
    d["hdu"] = g_last_read_info.hdu;
    d["used_cache"] = g_last_read_info.used_cache;
    d["used_mmap"] = g_last_read_info.used_mmap;
    d["used_buffered"] = g_last_read_info.used_buffered;
    d["used_parallel_columns"] = g_last_read_info.used_parallel_columns;
    d["path_used"] = g_last_read_info.path_used;
    return d;
}

// --- Templated Image Data Reading ---

template <typename T, int CfitsioType>
torch::Tensor read_image_data_typed(fitsfile* fptr, torch::Device device,
                                  const std::vector<long>& start, const std::vector<long>& shape) {
    // (debug scope removed)
    int status = 0;
    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) throw_fits_error(status, "Error getting image dimension");

    // Handle empty HDU (NAXIS=0) case - no image data
    if (naxis == 0) {
    // Empty HDU (NAXIS=0) -> return empty tensor
        torch::TensorOptions options = torch::TensorOptions().device(device);
        if      (std::is_same<T, uint8_t>::value) options = options.dtype(torch::kUInt8);
        else if (std::is_same<T, int16_t>::value) options = options.dtype(torch::kInt16);
        else if (std::is_same<T, int32_t>::value) options = options.dtype(torch::kInt32);
        else if (std::is_same<T, int64_t>::value) options = options.dtype(torch::kInt64);
        else if (std::is_same<T, float>::value)   options = options.dtype(torch::kFloat32);
        else if (std::is_same<T, double>::value)  options = options.dtype(torch::kFloat64);
        
        return torch::empty({0}, options);  // Return empty 1D tensor with 0 elements
    }

    std::vector<long> naxes(naxis);
    fits_get_img_size(fptr, naxis, naxes.data(), &status);
    if (status) throw_fits_error(status, "Error getting image size");

    torch::TensorOptions options = torch::TensorOptions().device(device);
    if      (std::is_same<T, uint8_t>::value) options = options.dtype(torch::kUInt8);
    else if (std::is_same<T, int16_t>::value) options = options.dtype(torch::kInt16);
    else if (std::is_same<T, int32_t>::value) options = options.dtype(torch::kInt32);
    else if (std::is_same<T, int64_t>::value) options = options.dtype(torch::kInt64);
    else if (std::is_same<T, float>::value)   options = options.dtype(torch::kFloat32);
    else if (std::is_same<T, double>::value)  options = options.dtype(torch::kFloat64);

    if (!start.empty()) {
        // Reading a subset
        std::vector<long> fpixel(naxis), lpixel(naxis), inc(naxis, 1);
        std::vector<int64_t> cutout_dims;

        for (int i = 0; i < naxis; ++i) {
            int fits_idx = naxis - 1 - i;
            fpixel[fits_idx] = start[i] + 1;
            long dim_size = (shape[i] == -1) ? (naxes[fits_idx] - start[i]) : shape[i];
            lpixel[fits_idx] = start[i] + dim_size;
            cutout_dims.push_back(dim_size);
        }

        torch::Tensor data = torch::empty(cutout_dims, options);
        fits_read_subset(fptr, CfitsioType, fpixel.data(), lpixel.data(), inc.data(),
                         nullptr, data.data_ptr<T>(), nullptr, &status);
        if (status) throw_fits_error(status, "Error reading data subset");
        return data;
    } else {
        // Reading the full image - CFITSIO API OPTIMIZATION
        std::vector<int64_t> torch_dims;
        for(long val : naxes) {
            torch_dims.push_back(val);
        }
        std::reverse(torch_dims.begin(), torch_dims.end());

        torch::Tensor data = torch::empty(torch_dims, options);
        long n_elements = data.numel();
        
        // OPTIMIZATION: Use fits_read_pix for better performance on full images
        // fits_read_pix is often faster than fits_read_img for full image reads
        long fpixel[MAX_COMPRESS_DIM] = {1,1,1,1,1,1};  // Start at pixel 1,1,1... in FITS convention
        
        fits_read_pix(fptr, CfitsioType, fpixel, n_elements, nullptr,
                      data.data_ptr<T>(), nullptr, &status);
        if (status) throw_fits_error(status, "Error reading full image with fits_read_pix");
        
        return data;
    }
}

torch::Tensor read_image_data(fitsfile* fptr, torch::Device device,
                           const std::vector<long>& start, const std::vector<long>& shape) {
    int status = 0;
    int bitpix;
    fits_get_img_type(fptr, &bitpix, &status);
    if (status) throw_fits_error(status, "Error getting image type");

    switch (bitpix) {
        case BYTE_IMG:     return read_image_data_typed<uint8_t, TBYTE>(fptr, device, start, shape);
        case SHORT_IMG:    return read_image_data_typed<int16_t, TSHORT>(fptr, device, start, shape);
        case LONG_IMG:     return read_image_data_typed<int32_t, TINT>(fptr, device, start, shape);
        case LONGLONG_IMG: return read_image_data_typed<int64_t, TLONGLONG>(fptr, device, start, shape);
        case FLOAT_IMG:    return read_image_data_typed<float, TFLOAT>(fptr, device, start, shape);
        case DOUBLE_IMG:   return read_image_data_typed<double, TDOUBLE>(fptr, device, start, shape);
        default: throw std::runtime_error("Unsupported FITS image data type: " + std::to_string(bitpix));
    }
}


// --- Table Data Reading ---

// Helper to get PyTorch dtype from CFITSIO type code
torch::Dtype get_torch_dtype(int typecode) {
    switch (typecode) {
        case TBYTE:     return torch::kUInt8;
        case 12:        return torch::kInt8;     // TSBYTE - 8-bit signed byte
        case 20:        return torch::kUInt16;   // TUSHORT - 16-bit unsigned integer  
        case TSHORT:    return torch::kInt16;
        case 30:        return torch::kUInt32;   // TUINT - 32-bit unsigned integer
        case TINT:      return torch::kInt32;
    case 40:
#ifdef __APPLE__
    return torch::kInt64;    // TULONG maps to 64-bit on macOS (LP64)
#else
    return torch::kInt32;    // TULONG - 32-bit elsewhere
#endif
    case 41:
#ifdef __APPLE__
        return torch::kInt64;    // TLONG - 64-bit signed long on macOS
#else
        return torch::kInt32;    // TLONG - 32-bit signed long elsewhere
#endif
        case TLONGLONG: return torch::kInt64;
        case TFLOAT:    return torch::kFloat32;
        case TDOUBLE:   return torch::kFloat64;
        case TLOGICAL:  return torch::kBool;
        default:        return torch::kFloat64; // Default for unsupported
    }
}

py::dict read_table_data(fitsfile* fptr, torch::Device device,
                         const py::object& columns_obj,
                         long start_row, const py::object& num_rows_obj) {
    // Simplified, production table reader (row-wise safe path retained)
    int status = 0;
    long total_rows;
    int total_cols;

    fits_get_num_rows(fptr, &total_rows, &status);
    if (status) throw_fits_error(status, "Error getting number of rows");

    fits_get_num_cols(fptr, &total_cols, &status);
    if (status) throw_fits_error(status, "Error getting number of columns");

    long rows_to_read = total_rows - start_row;
    if (!num_rows_obj.is_none()) {
        rows_to_read = std::min(rows_to_read, num_rows_obj.cast<long>());
    }
    
    if (rows_to_read <= 0) {
        return py::dict();
    }

    // Determine columns to read
    std::vector<std::string> selected_columns;
    if (!columns_obj.is_none()) {
        selected_columns = columns_obj.cast<std::vector<std::string>>();
    } else {
        // Get all column names by reading TTYPEi keywords (safer than deprecated helpers)
        selected_columns.reserve(total_cols);
        for (int i = 1; i <= total_cols; ++i) {
            char keyname[FLEN_KEYWORD];
            status = 0;
            if (fits_make_keyn("TTYPE", i, keyname, &status)) {
                throw_fits_error(status, "Error forming TTYPE keyword for column " + std::to_string(i));
            }
            char colname[FLEN_VALUE];
            char comment[FLEN_COMMENT];
            status = 0;
            if (fits_read_key(fptr, TSTRING, keyname, colname, comment, &status)) {
                throw_fits_error(status, "Error reading " + std::string(keyname) + " for column " + std::to_string(i));
            }
            // colname may be quoted; strip surrounding quotes if present
            std::string colname_str(colname);
            if (!colname_str.empty() && colname_str.front() == '\'' && colname_str.back() == '\'' && colname_str.size() >= 2) {
                colname_str = colname_str.substr(1, colname_str.size() - 2);
            }
            selected_columns.emplace_back(std::move(colname_str));
        }
    }

    // NOTE: Potential future parallel/optimized bulk paths removed pending safe re-introduction.
    
    // Separate string and numeric columns for optimized processing
    struct ColumnInfo {
        std::string name;
        int number;
        int typecode;
        long repeat;
        long width;
        bool is_string;
        bool is_varlen;
        std::string tform;
        int base_typecode; // For VLA, decoded from TFORM (e.g., 'D' -> TDOUBLE)
    };
    
    std::vector<ColumnInfo> column_info;
    column_info.reserve(selected_columns.size());
    
    for (const auto& col_name : selected_columns) {
        ColumnInfo info;
        info.name = col_name;
        
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &info.number, &status);
        if (status) throw_fits_error(status, "Error finding column: " + col_name);
        
        fits_get_coltype(fptr, info.number, &info.typecode, &info.repeat, &info.width, &status);
        if (status) throw_fits_error(status, "Error getting column type for: " + col_name);
        
        info.is_string = (info.typecode == TSTRING);
        // Initialize variable-length detection: negative typecode indicates VLA
        info.is_varlen = (info.typecode < 0);
        info.tform.clear();
        info.base_typecode = info.is_varlen ? -info.typecode : info.typecode;
        if (info.typecode < 0) {
            info.is_varlen = true;
            info.base_typecode = -info.typecode; // CFITSIO encodes VLA base type as negative
        }
        {
            char keyname[FLEN_KEYWORD];
            int st2 = 0;
            if (fits_make_keyn("TFORM", info.number, keyname, &st2) == 0) {
                char val[FLEN_VALUE];
                char com[FLEN_COMMENT];
                st2 = 0;
                if (fits_read_key(fptr, TSTRING, keyname, val, com, &st2) == 0) {
                    info.tform = std::string(val);
                    // strip quotes if present
                    if (!info.tform.empty() && info.tform.front()=='\'' && info.tform.back()=='\'' && info.tform.size()>=2) {
                        info.tform = info.tform.substr(1, info.tform.size()-2);
                    }
                    // variable-length if contains 'P' or 'Q' before base type letter
                    for (char &c : info.tform) c = (char)toupper(c);
                    if (info.tform.find('P') != std::string::npos || info.tform.find('Q') != std::string::npos) {
                        info.is_varlen = true;
                        // Derive base type code from first letter after P/Q markers
                        // Examples: "1PD(5)", "1PE(10)", "PJ(100)"
                        size_t ppos = info.tform.find('P');
                        if (ppos == std::string::npos) ppos = info.tform.find('Q');
                        if (ppos != std::string::npos) {
                            // find next alpha letter after ppos
                            int base = 0;
                            for (size_t i = ppos + 1; i < info.tform.size(); ++i) {
                                char ch = info.tform[i];
                                if (ch >= 'A' && ch <= 'Z') {
                                    switch (ch) {
                                        case 'L': base = TLOGICAL; break;
                                        case 'B': base = TBYTE; break;
                                        case 'I': base = TSHORT; break;
                                        case 'J': base = TINT; break; // 32-bit
                                        case 'K': base = TLONGLONG; break;
                                        case 'E': base = TFLOAT; break;
                                        case 'D': base = TDOUBLE; break;
                                        default: base = TDOUBLE; break; // fallback
                                    }
                                    break;
                                }
                            }
                            if (base != 0) info.base_typecode = base;
                        }
                    }
                }
            }
        }
        column_info.push_back(info);
    }

    py::dict result_dict;

    // Process string columns sequentially (required due to CFITSIO limitations)
    bool skip_string = false; // always process string columns
    for (const auto& info : column_info) {
        if (!info.is_string) continue;
        if (skip_string) { continue; }
        // Determine max element width (repeat often encodes width for TSTRING); use width if provided
        long elem_width = info.width > 0 ? info.width : info.repeat;
        if (elem_width <= 0) elem_width = 1;

        std::vector<char*> string_array(rows_to_read);
        std::vector<char> string_buffer(rows_to_read * (elem_width + 1), '\0');

        for (long i = 0; i < rows_to_read; i++) {
            string_array[i] = &string_buffer[i * (elem_width + 1)];
        }

        int anynul = 0;
        fits_read_col(fptr, TSTRING, info.number, start_row + 1, 1, rows_to_read,
                      nullptr, string_array.data(), &anynul, &status);
        if (status) throw_fits_error(status, "Error reading string column: " + info.name);

        py::list string_list;
        for (long i = 0; i < rows_to_read; i++) {
            // Ensure we strip trailing spaces similar to writer's padding behavior
            std::string raw(string_array[i]);
            size_t endpos = raw.find_last_not_of(' ');
            if (endpos != std::string::npos) raw.erase(endpos + 1); else raw.clear();
            string_list.append(py::str(raw));
        }
        result_dict[py::str(info.name)] = std::move(string_list);
    }

    // Process numeric columns with optimizations:
    // 1. Bulk per-column fits_read_col for scalar repeats (already implemented)
    // 2. Optional parallel column reads (env TORCHFITS_PAR_READ=1) for 4+ scalar numeric columns
    //    using separate FITS handles per thread (CFITSIO thread-safe with independent fitsfile*).
    // 3. Optional pinned host memory allocation (env TORCHFITS_PIN_MEMORY=1) to accelerate downstream GPU transfers.
    int numeric_processed = 0; // legacy diagnostic counter retained (unused threshold)
    auto elem_size_for = [](int typecode)->size_t {
        switch(typecode){
            case TBYTE: return 1;
            case 12: return 1;            // TSBYTE
            case 20: return 2;            // TUSHORT
            case TSHORT: return 2;
            case 30: return 4;            // TUINT
            case TINT: return 4;
            case 40:                       // TULONG
#ifdef __APPLE__
                return 8;                  // long is 64-bit on macOS
#else
                return 4;
#endif
            case 41:                       // TLONG
#ifdef __APPLE__
                return 8;
#else
                return 4;
#endif
            case TLONGLONG: return 8;
            case TFLOAT: return 4;
            case TDOUBLE: return 8;
            case TLOGICAL: return 1;
            default: return 8;
        }
    };
    // Identify scalar numeric columns eligible for parallelization
    std::vector<ColumnInfo> scalar_numeric;
    for (auto &ci : column_info) {
        if (!ci.is_string && !ci.is_varlen && ci.repeat == 1) scalar_numeric.push_back(ci);
    }
    bool enable_parallel = false;
    const char* par_env = std::getenv("TORCHFITS_PAR_READ");
    // Heuristic: auto-enable when many rows and at least a handful of scalar numeric columns
    bool reading_all_columns = columns_obj.is_none();
    if (par_env) {
        // Explicit override
        if (std::string(par_env) == "1" && scalar_numeric.size() >= 2) enable_parallel = true;
        if (std::string(par_env) == "0") enable_parallel = false;
    } else {
        bool enough_cols_all = reading_all_columns && scalar_numeric.size() >= 4;
        bool enough_cols_subset = (!reading_all_columns) && scalar_numeric.size() >= 4;
        if ((enough_cols_all || enough_cols_subset) && rows_to_read >= 100000) {
            enable_parallel = true;
        }
    }
    bool pin_memory = false;
    const char* pin_env = std::getenv("TORCHFITS_PIN_MEMORY");
    if (pin_env && std::string(pin_env) == "1") pin_memory = true;

    auto read_scalar_column = [&](const ColumnInfo &info)->std::pair<std::string, torch::Tensor> {
        int status_local = 0;
        // Open independent handle for this column to allow parallelism
        FITSFileWrapper f_local(fptr->Fptr->filename); // reuse helper requires filename; if not accessible we fallback to reopening via fits_open_file
        fitsfile* f2;
        if (fits_open_file(&f2, fptr->Fptr->filename, READONLY, &status_local)) {
            throw_fits_error(status_local, "Error reopening file for parallel column read: " + info.name);
        }
        // Move to same HDU number as original
        int hdutype=0;
        if (fits_movabs_hdu(f2, fptr->Fptr->curhdu+1, &hdutype, &status_local)) {
            fits_close_file(f2, &status_local);
            throw_fits_error(status_local, "Error moving to HDU for parallel column read: " + info.name);
        }
        torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
        auto opts = torch::TensorOptions().dtype(torch_dtype);
        if (pin_memory && device == torch::kCPU) opts = opts.pinned_memory(true);
        torch::Tensor col_data = torch::empty({rows_to_read}, opts);
        int anynul = 0; status_local = 0;
        fits_read_col(f2, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                      nullptr, col_data.data_ptr(), &anynul, &status_local);
        fits_close_file(f2, &status_local);
        if (status_local) {
            throw_fits_error(status_local, "Error reading column in parallel: " + info.name);
        }
        if (device != torch::kCPU) col_data = col_data.to(device);
        return {info.name, col_data};
    };

    if (enable_parallel) {
        // Limit concurrency to avoid oversubscription
    size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    // Allow up to 8 worker threads by default (avoid oversubscription)
    size_t cap = std::min<size_t>(hw, 8);
        if (const char* env = std::getenv("TORCHFITS_PAR_MAX_THREADS")) {
            try { cap = std::max<size_t>(1, std::stoul(env)); } catch (...) {}
        }
    cap = std::min(cap, scalar_numeric.size());

        // Partition columns into cap chunks; each worker opens a single handle and processes its chunk
        std::vector<std::vector<ColumnInfo>> chunks(cap);
        for (size_t i = 0; i < scalar_numeric.size(); ++i) {
            chunks[i % cap].push_back(scalar_numeric[i]);
        }

        auto worker = [&](const std::vector<ColumnInfo>& cols)->std::vector<std::pair<std::string, torch::Tensor>> {
            std::vector<std::pair<std::string, torch::Tensor>> out;
            if (cols.empty()) return out;
            int status_local = 0;
            fitsfile* f2 = nullptr;
            if (fits_open_file(&f2, fptr->Fptr->filename, READONLY, &status_local)) {
                throw_fits_error(status_local, "Error reopening file for parallel column chunk");
            }
            int hdutype=0;
            if (fits_movabs_hdu(f2, fptr->Fptr->curhdu+1, &hdutype, &status_local)) {
                fits_close_file(f2, &status_local);
                throw_fits_error(status_local, "Error moving to HDU for parallel column chunk");
            }
            out.reserve(cols.size());
            for (const auto& info : cols) {
                torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
                auto opts = torch::TensorOptions().dtype(torch_dtype);
                if (pin_memory && device == torch::kCPU) opts = opts.pinned_memory(true);
                torch::Tensor col_data = torch::empty({rows_to_read}, opts);
                int anynul = 0; int st = 0;
                fits_read_col(f2, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                              nullptr, col_data.data_ptr(), &anynul, &st);
                if (st) {
                    fits_close_file(f2, &status_local);
                    throw_fits_error(st, std::string("Error reading column in parallel chunk: ") + info.name);
                }
                if (device != torch::kCPU) col_data = col_data.to(device);
                out.emplace_back(info.name, col_data);
            }
            fits_close_file(f2, &status_local);
            return out;
        };

        std::vector<std::future<std::vector<std::pair<std::string, torch::Tensor>>>> futs;
        futs.reserve(cap);
        for (size_t i=0; i<cap; ++i) {
            futs.emplace_back(std::async(std::launch::async, worker, std::cref(chunks[i])));
        }
        for (auto &fut : futs) {
            auto vec = fut.get();
            for (auto &kv : vec) {
                result_dict[py::str(kv.first)] = kv.second;
            }
        }
        g_last_read_info.used_parallel_columns = true;
        // Mark processed so we don't re-read below
        std::unordered_set<std::string> done;
        for (auto &ci : scalar_numeric) done.insert(ci.name);
        // Continue with remaining (including vector repeat) columns below
        for (const auto& info : column_info) {
            if (info.is_string || done.count(info.name)) continue;
            if (info.is_varlen) {
                py::list col_list;
                for (long r = 0; r < rows_to_read; ++r) {
                    LONGLONG llength = 0; LONGLONG offset = 0; int st3 = 0; int anynul = 0; int stread = 0;
                    fits_read_descriptll(fptr, info.number, start_row + 1 + r, &llength, &offset, &st3);
                    if (st3) throw_fits_error(st3, "Error reading VLA descriptor for column: " + info.name);
                    long nelem = static_cast<long>(llength);
                    int read_code = info.base_typecode;
                    torch::Dtype torch_dtype = get_torch_dtype(read_code);
                    auto opts = torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU);
                    torch::Tensor row_tensor;
                    if (nelem > 0) {
                        row_tensor = torch::empty({nelem}, opts);
                        stread = 0; anynul = 0;
                        if (fits_read_col(fptr, read_code, info.number, start_row + 1 + r, 1, nelem,
                                          nullptr, row_tensor.data_ptr(), &anynul, &stread)) {
                            stread = 0; anynul = 0;
                            torch::Tensor tmp = torch::empty({nelem}, torch::TensorOptions().dtype(torch::kFloat64));
                            if (fits_read_col(fptr, TDOUBLE, info.number, start_row + 1 + r, 1, nelem,
                                              nullptr, tmp.data_ptr<double>(), &anynul, &stread)) {
                                throw_fits_error(stread, "Error reading VLA data for column: " + info.name);
                            }
                            row_tensor = tmp.to(torch_dtype);
                        }
                    } else {
                        row_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch_dtype));
                    }
                    col_list.append(row_tensor);
                }
                result_dict[py::str(info.name)] = std::move(col_list);
                continue;
            }
            torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
            torch::TensorOptions opts = torch::TensorOptions().dtype(torch_dtype);
            if (pin_memory && device == torch::kCPU) opts = opts.pinned_memory(true);
            torch::Tensor col_data;
            bool can_bulk = (info.repeat == 1);
            if (can_bulk) {
                col_data = torch::empty({rows_to_read}, opts);
                int anynul = 0; status = 0;
                fits_read_col(fptr, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                              nullptr, col_data.data_ptr(), &anynul, &status);
                if (status) throw_fits_error(status, "Error reading column: " + info.name);
            } else {
                // Read row-wise into a properly-typed temporary tensor to ensure alignment
                col_data = torch::empty({rows_to_read, info.repeat}, opts);
                size_t elem_size = elem_size_for(info.typecode);
                for (long r=0; r<rows_to_read; ++r) {
                    // Allocate a temporary 1D row tensor on CPU with correct dtype
                    torch::Tensor row_tmp = torch::empty({info.repeat}, torch::TensorOptions().dtype(torch_dtype));
                    int anynul=0; status=0;
                    fits_read_col(fptr, info.typecode, info.number, start_row + 1 + r, 1, info.repeat, nullptr,
                                   row_tmp.data_ptr(), &anynul, &status);
                    if (status) throw_fits_error(status, "Error reading column (row-wise): " + info.name);
                    // Copy bytes into destination row (avoid dtype conversions here)
                    char* dest = static_cast<char*>(col_data.data_ptr()) + static_cast<size_t>(r) * static_cast<size_t>(info.repeat) * elem_size;
                    std::memcpy(dest, row_tmp.data_ptr(), static_cast<size_t>(info.repeat) * elem_size);
                }
            }
            if (device != torch::kCPU) col_data = col_data.to(device);
            result_dict[py::str(info.name)] = (info.repeat == 1 ? col_data : col_data.squeeze());
        }
    return result_dict;
    }

    // Sequential path (existing logic with optional pinning)
    for (const auto& info : column_info) {
        if (info.is_string) continue;
        // Handle variable-length arrays as list[Tensor] per column
        if (info.is_varlen) {
            #ifdef TORCHFITS_DEBUG
            fprintf(stderr, "[TORCHFITS_DEBUG] VLA column %s base_type=%d rows=%ld\n", info.name.c_str(), info.base_typecode, rows_to_read);
            #endif
            // Heuristic/flag to parallelize per-row reads using independent CFITSIO handles
            bool vla_parallel = false;
            if (const char* env = std::getenv("TORCHFITS_VLA_PAR")) {
                vla_parallel = std::string(env) == "1";
            } else {
                vla_parallel = rows_to_read >= 4096; // heuristic threshold
            }
            int read_code = info.base_typecode;
            torch::Dtype torch_dtype = get_torch_dtype(read_code);
            auto opts = torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU);

            std::vector<torch::Tensor> row_tensors(static_cast<size_t>(rows_to_read));

            if (vla_parallel) {
                size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
                size_t cap = std::min<size_t>(hw, 8);
                if (const char* env = std::getenv("TORCHFITS_PAR_MAX_THREADS")) {
                    try { cap = std::max<size_t>(1, std::stoul(env)); } catch (...) {}
                }
                cap = std::min<size_t>(cap, static_cast<size_t>(rows_to_read));
                // Build chunks of row indices
                std::vector<std::vector<long>> chunks(cap);
                for (long r = 0; r < rows_to_read; ++r) chunks[static_cast<size_t>(r % cap)].push_back(r);
                auto worker = [&](const std::vector<long>& rows){
                    int st = 0; fitsfile* f2 = nullptr;
                    if (fits_open_file(&f2, fptr->Fptr->filename, READONLY, &st)) {
                        throw_fits_error(st, "Error reopening file for VLA parallel read");
                    }
                    int hdutype=0; if (fits_movabs_hdu(f2, fptr->Fptr->curhdu+1, &hdutype, &st)) { fits_close_file(f2, &st); throw_fits_error(st, "Error moving to HDU for VLA parallel read"); }
                    for (long rr : rows) {
                        LONGLONG llength = 0, offset = 0; int st3 = 0; int anynul = 0; int stread = 0;
                        fits_read_descriptll(f2, info.number, start_row + 1 + rr, &llength, &offset, &st3);
                        if (st3) throw_fits_error(st3, std::string("Error reading VLA descriptor for ") + info.name);
                        long nelem = static_cast<long>(llength);
                        torch::Tensor row_tensor;
                        if (nelem > 0) {
                            row_tensor = torch::empty({nelem}, opts);
                            stread = 0; anynul = 0;
                            if (fits_read_col(f2, read_code, info.number, start_row + 1 + rr, 1, nelem, nullptr, row_tensor.data_ptr(), &anynul, &stread)) {
                                // Fallback read as double then cast
                                stread = 0; anynul = 0;
                                torch::Tensor tmp = torch::empty({nelem}, torch::TensorOptions().dtype(torch::kFloat64));
                                if (fits_read_col(f2, TDOUBLE, info.number, start_row + 1 + rr, 1, nelem, nullptr, tmp.data_ptr<double>(), &anynul, &stread)) {
                                    throw_fits_error(stread, std::string("Error reading VLA data for ") + info.name);
                                }
                                row_tensor = tmp.to(torch_dtype);
                            }
                        } else {
                            row_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch_dtype));
                        }
                        row_tensors[static_cast<size_t>(rr)] = std::move(row_tensor);
                    }
                    fits_close_file(f2, &st);
                };
                std::vector<std::future<void>> futs; futs.reserve(cap);
                for (size_t i=0; i<cap; ++i) {
                    futs.emplace_back(std::async(std::launch::async, worker, std::cref(chunks[i])));
                }
                for (auto& fut : futs) fut.get();
            } else {
                for (long r = 0; r < rows_to_read; ++r) {
                    LONGLONG llength = 0; LONGLONG offset = 0; int st3 = 0; int anynul = 0; int stread = 0;
                    fits_read_descriptll(fptr, info.number, start_row + 1 + r, &llength, &offset, &st3);
                    if (st3) throw_fits_error(st3, "Error reading VLA descriptor for column: " + info.name);
                    long nelem = static_cast<long>(llength);
                    torch::Tensor row_tensor;
                    if (nelem > 0) {
                        row_tensor = torch::empty({nelem}, opts);
                        stread = 0; anynul = 0;
                        if (fits_read_col(fptr, read_code, info.number, start_row + 1 + r, 1, nelem,
                                          nullptr, row_tensor.data_ptr(), &anynul, &stread)) {
                            // Fallback: try reading as double then cast to target dtype
                            stread = 0; anynul = 0;
                            torch::Tensor tmp = torch::empty({nelem}, torch::TensorOptions().dtype(torch::kFloat64));
                            if (fits_read_col(fptr, TDOUBLE, info.number, start_row + 1 + r, 1, nelem,
                                              nullptr, tmp.data_ptr<double>(), &anynul, &stread)) {
                                throw_fits_error(stread, "Error reading VLA data for column: " + info.name);
                            }
                            row_tensor = tmp.to(torch_dtype);
                        }
                    } else {
                        row_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch_dtype));
                    }
                    row_tensors[static_cast<size_t>(r)] = std::move(row_tensor);
                }
            }
            py::list col_list;
            for (long r = 0; r < rows_to_read; ++r) {
                col_list.append(row_tensors[static_cast<size_t>(r)]);
            }
            result_dict[py::str(info.name)] = std::move(col_list);
            continue;
        }
        torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch_dtype);
        if (pin_memory && device == torch::kCPU) opts = opts.pinned_memory(true);
        torch::Tensor col_data;
        bool can_bulk = (info.repeat == 1); // Simplest safe case first (scalar columns)
        if (can_bulk) {
            col_data = torch::empty({rows_to_read}, opts);
            int anynul = 0; status = 0;
            // Single CFITSIO call for entire column segment
#ifdef TORCHFITS_DEBUG
            fprintf(stderr, "[TORCHFITS_DEBUG] Read numeric column %s typecode=%d col=%d start=%ld n=%ld\n", info.name.c_str(), info.typecode, info.number, start_row, rows_to_read);
#endif
            fits_read_col(fptr, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                          nullptr, col_data.data_ptr(), &anynul, &status);
#ifdef TORCHFITS_DEBUG
            if (!status) fprintf(stderr, "[TORCHFITS_DEBUG] ok column %s\n", info.name.c_str());
#endif
            if (status) {
                // Fallback: retry row-wise in case of intermittent failure
                status = 0;
                size_t elem_size = elem_size_for(info.typecode);
                for (long r=0; r<rows_to_read; ++r) {
                    // Read single element into a temporary properly-typed tensor
                    torch::Tensor row_tmp = torch::empty({1}, torch::TensorOptions().dtype(torch_dtype));
                    int anynul2=0; int st2=0;
                    if (fits_read_col(fptr, info.typecode, info.number, start_row + 1 + r, 1, 1, nullptr,
                                      row_tmp.data_ptr(), &anynul2, &st2)) {
                        throw_fits_error(st2, "Error reading column (row-wise fallback): " + info.name);
                    }
                    char* dest = static_cast<char*>(col_data.data_ptr()) + static_cast<size_t>(r) * elem_size;
                    std::memcpy(dest, row_tmp.data_ptr(), elem_size);
                }
            }
        } else {
            // Existing row-wise path for vector/repeat columns
            col_data = torch::empty({rows_to_read, info.repeat}, opts);
            size_t elem_size = elem_size_for(info.typecode);
            for (long r=0; r<rows_to_read; ++r) {
                // Properly-typed temporary row tensor
                torch::Tensor row_tmp = torch::empty({info.repeat}, torch::TensorOptions().dtype(torch_dtype));
                int anynul=0; status=0;
                fits_read_col(fptr, info.typecode, info.number, start_row + 1 + r, 1, info.repeat, nullptr,
                               row_tmp.data_ptr(), &anynul, &status);
                if (status) throw_fits_error(status, "Error reading column (row-wise): " + info.name);
                char* dest = static_cast<char*>(col_data.data_ptr()) + static_cast<size_t>(r) * static_cast<size_t>(info.repeat) * elem_size;
                std::memcpy(dest, row_tmp.data_ptr(), static_cast<size_t>(info.repeat) * elem_size);
            }
        }
        if (device != torch::kCPU) col_data = col_data.to(device);
        result_dict[py::str(info.name)] = (info.repeat == 1 ? col_data : col_data.squeeze());
        ++numeric_processed;
    }

    return result_dict;
}

// Helper: try to read TNULLi for a given column, returns (has_tnull, value)
static std::pair<bool, long long> get_tnull_for_column(fitsfile* fptr, int colnum) {
    int status = 0;
    char keyname[FLEN_KEYWORD];
    if (fits_make_keyn("TNULL", colnum, keyname, &status)) {
        return {false, 0};
    }
    // TNULL is integer; read as long long for safety
    long long val = 0;
    status = 0;
    if (fits_read_key_lnglng(fptr, keyname, &val, nullptr, &status) == 0) {
        return {true, val};
    }
    return {false, 0};
}

// Read table data and per-column null masks in one pass where possible (integer scalar columns)
// Returns a tuple: (data_dict, masks_dict)
static py::tuple read_table_data_with_masks(
    fitsfile* fptr, torch::Device device,
    const py::object& columns_obj,
    long start_row, const py::object& num_rows_obj) {
    int status = 0;
    long total_rows;
    int total_cols;

    fits_get_num_rows(fptr, &total_rows, &status);
    if (status) throw_fits_error(status, "Error getting number of rows");

    fits_get_num_cols(fptr, &total_cols, &status);
    if (status) throw_fits_error(status, "Error getting number of columns");

    long rows_to_read = total_rows - start_row;
    if (!num_rows_obj.is_none()) {
        rows_to_read = std::min(rows_to_read, num_rows_obj.cast<long>());
    }
    if (rows_to_read <= 0) {
        return py::make_tuple(py::dict(), py::dict());
    }

    // Determine columns to read
    std::vector<std::string> selected_columns;
    if (!columns_obj.is_none()) {
        selected_columns = columns_obj.cast<std::vector<std::string>>();
    } else {
        selected_columns.reserve(total_cols);
        for (int i = 1; i <= total_cols; ++i) {
            char keyname[FLEN_KEYWORD];
            status = 0;
            if (fits_make_keyn("TTYPE", i, keyname, &status)) {
                throw_fits_error(status, "Error forming TTYPE keyword for column " + std::to_string(i));
            }
            char colname[FLEN_VALUE];
            char comment[FLEN_COMMENT];
            status = 0;
            if (fits_read_key(fptr, TSTRING, keyname, colname, comment, &status)) {
                throw_fits_error(status, "Error reading " + std::string(keyname) + " for column " + std::to_string(i));
            }
            std::string colname_str(colname);
            if (!colname_str.empty() && colname_str.front() == '\'' && colname_str.back() == '\'' && colname_str.size() >= 2) {
                colname_str = colname_str.substr(1, colname_str.size() - 2);
            }
            selected_columns.emplace_back(std::move(colname_str));
        }
    }

    struct ColumnInfoMask {
        std::string name;
        int number;
        int typecode;
        long repeat;
        bool is_string;
    };

    std::vector<ColumnInfoMask> cols;
    cols.reserve(selected_columns.size());
    for (const auto& col_name : selected_columns) {
        ColumnInfoMask info;
        info.name = col_name;
        status = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &info.number, &status);
        if (status) throw_fits_error(status, "Error finding column: " + col_name);
        long width = 0;
        fits_get_coltype(fptr, info.number, &info.typecode, &info.repeat, &width, &status);
        if (status) throw_fits_error(status, "Error getting column type for: " + col_name);
        info.is_string = (info.typecode == TSTRING);
        cols.push_back(info);
    }

    py::dict data_dict;
    py::dict masks_dict;

    // Process string columns first (no masks)
    for (const auto& info : cols) {
        if (!info.is_string) continue;
        long elem_width = 1; // Conservative fallback width for strings
        {
            int stw = 0; int typecode=0; long repeat=0, width=0;
            fits_get_coltype(fptr, info.number, &typecode, &repeat, &width, &stw);
            if (!stw && (width > 0)) elem_width = width;
            else if (repeat > 0) elem_width = repeat;
        }
        std::vector<char*> string_array(rows_to_read);
        std::vector<char> string_buffer(rows_to_read * (elem_width + 1), '\0');
        for (long i = 0; i < rows_to_read; i++) {
            string_array[i] = &string_buffer[i * (elem_width + 1)];
        }
        int anynul = 0; status = 0;
        fits_read_col(fptr, TSTRING, info.number, start_row + 1, 1, rows_to_read,
                      nullptr, string_array.data(), &anynul, &status);
        if (status) throw_fits_error(status, "Error reading string column: " + info.name);
        py::list string_list;
        for (long i = 0; i < rows_to_read; i++) {
            std::string raw(string_array[i]);
            size_t endpos = raw.find_last_not_of(' ');
            if (endpos != std::string::npos) raw.erase(endpos + 1); else raw.clear();
            string_list.append(py::str(raw));
        }
        data_dict[py::str(info.name)] = std::move(string_list);
    }

    auto is_integer_like = [](int tc)->bool {
        return tc == TBYTE || tc == 12 || tc == 20 || tc == TSHORT || tc == 30 || tc == TINT || tc == 40 || tc == 41 || tc == TLONGLONG; };

    // Process numeric columns with optional one-pass null mask read for integer scalar columns
    for (const auto& info : cols) {
        if (info.is_string) continue;
        // Only scalar repeats handled for masks in this fast path
        if (info.repeat != 1) {
            // Fallback to existing bulk/row-wise path without masks
            torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
            auto opts = torch::TensorOptions().dtype(torch_dtype);
            torch::Tensor col_data = torch::empty({rows_to_read}, opts);
            int anynul = 0; status = 0;
            fits_read_col(fptr, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                          nullptr, col_data.data_ptr(), &anynul, &status);
            if (status) throw_fits_error(status, std::string("Error reading column: ") + info.name);
            if (device != torch::kCPU) col_data = col_data.to(device);
            data_dict[py::str(info.name)] = col_data;
            continue;
        }

        bool did_mask = false;
        if (is_integer_like(info.typecode)) {
            auto [has_tnull, tnull_val] = get_tnull_for_column(fptr, info.number);
            if (has_tnull) {
                // Allocate data tensor and null flags
                torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
                auto opts = torch::TensorOptions().dtype(torch_dtype);
                torch::Tensor col_data = torch::empty({rows_to_read}, opts);
                std::vector<char> nullflags(rows_to_read, 0);
                int anynul = 0; status = 0;
                // Use the generic API variant: it fills nullflags where values equal the column's TNULLn
                // CFITSIO will auto-compare against TNULLn in header when provided a nullarray.
                fits_read_colnull(
                    fptr,
                    info.typecode,
                    info.number,
                    start_row + 1,
                    1,
                    rows_to_read,
                    col_data.data_ptr(),
                    nullflags.data(),
                    &anynul,
                    &status
                );
                if (status) {
                    throw_fits_error(status, std::string("Error reading column with null mask: ") + info.name);
                }
                if (device != torch::kCPU) col_data = col_data.to(device);
                data_dict[py::str(info.name)] = col_data;
                // Build mask tensor (True where null). CFITSIO sets nullarray[i] = 1 for nulls.
                torch::Tensor mask = torch::empty({rows_to_read}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
                for (long i = 0; i < rows_to_read; ++i) {
                    static_cast<bool*>(mask.data_ptr())[i] = (nullflags[static_cast<size_t>(i)] != 0);
                }
                masks_dict[py::str(info.name)] = mask;
                did_mask = true;
            }
        }
        if (!did_mask) {
            // Fallback: no mask; just read data
            torch::Dtype torch_dtype = get_torch_dtype(info.typecode);
            auto opts = torch::TensorOptions().dtype(torch_dtype);
            torch::Tensor col_data = torch::empty({rows_to_read}, opts);
            int anynul = 0; status = 0;
            fits_read_col(fptr, info.typecode, info.number, start_row + 1, 1, rows_to_read,
                          nullptr, col_data.data_ptr(), &anynul, &status);
            if (status) throw_fits_error(status, std::string("Error reading column: ") + info.name);
            if (device != torch::kCPU) col_data = col_data.to(device);
            data_dict[py::str(info.name)] = col_data;
        }
    }

    return py::make_tuple(data_dict, masks_dict);
}

// Exposed helper: Open file/HDU, then read table with per-column null masks; returns (data, header, masks)
pybind11::object read_table_with_null_masks_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object columns,
    long start_row,
    pybind11::object num_rows,
    pybind11::str device_str
) {
    torch::Device device(device_str);
    try {
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
        std::string filename = RemoteFetcher::ensure_local(filename_or_url_str);

        int hdu_num = 1;
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        FITSFileWrapper f(filename);
        int status = 0;
        if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
        auto header = read_fits_header(f.get());
        int hdu_type = 0; status = 0; fits_get_hdu_type(f.get(), &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");
        if (!(hdu_type == BINARY_TBL || hdu_type == ASCII_TBL)) {
            throw std::runtime_error("read_table_with_null_masks: target HDU is not a table");
        }
        auto tup = read_table_data_with_masks(f.get(), device, columns, start_row, num_rows);
        return py::make_tuple(tup[0], py::cast(header), tup[1]);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}


// --- Main read implementation with CFITSIO enhancements ---
pybind11::object read_impl(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    pybind11::object start,
    pybind11::object shape,
    pybind11::object columns,
    long start_row,
    pybind11::object num_rows,
    size_t cache_capacity,
    pybind11::object enable_mmap,
    pybind11::object enable_buffered,
    pybind11::str device_str
) {
    torch::Device device(device_str);
    // read_impl core logic (debug instrumentation removed)

    try {
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
    // filename logged previously in debug mode
        
        // Handle remote URLs - download to cache if necessary
        std::string filename = RemoteFetcher::ensure_local(filename_or_url_str);

        int hdu_num = 1;
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        // === REAL CACHE CHECK: Try to get from cache first ===
        auto& real_cache = torchfits_real_cache::RealSmartCache::get_instance();
        std::ostringstream cache_key_stream;
        cache_key_stream << filename << "_hdu" << hdu_num;
        
        if (!start.is_none()) {
            auto start_vec = start.cast<std::vector<long>>();
            cache_key_stream << "_start";
            for (size_t i = 0; i < start_vec.size(); ++i) {
                cache_key_stream << "_" << start_vec[i];
            }
        }
        if (!shape.is_none()) {
            auto shape_vec = shape.cast<std::vector<long>>();
            cache_key_stream << "_shape";
            for (size_t i = 0; i < shape_vec.size(); ++i) {
                cache_key_stream << "_" << shape_vec[i];
            }
        }
        cache_key_stream << "_device_" << device_str.cast<std::string>();
        
        std::string cache_key = cache_key_stream.str();
        auto cached_result = real_cache.try_get(cache_key);
        if (cached_result.has_value()) {
            g_last_read_info.filename = filename;
            g_last_read_info.hdu = hdu_num;
            g_last_read_info.used_cache = true;
            g_last_read_info.used_mmap = false;
            g_last_read_info.used_buffered = false;
            g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns; // preserve
            g_last_read_info.path_used = std::string("cache");
            // cache hit
            // For cache hits, we need to read the header separately
            FITSFileWrapper f(filename);
            int status = 0;
            if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
                throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
            }
            auto header = read_fits_header(f.get());
            return py::make_tuple(py::cast(cached_result.value()), py::cast(header));
        }

    // Determine fast-path flags (explicit opt-in)
    bool try_mmap = false;
    if (!enable_mmap.is_none()) {
        try_mmap = enable_mmap.cast<bool>();
    }
    bool try_buffered = false;
    if (!enable_buffered.is_none()) {
        try_buffered = enable_buffered.cast<bool>();
    }

    // === PERFORMANCE OPTIMIZATION: Memory-mapped reading (full, uncompressed images) ===
    if (try_mmap && start.is_none() && shape.is_none()) {
            try {
                // Only attempt mmap for images (uncompressed primary or image extension)
                FITSFileWrapper fcheck(filename);
                int st_m = 0;
                if (fits_movabs_hdu(fcheck.get(), hdu_num, NULL, &st_m)) {
                    throw_fits_error(st_m, "Error moving to HDU " + std::to_string(hdu_num));
                }
                int hdu_type_m = 0; st_m = 0; fits_get_hdu_type(fcheck.get(), &hdu_type_m, &st_m);
                if (st_m) throw_fits_error(st_m, "Error getting HDU type");
                int stc_m = 0; int is_comp_m = fits_is_compressed_image(fcheck.get(), &stc_m); (void)stc_m;
                if ((hdu_type_m == IMAGE_HDU) && !is_comp_m) {
                    // Gate by size threshold (default 50MB, env override TORCHFITS_MMAP_MIN_MB)
                    size_t file_sz = 0;
                    {
                        std::ifstream fs(filename, std::ifstream::ate | std::ifstream::binary);
                        if (fs.is_open()) { file_sz = static_cast<size_t>(fs.tellg()); fs.close(); }
                    }
                    size_t min_mb = 50;
                    if (const char* env = std::getenv("TORCHFITS_MMAP_MIN_MB")) {
                        try { min_mb = std::stoul(env); } catch (...) {}
                    }
                    if (file_sz < min_mb * 1024ULL * 1024ULL) {
            throw std::runtime_error("mmap: below size threshold");
                    }
                    torchfits_cfitsio_enhanced::CFITSIOMemoryMapper mapper(filename);
                    torch::Tensor t = mapper.read_with_memory_mapping(hdu_num);
                    torch::Tensor device_t = t.to(device);
                    auto header = read_fits_header(fcheck.get());
                    // Cache and return
                    auto& real_cache = torchfits_real_cache::RealSmartCache::get_instance();
                    real_cache.put(cache_key_stream.str(), device_t);
                    g_last_read_info.filename = filename;
                    g_last_read_info.hdu = hdu_num;
                    g_last_read_info.used_cache = false;
                    g_last_read_info.used_mmap = true;
                    g_last_read_info.used_buffered = false;
                    g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
                    g_last_read_info.path_used = std::string("mmap");
                    return py::make_tuple(py::cast(device_t), py::cast(header));
                }
            } catch (const std::exception&) {
                // Fall through to buffered/standard path
            }
        }

    // === PERFORMANCE OPTIMIZATION: CFITSIO buffered/tiled reading for full images ===
    if (try_buffered && start.is_none() && shape.is_none()) {
            try {
                torchfits_cfitsio_enhanced::CFITSIOBufferedReader buffered_reader(filename);
                FITSFileWrapper f2(filename);
                int st_enh = 0;
                if (fits_movabs_hdu(f2.get(), hdu_num, NULL, &st_enh)) {
                    throw_fits_error(st_enh, "Error moving to HDU " + std::to_string(hdu_num));
                }
                int hdu_type_enh = 0; st_enh = 0; fits_get_hdu_type(f2.get(), &hdu_type_enh, &st_enh);
                if (st_enh) throw_fits_error(st_enh, "Error getting HDU type");
                // Prefer tiled path on compressed images
                int stc_enh = 0; int is_comp = fits_is_compressed_image(f2.get(), &stc_enh); (void)stc_enh;
                torch::Tensor result;
                if (hdu_type_enh == IMAGE_HDU || is_comp) {
                    if (is_comp) result = buffered_reader.read_tiled(hdu_num);
                    else result = buffered_reader.read_with_scaling(hdu_num);
                    torch::Tensor device_result = result.to(device);
                    real_cache.put(cache_key, device_result);
                    auto header = read_fits_header(f2.get());
                    g_last_read_info.filename = filename;
                    g_last_read_info.hdu = hdu_num;
                    g_last_read_info.used_cache = false;
                    g_last_read_info.used_mmap = false;
                    g_last_read_info.used_buffered = true;
                    g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
                    g_last_read_info.path_used = std::string("buffered");
                    return py::make_tuple(py::cast(device_result), py::cast(header));
                }
            } catch (const std::exception&) {
                // Fall through to standard path on any failure
            }
        }

        // === FALLBACK: Standard reading with performance optimization ===
        FITSFileWrapper f(filename);
        int status = 0;
        if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }

        // Apply CFITSIO performance optimizations to standard reading
        // OPTIMIZATION: Get actual file size for better buffer sizing
        std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
        size_t file_size = 1000000; // Default fallback
        if (file.is_open()) {
            file_size = static_cast<size_t>(file.tellg());
            file.close();
            // file size available for potential future optimization
        }
        
    // Temporarily disabled performance optimizer (suspected segfault source for table reads)
    // torchfits_cfitsio_enhanced::CFITSIOPerformanceOptimizer::optimize_file_access(f.get(), file_size);

        auto header = read_fits_header(f.get());

        int hdu_type;
        fits_get_hdu_type(f.get(), &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");

    status = 0;
    int is_comp_api = fits_is_compressed_image(f.get(), &status); status = 0;
    bool is_compressed_image = (is_comp_api != 0);

        // If user asked primary but it's empty (NAXIS=0), try advancing to first compressed image extension
        if (hdu_type == IMAGE_HDU) {
            int naxis = 0;
            fits_get_img_dim(f.get(), &naxis, &status);
            if (naxis == 0) {
                // Peek next HDU
                int total_hdus = 0; fits_get_num_hdus(f.get(), &total_hdus, &status); status = 0;
                if (total_hdus > 1) {
                    int hdutype2 = 0;
                    if (fits_movabs_hdu(f.get(), hdu_num + 1, &hdutype2, &status) == 0) {
                        int stc = 0; int is_comp2 = fits_is_compressed_image(f.get(), &stc); stc = 0;
                        auto hdr2 = read_fits_header(f.get());
                        if (is_comp2) {
                            // Treat this as image HDU
                            header = std::move(hdr2);
                            hdu_type = IMAGE_HDU; // Read using image routines
                            is_compressed_image = true;
                        } else {
                            // Move back to original HDU
                            status = 0; fits_movabs_hdu(f.get(), hdu_num, NULL, &status); status = 0;
                        }
                    }
                }
                // If still not a compressed image, return None for empty primary
                if (!is_compressed_image) {
                    return py::make_tuple(py::none(), py::cast(header));
                }
            }
        }

        if (hdu_type == IMAGE_HDU || is_compressed_image) {
            std::vector<long> start_vec, shape_vec;
            bool has_subset = false;
            if (!start.is_none()) {
                start_vec = start.cast<std::vector<long>>();
                shape_vec = shape.cast<std::vector<long>>();
                has_subset = true;
            }

            // Heuristic: for small subsets, read full image once and slice; reuse via cache across repeated calls
            if (has_subset) {
                int status_dims = 0, naxis = 0;
                fits_get_img_dim(f.get(), &naxis, &status_dims);
                if (status_dims) throw_fits_error(status_dims, "Error getting image dimension");
                std::vector<long> naxes(naxis);
                fits_get_img_size(f.get(), naxis, naxes.data(), &status_dims);
                if (status_dims) throw_fits_error(status_dims, "Error getting image size");

                // Compute total pixels and requested subset pixels
                long long total_pixels = 1;
                for (int i = 0; i < naxis; ++i) total_pixels *= static_cast<long long>(naxes[i]);
                long long subset_pixels = 1;
                for (size_t i = 0; i < shape_vec.size(); ++i) subset_pixels *= static_cast<long long>(shape_vec[i]);

                // Thresholds (env-overridable)
                double frac_thresh = 0.05; // default: 5%
                if (const char* env = std::getenv("TORCHFITS_SUBSET_FULL_FRAC")) {
                    try { frac_thresh = std::stod(env); } catch (...) {}
                }
                long long max_full_pixels = 2048LL * 2048LL; // default safety cap
                if (const char* env2 = std::getenv("TORCHFITS_SUBSET_FULL_MAX_PIXELS")) {
                    try { max_full_pixels = static_cast<long long>(std::stoll(env2)); } catch (...) {}
                }

                bool prefer_full_then_slice = (total_pixels > 0) &&
                                              (static_cast<double>(subset_pixels) / static_cast<double>(total_pixels) <= frac_thresh) &&
                                              (total_pixels <= max_full_pixels);

                if (prefer_full_then_slice) {
                    // Build a cache key for the full image (without start/shape)
                    std::ostringstream full_key_stream;
                    full_key_stream << filename << "_hdu" << hdu_num << "_device_" << device_str.cast<std::string>();
                    std::string full_cache_key = full_key_stream.str();

                    // Try to fetch full image from cache
                    auto cached_full = real_cache.try_get(full_cache_key);
                    torch::Tensor full;
                    if (cached_full.has_value()) {
                        full = cached_full.value();
                    } else {
                        // Read full image once
                        std::vector<long> empty_start, empty_shape;
                        full = read_image_data(f.get(), device, empty_start, empty_shape);
                        if (full.defined()) {
                            real_cache.put(full_cache_key, full);
                        }
                    }

                    // Slice the requested window from the full tensor; account for reversed dim order
                    torch::Tensor view = full;
                    for (size_t i = 0; i < shape_vec.size(); ++i) {
                        size_t rev = shape_vec.size() - 1 - i;
                        view = view.narrow(static_cast<int64_t>(rev), static_cast<int64_t>(start_vec[i]), static_cast<int64_t>(shape_vec[i]));
                    }

                    // Cache subset result under the regular cache_key as well
                    real_cache.put(cache_key, view);

                    g_last_read_info.filename = filename;
                    g_last_read_info.hdu = hdu_num;
                    g_last_read_info.used_cache = false;
                    g_last_read_info.used_mmap = false;
                    g_last_read_info.used_buffered = false;
                    g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
                    g_last_read_info.path_used = std::string("standard");

                    return py::make_tuple(py::cast(view), py::cast(header));
                }
            }

            // Default path: direct subset or full-image read
            torch::Tensor data = read_image_data(f.get(), device, start_vec, shape_vec);
            if (data.defined()) {
                real_cache.put(cache_key, data);
            }
            g_last_read_info.filename = filename;
            g_last_read_info.hdu = hdu_num;
            g_last_read_info.used_cache = false;
            g_last_read_info.used_mmap = false;
            g_last_read_info.used_buffered = false;
            g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
            g_last_read_info.path_used = std::string("standard");
            return py::make_tuple(data, py::cast(header));
        } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
            // Always proceed to read table data (early return diagnostic removed)
            py::dict table_data = read_table_data(f.get(), device, columns, start_row, num_rows);
            g_last_read_info.filename = filename;
            g_last_read_info.hdu = hdu_num;
            g_last_read_info.used_cache = false;
            g_last_read_info.used_mmap = false;
            g_last_read_info.used_buffered = false;
            g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
            g_last_read_info.path_used = std::string("standard");
            return py::make_tuple(table_data, py::cast(header));
        } else {
            // For unknown HDU types, return header but no data.
            g_last_read_info.filename = filename;
            g_last_read_info.hdu = hdu_num;
            g_last_read_info.used_cache = false;
            g_last_read_info.used_mmap = false;
            g_last_read_info.used_buffered = false;
            g_last_read_info.used_parallel_columns = g_last_read_info.used_parallel_columns;
            g_last_read_info.path_used = std::string("standard");
            return py::make_tuple(py::none(), py::cast(header));
        }

    } catch (const std::exception& e) {
        // Re-throw C++ exceptions as Python exceptions
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

// Optimized batched cutouts: one file open, many small reads
pybind11::object read_many_cutouts(
    pybind11::object filename_or_url,
    pybind11::object hdu,
    const std::vector<std::vector<long>>& starts,
    const std::vector<long>& shape,
    pybind11::str device_str
) {
    torch::Device device(device_str);
    try {
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
        std::string filename = RemoteFetcher::ensure_local(filename_or_url_str);

        int hdu_num = 1;
        if (!hdu.is_none()) {
            if (py::isinstance<py::str>(hdu)) {
                hdu_num = get_hdu_num_by_name(filename, hdu.cast<std::string>());
            } else {
                hdu_num = hdu.cast<int>();
            }
        }

        FITSFileWrapper f(filename);
        int status = 0;
        if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }

        int hdu_type = 0;
        fits_get_hdu_type(f.get(), &hdu_type, &status);
        if (status) throw_fits_error(status, "Error getting HDU type");
        if (hdu_type != IMAGE_HDU) {
            throw std::runtime_error("read_many_cutouts: target HDU is not an image");
        }

        // Heuristic: coalesce clustered cutouts into one bounding-box read to reduce I/O
        py::list out;
        bool can_coalesce = !starts.empty() && !shape.empty();
        // Allow disabling via env var TORCHFITS_CUTOUT_COALESCE=0
        if (const char* env = std::getenv("TORCHFITS_CUTOUT_COALESCE")) {
            if (std::string(env) == "0") can_coalesce = false;
        }
        if (can_coalesce) {
            // Compute bounding box [min_start, max_end) in user ordering
            std::vector<long> min_start(shape.size(), std::numeric_limits<long>::max());
            std::vector<long> max_end(shape.size(), std::numeric_limits<long>::min());
            for (const auto& st : starts) {
                if (st.size() != shape.size()) { can_coalesce = false; break; }
                for (size_t i = 0; i < shape.size(); ++i) {
                    min_start[i] = std::min(min_start[i], st[i]);
                    max_end[i]   = std::max(max_end[i],   st[i] + shape[i]);
                }
            }

            if (can_coalesce) {
                // Compute sizes and decide if it's worth coalescing
                std::vector<long> bbox_shape(shape.size());
                long long bbox_pixels = 1;
                long long per_cutout_pixels = 1;
                for (size_t i = 0; i < shape.size(); ++i) {
                    bbox_shape[i] = std::max<long>(0, max_end[i] - min_start[i]);
                    bbox_pixels *= static_cast<long long>(bbox_shape[i]);
                    per_cutout_pixels *= static_cast<long long>(shape[i]);
                }
                long long requested_pixels = per_cutout_pixels * static_cast<long long>(starts.size());
                double max_excess = 2.0; // default: allow up to 2x extra pixels in bbox
                if (const char* env = std::getenv("TORCHFITS_CUTOUT_COALESCE_MAX_EXCESS")) {
                    try { max_excess = std::stod(env); } catch (...) {}
                }
                // Benefit check: bbox should not be much larger than total requested area
                if (requested_pixels > 0 && static_cast<double>(bbox_pixels) <= max_excess * static_cast<double>(requested_pixels)) {
                    // Single read of the bounding region
                    torch::Tensor bbox = read_image_data(f.get(), device, min_start, bbox_shape);
                    // Slice each cutout from bbox (note: tensor dims are reversed vs user order in read_image_data)
                    // Build index slices for torch using reversed order
                    for (const auto& st : starts) {
                        // Compute local start within bbox
                        std::vector<long> local_start(shape.size());
                        for (size_t i = 0; i < shape.size(); ++i) local_start[i] = st[i] - min_start[i];

                        // Start from a view and narrow along each dim; account for reversed order
                        torch::Tensor view = bbox;
                        for (size_t i = 0; i < shape.size(); ++i) {
                            size_t rev = shape.size() - 1 - i; // tensor dimension index
                            view = view.narrow(static_cast<int64_t>(rev), static_cast<int64_t>(local_start[i]), static_cast<int64_t>(shape[i]));
                        }
                        out.append(py::cast(view));
                    }
                    return out;
                }
            }
        }

        // If coalescing wasn't beneficial, try a tile-aware path for compressed images to reuse tiles
        int st_comp = 0; int is_comp = fits_is_compressed_image(f.get(), &st_comp); st_comp = 0;
        if (is_comp) {
            // Only handle simple 2D images for now
            int naxis = 0; int st_na=0; fits_get_img_dim(f.get(), &naxis, &st_na);
            if (st_na) throw_fits_error(st_na, "Error getting image dimension");
            if (naxis == 2) {
                std::vector<long> naxes(2); int st_sz=0; fits_get_img_size(f.get(), 2, naxes.data(), &st_sz);
                if (st_sz) throw_fits_error(st_sz, "Error getting image size");
                // Read tile sizes from ZTILE1/2; if missing, default to full axis (single tile)
                long ztile1 = naxes[0]; long ztile2 = naxes[1];
                {
                    int st_k=0; long v=0;
                    if (fits_read_key_lng(f.get(), "ZTILE1", &v, nullptr, &st_k) == 0 && v > 0) ztile1 = v; st_k = 0;
                    if (fits_read_key_lng(f.get(), "ZTILE2", &v, nullptr, &st_k) == 0 && v > 0) ztile2 = v; st_k = 0;
                }
                // Optional global tile cache via RealSmartCache
                bool global_tile_cache = true;
                if (const char* env = std::getenv("TORCHFITS_TILE_CACHE")) {
                    if (std::string(env) == "0") global_tile_cache = false;
                }
                // Local per-call cache when global disabled
                struct TileKey { long ty; long tx; bool operator==(const TileKey& o) const { return ty==o.ty && tx==o.tx; } };
                struct TileKeyHash { size_t operator()(const TileKey& k) const { return (static_cast<size_t>(k.ty)<<32) ^ static_cast<size_t>(k.tx); } };
                std::unordered_map<TileKey, torch::Tensor, TileKeyHash> local_tile_cache;

                auto clamp = [](long v, long lo, long hi){ return std::max(lo, std::min(v, hi)); };

                auto get_tile = [&](long ty, long tx)->torch::Tensor {
                    TileKey key{ty, tx};
                    if (!global_tile_cache) {
                        auto it = local_tile_cache.find(key);
                        if (it != local_tile_cache.end()) return it->second;
                    } else {
                        // Try RealSmartCache
                        auto& rcache = torchfits_real_cache::RealSmartCache::get_instance();
                        std::ostringstream k;
                        k << "tile:" << filename << ":hdu=" << hdu_num << ":ty=" << ty << ":tx=" << tx;
                        // include dtype to avoid mismatches
                        int bitpix=0; int stt=0; fits_get_img_type(f.get(), &bitpix, &stt);
                        k << ":bitpix=" << bitpix << ":dev=" << device_str.cast<std::string>();
                        auto hit = rcache.try_get(k.str());
                        if (hit.has_value()) return hit.value();
                    }
                    long y0 = ty * ztile2; // FITS order: naxes = [NAXIS1(x), NAXIS2(y)], but our start uses [y,x]
                    long x0 = tx * ztile1;
                    long h = std::min<long>(ztile2, naxes[1] - y0);
                    long w = std::min<long>(ztile1, naxes[0] - x0);
                    std::vector<long> tstart{y0, x0};
                    std::vector<long> tshape{h, w};
                    torch::Tensor t = read_image_data(f.get(), torch::kCPU, tstart, tshape);
                    if (!global_tile_cache) {
                        local_tile_cache.emplace(key, t);
                    } else {
                        auto& rcache = torchfits_real_cache::RealSmartCache::get_instance();
                        std::ostringstream k;
                        k << "tile:" << filename << ":hdu=" << hdu_num << ":ty=" << ty << ":tx=" << tx;
                        int bitpix=0; int stt=0; fits_get_img_type(f.get(), &bitpix, &stt);
                        k << ":bitpix=" << bitpix << ":dev=" << device_str.cast<std::string>();
                        rcache.put(k.str(), t);
                    }
                    return t;
                };

                // Produce outputs by stitching overlapping tiles
                for (const auto& st : starts) {
                    long sy = st[0]; long sx = st[1]; long ch = shape[0]; long cw = shape[1];
                    long ey = sy + ch; long ex = sx + cw;
                    // Compute tile span
                    long ty0 = sy / ztile2; long ty1 = (ey-1) / ztile2;
                    long tx0 = sx / ztile1; long tx1 = (ex-1) / ztile1;
                    // Allocate destination with dtype of first tile
                    torch::Tensor dst;
                    bool dst_init = false;
                    for (long ty = ty0; ty <= ty1; ++ty) {
                        for (long tx = tx0; tx <= tx1; ++tx) {
                            torch::Tensor tile = get_tile(ty, tx);
                            if (!dst_init) { dst = torch::empty({ch, cw}, tile.options()); dst_init = true; }
                            long y0 = ty * ztile2; long x0 = tx * ztile1;
                            // Overlap rect in global coords
                            long oy0 = clamp(sy, y0, y0 + tile.size(0));
                            long oy1 = clamp(ey, y0, y0 + tile.size(0));
                            long ox0 = clamp(sx, x0, x0 + tile.size(1));
                            long ox1 = clamp(ex, x0, x0 + tile.size(1));
                            if (oy1 <= oy0 || ox1 <= ox0) continue;
                            // Map to tile-local and dst-local coords
                            long ly0 = oy0 - y0; long ly1 = oy1 - y0;
                            long lx0 = ox0 - x0; long lx1 = ox1 - x0;
                            long dy0 = oy0 - sy; long dy1 = oy1 - sy;
                            long dx0 = ox0 - sx; long dx1 = ox1 - sx;
                            // Slice-copy
                            torch::Tensor src_view = tile.narrow(0, ly0, ly1 - ly0).narrow(1, lx0, lx1 - lx0);
                            torch::Tensor dst_view = dst.narrow(0, dy0, dy1 - dy0).narrow(1, dx0, dx1 - dx0);
                            dst_view.copy_(src_view);
                        }
                    }
                    if (device != torch::kCPU) dst = dst.to(device);
                    out.append(py::cast(dst));
                }
                return out;
            }
        }

        // Fallback: independent reads per cutout
        for (const auto& st : starts) {
            torch::Tensor t = read_image_data(f.get(), device, st, shape);
            out.append(py::cast(t));
        }
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

// Batched small-cutout reader across multiple HDUs (MEF optimization)
pybind11::object read_many_cutouts_multi_hdu(
    pybind11::object filename_or_url,
    const std::vector<int>& hdus,
    const std::vector<std::vector<long>>& starts,
    const std::vector<long>& shape,
    pybind11::str device_str
) {
    torch::Device device(device_str);
    try {
        if (hdus.size() != starts.size()) {
            throw std::runtime_error("read_many_cutouts_multi_hdu: hdus and starts must have same length");
        }
        std::string filename_or_url_str = py::str(filename_or_url).cast<std::string>();
        std::string filename = RemoteFetcher::ensure_local(filename_or_url_str);
        FITSFileWrapper f(filename);
        int status = 0;
        // Group indices by HDU
        std::unordered_map<int, std::vector<size_t>> by_hdu;
        by_hdu.reserve(hdus.size());
        for (size_t i=0; i<hdus.size(); ++i) by_hdu[hdus[i]].push_back(i);
        py::list out; out.attr("extend")(py::list()); // ensure list type
        out = py::list();
        // Process each HDU: optionally coalesce, else tile-aware for compressed
        for (auto &kv : by_hdu) {
            int hdu_num = kv.first;
            const auto &idxs = kv.second;
            if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
                throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
            }
            int hdu_type = 0; status = 0; fits_get_hdu_type(f.get(), &hdu_type, &status);
            if (status) throw_fits_error(status, "Error getting HDU type");
            if (hdu_type != IMAGE_HDU) {
                throw std::runtime_error("read_many_cutouts_multi_hdu: non-image HDU encountered");
            }
            // Prepare per-HDU vectors
            std::vector<std::vector<long>> starts_hdu; starts_hdu.reserve(idxs.size());
            for (size_t j : idxs) starts_hdu.push_back(starts[j]);
            // Try coalescing within HDU
            bool can_coalesce = !starts_hdu.empty() && !shape.empty();
            if (const char* env = std::getenv("TORCHFITS_CUTOUT_COALESCE")) {
                if (std::string(env) == "0") can_coalesce = false;
            }
            py::list out_hdu;
            if (can_coalesce) {
                std::vector<long> min_start(shape.size(), std::numeric_limits<long>::max());
                std::vector<long> max_end(shape.size(), std::numeric_limits<long>::min());
                for (const auto& st : starts_hdu) {
                    for (size_t i = 0; i < shape.size(); ++i) {
                        min_start[i] = std::min(min_start[i], st[i]);
                        max_end[i]   = std::max(max_end[i],   st[i] + shape[i]);
                    }
                }
                std::vector<long> bbox_shape(shape.size());
                long long bbox_pixels = 1; long long per_cutout_pixels = 1;
                for (size_t i = 0; i < shape.size(); ++i) {
                    bbox_shape[i] = std::max<long>(0, max_end[i] - min_start[i]);
                    bbox_pixels *= static_cast<long long>(bbox_shape[i]);
                    per_cutout_pixels *= static_cast<long long>(shape[i]);
                }
                double max_excess = 2.0; if (const char* env = std::getenv("TORCHFITS_CUTOUT_COALESCE_MAX_EXCESS")) { try { max_excess = std::stod(env);} catch(...){} }
                if (per_cutout_pixels > 0 && static_cast<double>(bbox_pixels) <= max_excess * static_cast<double>(per_cutout_pixels * (long long)starts_hdu.size())) {
                    torch::Tensor bbox = read_image_data(f.get(), device, min_start, bbox_shape);
                    for (const auto& st : starts_hdu) {
                        std::vector<long> local_start(shape.size());
                        for (size_t i = 0; i < shape.size(); ++i) local_start[i] = st[i] - min_start[i];
                        torch::Tensor view = bbox;
                        for (size_t i = 0; i < shape.size(); ++i) {
                            size_t rev = shape.size() - 1 - i;
                            view = view.narrow(static_cast<int64_t>(rev), static_cast<int64_t>(local_start[i]), static_cast<int64_t>(shape[i]));
                        }
                        out_hdu.append(py::cast(view));
                    }
                } else {
                    can_coalesce = false;
                }
            }
            if (!can_coalesce) {
                // Tile-aware path mirrors single-HDU implementation
                int st_comp = 0; int is_comp = fits_is_compressed_image(f.get(), &st_comp); st_comp = 0;
                if (is_comp) {
                    int naxis = 0; int st_na=0; fits_get_img_dim(f.get(), &naxis, &st_na); if (st_na) throw_fits_error(st_na, "Error getting image dimension");
                    if (naxis != 2) throw std::runtime_error("read_many_cutouts_multi_hdu: only 2D images supported for compressed tile path");
                    std::vector<long> naxes(2); int st_sz=0; fits_get_img_size(f.get(), 2, naxes.data(), &st_sz); if (st_sz) throw_fits_error(st_sz, "Error getting image size");
                    long ztile1 = naxes[0]; long ztile2 = naxes[1];
                    { int st_k=0; long v=0; if (fits_read_key_lng(f.get(), "ZTILE1", &v, nullptr, &st_k) == 0 && v>0) ztile1=v; st_k=0; if (fits_read_key_lng(f.get(), "ZTILE2", &v, nullptr, &st_k) == 0 && v>0) ztile2=v; }
                    bool global_tile_cache = true; if (const char* env = std::getenv("TORCHFITS_TILE_CACHE")) { if (std::string(env) == "0") global_tile_cache = false; }
                    struct TileKey { long ty; long tx; bool operator==(const TileKey& o) const { return ty==o.ty && tx==o.tx; } };
                    struct TileKeyHash { size_t operator()(const TileKey& k) const { return (static_cast<size_t>(k.ty)<<32) ^ static_cast<size_t>(k.tx); } };
                    std::unordered_map<TileKey, torch::Tensor, TileKeyHash> local_tile_cache;
                    auto clamp = [](long v, long lo, long hi){ return std::max(lo, std::min(v, hi)); };
                    auto get_tile = [&](long ty, long tx)->torch::Tensor {
                        TileKey key{ty, tx};
                        if (!global_tile_cache) {
                            auto it = local_tile_cache.find(key); if (it != local_tile_cache.end()) return it->second;
                        } else {
                            auto& rcache = torchfits_real_cache::RealSmartCache::get_instance();
                            std::ostringstream k; k << "tile:" << filename << ":hdu=" << hdu_num << ":ty=" << ty << ":tx=" << tx;
                            int bitpix=0; int stt=0; fits_get_img_type(f.get(), &bitpix, &stt);
                            k << ":bitpix=" << bitpix << ":dev=" << device_str.cast<std::string>();
                            auto hit = rcache.try_get(k.str()); if (hit.has_value()) return hit.value();
                        }
                        long y0 = ty * ztile2; long x0 = tx * ztile1; long h = std::min<long>(ztile2, naxes[1] - y0); long w = std::min<long>(ztile1, naxes[0] - x0);
                        std::vector<long> tstart{y0, x0}; std::vector<long> tshape{h, w}; torch::Tensor t = read_image_data(f.get(), torch::kCPU, tstart, tshape);
                        if (!global_tile_cache) {
                            local_tile_cache.emplace(key, t);
                        } else {
                            auto& rcache = torchfits_real_cache::RealSmartCache::get_instance();
                            std::ostringstream k; k << "tile:" << filename << ":hdu=" << hdu_num << ":ty=" << ty << ":tx=" << tx;
                            int bitpix=0; int stt=0; fits_get_img_type(f.get(), &bitpix, &stt);
                            k << ":bitpix=" << bitpix << ":dev=" << device_str.cast<std::string>();
                            rcache.put(k.str(), t);
                        }
                        return t; };
                    for (const auto& stv : starts_hdu) {
                        long sy = stv[0]; long sx = stv[1]; long ch = shape[0]; long cw = shape[1]; long ey = sy + ch; long ex = sx + cw;
                        long ty0 = sy / ztile2; long ty1 = (ey-1) / ztile2; long tx0 = sx / ztile1; long tx1 = (ex-1) / ztile1;
                        torch::Tensor dst; bool dst_init=false;
                        for (long ty = ty0; ty <= ty1; ++ty) {
                            for (long tx = tx0; tx <= tx1; ++tx) {
                                torch::Tensor tile = get_tile(ty, tx);
                                if (!dst_init) { dst = torch::empty({ch, cw}, tile.options()); dst_init = true; }
                                long y0 = ty*ztile2; long x0 = tx*ztile1;
                                long oy0 = clamp(sy, y0, y0 + tile.size(0)); long oy1 = clamp(ey, y0, y0 + tile.size(0));
                                long ox0 = clamp(sx, x0, x0 + tile.size(1)); long ox1 = clamp(ex, x0, x0 + tile.size(1));
                                if (oy1 <= oy0 || ox1 <= ox0) continue;
                                long ly0 = oy0 - y0; long ly1 = oy1 - y0; long lx0 = ox0 - x0; long lx1 = ox1 - x0;
                                long dy0 = oy0 - sy; long dy1 = oy1 - sy; long dx0 = ox0 - sx; long dx1 = ox1 - sx;
                                torch::Tensor src_view = tile.narrow(0, ly0, ly1 - ly0).narrow(1, lx0, lx1 - lx0);
                                torch::Tensor dst_view = dst.narrow(0, dy0, dy1 - dy0).narrow(1, dx0, dx1 - dx0);
                                dst_view.copy_(src_view);
                            }
                        }
                        if (device != torch::kCPU) dst = dst.to(device);
                        out_hdu.append(py::cast(dst));
                    }
                } else {
                    for (const auto& stv : starts_hdu) {
                        torch::Tensor t = read_image_data(f.get(), device, stv, shape);
                        out_hdu.append(py::cast(t));
                    }
                }
            }
            // Append in the original input order: collect tensors and assign
            std::vector<py::object> collected;
            collected.reserve(starts_hdu.size());
            for (py::handle item : out_hdu) {
                collected.emplace_back(py::reinterpret_borrow<py::object>(item));
            }
            // Map back to global order
            size_t k = 0;
            for (size_t j : idxs) {
                (void)j;
                out.append(collected[k++]);
            }
        }
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}