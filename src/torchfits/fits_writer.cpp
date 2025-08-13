#include "fits_writer.h"
#include "fits_utils.h"
#include "debug.h" // retained for build compatibility; no debug macros used below
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include "cache.h"  // For clear_cache to invalidate after updates
#include "real_cache.h"  // For real smart cache invalidation
#include <torch/extension.h> // Ensure pybind11 type casters for torch::Tensor

namespace torchfits_writer {

void append_table_to_fits(const std::string& filename,
                         const py::dict& table_data,
                         const std::map<std::string, std::string>& header,
                         const std::vector<std::string>& column_units,
                         const std::vector<std::string>& column_descriptions,
                         const std::map<std::string, long>& null_sentinels) {
    if (table_data.size() == 0) {
        throw std::runtime_error("Cannot append empty table to FITS file");
    }

    fitsfile* fptr;
    int status = 0;

    fits_open_file(&fptr, filename.c_str(), READWRITE, &status);
    if (status) throw_fits_error(status, "Error opening FITS file for append: " + filename);

    try {
        // Move to last HDU so that the new table is appended at the end.
        int nhdus = 0;
        fits_get_num_hdus(fptr, &nhdus, &status);
        if (status) throw_fits_error(status, "Error getting number of HDUs before append");
        // Move to last existing HDU (nhdus). The subsequent fits_create_tbl will append a new HDU after it.
        fits_movabs_hdu(fptr, nhdus, nullptr, &status);
        if (status) throw_fits_error(status, "Error moving to last HDU before appending table");

        // Get column information (copied from write_table_to_fits)
        std::vector<std::string> column_names;
        std::vector<std::string> column_formats;
        std::vector<torch::Tensor> column_tensors;
        std::vector<bool> is_string_column;
        std::vector<std::vector<std::string>> string_columns;
        long nrows = 0;

        for (auto item : table_data) {
            std::string col_name = py::str(item.first).cast<std::string>();
            py::object value_obj = py::reinterpret_borrow<py::object>(item.second);

            bool is_tensor = false;
            try {
                if (py::isinstance<torch::Tensor>(value_obj)) {
                    is_tensor = true;
                } else if (py::hasattr(value_obj, "__class__") && py::str(value_obj.attr("__class__").attr("__name__")).cast<std::string>() == "Tensor") {
                    is_tensor = true;
                }
            } catch (...) {}

            if (is_tensor) {
                torch::Tensor tensor = value_obj.cast<torch::Tensor>();
                column_names.push_back(col_name);
                column_formats.push_back(get_fits_column_format(tensor));
                column_tensors.push_back(tensor.contiguous().cpu());
                is_string_column.push_back(false);
                string_columns.emplace_back();
                if (nrows == 0) {
                    nrows = tensor.size(0);
                } else if (tensor.size(0) != nrows) {
                    throw std::runtime_error("All table columns must have the same number of rows");
                }
            } else if (py::isinstance<py::list>(value_obj) || py::isinstance<py::tuple>(value_obj)) {
                py::sequence seq = value_obj.cast<py::sequence>();
                std::vector<std::string> strs;
                strs.reserve(seq.size());
                for (auto s : seq) {
                    if (!py::isinstance<py::str>(s)) {
                        throw std::runtime_error("Only list[str] supported for non-tensor column data");
                    }
                    strs.push_back(py::str(s).cast<std::string>());
                }
                if (strs.empty()) {
                    throw std::runtime_error("String column cannot be empty");
                }
                if (nrows == 0) nrows = strs.size();
                else if ((long)strs.size() != nrows) {
                    throw std::runtime_error("All table columns must have the same number of rows");
                }
                size_t maxlen = 1;
                for (auto &s : strs) maxlen = std::max(maxlen, s.size());
                column_names.push_back(col_name);
                column_formats.push_back(std::to_string(maxlen) + "A");
                column_tensors.emplace_back();
                is_string_column.push_back(true);
                string_columns.push_back(std::move(strs));
            } else {
                throw std::runtime_error("Unsupported column data type for writing (expected tensor or list[str])");
            }
        }

        int ncols = column_names.size();
        std::vector<char*> ttype(ncols);
        std::vector<char*> tform(ncols);
        for (int i = 0; i < ncols; ++i) {
            ttype[i] = const_cast<char*>(column_names[i].c_str());
            tform[i] = const_cast<char*>(column_formats[i].c_str());
        }

        fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype.data(), tform.data(), nullptr, nullptr, &status);
        if (status) throw_fits_error(status, "Error creating binary table");

        for (int i = 0; i < ncols; ++i) {
            if (is_string_column[i]) {
                std::vector<char*> cstrs;
                cstrs.reserve(nrows);
                for (auto &s : string_columns[i]) cstrs.push_back(const_cast<char*>(s.c_str()));
                fits_write_col(fptr, TSTRING, i + 1, 1, 1, nrows, cstrs.data(), &status);
                if (status) throw_fits_error(status, "Error writing string column " + column_names[i]);
            } else {
                torch::Tensor& tensor = column_tensors[i];
                int datatype = get_fits_table_datatype(tensor);
                fits_write_col(fptr, datatype, i + 1, 1, 1, nrows, tensor.data_ptr(), &status);
                if (status) throw_fits_error(status, "Error writing column " + column_names[i]);
            }
            if (i < (int)column_units.size() && !column_units[i].empty()) {
                std::string unit_key = "TUNIT" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, unit_key.c_str(), const_cast<char*>(column_units[i].c_str()), nullptr, &status);
            }
            if (i < (int)column_descriptions.size() && !column_descriptions[i].empty()) {
                std::string comm_key = "TCOMM" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, comm_key.c_str(), const_cast<char*>(column_descriptions[i].c_str()), nullptr, &status);
            }
            auto it = null_sentinels.find(column_names[i]);
            if (it != null_sentinels.end() && !is_string_column[i]) {
                std::string tnull_key = "TNULL" + std::to_string(i + 1);
                long sentinel = it->second;
                fits_write_key(fptr, TLONG, tnull_key.c_str(), &sentinel, nullptr, &status);
            }
        }

        write_header_keywords(fptr, header);

        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");

    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}
int get_fits_datatype(torch::Tensor tensor) {
    switch (tensor.dtype().toScalarType()) {
        case torch::kUInt8:   return BYTE_IMG;
        case torch::kInt16:   return SHORT_IMG;
        case torch::kInt32:   return LONG_IMG;
        case torch::kInt64:   return LONGLONG_IMG;
        case torch::kFloat32: return FLOAT_IMG;
        case torch::kFloat64: return DOUBLE_IMG;
        default:
            throw std::runtime_error("Unsupported tensor dtype for FITS writing: " + 
                                   torch::toString(tensor.dtype()));
    }
}

int get_fits_table_datatype(torch::Tensor tensor) {
    switch (tensor.dtype().toScalarType()) {
        case torch::kUInt8:   return TBYTE;
        case torch::kInt16:   return TSHORT;
        case torch::kInt32:   return TINT;
        case torch::kInt64:   return TLONGLONG;
        case torch::kFloat32: return TFLOAT;
        case torch::kFloat64: return TDOUBLE;
        case torch::kBool:    return TLOGICAL;
        default:
            throw std::runtime_error("Unsupported tensor dtype for FITS table writing: " + 
                                   torch::toString(tensor.dtype()));
    }
}

std::string get_fits_column_format(torch::Tensor tensor) {
    switch (tensor.dtype().toScalarType()) {
        case torch::kUInt8:   return "1B";
        case torch::kInt16:   return "1I";
        case torch::kInt32:   return "1J";
        case torch::kInt64:   return "1K";
        case torch::kFloat32: return "1E";
        case torch::kFloat64: return "1D";
        case torch::kBool:    return "1L";
        default:
            throw std::runtime_error("Unsupported tensor dtype for FITS column format: " + 
                                   torch::toString(tensor.dtype()));
    }
}

void write_header_keywords(fitsfile* fptr, const std::map<std::string, std::string>& header) {
    int status = 0;
    
    for (const auto& pair : header) {
        const std::string& key = pair.first;
        const std::string& value = pair.second;
        
        // Skip standard FITS keywords that are automatically handled
        if (key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" || 
            key.substr(0, 5) == "NAXIS" || key == "EXTEND") {
            continue;
        }
        
        // Try to determine if value is numeric or string
        char* endptr;
        double numeric_value = std::strtod(value.c_str(), &endptr);
        
        if (*endptr == '\0') {
            // It's a number
            if (value.find('.') != std::string::npos) {
                // Float
                fits_write_key(fptr, TDOUBLE, key.c_str(), &numeric_value, nullptr, &status);
            } else {
                // Integer
                long int_value = static_cast<long>(numeric_value);
                fits_write_key(fptr, TLONG, key.c_str(), &int_value, nullptr, &status);
            }
        } else {
            // It's a string
            fits_write_key(fptr, TSTRING, key.c_str(), 
                          const_cast<char*>(value.c_str()), nullptr, &status);
        }
        
        if (status) {
            // Silently continue on header keyword write failure (non-fatal)
            status = 0; // Continue with other keywords
        }
    }
}

void tensor_to_fits_image(fitsfile* fptr, torch::Tensor tensor) {
    int status = 0;
    
    // Ensure tensor is contiguous and on CPU
    tensor = tensor.contiguous().cpu();
    
    // Get tensor dimensions; reverse to match FITS axis ordering expected by CFITSIO (slowest varying first)
    auto sizes = tensor.sizes();
    std::vector<long> naxes(sizes.begin(), sizes.end());
    std::reverse(naxes.begin(), naxes.end());
    
    int bitpix = get_fits_datatype(tensor);
    
    // Create image HDU
    fits_create_img(fptr, bitpix, naxes.size(), naxes.data(), &status);
    if (status) throw_fits_error(status, "Error creating FITS image");
    
    // Write image data
    long nelements = tensor.numel();
    int datatype = get_fits_table_datatype(tensor);
    
    fits_write_img(fptr, datatype, 1, nelements, tensor.data_ptr(), &status);
    if (status) throw_fits_error(status, "Error writing image data to FITS");
    
    // Image write complete
}

void write_tensor_to_fits(const std::string& filename,
                         torch::Tensor data,
                         const std::map<std::string, std::string>& header,
                         bool overwrite) {
    // Writing single tensor FITS file
    
    fitsfile* fptr;
    int status = 0;
    
    std::string create_filename = filename;
    if (overwrite) {
        create_filename = "!" + filename; // CFITSIO overwrite syntax
    }
    
    fits_create_file(&fptr, create_filename.c_str(), &status);
    if (status) throw_fits_error(status, "Error creating FITS file: " + filename);
    
    try {
        // Write the primary HDU
        tensor_to_fits_image(fptr, data);
        
        // Write header keywords
        write_header_keywords(fptr, header);
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // Completed writing tensor FITS file
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void write_tensors_to_mef(const std::string& filename,
                         const std::vector<torch::Tensor>& tensors,
                         const std::vector<std::map<std::string, std::string>>& headers,
                         const std::vector<std::string>& extnames,
                         bool overwrite) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot write empty tensor list to FITS file");
    }
    
    // Writing MEF file
    
    fitsfile* fptr;
    int status = 0;
    
    std::string create_filename = filename;
    if (overwrite) {
        create_filename = "!" + filename;
    }
    
    fits_create_file(&fptr, create_filename.c_str(), &status);
    if (status) throw_fits_error(status, "Error creating FITS file: " + filename);
    
    try {
        // Standard FITS: first tensor becomes primary image HDU
        tensor_to_fits_image(fptr, tensors[0]);
        if (!headers.empty()) {
            write_header_keywords(fptr, headers[0]);
        }
        if (!extnames.empty() && !extnames[0].empty()) {
            fits_write_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extnames[0].c_str()), nullptr, &status);
        }
    // Remaining tensors as IMAGE extensions
        for (size_t i = 1; i < tensors.size(); ++i) {
            tensor_to_fits_image(fptr, tensors[i]);
            if (i < headers.size()) {
                write_header_keywords(fptr, headers[i]);
            }
            if (i < extnames.size() && !extnames[i].empty()) {
                fits_write_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extnames[i].c_str()), nullptr, &status);
            }
        }
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // Completed MEF write
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void write_table_to_fits(const std::string& filename,
                        const py::dict& table_data,
                        const std::map<std::string, std::string>& header,
                        const std::vector<std::string>& column_units,
                        const std::vector<std::string>& column_descriptions,
                        const std::map<std::string, long>& null_sentinels,
                        bool overwrite) {
    if (table_data.size() == 0) {
        throw std::runtime_error("Cannot write empty table to FITS file");
    }
    
    // Writing table to FITS
    
    fitsfile* fptr;
    int status = 0;
    
    std::string create_filename = filename;
    if (overwrite) {
        create_filename = "!" + filename;
    }
    
    fits_create_file(&fptr, create_filename.c_str(), &status);
    if (status) throw_fits_error(status, "Error creating FITS file: " + filename);
    
    try {
        // Create primary HDU (empty)
        fits_create_img(fptr, 8, 0, nullptr, &status);
        if (status) throw_fits_error(status, "Error creating primary HDU");
        
        // Get column information
        std::vector<std::string> column_names;
    std::vector<std::string> column_formats;
    std::vector<torch::Tensor> column_tensors; // numeric/bool tensors
    std::vector<bool> is_string_column;
    std::vector<std::vector<std::string>> string_columns; // store string data
        long nrows = 0;
        
        for (auto item : table_data) {
            std::string col_name = py::str(item.first).cast<std::string>();
            py::object value_obj = py::reinterpret_borrow<py::object>(item.second);

            bool is_tensor = false;
            try {
                if (py::isinstance<torch::Tensor>(value_obj)) {
                    is_tensor = true;
                } else if (py::hasattr(value_obj, "__class__") && py::str(value_obj.attr("__class__").attr("__name__")).cast<std::string>() == "Tensor") {
                    is_tensor = true; // Fallback detection
                }
            } catch (...) {}

            if (is_tensor) {
                torch::Tensor tensor = value_obj.cast<torch::Tensor>();
                column_names.push_back(col_name);
                column_formats.push_back(get_fits_column_format(tensor));
                column_tensors.push_back(tensor.contiguous().cpu());
                is_string_column.push_back(false);
                string_columns.emplace_back();
                if (nrows == 0) {
                    nrows = tensor.size(0);
                } else if (tensor.size(0) != nrows) {
                    throw std::runtime_error("All table columns must have the same number of rows");
                }
            } else if (py::isinstance<py::list>(value_obj) || py::isinstance<py::tuple>(value_obj)) {
                // Potential string column
                py::sequence seq = value_obj.cast<py::sequence>();
                std::vector<std::string> strs;
                strs.reserve(seq.size());
                for (auto s : seq) {
                    if (!py::isinstance<py::str>(s)) {
                        throw std::runtime_error("Only list[str] supported for non-tensor column data");
                    }
                    strs.push_back(py::str(s).cast<std::string>());
                }
                if (strs.empty()) {
                    throw std::runtime_error("String column cannot be empty");
                }
                if (nrows == 0) nrows = strs.size();
                else if ((long)strs.size() != nrows) {
                    throw std::runtime_error("All table columns must have the same number of rows");
                }
                // Determine max width
                size_t maxlen = 1;
                for (auto &s : strs) maxlen = std::max(maxlen, s.size());
                column_names.push_back(col_name);
                column_formats.push_back(std::to_string(maxlen) + "A");
                column_tensors.emplace_back(); // placeholder
                is_string_column.push_back(true);
                string_columns.push_back(std::move(strs));
            } else {
                throw std::runtime_error("Unsupported column data type for writing (expected tensor or list[str])");
            }
        }
        
        // Create binary table
        int ncols = column_names.size();
        std::vector<char*> ttype(ncols);
        std::vector<char*> tform(ncols);
        
        for (int i = 0; i < ncols; ++i) {
            ttype[i] = const_cast<char*>(column_names[i].c_str());
            tform[i] = const_cast<char*>(column_formats[i].c_str());
        }
        
        fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype.data(), tform.data(), 
                       nullptr, nullptr, &status);
        if (status) throw_fits_error(status, "Error creating binary table");
        
        // Write column data
        for (int i = 0; i < ncols; ++i) {
            if (is_string_column[i]) {
                std::vector<char*> cstrs;
                cstrs.reserve(nrows);
                for (auto &s : string_columns[i]) cstrs.push_back(const_cast<char*>(s.c_str()));
#ifdef TORCHFITS_DEBUG
                fprintf(stderr, "[TORCHFITS_DEBUG] Writing string column %s (n=%ld)\n", column_names[i].c_str(), (long)nrows);
#endif
                fits_write_col(fptr, TSTRING, i + 1, 1, 1, nrows, cstrs.data(), &status);
                if (status) throw_fits_error(status, "Error writing string column " + column_names[i]);
            } else {
                torch::Tensor& tensor = column_tensors[i];
                int datatype = get_fits_table_datatype(tensor);
#ifdef TORCHFITS_DEBUG
                fprintf(stderr, "[TORCHFITS_DEBUG] Writing numeric column %s dtype=%d rows=%ld\n", column_names[i].c_str(), datatype, (long)tensor.size(0));
#endif
                fits_write_col(fptr, datatype, i + 1, 1, 1, nrows,
                               tensor.data_ptr(), &status);
                if (status) throw_fits_error(status, "Error writing column " + column_names[i]);
            }
            // Column units
            if (i < (int)column_units.size() && !column_units[i].empty()) {
                std::string unit_key = "TUNIT" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, unit_key.c_str(),
                              const_cast<char*>(column_units[i].c_str()), nullptr, &status);
            }
            if (i < (int)column_descriptions.size() && !column_descriptions[i].empty()) {
                std::string comm_key = "TCOMM" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, comm_key.c_str(),
                              const_cast<char*>(column_descriptions[i].c_str()), nullptr, &status);
            }
            // Null sentinel if provided and numeric column
            auto it = null_sentinels.find(column_names[i]);
            if (it != null_sentinels.end() && !is_string_column[i]) {
                std::string tnull_key = "TNULL" + std::to_string(i + 1);
                long sentinel = it->second;
                fits_write_key(fptr, TLONG, tnull_key.c_str(), &sentinel, nullptr, &status);
            }
        }
        
        // Write table header keywords
    write_header_keywords(fptr, header);
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // Table write complete
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void write_fits_table(const std::string& filename,
                     const py::object& fits_table,
                     bool overwrite) {
    // Extract data dictionary from FitsTable
    py::dict table_data = fits_table.attr("data").cast<py::dict>();
    
    // Extract metadata if available
    std::map<std::string, std::string> header;
    std::vector<std::string> column_units;
    std::vector<std::string> column_descriptions;
    
    if (py::hasattr(fits_table, "column_info")) {
        py::dict column_info = fits_table.attr("column_info").cast<py::dict>();
        
        for (auto item : table_data) {
            std::string col_name = py::str(item.first).cast<std::string>();
            
            if (column_info.contains(py::str(col_name))) {
                py::object col_meta = column_info[py::str(col_name)];
                
                if (py::hasattr(col_meta, "unit") && !col_meta.attr("unit").is_none()) {
                    column_units.push_back(py::str(col_meta.attr("unit")).cast<std::string>());
                } else {
                    column_units.push_back("");
                }
                
                if (py::hasattr(col_meta, "description") && !col_meta.attr("description").is_none()) {
                    column_descriptions.push_back(py::str(col_meta.attr("description")).cast<std::string>());
                } else {
                    column_descriptions.push_back("");
                }
            }
        }
    }
    
    write_table_to_fits(filename, table_data, header, column_units, column_descriptions, std::map<std::string,long>(), overwrite);
}

void append_hdu_to_fits(const std::string& filename,
                       torch::Tensor data,
                       const std::map<std::string, std::string>& header,
                       const std::string& extname) {
    // Appending new HDU
    
    fitsfile* fptr;
    int status = 0;
    
    fits_open_file(&fptr, filename.c_str(), READWRITE, &status);
    if (status) throw_fits_error(status, "Error opening FITS file for appending: " + filename);
    
    try {
        // Move to end of file
        int hdu_num;
        fits_get_num_hdus(fptr, &hdu_num, &status);
        if (status) throw_fits_error(status, "Error getting number of HDUs");
        
        fits_movabs_hdu(fptr, hdu_num, nullptr, &status);
        if (status) throw_fits_error(status, "Error moving to last HDU");
        
        // Create new image extension
        tensor_to_fits_image(fptr, data);
        
        // Write header keywords
        write_header_keywords(fptr, header);
        
        // Add EXTNAME if provided
        if (!extname.empty()) {
            fits_write_key(fptr, TSTRING, "EXTNAME", 
                          const_cast<char*>(extname.c_str()), nullptr, &status);
        }
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // HDU append complete
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void update_fits_header(const std::string& filename,
                       int hdu_num,
                       const std::map<std::string, std::string>& updates) {
    // Updating header keywords
    
    fitsfile* fptr;
    int status = 0;
    
    fits_open_file(&fptr, filename.c_str(), READWRITE, &status);
    if (status) throw_fits_error(status, "Error opening FITS file for header update: " + filename);
    
    try {
        fits_movabs_hdu(fptr, hdu_num, nullptr, &status);
        if (status) throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        
        write_header_keywords(fptr, updates);
        
    fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // Header update complete
    // Invalidate any cached entries referencing this file
    try { clear_cache(); } catch (...) {}
    try { torchfits_real_cache::RealSmartCache::get_instance().clear(); } catch (...) {}
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void update_fits_data(const std::string& filename,
                     int hdu_num,
                     torch::Tensor new_data,
                     const std::vector<long>& start,
                     const std::vector<long>& shape) {
    // Updating data section
    
    fitsfile* fptr;
    int status = 0;
    
    fits_open_file(&fptr, filename.c_str(), READWRITE, &status);
    if (status) throw_fits_error(status, "Error opening FITS file for data update: " + filename);
    
    try {
        fits_movabs_hdu(fptr, hdu_num, nullptr, &status);
        if (status) throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        
        new_data = new_data.contiguous().cpu();
        int datatype = get_fits_table_datatype(new_data);
        
        if (start.empty()) {
            // Update entire image
            fits_write_img(fptr, datatype, 1, new_data.numel(), new_data.data_ptr(), &status);
            if (status) throw_fits_error(status, "Error updating entire image data");
        } else {
            // Update subset
            int naxis = start.size();
            std::vector<long> fpixel(naxis), lpixel(naxis), inc(naxis, 1);
            
            for (int i = 0; i < naxis; ++i) {
                fpixel[i] = start[i] + 1;  // FITS is 1-based
                if (i < shape.size()) {
                    lpixel[i] = start[i] + shape[i];
                } else {
                    lpixel[i] = start[i] + new_data.size(i);
                }
            }
            
            fits_write_subset(fptr, datatype, fpixel.data(), lpixel.data(), 
                             new_data.data_ptr(), &status);
            if (status) throw_fits_error(status, "Error updating image subset");
        }
        
    fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
    // Data update complete
    // Invalidate cache after pixel modification
    try { clear_cache(); } catch (...) {}
    try { torchfits_real_cache::RealSmartCache::get_instance().clear(); } catch (...) {}
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

// Close primary implementation section and re-open for stubs to avoid interleaving
} // namespace torchfits_writer

namespace torchfits_writer {
// --- Phase 2 advanced features currently disabled: provide stubs inside namespace ---
int get_fits_compression_type(CompressionType) { return 0; }

static std::string build_compress_suffix(const CompressionConfig& compression,
                                         const torch::Tensor& data) {
    // Map CompressionConfig to CFITSIO extended filename suffix.
    // Grammar examples:
    //   [compress]                     -> default (Rice, row-tiles)
    //   [compress RICE]                -> explicit algorithm
    //   [compress GZIP 100,100]        -> explicit tile dims
    //   [compress R 256,256;2]         -> noisebits = 2
    //   [compress HCOMPRESS]           -> HCOMPRESS (scale/smooth defaulted by CFITSIO)
    std::string algo;
    switch (compression.type) {
        case CompressionType::RICE: algo = "RICE"; break;
        case CompressionType::GZIP: algo = "GZIP"; break;
        case CompressionType::HCOMPRESS: algo = "HCOMPRESS"; break;
        case CompressionType::PLIO: algo = "PLIO"; break;
        case CompressionType::None: default: return std::string();
    }
    std::ostringstream oss;
    oss << "[compress ";
    // CFITSIO accepts full names; use short letter for Rice if desired, but names are clearer
    oss << algo;
    // Tile dimensions if provided
    int tx = compression.tile_dimensions[0];
    int ty = compression.tile_dimensions[1];
    if (tx > 0 && ty > 0) {
        oss << ' ' << tx << ',' << ty;
    }
    // Noise bits for floating types only (ignored for integer images)
    bool is_float = (data.scalar_type() == torch::kFloat32) || (data.scalar_type() == torch::kFloat64);
    if (is_float) {
        // quantize_level in our config maps to CFITSIO noisebits; default 0 means lossless-like
        if (compression.quantize_level > 0) {
            oss << ';' << compression.quantize_level;
        }
    }
    oss << ']';
    return oss.str();
}

void write_tensor_to_fits_advanced(const std::string& filename,
                                  torch::Tensor data,
                                  const std::map<std::string, std::string>& header,
                                  const CompressionConfig& compression,
                                  bool overwrite,
                                  bool checksum) {
    // If no compression requested, delegate to basic writer.
    if (compression.type == CompressionType::None) {
        write_tensor_to_fits(filename, data, header, overwrite);
        return;
    }

    // Build extended filename with compression request.
    std::string suffix = build_compress_suffix(compression, data);
    // If mapping failed, fallback to basic for safety
    if (suffix.empty()) {
        write_tensor_to_fits(filename, data, header, overwrite);
        return;
    }

    // Create compressed image using CFITSIO extended filename syntax.
    fitsfile* fptr = nullptr;
    int status = 0;
    std::string fname = filename;
    if (overwrite) {
        // CFITSIO overwrite handled by leading '!'; do not double-remove to avoid ENOENT
        if (!fname.empty() && fname[0] == '!') fname = fname.substr(1);
    }
    std::string create_name = (overwrite ? "!" : std::string()) + fname + suffix;

    if (fits_create_file(&fptr, create_name.c_str(), &status)) {
        throw_fits_error(status, "Error creating compressed FITS file: " + filename);
    }

    try {
        // Write the image (CFITSIO will produce a tile-compressed image container)
        tensor_to_fits_image(fptr, data);
        // Write header keywords
        write_header_keywords(fptr, header);
        // Optional checksum
        if (checksum) {
            int s2 = 0;
            ffpcks(fptr, &s2);
            // ignore checksum failure; non-fatal
        }
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing compressed FITS file");
    } catch (const std::exception&) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void write_variable_length_array(const std::string& filename,
                                const std::vector<torch::Tensor>& arrays,
                                const std::map<std::string, std::string>& header,
                                bool overwrite) {
    int status = 0;
    fitsfile* fptr = nullptr;
    std::string fname = filename;
    if (overwrite) std::remove(fname.c_str());
    std::string create_name = (fname.size() && fname[0] == '!') ? fname : ("!" + fname);
    if (fits_create_file(&fptr, create_name.c_str(), &status)) {
        throw_fits_error(status, "Error creating FITS file for variable length array");
    }
    // Primary HDU (empty image)
    if (fits_create_img(fptr, BYTE_IMG, 0, nullptr, &status)) {
        fits_close_file(fptr, &status);
        throw_fits_error(status, "Error creating primary HDU for variable length array");
    }
    // Table definition: single variable-length double column
    // Determine number of rows and maximum element length
    long nrows = arrays.size();
    long max_len = 0;
    for (const auto &t : arrays) {
        if (!(t.dtype() == torch::kFloat64 || t.dtype() == torch::kFloat32)) {
            fits_close_file(fptr, &status);
            throw std::runtime_error("Only float32/float64 tensors supported for variable length arrays");
        }
        max_len = std::max<long>(max_len, t.numel());
    }
    char *ttype[1];
    char *tform[1];
    std::string tname = "ARRAY_DATA";
    // CFITSIO variable-length column format: 'rPt(max)' where r is number of elements per cell (always 1 for scalar descriptors),
    // 'P' indicates variable-length, 'D' is datatype, and (max) optional maximum length hint.
    std::string tform_str = "1PD(" + std::to_string(max_len) + ")";
    ttype[0] = const_cast<char*>(tname.c_str());
    tform[0] = const_cast<char*>(tform_str.c_str());
    if (fits_create_tbl(fptr, BINARY_TBL, nrows, 1, ttype, tform, nullptr, nullptr, &status)) {
        fits_close_file(fptr, &status);
        throw_fits_error(status, "Error creating variable length array table");
    }
    // Write header keywords
    for (auto &kv : header) {
        fits_write_key(fptr, TSTRING, kv.first.c_str(), const_cast<char*>(kv.second.c_str()), nullptr, &status);
        if (status) {
            fits_close_file(fptr, &status);
            throw_fits_error(status, "Error writing header keyword");
        }
    }
    // Iterate rows writing data; CFITSIO creates descriptors automatically
    for (long row = 0; row < nrows; ++row) {
        const auto &tensor = arrays[row];
        torch::Tensor dbl = tensor.dtype() == torch::kFloat64 ? tensor : tensor.to(torch::kFloat64);
        long nelem = dbl.numel();
        if (fits_write_col(fptr, TDOUBLE, 1, row + 1, 1, nelem, (void*)dbl.data_ptr<double>(), &status)) {
            fits_close_file(fptr, &status);
            throw_fits_error(status, "Error writing variable array data");
        }
    // Diagnostic: read back descriptor to verify length/offset
    long repeat=0; long width=0; int typecode=0; int tmp_status=0;
    fits_binary_tform(const_cast<char*>(tform_str.c_str()), &typecode, &repeat, &width, &tmp_status);
    LONGLONG llength=0; LONGLONG offset=0; int d_status=0;
    fits_read_descriptll(fptr, 1, row+1, &llength, &offset, &d_status);
    fprintf(stderr, "[VARLEN DEBUG] row %ld wrote nelem=%ld descriptor length=%lld offset=%lld typecode=%d repeat=%ld width=%ld status=%d d_status=%d TFORM='%s'\n", row, nelem, llength, offset, typecode, repeat, width, tmp_status, d_status, tform_str.c_str());
    }
    fits_close_file(fptr, &status);
    if (status) throw_fits_error(status, "Error closing variable length array file");
}

StreamingWriter::StreamingWriter(const std::string& filename,
                               const std::vector<long>& dimensions,
                               torch::ScalarType dtype,
                               const CompressionConfig& compression,
                               bool overwrite)
  : filename_(filename), dimensions_(dimensions), dtype_(dtype), compression_(compression), current_position_(0), fptr_(nullptr), finalized_(true) {
    throw std::runtime_error("StreamingWriter temporarily disabled (namespace cleaned)");
}

StreamingWriter::~StreamingWriter() {}

void StreamingWriter::write_chunk(const torch::Tensor&, const std::vector<long>&) {
    throw std::runtime_error("StreamingWriter::write_chunk disabled (namespace cleaned)");
}

void StreamingWriter::write_sequential(const torch::Tensor&) {
    throw std::runtime_error("StreamingWriter::write_sequential disabled (namespace cleaned)");
}

void StreamingWriter::finalize(const std::map<std::string, std::string>&) {}

void write_variable_length_table(
    const std::string& filename,
    const std::map<std::string, std::vector<torch::Tensor>>& columns,
    const std::map<std::string, std::string>& header,
    bool overwrite) {
    if (columns.empty()) {
        throw std::invalid_argument("write_variable_length_table: columns must not be empty");
    }
    // Validate consistent row count
    long nrows = -1;
    long max_len_hint = 0;
    for (const auto& kv : columns) {
        const auto& name = kv.first;
        const auto& col = kv.second;
        if (nrows < 0) nrows = static_cast<long>(col.size());
        if (static_cast<long>(col.size()) != nrows) {
            throw std::invalid_argument("All VLA columns must have the same number of rows");
        }
        for (const auto& t : col) {
            if (!(t.dtype() == torch::kFloat64 || t.dtype() == torch::kFloat32)) {
                throw std::runtime_error("Only float32/float64 tensors supported for VLA columns");
            }
            max_len_hint = std::max<long>(max_len_hint, static_cast<long>(t.numel()));
        }
    }

    int status = 0;
    fitsfile* fptr = nullptr;
    std::string fname = filename;
    if (overwrite) std::remove(fname.c_str());
    std::string create_name = (fname.size() && fname[0] == '!') ? fname : ("!" + fname);
    if (fits_create_file(&fptr, create_name.c_str(), &status)) {
        throw_fits_error(status, "Error creating FITS file for VLA table");
    }
    if (fits_create_img(fptr, BYTE_IMG, 0, nullptr, &status)) {
        fits_close_file(fptr, &status);
        throw_fits_error(status, "Error creating primary HDU for VLA table");
    }
    // Prepare TTYPE/TFORM arrays
    const int ncols = static_cast<int>(columns.size());
    std::vector<char*> ttype(ncols);
    std::vector<char*> tform(ncols);
    std::vector<std::string> ttype_buf; ttype_buf.reserve(ncols);
    std::vector<std::string> tform_buf; tform_buf.reserve(ncols);
    std::vector<std::string> names; names.reserve(ncols);
    for (const auto& kv : columns) {
        std::string cname = kv.first;
        if (cname.empty()) cname = "COL";
        names.push_back(cname);
    }
    for (int i = 0; i < ncols; ++i) {
        ttype_buf.push_back(names[i]);
        ttype[i] = const_cast<char*>(ttype_buf.back().c_str());
        // Use double base type, with max length hint
        std::string tform_str = "1PD(" + std::to_string(std::max<long>(1, max_len_hint)) + ")";
        tform_buf.push_back(tform_str);
        tform[i] = const_cast<char*>(tform_buf.back().c_str());
    }
    if (fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype.data(), tform.data(), nullptr, nullptr, &status)) {
        fits_close_file(fptr, &status);
        throw_fits_error(status, "Error creating VLA table");
    }
    // Write header keywords
    for (auto &kv : header) {
        fits_write_key(fptr, TSTRING, kv.first.c_str(), const_cast<char*>(kv.second.c_str()), nullptr, &status);
        if (status) { fits_close_file(fptr, &status); throw_fits_error(status, "Error writing header keyword"); }
    }
    // Iterate rows and write each column's data for that row
    int col_index = 1;
    for (const auto& kv : columns) {
        const auto& col = kv.second;
        for (long row = 0; row < nrows; ++row) {
            const auto& tensor = col[row];
            torch::Tensor dbl = tensor.dtype() == torch::kFloat64 ? tensor : tensor.to(torch::kFloat64);
            long nelem = static_cast<long>(dbl.numel());
            if (nelem > 0) {
                if (fits_write_col(fptr, TDOUBLE, col_index, row + 1, 1, nelem, (void*)dbl.data_ptr<double>(), &status)) {
                    fits_close_file(fptr, &status);
                    throw_fits_error(status, "Error writing VLA column data");
                }
            } else {
                // Write zero-length descriptor by writing 0 elements (CFITSIO sets descriptor appropriately)
                double dummy = 0.0;
                if (fits_write_colnull(fptr, TDOUBLE, col_index, row + 1, 1, 0, &dummy, nullptr, &status)) {
                    // Some CFITSIO builds may not like this; ignore if it fails and continue
                    status = 0;
                }
            }
        }
        ++col_index;
    }
    fits_close_file(fptr, &status);
    if (status) throw_fits_error(status, "Error closing VLA table file");
}

} // namespace torchfits_writer
