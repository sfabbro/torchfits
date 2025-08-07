#include "fits_writer.h"
#include "fits_utils.h"
#include "debug.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace torchfits_writer {

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
            DEBUG_LOG("Warning: Failed to write header keyword " + key + ": " + std::to_string(status));
            status = 0; // Continue with other keywords
        }
    }
}

void tensor_to_fits_image(fitsfile* fptr, torch::Tensor tensor) {
    int status = 0;
    
    // Ensure tensor is contiguous and on CPU
    tensor = tensor.contiguous().cpu();
    
    // Get tensor dimensions (reverse for FITS column-major order)
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
    
    DEBUG_LOG("Successfully wrote tensor of shape " + 
              std::to_string(tensor.sizes().size()) + "D to FITS image");
}

void write_tensor_to_fits(const std::string& filename,
                         torch::Tensor data,
                         const std::map<std::string, std::string>& header,
                         bool overwrite) {
    DEBUG_LOG("Writing tensor to FITS file: " + filename);
    
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
        
        DEBUG_LOG("Successfully wrote tensor to FITS file: " + filename);
        
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
    
    DEBUG_LOG("Writing " + std::to_string(tensors.size()) + " tensors to MEF file: " + filename);
    
    fitsfile* fptr;
    int status = 0;
    
    std::string create_filename = filename;
    if (overwrite) {
        create_filename = "!" + filename;
    }
    
    fits_create_file(&fptr, create_filename.c_str(), &status);
    if (status) throw_fits_error(status, "Error creating FITS file: " + filename);
    
    try {
        // Write primary HDU (first tensor)
        tensor_to_fits_image(fptr, tensors[0]);
        
        // Write primary header
        if (!headers.empty()) {
            write_header_keywords(fptr, headers[0]);
        }
        
        // Add EXTNAME if provided
        if (!extnames.empty() && !extnames[0].empty()) {
            fits_write_key(fptr, TSTRING, "EXTNAME", 
                          const_cast<char*>(extnames[0].c_str()), nullptr, &status);
        }
        
        // Write extension HDUs
        for (size_t i = 1; i < tensors.size(); ++i) {
            // Create new image extension
            tensor_to_fits_image(fptr, tensors[i]);
            
            // Write extension header
            if (i < headers.size()) {
                write_header_keywords(fptr, headers[i]);
            }
            
            // Add EXTNAME if provided
            if (i < extnames.size() && !extnames[i].empty()) {
                fits_write_key(fptr, TSTRING, "EXTNAME", 
                              const_cast<char*>(extnames[i].c_str()), nullptr, &status);
            }
        }
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
        DEBUG_LOG("Successfully wrote MEF file with " + std::to_string(tensors.size()) + " extensions");
        
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
                        bool overwrite) {
    if (table_data.size() == 0) {
        throw std::runtime_error("Cannot write empty table to FITS file");
    }
    
    DEBUG_LOG("Writing table with " + std::to_string(table_data.size()) + " columns to FITS file: " + filename);
    
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
        std::vector<torch::Tensor> column_tensors;
        long nrows = 0;
        
        for (auto item : table_data) {
            std::string col_name = py::str(item.first).cast<std::string>();
            torch::Tensor tensor = item.second.cast<torch::Tensor>();
            
            column_names.push_back(col_name);
            column_formats.push_back(get_fits_column_format(tensor));
            column_tensors.push_back(tensor.contiguous().cpu());
            
            if (nrows == 0) {
                nrows = tensor.size(0);
            } else if (tensor.size(0) != nrows) {
                throw std::runtime_error("All table columns must have the same number of rows");
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
            torch::Tensor& tensor = column_tensors[i];
            int datatype = get_fits_table_datatype(tensor);
            
            fits_write_col(fptr, datatype, i + 1, 1, 1, nrows,
                          tensor.data_ptr(), &status);
            if (status) throw_fits_error(status, "Error writing column " + column_names[i]);
            
            // Write column units if provided
            if (i < column_units.size() && !column_units[i].empty()) {
                std::string unit_key = "TUNIT" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, unit_key.c_str(),
                              const_cast<char*>(column_units[i].c_str()), nullptr, &status);
            }
            
            // Write column descriptions if provided
            if (i < column_descriptions.size() && !column_descriptions[i].empty()) {
                std::string desc_key = "TTYPE" + std::to_string(i + 1);
                fits_write_key(fptr, TSTRING, desc_key.c_str(),
                              const_cast<char*>(column_descriptions[i].c_str()), nullptr, &status);
            }
        }
        
        // Write table header keywords
        write_header_keywords(fptr, header);
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
        DEBUG_LOG("Successfully wrote table with " + std::to_string(ncols) + 
                  " columns and " + std::to_string(nrows) + " rows");
        
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
    
    write_table_to_fits(filename, table_data, header, column_units, column_descriptions, overwrite);
}

void append_hdu_to_fits(const std::string& filename,
                       torch::Tensor data,
                       const std::map<std::string, std::string>& header,
                       const std::string& extname) {
    DEBUG_LOG("Appending HDU to FITS file: " + filename);
    
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
        
        DEBUG_LOG("Successfully appended HDU to FITS file");
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

void update_fits_header(const std::string& filename,
                       int hdu_num,
                       const std::map<std::string, std::string>& updates) {
    DEBUG_LOG("Updating header in FITS file: " + filename + ", HDU: " + std::to_string(hdu_num));
    
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
        
        DEBUG_LOG("Successfully updated " + std::to_string(updates.size()) + " header keywords");
        
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
    DEBUG_LOG("Updating data in FITS file: " + filename + ", HDU: " + std::to_string(hdu_num));
    
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
        
        DEBUG_LOG("Successfully updated FITS data");
        
    } catch (const std::exception& e) {
        fits_close_file(fptr, &status);
        throw;
    }
}

} // namespace torchfits_writer

// === Phase 2: Enhanced Writing Capabilities Implementation ===

namespace torchfits_writer {

int get_fits_compression_type(CompressionType comp_type) {
    switch (comp_type) {
        case CompressionType::None:      return 0;
        case CompressionType::GZIP:      return GZIP_1;
        case CompressionType::RICE:      return RICE_1;
        case CompressionType::HCOMPRESS: return HCOMPRESS_1;
        case CompressionType::PLIO:      return PLIO_1;
        default:                         return 0;
    }
}

void write_tensor_to_fits_advanced(const std::string& filename,
                                  torch::Tensor data,
                                  const std::map<std::string, std::string>& header,
                                  const CompressionConfig& compression,
                                  bool overwrite,
                                  bool checksum) {
    
    fitsfile* fptr = nullptr;
    int status = 0;
    
    try {
        // Prepare filename for creation
        std::string create_filename = filename;
        if (overwrite) {
            create_filename = "!" + filename;
        }
        
        // Create FITS file
        fits_create_file(&fptr, create_filename.c_str(), &status);
        if (status) throw_fits_error(status, "Error creating FITS file");
        
        // Convert tensor to contiguous CPU tensor if needed
        data = data.contiguous().cpu();
        
        // Get tensor dimensions (reverse for FITS column-major order)
        std::vector<long> naxes;
        for (int i = data.dim() - 1; i >= 0; --i) {
            naxes.push_back(data.size(i));
        }
        
        int fits_datatype = get_fits_datatype(data);
        
        // Create image HDU with compression if specified
        if (compression.type != CompressionType::None) {
            // Set compression parameters
            int comp_type = get_fits_compression_type(compression.type);
            fits_set_compression_type(fptr, comp_type, &status);
            if (status) throw_fits_error(status, "Error setting compression type");
            
            if (compression.quantize_level > 0) {
                fits_set_quantize_level(fptr, compression.quantize_level, &status);
                if (status) throw_fits_error(status, "Error setting quantization level");
            }
            
            if (compression.tile_dimensions[0] > 0 && compression.tile_dimensions[1] > 0) {
                long tile_dims[2] = {compression.tile_dimensions[0], compression.tile_dimensions[1]};
                fits_set_tile_dim(fptr, 2, tile_dims, &status);
                if (status) throw_fits_error(status, "Error setting tile dimensions");
            }
        }
        
        // Create primary image HDU
        fits_create_img(fptr, fits_datatype, naxes.size(), naxes.data(), &status);
        if (status) throw_fits_error(status, "Error creating image HDU");
        
        // Write image data
        fits_write_img(fptr, get_fits_table_datatype(data), 1, data.numel(), 
                      data.data_ptr(), &status);
        if (status) throw_fits_error(status, "Error writing image data");
        
        // Write header keywords
        for (const auto& [key, value] : header) {
            fits_write_key_str(fptr, key.c_str(), value.c_str(), nullptr, &status);
            if (status) DEBUG_LOG("Warning: Could not write header key " + key);
        }
        
        // Add checksum if requested
        if (checksum) {
            fits_write_chksum(fptr, &status);
            if (status) DEBUG_LOG("Warning: Could not write checksum");
        }
        
        // Close file
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
        DEBUG_LOG("Successfully wrote tensor to FITS with advanced options");
        
    } catch (const std::exception& e) {
        if (fptr) {
            fits_close_file(fptr, &status);
        }
        throw;
    }
}

void write_variable_length_array(const std::string& filename,
                                const std::vector<torch::Tensor>& arrays,
                                const std::map<std::string, std::string>& header,
                                bool overwrite) {
    
    if (arrays.empty()) {
        throw std::runtime_error("Cannot write empty variable-length array");
    }
    
    fitsfile* fptr = nullptr;
    int status = 0;
    
    try {
        // Prepare filename
        std::string create_filename = overwrite ? "!" + filename : filename;
        
        // Create FITS file
        fits_create_file(&fptr, create_filename.c_str(), &status);
        if (status) throw_fits_error(status, "Error creating FITS file");
        
        // Create primary HDU (empty)
        fits_create_img(fptr, BYTE_IMG, 0, nullptr, &status);
        if (status) throw_fits_error(status, "Error creating primary HDU");
        
        // Create binary table for variable-length arrays
        const char* ttype[] = {"ARRAY_DATA"};
        const char* tform[] = {"1PD"}; // Variable-length double array
        const char* tunit[] = {""};
        
        fits_create_tbl(fptr, BINARY_TBL, arrays.size(), 1, 
                       const_cast<char**>(ttype), 
                       const_cast<char**>(tform),
                       const_cast<char**>(tunit), 
                       "VARARRAY", &status);
        if (status) throw_fits_error(status, "Error creating binary table");
        
        // Write variable-length arrays
        for (size_t i = 0; i < arrays.size(); ++i) {
            torch::Tensor array = arrays[i].contiguous().cpu();
            if (array.dtype() != torch::kFloat64) {
                array = array.to(torch::kFloat64);
            }
            
            long row = i + 1; // FITS is 1-based
            long nelements = array.numel();
            
            fits_write_col(fptr, TDOUBLE, 1, row, 1, nelements,
                          array.data_ptr(), &status);
            if (status) throw_fits_error(status, "Error writing variable-length array");
        }
        
        // Write header keywords
        for (const auto& [key, value] : header) {
            fits_write_key_str(fptr, key.c_str(), value.c_str(), nullptr, &status);
            if (status) DEBUG_LOG("Warning: Could not write header key " + key);
        }
        
        fits_close_file(fptr, &status);
        if (status) throw_fits_error(status, "Error closing FITS file");
        
        DEBUG_LOG("Successfully wrote variable-length arrays");
        
    } catch (const std::exception& e) {
        if (fptr) {
            fits_close_file(fptr, &status);
        }
        throw;
    }
}

// StreamingWriter implementation
StreamingWriter::StreamingWriter(const std::string& filename, 
                               const std::vector<long>& dimensions,
                               torch::ScalarType dtype,
                               const CompressionConfig& compression,
                               bool overwrite)
    : filename_(filename), dimensions_(dimensions), dtype_(dtype), 
      compression_(compression), current_position_(0), fptr_(nullptr), finalized_(false) {
    
    int status = 0;
    
    try {
        // Prepare filename
        std::string create_filename = overwrite ? "!" + filename : filename;
        
        // Create FITS file
        fits_create_file(&fptr_, create_filename.c_str(), &status);
        if (status) throw_fits_error(status, "Error creating FITS file for streaming");
        
        // Set up compression if specified
        if (compression_.type != CompressionType::None) {
            int comp_type = get_fits_compression_type(compression_.type);
            fits_set_compression_type(fptr_, comp_type, &status);
            if (status) throw_fits_error(status, "Error setting compression for streaming");
        }
        
        // Create image HDU
        int fits_type = FLOAT_IMG;
        switch (dtype_) {
            case torch::kFloat32: fits_type = FLOAT_IMG; break;
            case torch::kFloat64: fits_type = DOUBLE_IMG; break;
            case torch::kInt16:   fits_type = SHORT_IMG; break;
            case torch::kInt32:   fits_type = LONG_IMG; break;
            case torch::kInt64:   fits_type = LONGLONG_IMG; break;
            default: fits_type = FLOAT_IMG; break;
        }
        
        fits_create_img(fptr_, fits_type, dimensions_.size(), 
                       const_cast<long*>(dimensions_.data()), &status);
        if (status) throw_fits_error(status, "Error creating streaming image HDU");
        
        DEBUG_LOG("StreamingWriter initialized for " + filename);
        
    } catch (const std::exception& e) {
        if (fptr_) {
            fits_close_file(fptr_, &status);
            fptr_ = nullptr;
        }
        throw;
    }
}

StreamingWriter::~StreamingWriter() {
    if (fptr_ && !finalized_) {
        finalize();
    }
}

void StreamingWriter::write_sequential(const torch::Tensor& data) {
    if (!fptr_) {
        throw std::runtime_error("StreamingWriter not initialized");
    }
    
    torch::Tensor write_data = data.contiguous().cpu();
    int status = 0;
    
    // Write data sequentially
    long first_pixel = current_position_ + 1; // FITS is 1-based
    int datatype = get_fits_table_datatype(write_data);
    
    fits_write_pix(fptr_, datatype, &first_pixel, write_data.numel(),
                  write_data.data_ptr(), &status);
    if (status) throw_fits_error(status, "Error in sequential write");
    
    current_position_ += write_data.numel();
}

void StreamingWriter::finalize(const std::map<std::string, std::string>& header) {
    if (!fptr_ || finalized_) return;
    
    int status = 0;
    
    // Write header keywords
    for (const auto& [key, value] : header) {
        fits_write_key_str(fptr_, key.c_str(), value.c_str(), nullptr, &status);
        if (status) DEBUG_LOG("Warning: Could not write header key " + key);
    }
    
    // Close file
    fits_close_file(fptr_, &status);
    if (status) throw_fits_error(status, "Error finalizing streaming writer");
    
    fptr_ = nullptr;
    finalized_ = true;
    
    DEBUG_LOG("StreamingWriter finalized");
}

} // namespace torchfits_writer
