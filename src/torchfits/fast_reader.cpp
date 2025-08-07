#include "fast_reader.h"
#include "fits_utils.h"
#include "debug.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace torchfits_fast {

// Maximum dimensions supported (matching fitsio's NUMPY_MAX_DIMS)
const int MAX_DIMS = 32;

torch::Tensor FastImageReader::read_image_fast(fitsfile* fptr, torch::Device device) {
    DEBUG_SCOPE;
    
    // Simple indication that fast reader is being used
    std::cerr << "[FAST_READER] Using FITSIO-optimized reader" << std::endl;
    
    int status = 0;
    int bitpix;
    fits_get_img_type(fptr, &bitpix, &status);
    if (status) throw_fits_error(status, "Error getting image type");

    switch (bitpix) {
        case BYTE_IMG:     return read_image_typed_fast<uint8_t, TBYTE>(fptr, device);
        case SHORT_IMG:    return read_image_typed_fast<int16_t, TSHORT>(fptr, device);
        case LONG_IMG:     return read_image_typed_fast<int32_t, TINT>(fptr, device);
        case LONGLONG_IMG: return read_image_typed_fast<int64_t, TLONGLONG>(fptr, device);
        case FLOAT_IMG:    return read_image_typed_fast<float, TFLOAT>(fptr, device);
        case DOUBLE_IMG:   return read_image_typed_fast<double, TDOUBLE>(fptr, device);
        default:
            throw std::runtime_error("Unsupported FITS image type: " + std::to_string(bitpix));
    }
}

template<typename T, int CfitsioType>
torch::Tensor FastImageReader::read_image_typed_fast(fitsfile* fptr, torch::Device device) {
    DEBUG_SCOPE;
    
    int status = 0;
    
    // Get image parameters using CFITSIO's efficient approach
    int maxdim = MAX_DIMS;
    int datatype = 0;
    int naxis = 0;
    LONGLONG naxes[MAX_DIMS];
    
    // Use fits_get_img_paramll like fitsio does (more efficient than fits_get_img_dim)
    if (fits_get_img_paramll(fptr, maxdim, &datatype, &naxis, naxes, &status)) {
        throw_fits_error(status, "Error getting image parameters");
    }
    
    // Calculate total size
    LONGLONG total_size = 1;
    std::vector<int64_t> torch_dims(naxis);
    for (int i = 0; i < naxis; i++) {
        total_size *= naxes[i];
        // Convert FITS order (fastest axis first) to PyTorch order (slowest axis first)
        torch_dims[naxis - 1 - i] = static_cast<int64_t>(naxes[i]);
    }
    
    // Create tensor with standard allocation (no alignment overhead for now)
    torch::Tensor data = torch::empty(torch_dims, torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>::value).device(device));
    
    // FITSIO OPTIMIZATION: Use fits_read_pixll like fitsio does
    // This is often faster than fits_read_pix for large arrays
    LONGLONG firstpixels[MAX_DIMS];
    for (int i = 0; i < naxis; i++) {
        firstpixels[i] = 1; // FITS uses 1-based indexing
    }
    
    int anynul = 0; // We don't use null value handling for now
    
    DEBUG_LOG("Reading " + std::to_string(total_size) + " pixels using fits_read_pixll");
    
    if (fits_read_pixll(fptr, CfitsioType, firstpixels, total_size, nullptr, 
                        data.data_ptr<T>(), &anynul, &status)) {
        throw_fits_error(status, "Error reading image with fits_read_pixll");
    }
    
    return data;
}

// Explicit instantiations for supported types
template torch::Tensor FastImageReader::read_image_typed_fast<uint8_t, TBYTE>(fitsfile* fptr, torch::Device device);
template torch::Tensor FastImageReader::read_image_typed_fast<int16_t, TSHORT>(fitsfile* fptr, torch::Device device);
template torch::Tensor FastImageReader::read_image_typed_fast<int32_t, TINT>(fitsfile* fptr, torch::Device device);
template torch::Tensor FastImageReader::read_image_typed_fast<int64_t, TLONGLONG>(fitsfile* fptr, torch::Device device);
template torch::Tensor FastImageReader::read_image_typed_fast<float, TFLOAT>(fitsfile* fptr, torch::Device device);
template torch::Tensor FastImageReader::read_image_typed_fast<double, TDOUBLE>(fitsfile* fptr, torch::Device device);

// Table reading implementation
pybind11::dict FastTableReader::read_table_fast(
    fitsfile* fptr,
    const std::vector<std::string>& columns,
    long start_row,
    long num_rows,
    torch::Device device
) {
    DEBUG_SCOPE;
    
    // Simple indication that fast table reader is being used
    std::cerr << "[FAST_READER] Using FITSIO-optimized table reader for " << columns.size() << " columns" << std::endl;
    
    auto column_info = analyze_columns(fptr, columns);
    pybind11::dict result;
    
    for (const auto& col : column_info) {
        DEBUG_LOG("Reading column: " + col.name + " (type " + std::to_string(col.fits_type) + ")");
        
        torch::Tensor tensor;
        switch (col.fits_type) {
            case TBYTE:
                tensor = read_column_fast<uint8_t, TBYTE>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            case TSHORT:
                tensor = read_column_fast<int16_t, TSHORT>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            case TINT:
                tensor = read_column_fast<int32_t, TINT>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            case TLONG:
                tensor = read_column_fast<int64_t, TLONG>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            case TFLOAT:
                tensor = read_column_fast<float, TFLOAT>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            case TDOUBLE:
                tensor = read_column_fast<double, TDOUBLE>(fptr, col.col_num, start_row, num_rows, col.repeat);
                break;
            default:
                DEBUG_LOG("Skipping unsupported column type: " + std::to_string(col.fits_type));
                continue;
        }
        
        if (device != torch::kCPU) {
            tensor = tensor.to(device);
        }
        
        result[col.name.c_str()] = tensor;
    }
    
    return result;
}

std::vector<FastTableReader::ColumnInfo> FastTableReader::analyze_columns(
    fitsfile* fptr, 
    const std::vector<std::string>& requested_columns
) {
    DEBUG_SCOPE;
    
    int status = 0;
    std::vector<ColumnInfo> result;
    
    // Get all columns if none specified
    std::vector<std::string> columns_to_process;
    if (requested_columns.empty()) {
        int ncols;
        fits_get_num_cols(fptr, &ncols, &status);
        if (status) throw_fits_error(status, "Error getting number of columns");
        
        for (int i = 1; i <= ncols; i++) {
            char colname[FLEN_VALUE];
            fits_get_acolparms(fptr, i, colname, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &status);
            if (status == 0) {
                columns_to_process.emplace_back(colname);
            }
            status = 0; // Reset for next iteration
        }
    } else {
        columns_to_process = requested_columns;
    }
    
    for (const auto& col_name : columns_to_process) {
        ColumnInfo info;
        info.name = col_name;
        
        // Get column number
        status = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &info.col_num, &status);
        if (status) {
            DEBUG_LOG("Column not found: " + col_name);
            continue; // Skip invalid columns
        }
        
        // Get column type information
        fits_get_coltype(fptr, info.col_num, &info.fits_type, &info.repeat, &info.width, &status);
        if (status) {
            DEBUG_LOG("Error getting column type for: " + col_name);
            continue;
        }
        
        // Map FITS type to torch dtype
        switch (info.fits_type) {
            case TBYTE:     info.torch_dtype = torch::kUInt8; break;
            case TSHORT:    info.torch_dtype = torch::kInt16; break;
            case TINT:      info.torch_dtype = torch::kInt32; break;
            case TLONG:     info.torch_dtype = torch::kInt64; break;
            case TFLOAT:    info.torch_dtype = torch::kFloat32; break;
            case TDOUBLE:   info.torch_dtype = torch::kFloat64; break;
            default:
                DEBUG_LOG("Unsupported column type: " + std::to_string(info.fits_type));
                continue;
        }
        
        result.push_back(info);
    }
    
    return result;
}

template<typename T, int CfitsioType>
torch::Tensor FastTableReader::read_column_fast(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat) {
    DEBUG_SCOPE;
    
    // Create tensor shape
    std::vector<int64_t> shape;
    if (repeat == 1) {
        shape = {num_rows};
    } else {
        shape = {num_rows, repeat};
    }
    
    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>::value));
    
    // FITSIO OPTIMIZATION: Use fits_read_col for efficient column reading
    int status = 0;
    int anynul = 0;
    
    // Read entire column at once (like fitsio does)
    if (fits_read_col(fptr, CfitsioType, col_num, start_row + 1, 1, num_rows * repeat,
                      nullptr, tensor.data_ptr<T>(), &anynul, &status)) {
        throw_fits_error(status, "Error reading column " + std::to_string(col_num));
    }
    
    DEBUG_LOG("Read " + std::to_string(num_rows * repeat) + " elements from column " + std::to_string(col_num));
    
    return tensor;
}

// Explicit instantiations for table reading
template torch::Tensor FastTableReader::read_column_fast<uint8_t, TBYTE>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
template torch::Tensor FastTableReader::read_column_fast<int16_t, TSHORT>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
template torch::Tensor FastTableReader::read_column_fast<int32_t, TINT>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
template torch::Tensor FastTableReader::read_column_fast<int64_t, TLONG>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
template torch::Tensor FastTableReader::read_column_fast<float, TFLOAT>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);
template torch::Tensor FastTableReader::read_column_fast<double, TDOUBLE>(fitsfile* fptr, int col_num, long start_row, long num_rows, long repeat);

} // namespace torchfits_fast
