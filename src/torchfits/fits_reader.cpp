#include "fits_reader.h"
#include "wcs_utils.h"
#include <sstream>
#include <algorithm>

// --- Core Data Reading Logic (Image HDUs) ---

torch::Tensor read_image_data(fitsfile* fptr, std::unique_ptr<wcsprm>& wcs) {
    int status = 0;
    int bitpix, naxis, anynul;
    long long naxes[3] = {1, 1, 1};  // CFITSIO supports up to 999 dimensions; we handle up to 3 here
    long long nelements;

    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) { //Use a more general function
        throw_fits_error(status, "Error getting image parameters");
    }

    // Check for supported dimensions (1D, 2D, or 3D)
    if (naxis < 1 || naxis > 3) {
        throw std::runtime_error("Unsupported number of dimensions: " + std::to_string(naxis) + ". Only 1D, 2D, and 3D images are supported.");
    }

    nelements = 1;
    for (int i = 0; i < naxis; ++i) {
        nelements *= naxes[i];
    }

    // --- WCS Handling ---
    // CFITSIO automatically updates CRPIX in the header when a cutout is read
    // or when opening with fits_file_open.
    auto updated_wcs = read_wcs_from_header(fptr); // wcs_utils.cpp
    if (updated_wcs) {
        wcs = std::move(updated_wcs);  // Take ownership if WCS is valid
    }

    torch::TensorOptions options;
    // Use a macro for code reuse.  This avoids code duplication.
    #define READ_AND_RETURN(cfitsio_type, torch_type, data_type) \
        options = torch::TensorOptions().dtype(torch_type); \
        /* Create tensor with correct dimensions, and order (z,y,x) for Pytorch compatibility*/  \
        auto data = torch::empty({(naxis > 2) ? naxes[2] : 1,  \
                                 (naxis > 1) ? naxes[1] : 1,  \
                                 naxes[0]}, options); \
        data_type* data_ptr = data.data_ptr<data_type>(); \
        if (fits_read_img(fptr, cfitsio_type, 1, nelements, nullptr, data_ptr, &anynul, &status)) { \
            throw_fits_error(status, "Error reading " #cfitsio_type " data"); \
        } \
        return data;

    // Select the appropriate read function based on BITPIX.
    if (bitpix == BYTE_IMG) {
        READ_AND_RETURN(TBYTE, torch::kUInt8, uint8_t);
    } else if (bitpix == SHORT_IMG) {
        READ_AND_RETURN(TSHORT, torch::kInt16, int16_t);
    } else if (bitpix == LONG_IMG) {
        READ_AND_RETURN(TINT, torch::kInt32, int32_t);
     } else if (bitpix == LONGLONG_IMG) {
        READ_AND_RETURN(TLONGLONG, torch::kInt64, int64_t);
    } else if (bitpix == FLOAT_IMG) {
        READ_AND_RETURN(TFLOAT, torch::kFloat32, float);
    } else if (bitpix == DOUBLE_IMG) {
        READ_AND_RETURN(TDOUBLE, torch::kFloat64, double);
    } else {
        throw_fits_error(0, "Unsupported data type (BITPIX = " + std::to_string(bitpix) + ")");
    }
    #undef READ_AND_RETURN
}

// --- Core Data Reading Logic (Binary and ASCII Tables) ---
std::map<std::string, torch::Tensor> read_table_data(fitsfile* fptr) {
    int status = 0;
    int num_cols, typecode;
    long long num_rows;

    if (fits_get_num_rowsll(fptr, &num_rows, &status)) {
        throw_fits_error(status, "Error getting number of rows in table");
    }
    if (fits_get_num_cols(fptr, &num_cols, &status)) {
        throw_fits_error(status, "Error getting number of columns in table");
    }

    std::map<std::string, torch::Tensor> table_data;

    char col_name[FLEN_VALUE]; //CFITSIO constant
    for (int col_num = 1; col_num <= num_cols; ++col_num) { //FITS standard: Columns start at 1
        long long repeat, width;
        if (fits_get_coltype(fptr, col_num, &typecode, &repeat, &width, &status)) {
            throw_fits_error(status, "Error getting column type for column " + std::to_string(col_num));
        }
        if (fits_get_colname(fptr, CASEINSEN, "*", col_name, &col_num, &status)) {
            throw_fits_error(status, "Error getting column name for column " + std::to_string(col_num));
        }
        std::string col_name_str(col_name);

        #define READ_COL_AND_STORE(cfitsio_type, torch_type, data_type) \
            { \
                auto tensor = torch::empty({num_rows}, torch::TensorOptions().dtype(torch_type)); \
                data_type* data_ptr = tensor.data_ptr<data_type>(); \
                if (fits_read_col(fptr, cfitsio_type, col_num, 1, 1, num_rows, nullptr, data_ptr, nullptr, &status)) { \
                    throw_fits_error(status, "Error reading column " + col_name_str + " (data type " #cfitsio_type ")"); \
                } \
                table_data[col_name_str] = tensor; \
            }

        // Select appropriate read function and PyTorch data type based on typecode.
        if (typecode == TBYTE) {
            READ_COL_AND_STORE(TBYTE, torch::kUInt8, uint8_t);
        } else if (typecode == TSHORT) {
            READ_COL_AND_STORE(TSHORT, torch::kInt16, int16_t);
        } else if (typecode == TINT) {
            READ_COL_AND_STORE(TINT, torch::kInt32, int32_t); //Should work in most cases
        }else if (typecode == TLONG) { //fits_read_col does not support int, so map TINT to int32_t, and TLONG to int
             READ_COL_AND_STORE(TLONG, torch::kInt32, int32_t);
        }else if (typecode == TLONGLONG) {
            READ_COL_AND_STORE(TLONGLONG, torch::kInt64, int64_t);
        } else if (typecode == TFLOAT) {
            READ_COL_AND_STORE(TFLOAT, torch::kFloat32, float);
        } else if (typecode == TDOUBLE) {
            READ_COL_AND_STORE(TDOUBLE, torch::kFloat64, double);
        } else if (typecode == TSTRING) {
            //For strings, we read an array of chars
            std::vector<char*> string_array(num_rows);
            for (int i = 0; i < num_rows; i++) {
                string_array[i] = new char[width + 1]; // +1 for null terminator
                string_array[i][width] = '\0'; // Ensure null termination
            }
            if (fits_read_col(fptr, TSTRING, col_num, 1, 1, num_rows, nullptr, string_array.data(), nullptr, &status)){
                 throw_fits_error(status, "Error reading column " + col_name_str + " (data type " #cfitsio_type ")");
            }
            //Convert to a list of strings to return
            std::vector<std::string> string_list;
            for (int i = 0; i < num_rows; i++) {
                string_list.emplace_back(string_array[i]);
                delete[] string_array[i];  // Free the allocated memory.
            }

           table_data[col_name_str] = pybind11::cast(string_list); //Return a python list of strings

        }
        else {
             if (fits_close_file(fptr, &status)) {
                 throw_fits_error(status, "Error closing file");
             }
            throw_fits_error(0, "Unsupported column data type (" + std::to_string(typecode) + ") in column " + col_name_str);
        }
        #undef READ_COL_AND_STORE
    }

    return table_data;
}
// --- Public API Functions ---

// Main read function: Handles images, tables, cutouts, and HDU selection.
pybind11::object read(const std::string& filename_with_cutout, pybind11::object hdu,
                      pybind11::object start, pybind11::object shape) {
    fitsfile* fptr = nullptr;
    int status = 0;
    int hdu_type;
    int hdu_num = 0; // Default to primary HDU (0 for CFITSIO).

    std::string filename;
    std::string cutout_str;

    //Check if filename_with_cutout contains brackets
    size_t first_bracket = filename_with_cutout.find('[');
    if(first_bracket !=  std::string::npos) {
        //Cutout
        filename = filename_with_cutout.substr(0, first_bracket);
        cutout_str = filename_with_cutout.substr(first_bracket);
    }
    else{
        //No cutout
        filename = filename_with_cutout;
        cutout_str = "";
    }


    // --- HDU Handling ---
    if (!hdu.is_none()) { //If hdu is provided
        if (pybind11::isinstance<pybind11::int_>(hdu)) {
            hdu_num = hdu.cast<int>();  // Use explicit HDU number.
            if (hdu_num < 0 )
            {
                 throw std::runtime_error("HDU number must be >= 0");
            }
        } else if (pybind11::isinstance<pybind11::str>(hdu)) {
            // Try to open as named extension.  CFITSIO will handle the lookup.
            filename = filename + "[" + hdu.cast<std::string>() + "]" + cutout_str; // reconstruct filename
        } else {
            throw std::runtime_error("Invalid 'hdu' argument.  Must be int or str.");
        }
    }
    else{ //If not, append cutout if exists
        filename = filename + cutout_str;
    }


    // --- Cutout Handling (start and shape) ---
    //If got start and shape arguments, create the cutout string
    if (!start.is_none() || !shape.is_none())
    {
          //If hdu is a number, reconstruct the full filename
         if (pybind11::isinstance<pybind11::int_>(hdu)) {
             filename = filename + "[" + std::to_string(hdu_num) + "]";
         } // else,  filename is ok.

        if (start.is_none() || shape.is_none()) {
            throw std::runtime_error("If 'start' is provided, 'shape' must also be provided, and vice-versa.");
        }
        if (!pybind11::isinstance<pybind11::sequence>(start) ||
            !pybind11::isinstance<pybind11::sequence>(shape)) {
            throw std::runtime_error("'start' and 'shape' must be sequences (e.g., lists or tuples).");
        }

        auto start_list = start.cast<std::vector<long>>();
        auto shape_list = shape.cast<std::vector<long>>();

        if (start_list.size() != shape_list.size()) {
            throw std::runtime_error("'start' and 'shape' must have the same number of dimensions.");
        }

        // Construct CFITSIO cutout string.
        std::stringstream cutout_builder;
        cutout_builder << "[";
        for (size_t i = 0; i < start_list.size(); ++i) {
            if(shape_list[i] <= 0 && ! (shape_list[i] == -1) ) // Use -1 as None
                throw std::runtime_error("Shape values must be > 0, or -1 (None)");
            // Special case: None in shape means read to the end.
            long long start_val = start_list[i] + 1; //FITS are 1-indexed
            long long end_val;
            if(shape_list[i] == -1) { //None, read all
                end_val = -1;
            }
            else{
                end_val = start_list[i] + shape_list[i]; //end is inclusive
            }

            cutout_builder << start_val << ":" << end_val;
            if (i < start_list.size() - 1) {
                cutout_builder << ",";
            }
        }
        cutout_builder << "]";
        filename = filename + cutout_builder.str(); //Append to filename

    }
    // --- File Opening (using constructed filename) ---
    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file/cutout: " + filename);
    }

    if (fits_get_hdu_type(fptr, &hdu_type, &status)) {
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error getting HDU type");
    }

    // --- Data Reading (Based on HDU Type) ---
    if (hdu_type == IMAGE_HDU) {
        auto wcs = read_wcs_from_header(fptr);  // From wcs_utils.cpp
        torch::Tensor data = read_image_data(fptr, wcs);
        std::map<std::string, std::string> header = read_fits_header(fptr);  // From fits_utils.cpp
          if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        return pybind11::make_tuple(data, header);  // Return tuple for images

    } else if (hdu_type == BINARY_TBL || hdu_type == ASCII_TBL) {
        std::map<std::string, torch::Tensor> table_data = read_table_data(fptr);
        std::map<std::string, std::string> header = read_fits_header(fptr); // From fits_utils.cpp
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        return pybind11::cast(table_data);  // Return dict for tables

    } else {
         if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw std::runtime_error("Unsupported HDU type.");
    }
}