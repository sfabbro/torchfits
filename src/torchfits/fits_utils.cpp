#include "fits_utils.h"
#include "debug.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower

void throw_fits_error(int status, const std::string& msg) {
    char err_text[FLEN_ERRMSG];
    fits_get_errstatus(status, err_text);
    throw std::runtime_error(msg + ": " + err_text);
}

std::map<std::string, std::string> read_fits_header(fitsfile* fptr) {
    int status = 0;
    int nkeys;
    std::map<std::string, std::string> header;

    if (fits_get_hdrspace(fptr, &nkeys, NULL, &status)) {
        throw_fits_error(status, "Error getting header space");
    }

    for (int i = 1; i <= nkeys; i++) {
        char card[FLEN_CARD];
        status = 0;
        if (fits_read_record(fptr, i, card, &status)) {
            if (status == KEY_NO_EXIST) continue;
            throw_fits_error(status, "Error reading header record");
        }

        // Skip comment and history records, but include user-defined keywords
        int keyclass = fits_get_keyclass(card);
        if (keyclass == TYP_COMM_KEY) {
            continue;
        }
        
        char key[FLEN_KEYWORD], value[FLEN_VALUE];
        int keylen;
        status = 0;
        if (fits_get_keyname(card, key, &keylen, &status)) {
            continue;
        }

        status = 0;
        if (fits_parse_value(card, value, NULL, &status)) {
            continue;
        }
        
        std::string val_str(value);

        // For string values, cfitsio returns them with quotes, e.g., "'a string   '".
        // The test expects the quotes to be removed from the final value.
        if (val_str.length() > 1 && val_str.front() == '\'') {
            size_t end_quote_pos = val_str.find_last_of('\'');
            if (end_quote_pos > 0) {
                std::string content = val_str.substr(1, end_quote_pos - 1);
                
                // Trim trailing spaces from the content
                size_t last_char = content.find_last_not_of(' ');
                if (std::string::npos != last_char) {
                    content.erase(last_char + 1);
                } else {
                    content.clear(); // all spaces
                }
                header[key] = content;
            } else {
                // Malformed, no closing quote.
                header[key] = val_str;
            }
        } else {
            // Not a quoted string. Trim leading/trailing whitespace.
            size_t first = val_str.find_first_not_of(" ");
            if (std::string::npos == first) {
                header[key] = "";
            } else {
                size_t last = val_str.find_last_not_of(" ");
                header[key] = val_str.substr(first, (last - first + 1));
            }
        }
    }
    return header;
}

int get_hdu_num_by_name(fitsfile* fptr, const std::string& hdu_name) {
    int status = 0;
    int num_hdus, initial_hdu, hdu_type;
    char extname[FLEN_VALUE];

    if (fits_get_num_hdus(fptr, &num_hdus, &status)) {
        throw_fits_error(status, "Error getting number of HDUs");
    }

    // Get current HDU position
    initial_hdu = 1; // Default fallback
    fits_get_hdu_num(fptr, &initial_hdu);

    for (int i = 1; i <= num_hdus; i++) {
        status = 0; // Reset status before each operation
        if (fits_movabs_hdu(fptr, i, &hdu_type, &status)) {
            // Attempt to move back before throwing
            status = 0;
            fits_movabs_hdu(fptr, initial_hdu, &hdu_type, &status);
            throw_fits_error(status, "Error moving to HDU " + std::to_string(i));
        }
        
        status = 0; // Reset status for key reading
        if (fits_read_key_str(fptr, "EXTNAME", extname, NULL, &status) == 0) {
            // Case-insensitive comparison
            std::string extname_str(extname);
            std::transform(extname_str.begin(), extname_str.end(), extname_str.begin(), ::toupper);
            std::string hdu_name_upper = hdu_name;
            std::transform(hdu_name_upper.begin(), hdu_name_upper.end(), hdu_name_upper.begin(), ::toupper);

            if (hdu_name_upper == extname_str) {
                // Found it. Restore original HDU and return.
                status = 0;
                fits_movabs_hdu(fptr, initial_hdu, &hdu_type, &status);
                return i;
            }
        } else if (status != KEY_NO_EXIST) {
            // Some other error occurred reading the key
            status = 0;
            fits_movabs_hdu(fptr, initial_hdu, &hdu_type, &status);
            throw_fits_error(status, "Error reading EXTNAME from HDU " + std::to_string(i));
        }
    }

    // If we get here, the HDU was not found. Restore original HDU and throw.
    status = 0;
    fits_movabs_hdu(fptr, initial_hdu, &hdu_type, &status);
    throw std::runtime_error("HDU with name '" + hdu_name + "' not found.");
}

std::vector<long> get_image_dims(fitsfile* fptr) {
    int status = 0;
    int naxis;
    if (fits_get_img_dim(fptr, &naxis, &status)) {
        throw_fits_error(status, "Error getting image dimensions");
    }

    if (naxis == 0) {
        return {};
    }

    std::vector<long> naxes(naxis);
    if (fits_get_img_size(fptr, naxis, naxes.data(), &status)) {
        throw_fits_error(status, "Error getting image size");
    }
    // FITS is column-major, C++ is row-major, reverse the dimensions
    std::reverse(naxes.begin(), naxes.end());
    return naxes;
}

std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num) {
    FITSFileWrapper f(filename);
    int status = 0;
    if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }
    return read_fits_header(f.get());
}

int get_num_hdus(const std::string& filename) {
    FITSFileWrapper f(filename);
    int status = 0;
    int num_hdus;
    if (fits_get_num_hdus(f.get(), &num_hdus, &status)) {
        throw_fits_error(status, "Error getting number of HDUs");
    }
    return num_hdus;
}

int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name) {
    FITSFileWrapper f(filename);
    return get_hdu_num_by_name(f.get(), hdu_name);
}

std::string get_hdu_type(const std::string& filename, int hdu_num) {
    FITSFileWrapper f(filename);
    int status = 0;
    int hdu_type;
    if (fits_movabs_hdu(f.get(), hdu_num, &hdu_type, &status)) {
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }
    
    switch (hdu_type) {
        case IMAGE_HDU: return "IMAGE";
        case ASCII_TBL: return "ASCII_TBL";
        case BINARY_TBL: return "BINARY_TBL";
        default: return "UNKNOWN";
    }
}

std::vector<long> get_dims(const std::string& filename, int hdu_num) {
    FITSFileWrapper f(filename);
    int status = 0;
    if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }
    return get_image_dims(f.get());
}