#include "fits_reader.h"
#include <algorithm>
#include <sstream>

// --- Utility Functions ---

// Convert a CFITSIO status code into a human-readable error message.
std::string fits_status_to_string(int status) {
    char err_msg[31]; // CFITSIO's error message buffer is 30 chars + null terminator
    fits_get_errstatus(status, err_msg);
    return std::string(err_msg);
}

// Throw a PyTorch RuntimeError with a formatted message, including CFITSIO error details.
void throw_fits_error(int status, const std::string& message) {
    std::stringstream ss;
    ss << message << " CFITSIO error: " << fits_status_to_string(status);
    throw std::runtime_error(ss.str());
}

// Parse a single 80-character FITS header card into key and value.
std::pair<std::string, std::string> parse_header_card(const char* card) {
    std::string card_str(card);  // Convert to std::string for easier manipulation
    std::string key, value;

    size_t eq_pos = card_str.find('=');
    if (eq_pos == std::string::npos || eq_pos >= 79) {
        // Handle cases where '=' is missing or misplaced (COMMENT, HISTORY, or blank).
        key = card_str;
        value = "";
    } else {
        key = card_str.substr(0, eq_pos);
        key.erase(0, key.find_first_not_of(" "));  // Trim leading spaces
        key.erase(key.find_last_not_of(" ") + 1); // Trim trailing spaces

        size_t val_start = eq_pos + 1;
        while (val_start < card_str.size() && card_str[val_start] == ' ') {
            val_start++; // Skip leading spaces after '='
        }

        size_t val_end;
        if (val_start < card_str.size() && card_str[val_start] == '\'') {
            // Handle quoted string values.
            val_start++;  // Skip the opening quote
            val_end = card_str.find('\'', val_start); // Find closing quote
            if (val_end == std::string::npos) {
                val_end = card_str.size(); // Handle unterminated string
            }
            value = card_str.substr(val_start, val_end - val_start);
        } else {
            // Handle non-string values (find comment delimiter or end of string).
            val_end = card_str.find('/', val_start);  // Find comment delimiter
            if (val_end == std::string::npos) {
                val_end = card_str.size(); // No comment, use the rest of the string
            }
            value = card_str.substr(val_start, val_end - val_start);
            value.erase(0, value.find_first_not_of(" "));  // Trim leading spaces
            value.erase(value.find_last_not_of(" ") + 1); // Trim trailing spaces
        }
    }
    return {key, value};
}

// Read the entire FITS header of the current HDU and return it as a map.
std::map<std::string, std::string> read_fits_header(fitsfile* fptr) {
    int status = 0;
    int num_keys;

    // Get the number of keywords in the current HDU.
    if (fits_get_hdrspace(fptr, &num_keys, nullptr, &status)) {
        throw_fits_error(status, "Error getting header size");
    }

    std::map<std::string, std::string> header;
    char card[FLEN_CARD]; // CFITSIO constant for card length

    // Iterate through header records (cards).
    for (int i = 1; i <= num_keys; ++i) {  // FITS indexing starts at 1
        if (fits_read_record(fptr, i, card, &status)) {
            if (status != END_OF_FILE) { // END_OF_FILE is expected at the end
                throw_fits_error(status, "Error reading header record " + std::to_string(i));
            }
            break;  // Exit loop when we reach END
        }
        auto [key, value] = parse_header_card(card); // Use structured bindings
        if (!key.empty()) {  // Ignore empty keys (e.g., blank cards)
            header[key] = value;
        }
    }
    status = 0; //reset status
    return header;
}

// Internal function to get the dimensions of an HDU.
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num) {
    fitsfile* fptr;
    int status = 0;
    int bitpix, naxis;
    long long naxes[3] = {0, 0, 0}; // Initialize to zeros.  We support up to 3 dimensions.

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, nullptr, &status)) { //Move to HDU
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
    }
    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) { //Get dimensions
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error getting image parameters");
        }
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }

    std::vector<long long> dims(naxes, naxes + std::min(naxis, 3));  // Convert long long array to vector
    return dims;
}

// Get the value of a *single* FITS header keyword as a string.
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key) {
    fitsfile* fptr;
    int status = 0;
    char value[FLEN_VALUE]; // CFITSIO constant for keyword value length

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, nullptr, &status)) { //Move to HDU
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
    }

    // Use fits_read_key_str for string values. Handles quoting and trimming.
    if (fits_read_key_str(fptr, key.c_str(), value, nullptr, &status)) {
        if (status) {
            fits_close_file(fptr, &status); // Close file
            if (status == KEY_NO_EXIST) {
                return "";  // Key not found, return empty string.
            }
            throw_fits_error(status, "Error reading header key: " + key);
        }
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }

    return std::string(value); // Return the value as a string.
}

// Get the type of a given HDU (IMAGE, TABLE, BINTABLE).
std::string get_hdu_type(const std::string& filename, int hdu_num) {
    fitsfile* fptr;
    int status = 0;
    int hdu_type;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, &hdu_type, &status)) { //Move to HDU
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }

    // Convert the CFITSIO integer code to a string representation.
    if (hdu_type == IMAGE_HDU) {
        return "IMAGE";
    } else if (hdu_type == ASCII_TBL) {
        return "TABLE";  // ASCII table
    } else if (hdu_type == BINARY_TBL) {
        return "BINTABLE"; // Binary table
    } else {
        return "UNKNOWN";
    }
}

// Get the total number of HDUs in a FITS file.
int get_num_hdus(const std::string& filename) {
    fitsfile* fptr;
    int status = 0;
    int num_hdus = 0;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_get_num_hdus(fptr, &num_hdus, &status)) {
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error getting number of HDUs");
        }
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }
    return num_hdus;
}

//Wrapper around read_fits_header, to open/close files.
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num)
{
    fitsfile* fptr;
    int status = 0;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, nullptr, &status)) {
        if (status) {
            fits_close_file(fptr, &status); // Close file
            throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
        }
    }

    //Call to the function to read the header
    std::map<std::string, std::string> header = read_fits_header(fptr);
    if (fits_close_file(fptr, &status)) {
        throw_fits_error(status, "Error closing FITS file");
    }

    return header;
}

//Wrapper around _get_hdu_dims
std::vector<long long> get_dims(const std::string& filename, int hdu_num){
    return _get_hdu_dims(filename, hdu_num);
}
