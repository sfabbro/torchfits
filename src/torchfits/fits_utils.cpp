#include "fits_reader.h" // Include the header file
#include <algorithm>
#include <sstream>

// --- Utility Functions ---

// Convert a CFITSIO status code into a human-readable error message.
std::string fits_status_to_string(int status) {
    char err_msg[31]; // CFITSIO's error message buffer
    fits_get_errstatus(status, err_msg);
    return std::string(err_msg);
}

// Throw a PyTorch RuntimeError with a formatted message, including CFITSIO error details.
void throw_fits_error(int status, const std::string& message = "") {
    std::stringstream ss;
    ss << message << " CFITSIO error: " << fits_status_to_string(status);
    throw std::runtime_error(ss.str());
}

// Parse a single 80-character FITS header card into key and value.
std::pair<std::string, std::string> parse_header_card(const char* card) {
    std::string card_str(card);
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
            val_start++; // Skip leading spaces
        }

        size_t val_end;
        if (val_start < card_str.size() && card_str[val_start] == '\'') {
            // Handle quoted string values.
            val_start++;  // Skip the opening quote
            val_end = card_str.find('\'', val_start);
            if (val_end == std::string::npos) {
                val_end = card_str.size(); // Handle unterminated string
            }
            value = card_str.substr(val_start, val_end - val_start);
        } else {
            // Handle non-string values (find comment delimiter or end of string).
            val_end = card_str.find('/', val_start);
            if (val_end == std::string::npos) {
                val_end = card_str.size();
            }
            value = card_str.substr(val_start, val_end - val_start);
            value.erase(0, value.find_first_not_of(" "));  // Trim
            value.erase(value.find_last_not_of(" ") + 1); // Trim
        }
    }
    return {key, value};
}

// Read the entire FITS header of the current HDU.
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
    for (int i = 1; i <= num_keys; ++i) {
        if (fits_read_record(fptr, i, card, &status)) {
            if (status != END_OF_FILE) { // END_OF_FILE is expected at the end of the header
                throw_fits_error(status, "Error reading header record " + std::to_string(i));
            }
            break;  // Exit loop when we reach the end
        }
        auto [key, value] = parse_header_card(card);
        if (!key.empty()) {  // Ignore empty keys
            header[key] = value;
        }
    }
    status = 0; //Reset status

    return header;
}

// Get the dimensions (NAXIS values) of a FITS image/cube HDU.
std::vector<long long> get_image_dims(const std::string& filename, int hdu_num) {
    fitsfile* fptr;
    int status = 0;
    int bitpix, naxis;
    long long naxes[3] = {0, 0, 0}; // Initialize to zeros

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, nullptr, &status)) { //Move to HDU
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }
    if (fits_get_img_paramll(fptr, 3, &bitpix, &naxis, naxes, &status)) { //Get dimensions
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error getting image parameters");
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }

    std::vector<long long> dims(naxes, naxes + naxis);  // Convert long long array to vector
    return dims;
}

// Get the value of a *single* FITS header keyword as a string.
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key) {
    fitsfile* fptr;
    int status = 0;
    char value[FLEN_VALUE]; // CFITSIO constant

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
    if (fits_movabs_hdu(fptr, hdu_num, nullptr, &status)) { //Move to HDU
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }

    // Use fits_read_key_str for string values. Handles quoting and trimming.
    if (fits_read_key_str(fptr, key.c_str(), value, nullptr, &status)) {
         if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        if (status == KEY_NO_EXIST) {
            return "";  // Key not found, return empty string.
        }
        throw_fits_error(status, "Error reading header key: " + key);
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
        if (fits_close_file(fptr, &status)) { // Close file
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
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
        if (fits_close_file(fptr, &status)) {
           throw_fits_error(status, "Error closing file");
        }
        throw_fits_error(status, "Error getting number of HDUs");
    }
    if (fits_close_file(fptr, &status)) { //Always close
        throw_fits_error(status, "Error closing FITS file");
    }
    return num_hdus;
}
