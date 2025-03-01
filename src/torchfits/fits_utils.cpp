#include "fits_reader.h"
#include <algorithm>
#include <sstream>
#include "fits_utils.h"

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
    std::string card_str(card);
    std::string key, value;
    
    // Trim the entire card
    auto trim_start = card_str.find_first_not_of(" ");
    if (trim_start == std::string::npos) {
        return {"", ""}; // Empty or whitespace-only card
    }
    card_str = card_str.substr(trim_start);
    
    // Find the equals sign (if present)
    size_t eq_pos = card_str.find('=');
    if (eq_pos == std::string::npos) {
        // No equals sign - this is a COMMENT, HISTORY, or similar
        key = card_str;
        auto key_end = key.find_first_of(" \t");
        if (key_end != std::string::npos) {
            key = key.substr(0, key_end);
        }
        return {key, ""};
    }
    
    // Process key (left of equals)
    key = card_str.substr(0, eq_pos);
    auto key_end = key.find_last_not_of(" \t");
    key = key.substr(0, key_end + 1);
    
    // Process value (right of equals)
    value = card_str.substr(eq_pos + 1);
    auto value_start = value.find_first_not_of(" \t");
    if (value_start == std::string::npos) {
        return {key, ""}; // Empty value
    }
    value = value.substr(value_start);
    
    // Handle quoted string values
    if (!value.empty() && value[0] == '\'') {
        // Find the closing quote, considering FITS' escaped quote rules ('')
        size_t pos = 1;
        std::string unquoted;
        bool escaped = false;
        
        while (pos < value.size()) {
            char c = value[pos++];
            if (c == '\'') {
                if (pos < value.size() && value[pos] == '\'') {
                    // Double single quote is an escaped quote in FITS
                    unquoted += '\'';
                    pos++;
                    escaped = false;
                } else {
                    // End of quoted string
                    escaped = true;
                    break;
                }
            } else {
                unquoted += c;
            }
        }
        
        if (escaped) {
            value = unquoted;
        }
    } else {
        // Non-quoted value - find comment delimiter
        auto comment_pos = value.find('/');
        if (comment_pos != std::string::npos) {
            value = value.substr(0, comment_pos);
        }
        
        // Trim trailing whitespace
        auto val_end = value.find_last_not_of(" \t");
        if (val_end != std::string::npos) {
            value = value.substr(0, val_end + 1);
        }
    }
    
    return {key, value};
}

// Read the entire FITS header of the current HDU and return it as a map.
std::map<std::string, std::string> read_fits_header(fitsfile* fptr) {
    int status = 0;
    int nkeys, keypos;
    char card[FLEN_CARD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    std::map<std::string, std::string> header;

    if (fits_get_hdrspace(fptr, &nkeys, NULL, &status)) {
        throw_fits_error(status, "Error getting header size");
    }

    for (int i = 1; i <= nkeys; i++) {
        status = 0;
        if (fits_read_record(fptr, i, card, &status)) {
            throw_fits_error(status, "Error reading header record");
        }

        char keyname[FLEN_KEYWORD];
        status = 0;
        if (fits_get_keyname(card, keyname, &keypos, &status) > 0) {
            // Valid keyword found, get its value
            status = 0;
            if (fits_parse_value(card, value, comment, &status) >= 0) {
                // Successfully parsed value
                header[keyname] = value;
            } else if (status == VALUE_UNDEFINED) {
                // NULL value
                header[keyname] = "UNDEFINED";
                status = 0; // Reset status
            }
        }
    }

    return header;
}

// Internal function to get the dimensions of an HDU.
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num) {
    FITSFile fits_file(filename);
    fits_file.move_to_hdu(hdu_num);
    
    int status = 0;
    int bitpix, naxis;
    
    // First get the number of axes
    if (fits_get_img_paramll(fits_file.get(), 0, &bitpix, &naxis, nullptr, &status)) {
        throw_fits_error(status, "Error getting image dimensions");
    }
    
    // Now allocate and get the actual dimensions
    std::vector<long long> naxes(naxis);
    if (naxis > 0) {
        if (fits_get_img_paramll(fits_file.get(), naxis, &bitpix, &naxis, naxes.data(), &status)) {
            throw_fits_error(status, "Error getting image dimensions");
        }
    }
    
    return naxes;
}

// Get the value of a *single* FITS header keyword as a string.
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key) {
    try {
        FITSFile fits_file(filename);
        fits_file.move_to_hdu(hdu_num);
        
        char value[FLEN_VALUE];
        int status = 0;
        if (fits_read_key_str(fits_file.get(), key.c_str(), value, nullptr, &status)) {
            if (status == KEY_NO_EXIST) {
                return ""; // Key not found, return empty string
            }
            throw_fits_error(status, "Error reading header key: " + key);
        }
        
        return std::string(value);
    } catch (const std::exception& e) {
        // Optionally log or wrap the exception
        throw;
    }
}

// Get the type of a given HDU (IMAGE, TABLE, BINTABLE).
std::string get_hdu_type(const std::string& filename, int hdu_num) {
    FITSFile fits_file(filename);
    int status = 0;
    int hdu_type;

    fits_file.move_to_hdu(hdu_num, &hdu_type);

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
    FITSFile fits_file(filename);
    fitsfile* fptr = fits_file.get();
    int status = 0;
    int num_hdus = 0;

    if (fits_get_num_hdus(fptr, &num_hdus, &status)) {
        throw_fits_error(status, "Error getting number of HDUs");
    }
    return num_hdus;
}

// Wrapper around read_fits_header, to open/close files.
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num) {
    FITSFile fits_file(filename);
    fits_file.move_to_hdu(hdu_num);
    return read_fits_header(fits_file.get());
}

// Wrapper around _get_hdu_dims
std::vector<long long> get_dims(const std::string& filename, int hdu_num) {
    return _get_hdu_dims(filename, hdu_num);
}

// Implementation of the FITSFile class
FITSFile::FITSFile(const std::string& filename) : fptr_(nullptr), owned_(true) {
    int status = 0;
    
    INFO_LOG("Opening FITS file: " + filename);
    if (fits_open_file(&fptr_, filename.c_str(), READONLY, &status)) {
        ERROR_LOG("Failed to open FITS file: " + fits_status_to_string(status));
        throw_fits_error(status, "Error opening FITS file: " + filename);
    }
}

// Implementation of move semantics for FITSFile
FITSFile::FITSFile(FITSFile&& other) noexcept 
    : fptr_(other.fptr_), owned_(other.owned_) {
    other.fptr_ = nullptr;
    other.owned_ = false;
}

FITSFile& FITSFile::operator=(FITSFile&& other) noexcept {
    if (this != &other) {
        if (owned_ && fptr_) {
            int status = 0;
            fits_close_file(fptr_, &status);
            if (status) {
                WARNING_LOG("Error closing FITS file in move assignment operator: " + 
                           fits_status_to_string(status));
            }
        }
        
        fptr_ = other.fptr_;
        owned_ = other.owned_;
        other.fptr_ = nullptr;
        other.owned_ = false;
    }
    return *this;
}

// Improved destructor with error logging
FITSFile::~FITSFile() {
    if (owned_ && fptr_) {
        int status = 0;
        INFO_LOG("FITS file being closed by destructor");
        fits_close_file(fptr_, &status);
        if (status) {
            WARNING_LOG("Error closing FITS file in destructor: " + 
                       fits_status_to_string(status));
        }
    }
}

// Enhanced close method with safety checks
void FITSFile::close() {
    if (owned_ && fptr_) {
        int status = 0;
        INFO_LOG("Closing FITS file");
        fits_close_file(fptr_, &status);
        if (status) {
            WARNING_LOG("Error closing FITS file: " + fits_status_to_string(status));
        }
        fptr_ = nullptr;
        owned_ = false;
    }
}

void FITSFile::move_to_hdu(int hdu_num, int* hdu_type) {
    int status = 0;
    if (fits_movabs_hdu(fptr_, hdu_num, hdu_type, &status)) {
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }
}

int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name) {
    FITSFile fits_file(filename);
    fitsfile* fptr = fits_file.get();
    int status = 0;
    int hdu_num;

    // Try to move to the HDU by name
    if (fits_movnam_hdu(fptr, ANY_HDU, const_cast<char*>(hdu_name.c_str()), 0, &status)) {
        throw_fits_error(status, "Error moving to HDU: " + hdu_name);
    }

    // Get the current HDU number
    if (fits_get_hdu_num(fptr, &hdu_num)) {
        throw std::runtime_error("Error getting HDU number for: " + hdu_name);
    }

    return hdu_num;
}
