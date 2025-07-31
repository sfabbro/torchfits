#ifndef TORCHFITS_FITS_UTILS_H
#define TORCHFITS_FITS_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fitsio.h>

// --- RAII Wrapper for fitsfile pointer ---
// Ensures that fits_close_file is called automatically
class FITSFileWrapper {
public:
    FITSFileWrapper(const std::string& filename, int mode = READONLY) {
        int status = 0;
        fits_open_file(&fptr_, filename.c_str(), mode, &status);
        if (status) {
            char err_text[FLEN_ERRMSG];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error opening FITS file '" + filename + "': " + err_text);
        }
    }

    ~FITSFileWrapper() {
        if (fptr_) {
            int status = 0;
            fits_close_file(fptr_, &status);
            // In a destructor, we shouldn't throw exceptions.
            // We could log an error here if needed.
        }
    }

    // Delete copy constructor and assignment operator
    FITSFileWrapper(const FITSFileWrapper&) = delete;
    FITSFileWrapper& operator=(const FITSFileWrapper&) = delete;

    // Allow move construction and assignment
    FITSFileWrapper(FITSFileWrapper&& other) noexcept : fptr_(other.fptr_) {
        other.fptr_ = nullptr;
    }
    FITSFileWrapper& operator=(FITSFileWrapper&& other) noexcept {
        if (this != &other) {
            if (fptr_) {
                int status = 0;
                fits_close_file(fptr_, &status);
            }
            fptr_ = other.fptr_;
            other.fptr_ = nullptr;
        }
        return *this;
    }

    fitsfile* get() const { return fptr_; }

private:
    fitsfile* fptr_ = nullptr;
};


// --- Utility Functions ---

// Throw a C++ exception with a FITS error message
void throw_fits_error(int status, const std::string& message);

// Read FITS header into a map
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);

// Get the number of HDUs in a FITS file
int get_num_hdus(const std::string& filename);

// Get the HDU number by its name
int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name);

// Get the type of a specific HDU
std::string get_hdu_type(const std::string& filename, int hdu_num);

// Get the dimensions of a FITS image/cube HDU
std::vector<long> get_dims(const std::string& filename, int hdu_num);

// Get the full header of a specific HDU
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num);


#endif // TORCHFITS_FITS_UTILS_H
