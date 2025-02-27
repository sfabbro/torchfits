#ifndef TORCHFITS_FITS_UTILS_H
#define TORCHFITS_FITS_UTILS_H

#include <fitsio.h>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

// FITS file handling RAII class
class FITSFile {
public:
    explicit FITSFile(const std::string& filename);
    ~FITSFile();
    
    fitsfile* get() const { return fptr_; }
    void move_to_hdu(int hdu_num, int* hdu_type = nullptr);
     void close();
    
    // No copying
    FITSFile(const FITSFile&) = delete;
    FITSFile& operator=(const FITSFile&) = delete;
    
private:
    fitsfile* fptr_ = nullptr;
};

// Utility functions
std::string fits_status_to_string(int status);
void throw_fits_error(int status, const std::string& message);
std::pair<std::string, std::string> parse_header_card(const char* card);
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num);
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key);
std::string get_hdu_type(const std::string& filename, int hdu_num);
int get_num_hdus(const std::string& filename);
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num);
std::vector<long long> get_dims(const std::string& filename, int hdu_num);
int get_hdu_num_by_name(const std::string& filename, const std::string& hdu_name);

#endif // TORCHFITS_FITS_UTILS_H
