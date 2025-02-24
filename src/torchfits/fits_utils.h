#ifndef FITS_UTILS_H
#define FITS_UTILS_H

#include <string>
#include <utility>  // For std::pair
#include <vector>
#include <map>
#include <fitsio.h> //For fitsfile and fits functions

// Function prototypes for utility functions (defined in fits_utils.cpp)
std::string fits_status_to_string(int status);
void throw_fits_error(int status, const std::string& message);
std::pair<std::string, std::string> parse_header_card(const char* card);
std::map<std::string, std::string> read_fits_header(fitsfile* fptr);
std::vector<long long> _get_hdu_dims(const std::string& filename, int hdu_num); // Now _get_hdu_dims
std::string get_header_value(const std::string& filename, int hdu_num, const std::string& key);
std::string get_hdu_type(const std::string& filename, int hdu_num);
int get_num_hdus(const std::string& filename);
std::map<std::string, std::string> get_header(const std::string& filename, int hdu_num); //Added wrapper
std::vector<long long> get_dims(const std::string& filename, int hdu_num); //Wrapper

#endif // FITS_UTILS_H