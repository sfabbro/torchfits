#include "fits_utils.h"  // For helper functions (like error handling)
#include "wcs_utils.h"
#include <sstream>
#include <wcslib/wcshdr.h>  // For wcspih

// Helper function for WCS struct cleanup
struct WCSDeleter {
    void operator()(struct wcsprm* wcs) {
        if (wcs) {
            wcsfree(wcs);
            free(wcs);
        }
    }
};

// Reads the WCS keywords from FITS header, constructs a wcsprm object.
std::unique_ptr<wcsprm> read_wcs_from_header(fitsfile* fptr) {
    int status = 0;
    char card[FLEN_CARD];
    std::stringstream wcs_header_stream;
    const char* wcs_keys[] = {"CTYPE", "CRPIX", "CRVAL", "CDELT", "PC", "CD", "NAXIS", nullptr}; // Null-terminated
    int key_index = 0;

    // Loop to add the basics to the header (to work with wcslib)
    while (wcs_keys[key_index] != nullptr) {
        const char* key = wcs_keys[key_index];
        int exact = 0;
        //Use CFITSIO function to read keywords
        if (fits_read_card(fptr, key, card, &status))
        {
            if (status == KEY_NO_EXIST) {
                //If a key doesn't exist, it's not a fatal error for WCS.
                //Just move to the next key. WCS can often still be determined.
                status = 0; // Reset the status
                key_index++;
                continue;
            }
            else{
               throw_fits_error(status, std::string("Error reading WCS key: ") + key);
            }
        }
        
        // Add header card and move to next key
        wcs_header_stream << card << "\n";
        key_index++;
    }

    // Create the WCS structure from the header
    int nkeyrec = 0, nreject = 0, nwcs = 0;
    wcsprm* wcs = nullptr;
    std::string header_str = wcs_header_stream.str();
    std::vector<char> header_cards(header_str.begin(), header_str.end());
    header_cards.push_back('\0');
    nkeyrec = header_cards.size() / 80;
    
    int wcs_status = wcspih(header_cards.data(), nkeyrec, WCSHDR_all, 0, &nreject, &nwcs, &wcs);

    // Define a consistent deleter for wcsprm
    auto wcs_deleter = [](wcsprm* p) {
        if (p) {
            wcsfree(p);
            free(p);
        }
    };

    // Improved error handling with consistent approach
    if (wcs_status != 0 || nwcs == 0) {
        if (wcs) wcs_deleter(wcs);
        throw std::runtime_error("Failed to create WCS from header: status=" + 
                                  std::to_string(wcs_status) + ", nwcs=" + std::to_string(nwcs));
    }
    
    if (wcsset(wcs) != 0) {
        wcs_deleter(wcs);
        throw std::runtime_error("Failed to initialize WCS structure");
    }

    return std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>(wcs, wcs_deleter);
}

// Standardized WCS deleter function that can be reused
auto wcs_deleter = [](wcsprm* p) {
    if (p) {
        wcsfree(p);
        free(p);
    }
};

// Improved create_wcs_from_header with consistent error handling
std::unique_ptr<wcsprm> create_wcs_from_header(
    const std::map<std::string, std::string>& header, 
    bool throw_on_error) {
    
    // Build header string from map
    std::stringstream header_stream;
    for (const auto& [key, value] : header) {
        std::string card = key;
        // Pad key to 8 characters
        card.resize(8, ' ');
        card += "= " + value;
        // Pad card to 80 characters
        card.resize(80, ' ');
        header_stream << card;
    }
    
    std::string header_str = header_stream.str();
    if (header_str.length() % 80 != 0) {
        std::string error_msg = "Invalid header length: " + std::to_string(header_str.length()) + 
                               " (must be multiple of 80)";
        if (throw_on_error) {
            throw std::runtime_error(error_msg);
        } else {
            WARNING_LOG(error_msg);
            return nullptr;
        }
    }
    
    int nkeyrec = header_str.length() / 80;
    int nreject = 0, nwcs = 0;
    wcsprm* wcs = nullptr;
    
    // Parse header
    int wcs_status = wcspih(header_str.c_str(), nkeyrec, WCSHDR_all, 0, &nreject, &nwcs, &wcs);
    
    // Handle errors consistently
    if (wcs_status == 0 && nwcs > 0 && wcs != nullptr) {
        if (wcsset(wcs) != 0) {
            if (throw_on_error) {
                wcs_deleter(wcs);
                throw std::runtime_error("Failed to initialize WCS structure");
            } else {
                WARNING_LOG("Failed to initialize WCS structure");
                wcs_deleter(wcs);
                return nullptr;
            }
        }
        
        return std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>(wcs, wcs_deleter);
    } else {
        if (wcs) {
            wcs_deleter(wcs);
        }
        
        if (throw_on_error) {
            std::string error_msg = "Failed to create WCS from header: status=" + 
                                   std::to_string(wcs_status) + ", nwcs=" + std::to_string(nwcs);
            throw std::runtime_error(error_msg);
        } else {
            WARNING_LOG("Failed to create WCS from header: status=" + 
                       std::to_string(wcs_status) + ", nwcs=" + std::to_string(nwcs));
            return nullptr;
        }
    }
}

// Helper function to create WCS from header map - add this BEFORE world_to_pixel function
std::unique_ptr<wcsprm, std::function<void(wcsprm*)>> read_wcs_from_header_map(
    const std::map<std::string, std::string>& header) 
{
    // Create a custom WCS deleter
    auto deleter = [](wcsprm* p) {
        if (p) {
            wcsfree(p);
            free(p);
        }
    };

    // Convert header map to raw character cards
    std::vector<char> header_cards;
    int nkeyrec = 0;
    
    for (const auto& [key, value] : header) {
        std::string card;
        
        // Skip keys that aren't valid WCS keywords
        // Only include keys that are likely to be WCS-related
        if (key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" || 
            key.substr(0, 5) == "NAXIS" || key.substr(0, 2) == "CD" ||
            key.substr(0, 4) == "CTYP" || key.substr(0, 4) == "CUNI" ||
            key.substr(0, 5) == "CRPIX" || key.substr(0, 5) == "CRVAL" ||
            key.substr(0, 5) == "CDELT" || key.substr(0, 5) == "CROTA" ||
            key.substr(0, 2) == "PC" || key.substr(0, 2) == "PV" ||
            key.substr(0, 5) == "LONPOLE" || key.substr(0, 5) == "LATPOLE" || 
            key.substr(0, 3) == "MJD" || key.substr(0, 5) == "EQUINOX" ||
            key.substr(0, 5) == "RADESYS") {
            
            // Format the card as "KEY     = 'VALUE'"
            card = key;
            while (card.length() < 8) {
                card += " ";
            }
            
            // For string values, add quotes
            card += "= ";
            if (value.find_first_not_of("0123456789.+-Ee") != std::string::npos) {
                card += "'" + value + "'";
            } else {
                card += value;
            }
            
            // Pad to 80 characters
            while (card.length() < 80) {
                card += " ";
            }
            
            // Add the card to our collection
            for (char c : card) {
                header_cards.push_back(c);
            }
            nkeyrec++;
        }
    }
    
    // Add END card
    std::string end_card = "END";
    while (end_card.length() < 80) {
        end_card += " ";
    }
    for (char c : end_card) {
        header_cards.push_back(c);
    }
    nkeyrec++;
    
    // Parse the header with wcslib
    int nreject = 0;
    int nwcs = 0;
    struct wcsprm* wcs = nullptr;
    
    int status = wcspih(header_cards.data(), nkeyrec, WCSHDR_all, 0, &nreject, &nwcs, &wcs);
    
    if (status != 0 || nwcs == 0 || wcs == nullptr) {
        // Clean up and return nullptr
        if (wcs) {
            wcsfree(wcs);
            free(wcs);
        }
        return std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>(nullptr, deleter);
    }
    
    // Initialize the WCS structure
    status = wcsset(wcs);
    if (status != 0) {
        wcsfree(wcs);
        free(wcs);
        return std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>(nullptr, deleter);
    }
    
    // Return the unique_ptr with our custom deleter
    return std::unique_ptr<wcsprm, std::function<void(wcsprm*)>>(wcs, deleter);
}

// Convert world coordinates to pixel coordinates.
std::tuple<torch::Tensor, torch::Tensor> world_to_pixel(
    torch::Tensor world_coords,
    std::map<std::string, std::string> header
) {
    // Validate input
    if (!world_coords.defined() || world_coords.numel() == 0) {
        throw std::runtime_error("World coordinates tensor is empty");
    }
    
    if (world_coords.dim() != 2) {
        throw std::runtime_error("World coordinates tensor must be 2D (N coordinates x M dimensions)");
    }
    
    if (header.empty()) {
        throw std::runtime_error("Header is empty, cannot perform WCS conversion");
    }
    
    // Read WCS from header
    auto wcs = read_wcs_from_header_map(header);
    if (!wcs) {
        throw std::runtime_error("Failed to parse WCS from header");
    }
    
    // Get dimensions
    int ncoords = world_coords.size(0);
    int nelem = world_coords.size(1);
    
    // Prepare output tensors
    torch::Tensor pixel_coords = torch::empty({ncoords, nelem}, torch::kFloat64);
    torch::Tensor status = torch::empty({ncoords}, torch::kInt32);
    
    // Access data pointers
    double* world_ptr = world_coords.data_ptr<double>();
    double* pixel_ptr = pixel_coords.data_ptr<double>();
    int* status_ptr = status.data_ptr<int>();
    
    // Temporary arrays for wcss2p intermediate results
    std::vector<double> imgcrd(nelem);
    std::vector<double> phi(1);
    std::vector<double> theta(1);
    std::vector<int> stat(1);
    
    // Convert each coordinate
    for (int i = 0; i < ncoords; i++) {
        // Call the WCSLIB function to do the conversion
        int result = wcss2p(
            wcs.get(), 
            1,                      // process 1 coordinate at a time
            nelem,                  // number of elements per coordinate
            &world_ptr[i * nelem],  // input world coordinates
            phi.data(),             // intermediate spherical coordinates
            theta.data(),           // intermediate spherical coordinates
            imgcrd.data(),          // intermediate image coordinates
            &pixel_ptr[i * nelem],  // output pixel coordinates
            stat.data()             // status code
        );
        
        status_ptr[i] = result != 0 ? result : stat[0];
    }
    
    return std::make_tuple(pixel_coords, status);
}

// Convert pixel coordinates to world coordinates.
std::tuple<torch::Tensor, torch::Tensor> pixel_to_world(
    torch::Tensor pixel_coords,
    std::map<std::string, std::string> header
) {
    // Validate input
    if (!pixel_coords.defined() || pixel_coords.numel() == 0) {
        throw std::runtime_error("Pixel coordinates tensor is empty");
    }
    
    if (pixel_coords.dim() != 2) {
        throw std::runtime_error("Pixel coordinates tensor must be 2D (N coordinates x M dimensions)");
    }
    
    if (header.empty()) {
        throw std::runtime_error("Header is empty, cannot perform WCS conversion");
    }
    
    // Read WCS from header
    auto wcs = read_wcs_from_header_map(header);
    if (!wcs) {
        throw std::runtime_error("Failed to parse WCS from header");
    }
    
    // Get dimensions
    int ncoords = pixel_coords.size(0);
    int nelem = pixel_coords.size(1);
    
    // Prepare output tensors
    torch::Tensor world_coords = torch::empty({ncoords, nelem}, torch::kFloat64);
    torch::Tensor status = torch::empty({ncoords}, torch::kInt32);
    
    // Access data pointers
    double* pixel_ptr = pixel_coords.data_ptr<double>();
    double* world_ptr = world_coords.data_ptr<double>();
    int* status_ptr = status.data_ptr<int>();
    
    // Temporary arrays for wcsp2s intermediate results
    std::vector<double> imgcrd(nelem);
    std::vector<double> phi(1);
    std::vector<double> theta(1);
    std::vector<int> stat(1);
    
    // Convert each coordinate
    for (int i = 0; i < ncoords; i++) {
        // Call the WCSLIB function to do the conversion
        int result = wcsp2s(
            wcs.get(), 
            1,                      // process 1 coordinate at a time
            nelem,                  // number of elements per coordinate
            &pixel_ptr[i * nelem],  // input pixel coordinates
            imgcrd.data(),          // intermediate image coordinates
            phi.data(),             // intermediate spherical coordinates
            theta.data(),           // intermediate spherical coordinates
            &world_ptr[i * nelem],  // output world coordinates
            stat.data()             // status code
        );
        
        status_ptr[i] = result != 0 ? result : stat[0];
    }
    
    return std::make_tuple(world_coords, status);
}
