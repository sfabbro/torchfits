#include "fits_reader.h"  // For helper functions (like error handling)
#include "wcs_utils.h"
#include <sstream>

// Reads the minimal set of WCS keywords, constructs a wcsprm object.
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
        if (fits_read_key_str(fptr, key, card, nullptr, &status))
        {
              if (status == KEY_NO_EXIST) {
                //If a key doesn't exist, it's not a fatal error for WCS.
                //Just move to the next key.  WCS can often still be determined.
                status = 0; // Reset the status
                key_index++;
                continue;

            }
            else{
               throw_fits_error(status, std::string("Error reading WCS key: ") + key);
            }
        }

        // fits_read_key_str gives at most 70 chars.
        //Pad with spaces up to 80 characters, as required by wcslib
        std::string card_str(card);
        int spaces_to_add = 80 - card_str.length() - strlen(key) - 2; // -2 to account for "= "
        if(spaces_to_add < 0) spaces_to_add = 0; //Just in case
        wcs_header_stream << key << " = " << card_str;
        for (int i = 0; i < spaces_to_add; ++i) {
            wcs_header_stream << " ";
        }

        key_index++;
    }

    // Add END card (required by wcspih).
    wcs_header_stream << "END";
    for (int i = 0; i < 77; ++i) {  // 80 - length("END")
        wcs_header_stream << " ";
    }

    // Parse the header string with wcslib to create the WCS object.
    int nreject, nwcs;
    struct wcsprm* wcs = nullptr;
    const std::string wcs_header_str = wcs_header_stream.str();
    int wcs_status = wcspih(wcs_header_str.c_str(), wcs_header_str.size(), 0, 0, &nreject, &nwcs, &wcs);

    if (wcs_status != 0 || nwcs == 0) {
        wcsfree(wcs, &nwcs); // Always free allocated memory
        return nullptr; // Return nullptr if no valid WCS
    }

    return std::unique_ptr<wcsprm>(wcs); // Return ownership via unique_ptr.
}

// Convert world coordinates to pixel coordinates.
std::pair<torch::Tensor, torch::Tensor> world_to_pixel(const torch::Tensor& world_coords, const std::map<std::string, std::string>& header) {

    //Reconstruct the wcs object from the header
    int nreject, nwcs;
    struct wcsprm* wcs = nullptr;
    std::stringstream header_stream;
    for (const auto& [key, value] : header) {
        header_stream << key;
        // Pad key to 8 characters if needed
        for(int i = key.length(); i < 8; ++i) {
            header_stream << " ";
        }
        header_stream << "= " << value;
        //Calculate the number of spaces needed to pad to 80
        int spaces_to_pad = 80 - (key.length() + 2 + value.length()); // +2 accounts for "= "
        //If there is no space, put at least one.
        if (spaces_to_pad<=0)
            header_stream << " ";
        else{
            for(int i = 0; i<spaces_to_pad; i++)
                header_stream << " ";
        }
    }

    const std::string header_str = header_stream.str();
    int wcs_status = wcspih(header_str.c_str(), header_str.size(), 0, 0, &nreject, &nwcs, &wcs);

    std::unique_ptr<wcsprm> wcs_ptr; //Use a unique_ptr for memory management
    if (wcs_status == 0 && nwcs > 0 && wcs != nullptr) {
        wcs_ptr = std::unique_ptr<wcsprm>(wcs);
    } else {
            wcsfree(wcs, &nwcs); //Free memory, since there is an error
        throw std::runtime_error("WCS parsing failed in world_to_pixel.");
    }
    //End reconstruction

    if (world_coords.size(1) != wcs_ptr->naxis)
    {
        throw std::runtime_error("World coordinates dimensions do not match the number of WCS axes.");
    }

    long npoints = world_coords.size(0);

    // Output arrays
    torch::Tensor pixel_coords = torch::zeros({npoints, wcs_ptr->naxis}, torch::kDouble); //Always use double for wcs transformation
    torch::Tensor imgcrd = torch::zeros({npoints, wcs_ptr->naxis}, torch::kDouble); // Intermediate image coordinates
    torch::Tensor phi = torch::zeros({npoints}, torch::kDouble);     // Intermediate values
    torch::Tensor theta = torch::zeros({npoints}, torch::kDouble);   // Intermediate values
    torch::Tensor stat = torch::zeros({npoints}, torch::kInt);    // Status array

    // Use tensor accessors for efficient access to tensor data.
    auto world_coords_acc = world_coords.accessor<double, 2>();
    auto pixel_coords_acc = pixel_coords.accessor<double, 2>();
    auto imgcrd_acc = imgcrd.accessor<double, 2>();
    auto phi_acc = phi.accessor<double, 1>();
    auto theta_acc = theta.accessor<double, 1>();
    auto stat_acc = stat.accessor<int,1>();

    // Prepare input data for wcslib (convert to a C-style array).
    std::vector<double> world_coords_carray(npoints * wcs_ptr->naxis);
    for (int i = 0; i < npoints; ++i) {
        for (int j = 0; j < wcs_ptr->naxis; ++j) {
            world_coords_carray[i * wcs_ptr->naxis + j] = world_coords_acc[i][j];
        }
    }

    // Perform the WCS transformation using wcslib.
    wcs_status = wcss2p(wcs_ptr.get(), npoints, wcs_ptr->naxis,
                             world_coords_carray.data(), // Input world coordinates
                             imgcrd_acc.data(),      // Output image coordinates (not used)
                             phi_acc.data(), theta_acc.data(),   // Output phi/theta (not used)
                             pixel_coords_acc.data(),  // Output pixel coordinates
                             stat_acc.data());      // Output status

    if (wcs_status != 0) {
        std::stringstream ss;
        ss << "WCS transformation (world_to_pixel) failed with status: " << wcs_status;
        throw std::runtime_error(ss.str());
    }
    return {pixel_coords, stat};
}

std::pair<torch::Tensor, torch::Tensor> pixel_to_world(const torch::Tensor& pixel_coords, const std::map<std::string, std::string>& header) {

    //Reconstruct the wcs object from the header
    int nreject, nwcs;
    struct wcsprm* wcs = nullptr;
    std::stringstream header_stream;
    for (const auto& [key, value] : header) {
        header_stream << key;
        // Pad key to 8 characters if needed
        for(int i = key.length(); i < 8; ++i) {
            header_stream << " ";
        }
        header_stream << "= " << value;
        //Calculate the number of spaces needed to pad to 80
        int spaces_to_pad = 80 - (key.length() + 2 + value.length()); // +2 accounts for "= "
        //If there is no space, put at least one.
        if (spaces_to_pad<=0)
            header_stream << " ";
        else{
            for(int i = 0; i<spaces_to_pad; i++)
                header_stream << " ";
        }
    }

    const std::string header_str = header_stream.str();
    int wcs_status = wcspih(header_str.c_str(), header_str.size(), 0, 0, &nreject, &nwcs, &wcs);

     std::unique_ptr<wcsprm> wcs_ptr; //Use a unique_ptr for memory management
    if (wcs_status == 0 && nwcs > 0 && wcs != nullptr) {
        wcs_ptr = std::unique_ptr<wcsprm>(wcs);
    } else {
        wcsfree(wcs, &nwcs); //Free memory, since there is an error
        throw std::runtime_error("WCS parsing failed in pixel_to_world.");
    }
    //End reconstruction

    if (pixel_coords.size(1) != wcs_ptr->naxis)
    {
        throw std::runtime_error("Pixel coordinates dimensions do not match the number of WCS axes.");
    }

    long npoints = pixel_coords.size(0);

    // Output arrays
    torch::Tensor world_coords = torch::zeros({npoints, wcs_ptr->naxis}, torch::kDouble);
    torch::Tensor imgcrd = torch::zeros({npoints, wcs_ptr->naxis}, torch::kDouble); // Intermediate image coordinates
    torch::Tensor phi = torch::zeros({npoints}, torch::kDouble);        // Intermediate values
    torch::Tensor theta = torch::zeros({npoints}, torch::kDouble);      // Intermediate values
    torch::Tensor stat = torch::zeros({npoints}, torch::kInt); //Status

    // Use tensor accessors for efficient access to tensor data.
    auto pixel_coords_acc = pixel_coords.accessor<double, 2>();
    auto world_coords_acc = world_coords.accessor<double, 2>();
    auto imgcrd_acc = imgcrd.accessor<double, 2>();
    auto phi_acc = phi.accessor<double, 1>();
    auto theta_acc = theta.accessor<double, 1>();
    auto stat_acc = stat.accessor<int,1>();

    // Prepare input for wcslib (convert to a C-style array).
    std::vector<double> pixel_coords_carray(npoints * wcs_ptr->naxis);
    for (int i = 0; i < npoints; ++i) {
        for (int j = 0; j < wcs_ptr->naxis; ++j) {
            pixel_coords_carray[i * wcs_ptr->naxis + j] = pixel_coords_acc[i][j];
        }
    }

    // Perform the WCS transformation using wcslib.
    wcs_status = wcsp2s(wcs_ptr.get(), npoints, wcs_ptr->naxis,
                            pixel_coords_carray.data(), // Input pixel coordinates
                            imgcrd_acc.data(),      // Output image coordinates (not used)
                            phi_acc.data(), theta_acc.data(),  // Output phi and theta (not used)
                            world_coords_acc.data(),     // Output world coordinates
                            stat_acc.data());      // Output status

    if (wcs_status != 0) {
        std::stringstream ss;
        ss << "WCS transformation (pixel_to_world) failed with status: " << wcs_status;
        throw std::runtime_error(ss.str());
    }

    return {world_coords, stat};
}