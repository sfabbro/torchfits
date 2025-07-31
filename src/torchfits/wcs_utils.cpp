#include "fits_utils.h"
#include "wcs_utils.h"
#include "debug.h"
#include <cstring>
#include <sstream>
#include <wcslib/wcshdr.h>

// Define a consistent deleter for the wcsprm struct to be used in unique_ptr.
// This ensures that wcsfree is always called, preventing memory leaks.
auto wcs_deleter = [](wcsprm* p) {
    if (p) {
        wcsfree(p);
        // wcsprm is allocated with malloc by wcslib, so we must use free.
        free(p);
    }
};

// Type alias for our unique_ptr with a custom deleter.
//using WcsUniquePtr = std::unique_ptr<wcsprm, decltype(wcs_deleter)>;

// Internal function to create a WCS object from a raw header string.
// This is the core function that handles parsing and error checking.
WcsUniquePtr _create_wcs_from_header_string(const std::string& header_str, bool throw_on_error) {
    if (header_str.length() % 80 != 0) {
        std::string error_msg = "Invalid header length: " + std::to_string(header_str.length()) + " (must be a multiple of 80)";
        if (throw_on_error) {
            throw std::runtime_error(error_msg);
        } else {
            WARNING_LOG(error_msg);
            return nullptr;
        }
    }

    int nkeyrec = header_str.length() / 80;
    int nreject = 0;
    int nwcs = 0;
    wcsprm* wcs = nullptr;

    // wcspih allocates memory for the wcsprm struct.
    int status = wcspih(const_cast<char*>(header_str.c_str()), nkeyrec, WCSHDR_all, 0, &nreject, &nwcs, &wcs);

    // Wrap the raw pointer in a unique_ptr immediately to ensure cleanup.
    WcsUniquePtr wcs_ptr(wcs, wcs_deleter);

    if (status != 0 || nwcs == 0 || !wcs) {
        // Error occurred or no WCS keywords found. The unique_ptr will handle cleanup.
        if (throw_on_error) {
            throw std::runtime_error("Failed to create WCS from header: wcspih status=" + std::to_string(status) + ", nwcs=" + std::to_string(nwcs));
        }
        return nullptr;
    }

    // wcsset initializes the struct from the parsed keywords.
    if (wcsset(wcs_ptr.get()) != 0) {
        // Error during initialization. The unique_ptr will handle cleanup.
        if (throw_on_error) {
            throw std::runtime_error("Failed to initialize WCS structure with wcsset");
        }
        return nullptr;
    }

    // Return the fully initialized and valid WCS object.
    return wcs_ptr;
}

// Reads WCS from an open FITS file.
WcsUniquePtr read_wcs_from_header(fitsfile* fptr) {
    int status = 0;
    int nkeys = 0;
    char* header_str = nullptr;

    if (fits_hdr2str(fptr, 1, NULL, 0, &header_str, &nkeys, &status)) {
        throw_fits_error(status, "Error converting FITS header to string");
    }

    // Ensure header_str is freed.
    std::string header(header_str);
    free(header_str);

    return _create_wcs_from_header_string(header, true);
}

// Creates a WCS object from a map of header key-value pairs.
WcsUniquePtr create_wcs_from_header(const std::map<std::string, std::string>& header, bool throw_on_error) {
    std::stringstream header_stream;
    for (const auto& [key, value] : header) {
        std::string card = key;
        card.resize(8, ' '); // Pad key to 8 characters.
        card += "= " + value;
        card.resize(80, ' '); // Pad card to 80 characters.
        header_stream << card;
    }
    return _create_wcs_from_header_string(header_stream.str(), throw_on_error);
}

// --- Coordinate Transformation Functions ---

// Template function to perform coordinate transformations.
template<typename Func>
std::tuple<torch::Tensor, torch::Tensor> transform_coords(
    torch::Tensor coords,
    const std::map<std::string, std::string>& header,
    Func wcs_func
) {
    if (!coords.defined() || coords.numel() == 0) {
        throw std::runtime_error("Input coordinates tensor is empty");
    }
    if (coords.dim() != 2) {
        throw std::runtime_error("Input coordinates tensor must be 2D (N_coords x N_dims)");
    }
    if (header.empty()) {
        throw std::runtime_error("Header is empty, cannot perform WCS conversion");
    }

    auto wcs = create_wcs_from_header(header, true);
    if (!wcs) {
        throw std::runtime_error("Failed to parse WCS from header");
    }

    int ncoords = coords.size(0);
    int nelem = coords.size(1);

    torch::Tensor out_coords = torch::empty({ncoords, nelem}, torch::kFloat64);
    torch::Tensor status = torch::empty({ncoords}, torch::kInt32);

    double* in_ptr = coords.to(torch::kFloat64).contiguous().data_ptr<double>();
    double* out_ptr = out_coords.data_ptr<double>();
    int* status_ptr = status.data_ptr<int>();

    std::vector<double> imgcrd(nelem), phi(nelem), theta(nelem);

    for (int i = 0; i < ncoords; ++i) {
        int stat_val;
        int result = wcs_func(
            wcs.get(), 1, nelem,
            in_ptr + i * nelem,
            imgcrd.data(), phi.data(), theta.data(),
            out_ptr + i * nelem,
            &stat_val
        );
        status_ptr[i] = (result != 0) ? result : stat_val;
    }

    return std::make_tuple(out_coords, status);
}

// Convert world coordinates to pixel coordinates.
std::tuple<torch::Tensor, torch::Tensor> world_to_pixel(
    torch::Tensor world_coords,
    const std::map<std::string, std::string>& header
) {
    return transform_coords(world_coords, header, wcss2p);
}

// Convert pixel coordinates to world coordinates.
std::tuple<torch::Tensor, torch::Tensor> pixel_to_world(
    torch::Tensor pixel_coords,
    const std::map<std::string, std::string>& header
) {
    return transform_coords(pixel_coords, header, wcsp2s);
}