#include "fits_reader.h"  // Include the header file
#include "fits_utils.h"
#include "wcs_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::map, etc.
#include <pybind11/numpy.h> // Not strictly required, but good practice
#include <torch/extension.h>

namespace py = pybind11;

// Add this type alias to make the function type clear
using ReadFunc = pybind11::object (*)(
    pybind11::object, pybind11::object, pybind11::object, 
    pybind11::object, pybind11::object, int, 
    pybind11::object, size_t, torch::Device
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read", &read_impl,
          py::arg("filename_or_url"),
          py::arg("hdu") = py::none(),
          py::arg("start") = py::none(),
          py::arg("shape") = py::none(),
          py::arg("columns") = py::none(),
          py::arg("start_row") = 0,
          py::arg("num_rows") = py::none(),
          py::arg("cache_capacity") = 0,
          py::arg("device") = torch::Device("cpu")
    );

    m.def("get_header", [](const std::string& filename, py::object hdu_spec) {
        if (py::isinstance<py::str>(hdu_spec)) {
            // Handle HDU name
            return get_header_by_name(filename, hdu_spec.cast<std::string>());
        } else {
            // Handle HDU number
            return get_header_by_number(filename, hdu_spec.cast<int>());
        }
    }, "Get FITS header as dictionary", py::arg("filename"), py::arg("hdu_num"));

    m.def("get_dims", &get_dims,
          "Get the dimensions of a FITS image/cube.\n\n"
          "Args:\n"
          "    filename (str): Path to the FITS file.\n"
          "    hdu_num (int or str): HDU number (1-based) or name.\n\n"
          "Returns:\n"
          "    List[int]: A list of dimensions.",
          py::arg("filename"), py::arg("hdu_num"));

    m.def("get_header_value", &get_header_value,
          "Get a single header keyword value.\n\n"
          "Args:\n"
          "    filename (str): Path to the FITS file.\n"
          "    hdu_num (int or str): HDU number (1-based) or name.\n"
          "    key (str): The header keyword.\n\n"
          "Returns:\n"
          "    str: The value of the keyword (empty string if not found).",
          py::arg("filename"), py::arg("hdu_num"), py::arg("key"));

    m.def("get_hdu_type", [](const std::string& filename, py::object hdu_spec) {
        int hdu_num;
        if (py::isinstance<py::str>(hdu_spec)) {
            hdu_num = get_hdu_num_by_name(filename, hdu_spec.cast<std::string>());
        } else {
            hdu_num = hdu_spec.cast<int>();
        }
        
        fitsfile* fptr;
        int status = 0;
        int hdutype;
        
        fits_open_file(&fptr, filename.c_str(), READONLY, &status);
        fits_movabs_hdu(fptr, hdu_num, &hdutype, &status);
        
        std::string type;
        if (hdutype == IMAGE_HDU) {
            type = "IMAGE";
        } else if (hdutype == BINARY_TBL) {
            type = "BINTABLE";
        } else if (hdutype == ASCII_TBL) {
            type = "TABLE";
        } else {
            type = "UNKNOWN";
        }
        
        fits_close_file(fptr, &status);
        return type;
    }, py::arg("filename"), py::arg("hdu_num"));

    m.def("get_num_hdus", &get_num_hdus,
          "Get the total number of HDUs in the FITS file.\n\n"
          "Args:\n"
          "    filename (str): Path to the FITS file.\n\n"
          "Returns:\n"
          "    int: The number of HDUs.",
          py::arg("filename"));

    // Expose helper functions for testing (optional, but good practice)
    m.def("_parse_header_card", &parse_header_card, "Parses a FITS header card (internal use).");
    m.def("_fits_status_to_string", &fits_status_to_string, "Converts CFITSIO status code to string.");
    // Expose cache clearing function
    m.def("_clear_cache", []() {
        DEBUG_LOG("Clearing cache from Python");
        if (cache) {
            cache->clear();
        }
    }, "Clear the internal LRU cache");

    // Expose the new functions for world-to-pixel and pixel-to-world coordinate transformations
    m.def("world_to_pixel", &world_to_pixel,
          "Convert world coordinates to pixel coordinates using WCS information from the header.\n\n"
          "Args:\n"
          "    world_coords (torch.Tensor): Tensor of world coordinates.\n"
          "    header (dict): FITS header dictionary containing WCS information.\n\n"
          "Returns:\n"
          "    Tuple[torch.Tensor, torch.Tensor]: A tuple (pixel_coords, status) where pixel_coords is a tensor of pixel coordinates and status is a tensor of status codes.",
          py::arg("world_coords"), py::arg("header"));

    m.def("pixel_to_world", &pixel_to_world,
          "Convert pixel coordinates to world coordinates using WCS information from the header.\n\n"
          "Args:\n"
          "    pixel_coords (torch.Tensor): Tensor of pixel coordinates.\n"
          "    header (dict): FITS header dictionary containing WCS information.\n\n"
          "Returns:\n"
          "    Tuple[torch.Tensor, torch.Tensor]: A tuple (world_coords, status) where world_coords is a tensor of world coordinates and status is a tensor of status codes.",
          py::arg("pixel_coords"), py::arg("header"));
}
