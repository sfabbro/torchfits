#include "fits_reader.h"  // Include our header file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector, std::map, etc.
#include <pybind11/numpy.h> // Not strictly needed, but good practice

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read", &read,
          "Read FITS data (image or table) with optional cutout.",
          py::arg("filename_with_cutout"),
          py::arg("hdu") = py::none(),
          py::arg("start") = py::none(),
          py::arg("shape") = py::none());

    m.def("get_header", &get_header,
          "Get the FITS header as a dictionary.",
          py::arg("filename"), py::arg("hdu_num"));

    m.def("get_dims", &get_dims,
          "Get the dimensions of a FITS image/cube.",
          py::arg("filename"), py::arg("hdu_num"));

    m.def("get_header_value", &get_header_value,
          "Get a single header keyword value.",
          py::arg("filename"), py::arg("hdu_num"), py::arg("key"));

    m.def("get_hdu_type", &get_hdu_type,
          "Get the HDU type (IMAGE, BINTABLE, TABLE).",
          py::arg("filename"), py::arg("hdu_num"));

    m.def("get_num_hdus", &get_num_hdus,
          "Get the total number of HDUs in the FITS file.",
          py::arg("filename"));

    // Expose helper functions for testing (optional, but good practice)
    m.def("_parse_header_card", &parse_header_card, "Parses a FITS header card (internal use).");
    m.def("_fits_status_to_string", &fits_status_to_string, "Converts CFITSIO status code to string.");

}