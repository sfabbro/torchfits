#include "fits_reader.h"  // Include the header file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::map, etc.
#include <pybind11/numpy.h> // Not strictly required, but good practice

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read", &read,
          "Read FITS data (image or table) with optional cutout.\n\n"
          "Args:\n"
          "    filename_with_cutout (str): Path to the FITS file, optionally with a cutout specification.\n"
          "    hdu (int or str, optional): HDU number (1-based) or name (string). Defaults to the primary HDU.\n"
          "    start (list[int], optional): Starting pixel coordinates (0-based) for a cutout.\n"
          "    shape (list[int], optional): Shape of the cutout. Use -1 for a dimension to read to the end.\n\n"
          "Returns:\n"
          "    Union[Tuple[torch.Tensor, Dict[str, str]], Dict[str, torch.Tensor]]:\n"
          "        A tuple (data, header) for image/cube HDUs, or a dictionary for table HDUs."
          ,
          py::arg("filename_with_cutout"),
          py::arg("hdu") = py::none(),
          py::arg("start") = py::none(),
          py::arg("shape") = py::none());

    m.def("get_header", &get_header,
          "Get the FITS header as a dictionary.\n\n"
          "Args:\n"
          "    filename (str): Path to the FITS file.\n"
          "    hdu_num (int or str): HDU number (1-based) or name.\n\n"
          "Returns:\n"
          "    Dict[str, str]: A dictionary of header keywords and values.",
          py::arg("filename"), py::arg("hdu_num"));

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

    m.def("get_hdu_type", &get_hdu_type,
          "Get the HDU type (IMAGE, BINTABLE, TABLE).\n\n"
          "Args:\n"
          "    filename (str): Path to the FITS file.\n"
          "    hdu_num (int or str): HDU number (1-based) or name.\n\n"
          "Returns:\n"
          "    str: The HDU type.",
          py::arg("filename"), py::arg("hdu_num"));

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
}