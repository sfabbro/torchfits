#include "fits_reader.h"  // Include the header file
#include "fits_utils.h"
#include "wcs_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::map, etc.
#include <pybind11/numpy.h> // Not strictly required, but good practice

namespace py = pybind11;

// Need to define this here, *before* we use LRUCache,
// since it's used by the cache.
struct CacheEntry {
    torch::Tensor data;
    std::map<std::string, std::string> header;
};

// Forward declare the LRUCache (so we can use it in _clear_cache)
class LRUCache {
public:
  LRUCache(size_t capacity) {} // Dummy constructor
  void clear() {} // Dummy
};
//Keep reference to the static cache, for the _clear_cache function
static std::unique_ptr<LRUCache> cache = nullptr;



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read", &read,
          "Read FITS data (image or table) with optional cutout, HDU selection, column selection and row selection.\n\n"
          "Args:\n"
          "    filename_or_url (str or dict): Path to the FITS file, or a dictionary with fsspec parameters, or a CFITSIO-compatible URL.\n"
          "    hdu (int or str, optional): HDU number (1-based) or name (string). Defaults to the primary HDU if no cutout string specifies it.\n"
          "    start (list[int], optional): Starting pixel coordinates (0-based) for a cutout.\n"
          "    shape (list[int], optional): Shape of the cutout. Use -1 for a dimension to read to the end.\n"
          "    columns (list[str], optional): List of column names to read from a table. Reads all if None.\n"
          "    start_row (int, optional): Starting row index (0-based) for table reads. Defaults to 0.\n"
          "    num_rows (int, optional): Number of rows to read from a table. Reads all remaining if None.\n"
          "    cache_capacity (int, optional): Capacity of the in-memory cache (in MB). Defaults to automatic sizing (25% of available RAM, up to 2GB).\n"
          "    device (str, optional): Device to place the tensor on ('cpu' or 'cuda'). Defaults to 'cpu'.\n\n"
          "Returns:\n"
          "    Union[Tuple[torch.Tensor, Dict[str, str]], Dict[str, torch::Tensor]]:\n"
          "        A tuple (data, header) for image/cube HDUs, or a dictionary for table HDUs."
          ,
          py::arg("filename_or_url"),
          py::arg("hdu") = py::none(),
          py::arg("start") = py::none(),
          py::arg("shape") = py::none(),
          py::arg("columns") = py::none(),
          py::arg("start_row") = 0,
          py::arg("num_rows") = py::none(),
          py::arg("cache_capacity") = 0,
          py::arg("device") = "cpu" // String default for ease of use
    );


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
    // Expose cache clearing function (for testing)
    m.def("_clear_cache", []() {
        if (cache) {  // Check if cache is initialized
            cache->clear();
        }
    }, "Clears the internal cache (for testing).");

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
