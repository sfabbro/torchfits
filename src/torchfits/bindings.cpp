#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <fitsio.h>
#include "fits_reader.h"
#include "fits_utils.h"
#include "wcs_utils.h"
#include "cache.h"

namespace py = pybind11;

// Helper function to resolve HDU (name or number)
int resolve_hdu(const std::string& filename, const pybind11::object& hdu) {
    if (hdu.is_none()) {
        return 1; // Default to primary HDU (HDU 1)
    } else if (py::isinstance<py::str>(hdu)) {
        return get_hdu_num_by_name(filename, hdu.cast<std::string>());
    } else if (py::isinstance<py::int_>(hdu)) {
        int hdu_num = hdu.cast<int>();
        if (hdu_num < 0) {
            throw py::value_error("HDU number must be >= 0");
        }
        return hdu_num;
    } else {
        throw py::type_error("HDU must be None, an integer, or a string.");
    }
}

PYBIND11_MODULE(fits_reader_cpp, m) {
    m.doc() = "Fast FITS reader for PyTorch";

   //Expose FITSFile
    py::class_<FITSFile, std::shared_ptr<FITSFile>>(m, "FITSFile")
        .def(py::init<const std::string&>(),py::arg("filename"))
        .def("move_to_hdu", &FITSFile::move_to_hdu, py::arg("hdu_num"), py::arg("hdu_type") = py::none())
        .def("get", &FITSFile::get)
        .def("close", &FITSFile::close);

    m.def("read", &read_impl,
        py::arg("filename_or_url"),
        py::arg("hdu") = py::none(),
        py::arg("start") = py::none(),
        py::arg("shape") = py::none(),
        py::arg("columns") = py::none(),
        py::arg("start_row") = 0,
        py::arg("num_rows") = py::none(),
        py::arg("cache_capacity") = 0,
        py::arg("device") = "cpu",
        "Reads data from a FITS file (image or table) into a PyTorch tensor."
    );

    m.def("get_header", [](const std::string& filename, pybind11::object hdu) {
        return get_header(filename, resolve_hdu(filename, hdu));
    }, py::arg("filename"), py::arg("hdu") = py::int_(1), "Get FITS header.");

    m.def("get_header_by_name", &get_header_by_name, py::arg("filename"), py::arg("hdu_name"), "Get FITS header.");
    m.def("get_header_by_number", &get_header_by_number, py::arg("filename"), py::arg("hdu_num"), "Get FITS header.");

    m.def("get_dims", [](const std::string& filename, pybind11::object hdu) {
        return get_dims(filename, resolve_hdu(filename, hdu));
    }, py::arg("filename"), py::arg("hdu") = py::int_(1), "Get the dimensions of a FITS HDU.");

    m.def("get_header_value", [](const std::string& filename, pybind11::object hdu, const std::string& key) {
        return get_header_value(filename, resolve_hdu(filename, hdu), key);
    }, py::arg("filename"), py::arg("hdu") = py::int_(1), py::arg("key"), "Get the value of a specific FITS header keyword.");
        
    m.def("get_hdu_type", [](const std::string& filename, pybind11::object hdu) {
        return get_hdu_type(filename, resolve_hdu(filename, hdu));
    }, py::arg("filename"), py::arg("hdu") = py::int_(1), "Get the type of a specific HDU.");

    m.def("get_num_hdus", &get_num_hdus, py::arg("filename"), "Get the number of HDUs in the FITS file.");

    // Expose cache management - keep only ONE definition
    m.def("clear_cache", []() {
        if (cache) {
            cache->clear();
        }
    }, "Clears the LRU cache");

    // Expose WCS functions
    m.def("world_to_pixel", &world_to_pixel,
        py::arg("world_coords"),
        py::arg("header"),
        "Converts world coordinates to pixel coordinates using WCS information");

    m.def("pixel_to_world", &pixel_to_world,
        py::arg("pixel_coords"),
        py::arg("header"),
        "Converts pixel coordinates to world coordinates using WCS information");

    // Expose cache functionality through a class, but don't use the same name twice
    py::class_<LRUCache>(m, "LRUCache")
        .def("clear", &LRUCache::clear, "Clear the cache")
        .def("size", &LRUCache::size, "Get current cache size in MB")
        .def("capacity", &LRUCache::capacity, "Get maximum cache capacity in MB");

    // Expose the global cache instance with a different name
    if (cache) {
        m.attr("global_cache") = cache.get();
    }

}
