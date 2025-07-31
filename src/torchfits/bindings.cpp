#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "fits_reader.h"
#include "fits_utils.h"
#include "cache.h"
#include "wcs_utils.h"
#include "cache.h"
#include "wcs_utils.h"

namespace py = pybind11;

py::object get_header_value(const std::string& filename, int hdu_num, const std::string& key) {
    FITSFileWrapper f(filename);
    int status = 0;
    if (fits_movabs_hdu(f.get(), hdu_num, NULL, &status)) {
        throw_fits_error(status, "Error moving to HDU " + std::to_string(hdu_num));
    }

    char value[FLEN_VALUE];
    if (fits_read_key_str(f.get(), key.c_str(), value, NULL, &status)) {
        if (status == KEY_NO_EXIST) {
            return py::none();
        }
        throw_fits_error(status, "Error reading key '" + key + "'");
    }
    return py::str(value);
}

PYBIND11_MODULE(fits_reader_cpp, m) {
    m.doc() = "Fast FITS reader for PyTorch C++ backend";

    m.def("read", [](py::object filename_or_url, py::object hdu, py::object start, py::object shape, py::object columns, long start_row, py::object num_rows, size_t cache_capacity, py::str device_str) {
        return read_impl(filename_or_url, hdu, start, shape, columns, start_row, num_rows, cache_capacity, device_str);
    },
        py::arg("filename_or_url"),
        py::arg("hdu") = 1,
        py::arg("start") = py::none(),
        py::arg("shape") = py::none(),
        py::arg("columns") = py::none(),
        py::arg("start_row") = 0,
        py::arg("num_rows") = py::none(),
        py::arg("cache_capacity") = 0,
        py::arg("device") = "cpu",
        "Reads data from a FITS file."
    );

    m.def("get_header", [](const std::string& filename, py::object hdu_spec) {
        int hdu_num = 1;
        if (py::isinstance<py::str>(hdu_spec)) {
            hdu_num = get_hdu_num_by_name(filename, hdu_spec.cast<std::string>());
        } else if (py::isinstance<py::int_>(hdu_spec)) {
            hdu_num = hdu_spec.cast<int>();
        }
        return get_header(filename, hdu_num);
    }, py::arg("filename"), py::arg("hdu_spec"), "Get FITS header.");

    m.def("get_dims", [](const std::string& filename, py::object hdu_spec) {
        int hdu_num = 1;
        if (py::isinstance<py::str>(hdu_spec)) {
            hdu_num = get_hdu_num_by_name(filename, hdu_spec.cast<std::string>());
        } else if (py::isinstance<py::int_>(hdu_spec)) {
            hdu_num = hdu_spec.cast<int>();
        }
        return get_dims(filename, hdu_num);
    }, py::arg("filename"), py::arg("hdu_spec"), "Get the dimensions of a FITS image/cube HDU.");

    m.def("get_num_hdus", &get_num_hdus, py::arg("filename"), "Get the number of HDUs in the FITS file.");

    m.def("get_hdu_type", [](const std::string& filename, py::object hdu_spec) {
        int hdu_num = 1;
        if (py::isinstance<py::str>(hdu_spec)) {
            hdu_num = get_hdu_num_by_name(filename, hdu_spec.cast<std::string>());
        } else if (py::isinstance<py::int_>(hdu_spec)) {
            hdu_num = hdu_spec.cast<int>();
        }
        return get_hdu_type(filename, hdu_num);
    }, py::arg("filename"), py::arg("hdu_spec"), "Get the HDU type.");

    m.def("get_header_value", [](const std::string& filename, py::object hdu_spec, const std::string& key) {
        int hdu_num = 1;
        if (py::isinstance<py::str>(hdu_spec)) {
            hdu_num = get_hdu_num_by_name(filename, hdu_spec.cast<std::string>());
        } else if (py::isinstance<py::int_>(hdu_spec)) {
            hdu_num = hdu_spec.cast<int>();
        }
        return get_header_value(filename, hdu_num, key);
    }, py::arg("filename"), py::arg("hdu_spec"), py::arg("key"), "Get the value of a single header keyword.");

    m.def("_clear_cache", &clear_cache, "Clear the FITS file cache.");

    m.def("world_to_pixel", &world_to_pixel, py::arg("world_coords"), py::arg("header"), "Convert world coordinates to pixel coordinates.");
    m.def("pixel_to_world", &pixel_to_world, py::arg("pixel_coords"), py::arg("header"), "Convert pixel coordinates to world coordinates.");
}
