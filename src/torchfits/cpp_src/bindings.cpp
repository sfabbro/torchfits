#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>

#include <Python.h>
#include <memory>
#include <cstring>

#include <torch/torch.h>
// #include <torch/csrc/autograd/python_variable.h> // Removed to avoid pybind11 conflict
#include <ATen/DLConvertor.h>
#undef READONLY

#include "torch_compat.h"

#include <fitsio.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <vector>
#include <unordered_map>

namespace nb = nanobind;

#include "hardware.h"
#include "fits.cpp"
#include "wcs.cpp"
#include "table.cpp"
#include "hardware.cpp"
#include "fast_io.cpp"

NB_MODULE(cpp, m) {
    nb::class_<torchfits::FITSFileV2>(m, "FITSFile")
        .def(nb::init<const char*, int>(), nb::arg("filename"), nb::arg("mode") = 0)
        .def("read_image", [](torchfits::FITSFileV2& self, int hdu_num, bool use_mmap) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_image(hdu_num, use_mmap);
            }
            return tensor_to_python(tensor);
        }, nb::arg("hdu_num"), nb::arg("use_mmap") = false)
        .def("read_header", &torchfits::FITSFileV2::get_header)
        .def("get_num_hdus", &torchfits::FITSFileV2::get_num_hdus)
        .def("get_hdu_type", &torchfits::FITSFileV2::get_hdu_type)
        .def("close", &torchfits::FITSFileV2::close)
        .def("write_image", [](torchfits::FITSFileV2& self, nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
            // nb::gil_scoped_release release; // nb::ndarray access might need GIL?
            return self.write_image(tensor, hdu_num, bscale, bzero);
        }, nb::arg("tensor"), nb::arg("hdu_num") = 0, nb::arg("bscale") = 1.0, nb::arg("bzero") = 0.0)
        .def("write_hdus", &torchfits::FITSFileV2::write_hdus)
        .def("compute_stats", &torchfits::FITSFileV2::compute_stats)
        .def("get_shape", &torchfits::FITSFileV2::get_shape)
        .def("get_dtype", &torchfits::FITSFileV2::get_dtype)
        .def("read_subset", [](torchfits::FITSFileV2& self, int hdu_num, long x1, long y1, long x2, long y2) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_subset(hdu_num, x1, y1, x2, y2);
            }
            return tensor_to_python(tensor);
        });

    m.def("read_full", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torchfits::FITSFileV2 file(filename.c_str(), 0);
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full", [](torchfits::FITSFileV2& file, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("file"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("compute_stats", [](torchfits::FITSFileV2& file, int hdu_num) {
        return file.compute_stats(hdu_num);
    });
    
    m.def("write_fits_file", [](const std::string& path, nb::list hdus, bool overwrite) {
        std::string final_path = path;
        if (overwrite) {
            final_path = "!" + path;
        }
        torchfits::FITSFileV2 file(final_path.c_str(), 1);
        file.write_hdus(hdus, overwrite);
    });

    m.def("write_fits_table", [](const std::string& filename, nb::dict tensor_dict, nb::dict header, bool overwrite) {
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite);
    });

    // Table operations with GIL release
    m.def("read_fits_table", [](const std::string& filename, int hdu_num) -> nb::object {
        // Release GIL for I/O operations
        nb::gil_scoped_release release;
        
        torchfits::TableReader reader(filename, hdu_num);
        // read_columns now returns C++ types (ColumnData), so it's safe to run without GIL
        auto result_map = reader.read_columns({}, 1, -1);
        
        // Acquire GIL to create Python objects
        nb::gil_scoped_acquire acquire;

        nb::dict result_dict;
        for (auto& [key, col_data] : result_map) {
            if (col_data.is_vla) {
                nb::list vla_list;
                for (const auto& tensor : col_data.vla_data) {
                    vla_list.append(tensor_to_python(tensor));
                }
                result_dict[key.c_str()] = vla_list;
            } else {
                result_dict[key.c_str()] = tensor_to_python(col_data.fixed_data);
            }
        }
        return result_dict;
    });

    m.def("read_fits_table_from_handle", [](torchfits::FITSFileV2& file, int hdu_num) -> nb::object {
        // try {
            nb::gil_scoped_release release;
            // Assuming TableReader has a constructor taking fitsfile*
            // If not, we need to add it.
            // For now, let's assume it exists or we will fix it in table.cpp
            torchfits::TableReader reader(file.get_fptr(), hdu_num);
            auto result_map = reader.read_columns({}, 1, -1);
            nb::gil_scoped_acquire acquire;
            
            nb::dict result_dict;
            for (auto& [key, col_data] : result_map) {
                if (col_data.is_vla) {
                    nb::list vla_list;
                    for (const auto& tensor : col_data.vla_data) {
                        vla_list.append(tensor_to_python(tensor));
                    }
                    result_dict[key.c_str()] = vla_list;
                } else {
                    result_dict[key.c_str()] = tensor_to_python(col_data.fixed_data);
                }
            }
            return result_dict;
        // } catch (const std::exception& e) {
        //     return nb::cast(nb::dict());
        // }
    });

    m.def("read_header_dict", [](const std::string& filename, int hdu_num) -> nb::dict {
        try {
            nb::gil_scoped_release release;
            // READONLY is usually 0 in cfitsio
            torchfits::FITSFileV2 file(filename.c_str(), 0);
            auto header = file.get_header(hdu_num);
            nb::gil_scoped_acquire acquire;
            nb::dict result;
            for (const auto& [key, value] : header) {
                result[key.c_str()] = value.c_str();
            }
            return result;
        } catch (const std::exception& e) {
            return nb::dict();
        }
    });

    m.def("read_fits_table", [](const std::string& filename, int hdu_num, const std::vector<std::string>& column_names, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        torchfits::TableReader reader(filename, hdu_num);
        
        if (mmap) {
            // Memory mapped reading (returns numpy arrays)
            // Must acquire GIL because read_columns_mmap creates Python objects
            nb::gil_scoped_acquire acquire;
            return reader.read_columns_mmap(column_names);
        } else {
            // Standard reading (returns tensors)
            auto result_map = reader.read_columns(column_names);
            nb::gil_scoped_acquire acquire;
            
            nb::dict result_dict;
            for (auto& [name, col_data] : result_map) {
                if (col_data.is_vla) {
                    // Convert vector<Tensor> to list
                    nb::list vla_list;
                    for (const auto& t : col_data.vla_data) {
                        vla_list.append(tensor_to_python(t));
                    }
                    result_dict[name.c_str()] = vla_list;
                } else {
                    // Convert Tensor to python object
                    result_dict[name.c_str()] = tensor_to_python(col_data.fixed_data);
                }
            }
            return nb::object(result_dict);
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1, nb::arg("column_names") = std::vector<std::string>(), nb::arg("mmap") = false);

    nb::class_<torchfits::HDUInfo>(m, "HDUInfo")
        .def_prop_rw("index", [](torchfits::HDUInfo& t) { return t.index; }, [](torchfits::HDUInfo& t, int v) { t.index = v; })
        .def_prop_rw("type", [](torchfits::HDUInfo& t) { return t.type; }, [](torchfits::HDUInfo& t, std::string v) { t.type = v; })
        .def_prop_rw("header", [](torchfits::HDUInfo& t) { return t.header; }, [](torchfits::HDUInfo& t, std::unordered_map<std::string, std::string> v) { t.header = v; });

    m.def("open_and_read_headers", [](const std::string& path, int mode) {
        nb::gil_scoped_release release;
        auto result = torchfits::open_and_read_headers(path, mode);
        nb::gil_scoped_acquire acquire;
        
        // Explicitly transfer ownership of the file pointer to Python
        nb::object file_obj = nb::cast(result.first, nb::rv_policy::take_ownership);
        nb::object infos_obj = nb::cast(result.second);
        
        return nb::make_tuple(file_obj, infos_obj);
    });

    // Fast I/O bindings
    m.def("read_image_fast", &torchfits::read_image_fast, 
          nb::arg("filename"), nb::arg("hdu_num") = 0, nb::arg("use_mmap") = true);
    
    m.def("open_fits_file", [](const std::string& path, const std::string& mode) {
        int mode_int = (mode == "w" || mode == "w+") ? 1 : 0;
        return new torchfits::FITSFileV2(path.c_str(), mode_int);
    }, nb::rv_policy::take_ownership);
    
    // close_fits_file removed, handled by destructor
    
    m.def("read_header", [](torchfits::FITSFileV2& file, int hdu_num) {
        return file.get_header(hdu_num);
    });

    m.def("get_num_hdus", [](torchfits::FITSFileV2& file) {
        return file.get_num_hdus();
    });

    m.def("get_hdu_type", [](torchfits::FITSFileV2& file, int hdu_num) {
        return file.get_hdu_type(hdu_num);
    });
    
    m.def("read_image_from_handle", [](torchfits::FITSFileV2& file, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image(hdu_num);
        }
        return tensor_to_python(tensor);
    });

    m.def("read_images_batch", [](const std::vector<std::string>& paths, int hdu_num) {
        nb::gil_scoped_release release;
        auto tensors = torchfits::read_images_batch(paths, hdu_num);
        nb::gil_scoped_acquire acquire;
        
        nb::list result;
        for (const auto& t : tensors) {
            result.append(tensor_to_python(t));
        }
        return result;
    });

    m.def("echo_tensor", [](nb::object obj) {
        return obj;
    });
}
