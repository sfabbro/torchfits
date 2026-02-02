#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <Python.h>
#include <memory>
#include <cstring>

#include "torchfits_torch.h"
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

NB_MODULE(cpp, m) {
    nb::class_<torchfits::FITSFile>(m, "FITSFile")
        .def(nb::init<const char*, int>(), nb::arg("filename"), nb::arg("mode") = 0)
        .def("read_image", [](torchfits::FITSFile& self, int hdu_num, bool use_mmap) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_image(hdu_num, use_mmap);
            }
            return tensor_to_python(tensor);
        }, nb::arg("hdu_num"), nb::arg("use_mmap") = true)
        .def("read_header", &torchfits::FITSFile::get_header)
        .def("get_num_hdus", &torchfits::FITSFile::get_num_hdus)
        .def("get_hdu_type", &torchfits::FITSFile::get_hdu_type)
        .def("close", &torchfits::FITSFile::close)
        .def("write_image", [](torchfits::FITSFile& self, nb::ndarray<> tensor, int hdu_num, double bscale, double bzero) {
            // nb::gil_scoped_release release; // nb::ndarray access might need GIL?
            return self.write_image(tensor, hdu_num, bscale, bzero);
        }, nb::arg("tensor"), nb::arg("hdu_num") = 0, nb::arg("bscale") = 1.0, nb::arg("bzero") = 0.0)
        .def("write_hdus", &torchfits::FITSFile::write_hdus)
        .def("compute_stats", &torchfits::FITSFile::compute_stats)
        .def("get_shape", &torchfits::FITSFile::get_shape)
        .def("get_dtype", &torchfits::FITSFile::get_dtype)
        .def("read_subset", [](torchfits::FITSFile& self, int hdu_num, long x1, long y1, long x2, long y2) {
            torch::Tensor tensor;
            {
                nb::gil_scoped_release release;
                tensor = self.read_subset(hdu_num, x1, y1, x2, y2);
            }
            return tensor_to_python(tensor);
        });

    nb::class_<torchfits::TableReader>(m, "TableReader")
        .def(nb::init<const std::string&, int>(), nb::arg("filename"), nb::arg("hdu_num") = 1)
        .def("__init__", [](torchfits::TableReader* self, torchfits::FITSFile& file, int hdu_num) {
            new (self) torchfits::TableReader(file.get_fptr(), hdu_num);
        }, nb::arg("file"), nb::arg("hdu_num") = 1)
        .def("read_rows", [](torchfits::TableReader& self,
                             const std::vector<std::string>& column_names,
                             long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            auto result_map = self.read_columns(column_names, start_row, num_rows);
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
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def_prop_ro("num_rows", &torchfits::TableReader::get_num_rows)
        .def_prop_ro("num_cols", &torchfits::TableReader::get_num_cols);

    m.def("read_full", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torchfits::FITSFile file(filename.c_str(), 0);
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_raw", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torchfits::FITSFile file(filename.c_str(), 0);
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image_raw(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_raw_with_scale", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torchfits::FITSFile file(filename.c_str(), 0);
        torch::Tensor tensor;
        torchfits::FITSFile::ScaleInfo scale_info;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image_raw(hdu_num, use_mmap);
            scale_info = file.get_scale_info_for_hdu(hdu_num);
        }
        return nb::make_tuple(
            tensor_to_python(tensor),
            scale_info.scaled,
            scale_info.bscale,
            scale_info.bzero
        );
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_scaled_cpu", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torchfits::FITSFile file(filename.c_str(), 0);
        torch::Tensor tensor;
        torchfits::FITSFile::ScaleInfo scale_info;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image_raw(hdu_num, use_mmap);
            scale_info = file.get_scale_info_for_hdu(hdu_num);
        }
        if (scale_info.scaled) {
            tensor = tensor.to(torch::kFloat32);
            if (scale_info.bscale != 1.0) {
                tensor.mul_(scale_info.bscale);
            }
            if (scale_info.bzero != 0.0) {
                tensor.add_(scale_info.bzero);
            }
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_unmapped", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_full_unmapped(filename, hdu_num);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_full_unmapped_raw", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_full_unmapped_raw(filename, hdu_num);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"));

    m.def("read_full_nocache", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_full_nocache(filename, hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full", [](torchfits::FITSFile& file, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file.read_image(hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("file"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("compute_stats", [](torchfits::FITSFile& file, int hdu_num) {
        return file.compute_stats(hdu_num);
    });
    
    m.def("write_fits_file", [](const std::string& path, nb::list hdus, bool overwrite) {
        std::string final_path = path;
        if (overwrite) {
            final_path = "!" + path;
        }
        torchfits::FITSFile file(final_path.c_str(), 1);
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

    m.def("read_fits_table_from_handle", [](torchfits::FITSFile& file, int hdu_num) -> nb::object {
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

    m.def("read_fits_table_rows_from_handle", [](torchfits::FITSFile& file, int hdu_num,
                                                 const std::vector<std::string>& column_names,
                                                 long start_row, long num_rows) -> nb::object {
        nb::gil_scoped_release release;
        torchfits::TableReader reader(file.get_fptr(), hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows);
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
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_header_dict", [](const std::string& filename, int hdu_num) -> nb::list {
        try {
            nb::gil_scoped_release release;
            // READONLY is usually 0 in cfitsio
            torchfits::FITSFile file(filename.c_str(), 0);
            auto header = file.get_header(hdu_num);
            nb::gil_scoped_acquire acquire;
            nb::list result;
            for (const auto& item : header) {
                result.append(nb::make_tuple(std::get<0>(item), std::get<1>(item), std::get<2>(item)));
            }
            return result;
        } catch (const std::exception& e) {
            return nb::list();
        }
    });

    m.def("read_fits_table", [](const std::string& filename, int hdu_num, const std::vector<std::string>& column_names, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        if (mmap) {
            torchfits::TableReader reader(filename, hdu_num);
            nb::gil_scoped_acquire acquire;
            return reader.read_columns_mmap(column_names);
        } else {
            torchfits::FITSFile file(filename.c_str(), 0);
            torchfits::TableReader reader(file.get_fptr(), hdu_num);
            auto result_map = reader.read_columns(column_names);
            nb::gil_scoped_acquire acquire;

            nb::dict result_dict;
            for (auto& [name, col_data] : result_map) {
                if (col_data.is_vla) {
                    nb::list vla_list;
                    for (const auto& t : col_data.vla_data) {
                        vla_list.append(tensor_to_python(t));
                    }
                    result_dict[name.c_str()] = vla_list;
                } else {
                    result_dict[name.c_str()] = tensor_to_python(col_data.fixed_data);
                }
            }
            return nb::object(result_dict);
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1, nb::arg("column_names") = std::vector<std::string>(), nb::arg("mmap") = false);

    m.def("read_fits_table_rows", [](const std::string& filename, int hdu_num,
                                     const std::vector<std::string>& column_names,
                                     long start_row, long num_rows, bool mmap) -> nb::object {
        nb::gil_scoped_release release;
        if (mmap) {
            torchfits::TableReader reader(filename, hdu_num);
            nb::gil_scoped_acquire acquire;
            return reader.read_columns_mmap(column_names, start_row, num_rows);
        } else {
            torchfits::FITSFile file(filename.c_str(), 0);
            torchfits::TableReader reader(file.get_fptr(), hdu_num);
            auto result_map = reader.read_columns(column_names, start_row, num_rows);
            nb::gil_scoped_acquire acquire;

            nb::dict result_dict;
            for (auto& [name, col_data] : result_map) {
                if (col_data.is_vla) {
                    nb::list vla_list;
                    for (const auto& t : col_data.vla_data) {
                        vla_list.append(tensor_to_python(t));
                    }
                    result_dict[name.c_str()] = vla_list;
                } else {
                    result_dict[name.c_str()] = tensor_to_python(col_data.fixed_data);
                }
            }
            return nb::object(result_dict);
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    nb::class_<torchfits::HDUInfo>(m, "HDUInfo")
        .def_prop_rw("index", [](torchfits::HDUInfo& t) { return t.index; }, [](torchfits::HDUInfo& t, int v) { t.index = v; })
        .def_prop_rw("type", [](torchfits::HDUInfo& t) { return t.type; }, [](torchfits::HDUInfo& t, std::string v) { t.type = v; })
        .def_prop_rw("header", [](torchfits::HDUInfo& t) { return t.header; }, [](torchfits::HDUInfo& t, std::vector<std::tuple<std::string, std::string, std::string>> v) { t.header = v; });

    m.def("open_and_read_headers", [](const std::string& path, int mode) {
        nb::gil_scoped_release release;
        auto result = torchfits::open_and_read_headers(path, mode);
        nb::gil_scoped_acquire acquire;
        
        // Explicitly transfer ownership of the file pointer to Python
        nb::object file_obj = nb::cast(result.first, nb::rv_policy::take_ownership);
        nb::object infos_obj = nb::cast(result.second);
        
        return nb::make_tuple(file_obj, infos_obj);
    });



    nb::class_<torchfits::WCS>(m, "WCS")
        .def(nb::init<const std::unordered_map<std::string, std::string>&>())
        .def("pixel_to_world", [](torchfits::WCS& self, nb::ndarray<> pixels) {
            // Convert nb::ndarray to torch::Tensor
            auto tensor = torch::from_blob(pixels.data(), {static_cast<long long>(pixels.shape(0)), static_cast<long long>(pixels.shape(1))}, 
                                         torch::TensorOptions().dtype(torch::kFloat64));
            
            // Check dimensions
            if (pixels.ndim() != 2) throw std::runtime_error("Input must be 2D array (N x naxis)");
            // if (pixels.shape(1) != 2) throw std::runtime_error("Last dimension must be 2"); // Removed check
            
            // Create tensor from ndarray (zero copy if possible)
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            
            torch::Tensor input_tensor;
            if (pixels.dtype() == nb::dtype<double>()) {
                input_tensor = torch::from_blob(pixels.data(), {static_cast<long long>(pixels.shape(0)), static_cast<long long>(pixels.shape(1))}, options);
            } else {
                // Copy and cast
                input_tensor = torch::from_blob(pixels.data(), {static_cast<long long>(pixels.shape(0)), static_cast<long long>(pixels.shape(1))}, options).clone(); 
            }
            
            torch::Tensor output = self.pixel_to_world(input_tensor);
            return tensor_to_python(output);
        })
        .def("world_to_pixel", [](torchfits::WCS& self, nb::ndarray<> coords) {
             auto options = torch::TensorOptions().dtype(torch::kFloat64);
             torch::Tensor input_tensor = torch::from_blob(coords.data(), {static_cast<long long>(coords.shape(0)), static_cast<long long>(coords.shape(1))}, options);
             torch::Tensor output = self.world_to_pixel(input_tensor);
             return tensor_to_python(output);
        })
        .def("get_footprint", [](torchfits::WCS& self) {
            return tensor_to_python(self.get_footprint());
        })
        .def_prop_ro("naxis", &torchfits::WCS::naxis)
        .def_prop_ro("crpix", [](torchfits::WCS& self) { return tensor_to_python(self.crpix()); })
        .def_prop_ro("crval", [](torchfits::WCS& self) { return tensor_to_python(self.crval()); })
        .def_prop_ro("cdelt", [](torchfits::WCS& self) { return tensor_to_python(self.cdelt()); })
        .def_prop_ro("pc", [](torchfits::WCS& self) { return tensor_to_python(self.pc()); })
        .def_prop_ro("ctype", &torchfits::WCS::ctype)
        .def_prop_ro("cunit", &torchfits::WCS::cunit)
        .def_prop_ro("lonpole", &torchfits::WCS::lonpole)
        .def_prop_ro("latpole", &torchfits::WCS::latpole);

    m.def("open_fits_file", [](const std::string& path, const std::string& mode) {
        int mode_int = (mode == "w" || mode == "w+") ? 1 : 0;
        return new torchfits::FITSFile(path.c_str(), mode_int);
    }, nb::rv_policy::take_ownership);
    
    // close_fits_file removed, handled by destructor
    
    m.def("read_header", [](torchfits::FITSFile& file, int hdu_num) {
        return file.get_header(hdu_num);
    });
    m.def("read_header_string", [](torchfits::FITSFile& file, int hdu_num) {
        return file.read_header_to_string(hdu_num);
    });
    m.def("configure_cache", &torchfits::configure_cache, nb::arg("max_files"), nb::arg("max_memory_mb"));
    m.def("clear_file_cache", &torchfits::clear_file_cache);
    m.def("get_cache_size", &torchfits::get_cache_size);

    m.def("get_num_hdus", [](torchfits::FITSFile& file) {
        return file.get_num_hdus();
    });

    m.def("get_hdu_type", [](torchfits::FITSFile& file, int hdu_num) {
        return file.get_hdu_type(hdu_num);
    });
    
    m.def("read_image_from_handle", [](torchfits::FITSFile& file, int hdu_num) {
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

    m.def("read_hdus_batch", [](const std::string& path, const std::vector<int>& hdus) {
        nb::gil_scoped_release release;
        auto tensors = torchfits::read_hdus_batch(path, hdus);
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
