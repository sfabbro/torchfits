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

namespace {

inline void move_to_hdu_0based(fitsfile* fptr, int hdu_num) {
    int status = 0;
    // CFITSIO uses 1-based HDU numbers for fits_movabs_hdu.
    fits_movabs_hdu(fptr, hdu_num + 1, nullptr, &status);
    if (status != 0) {
        throw std::runtime_error("Could not move to HDU");
    }
}

nb::object tensor_to_numpy_object(const torch::Tensor& tensor) {
    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor for NumPy conversion");
    }

    PyObject* numpy_obj = PyObject_CallMethod(tensor_obj, "numpy", nullptr);
    Py_DECREF(tensor_obj);
    if (!numpy_obj) {
        throw nb::python_error();
    }
    return nb::steal(numpy_obj);
}

// Allocate a CPU numpy.ndarray (via nanobind) backed by a Python-owned bytearray.
// This avoids needing NumPy headers while still returning a real numpy.ndarray when
// NumPy is installed (as it is in our benchmark env).
template <typename T>
nb::ndarray<nb::numpy, T, nb::c_contig> alloc_numpy_array(
    const std::vector<size_t>& shape
) {
    size_t nelem = 1;
    for (size_t d : shape) {
        nelem *= d;
    }
    const size_t nbytes = nelem * sizeof(T);

    PyObject* ba = PyByteArray_FromStringAndSize(nullptr, (Py_ssize_t) nbytes);
    if (!ba) {
        throw std::runtime_error("Failed to allocate bytearray for numpy result");
    }
    nb::object owner = nb::steal(ba);
    void* data = (void*) PyByteArray_AsString(owner.ptr());
    if (!data) {
        throw std::runtime_error("Failed to get bytearray buffer for numpy result");
    }
    return nb::ndarray<nb::numpy, T, nb::c_contig>(
        data, shape.size(), shape.data(), owner
    );
}

nb::dict table_result_to_python(
    const std::unordered_map<std::string, torchfits::TableReader::ColumnData>& result_map,
    bool as_numpy
) {
    nb::dict result_dict;
    for (auto& [key, col_data] : result_map) {
        if (col_data.is_vla) {
            if (as_numpy && col_data.vla_offsets.defined()) {
                result_dict[key.c_str()] = nb::make_tuple(
                    tensor_to_numpy_object(col_data.fixed_data),
                    tensor_to_numpy_object(col_data.vla_offsets)
                );
                continue;
            }
            nb::list vla_list;
            for (const auto& tensor : col_data.vla_data) {
                vla_list.append(as_numpy ? tensor_to_numpy_object(tensor) : tensor_to_python(tensor));
            }
            result_dict[key.c_str()] = vla_list;
        } else {
            result_dict[key.c_str()] = as_numpy ? tensor_to_numpy_object(col_data.fixed_data)
                                                : tensor_to_python(col_data.fixed_data);
        }
    }
    return result_dict;
}

}  // namespace

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
            return table_result_to_python(result_map, false);
        }, nb::arg("column_names") = std::vector<std::string>(),
           nb::arg("start_row") = 1, nb::arg("num_rows") = -1)
        .def("read_rows_numpy", [](torchfits::TableReader& self,
                                  const std::vector<std::string>& column_names,
                                  long start_row, long num_rows) -> nb::object {
            nb::gil_scoped_release release;
            auto result_map = self.read_columns(column_names, start_row, num_rows, true);
            nb::gil_scoped_acquire acquire;
            return table_result_to_python(result_map, true);
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

    m.def("read_full_cached", [](const std::string& filename, int hdu_num, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_full_cached(filename, hdu_num, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    m.def("read_full_numpy_cached", [](const std::string& filename, int hdu_num, bool use_mmap) -> nb::object {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_full_cached(filename, hdu_num, use_mmap);
        }
        return tensor_to_numpy_object(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num"), nb::arg("use_mmap") = true);

    // Numpy-returning fast path (for benchmarking vs fitsio/astropy without paying
    // torch.Tensor -> numpy conversion overhead).
    m.def("read_full_numpy", [](const std::string& filename, int hdu_num, bool use_mmap) -> nb::object {
        torchfits::FITSFile file(filename.c_str(), 0);
        fitsfile* fptr = file.get_fptr();

        int status = 0;
        file.ensure_hdu(hdu_num, &status);
        if (status != 0) {
            throw std::runtime_error("Could not move to HDU");
        }

        const int bitpix = file.get_dtype(hdu_num);
        const auto scale_info = file.get_scale_info_for_hdu(hdu_num);
        const bool scaled = scale_info.scaled;

        status = 0;
        const int is_comp = fits_is_compressed_image(fptr, &status);
        const bool compressed = (status == 0) && (is_comp != 0);
        if (status != 0) {
            status = 0;
        }

        // Shape in C-contiguous (numpy/torch) order.
        std::vector<long> shape_long = file.get_shape(hdu_num);
        std::vector<size_t> shape;
        shape.reserve(shape_long.size());
        for (long d : shape_long) {
            shape.push_back((size_t) d);
        }

        if (shape.empty()) {
            return alloc_numpy_array<uint8_t>({0}).cast();
        }

        int datatype = 0;
        nb::object out;
        void* dst = nullptr;

        if (scaled) {
            // Signed BYTE_IMG encoded with BZERO=-128 is a common case; match torchfits.read behavior.
            if (bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0) {
                auto arr = alloc_numpy_array<int8_t>(shape);
                dst = (void*) arr.data();
                datatype = TSBYTE;
                out = arr.cast();
            } else {
                auto arr = alloc_numpy_array<float>(shape);
                dst = (void*) arr.data();
                datatype = TFLOAT;
                out = arr.cast();
            }
        } else {
            switch (bitpix) {
                case BYTE_IMG: {
                    auto arr = alloc_numpy_array<uint8_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TBYTE;
                    out = arr.cast();
                    break;
                }
                case SHORT_IMG: {
                    auto arr = alloc_numpy_array<int16_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TSHORT;
                    out = arr.cast();
                    break;
                }
                case LONG_IMG: {
                    auto arr = alloc_numpy_array<int32_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TINT;
                    out = arr.cast();
                    break;
                }
                case LONGLONG_IMG: {
                    auto arr = alloc_numpy_array<int64_t>(shape);
                    dst = (void*) arr.data();
                    datatype = TLONGLONG;
                    out = arr.cast();
                    break;
                }
                case FLOAT_IMG: {
                    auto arr = alloc_numpy_array<float>(shape);
                    dst = (void*) arr.data();
                    datatype = TFLOAT;
                    out = arr.cast();
                    break;
                }
                case DOUBLE_IMG: {
                    auto arr = alloc_numpy_array<double>(shape);
                    dst = (void*) arr.data();
                    datatype = TDOUBLE;
                    out = arr.cast();
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported BITPIX for numpy read");
            }
        }

        // Fast uncompressed raw BYTE_IMG path: mmap/pread and memcpy into numpy buffer.
        // Also handle common signed-byte scaling (BITPIX=8 with BZERO=-128) by xor'ing
        // the sign bit in-place after copying.
        const bool signed_byte_scaled =
            scaled && bitpix == BYTE_IMG && scale_info.bscale == 1.0 && scale_info.bzero == -128.0;
        if (use_mmap && !compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled)) {
            if (filename.find('[') == std::string::npos) {
                LONGLONG headstart = 0, data_offset = 0, dataend = 0;
                status = 0;
                fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
                if (status == 0 && data_offset > 0) {
                    size_t nelem = 1;
                    for (size_t d : shape) nelem *= d;
                    const size_t nbytes = nelem;  // 1 byte/elem
                    const int fd = open_readonly_fd(filename);
                    if (fd != -1) {
                        struct stat sb {};
                        if (fstat(fd, &sb) == 0 &&
                            (size_t) sb.st_size >= (size_t) data_offset + nbytes) {
                            uint8_t* dst8 = static_cast<uint8_t*>(dst);
                            size_t remaining = nbytes;
                            off_t off = (off_t) data_offset;
                            bool ok = true;
                            while (remaining) {
                                ssize_t got = pread(fd, dst8, remaining, off);
                                if (got < 0) {
                                    if (errno == EINTR) {
                                        continue;
                                    }
                                    ok = false;
                                    break;
                                }
                                if (got == 0) {
                                    ok = false;
                                    break;
                                }
                                dst8 += (size_t) got;
                                off += (off_t) got;
                                remaining -= (size_t) got;
                            }
                            if (ok) {
                                ::close(fd);
                                if (signed_byte_scaled) {
                                    _xor_sign_bit_u8(static_cast<uint8_t*>(dst), nbytes);
                                }
                                return out;
                            }

                            void* map_ptr = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
                            if (map_ptr != MAP_FAILED) {
                                const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + data_offset;
                                std::memcpy(dst, src, nbytes);
                                munmap(map_ptr, sb.st_size);
                                ::close(fd);
                                if (signed_byte_scaled) {
                                    _xor_sign_bit_u8(static_cast<uint8_t*>(dst), nbytes);
                                }
                                return out;
                            }
                        }
                        ::close(fd);
                    }
                } else {
                    status = 0;
                }
            }
        }

        int anynul = 0;
        float fnullval = NAN;
        double dnullval = NAN;
        void* nullval_ptr = nullptr;
        if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
            if (has_compressed_nulls(fptr)) {
                nullval_ptr = (datatype == TFLOAT) ? (void*) &fnullval : (void*) &dnullval;
            }
        }

        {
            nb::gil_scoped_release release;
            status = 0;
            LONGLONG nelements = 1;
            for (size_t d : shape) {
                nelements *= (LONGLONG) d;
            }
            fits_read_img(fptr, datatype, 1, nelements, nullval_ptr, dst, &anynul, &status);
        }
        if (status != 0) {
            char err_text[31];
            fits_get_errstatus(status, err_text);
            throw std::runtime_error("Error reading image data (numpy): status=" + std::to_string(status) +
                                     " msg=" + std::string(err_text));
        }
        return out;
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
        // If this path was previously read via the unified cache, invalidate it so
        // subsequent opens see the new contents (mtime/size can be unchanged).
        torchfits::invalidate_cached(path);
        torchfits::invalidate_shared_meta(path);
        torchfits::FITSFile file(final_path.c_str(), 1);
        file.write_hdus(hdus, overwrite);
    });

    m.def("write_fits_file_compressed_images",
          [](const std::string& path, nb::list hdus, bool overwrite, const std::string& algorithm) {
              std::string final_path = path;
              if (overwrite) {
                  final_path = "!" + path;
              }

              int comptype = RICE_1;
              if (!algorithm.empty()) {
                  std::string a = algorithm;
                  std::transform(a.begin(), a.end(), a.begin(),
                                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                  if (a == "R" || a == "RICE" || a == "RICE_1") {
                      comptype = RICE_1;
                  } else if (a == "G" || a == "GZIP" || a == "GZIP_1") {
                      comptype = GZIP_1;
                  } else if (a == "GZIP_2") {
                      comptype = GZIP_2;
                  } else if (a == "H" || a == "HCOMPRESS" || a == "HCOMPRESS_1") {
                      comptype = HCOMPRESS_1;
                  } else if (a == "P" || a == "PLIO" || a == "PLIO_1") {
                      comptype = PLIO_1;
                  } else if (a == "NONE") {
                      comptype = 0;
                  } else {
                      throw std::runtime_error("Unsupported compression algorithm: " + algorithm);
                  }
              }

              torchfits::invalidate_cached(path);
              torchfits::invalidate_shared_meta(path);
              torchfits::FITSFile file(final_path.c_str(), 1);
              file.write_hdus_compressed_images(hdus, comptype);
          },
          nb::arg("path"), nb::arg("hdus"), nb::arg("overwrite"),
          nb::arg("algorithm") = std::string("RICE_1"));

    m.def("write_hdu_checksums", [](const std::string& path, int hdu_num) {
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, path.c_str(), 1 /* READWRITE */, &status);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for checksum writing");
        }
        move_to_hdu_0based(fptr, hdu_num);
        ffpcks(fptr, &status);  // compute + write DATASUM/CHECKSUM
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        if (status != 0 || close_status != 0) {
            throw std::runtime_error("Failed to write FITS checksums");
        }
    }, nb::arg("path"), nb::arg("hdu_num") = 0);

    m.def("verify_hdu_checksums", [](const std::string& path, int hdu_num) {
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, path.c_str(), 0 /* READONLY */, &status);
        if (status != 0 || !fptr) {
            throw std::runtime_error("Could not open FITS file for checksum verification");
        }
        int datastatus = -1;
        int hdustatus = -1;
        move_to_hdu_0based(fptr, hdu_num);
        ffvcks(fptr, &datastatus, &hdustatus, &status);
        int close_status = 0;
        fits_close_file(fptr, &close_status);
        if (status != 0 || close_status != 0) {
            throw std::runtime_error("Failed to verify FITS checksums");
        }
        return nb::make_tuple(datastatus, hdustatus);
    }, nb::arg("path"), nb::arg("hdu_num") = 0);

    m.def("write_fits_table", [](const std::string& filename, nb::dict tensor_dict, nb::dict header, bool overwrite,
                                 nb::object schema, const std::string& table_type) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite, schema, table_type);
    }, nb::arg("filename"), nb::arg("tensor_dict"), nb::arg("header"), nb::arg("overwrite"),
       nb::arg("schema") = nb::none(), nb::arg("table_type") = "binary");
    m.def("append_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        append_rows(filename.c_str(), hdu_num, tensor_dict);
    });
    m.def("insert_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        insert_rows(filename.c_str(), hdu_num, tensor_dict, start_row);
    });
    m.def("update_fits_table_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                       long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        update_rows(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });
    m.def("update_fits_table_rows_mmap", [](const std::string& filename, int hdu_num, nb::dict tensor_dict,
                                           long start_row, long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        update_rows_mmap(filename.c_str(), hdu_num, tensor_dict, start_row, num_rows);
    });
    m.def("rename_fits_table_columns", [](const std::string& filename, int hdu_num, nb::dict mapping) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        rename_columns(filename.c_str(), hdu_num, mapping);
    });
    m.def("drop_fits_table_columns", [](const std::string& filename, int hdu_num, nb::list columns) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        drop_columns(filename.c_str(), hdu_num, columns);
    });
    m.def("delete_fits_table_rows", [](const std::string& filename, int hdu_num, long start_row,
                                       long num_rows) {
        torchfits::invalidate_cached(filename);
        torchfits::invalidate_shared_meta(filename);
        delete_rows(filename.c_str(), hdu_num, start_row, num_rows);
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
            return table_result_to_python(result_map, false);
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
        return table_result_to_python(result_map, false);
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
            return nb::object(table_result_to_python(result_map, false));
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
            return nb::object(table_result_to_python(result_map, false));
        }
    }, nb::arg("filename"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1, nb::arg("mmap") = false);

    m.def("read_fits_table_rows_numpy_from_handle", [](torchfits::FITSFile& file, int hdu_num,
                                                       const std::vector<std::string>& column_names,
                                                       long start_row, long num_rows) -> nb::object {
        nb::gil_scoped_release release;
        torchfits::TableReader reader(file.get_fptr(), hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, true);
    }, nb::arg("file"), nb::arg("hdu_num") = 1,
       nb::arg("column_names") = std::vector<std::string>(),
       nb::arg("start_row") = 1, nb::arg("num_rows") = -1);

    m.def("read_fits_table_rows_numpy", [](const std::string& filename, int hdu_num,
                                           const std::vector<std::string>& column_names,
                                           long start_row, long num_rows, bool mmap) -> nb::object {
        if (mmap) {
            try {
                torchfits::TableReader reader(filename, hdu_num);
                nb::dict mapped = reader.read_columns_mmap(column_names, start_row, num_rows);
                nb::dict numpy_result;
                for (auto item : mapped) {
                    nb::handle key = item.first;
                    nb::handle value = item.second;
                    if (PyObject_HasAttrString(value.ptr(), "numpy")) {
                        PyObject* np_obj = PyObject_CallMethod(value.ptr(), "numpy", nullptr);
                        if (!np_obj) {
                            throw nb::python_error();
                        }
                        numpy_result[key] = nb::steal(np_obj);
                    } else {
                        numpy_result[key] = nb::borrow(value);
                    }
                }
                return numpy_result;
            } catch (...) {
                // Fallback to non-mmap path for unsupported table layouts (e.g. VLA/bit columns).
            }
        }
        nb::gil_scoped_release release;
        torchfits::FITSFile file(filename.c_str(), 0);
        torchfits::TableReader reader(file.get_fptr(), hdu_num);
        auto result_map = reader.read_columns(column_names, start_row, num_rows, true);
        nb::gil_scoped_acquire acquire;
        return table_result_to_python(result_map, true);
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
    m.def("clear_shared_read_meta_cache", &torchfits::clear_shared_read_meta_cache);
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

    m.def("read_hdus_batch", [](const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
        nb::gil_scoped_release release;
        auto tensors = torchfits::read_hdus_batch(path, hdus, use_mmap);
        nb::gil_scoped_acquire acquire;

        nb::list result;
        for (const auto& t : tensors) {
            result.append(tensor_to_python(t));
        }
        return result;
    }, nb::arg("path"), nb::arg("hdus"), nb::arg("use_mmap") = true);

    m.def("read_hdus_sequence_last", [](const std::string& path, const std::vector<int>& hdus, bool use_mmap) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = torchfits::read_hdus_sequence_last(path, hdus, use_mmap);
        }
        return tensor_to_python(tensor);
    }, nb::arg("path"), nb::arg("hdus"), nb::arg("use_mmap") = true);

    m.def("echo_tensor", [](nb::object obj) {
        return obj;
    });
}
