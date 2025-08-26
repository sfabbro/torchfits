#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>

#include <Python.h>
#include <dlpack/dlpack.h>
#include <memory>
#include <cstring>

#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <vector>
#include <unordered_map>

namespace nb = nanobind;

#include "fits.cpp"
#include "wcs.cpp"
#include "table.cpp"

// Provide a nanobind type_caster for at::Tensor using DLPack + PyCapsule
// exchange. This avoids depending on internal THPVariable symbols and
// keeps conversions zero-copy when the Python runtime supports DLPack.
//
// Approach summary:
// - To load (Python -> C++): accept a Python object; create a DLPack
//   capsule from it by calling `torch.utils.dlpack.to_dlpack` or
//   using the object's __dlpack__ if present. Then convert the
//   capsule to an at::Tensor using torch._C._from_dlpack (via the C API
//   surface exposed on the Python side). We keep ownership semantics
//   such that the capsule's deleter runs when appropriate.
// - To cast (C++ -> Python): obtain a DLManagedTensor capsule from the
//   at::Tensor using torch._C._to_dlpack (exposed on torch._C) and
//   return the result of `torch.utils.dlpack.from_dlpack(capsule)` so
//   Python receives a proper Tensor object and ownership is transferred.

namespace nanobind {
namespace detail {

    template <> struct type_caster<at::Tensor> {
    public:
        NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"));

        // Try to load a Python object as an at::Tensor using DLPack (zero-copy).
        NB_INLINE bool from_python(handle src, uint8_t flags, cleanup_list * /*cl*/) noexcept {
            if (!src) return false;

            try {
                nb::module_ dlpack_mod = nb::module_::import_("torch.utils.dlpack");

                // If the object implements __dlpack__, call it; otherwise ask
                // torch.utils.dlpack.to_dlpack to produce a capsule.
                nb::object py_capsule;
                if (nb::hasattr(src, "__dlpack__"))
                    py_capsule = src.attr("__dlpack__")();
                else
                    py_capsule = dlpack_mod.attr("to_dlpack")(src);

                // Extract the DLManagedTensor pointer from the capsule
                PyObject *caps = py_capsule.ptr();
                if (!caps) return false;

                void *ptr = PyCapsule_GetPointer(caps, "dltensor");
                if (!ptr) return false;

                DLManagedTensor *mt = reinterpret_cast<DLManagedTensor*>(ptr);

                // Construct an at::Tensor by consuming the DLManagedTensor.
                // at::fromDLPack consumes the DLManagedTensor and takes ownership
                // (invoking its deleter when appropriate).
                at::Tensor t = at::fromDLPack(mt);
                value = t;
                return true;

            } catch (...) {
                return false;
            }
        }

        // Cast an at::Tensor to a Python torch.Tensor using DLPack.
        static void dl_managed_tensor_deleter(DLManagedTensor* mt) {
            if (!mt) return;
            // manager_ctx holds a pointer to a heap-allocated shared_ptr<at::Tensor>
            auto holder = reinterpret_cast<std::shared_ptr<at::Tensor>*>(mt->manager_ctx);
            if (holder) delete holder;
            // free shape/strides arrays if present
            if (mt->dl_tensor.shape) delete [] mt->dl_tensor.shape;
            if (mt->dl_tensor.strides) delete [] mt->dl_tensor.strides;
            delete mt;
        }

        // Helper that constructs a Python tensor via DLPack from a C++ at::Tensor
        static handle tensor_to_python(const at::Tensor &src) {
            try {
                nb::module_ dlpack_mod = nb::module_::import_("torch.utils.dlpack");

                // Use ATen's toDLPack to create a DLManagedTensor with correct
                // ownership semantics. Then create a PyCapsule whose destructor
                // calls the DLManagedTensor deleter exactly once.
                DLManagedTensor *mt = at::toDLPack(src);
                if (!mt) {
                    return handle();
                }

                // Create a capsule destructor that calls the DLManagedTensor deleter
                auto capsule_destructor = [](PyObject *caps) {
                    DLManagedTensor *m = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(caps, "dltensor"));
                    if (m && m->deleter) {
                        m->deleter(m);
                    }
                };

                PyObject *caps = PyCapsule_New((void*)mt, "dltensor", /*destructor=*/(PyCapsule_Destructor) +[](PyObject *caps){
                    capsule_destructor(caps);
                });
                if (!caps) {
                    // If capsule creation failed, make sure to call deleter to avoid leak
                    if (mt->deleter) mt->deleter(mt);
                    return handle();
                }

                // Wrap the raw PyObject* into a nanobind object without altering refcount
                nb::object capsule_obj = nb::steal<nb::object>(nb::handle(caps));

                // Call torch.utils.dlpack.from_dlpack(capsule) to obtain a Python tensor
                nb::object py_tensor = dlpack_mod.attr("from_dlpack")(capsule_obj);
                return py_tensor;
            } catch (...) {
                return handle();
            }
        }

        // Overloads to satisfy nanobind's call patterns (pointer and const-ref/value)
        static handle from_cpp(at::Tensor *p, rv_policy /* policy */, cleanup_list * /* list */) {
            if (!p) return handle();
            return tensor_to_python(*p);
        }

        static handle from_cpp(const at::Tensor &src, rv_policy /* policy */, cleanup_list * /* list */) {
            return tensor_to_python(src);
        }
    };

} // namespace detail
} // namespace nanobind

NB_MODULE(cpp, m) {
    m.doc() = "torchfits C++ extension";
    
    // FITS file class
    nb::class_<torchfits::FITSFile>(m, "FITSFile")
        .def(nb::init<const std::string&, int>(), nb::arg("filename"), nb::arg("mode") = 0)
        .def("read_image", &torchfits::FITSFile::read_image, 
             nb::arg("hdu_num") = 0, nb::arg("use_mmap") = false)
        .def("write_image", &torchfits::FITSFile::write_image, 
             nb::arg("tensor"), nb::arg("hdu_num") = 0, 
             nb::arg("bscale") = 1.0, nb::arg("bzero") = 0.0)
        .def("get_header", &torchfits::FITSFile::get_header, nb::arg("hdu_num") = 0)
        .def("get_shape", &torchfits::FITSFile::get_shape, nb::arg("hdu_num") = 0)
        .def("get_dtype", &torchfits::FITSFile::get_dtype, nb::arg("hdu_num") = 0)
        .def("read_subset", &torchfits::FITSFile::read_subset)
        .def("compute_stats", &torchfits::FITSFile::compute_stats, nb::arg("hdu_num") = 0)
        .def("get_num_hdus", &torchfits::FITSFile::get_num_hdus)
        .def("get_hdu_type", &torchfits::FITSFile::get_hdu_type)
        .def("write_hdus", &torchfits::FITSFile::write_hdus);
    
    // WCS class
    nb::class_<torchfits::WCS>(m, "WCS")
        .def(nb::init<const std::unordered_map<std::string, std::string>&>())
        .def("pixel_to_world", &torchfits::WCS::pixel_to_world)
        .def("world_to_pixel", &torchfits::WCS::world_to_pixel)
        .def("get_footprint", &torchfits::WCS::get_footprint)
        .def_prop_ro("naxis", [](const torchfits::WCS &wcs) { return wcs.naxis(); })
        .def_prop_ro("crpix", [](const torchfits::WCS &wcs) { return wcs.crpix(); })
        .def_prop_ro("crval", [](const torchfits::WCS &wcs) { return wcs.crval(); })
                .def_prop_ro("cdelt", [](const torchfits::WCS &wcs) { return wcs.cdelt(); });

    

    
    
    // HDU operations
    m.def("open_fits_file", [](const std::string& path, const std::string& mode) {
        auto* file = new torchfits::FITSFile(path, mode == "r" ? 0 : 1);
        return reinterpret_cast<uintptr_t>(file);
    });
    
    m.def("close_fits_file", [](uintptr_t handle) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        delete file;
    });
    
    m.def("get_num_hdus", [](uintptr_t handle) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->get_num_hdus();
    });
    
    m.def("get_hdu_type", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->get_hdu_type(hdu_num);
    });
    
    m.def("read_header", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->get_header(hdu_num);
    });
    
    m.def("get_shape", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->get_shape(hdu_num);
    });
    
    m.def("get_dtype", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->get_dtype(hdu_num);
    });
    
    m.def("read_subset", [](uintptr_t handle, int hdu_num, long x1, long y1, long x2, long y2) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->read_subset(hdu_num, x1, y1, x2, y2);
    });
    
    m.def("read_full", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->read_image(hdu_num);
    });

    // Simple passthrough for testing DLPack caster: returns the same tensor
    m.def("echo_tensor", [](const at::Tensor &t) {
        return t;
    });
    
    m.def("compute_stats", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->compute_stats(hdu_num);
    });
    
    m.def("write_fits_file", [](const std::string& path, nb::list hdus, bool overwrite) {
        torchfits::FITSFile file(path, 1);
        file.write_hdus(hdus, overwrite);
    });
    
    // Table operations - create proper TensorFrame
    m.def("read_fits_table", [](const std::string& filename, int hdu_num) {
        void* reader = open_table_reader(filename.c_str(), hdu_num);
        if (!reader) {
            throw std::runtime_error("Failed to open table reader");
        }

        nb::dict result_dict;
        int status = read_table_columns(reader, nullptr, 0, 1, -1, &result_dict);
        close_table_reader(reader);

        if (status != 0) {
            throw std::runtime_error("Failed to read table columns");
        }

        return result_dict;
    });

    m.def("read_fits_table_from_handle", [](uintptr_t handle, int hdu_num) {
        void* reader = open_table_reader_from_handle(handle, hdu_num);
        if (!reader) {
            throw std::runtime_error("Failed to open table reader");
        }

        nb::dict result_dict;
        int status = read_table_columns(reader, nullptr, 0, 1, -1, &result_dict);
        close_table_reader(reader);

        if (status != 0) {
            throw std::runtime_error("Failed to read table columns");
        }

        return result_dict;
    });
    
    m.def("read_header_dict", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.get_header(hdu_num);
    });
    
    m.def("write_fits_table", [](const std::string& filename, nb::dict tensor_dict, nb::dict header, bool overwrite) {
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite);
    });

    m.def("append_rows", [](const std::string& filename, int hdu_num, nb::dict tensor_dict) {
        append_rows(filename.c_str(), hdu_num, tensor_dict);
    });
    
    // Simple cache management
    m.def("clear_file_cache", []() {
        torchfits::global_cache.clear();
    });
    
    m.def("get_cache_size", []() {
        return torchfits::global_cache.size();
    });
    
    // Performance hint: suggest optimal chunk size
    m.def("get_optimal_chunk_size", [](size_t data_size) {
        // Simple heuristic: use 64KB chunks for small data, 1MB for large
        if (data_size < 1024 * 1024) {
            return std::min(data_size, size_t(64 * 1024));
        } else {
            return std::min(data_size, size_t(1024 * 1024));
        }
    });
    
    // Direct read function for minimal overhead
    m.def("read_full", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.read_image(hdu_num);
    });
    
    // Echo tensor for DLPack round-trip testing (returns the same tensor)
    m.def("echo_tensor", [](const at::Tensor &t) {
        return t;
    });
    
    // Memory-mapped read function
    m.def("read_mmap", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.read_image(hdu_num, true);
    });
    
    // Mixed precision conversion
    // Accept a Python tensor object and perform the .to(dtype) call in Python
    // to avoid relying on a nanobind caster for torch::Tensor.
    m.def("convert_to_fp16", [](nb::object tensor_py) {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object dtype = torch.attr("float16");
        return tensor_py.attr("to")(dtype);
    });
    
    m.def("convert_to_bf16", [](nb::object tensor_py) {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object dtype = torch.attr("bfloat16");
        return tensor_py.attr("to")(dtype);
    });
    
    // CFITSIO native string parsing (e.g., "file.fits[0:200,400:600]")
    m.def("read_cfitsio_string", [](const std::string& cfitsio_string) {
        torchfits::FITSFile file(cfitsio_string, 0);
        return file.read_image(0);  // CFITSIO handles HDU and section parsing
    });
    
    // Hardware info functions
    m.def("get_hardware_info", []() {
        std::lock_guard<std::mutex> lock(torchfits::hw_mutex);
        if (!torchfits::hw_detected) {
            torchfits::hw_info = torchfits::detect_hardware();
            torchfits::hw_detected = true;
        }
        nb::dict info;
        info["l3_cache_size"] = torchfits::hw_info.l3_cache_size;
        info["memory_bandwidth"] = torchfits::hw_info.memory_bandwidth;
        info["available_memory"] = torchfits::hw_info.available_memory;
        info["is_nvme"] = torchfits::hw_info.is_nvme;
        return info;
    });
    
    // CFITSIO iterator function for optimal table processing
    m.def("iterate_table", [](const std::string& filename, int hdu_num, std::function<void(nb::dict)> work_func) {
        // Placeholder for fits_iterate_data implementation
        // This would be the "golden path" for table processing
        throw std::runtime_error("Table iteration not yet implemented");
    });
    
    // Get optimal row chunk size for table I/O
    m.def("get_optimal_rows", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        int status = 0;
        fits_movabs_hdu(file.get_fptr(), hdu_num + 1, nullptr, &status);
        
        long optimal_rows;
        fits_get_rowsize(file.get_fptr(), &optimal_rows, &status);
        
        if (status != 0) {
            throw std::runtime_error("Failed to get optimal row size");
        }
        
        return optimal_rows;
    });
}