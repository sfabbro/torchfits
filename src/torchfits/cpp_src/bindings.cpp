#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>

#include <Python.h>
#include <memory>
#include <cstring>

#include <torch/torch.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/DLConvertor.h>
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

// DISABLED: DLPack type_caster causes 7x overhead for int16!
// We now use explicit tensor_to_python() with THPVariable_Wrap instead
/*
namespace nanobind {
namespace detail {

template <> struct type_caster<torch::Tensor> {
    NB_TYPE_CASTER(torch::Tensor, const_name("torch.Tensor"));

    // Python -> C++ conversion
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        try {
            // Convert Python tensor to DLPack capsule
            object capsule_obj;

            // Try __dlpack__ protocol first (modern approach)
            if (PyObject_HasAttrString(src.ptr(), "__dlpack__")) {
                object dlpack_method = getattr(src, "__dlpack__");
                capsule_obj = dlpack_method();
            } else {
                // Fallback to torch.utils.dlpack.to_dlpack
                object torch_module = module_::import_("torch.utils.dlpack");
                object to_dlpack = torch_module.attr("to_dlpack");
                capsule_obj = to_dlpack(src);
            }

            // Extract DLManagedTensor from PyCapsule
            DLManagedTensor* dlmt = (DLManagedTensor*)PyCapsule_GetPointer(capsule_obj.ptr(), "dltensor");
            if (!dlmt) return false;

            // Convert DLPack to torch::Tensor
            value = torch::fromDLPack(dlmt);
            return true;
        } catch (...) {
            return false;
        }
    }

    // C++ -> Python conversion
    static handle from_cpp(const torch::Tensor &tensor, rv_policy policy, cleanup_list *cleanup) noexcept {
        try {
            // Convert at::Tensor to DLManagedTensor
            DLManagedTensor* dlmt = torch::toDLPack(tensor);
            if (!dlmt) return handle();

            // Wrap in PyCapsule
            PyObject *capsule = PyCapsule_New(dlmt, "dltensor", [](PyObject *obj) {
                DLManagedTensor* dlmt = (DLManagedTensor*)PyCapsule_GetPointer(obj, "dltensor");
                if (dlmt && dlmt->deleter) {
                    dlmt->deleter(dlmt);
                }
            });
            if (!capsule) return handle();

            // Import torch.utils.dlpack and call from_dlpack
            object torch_module = module_::import_("torch.utils.dlpack");
            object from_dlpack = torch_module.attr("from_dlpack");
            object result = from_dlpack(handle(capsule));

            return result.release();
        } catch (...) {
            return handle();
        }
    }
};

} // namespace detail
} // namespace nanobind
*/

// Helper function to convert torch::Tensor to Python object - FAST PATH
// EXPERIMENTAL: Return NumPy array for int16 to test if THPVariable_Wrap is the bottleneck
static nb::object tensor_to_python(const torch::Tensor& tensor) {
    // EXPERIMENT: For int16, return NumPy array instead of torch.Tensor
    // This tests whether THPVariable_Wrap is causing the 0.47ms overhead
    if (tensor.scalar_type() == torch::kInt16) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Create NumPy array from tensor data (zero-copy via capsule)
        auto shape = tensor.sizes();
        auto strides = tensor.strides();

        // Convert to bytes for nanobind
        std::vector<size_t> shape_vec(shape.begin(), shape.end());
        std::vector<int64_t> strides_vec;
        for (auto s : strides) {
            strides_vec.push_back(s * sizeof(int16_t));
        }

        // Create numpy array using nanobind
        // The tensor data is managed by the tensor itself, so we need to keep it alive
        auto* data_ptr = tensor.data_ptr<int16_t>();

        // Create a capsule to manage tensor lifetime
        auto tensor_copy = new torch::Tensor(tensor);  // Keep tensor alive
        auto capsule = nb::capsule(tensor_copy, [](void* p) noexcept {
            delete static_cast<torch::Tensor*>(p);
        });

        auto result = nb::ndarray<nb::numpy, int16_t>(
            data_ptr,
            shape_vec.size(),
            shape_vec.data(),
            capsule,
            strides_vec.data()
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        auto wrap_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        static int numpy_count = 0;
        if (numpy_count++ < 5) {
            fprintf(stderr, "[INT16] NumPy wrap: %.1fμs\n", wrap_us);
            fflush(stderr);
        }

        return nb::cast(result);
    }

    // For other types, use THPVariable_Wrap (works fine for uint8, etc.)
    auto t0 = std::chrono::high_resolution_clock::now();

    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wrap_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    static int tensor_count = 0;
    if (tensor.scalar_type() == torch::kUInt8 && tensor_count++ < 5) {
        fprintf(stderr, "[UINT8] Tensor wrap: %.1fμs\n", wrap_us);
        fflush(stderr);
    }

    return nb::steal(tensor_obj);
}

NB_MODULE(cpp, m) {
    m.doc() = "torchfits C++ extension";

    // Test function to verify compilation
    m.def("get_build_info", []() -> std::string {
        return "Build: NumPy int16 test - " __DATE__ " " __TIME__;
    });

    // Test tensor return for int16
    m.def("test_int16_return", []() -> nb::object {
        auto tensor = torch::empty({10, 10}, torch::TensorOptions().dtype(torch::kInt16));
        fprintf(stderr, "test_int16_return called!\n");
        fflush(stderr);
        return tensor_to_python(tensor);
    });

    // Simple test function to return a tensor
    m.def("test_tensor_return", []() -> nb::object {
        auto tensor = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat64));
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        return tensor_to_python(tensor);
    });

    // FITS file class with GIL release for I/O operations
    nb::class_<torchfits::FITSFile>(m, "FITSFile")
        .def(nb::init<const std::string&, int>(), nb::arg("filename"), nb::arg("mode") = 0)
        .def("read_image", [](torchfits::FITSFile& self, int hdu_num, bool use_mmap) {
            nb::gil_scoped_release release;
            auto tensor = self.read_image(hdu_num, use_mmap);
            return tensor_to_python(tensor);
        }, nb::arg("hdu_num") = 0, nb::arg("use_mmap") = false)
        .def("write_image", [](torchfits::FITSFile& self, const torch::Tensor& tensor, int hdu_num, double bscale, double bzero) {
            nb::gil_scoped_release release;
            return self.write_image(tensor, hdu_num, bscale, bzero);
        }, nb::arg("tensor"), nb::arg("hdu_num") = 0, nb::arg("bscale") = 1.0, nb::arg("bzero") = 0.0)
        .def("get_header", [](torchfits::FITSFile& self, int hdu_num) {
            nb::gil_scoped_release release;
            return self.get_header(hdu_num);
        }, nb::arg("hdu_num") = 0)
        .def("get_shape", &torchfits::FITSFile::get_shape, nb::arg("hdu_num") = 0)
        .def("get_dtype", &torchfits::FITSFile::get_dtype, nb::arg("hdu_num") = 0)
        .def("read_subset", [](torchfits::FITSFile& self, int hdu_num, long x1, long y1, long x2, long y2) {
            nb::gil_scoped_release release;
            auto tensor = self.read_subset(hdu_num, x1, y1, x2, y2);
            return tensor_to_python(tensor);
        })
        .def("compute_stats", [](torchfits::FITSFile& self, int hdu_num) {
            nb::gil_scoped_release release;
            return self.compute_stats(hdu_num);
        }, nb::arg("hdu_num") = 0)
        .def("get_num_hdus", &torchfits::FITSFile::get_num_hdus)
        .def("get_hdu_type", &torchfits::FITSFile::get_hdu_type)
        .def("write_hdus", [](torchfits::FITSFile& self, nb::list hdus, bool overwrite) {
            nb::gil_scoped_release release;
            return self.write_hdus(hdus, overwrite);
        })
        .def("get_fptr", &torchfits::FITSFile::get_fptr, nb::rv_policy::reference_internal);
    
    // WCS class
    nb::class_<torchfits::WCS>(m, "WCS")
        .def(nb::init<const std::unordered_map<std::string, std::string>&>())
        .def("pixel_to_world", &torchfits::WCS::pixel_to_world)
        .def("world_to_pixel", &torchfits::WCS::world_to_pixel)
        .def("get_footprint", &torchfits::WCS::get_footprint)
        .def("test_method", &torchfits::WCS::test_method)
        .def("naxis", &torchfits::WCS::naxis)
        .def("crpix", &torchfits::WCS::crpix)
        .def("crval", &torchfits::WCS::crval)
        .def("cdelt", &torchfits::WCS::cdelt);

    

    
    
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
    
    // Fast bulk header reading - OPTIMIZE.md Task #5
    m.def("read_header_string", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->read_header_to_string(hdu_num);
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
        auto tensor = file->read_subset(hdu_num, x1, y1, x2, y2);
        return tensor_to_python(tensor);
    });
    
    m.def("read_full", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);

        // PROFILING: Time the entire read_full operation
        auto t_start = std::chrono::high_resolution_clock::now();

        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            tensor = file->read_image(hdu_num);
        }

        auto t_before_wrap = std::chrono::high_resolution_clock::now();
        auto result = tensor_to_python(tensor);
        auto t_end = std::chrono::high_resolution_clock::now();

        // PROFILING: Write timing for first 10 calls
        static std::atomic<int> call_count{0};
        int count = call_count.fetch_add(1);
        if (count < 10 && tensor.scalar_type() == torch::kInt16) {
            auto total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            auto read_ms = std::chrono::duration<double, std::milli>(t_before_wrap - t_start).count();
            auto wrap_ms = std::chrono::duration<double, std::milli>(t_end - t_before_wrap).count();

            // Print timing to stderr
            fprintf(stderr, "[read_full INT16] total=%.3fms read=%.3fms wrap=%.3fms\n",
                   total_ms, read_ms, wrap_ms);
            fflush(stderr);
        }

        return result;
    });

    m.def("compute_stats", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->compute_stats(hdu_num);
    });
    
    m.def("write_fits_file", [](const std::string& path, nb::list hdus, bool overwrite) {
        torchfits::FITSFile file(path, 1);
        file.write_hdus(hdus, overwrite);
    });
    
    // Table operations with GIL release
    m.def("read_fits_table", [](const std::string& filename, int hdu_num) -> nb::object {
        try {
            nb::gil_scoped_release release;
            torchfits::TableReader reader(filename, hdu_num);
            
            // Read all columns if none specified
            std::vector<std::string> column_names = reader.get_column_names();
            auto result = reader.read_columns(column_names, 1, -1);
            
            return nb::cast(result);
        } catch (const std::exception& e) {
            // Return empty dict for failed reads
            return nb::cast(nb::dict());
        }
    });

    m.def("read_fits_table_from_handle", [](uintptr_t handle, int hdu_num) {
        try {
            nb::gil_scoped_release release;
            auto* fits_file = reinterpret_cast<torchfits::FITSFile*>(handle);
            torchfits::TableReader reader(fits_file->get_fptr(), hdu_num);
            
            // Read all columns if none specified  
            std::vector<std::string> column_names = reader.get_column_names();
            auto result = reader.read_columns(column_names, 1, -1);
            
            // Return dict with tensor_dict and col_stats for HDUList compatibility
            nb::dict full_result;
            full_result["tensor_dict"] = nb::cast(result);
            full_result["col_stats"] = nb::dict();  // Empty stats for now
            
            return full_result;
        } catch (const std::exception& e) {
            // Return empty result for failed reads
            nb::dict empty_result;
            empty_result["tensor_dict"] = nb::dict();
            empty_result["col_stats"] = nb::dict();
            return empty_result;
        }
    });
    
    m.def("read_header_dict", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.get_header(hdu_num);
    });

    // ULTRA-FAST: Read both data and header in SINGLE C++ call
    m.def("read_image_and_header", [](const std::string& filename, int hdu_num) -> nb::tuple {
        torch::Tensor tensor;
        std::unordered_map<std::string, std::string> header_map;
        {
            nb::gil_scoped_release release;
            torchfits::FITSFile file(filename, 0);
            tensor = file.read_image(hdu_num);
            header_map = file.get_header(hdu_num);
        }
        // Convert header map to Python dict (GIL reacquired here)
        nb::dict header;
        for (const auto& pair : header_map) {
            header[pair.first.c_str()] = pair.second.c_str();
        }
        return nb::make_tuple(tensor_to_python(tensor), header);
    }, nb::arg("filename"), nb::arg("hdu_num") = 0,
       "Read image data and header in single file open/close");

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
    
    // Configure cache function for Python cache module compatibility
    m.def("configure_cache", [](int max_files, int max_memory_mb) {
        // This is a placeholder for cache configuration
        // The actual cache configuration is handled by the global_cache object
        // which is configured at compile time or through other mechanisms
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
    
    // Direct read function for minimal overhead with GIL release
    m.def("read_full", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            torchfits::FITSFile file(filename, 0);
            tensor = file.read_image(hdu_num);
        }
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num") = 0);

    // Ultra-fast path: read image with minimal overhead - SINGLE C++ CALL
    m.def("read_image_fast", [](const std::string& filename, int hdu_num) -> nb::object {
        // Open, read, close in one function - minimize overhead
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            torchfits::FITSFile file(filename, 0);
            tensor = file.read_image(hdu_num);
        }
        // GIL reacquired here automatically
        return tensor_to_python(tensor);
    }, nb::arg("filename"), nb::arg("hdu_num") = 0,
       "Fast path: open->read->close in single C++ call");
    
    // Simple test function to return a tensor
    m.def("test_tensor_return", []() -> nb::object {
        auto tensor = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat64));
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        return tensor_to_python(tensor);
    });

    // Note: echo_tensor already defined above
    
    // Memory-mapped read function with GIL release
    m.def("read_mmap", [](const std::string& filename, int hdu_num) {
        torch::Tensor tensor;
        {
            nb::gil_scoped_release release;
            torchfits::FITSFile file(filename, 0);
            tensor = file.read_image(hdu_num, true);
        }
        return tensor_to_python(tensor);
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
        auto tensor = file.read_image(0);  // CFITSIO handles HDU and section parsing
        return tensor_to_python(tensor);
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
    
    // Simple passthrough for testing DLPack caster: returns the same tensor
    m.def("echo_tensor", [](const torch::Tensor &t) {
        return tensor_to_python(t);
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