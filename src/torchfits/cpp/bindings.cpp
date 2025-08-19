#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <fitsio.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

#include "fits.cpp"
#include "wcs.cpp"
#include "table.cpp"

PYBIND11_MODULE(cpp, m) {
    m.doc() = "torchfits C++ extension";
    
    // FITS file class
    py::class_<torchfits::FITSFile>(m, "FITSFile")
        .def(py::init<const std::string&, int>(), py::arg("filename"), py::arg("mode") = 0)
        .def("read_image", &torchfits::FITSFile::read_image, 
             py::arg("hdu_num") = 0, py::arg("use_mmap") = false)
        .def("write_image", &torchfits::FITSFile::write_image, 
             py::arg("tensor"), py::arg("hdu_num") = 0, 
             py::arg("bscale") = 1.0, py::arg("bzero") = 0.0)
        .def("get_header", &torchfits::FITSFile::get_header, py::arg("hdu_num") = 0)
        .def("get_shape", &torchfits::FITSFile::get_shape, py::arg("hdu_num") = 0)
        .def("get_dtype", &torchfits::FITSFile::get_dtype, py::arg("hdu_num") = 0)
        .def("read_subset", &torchfits::FITSFile::read_subset)
        .def("compute_stats", &torchfits::FITSFile::compute_stats, py::arg("hdu_num") = 0)
        .def("get_num_hdus", &torchfits::FITSFile::get_num_hdus)
        .def("get_hdu_type", &torchfits::FITSFile::get_hdu_type)
        .def("write_hdus", &torchfits::FITSFile::write_hdus);
    
    // WCS class
    py::class_<torchfits::WCS>(m, "WCS")
        .def(py::init<const std::unordered_map<std::string, std::string>&>())
        .def("pixel_to_world", &torchfits::WCS::pixel_to_world)
        .def("world_to_pixel", &torchfits::WCS::world_to_pixel)
        .def("get_footprint", &torchfits::WCS::get_footprint);
    
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
    
    m.def("compute_stats", [](uintptr_t handle, int hdu_num) {
        auto* file = reinterpret_cast<torchfits::FITSFile*>(handle);
        return file->compute_stats(hdu_num);
    });
    
    m.def("write_fits_file", [](const std::string& path, py::list hdus, bool overwrite) {
        torchfits::FITSFile file(path, 1);
        file.write_hdus(hdus, overwrite);
    });
    
    // Table operations - create proper TensorFrame
    m.def("read_fits_table", [](const std::string& filename, int hdu_num) {
        void* reader = open_table_reader(filename.c_str(), hdu_num);
        if (!reader) {
            throw std::runtime_error("Failed to open table reader");
        }

        py::dict result_dict;
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

        py::dict result_dict;
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
    
    m.def("write_fits_table", [](const std::string& filename, py::dict tensor_dict, py::dict header, bool overwrite) {
        write_fits_table(filename.c_str(), tensor_dict, header, overwrite);
    });

    m.def("append_rows", [](const std::string& filename, int hdu_num, py::dict tensor_dict) {
        append_rows(filename.c_str(), hdu_num, tensor_dict);
    });
    
    // Unified cache management
    m.def("clear_file_cache", []() {
        torchfits::global_cache.clear();
    });
    
    m.def("get_cache_size", []() {
        return torchfits::global_cache.size();
    });
    
    // Cloud/HPC cache configuration
    m.def("configure_cache", [](size_t max_files, size_t max_memory_mb) {
        torchfits::global_cache.clear();
        // Note: Would need to recreate cache with new parameters
    });
    
    // Direct read function for minimal overhead
    m.def("read_full", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.read_image(hdu_num);
    });
    
    // Memory-mapped read function
    m.def("read_mmap", [](const std::string& filename, int hdu_num) {
        torchfits::FITSFile file(filename, 0);
        return file.read_image(hdu_num, true);
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
        py::dict info;
        info["l3_cache_size"] = torchfits::hw_info.l3_cache_size;
        info["memory_bandwidth"] = torchfits::hw_info.memory_bandwidth;
        info["available_memory"] = torchfits::hw_info.available_memory;
        info["is_nvme"] = torchfits::hw_info.is_nvme;
        return info;
    });
    
    // CFITSIO iterator function for optimal table processing
    m.def("iterate_table", [](const std::string& filename, int hdu_num, py::function work_func) {
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