#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "fits_reader.h"
#include "fits_writer.h"
#include "fits_utils.h"
#include "cache.h"
#include "wcs_utils.h"
#include "performance.h"
#include "memory_optimizer.h"

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

    // Initialize performance optimizations
    // torchfits_perf::initialize_performance_optimizations();  // Disabled temporarily

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

    // --- Writing Functions (v1.0) ---
    m.def("write_tensor_to_fits", &torchfits_writer::write_tensor_to_fits,
        py::arg("filename"),
        py::arg("data"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("overwrite") = false,
        "Write a PyTorch tensor to a FITS file as an image HDU."
    );

    m.def("write_tensors_to_mef", &torchfits_writer::write_tensors_to_mef,
        py::arg("filename"),
        py::arg("tensors"),
        py::arg("headers") = std::vector<std::map<std::string, std::string>>(),
        py::arg("extnames") = std::vector<std::string>(),
        py::arg("overwrite") = false,
        "Write multiple tensors to a multi-extension FITS file."
    );

    m.def("write_table_to_fits", &torchfits_writer::write_table_to_fits,
        py::arg("filename"),
        py::arg("table_data"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("column_units") = std::vector<std::string>(),
        py::arg("column_descriptions") = std::vector<std::string>(),
        py::arg("overwrite") = false,
        "Write a dictionary of tensors (table data) to a FITS table."
    );

    m.def("write_fits_table", &torchfits_writer::write_fits_table,
        py::arg("filename"),
        py::arg("fits_table"),
        py::arg("overwrite") = false,
        "Write a FitsTable object to a FITS file."
    );

    m.def("append_hdu_to_fits", &torchfits_writer::append_hdu_to_fits,
        py::arg("filename"),
        py::arg("data"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("extname") = std::string(""),
        "Append an HDU to an existing FITS file."
    );

    m.def("update_fits_header", &torchfits_writer::update_fits_header,
        py::arg("filename"),
        py::arg("hdu_num"),
        py::arg("updates"),
        "Update header keywords in an existing FITS file."
    );

    m.def("update_fits_data", &torchfits_writer::update_fits_data,
        py::arg("filename"),
        py::arg("hdu_num"),
        py::arg("new_data"),
        py::arg("start") = std::vector<long>(),
        py::arg("shape") = std::vector<long>(),
        "Update data in an existing FITS file (in-place modification)."
    );

    // --- WCS Functions ---
    m.def("world_to_pixel", &world_to_pixel, py::arg("world_coords"), py::arg("header"), "Convert world coordinates to pixel coordinates.");
    m.def("pixel_to_world", &pixel_to_world, py::arg("pixel_coords"), py::arg("header"), "Convert pixel coordinates to world coordinates.");
    
    // === Phase 2: Enhanced Writing Capabilities ===
    
    // Compression types
    py::enum_<torchfits_writer::CompressionType>(m, "CompressionType")
        .value("None", torchfits_writer::CompressionType::None)
        .value("GZIP", torchfits_writer::CompressionType::GZIP)
        .value("RICE", torchfits_writer::CompressionType::RICE)
        .value("HCOMPRESS", torchfits_writer::CompressionType::HCOMPRESS)
        .value("PLIO", torchfits_writer::CompressionType::PLIO);
    
    // Compression configuration
    py::class_<torchfits_writer::CompressionConfig>(m, "CompressionConfig")
        .def(py::init<>())
        .def_readwrite("type", &torchfits_writer::CompressionConfig::type)
        .def_readwrite("quantize_level", &torchfits_writer::CompressionConfig::quantize_level)
        .def_readwrite("quantize_dither", &torchfits_writer::CompressionConfig::quantize_dither)
        .def_readwrite("preserve_zeros", &torchfits_writer::CompressionConfig::preserve_zeros);
    
    // Advanced writing functions
    m.def("write_tensor_to_fits_advanced", &torchfits_writer::write_tensor_to_fits_advanced,
        py::arg("filename"),
        py::arg("data"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("compression") = torchfits_writer::CompressionConfig(),
        py::arg("overwrite") = false,
        py::arg("checksum") = false,
        "Enhanced tensor writing with compression and advanced options."
    );
    
    m.def("write_variable_length_array", &torchfits_writer::write_variable_length_array,
        py::arg("filename"),
        py::arg("arrays"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("overwrite") = false,
        "Write tensor with variable-length array support."
    );
    
    // Temporarily disabled - function not implemented yet
    /*
    m.def("write_table_to_fits_advanced", &torchfits_writer::write_table_to_fits_advanced,
        py::arg("filename"),
        py::arg("table_data"),
        py::arg("header") = std::map<std::string, std::string>(),
        py::arg("column_units") = std::vector<std::string>(),
        py::arg("column_descriptions") = std::vector<std::string>(),
        py::arg("compression") = torchfits_writer::CompressionConfig(),
        py::arg("overwrite") = false,
        py::arg("checksum") = false,
        "Advanced table writing with compression and optimizations."
    );
    */
    
    // Streaming writer
    py::class_<torchfits_writer::StreamingWriter>(m, "StreamingWriter")
        .def(py::init<const std::string&, const std::vector<long>&, torch::ScalarType, 
                     const torchfits_writer::CompressionConfig&, bool>(),
             py::arg("filename"), py::arg("dimensions"), 
             py::arg("dtype") = torch::kFloat32,
             py::arg("compression") = torchfits_writer::CompressionConfig(),
             py::arg("overwrite") = false)
        .def("write_sequential", &torchfits_writer::StreamingWriter::write_sequential,
             py::arg("data"), "Write data sequentially (streaming mode)")
        .def("finalize", &torchfits_writer::StreamingWriter::finalize,
             py::arg("header") = std::map<std::string, std::string>(),
             "Finalize the file (write headers, checksums, etc.)")
        .def("get_position", &torchfits_writer::StreamingWriter::get_position,
             "Get current write position");
    
    // === Phase 1: Performance Features ===
    // Temporarily disabled until performance.cpp is fixed
    /*
        // GPU Pipeline bindings
    py::class_<torchfits_perf::GPUPipeline::GPUConfig>(m, "GPUConfig")
        .def(py::init<>())
        .def_readwrite("enable_gpu", &torchfits_perf::GPUPipeline::GPUConfig::enable_gpu)
        .def_readwrite("use_pinned_memory", &torchfits_perf::GPUPipeline::GPUConfig::use_pinned_memory)
        .def_readwrite("gpu_memory_pool_size", &torchfits_perf::GPUPipeline::GPUConfig::gpu_memory_pool_size)
        .def_readwrite("enable_async_copy", &torchfits_perf::GPUPipeline::GPUConfig::enable_async_copy);

    py::class_<torchfits_perf::GPUPipeline>(m, "GPUPipeline")
        .def_static("initialize", &torchfits_perf::GPUPipeline::initialize,
                   py::arg("config"),
                   "Initialize GPU pipeline with configuration")
        .def_static("is_available", &torchfits_perf::GPUPipeline::is_available,
                   "Check if GPU pipeline is available")
        .def_static("read_to_gpu", &torchfits_perf::GPUPipeline::read_to_gpu,
                   py::arg("filename"), py::arg("hdu") = 1, py::arg("device") = "cuda:0",
                   "Read FITS data directly to GPU");

    // Parallel FITS bindings
    py::class_<torchfits_perf::ParallelFITS::Statistics>(m, "Statistics")
        .def_readwrite("mean", &torchfits_perf::ParallelFITS::Statistics::mean)
        .def_readwrite("std", &torchfits_perf::ParallelFITS::Statistics::std)
        .def_readwrite("min", &torchfits_perf::ParallelFITS::Statistics::min)
        .def_readwrite("max", &torchfits_perf::ParallelFITS::Statistics::max)
        .def_readwrite("count", &torchfits_perf::ParallelFITS::Statistics::count);
    
    py::class_<torchfits_perf::ParallelFITS>(m, "ParallelFITS")
        .def_static("parallel_read_image", &torchfits_perf::ParallelFITS::parallel_read_image,
                   py::arg("filename"), py::arg("hdu_num") = 1, py::arg("num_threads") = 0,
                   "Read FITS image data in parallel chunks")
        .def_static("parallel_compute_stats", &torchfits_perf::ParallelFITS::parallel_compute_stats,
                   py::arg("filename"), py::arg("hdu_num") = 1, py::arg("num_threads") = 0,
                   "Parallel statistics computation");
    */
    
    // Memory optimization bindings
    py::class_<torchfits_mem::AlignedMemoryPool::MemoryStats>(m, "MemoryStats")
        .def_readonly("total_allocated_bytes", &torchfits_mem::AlignedMemoryPool::MemoryStats::total_allocated_bytes)
        .def_readonly("pooled_tensors_count", &torchfits_mem::AlignedMemoryPool::MemoryStats::pooled_tensors_count)
        .def_readonly("cache_hit_rate_percent", &torchfits_mem::AlignedMemoryPool::MemoryStats::cache_hit_rate_percent);

    m.def("get_memory_pool", []() -> torchfits_mem::AlignedMemoryPool& {
        return torchfits_mem::AlignedMemoryPool::instance();
    }, py::return_value_policy::reference, "Get the global memory pool instance");

    py::class_<torchfits_mem::AlignedMemoryPool>(m, "AlignedMemoryPool")
        .def("get_tensor", &torchfits_mem::AlignedMemoryPool::get_tensor,
             py::arg("shape"), py::arg("dtype"), py::arg("device") = torch::kCPU,
             "Get or create an aligned tensor from the pool")
        .def("return_tensor", &torchfits_mem::AlignedMemoryPool::return_tensor,
             py::arg("tensor"),
             "Return tensor to pool for reuse")
        .def("clear", &torchfits_mem::AlignedMemoryPool::clear,
             "Clear all cached tensors")
        .def("get_stats", &torchfits_mem::AlignedMemoryPool::get_stats,
             "Get memory usage statistics");

    m.def("create_aligned_tensor", &torchfits_mem::AlignedTensorFactory::create_aligned_tensor,
          py::arg("shape"), py::arg("dtype"), py::arg("device") = torch::kCPU, py::arg("fits_compatible") = true,
          "Create a memory-aligned tensor optimized for FITS data");

    m.def("is_optimally_aligned", &torchfits_mem::AlignedTensorFactory::is_optimally_aligned,
          py::arg("tensor"),
          "Check if tensor is optimally aligned for FITS operations");

    // Advanced CFITSIO features will be added in future versions
}
