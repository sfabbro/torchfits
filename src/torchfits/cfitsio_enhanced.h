#pragma once

#include <fitsio.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

// === REAL CFITSIO ENHANCED FEATURES ===
// Implementing the missing advanced CFITSIO functionality

namespace torchfits_cfitsio_enhanced {

/// Memory-mapped FITS file access using CFITSIO's memory mapping
class CFITSIOMemoryMapper {
public:
    CFITSIOMemoryMapper(const std::string& filename);
    ~CFITSIOMemoryMapper();
    
    /// Use CFITSIO's ffgmem for memory-mapped access
    torch::Tensor read_with_memory_mapping(int hdu_num = 1,
                                          const std::vector<long>& start = {},
                                          const std::vector<long>& shape = {});
    
    /// Check if file is suitable for memory mapping
    bool should_use_memory_mapping() const;
    
private:
    std::string filename_;
    fitsfile* fptr_;
    size_t file_size_;
    void* mapped_memory_;
    bool is_memory_mapped_;
};

/// Advanced buffered I/O using CFITSIO's buffering system
class CFITSIOBufferedReader {
public:
    CFITSIOBufferedReader(const std::string& filename);
    ~CFITSIOBufferedReader();
    
    /// Optimize buffer sizes based on file characteristics
    void optimize_buffers();
    
    /// Use fits_set_bscale for automatic scaling
    torch::Tensor read_with_scaling(int hdu_num = 1);
    
    /// Tile-based reading for compressed files
    torch::Tensor read_tiled(int hdu_num = 1, size_t tile_size = 1024);
    
private:
    std::string filename_;
    fitsfile* fptr_;
    size_t optimal_buffer_size_;
    bool is_compressed_;
};

/// Iterator-based data processing using fits_iterate_data
class CFITSIOIterator {
public:
    CFITSIOIterator(const std::string& filename, int hdu_num = 1);
    ~CFITSIOIterator();
    
    /// Process data using CFITSIO's iteration functions
    template<typename ProcessorFunc>
    void iterate_data(ProcessorFunc processor, size_t chunk_size = 10000);
    
    /// Parallel column processing for tables
    torch::Tensor parallel_column_read(const std::vector<std::string>& columns,
                                      size_t num_threads = 0);
    
    /// Row-wise filtering with CFITSIO optimization
    torch::Tensor filtered_read(const std::string& filter_expression);
    
private:
    std::string filename_;
    fitsfile* fptr_;
    int hdu_num_;
    int hdu_type_;
    
    static int iteration_callback(long total_rows, long offset, long first_rows,
                                 long n_values, int n_cols, iteratorCol* data,
                                 void* user_struct);
};

/// Compressed image handling with native decompression
class CFITSIOCompression {
public:
    /// Detect compression type
    static std::string detect_compression(const std::string& filename, int hdu_num = 1);
    
    /// Read compressed image with optimized decompression
    static torch::Tensor read_compressed_optimized(const std::string& filename,
                                                  int hdu_num = 1);
    
    /// Set compression parameters for better performance
    static void optimize_compression_reading(fitsfile* fptr);
    
private:
    static bool is_rice_compressed(fitsfile* fptr);
    static bool is_gzip_compressed(fitsfile* fptr);
    static bool is_hcompress_compressed(fitsfile* fptr);
};

/// Multi-HDU parallel processing
class CFITSIOParallelHDU {
public:
    CFITSIOParallelHDU(const std::string& filename);
    ~CFITSIOParallelHDU();
    
    /// Read multiple HDUs in parallel
    std::vector<torch::Tensor> read_hdus_parallel(const std::vector<int>& hdu_nums,
                                                  size_t num_threads = 0);
    
    /// Concurrent HDU access with thread safety
    torch::Tensor read_hdu_concurrent(int hdu_num);
    
private:
    std::string filename_;
    std::vector<std::unique_ptr<fitsfile*>> file_handles_;
    std::mutex access_mutex_;
    
    void initialize_handles(size_t num_handles);
};

/// Variable-length array (VLA) support
class CFITSIOVariableLength {
public:
    /// Read VLA columns efficiently
    static py::object read_vla_column(fitsfile* fptr, int col_num);
    
    /// Convert VLA to tensor format
    static torch::Tensor vla_to_tensor(const py::object& vla_data);
    
    /// Optimize VLA memory layout
    static void optimize_vla_access(fitsfile* fptr);
};

/// Real performance optimization using CFITSIO features
class CFITSIOPerformanceOptimizer {
public:
    /// Apply all CFITSIO-specific optimizations with known file size
    static void optimize_file_access(fitsfile* fptr, size_t file_size);
    
    /// Apply all CFITSIO-specific optimizations (backward compatibility)
    static void optimize_file_access(fitsfile* fptr);
    
    /// Set optimal I/O buffer sizes
    static void set_optimal_buffers(fitsfile* fptr, size_t file_size);
    
    /// Enable CFITSIO-level caching
    static void enable_cfitsio_caching(fitsfile* fptr);
    
    /// Optimize for specific data patterns
    static void optimize_for_pattern(fitsfile* fptr, const std::string& access_pattern);
    
private:
    static size_t calculate_optimal_buffer_size(size_t file_size, int data_type);
    static void set_compression_optimization(fitsfile* fptr);
    static void set_checksum_optimization(fitsfile* fptr);
};

} // namespace torchfits_cfitsio_enhanced
