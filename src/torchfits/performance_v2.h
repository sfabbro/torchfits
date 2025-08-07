#pragma once
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <unordered_map>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <fitsio.h>

namespace py = pybind11;

/// Performance optimizations for TorchFits v1.0
namespace torchfits_perf {

/// Memory mapping utilities for large files
class MemoryMapper {
private:
    struct MappedFile {
        void* ptr;
        size_t size;
        std::string filename;
        bool is_valid;
    };
    
    std::unordered_map<std::string, std::shared_ptr<MappedFile>> mapped_files;
    std::mutex mapping_mutex;
    
public:
    /// Memory map a FITS file for efficient access
    std::shared_ptr<MappedFile> map_file(const std::string& filename);
    
    /// Unmap a file
    void unmap_file(const std::string& filename);
    
    /// Check if memory mapping is beneficial for file
    bool should_use_memory_mapping(const std::string& filename, size_t file_size);
    
    /// Get memory-mapped data pointer for CFITSIO
    void* get_mapped_pointer(const std::string& filename);
    
    /// Cleanup all mappings
    void cleanup_all_mappings();
};

/// Buffer management for optimized I/O
class BufferManager {
private:
    struct BufferInfo {
        std::unique_ptr<char[]> buffer;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::vector<BufferInfo> buffers;
    std::mutex buffer_mutex;
    size_t max_buffer_size;
    
public:
    explicit BufferManager(size_t max_size = 64 * 1024 * 1024); // 64MB default
    
    /// Get optimal buffer for file type and size
    char* get_buffer(size_t required_size, const std::string& file_type = "");
    
    /// Return buffer to pool
    void return_buffer(char* buffer);
    
    /// Get optimal buffer size for operation
    size_t get_optimal_buffer_size(const std::string& filename, 
                                 const std::string& operation_type,
                                 size_t data_size);
    
    /// Cleanup unused buffers
    void cleanup_old_buffers();
};

/// Column reading task for parallel processing
struct ColumnTask {
    std::string name;
    int number;
    int typecode;
    long repeat;
    long width;
    long start_row;
    long num_rows;
    torch::Device device;
    bool use_iterator;  // Use fits_iterate_data for efficiency
};

/// Column result data
struct ColumnResult {
    std::string name;
    torch::Tensor data;
    bool success;
    std::string error_message;
    double read_time_ms;
};

/// Enhanced parallel table reader with v1.0 features
class ParallelTableReader {
private:
    std::vector<std::thread> worker_threads;
    std::queue<std::shared_ptr<ColumnTask>> task_queue;
    std::queue<std::shared_ptr<ColumnResult>> result_queue;
    std::mutex task_mutex;
    std::mutex result_mutex;
    std::condition_variable task_cv;
    std::condition_variable result_cv;
    bool stop_workers;
    int num_threads;
    
    /// Worker function that processes column reading tasks
    void worker_function(fitsfile* fptr);
    
    /// Read a single column with optimized type handling
    torch::Tensor read_column_optimized(fitsfile* fptr, const ColumnTask& task);
    
    /// Use fits_iterate_data for efficient column processing
    torch::Tensor read_column_with_iterator(fitsfile* fptr, const ColumnTask& task);
    
public:
    explicit ParallelTableReader(int threads = 0);
    ~ParallelTableReader();
    
    /// Read multiple columns in parallel with v1.0 optimizations
    py::dict read_columns_parallel(fitsfile* fptr, 
                                 const std::vector<std::string>& columns,
                                 long start_row,
                                 long num_rows,
                                 torch::Device device);
    
    /// Start worker threads
    void start_workers(fitsfile* fptr);
    
    /// Stop worker threads
    void stop_workers_and_wait();
};

/// Memory pool for tensor reuse with pinned memory support
class TensorMemoryPool {
private:
    struct PoolKey {
        std::vector<int64_t> shape;
        torch::Dtype dtype;
        torch::Device device;
        bool pinned;
        
        bool operator==(const PoolKey& other) const {
            return shape == other.shape && dtype == other.dtype && 
                   device == other.device && pinned == other.pinned;
        }
    };
    
    struct PoolKeyHash {
        std::size_t operator(>(const PoolKey& key) const {
            std::size_t h1 = std::hash<std::string>{}(torch::toString(key.dtype));
            std::size_t h2 = std::hash<std::string>{}(key.device.str());
            std::size_t h3 = std::hash<bool>{}(key.pinned);
            std::size_t h4 = 0;
            for (auto dim : key.shape) {
                h4 ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h4 << 6) + (h4 >> 2);
            }
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };
    
    std::unordered_map<PoolKey, std::queue<torch::Tensor>, PoolKeyHash> pools;
    std::mutex pool_mutex;
    size_t max_pool_size;
    size_t current_pinned_memory;
    size_t max_pinned_memory;
    
public:
    explicit TensorMemoryPool(size_t max_size = 100, size_t max_pinned_mb = 512);
    
    /// Get tensor from pool or create new one
    torch::Tensor get_tensor(const std::vector<int64_t>& shape, 
                           torch::Dtype dtype, 
                           torch::Device device,
                           bool pinned = false);
    
    /// Return tensor to pool for reuse
    void return_tensor(torch::Tensor tensor);
    
    /// Clear all pools
    void clear();
    
    /// Get memory usage statistics
    size_t get_pinned_memory_usage() const { return current_pinned_memory; }
};

/// GPU-direct tensor allocation with CUDA streams
class GPUTensorAllocator {
private:
    static std::unordered_map<torch::Device, std::vector<void*>> stream_pool;
    static std::mutex stream_mutex;
    
public:
    /// Allocate tensor directly on GPU with stream support
    static torch::Tensor allocate_gpu_tensor_direct(const std::vector<int64_t>& shape, 
                                                   torch::Dtype dtype,
                                                   torch::Device device,
                                                   void* cuda_stream = nullptr);
    
    /// Allocate pinned memory tensor for fast GPU transfer
    static torch::Tensor allocate_pinned_tensor(const std::vector<int64_t>& shape,
                                               torch::Dtype dtype);
    
    /// Check if device supports direct GPU allocation
    static bool supports_direct_allocation(torch::Device device);
    
    /// Get CUDA stream for async operations
    static void* get_cuda_stream(torch::Device device);
    
    /// Return CUDA stream to pool
    static void return_cuda_stream(torch::Device device, void* stream);
    
    /// Copy tensor to GPU asynchronously
    static std::future<torch::Tensor> copy_to_gpu_async(torch::Tensor cpu_tensor,
                                                       torch::Device gpu_device,
                                                       void* cuda_stream = nullptr);
};

/// SIMD optimization utilities with architecture detection
class SIMDOptimizer {
private:
    static bool avx2_available;
    static bool neon_available;
    static bool initialized;
    
    static void detect_capabilities();
    
public:
    /// Initialize SIMD detection
    static void initialize();
    
    /// Vectorized data conversion for float arrays
    static void convert_float_array_simd(const float* src, float* dst, size_t count);
    
    /// Vectorized data conversion for double arrays  
    static void convert_double_array_simd(const double* src, double* dst, size_t count);
    
    /// Vectorized byte swapping for endianness
    static void byte_swap_simd(void* data, size_t element_size, size_t count);
    
    /// Check if SIMD optimizations are available
    static bool simd_available();
    
    /// Get available SIMD instruction sets
    static std::vector<std::string> get_available_simd();
};

/// Iterator-based processing for large tables
class TableIterator {
private:
    fitsfile* fptr;
    long start_row;
    long end_row;
    long current_row;
    long chunk_size;
    std::vector<int> column_numbers;
    
public:
    TableIterator(fitsfile* fptr, const std::vector<int>& columns, 
                 long start = 1, long end = -1, long chunk = 10000);
    
    /// Process table data in chunks using fits_iterate_data
    template<typename Func>
    void iterate_chunks(Func processor);
    
    /// Get optimal chunk size for table
    static long get_optimal_chunk_size(fitsfile* fptr, const std::vector<int>& columns);
};

/// HDU parallel processing for MEF files
class HDUProcessor {
private:
    struct HDUTask {
        int hdu_number;
        std::string hdu_name;
        std::function<py::object(fitsfile*, int)> processor;
    };
    
    std::vector<std::thread> workers;
    std::queue<HDUTask> task_queue;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    bool stop_processing;
    
public:
    explicit HDUProcessor(int num_threads = 0);
    ~HDUProcessor();
    
    /// Process multiple HDUs in parallel
    std::vector<py::object> process_hdus_parallel(
        const std::string& filename,
        const std::vector<int>& hdu_numbers,
        std::function<py::object(fitsfile*, int)> processor);
    
    /// Start worker threads
    void start_workers();
    
    /// Stop worker threads
    void stop_workers();
};

/// Performance monitoring and optimization
class PerformanceMonitor {
private:
    struct OperationStats {
        double total_time_ms;
        long total_operations;
        double min_time_ms;
        double max_time_ms;
        std::string operation_type;
    };
    
    std::unordered_map<std::string, OperationStats> stats;
    std::mutex stats_mutex;
    
public:
    /// Record operation timing
    void record_operation(const std::string& operation, double time_ms);
    
    /// Get performance statistics
    py::dict get_statistics() const;
    
    /// Get optimization recommendations
    std::vector<std::string> get_recommendations() const;
    
    /// Reset statistics
    void reset();
};

/// Global instances
extern std::unique_ptr<MemoryMapper> global_memory_mapper;
extern std::unique_ptr<BufferManager> global_buffer_manager;
extern std::unique_ptr<TensorMemoryPool> global_memory_pool;
extern std::unique_ptr<PerformanceMonitor> global_performance_monitor;

/// Initialize all performance optimizations
void initialize_performance_optimizations();

/// Cleanup all performance optimizations
void cleanup_performance_optimizations();

/// Get performance optimization status
py::dict get_optimization_status();

} // namespace torchfits_perf
