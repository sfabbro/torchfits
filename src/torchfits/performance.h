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
#include "fits_utils.h"

// Platform-specific memory mapping
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace py = pybind11;

/// Performance optimizations for TorchFits table operations
namespace torchfits_perf {

/// Memory-mapped file handle for efficient large file access
struct MappedFile {
    void* ptr;
    size_t size;
    std::string filename;
    bool is_valid;
    
    // Platform-specific handles
#ifdef _WIN32
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int fd;
#endif

    MappedFile() : ptr(nullptr), size(0), is_valid(false) {
#ifdef _WIN32
        file_handle = INVALID_HANDLE_VALUE;
        mapping_handle = INVALID_HANDLE_VALUE;
#else
        fd = -1;
#endif
    }
    
    ~MappedFile();
};

/// Memory mapping manager for FITS files
class MemoryMapper {
private:
    std::unordered_map<std::string, std::shared_ptr<MappedFile>> mapped_files;
    std::mutex mapping_mutex;
    size_t min_file_size_for_mapping;  // Don't map small files
    size_t max_mapped_files;           // Limit number of mapped files
    
public:
    MemoryMapper(size_t min_size = 10 * 1024 * 1024,  // 10MB minimum
                 size_t max_files = 16);                // Max 16 mapped files
    ~MemoryMapper();
    
    /// Memory map a FITS file for efficient access
    std::shared_ptr<MappedFile> map_file(const std::string& filename);
    
    /// Unmap a file and free resources
    void unmap_file(const std::string& filename);
    
    /// Check if memory mapping would be beneficial for this file
    bool should_use_memory_mapping(const std::string& filename, size_t file_size = 0);
    
    /// Get memory-mapped data pointer for direct CFITSIO access
    void* get_mapped_pointer(const std::string& filename);
    
    /// Get file size from mapped file
    size_t get_mapped_size(const std::string& filename);
    
    /// Cleanup all mappings (called on shutdown)
    void cleanup_all_mappings();
    
    /// Get statistics about memory mapping usage
    struct MappingStats {
        size_t num_mapped_files;
        size_t total_mapped_bytes;
        size_t cache_hits;
        size_t cache_misses;
    };
    MappingStats get_stats() const;
};

/// Global memory mapper instance
extern std::unique_ptr<MemoryMapper> global_memory_mapper;

// === Buffered I/O System ===
// Provides efficient streaming I/O for large FITS files

struct BufferedIOConfig {
    size_t buffer_size = 1024 * 1024;  // 1MB default buffer
    size_t read_ahead_size = 4 * 1024 * 1024;  // 4MB read-ahead
    bool enable_async_prefetch = true;
    size_t max_concurrent_reads = 4;
};

class BufferedReader {
public:
    BufferedReader(const std::string& filename, const BufferedIOConfig& config = {});
    ~BufferedReader();
    
    /// Read data into buffer, returns bytes read
    size_t read(void* buffer, size_t size, size_t offset);
    
    /// Prefetch data at given offset (async)
    void prefetch(size_t offset, size_t size);
    
    /// Get file size
    size_t get_file_size() const { return file_size_; }
    
    /// Check if data is available in buffer
    bool is_buffered(size_t offset, size_t size) const;

private:
    std::string filename_;
    BufferedIOConfig config_;
    size_t file_size_;
    
    // Circular buffer system
    struct BufferBlock {
        std::vector<uint8_t> data;
        size_t file_offset;
        size_t valid_size;
        bool is_valid;
        std::chrono::steady_clock::time_point last_accessed;
    };
    
    mutable std::mutex buffer_mutex_;
    std::vector<BufferBlock> buffers_;
    
    // Async prefetching
    std::thread prefetch_thread_;
    std::queue<std::pair<size_t, size_t>> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    bool shutdown_prefetch_ = false;
    
    void prefetch_worker();
    BufferBlock* find_buffer_for_offset(size_t offset) const;
    BufferBlock* get_available_buffer();
    void load_buffer(BufferBlock* buffer, size_t offset, size_t size);
};

// === Iterator Functions ===
// Efficient iteration over FITS data without full loading

template<typename T>
class FITSDataIterator {
public:
    FITSDataIterator(const std::string& filename, int hdu_num = 1, size_t chunk_size = 1024);
    ~FITSDataIterator();
    
    /// Check if more data is available
    bool has_next() const { return current_position_ < total_elements_; }
    
    /// Get next chunk of data
    torch::Tensor next_chunk();
    
    /// Skip to specific position
    void seek(size_t position);
    
    /// Get current position
    size_t position() const { return current_position_; }
    
    /// Get total number of elements
    size_t size() const { return total_elements_; }
    
    /// Get data dimensions
    const std::vector<long>& dimensions() const { return dims_; }

private:
    std::string filename_;
    int hdu_num_;
    size_t chunk_size_;
    size_t current_position_;
    size_t total_elements_;
    std::vector<long> dims_;
    int fits_type_;
    
    // FITS file wrapper for efficient access
    std::unique_ptr<FITSFileWrapper> fits_file_;
    std::unique_ptr<BufferedReader> buffered_reader_;
    
    void initialize_iterator();
    torch::Tensor read_chunk_raw(size_t start_pos, size_t count);
};

// Specialized iterators for common data types
using Float32Iterator = FITSDataIterator<float>;
using Float64Iterator = FITSDataIterator<double>;
using Int32Iterator = FITSDataIterator<int32_t>;
using Int16Iterator = FITSDataIterator<int16_t>;

// === Multi-threading Support ===
// Parallel processing for FITS operations

class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    /// Submit a task to the thread pool
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (stop_) {
                throw std::runtime_error("submit on stopped ThreadPool");
            }
            
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    /// Wait for all tasks to complete
    void wait_all();
    
    /// Get number of threads
    size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

/// Parallel FITS operations
class ParallelFITS {
public:
    /// Read FITS image data in parallel chunks
    static torch::Tensor parallel_read_image(const std::string& filename, 
                                            int hdu_num = 1,
                                            size_t num_threads = 0);
    
    /// Write FITS image data in parallel
    static void parallel_write_image(const std::string& filename,
                                   const torch::Tensor& data,
                                   size_t num_threads = 0);
    
    /// Parallel histogram computation
    static torch::Tensor parallel_histogram(const std::string& filename,
                                          int hdu_num = 1,
                                          int bins = 256,
                                          size_t num_threads = 0);
    
    /// Parallel statistics computation
    struct Statistics {
        double mean;
        double std;
        double min;
        double max;
        size_t count;
    };
    
    static Statistics parallel_compute_stats(const std::string& filename,
                                           int hdu_num = 1,
                                           size_t num_threads = 0);

private:
    static ThreadPool& get_thread_pool();
};

// === GPU-Direct Pipeline ===
// Direct GPU memory allocation and transfer for FITS data

class GPUPipeline {
public:
    /// Configuration for GPU operations
    struct GPUConfig {
        bool enable_gpu = true;
        torch::Device device = torch::kCUDA;
        bool use_pinned_memory = true;
        size_t gpu_memory_pool_size = 2UL * 1024 * 1024 * 1024; // 2GB
        bool enable_async_copy = true;
    };
    
    /// Initialize GPU pipeline
    static void initialize(const GPUConfig& config);
    
    /// Check if GPU is available and initialized
    static bool is_available();
    
    /// Read FITS data directly to GPU memory
    static torch::Tensor read_to_gpu(const std::string& filename, 
                                   int hdu_num = 1,
                                   const torch::Device& device = torch::kCUDA);
    
    /// Stream FITS data to GPU in chunks
    class GPUStreamer {
    public:
        GPUStreamer(const std::string& filename, 
                   int hdu_num = 1,
                   size_t chunk_size = 1024 * 1024,
                   const torch::Device& device = torch::kCUDA);
        ~GPUStreamer();
        
        bool has_next() const;
        torch::Tensor next_chunk();
        void async_prefetch_next();
        
    private:
        std::string filename_;
        int hdu_num_;
        size_t chunk_size_;
        torch::Device device_;
        size_t current_position_;
        size_t total_elements_;
        
        // Async streaming
        std::thread prefetch_thread_;
        std::queue<torch::Tensor> gpu_buffer_queue_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        bool shutdown_ = false;
        
        void prefetch_worker();
    };
    
    /// Memory pool for GPU allocations
    class GPUMemoryPool {
    public:
        GPUMemoryPool(size_t pool_size, const torch::Device& device);
        ~GPUMemoryPool();
        
        torch::Tensor allocate(const torch::IntArrayRef& sizes, torch::ScalarType dtype);
        void deallocate(torch::Tensor& tensor);
        void cleanup();
        
        size_t get_available_memory() const;
        size_t get_total_memory() const { return pool_size_; }
        
    private:
        torch::Device device_;
        size_t pool_size_;
        size_t used_memory_;
        std::vector<torch::Tensor> free_tensors_;
        std::mutex pool_mutex_;
    };

private:
    static std::unique_ptr<GPUConfig> config_;
    static std::unique_ptr<GPUMemoryPool> gpu_pool_;
    static bool initialized_;
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
};

/// Column result data
struct ColumnResult {
    std::string name;
    torch::Tensor data;
    bool success;
    std::string error_message;
};

/// Parallel table reader for high-performance column processing
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
    
public:
    explicit ParallelTableReader(int threads = std::thread::hardware_concurrency());
    ~ParallelTableReader();
    
    /// Read multiple columns in parallel
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

/// Memory pool for tensor reuse
class TensorMemoryPool {
private:
    struct PoolKey {
        std::vector<int64_t> shape;
        torch::Dtype dtype;
        torch::Device device;
        
        bool operator==(const PoolKey& other) const {
            return shape == other.shape && dtype == other.dtype && device == other.device;
        }
    };
    
    struct PoolKeyHash {
        std::size_t operator()(const PoolKey& key) const {
            std::size_t h1 = std::hash<std::string>{}(torch::toString(key.dtype));
            std::size_t h2 = std::hash<std::string>{}(key.device.str());
            std::size_t h3 = 0;
            for (auto dim : key.shape) {
                h3 ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h3 << 6) + (h3 >> 2);
            }
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    
    std::unordered_map<PoolKey, std::queue<torch::Tensor>, PoolKeyHash> pools;
    std::mutex pool_mutex;
    size_t max_pool_size;
    
public:
    explicit TensorMemoryPool(size_t max_size = 100);
    
    /// Get tensor from pool or create new one
    torch::Tensor get_tensor(const std::vector<int64_t>& shape, 
                           torch::Dtype dtype, 
                           torch::Device device);
    
    /// Return tensor to pool for reuse
    void return_tensor(torch::Tensor tensor);
    
    /// Clear all pools
    void clear();
};

/// GPU-direct tensor allocation utilities
class GPUTensorAllocator {
public:
    /// Allocate tensor directly on GPU if available
    static torch::Tensor allocate_gpu_tensor_direct(const std::vector<int64_t>& shape, 
                                                   torch::Dtype dtype,
                                                   torch::Device device);
    
    /// Check if device supports direct GPU allocation
    static bool supports_direct_allocation(torch::Device device);
    
    /// Get optimal allocation strategy for device
    static std::string get_allocation_strategy(torch::Device device);
};

/// SIMD optimization utilities
class SIMDOptimizer {
public:
    /// Vectorized data conversion for float arrays
    static void convert_float_array_simd(const float* src, float* dst, size_t count);
    
    /// Vectorized data conversion for double arrays  
    static void convert_double_array_simd(const double* src, double* dst, size_t count);
    
    /// Check if SIMD optimizations are available
    static bool simd_available();
};

/// Global memory pool instance
extern TensorMemoryPool* global_memory_pool;

/// Initialize performance optimizations
void initialize_performance_optimizations();

/// Cleanup performance optimizations
void cleanup_performance_optimizations();

} // namespace torchfits_perf
