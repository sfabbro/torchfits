#pragma once
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <fitsio.h>

namespace py = pybind11;

/// Performance optimizations for TorchFits table operations
namespace torchfits_perf {

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
