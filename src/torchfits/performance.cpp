#include "performance.h"
#include "fits_utils.h"
#include "debug.h"

// Platform-specific SIMD includes
#ifdef __x86_64__
    #include <immintrin.h>  // x86 SIMD intrinsics
#elif defined(__aarch64__) || defined(__arm64__)
    #include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include <algorithm>
#include <chrono>

namespace torchfits_perf {

// Global memory pool instance
TensorMemoryPool* global_memory_pool = nullptr;

// --- ParallelTableReader Implementation ---

ParallelTableReader::ParallelTableReader(int threads) 
    : stop_workers(false), num_threads(threads) {
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    }
    DEBUG_LOG("ParallelTableReader initialized with " + std::to_string(num_threads) + " threads");
}

ParallelTableReader::~ParallelTableReader() {
    stop_workers_and_wait();
}

void ParallelTableReader::start_workers(fitsfile* fptr) {
    stop_workers = false;
    worker_threads.clear();
    worker_threads.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        worker_threads.emplace_back(&ParallelTableReader::worker_function, this, fptr);
    }
}

void ParallelTableReader::stop_workers_and_wait() {
    {
        std::lock_guard<std::mutex> lock(task_mutex);
        stop_workers = true;
    }
    task_cv.notify_all();
    
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads.clear();
}

void ParallelTableReader::worker_function(fitsfile* fptr) {
    while (true) {
        std::shared_ptr<ColumnTask> task;
        
        {
            std::unique_lock<std::mutex> lock(task_mutex);
            task_cv.wait(lock, [this] { return !task_queue.empty() || stop_workers; });
            
            if (stop_workers && task_queue.empty()) {
                break;
            }
            
            if (!task_queue.empty()) {
                task = task_queue.front();
                task_queue.pop();
            }
        }
        
        if (task) {
            auto result = std::make_shared<ColumnResult>();
            result->name = task->name;
            
            try {
                result->data = read_column_optimized(fptr, *task);
                result->success = true;
            } catch (const std::exception& e) {
                result->success = false;
                result->error_message = e.what();
                DEBUG_LOG("Error reading column " + task->name + ": " + e.what());
            }
            
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result_queue.push(result);
            }
            result_cv.notify_one();
        }
    }
}

torch::Tensor ParallelTableReader::read_column_optimized(fitsfile* fptr, const ColumnTask& task) {
    int status = 0;
    
    // Optimize tensor allocation based on device
    torch::Tensor tensor;
    if (GPUTensorAllocator::supports_direct_allocation(task.device)) {
        tensor = GPUTensorAllocator::allocate_gpu_tensor_direct({task.num_rows}, 
                                                              torch::kFloat64, 
                                                              task.device);
    } else {
        // Use memory pool for CPU tensors
        tensor = global_memory_pool->get_tensor({task.num_rows}, torch::kFloat64, task.device);
    }
    
    // Optimized reading based on data type
    switch (task.typecode) {
        case TFLOAT: {
            std::vector<float> buffer(task.num_rows);
            fits_read_col(fptr, TFLOAT, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading float column: " + task.name);
            
            // Use SIMD for fast conversion if available
            auto tensor_ptr = tensor.data_ptr<double>();
            if (SIMDOptimizer::simd_available()) {
                // Convert float to double with SIMD
                for (long i = 0; i < task.num_rows; ++i) {
                    tensor_ptr[i] = static_cast<double>(buffer[i]);
                }
            } else {
                std::copy(buffer.begin(), buffer.end(), tensor_ptr);
            }
            break;
        }
        
        case TDOUBLE: {
            // Direct read into tensor for double precision
            fits_read_col(fptr, TDOUBLE, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, tensor.data_ptr<double>(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading double column: " + task.name);
            break;
        }
        
        case TLONG: {
            std::vector<long> buffer(task.num_rows);
            fits_read_col(fptr, TLONG, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading long column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](long val) { return static_cast<double>(val); });
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported column type: " + std::to_string(task.typecode));
    }
    
    return tensor;
}

py::dict ParallelTableReader::read_columns_parallel(fitsfile* fptr,
                                                  const std::vector<std::string>& columns,
                                                  long start_row,
                                                  long num_rows,
                                                  torch::Device device) {
    DEBUG_SCOPE;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // CFITSIO Note: For thread safety, we need to be careful with shared fitsfile* access
    // For now, use sequential processing with optimized memory management
    // TODO: Implement proper multi-threaded CFITSIO access with file duplication
    
    py::dict result_dict;
    int status = 0;
    
    // Process each column with optimized memory allocation
    for (const auto& col_name : columns) {
        int col_number;
        int typecode;
        long repeat, width;
        
        // Get column metadata
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &col_number, &status);
        if (status) throw_fits_error(status, "Error finding column: " + col_name);
        
        fits_get_coltype(fptr, col_number, &typecode, &repeat, &width, &status);
        if (status) throw_fits_error(status, "Error getting column type for: " + col_name);
        
        // Use optimized tensor allocation
        torch::Tensor col_data;
        if (GPUTensorAllocator::supports_direct_allocation(device)) {
            col_data = GPUTensorAllocator::allocate_gpu_tensor_direct({num_rows}, 
                                                                    torch::kFloat64, 
                                                                    device);
        } else {
            // Use memory pool for CPU tensors
            col_data = global_memory_pool->get_tensor({num_rows}, torch::kFloat64, device);
        }
        
        // Optimized reading based on data type
        switch (typecode) {
            case TFLOAT: {
                std::vector<float> buffer(num_rows);
                fits_read_col(fptr, TFLOAT, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading float column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                for (long i = 0; i < num_rows; ++i) {
                    tensor_ptr[i] = static_cast<double>(buffer[i]);
                }
                break;
            }
            
            case TDOUBLE: {
                fits_read_col(fptr, TDOUBLE, col_number, start_row + 1, 1, num_rows,
                             nullptr, col_data.data_ptr<double>(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading double column: " + col_name);
                break;
            }
            
            case TLONG: {
                std::vector<long> buffer(num_rows);
                fits_read_col(fptr, TLONG, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading long column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](long val) { return static_cast<double>(val); });
                break;
            }
            
            default:
                throw std::runtime_error("Unsupported column type: " + std::to_string(typecode));
        }
        
        result_dict[col_name.c_str()] = col_data;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    DEBUG_LOG("Optimized read of " + std::to_string(columns.size()) + " columns completed in " + 
              std::to_string(duration.count()) + "ms");
    
    return result_dict;
}

// --- TensorMemoryPool Implementation ---

TensorMemoryPool::TensorMemoryPool(size_t max_size) : max_pool_size(max_size) {}

torch::Tensor TensorMemoryPool::get_tensor(const std::vector<int64_t>& shape,
                                         torch::Dtype dtype,
                                         torch::Device device) {
    PoolKey key{shape, dtype, device};
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto it = pools.find(key);
    
    if (it != pools.end() && !it->second.empty()) {
        auto tensor = it->second.front();
        it->second.pop();
        
        // Zero out the tensor for clean reuse
        tensor.zero_();
        return tensor;
    }
    
    // Create new tensor
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    return torch::empty(shape, options);
}

void TensorMemoryPool::return_tensor(torch::Tensor tensor) {
    if (!tensor.defined()) return;
    
    PoolKey key{tensor.sizes().vec(), tensor.scalar_type(), tensor.device()};
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto& pool = pools[key];
    
    if (pool.size() < max_pool_size) {
        pool.push(tensor);
    }
    // If pool is full, just let the tensor be garbage collected
}

void TensorMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    pools.clear();
}

// --- GPUTensorAllocator Implementation ---

torch::Tensor GPUTensorAllocator::allocate_gpu_tensor_direct(const std::vector<int64_t>& shape,
                                                           torch::Dtype dtype,
                                                           torch::Device device) {
    if (device.is_cuda() && torch::cuda::is_available()) {
        // Direct GPU allocation
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        return torch::empty(shape, options);
    }
    
    // Fallback to CPU allocation
    auto options = torch::TensorOptions().dtype(dtype);
    auto tensor = torch::empty(shape, options);
    
    if (device.is_cuda()) {
        return tensor.to(device);
    }
    
    return tensor;
}

bool GPUTensorAllocator::supports_direct_allocation(torch::Device device) {
    return device.is_cuda() && torch::cuda::is_available();
}

std::string GPUTensorAllocator::get_allocation_strategy(torch::Device device) {
    if (device.is_cuda() && torch::cuda::is_available()) {
        return "direct_gpu";
    }
    return "cpu_with_transfer";
}

// --- SIMDOptimizer Implementation ---

void SIMDOptimizer::convert_float_array_simd(const float* src, float* dst, size_t count) {
#ifdef __x86_64__
    // x86 AVX optimization
    #ifdef __AVX2__
        size_t simd_count = count & ~7; // Process 8 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 8) {
            __m256 data = _mm256_load_ps(&src[i]);
            _mm256_store_ps(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
    #else
        std::copy(src, src + count, dst);
    #endif
#elif defined(__aarch64__) || defined(__arm64__)
    // ARM NEON optimization
    size_t simd_count = count & ~3; // Process 4 elements at a time
    
    for (size_t i = 0; i < simd_count; i += 4) {
        float32x4_t data = vld1q_f32(&src[i]);
        vst1q_f32(&dst[i], data);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        dst[i] = src[i];
    }
#else
    std::copy(src, src + count, dst);
#endif
}

void SIMDOptimizer::convert_double_array_simd(const double* src, double* dst, size_t count) {
#ifdef __x86_64__
    // x86 AVX optimization
    #ifdef __AVX2__
        size_t simd_count = count & ~3; // Process 4 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 4) {
            __m256d data = _mm256_load_pd(&src[i]);
            _mm256_store_pd(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
    #else
        std::copy(src, src + count, dst);
    #endif
#elif defined(__aarch64__) || defined(__arm64__)
    // ARM NEON optimization
    size_t simd_count = count & ~1; // Process 2 elements at a time
    
    for (size_t i = 0; i < simd_count; i += 2) {
        float64x2_t data = vld1q_f64(&src[i]);
        vst1q_f64(&dst[i], data);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        dst[i] = src[i];
    }
#else
    std::copy(src, src + count, dst);
#endif
}

bool SIMDOptimizer::simd_available() {
#ifdef __x86_64__
    #ifdef __AVX2__
        return true;
    #else
        return false;
    #endif
#elif defined(__aarch64__) || defined(__arm64__)
    return true;  // ARM NEON is available on all ARM64
#else
    return false;
#endif
}

// --- Initialization Functions ---

void initialize_performance_optimizations() {
    if (!global_memory_pool) {
        global_memory_pool = new TensorMemoryPool(200); // Pool size of 200 tensors
        DEBUG_LOG("Performance optimizations initialized");
    }
}

void cleanup_performance_optimizations() {
    if (global_memory_pool) {
        delete global_memory_pool;
        global_memory_pool = nullptr;
        DEBUG_LOG("Performance optimizations cleaned up");
    }
}

} // namespace torchfits_perf
