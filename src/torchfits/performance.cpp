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
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

namespace torchfits_perf {

// Global instances
TensorMemoryPool* global_memory_pool = nullptr;
std::unique_ptr<MemoryMapper> global_memory_mapper = nullptr;

// --- MappedFile Implementation ---

MappedFile::~MappedFile() {
    if (!is_valid || ptr == nullptr) return;
    
#ifdef _WIN32
    if (ptr != nullptr) {
        UnmapViewOfFile(ptr);
        ptr = nullptr;
    }
    if (mapping_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(mapping_handle);
        mapping_handle = INVALID_HANDLE_VALUE;
    }
    if (file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
    }
#else
    if (ptr != MAP_FAILED && ptr != nullptr) {
        munmap(ptr, size);
        ptr = nullptr;
    }
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
#endif
    is_valid = false;
}

// --- MemoryMapper Implementation ---

MemoryMapper::MemoryMapper(size_t min_size, size_t max_files) 
    : min_file_size_for_mapping(min_size), max_mapped_files(max_files) {
    DEBUG_LOG("MemoryMapper initialized: min_size=" + std::to_string(min_size) + 
              ", max_files=" + std::to_string(max_files));
}

MemoryMapper::~MemoryMapper() {
    cleanup_all_mappings();
}

std::shared_ptr<MappedFile> MemoryMapper::map_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    
    // Check if already mapped
    auto it = mapped_files.find(filename);
    if (it != mapped_files.end()) {
        DEBUG_LOG("Memory mapping cache hit for " + filename);
        return it->second;
    }
    
    // Check if we should memory map this file
    if (!should_use_memory_mapping(filename)) {
        DEBUG_LOG("File " + filename + " not suitable for memory mapping");
        return nullptr;
    }
    
    // Check if we're at the mapping limit
    if (mapped_files.size() >= max_mapped_files) {
        // Remove oldest mapping (simple LRU-like strategy)
        DEBUG_LOG("Memory mapping limit reached, cleaning up oldest mapping");
        auto oldest = mapped_files.begin();
        mapped_files.erase(oldest);
    }
    
    auto mapped = std::make_shared<MappedFile>();
    mapped->filename = filename;
    
#ifdef _WIN32
    // Windows implementation
    mapped->file_handle = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    
    if (mapped->file_handle == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("Failed to open file for memory mapping: " + filename);
        return nullptr;
    }
    
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(mapped->file_handle, &file_size)) {
        CloseHandle(mapped->file_handle);
        return nullptr;
    }
    mapped->size = static_cast<size_t>(file_size.QuadPart);
    
    mapped->mapping_handle = CreateFileMappingA(
        mapped->file_handle,
        nullptr,
        PAGE_READONLY,
        0, 0,
        nullptr
    );
    
    if (mapped->mapping_handle == nullptr) {
        CloseHandle(mapped->file_handle);
        return nullptr;
    }
    
    mapped->ptr = MapViewOfFile(
        mapped->mapping_handle,
        FILE_MAP_READ,
        0, 0, 0
    );
    
    if (mapped->ptr == nullptr) {
        CloseHandle(mapped->mapping_handle);
        CloseHandle(mapped->file_handle);
        return nullptr;
    }
    
#else
    // Unix implementation (Linux, macOS)
    mapped->fd = open(filename.c_str(), O_RDONLY);
    if (mapped->fd < 0) {
        DEBUG_LOG("Failed to open file for memory mapping: " + filename);
        return nullptr;
    }
    
    struct stat st;
    if (fstat(mapped->fd, &st) < 0) {
        close(mapped->fd);
        return nullptr;
    }
    mapped->size = static_cast<size_t>(st.st_size);
    
    mapped->ptr = mmap(nullptr, mapped->size, PROT_READ, MAP_PRIVATE, mapped->fd, 0);
    if (mapped->ptr == MAP_FAILED) {
        close(mapped->fd);
        DEBUG_LOG("mmap failed for file: " + filename);
        return nullptr;
    }
    
    // Advise the kernel about our access pattern
    if (madvise(mapped->ptr, mapped->size, MADV_SEQUENTIAL | MADV_WILLNEED) < 0) {
        // Non-fatal, just log
        DEBUG_LOG("madvise failed for file: " + filename);
    }
#endif
    
    mapped->is_valid = true;
    mapped_files[filename] = mapped;
    
    DEBUG_LOG("Successfully memory mapped file: " + filename + 
              " (size: " + std::to_string(mapped->size) + " bytes)");
    
    return mapped;
}

void MemoryMapper::unmap_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    mapped_files.erase(filename);
    DEBUG_LOG("Unmapped file: " + filename);
}

bool MemoryMapper::should_use_memory_mapping(const std::string& filename, size_t file_size) {
    // Get file size if not provided
    if (file_size == 0) {
#ifdef _WIN32
        HANDLE hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        
        LARGE_INTEGER size;
        bool got_size = GetFileSizeEx(hFile, &size);
        CloseHandle(hFile);
        
        if (!got_size) return false;
        file_size = static_cast<size_t>(size.QuadPart);
#else
        struct stat st;
        if (stat(filename.c_str(), &st) < 0) return false;
        file_size = static_cast<size_t>(st.st_size);
#endif
    }
    
    // Only map files larger than minimum threshold
    if (file_size < min_file_size_for_mapping) {
        return false;
    }
    
    // Don't map if we're at the limit and this would be a new mapping
    std::lock_guard<std::mutex> lock(mapping_mutex);
    if (mapped_files.find(filename) == mapped_files.end() && 
        mapped_files.size() >= max_mapped_files) {
        return false;
    }
    
    return true;
}

void* MemoryMapper::get_mapped_pointer(const std::string& filename) {
    auto mapped = map_file(filename);
    return mapped ? mapped->ptr : nullptr;
}

size_t MemoryMapper::get_mapped_size(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    auto it = mapped_files.find(filename);
    return (it != mapped_files.end()) ? it->second->size : 0;
}

void MemoryMapper::cleanup_all_mappings() {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    mapped_files.clear();
    DEBUG_LOG("Cleaned up all memory mappings");
}

MemoryMapper::MappingStats MemoryMapper::get_stats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mapping_mutex));
    
    MappingStats stats = {};
    stats.num_mapped_files = mapped_files.size();
    
    for (const auto& pair : mapped_files) {
        stats.total_mapped_bytes += pair.second->size;
    }
    
    return stats;
}

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
        
        case TLONGLONG: {
            std::vector<LONGLONG> buffer(task.num_rows);
            fits_read_col(fptr, TLONGLONG, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading long long column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](LONGLONG val) { return static_cast<double>(val); });
            break;
        }
        
        case TINT: {
            std::vector<int> buffer(task.num_rows);
            fits_read_col(fptr, TINT, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading int column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](int val) { return static_cast<double>(val); });
            break;
        }
        
        case TSHORT: {
            std::vector<short> buffer(task.num_rows);
            fits_read_col(fptr, TSHORT, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading short column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](short val) { return static_cast<double>(val); });
            break;
        }
        
        case TBYTE: {
            std::vector<unsigned char> buffer(task.num_rows);
            fits_read_col(fptr, TBYTE, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading byte column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](unsigned char val) { return static_cast<double>(val); });
            break;
        }
        
        case TLOGICAL: {
            std::vector<char> buffer(task.num_rows);
            fits_read_col(fptr, TLOGICAL, task.number, task.start_row + 1, 1, task.num_rows,
                         nullptr, buffer.data(), nullptr, &status);
            if (status) throw_fits_error(status, "Error reading logical column: " + task.name);
            
            auto tensor_ptr = tensor.data_ptr<double>();
            std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                          [](char val) { return static_cast<double>(val == 'T' ? 1 : 0); });
            break;
        }
        
        case TSTRING: {
            // String columns cannot be converted to numeric tensors
            // They are handled by the main reader function
            throw std::runtime_error("String columns not supported in optimized numeric processing");
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
            
            case TLONGLONG: {
                std::vector<LONGLONG> buffer(num_rows);
                fits_read_col(fptr, TLONGLONG, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading long long column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](LONGLONG val) { return static_cast<double>(val); });
                break;
            }
            
            case TINT: {
                std::vector<int> buffer(num_rows);
                fits_read_col(fptr, TINT, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading int column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](int val) { return static_cast<double>(val); });
                break;
            }
            
            case TSHORT: {
                std::vector<short> buffer(num_rows);
                fits_read_col(fptr, TSHORT, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading short column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](short val) { return static_cast<double>(val); });
                break;
            }
            
            case TBYTE: {
                std::vector<unsigned char> buffer(num_rows);
                fits_read_col(fptr, TBYTE, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading byte column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](unsigned char val) { return static_cast<double>(val); });
                break;
            }
            
            case TLOGICAL: {
                std::vector<char> buffer(num_rows);
                fits_read_col(fptr, TLOGICAL, col_number, start_row + 1, 1, num_rows,
                             nullptr, buffer.data(), nullptr, &status);
                if (status) throw_fits_error(status, "Error reading logical column: " + col_name);
                
                auto tensor_ptr = col_data.data_ptr<double>();
                std::transform(buffer.begin(), buffer.end(), tensor_ptr,
                              [](char val) { return static_cast<double>(val == 'T' ? 1 : 0); });
                break;
            }
            
            case TSTRING: {
                // String columns cannot be converted to numeric tensors
                // Skip them - they will be handled by the main reader function
                continue;
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
    
    // Use duration to avoid unused variable warning
    (void)duration;
    
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

// === BufferedReader Implementation ===

torchfits_perf::BufferedReader::BufferedReader(const std::string& filename, const BufferedIOConfig& config)
    : filename_(filename), config_(config), file_size_(0) {
    
    // Get file size
    struct stat st;
    if (stat(filename.c_str(), &st) == 0) {
        file_size_ = st.st_size;
    } else {
        throw std::runtime_error("Cannot stat file: " + filename);
    }
    
    // Initialize buffer blocks (multiple buffers for better caching)
    size_t num_buffers = std::max(2UL, config_.max_concurrent_reads);
    buffers_.resize(num_buffers);
    
    for (auto& buffer : buffers_) {
        buffer.data.resize(config_.buffer_size);
        buffer.file_offset = 0;
        buffer.valid_size = 0;
        buffer.is_valid = false;
        buffer.last_accessed = std::chrono::steady_clock::now();
    }
    
    // Start prefetch worker thread if enabled
    if (config_.enable_async_prefetch) {
        prefetch_thread_ = std::thread(&BufferedReader::prefetch_worker, this);
    }
}

torchfits_perf::BufferedReader::~BufferedReader() {
    // Shutdown prefetch thread
    if (config_.enable_async_prefetch && prefetch_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            shutdown_prefetch_ = true;
        }
        prefetch_cv_.notify_all();
        prefetch_thread_.join();
    }
}

size_t torchfits_perf::BufferedReader::read(void* buffer, size_t size, size_t offset) {
    if (offset >= file_size_) {
        return 0;
    }
    
    size_t bytes_to_read = std::min(size, file_size_ - offset);
    size_t bytes_read = 0;
    uint8_t* output = static_cast<uint8_t*>(buffer);
    
    while (bytes_read < bytes_to_read) {
        size_t current_offset = offset + bytes_read;
        size_t remaining = bytes_to_read - bytes_read;
        
        // Try to find data in existing buffers
        BufferBlock* buffer_block = find_buffer_for_offset(current_offset);
        
        if (buffer_block && buffer_block->is_valid) {
            // Data found in buffer
            size_t block_start = buffer_block->file_offset;
            size_t block_end = block_start + buffer_block->valid_size;
            
            if (current_offset >= block_start && current_offset < block_end) {
                size_t buffer_offset = current_offset - block_start;
                size_t available_in_buffer = block_end - current_offset;
                size_t to_copy = std::min(remaining, available_in_buffer);
                
                std::memcpy(output + bytes_read, 
                           buffer_block->data.data() + buffer_offset, 
                           to_copy);
                
                bytes_read += to_copy;
                buffer_block->last_accessed = std::chrono::steady_clock::now();
                continue;
            }
        }
        
        // Data not in buffer, need to load it
        BufferBlock* available_buffer = get_available_buffer();
        if (available_buffer) {
            size_t load_size = std::min(config_.buffer_size, file_size_ - current_offset);
            load_buffer(available_buffer, current_offset, load_size);
            
            // Now copy from the loaded buffer
            size_t to_copy = std::min(remaining, available_buffer->valid_size);
            std::memcpy(output + bytes_read, available_buffer->data.data(), to_copy);
            bytes_read += to_copy;
            available_buffer->last_accessed = std::chrono::steady_clock::now();
        } else {
            // Fallback: direct read if no buffer available
            std::ifstream file(filename_, std::ios::binary);
            file.seekg(current_offset);
            file.read(reinterpret_cast<char*>(output + bytes_read), remaining);
            bytes_read += file.gcount();
            break;
        }
    }
    
    return bytes_read;
}

void torchfits_perf::BufferedReader::prefetch(size_t offset, size_t size) {
    if (!config_.enable_async_prefetch) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    prefetch_queue_.push({offset, size});
    prefetch_cv_.notify_one();
}

bool torchfits_perf::BufferedReader::is_buffered(size_t offset, size_t size) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    for (const auto& buffer : buffers_) {
        if (buffer.is_valid) {
            size_t block_start = buffer.file_offset;
            size_t block_end = block_start + buffer.valid_size;
            
            if (offset >= block_start && offset + size <= block_end) {
                return true;
            }
        }
    }
    return false;
}

void torchfits_perf::BufferedReader::prefetch_worker() {
    while (!shutdown_prefetch_) {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this] { return !prefetch_queue_.empty() || shutdown_prefetch_; });
        
        if (shutdown_prefetch_) {
            break;
        }
        
        auto [offset, size] = prefetch_queue_.front();
        prefetch_queue_.pop();
        lock.unlock();
        
        // Load data into available buffer
        BufferBlock* buffer = get_available_buffer();
        if (buffer) {
            size_t load_size = std::min(config_.read_ahead_size, file_size_ - offset);
            load_buffer(buffer, offset, load_size);
        }
    }
}

torchfits_perf::BufferedReader::BufferBlock* torchfits_perf::BufferedReader::find_buffer_for_offset(size_t offset) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    for (auto& buffer : buffers_) {
        if (buffer.is_valid) {
            size_t block_start = buffer.file_offset;
            size_t block_end = block_start + buffer.valid_size;
            
            if (offset >= block_start && offset < block_end) {
                return &buffer;
            }
        }
    }
    return nullptr;
}

torchfits_perf::BufferedReader::BufferBlock* torchfits_perf::BufferedReader::get_available_buffer() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // First try to find an invalid buffer
    for (auto& buffer : buffers_) {
        if (!buffer.is_valid) {
            return &buffer;
        }
    }
    
    // If all buffers are valid, evict the least recently used
    auto lru_it = std::min_element(buffers_.begin(), buffers_.end(),
        [](const BufferBlock& a, const BufferBlock& b) {
            return a.last_accessed < b.last_accessed;
        });
    
    return &(*lru_it);
}

void torchfits_perf::BufferedReader::load_buffer(BufferBlock* buffer, size_t offset, size_t size) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file) {
        buffer->is_valid = false;
        return;
    }
    
    file.seekg(offset);
    file.read(reinterpret_cast<char*>(buffer->data.data()), size);
    
    buffer->file_offset = offset;
    buffer->valid_size = file.gcount();
    buffer->is_valid = buffer->valid_size > 0;
    buffer->last_accessed = std::chrono::steady_clock::now();
}

// === ThreadPool Implementation ===

torchfits_perf::ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
            }
        });
    }
}

torchfits_perf::ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void torchfits_perf::ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty(); });
}

// === ParallelFITS Implementation ===

torchfits_perf::ThreadPool& torchfits_perf::ParallelFITS::get_thread_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}

torch::Tensor ParallelFITS::parallel_read_image(const std::string& filename, 
                                               int hdu_num,
                                               size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    // Get image dimensions first
    FITSFileWrapper fits_file(filename);
    int status = 0;
    if (fits_movabs_hdu(fits_file.get(), hdu_num, nullptr, &status)) {
        throw std::runtime_error("Error moving to HDU " + std::to_string(hdu_num));
    }
    
    std::vector<long> dims = get_image_dims(fits_file.get());
    if (dims.empty()) {
        throw std::runtime_error("Cannot read empty image");
    }
    
    // Calculate total elements
    size_t total_elements = 1;
    for (long dim : dims) {
        total_elements *= dim;
    }
    
    // Determine data type
    int bitpix;
    if (fits_get_img_type(fits_file.get(), &bitpix, &status)) {
        throw std::runtime_error("Error getting image type");
    }
    
    // Create output tensor based on FITS data type
    torch::Tensor result;
    torch::IntArrayRef tensor_dims(dims.data(), dims.size());
    
    switch (bitpix) {
        case FLOAT_IMG:
            result = torch::empty(tensor_dims, torch::dtype(torch::kFloat32));
            break;
        case DOUBLE_IMG:
            result = torch::empty(tensor_dims, torch::dtype(torch::kFloat64));
            break;
        case SHORT_IMG:
            result = torch::empty(tensor_dims, torch::dtype(torch::kInt16));
            break;
        case LONG_IMG:
            result = torch::empty(tensor_dims, torch::dtype(torch::kInt32));
            break;
        case LONGLONG_IMG:
            result = torch::empty(tensor_dims, torch::dtype(torch::kInt64));
            break;
        default:
            result = torch::empty(tensor_dims, torch::dtype(torch::kFloat32));
    }
    
    // Divide work among threads
    size_t chunk_size = total_elements / num_threads;
    if (chunk_size == 0) {
        chunk_size = total_elements;
        num_threads = 1;
    }
    
    auto& pool = get_thread_pool();
    std::vector<std::future<void>> futures;
    
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        size_t start_pixel = thread_id * chunk_size;
        size_t end_pixel = (thread_id == num_threads - 1) ? 
                          total_elements : (thread_id + 1) * chunk_size;
        
        if (start_pixel >= total_elements) break;
        
        auto future = pool.submit([&, start_pixel, end_pixel, bitpix, hdu_num]() {
            // Each thread opens its own FITS file handle
            FITSFileWrapper thread_fits(filename);
            int thread_status = 0;
            
            if (fits_movabs_hdu(thread_fits.get(), hdu_num, nullptr, &thread_status)) {
                throw std::runtime_error("Thread error moving to HDU");
            }
            
            // Read chunk data
            long nulval = 0;
            int anynul = 0;
            long first_pixel = start_pixel + 1; // FITS is 1-indexed
            long num_pixels = end_pixel - start_pixel;
            
            // Get pointer to result tensor data
            void* data_ptr = static_cast<char*>(result.data_ptr()) + 
                           start_pixel * result.element_size();
            
            switch (bitpix) {
                case FLOAT_IMG:
                    fits_read_pix(thread_fits.get(), TFLOAT, &first_pixel, num_pixels,
                                &nulval, data_ptr, &anynul, &thread_status);
                    break;
                case DOUBLE_IMG:
                    fits_read_pix(thread_fits.get(), TDOUBLE, &first_pixel, num_pixels,
                                &nulval, data_ptr, &anynul, &thread_status);
                    break;
                case SHORT_IMG:
                    fits_read_pix(thread_fits.get(), TSHORT, &first_pixel, num_pixels,
                                &nulval, data_ptr, &anynul, &thread_status);
                    break;
                case LONG_IMG:
                    fits_read_pix(thread_fits.get(), TINT, &first_pixel, num_pixels,
                                &nulval, data_ptr, &anynul, &thread_status);
                    break;
                case LONGLONG_IMG:
                    fits_read_pix(thread_fits.get(), TLONGLONG, &first_pixel, num_pixels,
                                &nulval, data_ptr, &anynul, &thread_status);
                    break;
            }
            
            if (thread_status) {
                throw std::runtime_error("Error reading pixel data in thread");
            }
        });
        
        futures.push_back(std::move(future));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    
    return result;
}

ParallelFITS::Statistics ParallelFITS::parallel_compute_stats(const std::string& filename,
                                                             int hdu_num,
                                                             size_t num_threads) {
    // Read data in parallel
    torch::Tensor data = parallel_read_image(filename, hdu_num, num_threads);
    
    // Use PyTorch's optimized statistics functions
    Statistics stats;
    stats.mean = data.mean().item<double>();
    stats.std = data.std().item<double>();
    stats.min = data.min().item<double>();
    stats.max = data.max().item<double>();
    stats.count = data.numel();
    
    return stats;
}

// === GPU Pipeline Implementation ===

std::unique_ptr<GPUPipeline::GPUConfig> GPUPipeline::config_ = nullptr;
std::unique_ptr<GPUPipeline::GPUMemoryPool> GPUPipeline::gpu_pool_ = nullptr;
bool GPUPipeline::initialized_ = false;

void GPUPipeline::initialize(const GPUConfig& config) {
    if (initialized_) return;
    
    config_ = std::make_unique<GPUConfig>(config);
    
    // Check if CUDA is available
    if (config.enable_gpu && torch::cuda::is_available()) {
        try {
            // Initialize GPU memory pool
            gpu_pool_ = std::make_unique<GPUMemoryPool>(
                config.gpu_memory_pool_size, config.device);
            
            DEBUG_PRINT("GPU Pipeline initialized with device: " << config.device);
            initialized_ = true;
            
        } catch (const std::exception& e) {
            DEBUG_PRINT("Failed to initialize GPU pipeline: " << e.what());
            config_->enable_gpu = false;
        }
    } else {
        DEBUG_PRINT("GPU not available, using CPU fallback");
        config_->enable_gpu = false;
    }
    
    initialized_ = true;
}

bool GPUPipeline::is_available() {
    return initialized_ && config_ && config_->enable_gpu && torch::cuda::is_available();
}

torch::Tensor GPUPipeline::read_to_gpu(const std::string& filename, 
                                      int hdu_num,
                                      const torch::Device& device) {
    if (!is_available()) {
        // Fallback to CPU read then move to GPU
        torch::Tensor cpu_data = ParallelFITS::parallel_read_image(filename, hdu_num);
        return cpu_data.to(device);
    }
    
    try {
        // Read to pinned CPU memory first for faster GPU transfer
        torch::Tensor cpu_data = ParallelFITS::parallel_read_image(filename, hdu_num);
        
        if (config_->use_pinned_memory) {
            // Pin memory for faster GPU transfer
            cpu_data = cpu_data.pin_memory();
        }
        
        // Async copy to GPU if enabled
        torch::Tensor gpu_data;
        if (config_->enable_async_copy) {
            gpu_data = cpu_data.to(device, /*non_blocking=*/true);
        } else {
            gpu_data = cpu_data.to(device);
        }
        
        return gpu_data;
        
    } catch (const std::exception& e) {
        DEBUG_PRINT("GPU read failed, falling back to CPU: " << e.what());
        torch::Tensor cpu_data = ParallelFITS::parallel_read_image(filename, hdu_num);
        return cpu_data.to(device);
    }
}

// GPUStreamer implementation
GPUPipeline::GPUStreamer::GPUStreamer(const std::string& filename, 
                                     int hdu_num,
                                     size_t chunk_size,
                                     const torch::Device& device)
    : filename_(filename), hdu_num_(hdu_num), chunk_size_(chunk_size), 
      device_(device), current_position_(0), total_elements_(0) {
    
    // Get total data size
    FITSFileWrapper fits_file(filename);
    int status = 0;
    if (fits_movabs_hdu(fits_file.get(), hdu_num, nullptr, &status)) {
        throw std::runtime_error("Error moving to HDU");
    }
    
    std::vector<long> dims = get_image_dims(fits_file.get());
    total_elements_ = 1;
    for (long dim : dims) {
        total_elements_ *= dim;
    }
    
    // Start prefetch worker
    if (GPUPipeline::is_available()) {
        prefetch_thread_ = std::thread(&GPUStreamer::prefetch_worker, this);
    }
}

GPUPipeline::GPUStreamer::~GPUStreamer() {
    if (prefetch_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_ = true;
        }
        queue_cv_.notify_all();
        prefetch_thread_.join();
    }
}

bool GPUPipeline::GPUStreamer::has_next() const {
    return current_position_ < total_elements_;
}

torch::Tensor GPUPipeline::GPUStreamer::next_chunk() {
    if (!has_next()) {
        throw std::runtime_error("No more chunks available");
    }
    
    // Try to get from prefetch queue first
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!gpu_buffer_queue_.empty()) {
            torch::Tensor chunk = gpu_buffer_queue_.front();
            gpu_buffer_queue_.pop();
            current_position_ += chunk.numel();
            return chunk;
        }
    }
    
    // Fallback: read directly
    size_t remaining = total_elements_ - current_position_;
    size_t elements_to_read = std::min(chunk_size_, remaining);
    
    // This is a simplified implementation - in practice, you'd need
    // to implement chunk-based FITS reading
    torch::Tensor chunk = torch::empty({static_cast<long>(elements_to_read)}, 
                                      torch::dtype(torch::kFloat32));
    
    current_position_ += elements_to_read;
    
    if (GPUPipeline::is_available()) {
        return chunk.to(device_);
    }
    return chunk;
}

void GPUPipeline::GPUStreamer::prefetch_worker() {
    // Simplified prefetch worker - would need full implementation
    // for production use
    while (!shutdown_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
        
        if (shutdown_) break;
        
        // Prefetch logic here
        // This would read ahead and populate gpu_buffer_queue_
    }
}

// GPUMemoryPool implementation
GPUPipeline::GPUMemoryPool::GPUMemoryPool(size_t pool_size, const torch::Device& device)
    : device_(device), pool_size_(pool_size), used_memory_(0) {
    
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA not available for GPU memory pool");
    }
    
    DEBUG_PRINT("Initialized GPU memory pool with " << pool_size / (1024*1024) << " MB");
}

GPUPipeline::GPUMemoryPool::~GPUMemoryPool() {
    cleanup();
}

torch::Tensor GPUPipeline::GPUMemoryPool::allocate(const torch::IntArrayRef& sizes, 
                                                  torch::ScalarType dtype) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Calculate required memory
    size_t element_size = torch::elementSize(dtype);
    size_t total_elements = 1;
    for (auto size : sizes) {
        total_elements *= size;
    }
    size_t required_memory = total_elements * element_size;
    
    // Check if we have enough space
    if (used_memory_ + required_memory > pool_size_) {
        throw std::runtime_error("GPU memory pool exhausted");
    }
    
    // Try to reuse existing tensor
    for (auto it = free_tensors_.begin(); it != free_tensors_.end(); ++it) {
        if (it->numel() * it->element_size() >= required_memory &&
            it->scalar_type() == dtype) {
            torch::Tensor tensor = *it;
            free_tensors_.erase(it);
            return tensor.view(sizes);
        }
    }
    
    // Allocate new tensor
    torch::Tensor tensor = torch::empty(sizes, 
        torch::TensorOptions().dtype(dtype).device(device_));
    
    used_memory_ += required_memory;
    return tensor;
}

void GPUPipeline::GPUMemoryPool::deallocate(torch::Tensor& tensor) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (tensor.device() == device_) {
        free_tensors_.push_back(tensor);
    }
    
    tensor = torch::Tensor(); // Clear reference
}

void GPUPipeline::GPUMemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    free_tensors_.clear();
    used_memory_ = 0;
}

size_t GPUPipeline::GPUMemoryPool::get_available_memory() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return pool_size_ - used_memory_;
}
