#include "performance_v2.h"
#include "fits_utils.h"
#include "debug.h"

// Platform-specific includes for memory mapping
#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// Platform-specific SIMD includes
#ifdef __x86_64__
    #include <immintrin.h>  // x86 SIMD intrinsics
    #include <cpuid.h>
#elif defined(__aarch64__) || defined(__arm64__)
    #include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include <algorithm>
#include <chrono>
#include <iostream>

namespace torchfits_perf {

// Global instances
std::unique_ptr<MemoryMapper> global_memory_mapper;
std::unique_ptr<BufferManager> global_buffer_manager;
std::unique_ptr<TensorMemoryPool> global_memory_pool;
std::unique_ptr<PerformanceMonitor> global_performance_monitor;

// Static member initialization
bool SIMDOptimizer::avx2_available = false;
bool SIMDOptimizer::neon_available = false;
bool SIMDOptimizer::initialized = false;
std::unordered_map<torch::Device, std::vector<void*>> GPUTensorAllocator::stream_pool;
std::mutex GPUTensorAllocator::stream_mutex;

// --- MemoryMapper Implementation ---

std::shared_ptr<MemoryMapper::MappedFile> MemoryMapper::map_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    
    auto it = mapped_files.find(filename);
    if (it != mapped_files.end() && it->second->is_valid) {
        return it->second;
    }
    
    auto mapped_file = std::make_shared<MappedFile>();
    mapped_file->filename = filename;
    mapped_file->is_valid = false;
    
#ifdef _WIN32
    // Windows memory mapping
    HANDLE file_handle = CreateFileA(filename.c_str(), GENERIC_READ, 
                                   FILE_SHARE_READ, NULL, OPEN_EXISTING, 
                                   FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("Failed to open file for memory mapping: " + filename);
        return mapped_file;
    }
    
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        CloseHandle(file_handle);
        DEBUG_LOG("Failed to get file size for memory mapping: " + filename);
        return mapped_file;
    }
    
    HANDLE mapping_handle = CreateFileMapping(file_handle, NULL, PAGE_READONLY, 
                                            file_size.HighPart, file_size.LowPart, NULL);
    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        DEBUG_LOG("Failed to create file mapping: " + filename);
        return mapped_file;
    }
    
    void* ptr = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, file_size.QuadPart);
    if (ptr == NULL) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        DEBUG_LOG("Failed to map view of file: " + filename);
        return mapped_file;
    }
    
    mapped_file->ptr = ptr;
    mapped_file->size = file_size.QuadPart;
    mapped_file->is_valid = true;
    
    CloseHandle(mapping_handle);
    CloseHandle(file_handle);
    
#else
    // Unix/Linux memory mapping
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        DEBUG_LOG("Failed to open file for memory mapping: " + filename);
        return mapped_file;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        DEBUG_LOG("Failed to get file stats for memory mapping: " + filename);
        return mapped_file;
    }
    
    void* ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        DEBUG_LOG("Failed to memory map file: " + filename);
        return mapped_file;
    }
    
    // Advise kernel about access pattern
    madvise(ptr, sb.st_size, MADV_RANDOM);
    
    mapped_file->ptr = ptr;
    mapped_file->size = sb.st_size;
    mapped_file->is_valid = true;
    
    close(fd);
#endif
    
    mapped_files[filename] = mapped_file;
    DEBUG_LOG("Successfully memory mapped file: " + filename + 
              " (size: " + std::to_string(mapped_file->size) + " bytes)");
    
    return mapped_file;
}

void MemoryMapper::unmap_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    
    auto it = mapped_files.find(filename);
    if (it == mapped_files.end()) {
        return;
    }
    
    auto mapped_file = it->second;
    if (mapped_file->is_valid && mapped_file->ptr) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_file->ptr);
#else
        munmap(mapped_file->ptr, mapped_file->size);
#endif
        mapped_file->is_valid = false;
        DEBUG_LOG("Unmapped file: " + filename);
    }
    
    mapped_files.erase(it);
}

bool MemoryMapper::should_use_memory_mapping(const std::string& filename, size_t file_size) {
    // Use memory mapping for files larger than 10MB
    const size_t MIN_SIZE_FOR_MAPPING = 10 * 1024 * 1024;
    
    // Don't use memory mapping for compressed files (for now)
    if (filename.find(".gz") != std::string::npos ||
        filename.find(".bz2") != std::string::npos) {
        return false;
    }
    
    return file_size > MIN_SIZE_FOR_MAPPING;
}

void* MemoryMapper::get_mapped_pointer(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    
    auto it = mapped_files.find(filename);
    if (it != mapped_files.end() && it->second->is_valid) {
        return it->second->ptr;
    }
    
    return nullptr;
}

void MemoryMapper::cleanup_all_mappings() {
    std::lock_guard<std::mutex> lock(mapping_mutex);
    
    for (auto& pair : mapped_files) {
        auto mapped_file = pair.second;
        if (mapped_file->is_valid && mapped_file->ptr) {
#ifdef _WIN32
            UnmapViewOfFile(mapped_file->ptr);
#else
            munmap(mapped_file->ptr, mapped_file->size);
#endif
            mapped_file->is_valid = false;
        }
    }
    
    mapped_files.clear();
    DEBUG_LOG("Cleaned up all memory mappings");
}

// --- BufferManager Implementation ---

BufferManager::BufferManager(size_t max_size) : max_buffer_size(max_size) {
    DEBUG_LOG("BufferManager initialized with max size: " + std::to_string(max_size));
}

char* BufferManager::get_buffer(size_t required_size, const std::string& file_type) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    // Find suitable buffer
    for (auto& buffer_info : buffers) {
        if (!buffer_info.in_use && buffer_info.size >= required_size) {
            buffer_info.in_use = true;
            buffer_info.last_used = std::chrono::steady_clock::now();
            return buffer_info.buffer.get();
        }
    }
    
    // Create new buffer if needed
    if (required_size <= max_buffer_size) {
        BufferInfo new_buffer;
        new_buffer.buffer = std::make_unique<char[]>(required_size);
        new_buffer.size = required_size;
        new_buffer.in_use = true;
        new_buffer.last_used = std::chrono::steady_clock::now();
        
        char* ptr = new_buffer.buffer.get();
        buffers.push_back(std::move(new_buffer));
        
        DEBUG_LOG("Created new buffer of size: " + std::to_string(required_size));
        return ptr;
    }
    
    // For very large requests, allocate directly
    return new char[required_size];
}

void BufferManager::return_buffer(char* buffer) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    for (auto& buffer_info : buffers) {
        if (buffer_info.buffer.get() == buffer) {
            buffer_info.in_use = false;
            buffer_info.last_used = std::chrono::steady_clock::now();
            return;
        }
    }
    
    // If not found in pool, it was allocated directly
    delete[] buffer;
}

size_t BufferManager::get_optimal_buffer_size(const std::string& filename, 
                                            const std::string& operation_type,
                                            size_t data_size) {
    // Default buffer size
    size_t base_size = 1024 * 1024; // 1MB
    
    // Adjust based on operation type
    if (operation_type == "table_read") {
        base_size = std::min(data_size, 4 * 1024 * 1024UL); // Up to 4MB for tables
    } else if (operation_type == "image_read") {
        base_size = std::min(data_size, 8 * 1024 * 1024UL); // Up to 8MB for images
    } else if (operation_type == "cutout") {
        base_size = std::min(data_size * 2, 2 * 1024 * 1024UL); // Up to 2MB for cutouts
    }
    
    // Ensure it's within limits
    return std::min(base_size, max_buffer_size);
}

void BufferManager::cleanup_old_buffers() {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    auto now = std::chrono::steady_clock::now();
    auto threshold = std::chrono::minutes(5); // 5 minutes
    
    auto it = buffers.begin();
    while (it != buffers.end()) {
        if (!it->in_use && (now - it->last_used) > threshold) {
            DEBUG_LOG("Cleaned up unused buffer of size: " + std::to_string(it->size));
            it = buffers.erase(it);
        } else {
            ++it;
        }
    }
}

// --- SIMDOptimizer Implementation ---

void SIMDOptimizer::initialize() {
    if (initialized) return;
    
    detect_capabilities();
    initialized = true;
    
    DEBUG_LOG("SIMD capabilities - AVX2: " + std::string(avx2_available ? "yes" : "no") +
              ", NEON: " + std::string(neon_available ? "yes" : "no"));
}

void SIMDOptimizer::detect_capabilities() {
#ifdef __x86_64__
    // Check for AVX2 support
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        avx2_available = (ebx & (1 << 5)) != 0;
    }
#elif defined(__aarch64__) || defined(__arm64__)
    // ARM NEON is standard on ARM64
    neon_available = true;
#endif
}

void SIMDOptimizer::convert_float_array_simd(const float* src, float* dst, size_t count) {
    if (!simd_available()) {
        std::memcpy(dst, src, count * sizeof(float));
        return;
    }
    
#ifdef __x86_64__
    if (avx2_available) {
        size_t simd_count = count & ~7; // Process 8 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 8) {
            __m256 data = _mm256_loadu_ps(&src[i]);
            _mm256_storeu_ps(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
        return;
    }
#elif defined(__aarch64__) || defined(__arm64__)
    if (neon_available) {
        size_t simd_count = count & ~3; // Process 4 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 4) {
            float32x4_t data = vld1q_f32(&src[i]);
            vst1q_f32(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
        return;
    }
#endif
    
    // Fallback
    std::memcpy(dst, src, count * sizeof(float));
}

void SIMDOptimizer::convert_double_array_simd(const double* src, double* dst, size_t count) {
    if (!simd_available()) {
        std::memcpy(dst, src, count * sizeof(double));
        return;
    }
    
#ifdef __x86_64__
    if (avx2_available) {
        size_t simd_count = count & ~3; // Process 4 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 4) {
            __m256d data = _mm256_loadu_pd(&src[i]);
            _mm256_storeu_pd(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
        return;
    }
#elif defined(__aarch64__) || defined(__arm64__)
    if (neon_available) {
        size_t simd_count = count & ~1; // Process 2 elements at a time
        
        for (size_t i = 0; i < simd_count; i += 2) {
            float64x2_t data = vld1q_f64(&src[i]);
            vst1q_f64(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dst[i] = src[i];
        }
        return;
    }
#endif
    
    // Fallback
    std::memcpy(dst, src, count * sizeof(double));
}

bool SIMDOptimizer::simd_available() {
    if (!initialized) initialize();
    return avx2_available || neon_available;
}

std::vector<std::string> SIMDOptimizer::get_available_simd() {
    if (!initialized) initialize();
    
    std::vector<std::string> available;
    if (avx2_available) available.push_back("AVX2");
    if (neon_available) available.push_back("NEON");
    
    return available;
}

// --- Global initialization functions ---

void initialize_performance_optimizations() {
    DEBUG_LOG("Initializing TorchFits v1.0 performance optimizations");
    
    global_memory_mapper = std::make_unique<MemoryMapper>();
    global_buffer_manager = std::make_unique<BufferManager>();
    global_memory_pool = std::make_unique<TensorMemoryPool>();
    global_performance_monitor = std::make_unique<PerformanceMonitor>();
    
    SIMDOptimizer::initialize();
    
    DEBUG_LOG("Performance optimizations initialized successfully");
}

void cleanup_performance_optimizations() {
    DEBUG_LOG("Cleaning up performance optimizations");
    
    if (global_memory_mapper) {
        global_memory_mapper->cleanup_all_mappings();
        global_memory_mapper.reset();
    }
    
    if (global_buffer_manager) {
        global_buffer_manager->cleanup_old_buffers();
        global_buffer_manager.reset();
    }
    
    if (global_memory_pool) {
        global_memory_pool->clear();
        global_memory_pool.reset();
    }
    
    global_performance_monitor.reset();
    
    DEBUG_LOG("Performance optimizations cleaned up");
}

py::dict get_optimization_status() {
    py::dict status;
    
    status["memory_mapping"] = global_memory_mapper != nullptr;
    status["buffer_management"] = global_buffer_manager != nullptr;
    status["tensor_pooling"] = global_memory_pool != nullptr;
    status["simd_available"] = SIMDOptimizer::simd_available();
    status["simd_instructions"] = SIMDOptimizer::get_available_simd();
    
    if (global_performance_monitor) {
        status["statistics"] = global_performance_monitor->get_statistics();
    }
    
    return status;
}

} // namespace torchfits_perf

// --- Additional implementations ---

namespace torchfits_perf {

// --- TensorMemoryPool Implementation ---

TensorMemoryPool::TensorMemoryPool(size_t max_size, size_t max_pinned_mb) 
    : max_pool_size(max_size), current_pinned_memory(0), 
      max_pinned_memory(max_pinned_mb * 1024 * 1024) {
    DEBUG_LOG("TensorMemoryPool initialized - max_size: " + std::to_string(max_size) + 
              ", max_pinned_mb: " + std::to_string(max_pinned_mb));
}

torch::Tensor TensorMemoryPool::get_tensor(const std::vector<int64_t>& shape, 
                                         torch::Dtype dtype, 
                                         torch::Device device,
                                         bool pinned) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    PoolKey key{shape, dtype, device, pinned};
    
    auto it = pools.find(key);
    if (it != pools.end() && !it->second.empty()) {
        torch::Tensor tensor = it->second.front();
        it->second.pop();
        return tensor;
    }
    
    // Create new tensor
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(device);
    if (pinned && device.is_cpu()) {
        options = options.pinned_memory(true);
        
        // Check pinned memory limit
        size_t tensor_size = 1;
        for (auto dim : shape) tensor_size *= dim;
        tensor_size *= torch::elementSize(dtype);
        
        if (current_pinned_memory + tensor_size > max_pinned_memory) {
            // Don't use pinned memory if we'd exceed the limit
            options = options.pinned_memory(false);
        } else {
            current_pinned_memory += tensor_size;
        }
    }
    
    return torch::empty(shape, options);
}

void TensorMemoryPool::return_tensor(torch::Tensor tensor) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    PoolKey key{tensor.sizes().vec(), tensor.dtype(), tensor.device(), tensor.is_pinned()};
    
    auto& queue = pools[key];
    if (queue.size() < max_pool_size) {
        queue.push(tensor);
    } else {
        // Pool is full, let tensor be destroyed
        if (tensor.is_pinned()) {
            size_t tensor_size = tensor.numel() * torch::elementSize(tensor.dtype());
            current_pinned_memory = std::max(current_pinned_memory - tensor_size, size_t(0));
        }
    }
}

void TensorMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    pools.clear();
    current_pinned_memory = 0;
    DEBUG_LOG("TensorMemoryPool cleared");
}

// --- GPUTensorAllocator Implementation ---

torch::Tensor GPUTensorAllocator::allocate_gpu_tensor_direct(const std::vector<int64_t>& shape, 
                                                           torch::Dtype dtype,
                                                           torch::Device device,
                                                           void* cuda_stream) {
    if (!device.is_cuda()) {
        return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
    }
    
    // Use global memory pool if available
    if (global_memory_pool) {
        return global_memory_pool->get_tensor(shape, dtype, device);
    }
    
    return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
}

torch::Tensor GPUTensorAllocator::allocate_pinned_tensor(const std::vector<int64_t>& shape,
                                                       torch::Dtype dtype) {
    if (global_memory_pool) {
        return global_memory_pool->get_tensor(shape, dtype, torch::kCPU, true);
    }
    
    return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
}

bool GPUTensorAllocator::supports_direct_allocation(torch::Device device) {
    return device.is_cuda() && torch::cuda::is_available();
}

void* GPUTensorAllocator::get_cuda_stream(torch::Device device) {
    if (!device.is_cuda()) return nullptr;
    
    std::lock_guard<std::mutex> lock(stream_mutex);
    
    auto& streams = stream_pool[device];
    if (!streams.empty()) {
        void* stream = streams.back();
        streams.pop_back();
        return stream;
    }
    
    // Create new stream (implementation would depend on CUDA availability)
    return nullptr; // Simplified for now
}

void GPUTensorAllocator::return_cuda_stream(torch::Device device, void* stream) {
    if (!device.is_cuda() || !stream) return;
    
    std::lock_guard<std::mutex> lock(stream_mutex);
    stream_pool[device].push_back(stream);
}

std::future<torch::Tensor> GPUTensorAllocator::copy_to_gpu_async(torch::Tensor cpu_tensor,
                                                               torch::Device gpu_device,
                                                               void* cuda_stream) {
    return std::async(std::launch::async, [cpu_tensor, gpu_device]() {
        return cpu_tensor.to(gpu_device, /*non_blocking=*/true);
    });
}

// --- PerformanceMonitor Implementation ---

void PerformanceMonitor::record_operation(const std::string& operation, double time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex);
    
    auto& stat = stats[operation];
    stat.operation_type = operation;
    stat.total_time_ms += time_ms;
    stat.total_operations++;
    
    if (stat.total_operations == 1) {
        stat.min_time_ms = time_ms;
        stat.max_time_ms = time_ms;
    } else {
        stat.min_time_ms = std::min(stat.min_time_ms, time_ms);
        stat.max_time_ms = std::max(stat.max_time_ms, time_ms);
    }
}

py::dict PerformanceMonitor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    
    py::dict result;
    
    for (const auto& pair : stats) {
        const auto& stat = pair.second;
        py::dict op_stats;
        
        op_stats["total_operations"] = stat.total_operations;
        op_stats["total_time_ms"] = stat.total_time_ms;
        op_stats["average_time_ms"] = stat.total_time_ms / stat.total_operations;
        op_stats["min_time_ms"] = stat.min_time_ms;
        op_stats["max_time_ms"] = stat.max_time_ms;
        
        result[pair.first] = op_stats;
    }
    
    return result;
}

std::vector<std::string> PerformanceMonitor::get_recommendations() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    
    std::vector<std::string> recommendations;
    
    for (const auto& pair : stats) {
        const auto& stat = pair.second;
        double avg_time = stat.total_time_ms / stat.total_operations;
        
        if (avg_time > 100.0 && pair.first.find("table") != std::string::npos) {
            recommendations.push_back("Consider using parallel column reading for table operations");
        }
        
        if (avg_time > 50.0 && pair.first.find("image") != std::string::npos) {
            recommendations.push_back("Consider using memory mapping for large image files");
        }
    }
    
    return recommendations;
}

void PerformanceMonitor::reset() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    stats.clear();
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
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            try {
                if (task->use_iterator) {
                    result->data = read_column_with_iterator(fptr, *task);
                } else {
                    result->data = read_column_optimized(fptr, *task);
                }
                result->success = true;
            } catch (const std::exception& e) {
                result->success = false;
                result->error_message = e.what();
                DEBUG_LOG("Error reading column " + task->name + ": " + e.what());
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result->read_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
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
    
    // Get PyTorch dtype from CFITSIO typecode
    torch::Dtype torch_dtype;
    switch (task.typecode) {
        case TBYTE:     torch_dtype = torch::kUInt8; break;
        case TSHORT:    torch_dtype = torch::kInt16; break;
        case TINT:      torch_dtype = torch::kInt32; break;
        case TLONGLONG: torch_dtype = torch::kInt64; break;
        case TFLOAT:    torch_dtype = torch::kFloat32; break;
        case TDOUBLE:   torch_dtype = torch::kFloat64; break;
        default:        torch_dtype = torch::kFloat32; break;
    }
    
    // Create tensor with appropriate shape
    std::vector<int64_t> tensor_shape = {task.num_rows};
    if (task.repeat > 1) tensor_shape.push_back(task.repeat);
    
    torch::Tensor column_data;
    if (global_memory_pool) {
        column_data = global_memory_pool->get_tensor(tensor_shape, torch_dtype, task.device);
    } else {
        column_data = torch::empty(tensor_shape, torch::TensorOptions().dtype(torch_dtype).device(task.device));
    }
    
    // Read column data
    fits_read_col(fptr, task.typecode, task.number, task.start_row + 1, 1, task.num_rows,
                 nullptr, column_data.data_ptr(), nullptr, &status);
    
    if (status) {
        throw std::runtime_error("Error reading column " + task.name + ": CFITSIO error " + std::to_string(status));
    }
    
    return column_data;
}

torch::Tensor ParallelTableReader::read_column_with_iterator(fitsfile* fptr, const ColumnTask& task) {
    // For now, fall back to optimized reading
    // In future versions, this would use fits_iterate_data for better performance
    return read_column_optimized(fptr, task);
}

py::dict ParallelTableReader::read_columns_parallel(fitsfile* fptr, 
                                                  const std::vector<std::string>& columns,
                                                  long start_row,
                                                  long num_rows,
                                                  torch::Device device) {
    py::dict result;
    
    // Start worker threads
    start_workers(fptr);
    
    // Create tasks for each column
    int status = 0;
    for (const std::string& col_name : columns) {
        int col_num;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), &col_num, &status);
        if (status) {
            status = 0; // Reset and skip this column
            continue;
        }
        
        int typecode;
        long repeat, width;
        fits_get_coltype(fptr, col_num, &typecode, &repeat, &width, &status);
        if (status) {
            status = 0; // Reset and skip this column
            continue;
        }
        
        auto task = std::make_shared<ColumnTask>();
        task->name = col_name;
        task->number = col_num;
        task->typecode = typecode;
        task->repeat = repeat;
        task->width = width;
        task->start_row = start_row;
        task->num_rows = num_rows;
        task->device = device;
        task->use_iterator = num_rows > 10000; // Use iterator for large datasets
        
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            task_queue.push(task);
        }
        task_cv.notify_one();
    }
    
    // Collect results
    size_t expected_results = columns.size();
    size_t collected_results = 0;
    
    while (collected_results < expected_results) {
        std::shared_ptr<ColumnResult> column_result;
        
        {
            std::unique_lock<std::mutex> lock(result_mutex);
            result_cv.wait(lock, [this] { return !result_queue.empty(); });
            
            if (!result_queue.empty()) {
                column_result = result_queue.front();
                result_queue.pop();
            }
        }
        
        if (column_result) {
            if (column_result->success) {
                result[py::str(column_result->name)] = column_result->data;
            } else {
                DEBUG_LOG("Failed to read column " + column_result->name + ": " + column_result->error_message);
            }
            collected_results++;
        }
    }
    
    // Stop worker threads
    stop_workers_and_wait();
    
    return result;
}

} // namespace torchfits_perf
