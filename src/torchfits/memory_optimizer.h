#ifndef TORCHFITS_MEMORY_OPTIMIZER_H
#define TORCHFITS_MEMORY_OPTIMIZER_H

#include <torch/extension.h>
#include <fitsio.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstring>
#include "debug.h"

namespace torchfits_mem {

// Memory alignment constants
constexpr size_t CACHE_LINE_SIZE = 64;  // Most modern CPUs
constexpr size_t SIMD_ALIGNMENT = 32;   // AVX2 alignment
constexpr size_t FITS_ALIGNMENT = 8;    // FITS data alignment

/**
 * Memory-aligned tensor factory for optimal FITS data loading
 * 
 * This class creates PyTorch tensors with memory layouts optimized for:
 * 1. FITS data alignment (matching CFITSIO expectations)
 * 2. SIMD operations (AVX2/AVX-512 compatibility)
 * 3. Cache line alignment (reduced cache misses)
 * 4. Zero-copy operations where possible
 */
class AlignedTensorFactory {
public:
    /**
     * Create a memory-aligned tensor optimized for FITS data
     * 
     * @param shape Tensor dimensions
     * @param dtype PyTorch data type
     * @param device Target device (CPU/GPU)
     * @param fits_compatible Align for direct CFITSIO operations
     * @return Optimally aligned tensor
     */
    static torch::Tensor create_aligned_tensor(
        const std::vector<int64_t>& shape,
        torch::Dtype dtype,
        torch::Device device = torch::kCPU,
        bool fits_compatible = true
    );

    /**
     * Create tensor with pre-allocated aligned memory buffer
     * Useful for zero-copy FITS reading operations
     */
    static torch::Tensor create_from_aligned_buffer(
        void* aligned_buffer,
        const std::vector<int64_t>& shape,
        torch::Dtype dtype,
        torch::Device device = torch::kCPU
    );

    /**
     * Check if tensor is optimally aligned for FITS operations
     */
    static bool is_optimally_aligned(const torch::Tensor& tensor);

    /**
     * Get optimal alignment for given data type and operation
     */
    static size_t get_optimal_alignment(torch::Dtype dtype, bool fits_compatible = true);

private:
    // Internal aligned memory allocator
    static void* allocate_aligned_memory(size_t size, size_t alignment);
    static void deallocate_aligned_memory(void* ptr);

public:
    // Make memory functions accessible for internal use
    static void* public_allocate_aligned_memory(size_t size, size_t alignment) {
        return allocate_aligned_memory(size, alignment);
    }
    static void public_deallocate_aligned_memory(void* ptr) {
        deallocate_aligned_memory(ptr);
    }
};

/**
 * Optimized table reading with memory-aligned tensor creation
 * Implements bulk binary reading strategy inspired by fitsio's fits_read_tblbytes
 */
class OptimizedTableReader {
public:
    struct ColumnMetadata {
        std::string name;
        int fits_type;
        int fits_column_num;
        torch::Dtype torch_dtype;
        long repeat_count;
        long byte_width;
        size_t byte_offset;     // Offset within row
        bool is_string;
        bool is_variable_length;
    };

    /**
     * Read table data with optimal memory alignment and bulk operations
     * 
     * @param fptr CFITSIO file pointer
     * @param columns Column names to read (empty = all columns)
     * @param start_row Starting row (0-indexed)
     * @param num_rows Number of rows to read
     * @param device Target device
     * @return Dictionary of column_name -> aligned tensor
     */
    static pybind11::dict read_table_optimized(
        fitsfile* fptr,
        const std::vector<std::string>& columns,
        long start_row,
        long num_rows,
        torch::Device device = torch::kCPU
    );

private:
    // Analyze table structure for optimal reading strategy
    static std::vector<ColumnMetadata> analyze_table_structure(
        fitsfile* fptr,
        const std::vector<std::string>& requested_columns
    );

    // Bulk binary reading using fits_read_tblbytes approach
    // Deleter for aligned memory
    struct AlignedDeleter {
        void operator()(uint8_t* ptr) {
            AlignedTensorFactory::public_deallocate_aligned_memory(static_cast<void*>(ptr));
        }
    };

    // Bulk binary table reading with aligned memory management
    static std::unique_ptr<uint8_t[], AlignedDeleter> read_table_bulk_binary(
        fitsfile* fptr,
        long start_row,
        long num_rows,
        long row_length
    );

    // Parse binary data into aligned tensors
    static pybind11::dict parse_binary_to_tensors(
        const uint8_t* binary_data,
        const std::vector<ColumnMetadata>& columns,
        long num_rows,
        torch::Device device
    );

    // Direct memory copying for compatible layouts
    static torch::Tensor create_tensor_from_binary(
        const uint8_t* data,
        const ColumnMetadata& column,
        long num_rows,
        torch::Device device
    );
};

/**
 * Memory pool for reusing aligned tensor allocations
 * Reduces allocation overhead for repeated FITS operations
 */
class AlignedMemoryPool {
public:
    static AlignedMemoryPool& instance();

    /**
     * Get or create an aligned tensor from the pool
     */
    torch::Tensor get_tensor(
        const std::vector<int64_t>& shape,
        torch::Dtype dtype,
        torch::Device device = torch::kCPU
    );

    /**
     * Return tensor to pool for reuse
     */
    void return_tensor(torch::Tensor&& tensor);

    /**
     * Clear all cached tensors
     */
    void clear();

    /**
     * Get memory usage statistics
     */
    struct MemoryStats {
        size_t total_allocated_bytes;
        size_t pooled_tensors_count;
        size_t cache_hit_rate_percent;
    };
    MemoryStats get_stats() const;

private:
    AlignedMemoryPool() = default;
    
    struct TensorCacheKey {
        std::vector<int64_t> shape;
        torch::Dtype dtype;
        torch::Device device;
        
        bool operator==(const TensorCacheKey& other) const;
    };
    
    struct TensorCacheKeyHash {
        size_t operator()(const TensorCacheKey& key) const;
    };
    
    mutable std::mutex pool_mutex_;
    std::unordered_map<TensorCacheKey, std::vector<torch::Tensor>, TensorCacheKeyHash> tensor_pool_;
    
    // Statistics
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;
    mutable size_t total_allocated_bytes_ = 0;
};

} // namespace torchfits_mem

#endif // TORCHFITS_MEMORY_OPTIMIZER_H
