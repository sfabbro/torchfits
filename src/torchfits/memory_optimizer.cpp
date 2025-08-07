#include "memory_optimizer.h"
#include "fits_utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace torchfits_mem {

// --- AlignedTensorFactory Implementation ---

torch::Tensor AlignedTensorFactory::create_aligned_tensor(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype,
    torch::Device device,
    bool fits_compatible
) {
    DEBUG_SCOPE;
    
    // Calculate total elements and byte size
    int64_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    size_t element_size = torch::elementSize(dtype);
    size_t total_bytes = total_elements * element_size;
    
    // Determine optimal alignment
    size_t alignment = get_optimal_alignment(dtype, fits_compatible);
    
    if (device == torch::kCPU) {
        // Create CPU tensor with aligned memory
        void* aligned_buffer = allocate_aligned_memory(total_bytes, alignment);
        if (!aligned_buffer) {
            // Fallback to standard allocation
            DEBUG_LOG("Aligned allocation failed, falling back to standard allocation");
            return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
        }
        
        // Create tensor from aligned buffer
        auto deleter = [](void* ptr) { deallocate_aligned_memory(ptr); };
        torch::Tensor tensor = torch::from_blob(
            aligned_buffer, shape, deleter, torch::TensorOptions().dtype(dtype)
        );
        
        DEBUG_LOG("Created aligned tensor with " + std::to_string(alignment) + 
                  " byte alignment, size: " + std::to_string(total_bytes) + " bytes");
        
        return tensor;
    } else {
        // For GPU tensors, create on CPU first then transfer
        // This ensures optimal memory layout before GPU transfer
        auto cpu_tensor = create_aligned_tensor(shape, dtype, torch::kCPU, fits_compatible);
        return cpu_tensor.to(device);
    }
}

torch::Tensor AlignedTensorFactory::create_from_aligned_buffer(
    void* aligned_buffer,
    const std::vector<int64_t>& shape,
    torch::Dtype dtype,
    torch::Device device
) {
    DEBUG_SCOPE;
    
    if (device == torch::kCPU) {
        // Direct tensor creation from buffer (no ownership transfer)
        return torch::from_blob(
            aligned_buffer, shape, torch::TensorOptions().dtype(dtype)
        );
    } else {
        // Copy to device
        auto cpu_tensor = torch::from_blob(
            aligned_buffer, shape, torch::TensorOptions().dtype(dtype)
        );
        return cpu_tensor.to(device);
    }
}

bool AlignedTensorFactory::is_optimally_aligned(const torch::Tensor& tensor) {
    if (!tensor.is_contiguous()) return false;
    
    uintptr_t ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    size_t optimal_alignment = get_optimal_alignment(tensor.scalar_type());
    
    return (ptr % optimal_alignment) == 0;
}

size_t AlignedTensorFactory::get_optimal_alignment(torch::Dtype dtype, bool fits_compatible) {
    size_t element_size = torch::elementSize(dtype);
    
    // Start with FITS alignment requirement
    size_t alignment = FITS_ALIGNMENT;
    
    // Increase for SIMD operations
    alignment = std::max(alignment, SIMD_ALIGNMENT);
    
    // Ensure alignment is multiple of element size
    alignment = std::max(alignment, element_size);
    
    // For optimal cache performance
    if (element_size >= 8) {
        alignment = std::max(alignment, CACHE_LINE_SIZE);
    }
    
    return alignment;
}

void* AlignedTensorFactory::allocate_aligned_memory(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void AlignedTensorFactory::deallocate_aligned_memory(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

// --- OptimizedTableReader Implementation ---

pybind11::dict OptimizedTableReader::read_table_optimized(
    fitsfile* fptr,
    const std::vector<std::string>& columns,
    long start_row,
    long num_rows,
    torch::Device device
) {
    DEBUG_SCOPE;
    
    if (num_rows <= 0) {
        return pybind11::dict();
    }
    
    // Analyze table structure
    auto column_metadata = analyze_table_structure(fptr, columns);
    if (column_metadata.empty()) {
        return pybind11::dict();
    }
    
    // Check if we can use bulk binary reading
    bool has_variable_length = std::any_of(column_metadata.begin(), column_metadata.end(),
        [](const ColumnMetadata& col) { return col.is_variable_length; });
    
    bool has_strings = std::any_of(column_metadata.begin(), column_metadata.end(),
        [](const ColumnMetadata& col) { return col.is_string; });
    
    // For now, disable bulk binary approach and focus on aligned tensor creation
    // TODO: Re-enable bulk reading after debugging
    /*
    // For tables with only numeric fixed-length columns, use bulk binary approach
    if (!has_variable_length && !has_strings && column_metadata.size() >= 2 && num_rows >= 100) {
        DEBUG_LOG("Using bulk binary table reading for optimized performance");
        
        int status = 0;
        long row_length;
        fits_get_rowsize(fptr, &row_length, &status);
        if (status == 0 && row_length > 0) {
            try {
                // Read raw table data in bulk
                auto binary_data = read_table_bulk_binary(fptr, start_row, num_rows, row_length);
                if (binary_data) {
                    // Parse binary data into aligned tensors
                    return parse_binary_to_tensors(
                        binary_data.get(), column_metadata, num_rows, device
                    );
                }
            } catch (const std::exception& e) {
                DEBUG_LOG("Bulk binary read failed, falling back to column-by-column: " + 
                          std::string(e.what()));
            }
        }
    }
    */
    
    // Fallback to optimized column-by-column reading with aligned tensors
    DEBUG_LOG("Using column-by-column reading with aligned tensors");
    pybind11::dict result;
    
    for (const auto& col : column_metadata) {
        if (col.is_string) {
            // Handle string columns traditionally
            std::vector<char*> string_array(num_rows);
            std::vector<char> string_buffer(num_rows * (col.repeat_count + 1));
            
            for (long i = 0; i < num_rows; i++) {
                string_array[i] = &string_buffer[i * (col.repeat_count + 1)];
            }
            
            int status = 0;
            fits_read_col_str(fptr, col.fits_column_num, start_row + 1, 1, num_rows,
                              nullptr, string_array.data(), nullptr, &status);
            
            if (status) {
                throw_fits_error(status, "Error reading string column: " + col.name);
            }
            
            pybind11::list string_list;
            for (long i = 0; i < num_rows; i++) {
                string_list.append(pybind11::str(string_array[i]));
            }
            result[pybind11::str(col.name)] = std::move(string_list);
        } else {
            // Create aligned tensor for numeric data
            std::vector<int64_t> shape;
            if (col.repeat_count == 1) {
                shape = {num_rows};
            } else {
                shape = {num_rows, col.repeat_count};
            }
            
            auto tensor = AlignedTensorFactory::create_aligned_tensor(
                shape, col.torch_dtype, torch::kCPU, true
            );
            
            // Read data directly into aligned tensor
            int status = 0;
            fits_read_col(fptr, col.fits_type, col.fits_column_num, start_row + 1, 1,
                         num_rows * col.repeat_count,
                         nullptr, tensor.data_ptr(), nullptr, &status);
            
            if (status) {
                throw_fits_error(status, "Error reading column: " + col.name);
            }
            
            // Transfer to target device if needed
            if (device != torch::kCPU) {
                tensor = tensor.to(device);
            }
            
            result[pybind11::str(col.name)] = tensor.squeeze();
        }
    }
    
    return result;
}

std::vector<OptimizedTableReader::ColumnMetadata> OptimizedTableReader::analyze_table_structure(
    fitsfile* fptr,
    const std::vector<std::string>& requested_columns
) {
    DEBUG_SCOPE;
    
    int status = 0;
    int total_cols;
    fits_get_num_cols(fptr, &total_cols, &status);
    if (status) {
        throw_fits_error(status, "Error getting number of columns");
    }
    
    std::vector<std::string> columns_to_process;
    if (requested_columns.empty()) {
        // Get all column names
        for (int i = 1; i <= total_cols; ++i) {
            char colname[FLEN_VALUE];
            status = 0;
            fits_get_bcolparms(fptr, i, colname, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &status);
            if (status == 0) {
                columns_to_process.emplace_back(colname);
            }
        }
    } else {
        columns_to_process = requested_columns;
    }
    
    std::vector<ColumnMetadata> metadata;
    
    // Get the total row length from FITS header for proper byte offset calculation
    long fits_row_length;
    status = 0;
    fits_get_rowsize(fptr, &fits_row_length, &status);
    if (status) {
        throw_fits_error(status, "Error getting FITS table row size");
    }
    
    for (const auto& col_name : columns_to_process) {
        ColumnMetadata col_meta;
        col_meta.name = col_name;
        
        // Get column number
        status = 0;
        fits_get_colnum(fptr, CASEINSEN, const_cast<char*>(col_name.c_str()), 
                        &col_meta.fits_column_num, &status);
        if (status) continue; // Skip invalid columns
        
        // Get column type information
        fits_get_coltype(fptr, col_meta.fits_column_num, &col_meta.fits_type, 
                         &col_meta.repeat_count, &col_meta.byte_width, &status);
        if (status) continue;
        
        // Calculate byte offset manually - CFITSIO doesn't have fits_get_col_offset
        // We need to traverse columns to calculate offsets
        size_t offset = 0;
        for (int c = 1; c < col_meta.fits_column_num; ++c) {
            int type;
            long repeat, width;
            fits_get_coltype(fptr, c, &type, &repeat, &width, &status);
            if (status == 0) {
                offset += static_cast<size_t>(repeat * width);
            }
            status = 0; // Reset for next iteration
        }
        
        col_meta.byte_offset = offset;
        col_meta.byte_width = static_cast<size_t>(fits_row_length); // Store total row width for parsing
        
        col_meta.is_string = (col_meta.fits_type == TSTRING);
        col_meta.is_variable_length = (col_meta.repeat_count < 0);
        
        // Map FITS type to torch dtype
        switch (col_meta.fits_type) {
            case TBYTE:     col_meta.torch_dtype = torch::kUInt8; break;
            case TSHORT:    col_meta.torch_dtype = torch::kInt16; break;
            case TINT:      col_meta.torch_dtype = torch::kInt32; break;
            case TLONG:     col_meta.torch_dtype = torch::kInt64; break;
            case TFLOAT:    col_meta.torch_dtype = torch::kFloat32; break;
            case TDOUBLE:   col_meta.torch_dtype = torch::kFloat64; break;
            case TLOGICAL:  col_meta.torch_dtype = torch::kBool; break;
            default:        col_meta.torch_dtype = torch::kFloat64; break;
        }
        
        metadata.push_back(col_meta);
    }
    
    return metadata;
}

std::unique_ptr<uint8_t[], OptimizedTableReader::AlignedDeleter> OptimizedTableReader::read_table_bulk_binary(
    fitsfile* fptr,
    long start_row,
    long num_rows,
    long row_length
) {
    DEBUG_SCOPE;
    
    // Calculate total bytes to read
    size_t total_bytes = static_cast<size_t>(num_rows * row_length);
    
    // Allocate aligned buffer for bulk read
    size_t alignment = AlignedTensorFactory::get_optimal_alignment(torch::kUInt8, true);
    void* aligned_buffer = AlignedTensorFactory::public_allocate_aligned_memory(total_bytes, alignment);
    
    if (!aligned_buffer) {
        throw std::runtime_error("Failed to allocate aligned memory for bulk table read");
    }
    
    // Use fits_read_tblbytes for maximum performance
    int status = 0;
    LONGLONG first_char = 1;  // FITS uses 1-based indexing
    LONGLONG num_chars = static_cast<LONGLONG>(total_bytes);
    
    fits_read_tblbytes(fptr, start_row + 1, first_char, num_chars, 
                       static_cast<unsigned char*>(aligned_buffer), &status);
    
    if (status) {
        AlignedTensorFactory::public_deallocate_aligned_memory(aligned_buffer);
        throw_fits_error(status, "Error in bulk binary table read");
    }
    
    DEBUG_LOG("Successfully read " + std::to_string(total_bytes) + 
              " bytes using bulk binary approach");
    
    // Return unique_ptr with aligned memory
    return std::unique_ptr<uint8_t[], AlignedDeleter>(static_cast<uint8_t*>(aligned_buffer));
}

pybind11::dict OptimizedTableReader::parse_binary_to_tensors(
    const uint8_t* binary_data,
    const std::vector<ColumnMetadata>& columns,
    long num_rows,
    torch::Device device
) {
    DEBUG_SCOPE;
    
    pybind11::dict result;
    
    for (const auto& col : columns) {
        if (col.is_string || col.is_variable_length) {
            continue; // Skip non-numeric columns in binary parsing
        }
        
        // Create aligned tensor
        std::vector<int64_t> shape;
        if (col.repeat_count == 1) {
            shape = {num_rows};
        } else {
            shape = {num_rows, col.repeat_count};
        }
        
        auto tensor = create_tensor_from_binary(binary_data, col, num_rows, device);
        result[pybind11::str(col.name)] = tensor.squeeze();
    }
    
    return result;
}

torch::Tensor OptimizedTableReader::create_tensor_from_binary(
    const uint8_t* data,
    const ColumnMetadata& column,
    long num_rows,
    torch::Device device
) {
    DEBUG_SCOPE;
    
    std::vector<int64_t> shape;
    if (column.repeat_count == 1) {
        shape = {num_rows};
    } else {
        shape = {num_rows, column.repeat_count};
    }
    
    // Create aligned tensor
    auto tensor = AlignedTensorFactory::create_aligned_tensor(
        shape, column.torch_dtype, torch::kCPU, true
    );
    
    // Extract column data from binary buffer
    size_t element_size = torch::elementSize(column.torch_dtype);
    size_t col_elements_per_row = column.repeat_count;
    size_t col_bytes_per_row = col_elements_per_row * element_size;
    
    // FITS tables are row-major: each row contains all columns sequentially
    // We need to extract our specific column from each row
    char* tensor_data = static_cast<char*>(tensor.data_ptr());
    
    for (long row = 0; row < num_rows; ++row) {
        // Calculate source position: row start + column offset
        const char* row_start = reinterpret_cast<const char*>(data + row * column.byte_width);
        const char* col_data = row_start + column.byte_offset;
        
        // Calculate destination position in tensor
        char* dest = tensor_data + row * col_bytes_per_row;
        
        // Copy column data for this row
        std::memcpy(dest, col_data, col_bytes_per_row);
    }
    
    DEBUG_LOG("Extracted column '" + column.name + "' from binary data: " +
              std::to_string(num_rows) + " rows × " + std::to_string(col_elements_per_row) + " elements");
    
    if (device != torch::kCPU) {
        tensor = tensor.to(device);
    }
    
    return tensor;
}

// --- AlignedMemoryPool Implementation ---

AlignedMemoryPool& AlignedMemoryPool::instance() {
    static AlignedMemoryPool instance;
    return instance;
}

torch::Tensor AlignedMemoryPool::get_tensor(
    const std::vector<int64_t>& shape,
    torch::Dtype dtype,
    torch::Device device
) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    TensorCacheKey key{shape, dtype, device};
    
    auto it = tensor_pool_.find(key);
    if (it != tensor_pool_.end() && !it->second.empty()) {
        // Reuse cached tensor
        torch::Tensor tensor = std::move(it->second.back());
        it->second.pop_back();
        
        if (it->second.empty()) {
            tensor_pool_.erase(it);
        }
        
        cache_hits_++;
        return tensor;
    }
    
    // Create new aligned tensor
    cache_misses_++;
    auto tensor = AlignedTensorFactory::create_aligned_tensor(shape, dtype, device, true);
    
    // Update statistics
    int64_t elements = 1;
    for (auto dim : shape) elements *= dim;
    size_t bytes = elements * torch::elementSize(dtype);
    total_allocated_bytes_ += bytes;
    
    return tensor;
}

void AlignedMemoryPool::return_tensor(torch::Tensor&& tensor) {
    if (!tensor.defined() || !tensor.is_contiguous()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    TensorCacheKey key{tensor.sizes().vec(), tensor.scalar_type(), tensor.device()};
    tensor_pool_[key].emplace_back(std::move(tensor));
}

void AlignedMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    tensor_pool_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
    total_allocated_bytes_ = 0;
}

AlignedMemoryPool::MemoryStats AlignedMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t pooled_count = 0;
    for (const auto& pair : tensor_pool_) {
        pooled_count += pair.second.size();
    }
    
    size_t total_requests = cache_hits_ + cache_misses_;
    size_t hit_rate = (total_requests > 0) ? (cache_hits_ * 100 / total_requests) : 0;
    
    return MemoryStats{
        total_allocated_bytes_,
        pooled_count,
        hit_rate
    };
}

// TensorCacheKey implementations
bool AlignedMemoryPool::TensorCacheKey::operator==(const TensorCacheKey& other) const {
    return shape == other.shape && dtype == other.dtype && device == other.device;
}

size_t AlignedMemoryPool::TensorCacheKeyHash::operator()(const TensorCacheKey& key) const {
    size_t hash = 0;
    
    // Hash shape
    for (auto dim : key.shape) {
        hash ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    
    // Hash dtype and device
    hash ^= std::hash<int8_t>{}(static_cast<int8_t>(key.dtype)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int8_t>{}(static_cast<int8_t>(key.device.type())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    
    return hash;
}

} // namespace torchfits_mem
