#pragma once

#include <cstdlib>
#include <string>
#include <mutex>

namespace torchfits {

struct HardwareInfo {
    size_t l3_cache_size = 8 * 1024 * 1024;  // 8MB default
    size_t memory_bandwidth = 50 * 1024 * 1024 * 1024ULL;  // 50GB/s default
    size_t available_memory = 8 * 1024 * 1024 * 1024ULL;   // 8GB default
    bool is_nvme = true;  // Assume fast storage
    size_t storage_bandwidth = 3 * 1024 * 1024 * 1024ULL;  // 3GB/s default (NVMe)
};

// Global hardware info cache
extern HardwareInfo hw_info;
extern bool hw_detected;
extern std::mutex hw_mutex;

// Function declarations
HardwareInfo detect_hardware();
size_t calculate_optimal_chunk_size(size_t data_size, const HardwareInfo& hw, const std::string& filepath = "");
size_t detect_storage_speed(const std::string& filepath);

// RAII wrapper for mmap
class MMapHandle {
public:
    void* ptr = nullptr;
    size_t size = 0;
    int fd = -1;
    bool owner = false;

    MMapHandle() = default;
    explicit MMapHandle(const std::string& filename);
    
    // Move constructor
    MMapHandle(MMapHandle&& other) noexcept 
        : ptr(other.ptr), size(other.size), fd(other.fd), owner(other.owner) {
        other.ptr = nullptr;
        other.size = 0;
        other.fd = -1;
        other.owner = false;
    }
    
    // Move assignment
    MMapHandle& operator=(MMapHandle&& other) noexcept {
        if (this != &other) {
            cleanup();
            ptr = other.ptr;
            size = other.size;
            fd = other.fd;
            owner = other.owner;
            other.ptr = nullptr;
            other.size = 0;
            other.fd = -1;
            other.owner = false;
        }
        return *this;
    }

    ~MMapHandle() {
        cleanup();
    }
    
    void cleanup(); // Implementation in hardware.cpp or inline if header-only
};

// Helper to swap bytes for int16 (Big Endian -> Little Endian)
inline int16_t bswap_16(int16_t x) {
    return __builtin_bswap16(x);
}

// Helper to swap bytes for int32/float32 (Big Endian -> Little Endian)
inline int32_t bswap_32(int32_t x) {
    return __builtin_bswap32(x);
}

// Helper to swap bytes for int64/double (Big Endian -> Little Endian)
inline int64_t bswap_64(int64_t x) {
    return __builtin_bswap64(x);
}

}  // namespace torchfits