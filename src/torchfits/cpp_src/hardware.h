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

}  // namespace torchfits