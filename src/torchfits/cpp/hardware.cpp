#include "hardware.h"
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>

namespace torchfits {

// Global hardware info cache
HardwareInfo hw_info;
bool hw_detected = false;
std::mutex hw_mutex;

size_t detect_storage_speed(const std::string& filepath) {
    // Simple storage speed detection
    #ifdef __APPLE__
    // Check if on SSD (most Macs are NVMe)
    return 3 * 1024 * 1024 * 1024ULL;  // 3GB/s NVMe
    #endif
    
    #ifdef __linux__
    // Check filesystem type or device info
    // For now, assume NVMe if file path suggests it
    if (filepath.find("/dev/nvme") != std::string::npos) {
        return 3 * 1024 * 1024 * 1024ULL;  // 3GB/s NVMe
    } else if (filepath.find("/dev/sd") != std::string::npos) {
        return 500 * 1024 * 1024ULL;  // 500MB/s SATA SSD
    }
    #endif
    
    // Default to fast storage
    return 3 * 1024 * 1024 * 1024ULL;
}

HardwareInfo detect_hardware() {
    HardwareInfo info;
    
    // Detect L3 cache size (Linux/macOS)
    #ifdef __APPLE__
    FILE* fp = popen("sysctl -n hw.l3cachesize", "r");
    if (fp) {
        size_t cache_size;
        if (fscanf(fp, "%zu", &cache_size) == 1) {
            info.l3_cache_size = cache_size;
        }
        pclose(fp);
    }
    
    // Detect memory size
    fp = popen("sysctl -n hw.memsize", "r");
    if (fp) {
        size_t mem_size;
        if (fscanf(fp, "%zu", &mem_size) == 1) {
            info.available_memory = mem_size;
        }
        pclose(fp);
    }
    #endif
    
    #ifdef __linux__
    // Read /proc/cpuinfo for cache info
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cache size") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string cache_str = line.substr(pos + 1);
                size_t cache_kb = std::stoul(cache_str);
                info.l3_cache_size = cache_kb * 1024;
                break;
            }
        }
    }
    
    // Read /proc/meminfo for memory
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string mem_str = line.substr(pos + 1);
                size_t mem_kb = std::stoul(mem_str);
                info.available_memory = mem_kb * 1024;
                break;
            }
        }
    }
    #endif
    
    return info;
}

size_t calculate_optimal_chunk_size(size_t data_size, const HardwareInfo& hw, const std::string& filepath) {
    // CFITSIO MINDIRECT threshold - 3 FITS blocks (8640 bytes)
    const size_t CFITSIO_MINDIRECT = 8640;
    
    // Small data: single read if fits in L3 cache OR above MINDIRECT threshold
    if (data_size <= hw.l3_cache_size || data_size >= CFITSIO_MINDIRECT) {
        return data_size;
    }
    
    // Detect storage speed for this file
    size_t storage_speed = detect_storage_speed(filepath);
    
    // Medium data: use L3 cache sized chunks
    if (data_size <= hw.available_memory / 4) {
        return std::min(hw.l3_cache_size, data_size / 4);
    }
    
    // Large data: optimize for storage vs memory bandwidth
    // Target ~100ms per chunk at limiting bandwidth
    size_t storage_chunk = storage_speed / 10;  // 100ms at storage speed
    size_t memory_chunk = hw.memory_bandwidth / 10;  // 100ms at memory speed
    
    // Use the limiting factor
    size_t bandwidth_chunk = std::min(storage_chunk, memory_chunk);
    
    // But don't exceed 1/8 of available memory
    size_t memory_limit = hw.available_memory / 8;
    
    return std::min(std::min(bandwidth_chunk, memory_limit), data_size / 2);
}

}  // namespace torchfits