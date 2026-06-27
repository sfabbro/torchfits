#include "hardware.h"
#include <sys/stat.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace torchfits {

// Security check: Prevent command injection via cfitsio pipe syntax
void validate_fits_filename(const std::string& filename) {
    if (!filename.empty()) {
        size_t first = filename.find_first_not_of(" \t");
        size_t last = filename.find_last_not_of(" \t");

        if (first != std::string::npos) {
            if (filename[first] == '|' || filename[last] == '|') {
                throw std::runtime_error("Security Error: Filenames starting or ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

// Global hardware info cache
HardwareInfo hw_info;
bool hw_detected = false;
std::mutex hw_mutex;

HardwareInfo detect_hardware() {
    HardwareInfo info;

    // Detect L3 cache size (Linux/macOS)
    #ifdef __APPLE__
    uint64_t cache_size = 0;
    size_t size = sizeof(cache_size);
    if (sysctlbyname("hw.l3cachesize", &cache_size, &size, NULL, 0) == 0) {
        info.l3_cache_size = static_cast<size_t>(cache_size);
    }

    // Detect memory size
    uint64_t mem_size = 0;
    size = sizeof(mem_size);
    if (sysctlbyname("hw.memsize", &mem_size, &size, NULL, 0) == 0) {
        info.available_memory = static_cast<size_t>(mem_size);
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

MMapHandle::MMapHandle(const std::string& filename) : MMapHandle(filename, false) {}

MMapHandle::MMapHandle(void* ptr, size_t size, int fd, bool owner)
    : ptr(ptr), size(size), fd(fd), owner(owner) {}

MMapHandle::MMapHandle(const std::string& filename, bool writable) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file descriptor: " + filename);
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        throw std::runtime_error("Failed to stat file: " + filename);
    }
    size = st.st_size;

    int prot = PROT_READ | (writable ? PROT_WRITE : 0);
    ptr = mmap(nullptr, size, prot, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filename);
    }
    owner = true;
}

void MMapHandle::cleanup() {
    if (ptr) {
        munmap(ptr, size);
        ptr = nullptr;
    }
    if (owner && fd != -1) {
        close(fd);
        fd = -1;
    }
    size = 0;
}

}  // namespace torchfits
