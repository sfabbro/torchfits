#include "cache.h"
#include "debug.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream> // For basic logging (can be replaced)

// --- Cross-Platform Memory Check ---
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

// Global cache instance
std::unique_ptr<LRUCache> cache;

// Get available system memory in bytes
size_t get_available_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
    mach_port_t host_port = mach_host_self();
    mach_msg_type_number_t host_size = sizeof(vm_statistics_data_t) / sizeof(integer_t);
    vm_size_t page_size;
    vm_statistics_data_t vm_stat;

    host_page_size(host_port, &page_size);
    if (host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        return 0; // Return 0 on error
    }

    return static_cast<size_t>(vm_stat.free_count) * static_cast<size_t>(page_size);
#else
    struct sysinfo info;
    sysinfo(&info);
    return static_cast<size_t>(info.freeram * info.mem_unit);
#endif
}

// Thread-safe cache initialization (simplified)
void ensure_cache_initialized(size_t capacity_mb) {
    static std::mutex cache_init_mutex;
    std::lock_guard<std::mutex> lock(cache_init_mutex);

    if (!cache || cache->capacity() != capacity_mb) {
        std::cerr << "Initializing cache with capacity: " << capacity_mb << " MB" << std::endl;
        // Set reasonable cache size (25% of available memory or specified size)
        size_t capacity = (capacity_mb > 0) ?
            capacity_mb :
            static_cast<size_t>(0.25 * get_available_memory() / (1024 * 1024));
        capacity = std::min(capacity, static_cast<size_t>(2048)); // Limit to 2GB
        std::cerr << "Final cache capacity: " << capacity << " MB" << std::endl;
        cache = std::make_unique<LRUCache>(capacity);
    }
}

// --- LRUCache Implementation ---

LRUCache::LRUCache(size_t capacity_mb)
    : capacity_mb_(capacity_mb), current_size_(0) {
    std::cerr << "Initializing LRU cache with capacity: " << capacity_mb << " MB" << std::endl;
}

size_t LRUCache::capacity() const { return capacity_mb_; }
size_t LRUCache::size() const { return current_size_; }

void LRUCache::clear() {
    std::cerr << "Clearing LRU cache" << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    cache_list_.clear();
    current_size_ = 0;
}

// Improved LRU cache implementation
void LRUCache::put(const std::string& key, const std::shared_ptr<CacheEntry>& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!entry || !entry->data.defined()) {
        WARNING_LOG("Attempted to cache null or undefined entry");
        return;
    }
    
    // Calculate entry size safely
    size_t entry_size;
    try {
        entry_size = entry->size();
    } catch (const std::exception& e) {
        ERROR_LOG("Error calculating cache entry size: " + std::string(e.what()));
        return;
    }
    
    // Check if entry is too large for cache
    if (entry_size > capacity_mb_) {
        WARNING_LOG("Cache entry too large: " + std::to_string(entry_size) + 
                   "MB exceeds cache capacity of " + std::to_string(capacity_mb_) + "MB");
        return; // Skip caching instead of throwing
    }
    
    // If entry already exists, update it
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        current_size_ -= std::distance(cache_list_.begin(), it->second)->second->size();
        cache_list_.erase(it->second);
        cache_map_.erase(it);
    }
    
    // Make space for new entry
    while (!cache_list_.empty() && (current_size_ + entry_size > capacity_mb_)) {
        auto last = cache_list_.back();
        current_size_ -= last.second->size();
        cache_map_.erase(last.first);
        cache_list_.pop_back();
        INFO_LOG("Evicted cache entry: " + last.first);
    }
    
    // Add new entry to cache
    cache_list_.push_front(std::make_pair(key, entry));
    cache_map_[key] = cache_list_.begin();
    current_size_ += entry_size;
    
    INFO_LOG("Added to cache: " + key + " (" + std::to_string(entry_size) + 
             "MB, current total: " + std::to_string(current_size_) + "MB)");
}

std::shared_ptr<CacheEntry> LRUCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        return nullptr;
    }

    // Move to front (most recently used)
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
}
