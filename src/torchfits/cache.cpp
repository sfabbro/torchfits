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
    
    // If capacity is 0, disable cache completely
    if (capacity_mb == 0) {
        DEBUG_LOG("Cache disabled (capacity = 0)");
        cache.reset();
        return;
    }
    
    size_t capacity_bytes_requested = capacity_mb * 1024 * 1024;

    // Check if cache exists and capacity matches requested bytes
    if (!cache || cache->capacity_bytes() != capacity_bytes_requested) {
        DEBUG_LOG("Initializing cache. Requested capacity: " + std::to_string(capacity_mb) + " MB");
        
        // Calculate default capacity (25% of available memory in bytes)
        size_t available_memory = get_available_memory();
        size_t default_capacity_bytes = static_cast<size_t>(0.25 * available_memory);
        
        // Determine final capacity in bytes
        size_t capacity_bytes = (capacity_mb > 0) ? capacity_bytes_requested : default_capacity_bytes;

        // Limit to 2GB (adjust as needed)
        size_t max_capacity_bytes = 2ULL * 1024 * 1024 * 1024;
        capacity_bytes = std::min(capacity_bytes, max_capacity_bytes);
        capacity_bytes = std::max(capacity_bytes, (size_t)1024*1024); // Ensure at least 1MB if possible
        
        DEBUG_LOG("Final cache capacity: " + std::to_string(capacity_bytes / (1024*1024)) + " MB (" + std::to_string(capacity_bytes) + " bytes)");
        cache = std::make_unique<LRUCache>(capacity_bytes); // Pass bytes to constructor
    }
}

// --- LRUCache Implementation ---

LRUCache::LRUCache(size_t capacity_bytes) // Constructor takes bytes
    : capacity_bytes_(capacity_bytes), current_size_bytes_(0) {
    DEBUG_LOG("Initializing LRU cache with capacity: " + std::to_string(capacity_bytes_ / (1024*1024)) + " MB (" + std::to_string(capacity_bytes_) + " bytes)");
}

size_t LRUCache::capacity_bytes() const { return capacity_bytes_; } // Renamed accessor
size_t LRUCache::size_bytes() const { return current_size_bytes_; }     // Renamed accessor

void LRUCache::clear() {
    DEBUG_LOG("Clearing LRU cache");
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    cache_list_.clear();
    current_size_bytes_ = 0;
}

// Improved LRU cache implementation using bytes
void LRUCache::put(const std::string& key, const std::shared_ptr<CacheEntry>& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!entry || !entry->data.defined()) {
        WARNING_LOG("Attempted to cache null or undefined entry");
        return;
    }
    
    size_t entry_size_bytes;
    try {
        entry_size_bytes = entry->size(); // size() now returns bytes
    } catch (const std::exception& e) {
        ERROR_LOG("Error calculating cache entry size: " + std::string(e.what()));
        return;
    }

    // Size check using bytes
    if (entry_size_bytes == 0) { // Don't cache zero-size entries
        WARNING_LOG("Skipping caching of zero-size entry for key: " + key);
        return;
    }

    if (entry_size_bytes > capacity_bytes_) {
        WARNING_LOG("Cache entry too large: " + std::to_string(entry_size_bytes) + 
                   " bytes exceeds cache capacity of " + std::to_string(capacity_bytes_) + " bytes");
        return; 
    }
    
    // Eviction loop using bytes
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        // Entry exists, remove its size before updating
        current_size_bytes_ -= (*it->second).second->size(); 
        cache_list_.erase(it->second);
        cache_map_.erase(it);
    }
    
    // Make space for new entry (use bytes)
    while (!cache_list_.empty() && (current_size_bytes_ + entry_size_bytes > capacity_bytes_)) {
        auto last = cache_list_.back();
        size_t evicted_size = last.second->size(); // Get size in bytes
        current_size_bytes_ -= evicted_size;
        cache_map_.erase(last.first);
        cache_list_.pop_back();
        DEBUG_LOG("Evicted cache entry: " + last.first + " (" + std::to_string(evicted_size) + " bytes)");
    }
    
    // Add new entry
    cache_list_.push_front(std::make_pair(key, entry));
    cache_map_[key] = cache_list_.begin();
    current_size_bytes_ += entry_size_bytes;
    
    DEBUG_LOG("Added to cache: " + key + " (" + std::to_string(entry_size_bytes) + 
             " bytes, current total: " + std::to_string(current_size_bytes_) + " bytes)");
}

std::shared_ptr<CacheEntry> LRUCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If cache is disabled (nullptr), always return nullptr
    if (!cache) {
        return nullptr;
    }
    
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        return nullptr;
    }

    // Move to front (most recently used)
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return (*it->second).second;
}
