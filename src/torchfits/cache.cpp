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

void LRUCache::put(const std::string& key, const std::shared_ptr<CacheEntry>& entry) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate entry size in MB
    size_t entry_size = entry->size();
    if (entry_size > capacity_mb_) {
        std::stringstream ss;
        ss << "Cannot insert element, the element size (" << entry_size << "MB) is greater than the cache capacity (" << capacity_mb_ << "MB)";
        throw std::runtime_error(ss.str());
    }

    // Check if entry already exists
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        // Move to front (most recently used)
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        //Update the size
        current_size_ -= it->second->second->size();
        // Update entry
        it->second->second = entry;
        current_size_ += entry_size;
        return;
    }

    // Add new entry to front
    cache_list_.push_front({ key, entry });
    cache_map_[key] = cache_list_.begin();
    current_size_ += entry_size;

    // Evict least recently used entries if needed
    while (current_size_ > capacity_mb_ && !cache_list_.empty()) {
        auto last = std::prev(cache_list_.end());
        // Remove entry
        current_size_ = (current_size_ > last->second->size()) ?
            (current_size_ - last->second->size()) : 0;
        cache_map_.erase(last->first);
        cache_list_.pop_back();
    }
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
