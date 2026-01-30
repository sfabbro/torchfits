#pragma once
#include <unordered_map>
#include <list>
#include <mutex>
#include <string>
#include <memory>
#include <fitsio.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/statfs.h>
#endif
#ifdef __APPLE__
#include <sys/mount.h>
#endif

#include "security.h"

namespace torchfits {

// Simplified cache entry
struct CacheEntry {
    fitsfile* fptr = nullptr;
    std::list<std::string>::iterator lru_iter;
};

class UnifiedCache {
public:
    UnifiedCache(size_t max_files = 100, size_t max_memory_mb = 1024)
        : max_files_(max_files) {}

    fitsfile* get_or_open(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(filepath);
        if (it != cache_.end()) {
            // Update LRU - move to front
            lru_list_.erase(it->second.lru_iter);
            lru_list_.push_front(filepath);
            it->second.lru_iter = lru_list_.begin();
            return it->second.fptr;
        }

        // Open new file
        // Security check: Prevent command injection via cfitsio pipe syntax
        validate_fits_filename(filepath);

        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filepath.c_str(), READONLY, &status);

        if (status != 0) return nullptr;

        // Add to cache
        CacheEntry entry;
        entry.fptr = fptr;
        lru_list_.push_front(filepath);
        entry.lru_iter = lru_list_.begin();
        cache_[filepath] = entry;

        // Simple LRU eviction by count only
        if (cache_.size() > max_files_) {
            evict_lru();
        }

        return fptr;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [path, entry] : cache_) {
            if (entry.fptr) {
                int status = 0;
                fits_close_file(entry.fptr, &status);
            }
        }
        cache_.clear();
        lru_list_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

private:
    std::unordered_map<std::string, CacheEntry> cache_;
    std::list<std::string> lru_list_;
    mutable std::mutex mutex_;
    size_t max_files_;

    void evict_lru() {
        if (lru_list_.empty()) return;

        std::string path = lru_list_.back();
        auto it = cache_.find(path);

        if (it != cache_.end()) {
            if (it->second.fptr) {
                int status = 0;
                fits_close_file(it->second.fptr, &status);
            }
            cache_.erase(it);
        }

        lru_list_.pop_back();
    }
};

// Global cache instance
static UnifiedCache global_cache;

}  // namespace torchfits