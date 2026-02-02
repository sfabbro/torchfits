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

namespace torchfits {

// Simplified cache entry
struct CacheEntry {
    fitsfile* fptr = nullptr;
    std::list<std::string>::iterator lru_iter;
    size_t refcount = 0;
};

class UnifiedCache {
public:
    UnifiedCache(size_t max_files = 100, size_t max_memory_mb = 1024)
        : max_files_(max_files), max_memory_mb_(max_memory_mb) {}

    void configure(size_t max_files, size_t max_memory_mb) {
        max_files_ = max_files;
        max_memory_mb_ = max_memory_mb;
    }

    fitsfile* get_or_open(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(filepath);
        if (it != cache_.end()) {
            // Update LRU - move to front
            lru_list_.erase(it->second.lru_iter);
            lru_list_.push_front(filepath);
            it->second.lru_iter = lru_list_.begin();
            it->second.refcount += 1;
            return it->second.fptr;
        }

        // Open new file
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filepath.c_str(), READONLY, &status);

        if (status != 0) return nullptr;

        // Add to cache
        CacheEntry entry;
        entry.fptr = fptr;
        lru_list_.push_front(filepath);
        entry.lru_iter = lru_list_.begin();
        entry.refcount = 1;
        cache_[filepath] = entry;

        // Simple LRU eviction by count only
        if (cache_.size() > max_files_) {
            evict_lru();
        }

        return fptr;
    }

    void release(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(filepath);
        if (it == cache_.end()) {
            return;
        }
        if (it->second.refcount > 0) {
            it->second.refcount -= 1;
        }
        if (cache_.size() > max_files_) {
            evict_lru();
        }
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
    size_t max_memory_mb_;

    void evict_lru() {
        if (lru_list_.empty()) return;

        // Evict the least-recently used entry that is not in use.
        for (auto it_list = lru_list_.rbegin(); it_list != lru_list_.rend(); ++it_list) {
            const std::string& path = *it_list;
            auto it = cache_.find(path);
            if (it == cache_.end()) {
                continue;
            }
            if (it->second.refcount != 0) {
                continue;
            }
            if (it->second.fptr) {
                int status = 0;
                fits_close_file(it->second.fptr, &status);
            }
            // erase from LRU list
            lru_list_.erase(it->second.lru_iter);
            cache_.erase(it);
            break;
        }
    }
};

// Global cache instance
static UnifiedCache global_cache;

// C-style helpers for bindings
inline void configure_cache(size_t max_files, size_t max_memory_mb) {
    global_cache.configure(max_files, max_memory_mb);
}

inline void clear_file_cache() {
    global_cache.clear();
}

inline size_t get_cache_size() {
    return global_cache.size();
}

inline fitsfile* get_or_open_cached(const std::string& filepath) {
    return global_cache.get_or_open(filepath);
}

inline void release_cached(const std::string& filepath) {
    global_cache.release(filepath);
}

}  // namespace torchfits
