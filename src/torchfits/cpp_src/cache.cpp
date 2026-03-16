#pragma once
#include <unordered_map>
#include <list>
#include <mutex>
#include <string>
#include <memory>
#include <chrono>
#include <fitsio.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/statfs.h>
#endif
#ifdef __APPLE__
#include <sys/mount.h>
#endif

namespace torchfits {

namespace {
inline bool env_flag_default_true(const char* name) {
    const char* v = std::getenv(name);
    if (!v) {
        return true;
    }
    std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF" ||
             s == "no" || s == "NO");
}

inline int64_t env_nonnegative_int(const char* name, int64_t default_value) {
    const char* v = std::getenv(name);
    if (!v) {
        return default_value;
    }
    try {
        int64_t parsed = std::stoll(std::string(v));
        return parsed < 0 ? 0 : parsed;
    } catch (...) {
        return default_value;
    }
}

inline int64_t monotonic_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Validate cached handles against filesystem state. This adds a stat() on cache hits.
// Keep this enabled by default for correctness when files are modified externally;
// users can explicitly disable it via TORCHFITS_CACHE_VALIDATE=0 for pure throughput.
const bool kValidateCache = []() {
    return env_flag_default_true("TORCHFITS_CACHE_VALIDATE");
}();

// To avoid paying stat() on every tiny hot-loop read, validate at most once per
// interval per path by default. Set TORCHFITS_CACHE_VALIDATE_INTERVAL_MS=0 for
// strict per-access validation.
const int64_t kValidateIntervalNs = []() {
    // Balance stale-file detection with hot-path latency.
    // A longer default interval reduces repeated stat() overhead in hot loops.
    constexpr int64_t kDefaultMs = 1000;
    return env_nonnegative_int("TORCHFITS_CACHE_VALIDATE_INTERVAL_MS", kDefaultMs) * 1000000LL;
}();
}  // namespace

// Simplified cache entry
struct CacheEntry {
    fitsfile* fptr = nullptr;
    std::list<std::string>::iterator lru_iter;
    size_t refcount = 0;
    bool has_stat = false;
    off_t size = 0;
    int64_t mtime_ns = 0;
    ino_t inode = 0;
    bool stale = false;
    int64_t last_validate_ns = 0;
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

        auto mtime_ns_from_stat = [&](const struct stat& st) -> int64_t {
#if defined(__APPLE__)
            return (static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL) +
                   static_cast<int64_t>(st.st_mtimespec.tv_nsec);
#else
            return (static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL) +
                   static_cast<int64_t>(st.st_mtim.tv_nsec);
#endif
        };

        auto read_stat = [&](bool* out_has_stat, off_t* out_size, int64_t* out_mtime_ns, ino_t* out_inode) {
            *out_has_stat = false;
            *out_size = 0;
            *out_mtime_ns = 0;
            *out_inode = 0;
            if (filepath.find('[') != std::string::npos) {
                return;
            }
            struct stat st {};
            if (stat(filepath.c_str(), &st) == 0) {
                *out_has_stat = true;
                *out_size = st.st_size;
                *out_mtime_ns = mtime_ns_from_stat(st);
                *out_inode = st.st_ino;
            }
        };

        auto it = cache_.find(filepath);
        if (it != cache_.end()) {
            if (kValidateCache) {
                const int64_t now_ns = monotonic_now_ns();
                const bool should_validate =
                    (kValidateIntervalNs <= 0 || it->second.last_validate_ns == 0 ||
                     (now_ns - it->second.last_validate_ns) >= kValidateIntervalNs);
                if (should_validate) {
                    bool cur_has_stat = false;
                    off_t cur_size = 0;
                    int64_t cur_mtime_ns = 0;
                    ino_t cur_inode = 0;
                    read_stat(&cur_has_stat, &cur_size, &cur_mtime_ns, &cur_inode);
                    it->second.last_validate_ns = now_ns;

                    // If the underlying file changed and this cached handle isn't in use,
                    // drop it so subsequent reads see the new file contents.
                    if (cur_has_stat && it->second.has_stat &&
                        (it->second.size != cur_size ||
                         it->second.mtime_ns != cur_mtime_ns ||
                         it->second.inode != cur_inode)) {
                        it->second.stale = true;
                    }
                }
            }
            if (it->second.stale && it->second.refcount == 0) {
                if (it->second.fptr) {
                    int status = 0;
                    fits_close_file(it->second.fptr, &status);
                }
                // Remove and fall through to re-open.
                lru_list_.erase(it->second.lru_iter);
                cache_.erase(it);
            } else {
            // Update LRU - move to front
            lru_list_.erase(it->second.lru_iter);
            lru_list_.push_front(filepath);
            it->second.lru_iter = lru_list_.begin();
            it->second.refcount += 1;
            return it->second.fptr;
            }
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
        if (kValidateCache) {
            read_stat(&entry.has_stat, &entry.size, &entry.mtime_ns, &entry.inode);
            entry.last_validate_ns = monotonic_now_ns();
        }
        entry.stale = false;
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
        // If the file changed while a handle was outstanding, close the cached handle
        // as soon as it is no longer referenced, so future opens see fresh contents.
        if (it->second.stale && it->second.refcount == 0) {
            if (it->second.fptr) {
                int status = 0;
                fits_close_file(it->second.fptr, &status);
            }
            lru_list_.erase(it->second.lru_iter);
            cache_.erase(it);
            return;
        }
        if (cache_.size() > max_files_) {
            evict_lru();
        }
    }

    // Remove a path from the cache. If it's currently referenced, mark stale and
    // it will be dropped when the last reference is released.
    void invalidate(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(filepath);
        if (it == cache_.end()) {
            return;
        }
        it->second.stale = true;
        if (it->second.refcount == 0) {
            if (it->second.fptr) {
                int status = 0;
                fits_close_file(it->second.fptr, &status);
            }
            lru_list_.erase(it->second.lru_iter);
            cache_.erase(it);
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

inline void invalidate_cached(const std::string& filepath) {
    global_cache.invalidate(filepath);
}

}  // namespace torchfits
