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

// Unified cache entry
struct CacheEntry {
    fitsfile* fptr = nullptr;
    size_t access_count = 0;
    std::chrono::steady_clock::time_point last_access;
    bool is_remote = false;
    size_t file_size = 0;
};

class UnifiedCache {
public:
    UnifiedCache(size_t max_files = 100, size_t max_memory_mb = 1024) 
        : max_files_(max_files), max_memory_bytes_(max_memory_mb * 1024 * 1024) {}
    
    fitsfile* get_or_open(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(filepath);
        if (it != cache_.end()) {
            // Update LRU
            lru_list_.erase(it->second.lru_iter);
            lru_list_.push_front(filepath);
            it->second.lru_iter = lru_list_.begin();
            it->second.entry.access_count++;
            it->second.entry.last_access = std::chrono::steady_clock::now();
            return it->second.entry.fptr;
        }
        
        // Open new file
        fitsfile* fptr = nullptr;
        int status = 0;
        fits_open_file(&fptr, filepath.c_str(), READONLY, &status);
        
        if (status != 0) return nullptr;
        
        // Detect if remote (cloud/HPC common)
        bool is_remote = is_remote_file(filepath);
        size_t file_size = get_file_size(fptr);
        
        // Add to cache
        CacheItem item;
        item.entry.fptr = fptr;
        item.entry.is_remote = is_remote;
        item.entry.file_size = file_size;
        item.entry.access_count = 1;
        item.entry.last_access = std::chrono::steady_clock::now();
        
        lru_list_.push_front(filepath);
        item.lru_iter = lru_list_.begin();
        
        cache_[filepath] = std::move(item);
        current_memory_ += file_size;
        
        // Evict if needed
        evict_if_needed();
        
        return fptr;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [path, item] : cache_) {
            if (item.entry.fptr) {
                int status = 0;
                fits_close_file(item.entry.fptr, &status);
            }
        }
        cache_.clear();
        lru_list_.clear();
        current_memory_ = 0;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

private:
    struct CacheItem {
        CacheEntry entry;
        std::list<std::string>::iterator lru_iter;
    };
    
    std::unordered_map<std::string, CacheItem> cache_;
    std::list<std::string> lru_list_;
    mutable std::mutex mutex_;
    size_t max_files_;
    size_t max_memory_bytes_;
    size_t current_memory_ = 0;
    
    bool is_remote_file(const std::string& path) {
        // Network protocols
        if (path.find("http://") == 0 || path.find("https://") == 0 || 
            path.find("ftp://") == 0 || path.find("s3://") == 0) {
            return true;
        }
        
        // Check filesystem type via statfs/statvfs
        #ifdef __linux__
        struct statfs fs;
        if (statfs(path.c_str(), &fs) == 0) {
            // Network filesystems
            return fs.f_type == 0x6969 ||  // NFS
                   fs.f_type == 0x517B ||  // SMB
                   fs.f_type == 0x65735546; // FUSE (often network)
        }
        #endif
        
        #ifdef __APPLE__
        struct statfs fs;
        if (statfs(path.c_str(), &fs) == 0) {
            // Network filesystems on macOS
            return strcmp(fs.f_fstypename, "nfs") == 0 ||
                   strcmp(fs.f_fstypename, "smbfs") == 0;
        }
        #endif
        
        return false;
    }
    
    size_t get_file_size(fitsfile* fptr) {
        long naxes[10];
        int naxis, status = 0;
        fits_get_img_param(fptr, 10, nullptr, &naxis, naxes, &status);
        
        if (status != 0) return 1024 * 1024; // 1MB default
        
        size_t total_pixels = 1;
        for (int i = 0; i < naxis; i++) {
            total_pixels *= naxes[i];
        }
        return total_pixels * 4; // Assume float32
    }
    
    void evict_if_needed() {
        // Evict by memory pressure first (cloud/HPC critical)
        while (current_memory_ > max_memory_bytes_ && !lru_list_.empty()) {
            evict_lru();
        }
        
        // Then by file count
        while (cache_.size() > max_files_ && !lru_list_.empty()) {
            evict_lru();
        }
    }
    
    void evict_lru() {
        if (lru_list_.empty()) return;
        
        std::string path = lru_list_.back();
        auto it = cache_.find(path);
        
        if (it != cache_.end()) {
            if (it->second.entry.fptr) {
                int status = 0;
                fits_close_file(it->second.entry.fptr, &status);
            }
            current_memory_ -= it->second.entry.file_size;
            cache_.erase(it);
        }
        
        lru_list_.pop_back();
    }
};

// Global cache instance
static UnifiedCache global_cache;

}  // namespace torchfits