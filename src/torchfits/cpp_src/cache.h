/**
 * Multi-level caching system header
 *
 * Implements L1 (memory) and L2 (disk) caching for FITS data
 */

#pragma once

#include <string>
#include <memory>
#include <fitsio.h>

namespace torchfits {

// Cache initialization/configuration
void configure_cache(size_t max_files, size_t max_memory_mb);
void clear_file_cache();
size_t get_cache_size();
fitsfile* get_or_open_cached(const std::string& filepath);
void release_cached(const std::string& filepath);
void invalidate_cached(const std::string& filepath);

// Cache initialization (old API if any remains)
void init_cache(size_t memory_limit_mb, const std::string& disk_cache_dir = "");
void clear_cache();

class CacheManager {
public:
    static CacheManager& instance();

    void set_memory_limit(size_t limit_mb);
    void set_disk_cache_dir(const std::string& dir);
    void clear_all();

private:
    CacheManager() = default;
    size_t memory_limit_ = 0;
    std::string disk_cache_dir_;
};

} // namespace torchfits
