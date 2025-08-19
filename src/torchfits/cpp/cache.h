/**
 * Multi-level caching system header
 * 
 * Implements L1 (memory) and L2 (disk) caching for FITS data
 */

#pragma once

#include <string>
#include <memory>

// Cache initialization
void init_cache(size_t memory_limit_mb, const std::string& disk_cache_dir = "");

// Cache operations
void clear_cache();

// Internal cache management (would be expanded in full implementation)
class CacheManager {
public:
    static CacheManager& instance();
    
    void set_memory_limit(size_t limit_mb);
    void set_disk_cache_dir(const std::string& dir);
    void clear_all();
    
private:
    CacheManager() = default;
    size_t memory_limit_;
    std::string disk_cache_dir_;
};