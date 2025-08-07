#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>
#include <memory>
#include <optional>

// === REAL SMART CACHE IMPLEMENTATION ===
// Actually integrated into the reading pipeline

namespace torchfits_real_cache {

/// Simple but effective smart cache entry
struct RealCacheEntry {
    torch::Tensor data;
    std::chrono::steady_clock::time_point last_accessed;
    size_t access_count = 0;
    size_t memory_usage = 0;
};

/// Actually working smart cache integrated into reads
class RealSmartCache {
public:
    // Singleton access
    static RealSmartCache& get_instance();
    
    RealSmartCache(size_t max_memory_mb = 512);
    ~RealSmartCache();
    
    /// Try to get from cache - returns optional tensor
    std::optional<torch::Tensor> try_get(const std::string& key);
    
    /// Put into cache with automatic eviction
    void put(const std::string& key, const torch::Tensor& data);
    
    /// Clear cache
    void clear();
    
    /// Get statistics
    struct Stats {
        size_t total_entries = 0;
        size_t memory_usage_mb = 0;
        size_t hits = 0;
        size_t misses = 0;
        double hit_rate = 0.0;
    };
    
    Stats get_stats() const;

private:
    size_t max_memory_bytes_;
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, std::unique_ptr<RealCacheEntry>> cache_;
    
    // Statistics
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
    
    void evict_if_needed();
    std::string find_lru_key() const;
    size_t calculate_current_memory() const;
};

/// Global cache instance
extern std::unique_ptr<RealSmartCache> global_real_cache;

/// Initialize real cache
void initialize_real_cache(size_t max_memory_mb = 512);

/// Make cache key for file/hdu combination
std::string make_cache_key(const std::string& filename, int hdu_num,
                          const std::vector<long>& start = {},
                          const std::vector<long>& shape = {});

} // namespace torchfits_real_cache
