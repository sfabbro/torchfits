#include "real_cache.h"
#include "debug.h"
#include <algorithm>

namespace torchfits_real_cache {

/// Global cache instance definition
std::unique_ptr<RealSmartCache> global_real_cache = nullptr;

/// Global singleton instance
RealSmartCache& RealSmartCache::get_instance() {
    static RealSmartCache instance;
    return instance;
}

RealSmartCache::RealSmartCache(size_t max_memory_mb) 
    : max_memory_bytes_(max_memory_mb * 1024 * 1024), hits_(0), misses_(0) {
    DEBUG_LOG("RealSmartCache initialized with " + std::to_string(max_memory_mb) + " MB capacity");
}

RealSmartCache::~RealSmartCache() {
    DEBUG_LOG("RealSmartCache destroyed");
}

std::optional<torch::Tensor> RealSmartCache::try_get(const std::string& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Cache hit
        it->second->last_accessed = std::chrono::steady_clock::now();
        it->second->access_count++;
        hits_++;
        
        DEBUG_LOG("Cache HIT for " + key);
        return it->second->data.clone();
    }
    
    // Cache miss
    misses_++;
    DEBUG_LOG("Cache MISS for " + key);
    return std::nullopt; // Return empty optional
}

void RealSmartCache::put(const std::string& key, const torch::Tensor& data) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check if already exists
    if (cache_.find(key) != cache_.end()) {
        return; // Already cached
    }
    
    // Create new entry
    auto entry = std::make_unique<RealCacheEntry>();
    entry->data = data.clone();
    entry->last_accessed = std::chrono::steady_clock::now();
    entry->access_count = 1;
    entry->memory_usage = data.numel() * data.element_size();
    
    // Evict if necessary before adding
    evict_if_needed();
    
    // Add to cache
    cache_[key] = std::move(entry);
    
    DEBUG_LOG("Cached " + key + " (" + std::to_string(data.numel() * data.element_size() / 1024) + " KB)");
}

void RealSmartCache::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    hits_ = 0;
    misses_ = 0;
    DEBUG_LOG("Cache cleared");
}

RealSmartCache::Stats RealSmartCache::get_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    Stats stats;
    stats.total_entries = cache_.size();
    stats.memory_usage_mb = calculate_current_memory() / (1024 * 1024);
    stats.hits = hits_;
    stats.misses = misses_;
    stats.hit_rate = (hits_ + misses_ > 0) ? 
                     static_cast<double>(hits_) / (hits_ + misses_) : 0.0;
    
    return stats;
}

void RealSmartCache::evict_if_needed() {
    size_t current_memory = calculate_current_memory();
    
    // Evict if we're over 80% capacity
    while (current_memory > max_memory_bytes_ * 0.8 && !cache_.empty()) {
        std::string lru_key = find_lru_key();
        if (!lru_key.empty()) {
            auto it = cache_.find(lru_key);
            if (it != cache_.end()) {
                DEBUG_LOG("Evicting " + lru_key + " (LRU)");
                current_memory -= it->second->memory_usage;
                cache_.erase(it);
            }
        } else {
            break; // Safety check
        }
    }
}

std::string RealSmartCache::find_lru_key() const {
    if (cache_.empty()) return "";
    
    auto oldest_time = std::chrono::steady_clock::now();
    std::string lru_key;
    
    for (const auto& [key, entry] : cache_) {
        if (entry->last_accessed < oldest_time) {
            oldest_time = entry->last_accessed;
            lru_key = key;
        }
    }
    
    return lru_key;
}

size_t RealSmartCache::calculate_current_memory() const {
    size_t total = 0;
    for (const auto& [key, entry] : cache_) {
        total += entry->memory_usage;
    }
    return total;
}

void initialize_real_cache(size_t max_memory_mb) {
    if (!global_real_cache) {
        global_real_cache = std::make_unique<RealSmartCache>(max_memory_mb);
        DEBUG_LOG("Global real cache initialized");
    }
}

std::string make_cache_key(const std::string& filename, int hdu_num,
                          const std::vector<long>& start,
                          const std::vector<long>& shape) {
    std::string key = filename + ":" + std::to_string(hdu_num);
    
    if (!start.empty()) {
        key += ":start=";
        for (size_t i = 0; i < start.size(); ++i) {
            if (i > 0) key += ",";
            key += std::to_string(start[i]);
        }
    }
    
    if (!shape.empty()) {
        key += ":shape=";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) key += ",";
            key += std::to_string(shape[i]);
        }
    }
    
    return key;
}

} // namespace torchfits_real_cache
