#ifndef TORCHFITS_CACHE_H
#define TORCHFITS_CACHE_H

#include <list>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <map>
#include <vector>

// CacheEntry structure to store data in the cache
struct CacheEntry {
    torch::Tensor data;
    std::map<std::string, std::string> header;
    std::map<std::string, std::vector<std::string>> string_data;

    CacheEntry(torch::Tensor d, std::map<std::string, std::string> h)
        : data(std::move(d)), header(std::move(h)) {}

    // Add explicit copy constructor
    CacheEntry(const CacheEntry& other)
        : data(other.data.defined() ? other.data.clone() : torch::Tensor()),
          header(other.header),
          string_data(other.string_data) {}

    // Add assignment operator
    CacheEntry& operator=(const CacheEntry& other) {
        if (this != &other) {
            data = other.data.defined() ? other.data.clone() : torch::Tensor();
            header = other.header;
            string_data = other.string_data;
        }
        return *this;
    }

    size_t size() const {
        size_t total_size = 0;

        // Size of the data tensor
        if (data.defined() && data.numel() > 0) {
            total_size += data.nbytes();
        }

        // Size of the header (estimate based on number of key-value pairs)
        total_size += header.size() * 80; // Assuming each header card is ~80 bytes

        // Size of string data
        for (const auto& [col_name, string_list] : string_data) {
            for (const auto& str : string_list) {
                total_size += str.size(); // Size of each string
                total_size += 1; // Null terminator
            }
        }

        return total_size; // Return size in bytes
    }
};

// LRU Cache for FITS data
class LRUCache {
public:
    explicit LRUCache(size_t capacity_mb = 256);

    // Accessors
    size_t capacity_bytes() const;
    size_t size_bytes() const;

    // Operations
    void clear();
    void put(const std::string& key, const std::shared_ptr<CacheEntry>& entry);
    std::shared_ptr<CacheEntry> get(const std::string& key);

private:
    size_t capacity_bytes_;       // Max size in bytes
    size_t current_size_bytes_;   // Current size in bytes
    std::mutex mutex_;         // Thread safety

    // Cache storage structure
    using ListEntry = std::pair<std::string, std::shared_ptr<CacheEntry>>;
    std::list<ListEntry> cache_list_;  // MRU to LRU order
    std::unordered_map<std::string, typename std::list<ListEntry>::iterator> cache_map_;  // Key to list iterator
};

// Global cache instance
extern std::unique_ptr<LRUCache> cache;

// Initialize cache with appropriate capacity (simplified)
void ensure_cache_initialized(size_t capacity_mb);

#endif // TORCHFITS_CACHE_H
