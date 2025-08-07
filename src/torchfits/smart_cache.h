#pragma once

#include <torch/torch.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>
#include <queue>
#include <functional>
#include <thread>
#include <condition_variable>

// === Phase 3: Smart Caching System ===
// Intelligent data handling with ML-optimized prefetching

namespace torchfits_cache {

/// Cache entry for FITS data
struct CacheEntry {
    torch::Tensor data;
    std::map<std::string, std::string> metadata;
    std::chrono::steady_clock::time_point last_accessed;
    std::chrono::steady_clock::time_point created;
    size_t access_count = 0;
    size_t memory_usage = 0;
    
    // ML features for intelligent caching
    double access_frequency = 0.0;
    double predicted_reuse_probability = 0.0;
    std::vector<std::string> access_patterns;
};

/// Smart cache configuration
struct SmartCacheConfig {
    size_t max_memory_bytes = 2ULL * 1024 * 1024 * 1024; // 2GB default
    size_t max_entries = 1000;
    bool enable_ml_prefetching = true;
    bool enable_compression = true;
    bool enable_pattern_learning = true;
    double eviction_threshold = 0.8; // Evict when 80% full
    size_t prefetch_lookahead = 10;  // How many files to prefetch ahead
};

/// Intelligent FITS data cache with ML-optimized prefetching
class SmartCache {
public:
    SmartCache(const SmartCacheConfig& config = {});
    ~SmartCache();
    
    /// Get data from cache or load if not present
    torch::Tensor get(const std::string& filename, 
                     int hdu_num = 1,
                     const std::vector<long>& start = {},
                     const std::vector<long>& shape = {});
    
    /// Put data into cache
    void put(const std::string& key, 
            const torch::Tensor& data,
            const std::map<std::string, std::string>& metadata = {});
    
    /// Prefetch data based on access patterns
    void prefetch(const std::vector<std::string>& filenames,
                 const std::vector<int>& hdu_nums = {});
    
    /// Clear cache or specific entries
    void clear();
    void remove(const std::string& key);
    
    /// Cache statistics and management
    struct CacheStats {
        size_t total_entries;
        size_t memory_usage;
        size_t hits;
        size_t misses;
        double hit_rate;
        size_t prefetch_hits;
        size_t ml_predictions_correct;
    };
    
    CacheStats get_stats() const;
    void optimize_cache(); // ML-based cache optimization
    
private:
    SmartCacheConfig config_;
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, std::unique_ptr<CacheEntry>> cache_;
    
    // Statistics
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
    mutable size_t prefetch_hits_ = 0;
    mutable size_t ml_predictions_correct_ = 0;
    
    // ML and pattern learning
    std::unique_ptr<class AccessPatternLearner> pattern_learner_;
    std::unique_ptr<class PrefetchPredictor> prefetch_predictor_;
    
    // Background operations
    std::thread background_thread_;
    std::queue<std::function<void()>> background_tasks_;
    std::mutex background_mutex_;
    std::condition_variable background_cv_;
    bool shutdown_ = false;
    
    void background_worker();
    void evict_if_needed();
    std::string make_cache_key(const std::string& filename, int hdu_num, 
                              const std::vector<long>& start, 
                              const std::vector<long>& shape) const;
    
    CacheEntry* find_victim_for_eviction();
    void update_access_patterns(const std::string& key);
};

/// Learn access patterns for intelligent prefetching
class AccessPatternLearner {
public:
    AccessPatternLearner();
    ~AccessPatternLearner();
    
    /// Record an access event
    void record_access(const std::string& filename, int hdu_num,
                      const std::chrono::steady_clock::time_point& timestamp);
    
    /// Predict next likely accesses based on current access
    std::vector<std::pair<std::string, double>> predict_next_accesses(
        const std::string& current_file, int current_hdu, size_t num_predictions = 5);
    
    /// Get sequential access probability
    double get_sequential_probability(const std::string& filename) const;
    
    /// Train the pattern learning model
    void train_model();

private:
    struct AccessEvent {
        std::string filename;
        int hdu_num;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<AccessEvent> access_history_;
    std::unordered_map<std::string, std::vector<std::string>> sequence_patterns_;
    std::unordered_map<std::string, double> file_frequencies_;
    mutable std::mutex learner_mutex_;
    
    void analyze_sequences();
    void update_transition_matrix();
};

/// ML-based prefetch prediction
class PrefetchPredictor {
public:
    PrefetchPredictor(const SmartCacheConfig& config);
    ~PrefetchPredictor();
    
    /// Predict which files should be prefetched
    std::vector<std::string> predict_prefetch_candidates(
        const std::string& current_file,
        const AccessPatternLearner& learner,
        size_t max_candidates = 5) const;
    
    /// Update prediction model based on actual usage
    void update_model(const std::string& predicted_file, 
                     const std::string& actual_file,
                     bool was_useful);
    
    /// Get prefetch priority for a file
    double get_prefetch_priority(const std::string& filename,
                               const AccessPatternLearner& learner) const;

private:
    SmartCacheConfig config_;
    
    // Simple ML model for prefetch prediction
    struct PredictionModel {
        std::unordered_map<std::string, double> file_weights;
        std::unordered_map<std::string, std::vector<double>> feature_weights;
        double learning_rate = 0.01;
    } model_;
    
    mutable std::mutex predictor_mutex_;
    
    std::vector<double> extract_features(const std::string& filename,
                                       const AccessPatternLearner& learner) const;
    void train_on_feedback(const std::vector<double>& features, 
                          bool was_useful);
};

/// Torch-Frame integration for advanced data handling
class TorchFrameIntegration {
public:
    /// Convert FITS table to torch-frame compatible format
    static py::object fits_to_torch_frame(const std::string& filename, int hdu_num = 2);
    
    /// Convert torch-frame data back to FITS table
    static void torch_frame_to_fits(const py::object& torch_frame,
                                   const std::string& filename,
                                   bool overwrite = false);
    
    /// Optimize data types for torch-frame usage
    static py::object optimize_for_torch_frame(const py::object& torch_frame);
    
    /// Smart column type inference
    static std::map<std::string, std::string> infer_column_types(
        const py::dict& table_data);
};

/// Global smart cache instance
extern std::unique_ptr<SmartCache> global_smart_cache;

/// Initialize smart caching system
void initialize_smart_cache(const SmartCacheConfig& config = {});

/// Cleanup smart caching system
void cleanup_smart_cache();

} // namespace torchfits_cache
