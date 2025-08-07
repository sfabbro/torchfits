#include "smart_cache.h"
#include "fits_reader.h"
#include "performance.h"
#include "debug.h"
#include <algorithm>
#include <random>
#include <cmath>

namespace torchfits_cache {

// Global smart cache instance
std::unique_ptr<SmartCache> global_smart_cache = nullptr;

// === SmartCache Implementation ===

SmartCache::SmartCache(const SmartCacheConfig& config) 
    : config_(config), hits_(0), misses_(0), prefetch_hits_(0), ml_predictions_correct_(0) {
    
    // Initialize ML components
    if (config_.enable_pattern_learning) {
        pattern_learner_ = std::make_unique<AccessPatternLearner>();
    }
    
    if (config_.enable_ml_prefetching) {
        prefetch_predictor_ = std::make_unique<PrefetchPredictor>(config_);
    }
    
    // Start background worker thread
    background_thread_ = std::thread(&SmartCache::background_worker, this);
    
    DEBUG_LOG("SmartCache initialized with " << config_.max_memory_bytes / (1024*1024) << " MB capacity");
}

SmartCache::~SmartCache() {
    // Shutdown background thread
    {
        std::lock_guard<std::mutex> lock(background_mutex_);
        shutdown_ = true;
    }
    background_cv_.notify_all();
    
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
    
    DEBUG_LOG("SmartCache destroyed");
}

torch::Tensor SmartCache::get(const std::string& filename, 
                             int hdu_num,
                             const std::vector<long>& start,
                             const std::vector<long>& shape) {
    std::string key = make_cache_key(filename, hdu_num, start, shape);
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Cache hit
            it->second->last_accessed = std::chrono::steady_clock::now();
            it->second->access_count++;
            hits_++;
            
            // Record access pattern
            if (pattern_learner_) {
                update_access_patterns(key);
            }
            
            DEBUG_LOG("Cache hit for " << filename << " HDU " << hdu_num);
            return it->second->data.clone();
        }
        
        misses_++;
    }
    
    // Cache miss - load data
    DEBUG_LOG("Cache miss for " << filename << " HDU " << hdu_num << " - loading");
    
    torch::Tensor data;
    std::map<std::string, std::string> metadata;
    
    try {
        // Load data using existing reader
        if (start.empty() && shape.empty()) {
            // Full image/table read
            auto result = read_impl(py::str(filename), py::int_(hdu_num), 
                                  py::none(), py::none(), py::none(), 
                                  0, py::none(), 0, py::str("cpu"));
            
            if (py::isinstance<torch::Tensor>(result)) {
                data = result.cast<torch::Tensor>();
            } else {
                // Handle table or other data types
                DEBUG_LOG("Non-tensor data type - creating placeholder");
                data = torch::empty({1}, torch::kFloat32);
            }
        } else {
            // Partial read with start/shape
            auto result = read_impl(py::str(filename), py::int_(hdu_num), 
                                  py::cast(start), py::cast(shape), py::none(),
                                  0, py::none(), 0, py::str("cpu"));
            
            if (py::isinstance<torch::Tensor>(result)) {
                data = result.cast<torch::Tensor>();
            } else {
                data = torch::empty({1}, torch::kFloat32);
            }
        }
        
        // Store in cache
        put(key, data, metadata);
        
        // Trigger prefetching based on access patterns
        if (config_.enable_ml_prefetching && prefetch_predictor_ && pattern_learner_) {
            auto candidates = prefetch_predictor_->predict_prefetch_candidates(
                filename, *pattern_learner_, config_.prefetch_lookahead);
            
            if (!candidates.empty()) {
                // Schedule background prefetching
                std::lock_guard<std::mutex> bg_lock(background_mutex_);
                background_tasks_.push([this, candidates]() {
                    for (const auto& candidate : candidates) {
                        try {
                            get(candidate, 1); // Prefetch primary HDU
                        } catch (...) {
                            // Ignore prefetch failures
                        }
                    }
                });
                background_cv_.notify_one();
            }
        }
        
    } catch (const std::exception& e) {
        DEBUG_LOG("Error loading data for cache: " << e.what());
        throw;
    }
    
    return data;
}

void SmartCache::put(const std::string& key, 
                    const torch::Tensor& data,
                    const std::map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Check if we need to evict entries
    evict_if_needed();
    
    // Create cache entry
    auto entry = std::make_unique<CacheEntry>();
    entry->data = data.clone();
    entry->metadata = metadata;
    entry->last_accessed = std::chrono::steady_clock::now();
    entry->created = entry->last_accessed;
    entry->access_count = 1;
    entry->memory_usage = data.numel() * data.element_size();
    
    // ML features
    entry->access_frequency = 1.0;
    entry->predicted_reuse_probability = 0.5; // Default neutral probability
    
    cache_[key] = std::move(entry);
    
    DEBUG_LOG("Cached data: " << key << " (" << data.numel() * data.element_size() / 1024 << " KB)");
}

void SmartCache::prefetch(const std::vector<std::string>& filenames,
                         const std::vector<int>& hdu_nums) {
    // Schedule prefetching in background
    std::lock_guard<std::mutex> lock(background_mutex_);
    
    background_tasks_.push([this, filenames, hdu_nums]() {
        for (size_t i = 0; i < filenames.size(); ++i) {
            int hdu = (i < hdu_nums.size()) ? hdu_nums[i] : 1;
            try {
                get(filenames[i], hdu);
                prefetch_hits_++;
            } catch (const std::exception& e) {
                DEBUG_LOG("Prefetch failed for " << filenames[i] << ": " << e.what());
            }
        }
    });
    
    background_cv_.notify_one();
}

SmartCache::CacheStats SmartCache::get_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    size_t total_memory = 0;
    for (const auto& [key, entry] : cache_) {
        total_memory += entry->memory_usage;
    }
    
    CacheStats stats;
    stats.total_entries = cache_.size();
    stats.memory_usage = total_memory;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.hit_rate = (hits_ + misses_ > 0) ? 
                     static_cast<double>(hits_) / (hits_ + misses_) : 0.0;
    stats.prefetch_hits = prefetch_hits_;
    stats.ml_predictions_correct = ml_predictions_correct_;
    
    return stats;
}

void SmartCache::background_worker() {
    while (!shutdown_) {
        std::unique_lock<std::mutex> lock(background_mutex_);
        background_cv_.wait(lock, [this] { 
            return !background_tasks_.empty() || shutdown_; 
        });
        
        while (!background_tasks_.empty() && !shutdown_) {
            auto task = std::move(background_tasks_.front());
            background_tasks_.pop();
            lock.unlock();
            
            try {
                task();
            } catch (const std::exception& e) {
                DEBUG_LOG("Background task failed: " << e.what());
            }
            
            lock.lock();
        }
    }
}

void SmartCache::evict_if_needed() {
    size_t total_memory = 0;
    for (const auto& [key, entry] : cache_) {
        total_memory += entry->memory_usage;
    }
    
    if (total_memory > config_.max_memory_bytes * config_.eviction_threshold ||
        cache_.size() > config_.max_entries) {
        
        // Find victim for eviction using ML-based scoring
        CacheEntry* victim = find_victim_for_eviction();
        if (victim) {
            // Remove the victim
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.get() == victim) {
                    DEBUG_LOG("Evicting cache entry: " << it->first);
                    cache_.erase(it);
                    break;
                }
            }
        }
    }
}

CacheEntry* SmartCache::find_victim_for_eviction() {
    if (cache_.empty()) return nullptr;
    
    CacheEntry* victim = nullptr;
    double lowest_score = std::numeric_limits<double>::max();
    
    auto now = std::chrono::steady_clock::now();
    
    for (const auto& [key, entry] : cache_) {
        // Calculate eviction score (lower = more likely to evict)
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - entry->last_accessed).count();
        
        double score = entry->access_frequency 
                      + entry->predicted_reuse_probability
                      - (age / 3600.0); // Age penalty (hours)
        
        if (score < lowest_score) {
            lowest_score = score;
            victim = entry.get();
        }
    }
    
    return victim;
}

std::string SmartCache::make_cache_key(const std::string& filename, int hdu_num,
                                      const std::vector<long>& start,
                                      const std::vector<long>& shape) const {
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

void SmartCache::update_access_patterns(const std::string& key) {
    if (pattern_learner_) {
        // Extract filename from key
        size_t colon_pos = key.find(':');
        if (colon_pos != std::string::npos) {
            std::string filename = key.substr(0, colon_pos);
            // Simple HDU extraction
            int hdu_num = 1;
            
            pattern_learner_->record_access(filename, hdu_num, 
                std::chrono::steady_clock::now());
        }
    }
}

// === AccessPatternLearner Implementation ===

AccessPatternLearner::AccessPatternLearner() {
    DEBUG_LOG("AccessPatternLearner initialized");
}

AccessPatternLearner::~AccessPatternLearner() = default;

void AccessPatternLearner::record_access(const std::string& filename, int hdu_num,
                                        const std::chrono::steady_clock::time_point& timestamp) {
    std::lock_guard<std::mutex> lock(learner_mutex_);
    
    AccessEvent event{filename, hdu_num, timestamp};
    access_history_.push_back(event);
    
    // Update file frequencies
    file_frequencies_[filename]++;
    
    // Limit history size
    if (access_history_.size() > 10000) {
        access_history_.erase(access_history_.begin(), 
                             access_history_.begin() + 1000);
    }
    
    // Periodically analyze patterns
    if (access_history_.size() % 100 == 0) {
        analyze_sequences();
    }
}

std::vector<std::pair<std::string, double>> AccessPatternLearner::predict_next_accesses(
    const std::string& current_file, int current_hdu, size_t num_predictions) {
    
    std::lock_guard<std::mutex> lock(learner_mutex_);
    
    std::vector<std::pair<std::string, double>> predictions;
    
    // Simple sequence-based prediction
    auto it = sequence_patterns_.find(current_file);
    if (it != sequence_patterns_.end()) {
        std::map<std::string, double> candidates;
        
        for (const auto& next_file : it->second) {
            candidates[next_file] += 1.0;
        }
        
        // Convert to vector and sort by probability
        for (const auto& [file, count] : candidates) {
            double probability = count / it->second.size();
            predictions.emplace_back(file, probability);
        }
        
        std::sort(predictions.begin(), predictions.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (predictions.size() > num_predictions) {
            predictions.resize(num_predictions);
        }
    }
    
    return predictions;
}

void AccessPatternLearner::analyze_sequences() {
    if (access_history_.size() < 2) return;
    
    // Build sequence patterns
    for (size_t i = 1; i < access_history_.size(); ++i) {
        const auto& prev = access_history_[i-1];
        const auto& curr = access_history_[i];
        
        // Check if accesses are close in time (within 10 seconds)
        auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
            curr.timestamp - prev.timestamp).count();
        
        if (time_diff <= 10) {
            sequence_patterns_[prev.filename].push_back(curr.filename);
        }
    }
}

// === PrefetchPredictor Implementation ===

PrefetchPredictor::PrefetchPredictor(const SmartCacheConfig& config) 
    : config_(config) {
    DEBUG_LOG("PrefetchPredictor initialized");
}

PrefetchPredictor::~PrefetchPredictor() = default;

std::vector<std::string> PrefetchPredictor::predict_prefetch_candidates(
    const std::string& current_file,
    const AccessPatternLearner& learner,
    size_t max_candidates) const {
    
    auto predictions = learner.predict_next_accesses(current_file, 1, max_candidates);
    
    std::vector<std::string> candidates;
    for (const auto& [file, probability] : predictions) {
        if (probability > 0.1) { // Minimum threshold
            candidates.push_back(file);
        }
    }
    
    return candidates;
}

// === Global Functions ===

void initialize_smart_cache(const SmartCacheConfig& config) {
    if (!global_smart_cache) {
        global_smart_cache = std::make_unique<SmartCache>(config);
        DEBUG_LOG("Global smart cache initialized");
    }
}

void cleanup_smart_cache() {
    global_smart_cache.reset();
    DEBUG_LOG("Global smart cache cleaned up");
}

} // namespace torchfits_cache
