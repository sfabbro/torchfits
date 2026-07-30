/**
 * CFITSIO handle-cache API (Option A: unused on hot paths).
 *
 * Shared fitsfile* pooling was removed: concurrent readers must not share a
 * fitsfile* (single CHDU cursor). Live shared state is SharedReadMeta + raw fd
 * (fits_detail.h). These entry points remain as stable no-ops / close helpers
 * so existing bindings and invalidate call sites keep linking.
 */

#include "cache.h"

#include <stdexcept>

namespace torchfits {

void configure_cache(size_t /*max_files*/, size_t /*max_memory_mb*/) {}

void clear_file_cache() {}

void invalidate_file_cache(const std::string& /*filepath*/) {}

size_t get_cache_size() { return 0; }

fitsfile* get_or_open_cached(const std::string& /*filepath*/) {
    // Hot paths open private handles; do not resurrect shared-handle pooling.
    throw std::runtime_error(
        "get_or_open_cached is disabled (CFITSIO Option A: private handles only)");
}

void release_cached(const std::string& /*filepath*/) {}

void invalidate_cached(const std::string& /*filepath*/) {}

}  // namespace torchfits
