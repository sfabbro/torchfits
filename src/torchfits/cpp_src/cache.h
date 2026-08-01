/**
 * Cache API retained for bindings after Option A (private per-call handles).
 *
 * Shared fitsfile* LRU is gone. invalidate/clear are no-ops here; callers also
 * invoke invalidate_shared_meta for live SharedReadMeta state.
 */

#pragma once

#include <string>
#include <fitsio.h>

namespace torchfits {

void configure_cache(size_t max_files, size_t max_memory_mb);
void clear_file_cache();
void invalidate_file_cache(const std::string& filepath);
size_t get_cache_size();
fitsfile* get_or_open_cached(const std::string& filepath);
void release_cached(const std::string& filepath);
void invalidate_cached(const std::string& filepath);

// RAII close for a privately owned fitsfile*. Move-only: a copied guard would
// close the same handle twice.
struct FitsHandleGuard {
    fitsfile* fptr = nullptr;

    FitsHandleGuard() = default;
    FitsHandleGuard(const FitsHandleGuard&) = delete;
    FitsHandleGuard& operator=(const FitsHandleGuard&) = delete;

    FitsHandleGuard(FitsHandleGuard&& other) noexcept : fptr(other.fptr) {
        other.fptr = nullptr;
    }
    FitsHandleGuard& operator=(FitsHandleGuard&& other) noexcept {
        if (this != &other) {
            release();
            fptr = other.fptr;
            other.fptr = nullptr;
        }
        return *this;
    }

    ~FitsHandleGuard() { release(); }

    void release() {
        if (!fptr) return;
        int status = 0;
        fits_close_file(fptr, &status);
        fptr = nullptr;
    }
};

} // namespace torchfits
