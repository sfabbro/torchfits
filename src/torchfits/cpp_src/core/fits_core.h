#pragma once

// Torch-free half of the FITS read stack.
//
// Everything declared here used to live inline in ``fits_detail.h``, which the
// torch-linked extension pulled in for ATen and nanobind. None of it needs
// either: it is CFITSIO calls, FITS text handling, and the shared per-path
// metadata cache. Moving it into libtorchfits_core lets the metadata entry
// points run from a library that never loads libtorch, and gives _C and _core
// one implementation instead of two that can drift.
//
// The namespace is unchanged (``torchfits::detail``) so every existing call
// site keeps compiling against this header.

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fitsio.h>

#include "core/core_export.h"

namespace torchfits {
namespace detail {

// The core library's own build identity, read from the loaded shared object
// rather than from the module that was compiled alongside it. A stale
// libtorchfits_core sitting next to a fresh _core.so / _C.so passes every
// compile-time check and only fails when a struct layout disagrees; this is
// the value the process actually got.
TORCHFITS_CORE_API const char* core_library_build_id();

// ---------------------------------------------------------------------------
// Header shape helpers
// ---------------------------------------------------------------------------
TORCHFITS_CORE_API bool is_unsigned_short_offset(double offset);
TORCHFITS_CORE_API bool is_unsigned_long_offset(double offset);
TORCHFITS_CORE_API void validate_image_naxis(int naxis);

// Reject GROUPS=T and read BITPIX/NAXIS/NAXISn in one pass.
TORCHFITS_CORE_API void read_image_params_9d(
    fitsfile* fptr,
    int* bitpix,
    int* naxis,
    std::array<LONGLONG, 9>& naxes,
    int* status
);

// Sign-bit XOR for the FITS signed-byte convention, parallelized by the core
// pool (at::parallel_for is not available here).
TORCHFITS_CORE_API void xor_sign_bit_u8(uint8_t* p, size_t nbytes);

// ---------------------------------------------------------------------------
// BSCALE/BZERO detection
// ---------------------------------------------------------------------------
struct ScaleDetectionResult {
    bool scaled = false;
    bool trusted = true;
    double bscale = 1.0;
    double bzero = 0.0;
};

TORCHFITS_CORE_API ScaleDetectionResult detect_scale_info_fast(fitsfile* fptr, int bitpix);

// ---------------------------------------------------------------------------
// Shared read metadata cache
// ---------------------------------------------------------------------------
// Refcounted raw file descriptor. Readers borrow a shared_ptr for the
// duration of a pread/mmap so an invalidation (file replaced/truncated) can
// reset meta->raw_fd without closing an fd that is still in flight — the fd is
// only closed when the last holder releases it.
struct TORCHFITS_CORE_API RawFdHolder {
    int fd = -1;
    explicit RawFdHolder(int f = -1) : fd(f) {}
    ~RawFdHolder();
    RawFdHolder(const RawFdHolder&) = delete;
    RawFdHolder& operator=(const RawFdHolder&) = delete;
};

struct TORCHFITS_CORE_API SharedReadMeta {
    // Rotated whenever out-of-band file changes are detected, so caches keyed
    // by uid (e.g. the per-thread HDU metadata cache in read_full_cached)
    // cannot serve entries from a previous file generation. Atomic because
    // readers sample it without holding meta->mutex.
    std::atomic<uint64_t> uid{0};
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache;
    std::unordered_map<int, bool> compressed_cache;
    std::unordered_map<int, std::tuple<bool, bool, double, double>> scale_cache;
    std::unordered_map<std::string, int> hdu_name_cache;
    // Structural table metadata, keyed by HDU. These exist so the "skinny"
    // probes (read_nrows / read_colnames / read_hdu_type / read_num_hdus) can
    // answer without re-opening the file: finding a table HDU means scanning
    // the headers that precede it, so the cost grows with header size while the
    // Python-side result stays O(1). Cleared with the other caches below.
    std::unordered_map<int, long long> nrows_cache;
    std::unordered_map<int, std::vector<std::string>> colnames_cache;
    std::unordered_map<int, std::string> hdu_type_cache;
    int num_hdus = -1;
    bool has_stat = false;
    off_t size = 0;
    int64_t mtime_ns = 0;
    ino_t inode = 0;
    int64_t last_stat_check_ns = 0;
    std::shared_ptr<RawFdHolder> raw_fd;
    std::shared_mutex mutex;
};

extern TORCHFITS_CORE_API std::mutex g_shared_meta_mutex;
extern TORCHFITS_CORE_API std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
extern TORCHFITS_CORE_API std::atomic<uint64_t> g_shared_meta_uid;

extern TORCHFITS_CORE_API const bool kValidateSharedMeta;
extern TORCHFITS_CORE_API const int64_t kSharedMetaValidateIntervalNs;

TORCHFITS_CORE_API std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename);
TORCHFITS_CORE_API int open_readonly_fd(const std::string& filename);
// Prefer fits_open_diskfile for plain local paths (skips CFITSIO extended-syntax parse).
TORCHFITS_CORE_API int open_fits_readonly(fitsfile** fptr, const std::string& path);
TORCHFITS_CORE_API std::shared_ptr<RawFdHolder> get_shared_raw_fd(
    const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename);
TORCHFITS_CORE_API bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes);
TORCHFITS_CORE_API void invalidate_shared_meta_for_path(const std::string& filename);
TORCHFITS_CORE_API void clear_shared_meta_cache();

// ---------------------------------------------------------------------------
// FITS text handling
// ---------------------------------------------------------------------------
// Lenient: drops bytes outside printable ASCII so sloppy files stay readable.
TORCHFITS_CORE_API std::string sanitize_fits_string(const std::string& input);
// Drop only bytes that cannot appear in UTF-8 (>= 0x80). nanobind converts a
// std::string to Python `str` by UTF-8 decoding, so one stray high-bit byte in
// a header made read_header_string() raise; the caller then quietly fell back
// to the raw string dict, where every value loses its type (17 -> '17').
TORCHFITS_CORE_API std::string drop_non_ascii_bytes(const std::string& input);

}  // namespace detail
}  // namespace torchfits
