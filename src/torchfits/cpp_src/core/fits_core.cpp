// Implementation of the torch-free FITS primitives declared in fits_core.h.
//
// The bodies were moved verbatim out of fits_detail.h (which still includes
// this header) so the metadata path and the tensor path cannot drift apart.
// Only two things changed mechanically: `inline` is gone (these are now
// out-of-line exports of libtorchfits_core), and the sign-bit XOR uses the
// core thread pool instead of at::parallel_for, which libtorchfits_core does
// not link.

#include "core/fits_core.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/parallel.h"
#include "internal_utils.h"
#include "security.h"

namespace torchfits {
namespace detail {

#ifndef TORCHFITS_CORE_BUILD_ID
#define TORCHFITS_CORE_BUILD_ID "unknown"
#endif

const char* core_library_build_id() {
    return TORCHFITS_CORE_BUILD_ID;
}

RawFdHolder::~RawFdHolder() {
    if (fd != -1) ::close(fd);
}

// FITS stores BZERO/TZERO as ASCII decimals; writers may serialize the
// canonical unsigned offsets with rounding error (e.g. 32767.99999 after a
// float32 round-trip). Match them within the same tight tolerance the table
// reader uses, so image and table paths agree on the unsigned conventions.
bool is_unsigned_short_offset(double offset) {
    return std::abs(offset - 32768.0) < 1e-5;
}
bool is_unsigned_long_offset(double offset) {
    return std::abs(offset - 2147483648.0) < 1e-5;
}

void validate_image_naxis(int naxis) {
    if (naxis < 0 || naxis > 9) {
        throw std::runtime_error(
            "torchfits supports FITS images with at most 9 axes; got NAXIS=" +
            std::to_string(naxis));
    }
}

void read_image_params_9d(
    fitsfile* fptr,
    int* bitpix,
    int* naxis,
    std::array<LONGLONG, 9>& naxes,
    int* status
) {
    // Random Groups data (GROUPS=T with PCOUNT/GCOUNT) is not supported:
    // decoding it as an ordinary image would silently return wrong-shaped
    // values. Fail loudly instead of guessing. Compressed-image HDUs are
    // unaffected — cfitsio virtualizes them as plain images and they never
    // carry a GROUPS keyword.
    {
        int groups_status = 0;
        int groups_val = 0;
        fits_read_key(fptr, TLOGICAL, "GROUPS", &groups_val, nullptr, &groups_status);
        if (groups_status == 0 && groups_val != 0) {
            throw std::runtime_error(
                "Random Groups FITS images (GROUPS=T) are not supported");
        }
    }
    int declared_naxis = 0;
    fits_read_key(fptr, TINT, "NAXIS", &declared_naxis, nullptr, status);
    if (*status != 0) return;
    validate_image_naxis(declared_naxis);
    fits_get_img_paramll(fptr, 9, bitpix, naxis, naxes.data(), status);
}

// ---------------------------------------------------------------------------
// Sign-bit XOR for signed-byte encoding
// ---------------------------------------------------------------------------
void xor_sign_bit_u8(uint8_t* p, size_t nbytes) {
    if (!p || nbytes == 0) return;
    static const size_t kParallelMinBytes = []() -> size_t {
        constexpr int64_t kDefault = 1 << 18;
        int64_t parsed = kDefault;
        if (const char* v = std::getenv("TORCHFITS_XOR_PARALLEL_MIN_BYTES")) {
            try { parsed = std::stoll(std::string(v)); } catch (...) { parsed = kDefault; }
        }
        return parsed <= 0 ? 1 : static_cast<size_t>(parsed);
    }();

    auto xor_block = [](uint8_t* ptr, size_t len) {
        if (!ptr || len == 0) return;
        constexpr uint64_t kMask64 = 0x8080808080808080ULL;
        size_t i = 0;
        while (i < len && ((reinterpret_cast<uintptr_t>(ptr + i) & 7u) != 0u)) {
            ptr[i] ^= 0x80;
            ++i;
        }
        uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr + i);
        const size_t n64 = (len - i) / sizeof(uint64_t);
        for (size_t j = 0; j < n64; ++j) p64[j] ^= kMask64;
        i += n64 * sizeof(uint64_t);
        while (i < len) { ptr[i] ^= 0x80; ++i; }
    };

    if (nbytes < kParallelMinBytes) {
        xor_block(p, nbytes);
        return;
    }
    core::parallel_for(0, static_cast<int64_t>(nbytes), 1 << 20, [=](int64_t begin, int64_t end) {
        xor_block(p + begin, static_cast<size_t>(end - begin));
    });
}

ScaleDetectionResult detect_scale_info_fast(fitsfile* fptr, int bitpix) {
    ScaleDetectionResult out;
    if (!fptr || bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) return out;

    // BLANK means undefined pixels. Identity BSCALE/BZERO is otherwise the
    // unscaled-int16 fast path, which cannot hold NaN — promote to scaled
    // float so nulval=NaN applies (quantize + integer images with BLANK).
    int blank_status = 0;
    long blank_val = 0;
    fits_read_key(fptr, TLONG, "BLANK", &blank_val, nullptr, &blank_status);
    const bool has_blank = (blank_status == 0);

    int equiv_status = 0;
    int equiv_type = bitpix;
    fits_get_img_equivtype(fptr, &equiv_type, &equiv_status);
    if (equiv_status == 0) {
        // CFITSIO has no unsigned 64-bit convention: fits_get_img_equivtype
        // reports TLONGLONG for BITPIX=64 regardless of BZERO, so the shortcut
        // would hide a uint64 convention (BZERO=2^63). Always read BSCALE/BZERO
        // for LONGLONG_IMG. BLANK also skips the shortcut so we still load
        // BSCALE/BZERO before marking scaled.
        if (equiv_type == bitpix && bitpix != LONGLONG_IMG && !has_blank) return out;
        if (bitpix == BYTE_IMG && equiv_type == SBYTE_IMG) {
            out.scaled = true; out.bscale = 1.0; out.bzero = -128.0;
            return out;
        }
    }
    double bscale = 1.0;
    double bzero = 0.0;
    int s1 = 0;
    fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, nullptr, &s1);
    if (s1 == 0) {
        out.bscale = bscale;
        if (bscale != 1.0) out.scaled = true;
    } else if (s1 != KEY_NO_EXIST) {
        out.scaled = true; out.trusted = false;
    }
    int s2 = 0;
    fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, nullptr, &s2);
    if (s2 == 0) {
        out.bzero = bzero;
        if (bzero != 0.0) out.scaled = true;
    } else if (s2 != KEY_NO_EXIST) {
        out.scaled = true; out.trusted = false;
    }
    if (equiv_status == 0 && equiv_type != bitpix) out.scaled = true;
    if (has_blank) out.scaled = true;
    return out;
}

std::mutex g_shared_meta_mutex;
std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
std::atomic<uint64_t> g_shared_meta_uid{1};

const bool kValidateSharedMeta = []() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_SHARED_META_VALIDATE");
}();

const int64_t kSharedMetaValidateIntervalNs = []() {
    constexpr int64_t kDefaultMs = 1000;
    return torchfits::internal::env_nonnegative_int(
        "TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS", kDefaultMs) * 1000000LL;
}();

std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
    bool can_stat = kValidateSharedMeta && !has_cfitsio_extended_filename_syntax(filename);
    std::shared_ptr<SharedReadMeta> meta;
    {
        std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
        auto it = g_shared_meta.find(filename);
        if (it == g_shared_meta.end()) {
            meta = std::make_shared<SharedReadMeta>();
            meta->uid.store(g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed),
                            std::memory_order_relaxed);
            g_shared_meta.emplace(filename, meta);
        } else {
            meta = it->second;
        }
    }
    if (!can_stat) return meta;
    const int64_t now_ns = torchfits::internal::monotonic_now_ns();
    std::unique_lock<std::shared_mutex> meta_lock(meta->mutex);
    if (kSharedMetaValidateIntervalNs > 0 && meta->last_stat_check_ns != 0 &&
        (now_ns - meta->last_stat_check_ns) < kSharedMetaValidateIntervalNs) {
        return meta;
    }
    meta->last_stat_check_ns = now_ns;
    struct stat st {};
    if (stat(filename.c_str(), &st) == 0) {
        int64_t cur_mtime_ns = torchfits::internal::mtime_ns_from_stat(st);
        if (!meta->has_stat || meta->size != st.st_size ||
            meta->mtime_ns != cur_mtime_ns || meta->inode != st.st_ino) {
            // Drop the current fd. In-flight readers hold their own shared_ptr,
            // so the close is deferred until they release it — a pread/mmap can
            // never race a close of the same fd.
            meta->raw_fd.reset();
            meta->image_info_cache.clear();
            meta->compressed_cache.clear();
            meta->scale_cache.clear();
            // hdu_name_cache is keyed by EXTNAME alone, so it must be dropped
            // with the rest: a rewrite can move a name to a different index
            // (or reuse it for another HDU), and a surviving entry would make
            // resolution silently read the wrong extension.
            meta->hdu_name_cache.clear();
            meta->nrows_cache.clear();
            meta->colnames_cache.clear();
            meta->hdu_type_cache.clear();
            meta->num_hdus = -1;
            // Rotate identity so per-thread caches keyed by {uid, hdu} cannot
            // pair stale shape/dtype/scale metadata with the replaced file
            // (R7-CPP2). In-flight readers keep their own shared_ptr and the
            // data they already decoded from the previous generation.
            meta->uid.store(g_shared_meta_uid.fetch_add(1, std::memory_order_relaxed),
                            std::memory_order_relaxed);
            meta->has_stat = true;
            meta->size = st.st_size;
            meta->mtime_ns = cur_mtime_ns;
            meta->inode = st.st_ino;
        }
    }
    return meta;
}

int open_readonly_fd(const std::string& filename) {
#ifdef O_CLOEXEC
    int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != -1) return fd;
#endif
    return ::open(filename.c_str(), O_RDONLY);
}

// Prefer fits_open_diskfile for plain local paths (skips CFITSIO extended-syntax parse).
int open_fits_readonly(fitsfile** fptr, const std::string& path) {
    int status = 0;
    if (has_cfitsio_extended_filename_syntax(path) || path.find("://") != std::string::npos) {
        fits_open_file(fptr, path.c_str(), 0 /* READONLY */, &status);
    } else {
        fits_open_diskfile(fptr, path.c_str(), 0 /* READONLY */, &status);
    }
    return status;
}

std::shared_ptr<RawFdHolder> get_shared_raw_fd(
    const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename) {
    if (!meta || has_cfitsio_extended_filename_syntax(filename)) return nullptr;
    std::unique_lock<std::shared_mutex> lock(meta->mutex);
    if (!meta->raw_fd) {
        meta->raw_fd = std::make_shared<RawFdHolder>(open_readonly_fd(filename));
    }
    // Open-time snapshot (r9b-06): the fd must describe the same file
    // generation the shared stat snapshot does. A replacement that has not
    // been observed yet (or a lazy open after one) would otherwise hand
    // another file's bytes to readers holding old-generation metadata.
    // Callers fall back to their own CFITSIO handle when this returns null.
    if (meta->raw_fd && meta->raw_fd->fd != -1 && meta->has_stat) {
        struct stat st {};
        if (fstat(meta->raw_fd->fd, &st) != 0 ||
            st.st_ino != meta->inode ||
            static_cast<off_t>(st.st_size) != meta->size ||
            torchfits::internal::mtime_ns_from_stat(st) != meta->mtime_ns) {
            return nullptr;
        }
    }
    return meta->raw_fd;
}

bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes) {
    if (fd == -1 || !dst_void || nbytes == 0) return false;
    // Hint the kernel to start async page-in before the synchronous pread loop.
    // This overlaps I/O with any preceding computation on the calling thread.
#if defined(__linux__)
    (void)::posix_fadvise(fd, offset, static_cast<off_t>(nbytes), POSIX_FADV_WILLNEED);
#endif
    uint8_t* dst = static_cast<uint8_t*>(dst_void);
    size_t remaining = nbytes;
    off_t off = offset;
    while (remaining > 0) {
        ssize_t got = ::pread(fd, dst, remaining, off);
        if (got < 0) { if (errno == EINTR) continue; break; }
        if (got == 0) break;
        dst += static_cast<size_t>(got);
        off += static_cast<off_t>(got);
        remaining -= static_cast<size_t>(got);
    }
    if (remaining == 0) return true;
    struct stat sb {};
    if (fstat(fd, &sb) != 0) return false;
    if (offset < 0) return false;
    if (static_cast<size_t>(sb.st_size) < static_cast<size_t>(offset) + nbytes) return false;
    static const long kPageSize = sysconf(_SC_PAGESIZE);
    const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
    const off_t page_offset = offset & ~page_mask;
    const size_t map_len = static_cast<size_t>(nbytes + (offset - page_offset));
    void* map_ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, page_offset);
    if (map_ptr == MAP_FAILED) return false;
    const uint8_t* src = static_cast<const uint8_t*>(map_ptr) + (offset - page_offset);
    std::memcpy(dst_void, src, nbytes);
    munmap(map_ptr, map_len);
    return true;
}

void invalidate_shared_meta_for_path(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.erase(filename);
}

void clear_shared_meta_cache() {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.clear();
}

std::string sanitize_fits_string(const std::string& input) {
    std::string output = input;
    output.erase(std::remove_if(output.begin(), output.end(), [](unsigned char c) {
        return c < 32 || c > 126;
    }), output.end());
    return output;
}

// Drop only the bytes that cannot appear in UTF-8 (>= 0x80), leaving every
// card's structure intact. nanobind converts a std::string to Python `str` by
// UTF-8 decoding, so one stray high-bit byte in a header made
// read_header_string() raise; the caller then quietly fell back to the raw
// string dict, where every value loses its type (17 -> '17', T -> 'T'). Card
// text is written in printable ASCII, so a byte >= 0x80 is always corruption,
// and the read side already drops exactly such bytes from header values via
// sanitize_fits_string().
std::string drop_non_ascii_bytes(const std::string& input) {
    if (std::none_of(input.begin(), input.end(), [](unsigned char c) {
            return c >= 0x80;
        })) {
        return input;
    }
    std::string output;
    output.reserve(input.size());
    for (unsigned char c : input) {
        if (c < 0x80) output.push_back(static_cast<char>(c));
    }
    return output;
}

}  // namespace detail
}  // namespace torchfits
