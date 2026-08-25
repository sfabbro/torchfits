#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <array>
#include <tuple>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif
#include <ATen/Parallel.h>
#include <fitsio.h>

#include "internal_utils.h"
#include "hardware.h"
#include "security.h"

namespace torchfits {
namespace detail {

// FITS stores BZERO/TZERO as ASCII decimals; writers may serialize the
// canonical unsigned offsets with rounding error (e.g. 32767.99999 after a
// float32 round-trip). Match them within the same tight tolerance the table
// reader uses, so image and table paths agree on the unsigned conventions.
inline bool is_unsigned_short_offset(double offset) {
    return std::abs(offset - 32768.0) < 1e-5;
}
inline bool is_unsigned_long_offset(double offset) {
    return std::abs(offset - 2147483648.0) < 1e-5;
}

inline void validate_image_naxis(int naxis) {
    if (naxis < 0 || naxis > 9) {
        throw std::runtime_error(
            "torchfits supports FITS images with at most 9 axes; got NAXIS=" +
            std::to_string(naxis));
    }
}

// Checked NAXISn product — unsigned multiply with overflow detection so a
// pathological header cannot wrap LONGLONG and under-allocate a read buffer.
inline LONGLONG checked_nelements_product(const LONGLONG* naxes, int naxis) {
    if (naxis <= 0) {
        return 0;
    }
    unsigned long long product = 1ULL;
    for (int i = 0; i < naxis; ++i) {
        const LONGLONG dim = naxes[i];
        if (dim < 0) {
            throw std::runtime_error("NAXIS product overflow: negative axis length");
        }
        if (dim == 0) {
            return 0;
        }
        const unsigned long long udim = static_cast<unsigned long long>(dim);
        if (product > std::numeric_limits<unsigned long long>::max() / udim) {
            throw std::runtime_error("NAXIS product overflow");
        }
        product *= udim;
        // Leave headroom for element size up to 16 bytes (complex128).
        if (product > static_cast<unsigned long long>(std::numeric_limits<LONGLONG>::max() / 16)) {
            throw std::runtime_error("NAXIS product overflow");
        }
    }
    return static_cast<LONGLONG>(product);
}

template <typename DimT>
inline LONGLONG checked_nelements_product(const std::vector<DimT>& naxes) {
    if (naxes.empty()) {
        return 0;
    }
    std::array<LONGLONG, 9> buf{};
    if (naxes.size() > 9) {
        throw std::runtime_error("NAXIS product overflow: too many axes");
    }
    for (size_t i = 0; i < naxes.size(); ++i) {
        buf[i] = static_cast<LONGLONG>(naxes[i]);
    }
    return checked_nelements_product(buf.data(), static_cast<int>(naxes.size()));
}

inline void read_image_params_9d(
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
inline void _xor_sign_bit_u8(uint8_t* p, size_t nbytes) {
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
    at::parallel_for(0, static_cast<int64_t>(nbytes), 1 << 20, [&](int64_t begin, int64_t end) {
        xor_block(p + begin, static_cast<size_t>(end - begin));
    });
}

// ---------------------------------------------------------------------------
// Scale detection
// ---------------------------------------------------------------------------
struct ScaleDetectionResult {
    bool scaled = false;
    bool trusted = true;
    double bscale = 1.0;
    double bzero = 0.0;
};

inline ScaleDetectionResult detect_scale_info_fast(fitsfile* fptr, int bitpix) {
    ScaleDetectionResult out;
    if (!fptr || bitpix == FLOAT_IMG || bitpix == DOUBLE_IMG) return out;
    int equiv_status = 0;
    int equiv_type = bitpix;
    fits_get_img_equivtype(fptr, &equiv_type, &equiv_status);
    if (equiv_status == 0) {
        // CFITSIO has no unsigned 64-bit convention: fits_get_img_equivtype
        // reports TLONGLONG for BITPIX=64 regardless of BZERO, so the shortcut
        // would hide a uint64 convention (BZERO=2^63). Always read BSCALE/BZERO
        // for LONGLONG_IMG.
        if (equiv_type == bitpix && bitpix != LONGLONG_IMG) return out;
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
    return out;
}

// ---------------------------------------------------------------------------
// ResolvedFITSMeta — flattened metadata for canonical image read
// ---------------------------------------------------------------------------
struct ResolvedFITSMeta {
    int bitpix = 0;
    int naxis = 0;
    std::array<LONGLONG, 9> naxes_ll{};
    bool scaled = false;
    double bscale = 1.0;
    double bzero = 0.0;
    bool compressed = false;
};

// ---------------------------------------------------------------------------
// Shared read metadata cache
// ---------------------------------------------------------------------------
// Refcounted raw file descriptor. Readers borrow a shared_ptr for the
// duration of a pread/mmap so an invalidation (file replaced/truncated) can
// reset meta->raw_fd without closing an fd that is still in flight — the fd is
// only closed when the last holder releases it.
struct RawFdHolder {
    int fd = -1;
    explicit RawFdHolder(int f = -1) : fd(f) {}
    ~RawFdHolder() {
        if (fd != -1) ::close(fd);
    }
    RawFdHolder(const RawFdHolder&) = delete;
    RawFdHolder& operator=(const RawFdHolder&) = delete;
};

struct SharedReadMeta {
    // Rotated whenever out-of-band file changes are detected, so caches keyed
    // by uid (e.g. the per-thread HDU metadata cache in read_full_cached)
    // cannot serve entries from a previous file generation. Atomic because
    // readers sample it without holding meta->mutex.
    std::atomic<uint64_t> uid{0};
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache;
    std::unordered_map<int, bool> compressed_cache;
    std::unordered_map<int, std::tuple<bool, bool, double, double>> scale_cache;
    std::unordered_map<std::string, int> hdu_name_cache;
    bool has_stat = false;
    off_t size = 0;
    int64_t mtime_ns = 0;
    ino_t inode = 0;
    int64_t last_stat_check_ns = 0;
    std::shared_ptr<RawFdHolder> raw_fd;
    // Absolute CFITSIO HDU the shared fptr is currently on (-1 = unknown).
    // Lets one-shot FITSFile wrappers skip redundant fits_movabs_hdu.
    int current_fits_hdu = -1;
    std::shared_mutex mutex;
};

inline std::mutex g_shared_meta_mutex;
inline std::unordered_map<std::string, std::shared_ptr<SharedReadMeta>> g_shared_meta;
inline std::atomic<uint64_t> g_shared_meta_uid{1};

inline const bool kValidateSharedMeta = []() {
    return torchfits::internal::env_flag_default_true("TORCHFITS_SHARED_META_VALIDATE");
}();

inline const int64_t kSharedMetaValidateIntervalNs = []() {
    constexpr int64_t kDefaultMs = 1000;
    return torchfits::internal::env_nonnegative_int(
        "TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS", kDefaultMs) * 1000000LL;
}();

inline std::shared_ptr<SharedReadMeta> get_shared_meta_for_path(const std::string& filename) {
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
            meta->current_fits_hdu = -1;
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

inline int open_readonly_fd(const std::string& filename) {
#ifdef O_CLOEXEC
    int fd = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != -1) return fd;
#endif
    return ::open(filename.c_str(), O_RDONLY);
}

// Prefer fits_open_diskfile for plain local paths (skips CFITSIO extended-syntax parse).
inline int open_fits_readonly(fitsfile** fptr, const std::string& path) {
    int status = 0;
    if (has_cfitsio_extended_filename_syntax(path) || path.find("://") != std::string::npos) {
        fits_open_file(fptr, path.c_str(), 0 /* READONLY */, &status);
    } else {
        fits_open_diskfile(fptr, path.c_str(), 0 /* READONLY */, &status);
    }
    return status;
}

inline std::shared_ptr<RawFdHolder> get_shared_raw_fd(
    const std::shared_ptr<SharedReadMeta>& meta, const std::string& filename) {
    if (!meta || has_cfitsio_extended_filename_syntax(filename)) return nullptr;
    std::unique_lock<std::shared_mutex> lock(meta->mutex);
    if (!meta->raw_fd) {
        meta->raw_fd = std::make_shared<RawFdHolder>(open_readonly_fd(filename));
    }
    return meta->raw_fd;
}

inline bool read_region_via_fd(int fd, off_t offset, void* dst_void, size_t nbytes) {
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

inline void invalidate_shared_meta_for_path(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.erase(filename);
}

inline void clear_shared_meta_cache() {
    std::lock_guard<std::mutex> lock(g_shared_meta_mutex);
    g_shared_meta.clear();
}

inline size_t datatype_elem_size(int datatype) {
    switch (datatype) {
        case TBYTE:
        case TSBYTE:    return sizeof(uint8_t);
        case TSHORT:
        case TUSHORT:   return sizeof(uint16_t);
        case TINT:
        case TUINT:     return sizeof(uint32_t);
        case TLONGLONG: return sizeof(uint64_t);
        case TFLOAT:    return sizeof(float);
        case TDOUBLE:   return sizeof(double);
        default:        return 0;
    }
}

// ---------------------------------------------------------------------------
// read_tensor_canonical — shared core for all image read paths
// ---------------------------------------------------------------------------
inline torch::Tensor read_tensor_canonical(
    fitsfile* fptr,
    const std::string& path,
    const ResolvedFITSMeta& meta,
    bool use_mmap,
    int raw_fd,
    bool use_chunking = false
) {
    const int bitpix = meta.bitpix;
    const int naxis = meta.naxis;
    const std::array<LONGLONG, 9>& naxes_ll = meta.naxes_ll;
    const bool scaled = meta.scaled;
    const double bscale = meta.bscale;
    const double bzero = meta.bzero;
    const bool compressed = meta.compressed;

    validate_image_naxis(naxis);
    if (naxis == 0) {
        torch::ScalarType dtype = torch::kUInt8;
        switch (bitpix) {
            case BYTE_IMG: dtype = torch::kUInt8; break;
            case SHORT_IMG: dtype = torch::kInt16; break;
            case LONG_IMG: dtype = torch::kInt32; break;
            case LONGLONG_IMG: dtype = torch::kInt64; break;
            case FLOAT_IMG: dtype = torch::kFloat32; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; break;
            default: break;
        }
        return torch::empty({0}, torch::TensorOptions().dtype(dtype));
    }

    LONGLONG nelements = checked_nelements_product(naxes_ll.data(), naxis);

    int64_t torch_shape[9];
    for (int i = 0; i < naxis; ++i)
        torch_shape[i] = static_cast<int64_t>(naxes_ll[naxis - 1 - i]);

    // Unsigned conventions
    const bool unsigned_short = scaled && bitpix == SHORT_IMG && bscale == 1.0 && is_unsigned_short_offset(bzero);
    const bool unsigned_long  = scaled && bitpix == LONG_IMG  && bscale == 1.0 && is_unsigned_long_offset(bzero);

    torch::ScalarType dtype;
    int datatype;
    if (scaled) {
        if (bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0) {
            dtype = at::kChar; datatype = TSBYTE;
        } else if (unsigned_short) {
            dtype = torch::kUInt16; datatype = TUSHORT;
        } else if (unsigned_long) {
            dtype = torch::kUInt32; datatype = TUINT;
        } else {
            dtype = torch::kFloat32; datatype = TFLOAT;
        }
    } else {
        switch (bitpix) {
            case BYTE_IMG: dtype = torch::kUInt8; datatype = TBYTE; break;
            case SHORT_IMG: dtype = torch::kInt16; datatype = TSHORT; break;
            case LONG_IMG: dtype = torch::kInt32; datatype = TINT; break;
            case LONGLONG_IMG: dtype = torch::kInt64; datatype = TLONGLONG; break;
            case FLOAT_IMG: dtype = torch::kFloat32; datatype = TFLOAT; break;
            case DOUBLE_IMG: dtype = torch::kFloat64; datatype = TDOUBLE; break;
            default: throw std::runtime_error("Unsupported BITPIX");
        }
    }

    auto tensor = torch::empty(at::IntArrayRef(torch_shape, naxis), torch::TensorOptions().dtype(dtype));

    // BYTE_IMG direct pread — works for mmap on/off (pread is buffered I/O,
    // not a secret mmap). Beats CFITSIO fits_read_img on large int8 payloads.
    const bool signed_byte_scaled = scaled && bitpix == BYTE_IMG && bscale == 1.0 && bzero == -128.0;
    if (!compressed && bitpix == BYTE_IMG && (!scaled || signed_byte_scaled)) {
        int status = 0;
        LONGLONG headstart = 0, data_offset = 0, dataend = 0;
        fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
        if (status == 0 && data_offset > 0) {
            const size_t nbytes = static_cast<size_t>(nelements);
            const int fd = raw_fd;
            if (fd != -1 && read_region_via_fd(fd, static_cast<off_t>(data_offset), tensor.data_ptr(), nbytes)) {
                if (signed_byte_scaled)
                    _xor_sign_bit_u8(static_cast<uint8_t*>(tensor.data_ptr()), nbytes);
                return tensor;
            }
        }
    }

    // Multi-byte mmap fast path — SIMD endian convert while copying (all sizes).
    const bool multi_byte_mmap_ok =
        use_mmap && !compressed && (!scaled || unsigned_short || unsigned_long) &&
        !has_cfitsio_extended_filename_syntax(path);
    if (multi_byte_mmap_ok) {
        size_t elem_size = 0;
        switch (bitpix) {
            case SHORT_IMG: elem_size = sizeof(uint16_t); break;
            // LONG_IMG: enable for both plain int32 and unsigned_long.
            // The unsigned offset (+2147483648u) is applied only when
            // unsigned_long is true — see the bswap dispatch below.
            case LONG_IMG: elem_size = sizeof(uint32_t); break;
            case LONGLONG_IMG: elem_size = sizeof(uint64_t); break;
            // FLOAT_IMG: bswap_32 is identical to int32 — the raw bits are
            // the same 4-byte big-endian pattern regardless of interpretation.
            case FLOAT_IMG: elem_size = sizeof(uint32_t); break;
            case DOUBLE_IMG: elem_size = sizeof(uint64_t); break;
            default: break;
        }
        if (elem_size > 0) {
            int status = 0;
            LONGLONG headstart = 0, data_offset = 0, dataend = 0;
            fits_get_hduaddrll(fptr, &headstart, &data_offset, &dataend, &status);
            if (status == 0 && data_offset > 0 && raw_fd != -1) {
                const size_t nbytes = static_cast<size_t>(nelements) * elem_size;
                struct stat sb {};
                if (nbytes > 0 && fstat(raw_fd, &sb) == 0 &&
                    static_cast<size_t>(sb.st_size) >= static_cast<size_t>(data_offset) + nbytes) {
                    static const long kPageSize = sysconf(_SC_PAGESIZE);
                    const off_t page_mask = kPageSize > 0 ? static_cast<off_t>(kPageSize - 1) : 0;
                    const off_t page_offset = data_offset & ~page_mask;
                    const size_t map_len = static_cast<size_t>(nbytes + (data_offset - page_offset));
                    void* map_ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, raw_fd, page_offset);
                    if (map_ptr != MAP_FAILED) {
#if defined(MADV_SEQUENTIAL) && defined(MADV_WILLNEED)
                        madvise(map_ptr, map_len, MADV_SEQUENTIAL | MADV_WILLNEED);
#endif
                        const size_t src_offset = static_cast<size_t>(data_offset - page_offset);
                        if (host_is_little_endian()) {
                            if (elem_size == sizeof(uint16_t)) {
                                const auto* src = reinterpret_cast<const uint16_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint16_t*>(tensor.data_ptr());
                                if (unsigned_short) {
                                    internal::bswap16_copy_u16_offset(
                                        src, dst, static_cast<size_t>(nelements),
                                        static_cast<uint16_t>(32768));
                                } else {
                                    internal::bswap16_copy(
                                        src, dst, static_cast<size_t>(nelements));
                                }
                            } else if (elem_size == sizeof(uint32_t)) {
                                const auto* src = reinterpret_cast<const uint32_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint32_t*>(tensor.data_ptr());
                                if (unsigned_long) {
                                    internal::bswap32_copy_u32_offset(
                                        src, dst, static_cast<size_t>(nelements),
                                        2147483648u);
                                } else {
                                    internal::bswap32_copy(
                                        src, dst, static_cast<size_t>(nelements));
                                }
                            } else if (elem_size == sizeof(uint64_t)) {
                                const auto* src = reinterpret_cast<const uint64_t*>(
                                    static_cast<const uint8_t*>(map_ptr) + src_offset);
                                auto* dst = static_cast<uint64_t*>(tensor.data_ptr());
                                internal::bswap64_copy(
                                    src, dst, static_cast<size_t>(nelements));
                            }
                        } else {
                            std::memcpy(tensor.data_ptr(), static_cast<const uint8_t*>(map_ptr) + src_offset, nbytes);
                        }
                        munmap(map_ptr, map_len);
                        return tensor;
                    }
                }
            }
        }
    }

    // CFITSIO fallback. Compressed images may hold undefined pixels; always
    // substitute NaN so nulls can never masquerade as 0 (CFITSIO only applies
    // nulval where a pixel is actually undefined, so this is free otherwise).
    float fnullval = NAN;
    double dnullval = NAN;
    void* nullval_ptr = nullptr;
    if ((datatype == TFLOAT || datatype == TDOUBLE) && compressed) {
        nullval_ptr = (datatype == TFLOAT) ? (void*)&fnullval : (void*)&dnullval;
    }

    int status = 0;
    if (!use_chunking) {
        int anynul = 0;
        fits_read_img(fptr, datatype, 1, nelements, nullval_ptr, tensor.data_ptr(), &anynul, &status);
    } else {
        static const size_t kChunkSizeBytes = 128 * 1024 * 1024;
        const size_t pixel_size = datatype_elem_size(datatype);
        const size_t effective_pixel_size = pixel_size > 0 ? pixel_size : 1;
        const LONGLONG chunk_pixels = static_cast<LONGLONG>(kChunkSizeBytes / effective_pixel_size);

        if (nelements <= chunk_pixels) {
            int anynul = 0;
            fits_read_img(fptr, datatype, 1, nelements, nullval_ptr, tensor.data_ptr(), &anynul, &status);
        } else {
            LONGLONG remain = nelements;
            LONGLONG offset = 0;
            char* dst_ptr = static_cast<char*>(tensor.data_ptr());
            while (remain > 0 && status == 0) {
                LONGLONG n_read = (remain > chunk_pixels) ? chunk_pixels : remain;
                int anynul = 0;
                fits_read_img(fptr, datatype, 1 + offset, n_read, nullval_ptr,
                              static_cast<void*>(dst_ptr + (offset * effective_pixel_size)), &anynul, &status);
                offset += n_read;
                remain -= n_read;
            }
        }
    }

    if (status != 0) {
        char err_text[31];
        fits_get_errstatus(status, err_text);
        throw std::runtime_error("Error reading image data: status=" + std::to_string(status) +
                                 " msg=" + std::string(err_text));
    }

    return tensor;
}

// ---------------------------------------------------------------------------
// FITS string sanitization
// ---------------------------------------------------------------------------
inline std::string sanitize_fits_string(const std::string& input) {
    std::string output = input;
    output.erase(std::remove_if(output.begin(), output.end(), [](unsigned char c) {
        return c < 32 || c > 126;
    }), output.end());
    return output;
}

inline std::string sanitize_fits_key(const std::string& input) {
    // Preserve long / hierarchical keywords (spaces allowed). Short FITS
    // keywords stay [A-Z0-9_-] only, uppercased.
    std::string raw = input;
    while (!raw.empty() && (raw.front() == ' ' || raw.front() == '\t')) {
        raw.erase(raw.begin());
    }
    while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\t')) {
        raw.pop_back();
    }
    std::string upper;
    upper.reserve(raw.size());
    for (unsigned char c : raw) {
        upper.push_back(static_cast<char>(std::toupper(c)));
    }
    // CFITSIO adds the HIERARCH token itself for long keys — strip a leading
    // "HIERARCH " so we do not double-prefix.
    if (upper.rfind("HIERARCH ", 0) == 0) {
        raw = raw.substr(9);
        while (!raw.empty() && raw.front() == ' ') {
            raw.erase(raw.begin());
        }
        upper = upper.substr(9);
        while (!upper.empty() && upper.front() == ' ') {
            upper.erase(upper.begin());
        }
    }

    const bool allow_spaces = raw.size() > 8 || raw.find(' ') != std::string::npos;
    std::string output;
    output.reserve(raw.size());
    for (unsigned char c : raw) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.' ||
            (allow_spaces && c == ' ')) {
            output.push_back(
                allow_spaces ? static_cast<char>(c)
                             : static_cast<char>(std::toupper(c)));
        }
    }
    if (allow_spaces && !output.empty()) {
        // Uppercase alphanumeric runs but keep spaces (ESO-style keys).
        for (char& c : output) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
        }
    }
    return output.empty() ? "UNKNOWN" : output;
}

} // namespace detail
} // namespace torchfits
