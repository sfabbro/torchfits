#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <sys/stat.h>

#include <nanobind/ndarray.h>

namespace nb = nanobind;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#endif

namespace torchfits {
namespace internal {

/// Returns true unless the environment variable is explicitly falsy.
/// Lowercases the value before comparison to accept "0", "false", "off", "no"
/// in any casing.
inline bool env_flag_default_true(const char* name) {
    const char* v = std::getenv(name);
    if (!v) {
        return true;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s == "0" || s == "false" || s == "off" || s == "no");
}

/// Returns the non-negative integer value of an environment variable,
/// or `default_value` when unset / unparseable / negative.
inline int64_t env_nonnegative_int(const char* name, int64_t default_value) {
    const char* v = std::getenv(name);
    if (!v) {
        return default_value;
    }
    try {
        int64_t parsed = std::stoll(std::string(v));
        return parsed < 0 ? 0 : parsed;
    } catch (...) {
        return default_value;
    }
}

/// Monotonic clock in nanoseconds (steady_clock).
inline int64_t monotonic_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Extract file mtime as nanoseconds since epoch from a `struct stat`.
inline int64_t mtime_ns_from_stat(const struct stat& st) {
#if defined(__APPLE__)
    return (static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL) +
           static_cast<int64_t>(st.st_mtimespec.tv_nsec);
#else
    return (static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL) +
           static_cast<int64_t>(st.st_mtim.tv_nsec);
#endif
}

/// Byte-swap helpers for converting FITS big-endian data to host byte order.
/// Accept unsigned types to match the raw byte patterns from CFITSIO.
inline uint16_t bswap_16(uint16_t x) { return __builtin_bswap16(x); }
inline uint32_t bswap_32(uint32_t x) { return __builtin_bswap32(x); }
inline uint64_t bswap_64(uint64_t x) { return __builtin_bswap64(x); }

template <typename T>
inline T load_unaligned(const uint8_t* src) {
    T value;
    std::memcpy(&value, src, sizeof(T));
    return value;
}

template <typename T>
inline void store_unaligned(uint8_t* dst, T value) {
    std::memcpy(dst, &value, sizeof(T));
}

/// Scalar single-value helpers are spelled `bswap_XX`; no undecorated aliases
/// (they duplicated the canonical names and had zero call sites).

/// Vectorized big-endian → host endian copies for image mmap paths.
inline void bswap16_copy(const void* src, void* dst, size_t n) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; n - i >= 8; i += 8) {
        uint8x16_t v = vld1q_u8(src_bytes + i * sizeof(uint16_t));
        uint8x16_t b = vrev16q_u8(v);
        vst1q_u8(dst_bytes + i * sizeof(uint16_t), b);
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (; n - i >= 16; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src_bytes + i * sizeof(uint16_t)));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst_bytes + i * sizeof(uint16_t)),
            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (; n - i >= 8; i += 8) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(src_bytes + i * sizeof(uint16_t)));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst_bytes + i * sizeof(uint16_t)),
            _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) {
        const size_t byte_offset = i * sizeof(uint16_t);
        store_unaligned<uint16_t>(
            dst_bytes + byte_offset,
            bswap_16(load_unaligned<uint16_t>(src_bytes + byte_offset)));
    }
}

inline void bswap16_copy_u16_offset(const void* src, void* dst, size_t n,
                                    uint16_t offset) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint16x8_t off = vdupq_n_u16(offset);
    for (; n - i >= 8; i += 8) {
        uint8x16_t v = vld1q_u8(src_bytes + i * sizeof(uint16_t));
        uint8x16_t b = vrev16q_u8(v);
        vst1q_u8(dst_bytes + i * sizeof(uint16_t),
                 vreinterpretq_u8_u16(vaddq_u16(vreinterpretq_u16_u8(b), off)));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m256i off = _mm256_set1_epi16(static_cast<int16_t>(offset));
    for (; n - i >= 16; i += 16) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src_bytes + i * sizeof(uint16_t)));
        v = _mm256_shuffle_epi8(v, shuffle);
        v = _mm256_add_epi16(v, off);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst_bytes + i * sizeof(uint16_t)), v);
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m128i off = _mm_set1_epi16(static_cast<int16_t>(offset));
    for (; n - i >= 8; i += 8) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(src_bytes + i * sizeof(uint16_t)));
        v = _mm_shuffle_epi8(v, shuffle);
        v = _mm_add_epi16(v, off);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst_bytes + i * sizeof(uint16_t)), v);
    }
#endif
    for (; i < n; ++i) {
        const size_t byte_offset = i * sizeof(uint16_t);
        const uint16_t value =
            bswap_16(load_unaligned<uint16_t>(src_bytes + byte_offset));
        store_unaligned<uint16_t>(
            dst_bytes + byte_offset, static_cast<uint16_t>(value + offset));
    }
}

inline void bswap32_copy(const void* src, void* dst, size_t n) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; n - i >= 4; i += 4) {
        uint8x16_t v = vld1q_u8(src_bytes + i * sizeof(uint32_t));
        uint8x16_t b = vrev32q_u8(v);
        vst1q_u8(dst_bytes + i * sizeof(uint32_t), b);
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    for (; n - i >= 8; i += 8) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src_bytes + i * sizeof(uint32_t)));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst_bytes + i * sizeof(uint32_t)),
            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    for (; n - i >= 4; i += 4) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(src_bytes + i * sizeof(uint32_t)));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst_bytes + i * sizeof(uint32_t)),
            _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) {
        const size_t byte_offset = i * sizeof(uint32_t);
        store_unaligned<uint32_t>(
            dst_bytes + byte_offset,
            bswap_32(load_unaligned<uint32_t>(src_bytes + byte_offset)));
    }
}

inline void bswap32_copy_u32_offset(const void* src, void* dst, size_t n,
                                    uint32_t offset) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint32x4_t off = vdupq_n_u32(offset);
    for (; n - i >= 4; i += 4) {
        uint8x16_t v = vld1q_u8(src_bytes + i * sizeof(uint32_t));
        uint8x16_t b = vrev32q_u8(v);
        vst1q_u8(dst_bytes + i * sizeof(uint32_t),
                 vreinterpretq_u8_u32(vaddq_u32(vreinterpretq_u32_u8(b), off)));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i off = _mm256_set1_epi32(static_cast<int32_t>(offset));
    for (; n - i >= 8; i += 8) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src_bytes + i * sizeof(uint32_t)));
        v = _mm256_shuffle_epi8(v, shuffle);
        v = _mm256_add_epi32(v, off);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst_bytes + i * sizeof(uint32_t)), v);
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    const __m128i off = _mm_set1_epi32(static_cast<int32_t>(offset));
    for (; n - i >= 4; i += 4) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(src_bytes + i * sizeof(uint32_t)));
        v = _mm_shuffle_epi8(v, shuffle);
        v = _mm_add_epi32(v, off);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst_bytes + i * sizeof(uint32_t)), v);
    }
#endif
    for (; i < n; ++i) {
        const size_t byte_offset = i * sizeof(uint32_t);
        const uint32_t value =
            bswap_32(load_unaligned<uint32_t>(src_bytes + byte_offset));
        store_unaligned<uint32_t>(dst_bytes + byte_offset, value + offset);
    }
}

inline void bswap64_copy(const void* src, void* dst, size_t n) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; n - i >= 2; i += 2) {
        uint8x16_t v = vld1q_u8(src_bytes + i * sizeof(uint64_t));
        uint8x16_t b = vrev64q_u8(v);
        vst1q_u8(dst_bytes + i * sizeof(uint64_t), b);
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    for (; n - i >= 4; i += 4) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src_bytes + i * sizeof(uint64_t)));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst_bytes + i * sizeof(uint64_t)),
            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    for (; n - i >= 2; i += 2) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(src_bytes + i * sizeof(uint64_t)));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst_bytes + i * sizeof(uint64_t)),
            _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) {
        const size_t byte_offset = i * sizeof(uint64_t);
        store_unaligned<uint64_t>(
            dst_bytes + byte_offset,
            bswap_64(load_unaligned<uint64_t>(src_bytes + byte_offset)));
    }
}

// ---------------------------------------------------------------------------
// nanobind ndarray helpers shared by the image and table write paths
// ---------------------------------------------------------------------------

/// True when a nanobind ndarray view is C-contiguous. Signed math: a negative
/// stride must compare unequal to a positive expected stride, never wrap to a
/// huge size_t that looks contiguous.
inline bool ndarray_is_c_contiguous(const nb::ndarray<>& t) {
    std::ptrdiff_t expect = 1;
    for (size_t d = t.ndim(); d-- > 0;) {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(t.shape(d));
        if (n <= 1) {
            continue;
        }
        if (static_cast<std::ptrdiff_t>(t.stride(d)) != expect) {
            return false;
        }
        expect *= n;
    }
    return true;
}

/// Copy a possibly non-contiguous ndarray view into `buf` (grown as needed)
/// and return a pointer to contiguous data; returns the source pointer
/// unchanged when already contiguous. Strides are signed element offsets:
/// negative strides address earlier elements, so use ptrdiff_t.
inline void* ensure_c_contiguous_ndarray(
    nb::ndarray<>& t, long nelements, std::vector<uint8_t>& buf
) {
    const size_t item = (static_cast<size_t>(t.dtype().bits) + 7) / 8;
    if (ndarray_is_c_contiguous(t)) {
        return t.data();
    }
    if (nelements < 0) {
        throw std::runtime_error("negative element count for contiguous copy");
    }
    if (item != 0 && static_cast<uint64_t>(nelements) > SIZE_MAX / item) {
        throw std::runtime_error("array byte size overflows size_t");
    }
    buf.resize(static_cast<size_t>(nelements) * item);
    auto* dst = buf.data();
    const auto* base = static_cast<const uint8_t*>(t.data());
    const size_t ndim = t.ndim();
    if (ndim == 1) {
        const std::ptrdiff_t s0 =
            static_cast<std::ptrdiff_t>(t.stride(0)) * static_cast<std::ptrdiff_t>(item);
        for (long i = 0; i < nelements; ++i) {
            std::memcpy(dst + static_cast<size_t>(i) * item,
                        base + static_cast<std::ptrdiff_t>(i) * s0, item);
        }
        return dst;
    }
    if (ndim == 2) {
        const size_t n0 = static_cast<size_t>(t.shape(0));
        const size_t n1 = static_cast<size_t>(t.shape(1));
        const std::ptrdiff_t s0 =
            static_cast<std::ptrdiff_t>(t.stride(0)) * static_cast<std::ptrdiff_t>(item);
        const std::ptrdiff_t s1 =
            static_cast<std::ptrdiff_t>(t.stride(1)) * static_cast<std::ptrdiff_t>(item);
        size_t out = 0;
        for (size_t i0 = 0; i0 < n0; ++i0) {
            for (size_t i1 = 0; i1 < n1; ++i1) {
                std::memcpy(dst + out * item,
                            base + static_cast<std::ptrdiff_t>(i0) * s0
                                 + static_cast<std::ptrdiff_t>(i1) * s1, item);
                ++out;
            }
        }
        return dst;
    }
    // Generic N-dim strided copy: row-major linear index -> byte offset.
    std::vector<std::ptrdiff_t> strides(ndim);
    std::vector<size_t> shape(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        shape[d] = static_cast<size_t>(t.shape(d));
        strides[d] = static_cast<std::ptrdiff_t>(t.stride(d)) * static_cast<std::ptrdiff_t>(item);
    }
    const size_t n = static_cast<size_t>(nelements);
    for (size_t lin = 0; lin < n; ++lin) {
        size_t rem = lin;
        std::ptrdiff_t off = 0;
        for (size_t d = ndim; d-- > 0;) {
            const size_t dim = shape[d];
            if (dim == 0) break;
            const size_t idx = rem % dim;
            rem /= dim;
            off += static_cast<std::ptrdiff_t>(idx) * strides[d];
        }
        std::memcpy(dst + lin * item, base + off, item);
    }
    return dst;
}

/// Pad/truncate string values to a FITS character-column width: longer values
/// are clipped, shorter values right-padded with ASCII spaces.
inline std::vector<std::string> pad_fits_strings(
    const std::vector<std::string>& values, long width_chars
) {
    std::vector<std::string> padded;
    padded.reserve(values.size());
    for (const auto& v : values) {
        std::string s = v;
        if (static_cast<long>(s.size()) > width_chars) {
            s = s.substr(0, static_cast<size_t>(width_chars));
        } else if (static_cast<long>(s.size()) < width_chars) {
            s.append(static_cast<size_t>(width_chars - s.size()), ' ');
        }
        padded.push_back(std::move(s));
    }
    return padded;
}

}  // namespace internal
}  // namespace torchfits
