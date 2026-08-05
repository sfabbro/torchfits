#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <sys/stat.h>

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

/// Canonical aliases for code that expects the undecorated names.
inline uint32_t bswap32(uint32_t x) { return bswap_32(x); }
inline uint64_t bswap64(uint64_t x) { return bswap_64(x); }

/// Vectorized big-endian → host endian copies for image mmap paths.
inline void bswap16_copy(const uint16_t* src, uint16_t* dst, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint8x16_t b = vrev16q_u8(vreinterpretq_u8_u16(v));
        vst1q_u16(dst + i, vreinterpretq_u16_u8(b));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) dst[i] = bswap_16(src[i]);
}

inline void bswap16_copy_u16_offset(const uint16_t* src, uint16_t* dst, size_t n,
                                    uint16_t offset) {
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint16x8_t off = vdupq_n_u16(offset);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint8x16_t b = vrev16q_u8(vreinterpretq_u8_u16(v));
        vst1q_u16(dst + i, vaddq_u16(vreinterpretq_u16_u8(b), off));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m256i off = _mm256_set1_epi16(static_cast<int16_t>(offset));
    for (; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        v = _mm256_shuffle_epi8(v, shuffle);
        v = _mm256_add_epi16(v, off);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m128i off = _mm_set1_epi16(static_cast<int16_t>(offset));
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        v = _mm_shuffle_epi8(v, shuffle);
        v = _mm_add_epi16(v, off);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), v);
    }
#endif
    for (; i < n; ++i) dst[i] = static_cast<uint16_t>(bswap_16(src[i]) + offset);
}

inline void bswap32_copy(const uint32_t* src, uint32_t* dst, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 4 <= n; i += 4) {
        uint32x4_t v = vld1q_u32(src + i);
        uint8x16_t b = vrev32q_u8(vreinterpretq_u8_u32(v));
        vst1q_u32(dst + i, vreinterpretq_u32_u8(b));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    for (; i + 4 <= n; i += 4) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) dst[i] = bswap_32(src[i]);
}

inline void bswap32_copy_u32_offset(const uint32_t* src, uint32_t* dst, size_t n,
                                    uint32_t offset) {
    bswap32_copy(src, dst, n);
    for (size_t i = 0; i < n; ++i) dst[i] += offset;
}

inline void bswap64_copy(const uint64_t* src, uint64_t* dst, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 2 <= n; i += 2) {
        uint64x2_t v = vld1q_u64(src + i);
        uint8x16_t b = vrev64q_u8(vreinterpretq_u8_u64(v));
        vst1q_u64(dst + i, vreinterpretq_u64_u8(b));
    }
#elif defined(__AVX2__)
    const __m256i shuffle = _mm256_set_epi8(
        24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    for (; i + 4 <= n; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_shuffle_epi8(v, shuffle));
    }
#elif defined(__SSSE3__)
    const __m128i shuffle = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    for (; i + 2 <= n; i += 2) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_shuffle_epi8(v, shuffle));
    }
#endif
    for (; i < n; ++i) dst[i] = bswap_64(src[i]);
}

}  // namespace internal
}  // namespace torchfits
