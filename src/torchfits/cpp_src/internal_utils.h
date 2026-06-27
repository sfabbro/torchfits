#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <sys/stat.h>

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

}  // namespace internal
}  // namespace torchfits
