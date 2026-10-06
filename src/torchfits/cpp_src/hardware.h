#pragma once

#include <cstdlib>
#include <string>
#include <mutex>

#include "core/core_export.h"
#include "internal_utils.h"

namespace torchfits {

inline bool host_is_little_endian() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

// RAII wrapper for mmap
class MMapHandle {
public:
    void* ptr = nullptr;
    size_t size = 0;
    int fd = -1;
    bool owner = false;

    MMapHandle() = default;
    TORCHFITS_CORE_API explicit MMapHandle(const std::string& filename);
    TORCHFITS_CORE_API explicit MMapHandle(const std::string& filename, bool writable);
    TORCHFITS_CORE_API explicit MMapHandle(void* ptr, size_t size, int fd, bool owner = true);

    // Move constructor
    MMapHandle(MMapHandle&& other) noexcept
        : ptr(other.ptr), size(other.size), fd(other.fd), owner(other.owner) {
        other.ptr = nullptr;
        other.size = 0;
        other.fd = -1;
        other.owner = false;
    }

    // Move assignment
    MMapHandle& operator=(MMapHandle&& other) noexcept {
        if (this != &other) {
            cleanup();
            ptr = other.ptr;
            size = other.size;
            fd = other.fd;
            owner = other.owner;
            other.ptr = nullptr;
            other.size = 0;
            other.fd = -1;
            other.owner = false;
        }
        return *this;
    }

    ~MMapHandle() {
        cleanup();
    }

    // Out-of-line definitions live in libtorchfits_core (hardware.cpp), so
    // the torch-linked extension and the torch-free metadata core map the same
    // file rather than each compiling their own copy of the mmap bookkeeping.
    TORCHFITS_CORE_API void cleanup();

    // Give up ownership of the descriptor without closing it, for the
    // open -> fstat -> validate -> mmap sequence: a guard has to own the fd
    // from the moment it is opened (the validation step can throw, and a raw
    // int on the stack is not closed by unwinding), but the mapping that
    // finally owns it does not exist yet. Only valid on a handle constructed
    // as an fd-only owner (ptr == nullptr) -- calling it on a mapped handle
    // would strand the mapping. Returns the descriptor, or -1 if this handle
    // does not own one.
    int detach_fd() noexcept {
        if (!owner) {
            return -1;
        }
        owner = false;
        const int released = fd;
        fd = -1;
        return released;
    }
};

// Convenience wrappers that accept signed integer types commonly used in
// torchfits table/image code. Delegates to the canonical unsigned helpers
// in internal_utils.h.
inline int16_t bswap_16(int16_t x) { return static_cast<int16_t>(internal::bswap_16(static_cast<uint16_t>(x))); }
inline int32_t bswap_32(int32_t x) { return static_cast<int32_t>(internal::bswap_32(static_cast<uint32_t>(x))); }
inline int64_t bswap_64(int64_t x) { return static_cast<int64_t>(internal::bswap_64(static_cast<uint64_t>(x))); }

// Re-export the unsigned overloads from internal_utils.h so callers with
// uintXX_t arguments (e.g. from raw memory byte-swaps) get the canonical
// helpers directly without implicit signed/unsigned conversions.
using internal::bswap_16;
using internal::bswap_32;
using internal::bswap_64;

}  // namespace torchfits
