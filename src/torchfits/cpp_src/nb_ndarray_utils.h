#pragma once

// nanobind ndarray helpers shared by the image and table write paths.
//
// These live apart from internal_utils.h so the torch-free core can include
// the latter: libtorchfits_core links CFITSIO and std::thread, never Python or
// nanobind. Nothing in this file may be reachable from core/.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace torchfits {
namespace internal {

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
    // `nelements` is how many elements the caller will hand to CFITSIO, not how
    // many the array holds. It has to be checked here rather than trusted: the
    // contiguous fast path below returns t.data() without reading it, and the
    // caller then reads nelements from that buffer, so an over-large count is a
    // heap overread past the array. Callers currently derive nelements from the
    // array itself; this keeps that an enforced invariant rather than a
    // convention. Values below size() are legitimate (packing a 2D column).
    if (nelements < 0) {
        throw std::runtime_error("negative element count for contiguous copy");
    }
    if (static_cast<uint64_t>(nelements) > static_cast<uint64_t>(t.size())) {
        throw std::runtime_error(
            "element count " + std::to_string(nelements) +
            " exceeds array size " + std::to_string(t.size()));
    }
    const size_t item = (static_cast<size_t>(t.dtype().bits) + 7) / 8;
    if (ndarray_is_c_contiguous(t)) {
        return t.data();
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
}  // namespace internal
}  // namespace torchfits
