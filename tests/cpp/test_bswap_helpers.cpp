// Standalone self-check: the big-endian -> host byte-swap helpers in
// internal_utils.h.
//
// These have no Python test, and the real-observation suites cannot reach them:
// the unsigned-int16/32 fast paths in read_tensor_canonical are gated on
// `!compressed` (the mmap path skips compressed images), while the CFHT corpus
// is Rice-compressed throughout. A wrong shuffle here is silent -- it produces
// plausible pixels that are simply byte-reversed -- so "no test" and "correct"
// look identical until someone compares against astropy on an uncompressed
// unsigned frame. (tests/test_byteswap.py compiles and runs this file, so the
// helpers are not untested; this file is where the comparison itself lives.)
//
// Three things are checked, because a byte-wise comparison alone is weaker than
// it looks:
//
//  1. The reference. Each helper is compared against a reference that moves
//     bytes one at a time (src[width-1] -> dst[0], and so on). It is
//     deliberately NOT __builtin_bswap*: that is exactly what the helpers' own
//     scalar tails call (internal::bswap_XX is a one-line alias for the
//     builtin), so a builtin-based reference shares code with the code under
//     test and cannot check it. Measured over every 16-bit value and 2M
//     32/64-bit samples, the builtin does agree with a hand-written reversal,
//     so nothing was broken -- but the check no longer depends on that.
//  2. Host order. A byte permutation compared with a byte permutation only
//     proves the two agree, not that either is the right one, so a known
//     big-endian FITS value is decoded and compared as a host-order integer.
//  3. Alignment. The SIMD branches use vld1q/vst1q and loadu/storeu, and the
//     tails use memcpy, all of which tolerate unaligned pointers -- but nothing
//     pinned that. SubsetReader::try_read_via_mmap passes
//     pixel_base_ + (y * naxis1 + x1) * elem_bytes_, so a cutout with an odd x1
//     lands 2 or 6 bytes past a page-aligned base on a 2-byte frame (verified:
//     a cutout at x1=1 and x1=3 reads back exactly). Every helper is therefore
//     also run with deliberately misaligned source and destination pointers.
//
// Element counts straddle every vector width and the scalar tail: 0, 1, 2, 3,
// 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 100, 257, 1000. The offset
// variants are checked with the two offsets the FITS unsigned conventions
// actually use (BZERO=32768 for int16, BZERO=2147483648 for int32), including
// the wrap, so the add is proven modulo 2^16/2^32 rather than in a range where
// saturation would not show.
//
// Which SIMD branch is compiled in is printed, because the three branches are
// selected by #if and a silent no-op would still "pass" a bad comparison. The
// scalar tail is the fallback when no branch is taken, so on an unknown
// architecture this probe still checks the tail -- against a reference that
// shares no code with it.
//
// Build (header-only; no project library needed):
//   c++ -std=c++17 -O2 -I src/torchfits/cpp_src \
//       tests/cpp/test_bswap_helpers.cpp -o /tmp/probe
// Run: /tmp/probe
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "internal_utils.h"

namespace {

using torchfits::internal::bswap16_copy;
using torchfits::internal::bswap16_copy_u16_offset;
using torchfits::internal::bswap32_copy;
using torchfits::internal::bswap32_copy_u32_offset;
using torchfits::internal::bswap64_copy;

int failures = 0;
std::string current_case;

void expect(bool ok) {
    if (!ok) {
        std::printf(" FAIL  %s\n", current_case.c_str());
        ++failures;
    }
}

// Deterministic bytes: fixed seed so a failure is reproducible, and a full
// sweep of all 256 byte values (measured: 256 distinct values in the 8192-byte
// buffer) so no lane can hide a permutation error.
std::vector<uint8_t> make_buffer() {
    std::vector<uint8_t> buf(8192);
    uint32_t s = 12345u;
    for (auto& b : buf) {
        s = s * 1103515245u + 12345u;
        b = static_cast<uint8_t>(s >> 16);
    }
    return buf;
}

// The reference: one byte at a time, no builtin, no intrinsic.
void ref_reverse(const uint8_t* src, uint8_t* dst, size_t width) {
    for (size_t j = 0; j < width; ++j) dst[j] = src[width - 1 - j];
}

template <typename T>
T load_host(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

template <typename T>
void store_host(uint8_t* p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

using CopyFn = std::function<void(const void*, void*, size_t)>;
using OffsetFn = std::function<void(const void*, void*, size_t, size_t)>;

// (src offset, dst offset) pairs. The first is the aligned case; the others put
// the pointers at every residue modulo 16 that matters for 2-, 4- and 8-byte
// elements, including source and destination disagreeing.
constexpr size_t kAlignments[][2] = {{0, 0}, {1, 1}, {3, 5}, {7, 2}};
constexpr size_t kAlignmentCount = sizeof(kAlignments) / sizeof(kAlignments[0]);

// Fill the destination with a byte the helpers never write, so an untouched
// byte is a mismatch rather than an accident.
constexpr uint8_t kFiller = 0xAA;

// Compare one plain helper at one (size, alignment).
void check_plain(
    const std::vector<uint8_t>& buf,
    size_t n,
    size_t width,
    size_t src_off,
    size_t dst_off,
    const CopyFn& fn
) {
    const size_t span = dst_off + n * width;
    std::vector<uint8_t> got(span, kFiller), want(span, kFiller);
    fn(buf.data() + src_off, got.data() + dst_off, n);
    for (size_t i = 0; i < n; ++i) {
        ref_reverse(buf.data() + src_off + i * width, want.data() + dst_off + i * width, width);
    }
    expect(std::memcmp(got.data(), want.data(), span) == 0);
}

// Same, for the BZERO variants: reverse the bytes, then do the offset add in
// host order -- the arithmetic is plain integer addition, so the part that
// needs an independent reference is the permutation, which ref_reverse gives.
template <typename T>
void check_offset(
    const std::vector<uint8_t>& buf,
    size_t n,
    size_t width,
    T offset,
    size_t src_off,
    size_t dst_off,
    const OffsetFn& fn
) {
    const size_t span = dst_off + n * width;
    std::vector<uint8_t> got(span, kFiller), want(span, kFiller);
    fn(buf.data() + src_off, got.data() + dst_off, n, offset);
    for (size_t i = 0; i < n; ++i) {
        uint8_t tmp[8];
        ref_reverse(buf.data() + src_off + i * width, tmp, width);
        const T v = static_cast<T>(load_host<T>(tmp) + offset);
        store_host<T>(want.data() + dst_off + i * width, v);
    }
    expect(std::memcmp(got.data(), want.data(), span) == 0);
}

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
// The permutation, tied to the FITS convention rather than to the reference: a
// big-endian value read through a helper must come out as its numeric value in
// host order. Run at a vector width so the SIMD branch, not the tail, is what
// gets pinned. (The helpers are only ever called on a little-endian host;
// build_arm64.cpp and friends guard the call sites with host_is_little_endian.)
void check_host_order() {
    {
        const uint8_t be[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xF0, 0xDE,
                              0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
        uint8_t out[16];
        std::memset(out, kFiller, sizeof out);
        current_case = "host order 16-bit: FITS bytes decode to the host value";
        bswap16_copy(be, out, 8);
        // Read big-endian, written as the number a human would read off the
        // FITS bytes: {0x12,0x34} is 0x1234, not 0x3412.
        const uint16_t want[8] = {0x1234, 0x5678, 0x9ABC, 0xF0DE,
                                  0x1122, 0x3344, 0x5566, 0x7788};
        bool ok = true;
        for (size_t i = 0; i < 8; ++i) ok = ok && load_host<uint16_t>(out + i * 2) == want[i];
        expect(ok);
    }
    {
        const uint8_t be[] = {0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
                              0x89, 0xAB, 0xCD, 0xEF, 0x89, 0xAB, 0xCD, 0xEF};
        uint8_t out[16];
        std::memset(out, kFiller, sizeof out);
        current_case = "host order 32-bit: FITS bytes decode to the host value";
        bswap32_copy(be, out, 4);
        const uint32_t want[4] = {0x89ABCDEFu, 0x01234567u, 0x89ABCDEFu, 0x89ABCDEFu};
        bool ok = true;
        for (size_t i = 0; i < 4; ++i) ok = ok && load_host<uint32_t>(out + i * 4) == want[i];
        expect(ok);
    }
    {
        const uint8_t be[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                              0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
        uint8_t out[16];
        std::memset(out, kFiller, sizeof out);
        current_case = "host order 64-bit: FITS bytes decode to the host value";
        bswap64_copy(be, out, 2);
        bool ok = load_host<uint64_t>(out) == 0x0123456789ABCDEFull;
        ok = ok && load_host<uint64_t>(out + 8) == 0xFEDCBA9876543210ull;
        expect(ok);
    }
}
#endif

}  // namespace

int main() {
    std::printf(
        "SIMD path compiled in -- NEON: %s  SSSE3: %s  AVX2: %s\n",
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        "yes",
#else
        "no",
#endif
#if defined(__SSSE3__)
        "yes",
#else
        "no",
#endif
#if defined(__AVX2__)
        "yes"
#else
        "no"
#endif
    );

    const std::vector<uint8_t> buf = make_buffer();
    const size_t sizes[] = {0,   1,   2,    3,    4,    5,    7,    8,    9,   15,
                            16,  17,  31,   32,   33,   63,   100,  257,  1000};

    for (size_t n : sizes) {
        char tag[64];
        std::snprintf(tag, sizeof tag, "n=%zu", n);
        for (size_t a = 0; a < kAlignmentCount; ++a) {
            const size_t src_off = kAlignments[a][0];
            const size_t dst_off = kAlignments[a][1];
            char where[96];
            std::snprintf(
                where, sizeof where, " (src+%zu dst+%zu)", src_off, dst_off);

            current_case = std::string("bswap16_copy ") + tag + where;
            check_plain(buf, n, 2, src_off, dst_off, bswap16_copy);

            current_case = std::string("bswap16_copy_u16_offset ") + tag + where;
            check_offset<uint16_t>(buf, n, 2, 32768, src_off, dst_off,
                                   bswap16_copy_u16_offset);

            current_case = std::string("bswap32_copy ") + tag + where;
            check_plain(buf, n, 4, src_off, dst_off, bswap32_copy);

            current_case = std::string("bswap32_copy_u32_offset ") + tag + where;
            check_offset<uint32_t>(buf, n, 4, 2147483648u, src_off, dst_off,
                                   bswap32_copy_u32_offset);

            current_case = std::string("bswap64_copy ") + tag + where;
            check_plain(buf, n, 8, src_off, dst_off, bswap64_copy);
        }
    }

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    check_host_order();
#endif

    if (failures == 0) {
        std::printf(
            "all %zu sizes x 5 helpers x %zu alignments match the byte-wise reference\n",
            sizeof(sizes) / sizeof(sizes[0]),
            kAlignmentCount
        );
    } else {
        std::printf("FAILURES: %d\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
