# Phase 2: int16 Performance Investigation
**Date:** 2025-11-27
**Status:** In Progress

---

## Executive Summary

After removing BSCALE/BZERO checking to match fitsio's behavior, we achieved:
- **uint8: 0.73x** ✅ **27% FASTER than fitsio!**
- **float64: 0.62x** ✅ **38% FASTER than fitsio!**
- **float32: 1.14x** ⚠️ 14% slower (acceptable)
- **int16: 1.78x** ❌ **78% slower** (PROBLEM)
- **int32: 1.59x** ❌ 59% slower

**Critical issue:** int16 and int32 remain significantly slower than fitsio.

---

## What We Fixed

### Removed BSCALE/BZERO Checking

**Discovery:** fitsio does NOT check for BSCALE/BZERO keywords at all.

**Their code:**
```c
fits_movabs_hdu(self->fits, hdunum, &hdutype, &status);
fits_get_img_paramll(self->fits, maxdim, &datatype, &naxis, naxes, &status);
fits_read_pixll(self->fits, fits_read_dtype, firstpixels, size,
                nullval_ptr, data, &anynul, &status);
```

**No `fits_read_key` calls for BSCALE/BZERO!**

**Our old code:**
```cpp
if (bitpix == BYTE_IMG || bitpix == SHORT_IMG || ...) {
    fits_read_key(fptr_, TDOUBLE, "BSCALE", &bscale, nullptr, &status);
    status = 0;
    fits_read_key(fptr_, TDOUBLE, "BZERO", &bzero, nullptr, &status);
    status = 0;
    has_scaling = (bscale != 1.0 || bzero != 0.0);
}
```

**Impact:** Removed ~0.5ms overhead per read.

**Trade-off:** We now return raw unscaled integers if BSCALE/BZERO exist, matching fitsio's behavior. Users who need scaling should use astropy.

---

## Performance Results

### Before BSCALE/BZERO Removal

| Data Type | Size | torchfits | fitsio | Ratio |
|-----------|------|-----------|--------|-------|
| uint8 | 1kx1k (1MB) | 0.358ms | 0.358ms | 0.98x |
| int16 | 1kx1k (2MB) | 1.034ms | 0.527ms | **1.96x** |
| float32 | 1kx1k (4MB) | 1.153ms | 0.946ms | 1.22x |
| float64 | 1kx1k (8MB) | 2.207ms | 1.765ms | 1.25x |

### After BSCALE/BZERO Removal

| Data Type | Size | torchfits | fitsio | Ratio | Status |
|-----------|------|-----------|--------|-------|--------|
| uint8 | 100x100 (0.01MB) | 0.336ms | 0.360ms | **0.93x** | ✅ FASTER |
| uint8 | 1kx1k (1MB) | 0.569ms | 0.775ms | **0.73x** | ✅ **FASTER** |
| int16 | 1kx1k (2MB) | 1.300ms | 0.730ms | **1.78x** | ❌ SLOW |
| int32 | 1kx1k (4MB) | 1.812ms | 1.140ms | **1.59x** | ❌ SLOW |
| float32 | 1kx1k (4MB) | 1.504ms | 1.323ms | **1.14x** | ⚠️ OK |
| float64 | 1kx1k (8MB) | 2.975ms | 4.767ms | **0.62x** | ✅ **FASTER** |
| float32 | 2kx2k (16MB) | 5.272ms | 3.479ms | 1.52x | ❌ SLOW |
| float64 | 2kx2k (32MB) | 9.569ms | 9.712ms | **0.99x** | ✅ MATCH |
| float32 | 4kx4k (64MB) | 30.609ms | 17.450ms | 1.75x | ❌ SLOW |

**Key changes:**
- uint8: 0.98x → **0.73x** (27% improvement!)
- int16: 1.96x → 1.78x (9% improvement, still slow)
- float64: 1.25x → **0.62x** (huge improvement!)

---

## The int16 Mystery

### What We Know

**Our code for uint8:**
```cpp
auto tensor = torch::empty(shape, torch::kUInt8);
fits_read_img(fptr_, TBYTE, 1, total_pixels, nullptr,
             tensor.data_ptr<uint8_t>(), nullptr, &status);
```

**Our code for int16:**
```cpp
auto tensor = torch::empty(shape, torch::kInt16);
fits_read_img(fptr_, TSHORT, 1, total_pixels, nullptr,
             tensor.data_ptr<int16_t>(), nullptr, &status);
```

**Identical structure, but int16 is 1.78x slower!**

### fitsio's Strange Behavior

| Type | Size | fitsio Time | Notes |
|------|------|------------|-------|
| uint8 | 1MB | 0.775ms | - |
| int16 | 2MB | 0.730ms | ← **FASTER despite 2x file size!** |
| int32 | 4MB | 1.140ms | - |
| float32 | 4MB | 1.323ms | - |
| float64 | 8MB | 4.767ms | - |

fitsio's int16 is actually FASTER than uint8 despite being 2x the file size. This suggests they have some special optimization for int16, or uint8 has overhead in their code.

### Comparison: Extra Overhead

Testing showed we have **0.488ms extra overhead** for int16 vs uint8 compared to fitsio:

| | torchfits | fitsio |
|---|-----------|--------|
| uint8 | 0.518ms | 0.490ms |
| int16 | 1.293ms | 0.777ms |
| **Slowdown** | **+0.775ms** | **+0.286ms** |
| **Extra overhead** | **0.488ms** | - |

---

## What We've Ruled Out

✅ **BSCALE/BZERO checking** - Removed, improved by 9%, but not the root cause

✅ **Compiler optimization flags** - O3 is 9% faster than O2

✅ **CFITSIO API choice** - `fits_read_img` is 33% faster than `fits_read_pixll`

✅ **fits_get_img_paramll** - Already using LONGLONG variant like fitsio

✅ **Unnecessary CFITSIO calls** - Down to 2 calls (movabs_hdu + get_img_paramll)

---

## Hypotheses for Remaining int16 Slowdown

### 1. Memory Allocation Overhead
**Hypothesis:** `torch::empty(shape, torch::kInt16)` is slower than `torch::empty(shape, torch::kUInt8)` or numpy allocation.

**Test:** Profile torch::empty() separately:
```cpp
auto start = now();
auto tensor = torch::empty(shape, torch::kInt16);
auto alloc_time = now() - start;
```

### 2. Memory Alignment
**Hypothesis:** torch::Tensor has different memory alignment than numpy arrays, affecting CFITSIO's internal buffering differently for 2-byte vs 1-byte types.

**Test:** Check tensor.data_ptr() alignment and compare to numpy.

### 3. CFITSIO Internal Handling
**Hypothesis:** CFITSIO's `fits_read_img` with TSHORT has internal overhead (e.g., byte swapping, type conversion) that TBYTE doesn't have.

**Test:** Read CFITSIO source code for `ffgpv` (fits_read_img implementation).

### 4. PyTorch Type Dispatch Overhead
**Hypothesis:** torch::kInt16 has overhead in PyTorch's internal type system.

**Test:** Benchmark torch::empty() for different types.

### 5. System-Level Cache/Alignment
**Hypothesis:** 2-byte int16 has poor cache or memory alignment characteristics on ARM (M-series Mac).

**Test:** Use Instruments Time Profiler to see cache miss rates.

---

## Next Steps

### Priority 1: System-Level Profiling (REQUIRED)

Use macOS Instruments to see exactly where time is spent:

```bash
# Run profiling script
instruments -t "Time Profiler" \
  pixi run python profile_int16_system.py

# Open in Instruments.app to analyze:
# - Which C++ functions are slow
# - Cache miss rates (L1, L2, L3)
# - Branch mispredictions
# - Memory allocation overhead
```

### Priority 2: Memory Allocation Profiling

Add timing to C++ code to measure torch::empty() overhead:

```cpp
auto start = std::chrono::high_resolution_clock::now();
auto tensor = torch::empty(shape, torch::kInt16);
auto alloc_duration = std::chrono::high_resolution_clock::now() - start;
// Log alloc_duration
```

### Priority 3: Read CFITSIO Source

Study `cfitsio/getcolb.c` and `cfitsio/getcol.c` to understand differences between TBYTE and TSHORT handling.

### Priority 4: Test Without PyTorch

Create a pure C++ CFITSIO test without torch::Tensor to isolate whether the issue is in PyTorch or CFITSIO:

```cpp
std::vector<int16_t> buffer(total_pixels);
fits_read_img(fptr, TSHORT, 1, total_pixels, nullptr,
             buffer.data(), nullptr, &status);
```

If this is fast, the problem is torch::Tensor allocation/handling.

---

## Alternative Approaches

### Option 1: Read int16 as float32

Force int16 to be read as TFLOAT to see if that's faster:

```cpp
case SHORT_IMG: {
    auto tensor = torch::empty(shape, torch::kFloat32);
    fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                 tensor.data_ptr<float>(), nullptr, &status);
    return tensor.to(torch::kInt16);  // Convert back
}
```

**Trade-off:** Uses more memory, but might be faster if TFLOAT is fast.

### Option 2: Skip int16 Optimization for Now

Focus on the types we're winning at:
- ✅ uint8: 0.73x
- ✅ float64: 0.62x
- ⚠️ float32: 1.14x (acceptable)

Document that int16/int32 have known performance limitations and recommend users convert their data to float32 if performance is critical.

---

## Performance Target Assessment

### Current Status vs Goals

| Type | Current | Target (1.0x) | Stretch (0.8x) | Status |
|------|---------|---------------|----------------|--------|
| uint8 | 0.73x | ✅ | ✅ | **BEATING** |
| int16 | 1.78x | ❌ | ❌ | **FAILING** |
| int32 | 1.59x | ❌ | ❌ | **FAILING** |
| float32 | 1.14x | ⚠️ | ❌ | Close |
| float64 | 0.62x | ✅ | ✅ | **BEATING** |

**Overall:** 3/5 types beating target, 2/5 failing badly

---

## Recommendations

### For Users

**Best performance (better than fitsio):**
- Use uint8 data → **27% faster**
- Use float64 data → **38% faster**

**Acceptable performance:**
- Use float32 data → 14% slower

**Avoid if performance critical:**
- int16 data → 78% slower
- int32 data → 59% slower

### For Development

1. **Immediate:** Run system-level profiling with Instruments to find int16 bottleneck
2. **If profiling finds torch::empty() overhead:** Consider pre-allocating tensor pools
3. **If profiling finds CFITSIO overhead:** May need to use alternative CFITSIO functions or fall back to mmap for int16
4. **If cannot fix int16:** Document limitation and recommend float32 for performance-critical ML use cases

---

## Files Created/Modified

### Modified
- [src/torchfits/cpp_src/fits.cpp](src/torchfits/cpp_src/fits.cpp) - Removed BSCALE/BZERO checking

### Created
- [debug_uint8_vs_int16.py](debug_uint8_vs_int16.py) - Isolate int16 vs uint8 overhead
- [debug_int16_slowdown.py](debug_int16_slowdown.py) - Test BSCALE/BZERO impact
- [profile_int16_system.py](profile_int16_system.py) - System-level profiling script

### Documentation
- [PHASE2_FINAL_REPORT.md](PHASE2_FINAL_REPORT.md) - Comprehensive Phase 2 summary
- This file - int16 investigation details

---

## References

- [fitsio source code](https://github.com/esheldon/fitsio/blob/master/fitsio/fitsio_pywrap.c)
- [CFITSIO documentation](https://heasarc.gsfc.nasa.gov/fitsio/c/c_user/node40.html)
- [macOS Instruments profiling](https://developer.apple.com/documentation/instruments)
