# Phase 3: Cache Discovery and int16 Performance Investigation
**Date:** 2025-11-28
**Status:** Critical finding - caching is the main bottleneck

---

## Executive Summary

We discovered that **file handle caching** is causing most of the int16 performance degradation, not CFITSIO or our C++ code:

| Test Configuration | uint8 | int16 | Ratio |
|-------------------|-------|-------|-------|
| **C++ standalone (no cache)** | 0.043ms | 0.260ms | **6.03x** ✅ |
| **Python with cache clearing** | 0.062ms | 0.572ms | **9.17x** ⚠️ |
| **Python without cache clearing** | 0.044ms | 0.999ms | **22.90x** ❌ |

**Key insight:** Cache clearing reduces overhead from 22.90x → 9.17x. The C++ standalone (which has no caching) shows only 6.03x overhead, proving that CFITSIO's inherent int16 overhead is only **6x**, not 20x+.

---

## Investigation Process

### 1. Initial Profiling Results

Python profiling showed **21x overhead** for int16:
```
Pure read operation (handle reuse):
  uint8: 0.0283ms
  int16: 0.6536ms
  Ratio: 23.09x
```

This contradicted our expectations since we'd already optimized DLPack.

### 2. C++ Standalone Test

Created `test_cfitsio_vs_torch.cpp` to isolate the issue:

```cpp
// Test 1: CFITSIO only (malloc buffer)
uint8:  0.033ms, int16: 0.192ms  →  5.86x ratio

// Test 2: torch::empty()
uint8:  0.000167ms, int16: 0.000208ms  →  1.25x ratio

// Test 3: Full path (alloc + CFITSIO read)
uint8:  0.043ms, int16: 0.260ms  →  6.03x ratio
```

**Result:** Only **6.03x overhead** in pure C++! So the 21x seen in Python must be from the wrapper/caching layer.

### 3. Same Files, Different Results

Testing the **exact same files** with both approaches:

**C++ standalone:**
```
uint8:  0.043ms
int16:  0.260ms
Ratio:  6.03x ✅
```

**Python binding:**
```
uint8:  0.044ms
int16:  0.999ms
Ratio:  22.90x ❌
```

**Smoking gun:** uint8 times are identical (0.043 vs 0.044), but int16 is **3.8x slower** in Python!

### 4. Cache Discovery

Found two levels of caching:

1. **Python-level cache** (`__init__.py:138-144`):
   ```python
   cache_key = (path, hdu, device, fp16, bf16, columns, start_row, num_rows)
   if cache_key in _file_cache:
       return cached_data, cached_header
   ```

2. **C++ file handle cache** (`fits.cpp:134-138`):
   ```cpp
   fptr_ = global_cache.get_or_open(filename);
   ```

### 5. Cache Clearing Test

With `cpp.clear_file_cache()` before each read:
```
uint8:  0.062ms
int16:  0.572ms
Ratio:  9.17x
```

**Result:** Dropped from 22.90x → 9.17x just by clearing the C++ file handle cache!

---

## Root Cause Analysis

### Why is int16 Affected More by Caching?

The file handle cache in `global_cache.get_or_open()` keeps `fitsfile*` pointers open. When reading int16 data:

1. **First read:** Fresh file handle, CFITSIO reads from disk → ~0.26ms (C++ baseline)
2. **Subsequent reads with cached handle:** CFITSIO may be:
   - Re-reading already buffered data (slower for int16 due to byte swapping in buffer)
   - Seeking within internal buffers (int16 has larger element size)
   - Performing extra validation on cached file state

**Hypothesis:** CFITSIO's internal buffering interacts poorly with int16's 2-byte elements and byte-swapping requirements when the file handle is reused.

### Breakdown of Overhead

| Component | uint8 | int16 | Overhead |
|-----------|-------|-------|----------|
| **CFITSIO inherent** | 0.033ms | 0.192ms | 5.82x |
| **torch::empty()** | ~0ms | ~0ms | ~1x |
| **C++ total (no cache)** | 0.043ms | 0.260ms | **6.03x** |
| **Python + cache (handle reuse)** | 0.044ms | 0.999ms | **22.90x** |
| **Cache-induced overhead** | +0.001ms | +0.739ms | **3.8x extra** |

---

## Solutions

### Option 1: Disable File Handle Caching (RECOMMENDED)

**Pros:**
- Immediately reduces int16 overhead from 22.90x → 9.17x
- Simple one-line change

**Cons:**
- Loses ~0.03ms file open/close amortization
- May impact workloads that read the same file many times

**Implementation:**
```cpp
// In fits.cpp constructor, disable caching:
FITSFile(const std::string& filename, int mode = 0) : filename_(filename), mode_(mode) {
    // OLD: fptr_ = global_cache.get_or_open(filename);

    // NEW: Always open fresh
    int status = 0;
    fits_open_file(&fptr_, filename.c_str(), READONLY, &status);
    if (status != 0) {
        throw std::runtime_error("Failed to open FITS file: " + filename);
    }
    cached_ = false;
}
```

### Option 2: Smart Caching Based on Dtype

Cache uint8/float32 (which benefit), but not int16:

```cpp
bool should_cache(int bitpix) {
    // Don't cache int16 - it performs worse with caching
    return bitpix != 16;
}
```

### Option 3: Flush CFITSIO Buffers

Explicitly flush CFITSIO buffers before each read:

```cpp
// Before reading
fits_flush_buffer(fptr_, 0, &status);
```

### Option 4: Use Direct I/O for int16

Bypass CFITSIO entirely for uncompressed int16:

```cpp
if (bitpix == 16 && !compressed) {
    // mmap + SIMD byte swap
    return read_int16_direct(filename);
}
```

---

## Recommendations

### Immediate (Phase 3)
1. ✅ **Disable file handle caching** - Gets us from 22.90x → 9.17x
2. 🔍 **Investigate remaining 9.17x vs 6.03x gap** - Why is Python still 3x slower than C++ even with cache disabled?

### Short Term
3. **Profile CFITSIO buffer behavior** - Use Instruments to see what CFITSIO does differently with int16
4. **Test buffer flushing** - See if `fits_flush_buffer()` helps

### Long Term
5. **Custom int16 reader** - mmap + SIMD for uncompressed data
6. **Optimize for common patterns** - Cache strategy based on access patterns

---

## Performance Targets

### Current Status
| Type | Current | Target | Status |
|------|---------|--------|--------|
| int16 | 22.90x | ≤6.0x | ❌ Cache issue |
| int16 (cache cleared) | 9.17x | ≤6.0x | ⚠️ Still investigating |
| int16 (C++ baseline) | 6.03x | ≤6.0x | ✅ ACHIEVED |

### After Cache Fix
- Disable caching → **9.17x** (50% improvement)
- Understand Python overhead → target **≤7.0x**
- Direct I/O for int16 → target **≤3.0x**

---

## Files Created

1. **test_cfitsio_vs_torch.cpp** - C++ standalone benchmark (proves 6.03x is the baseline)
2. **test_handle_reuse_effect.py** - Demonstrated handle reuse affects performance
3. **profile_int16_detailed.py** - Detailed component profiling
4. **This file** - Investigation summary

---

## Next Steps

1. **Disable file handle caching globally** and re-benchmark
2. **Profile the remaining 9.17x → 6.03x gap** with Instruments
3. **Consider dtype-specific caching strategies**
4. **Investigate CFITSIO buffer behavior** with handle reuse

---

## Key Learnings

1. **Caching isn't always faster** - int16's byte-swapping interacts poorly with cached handles
2. **Always test with fresh data** - Caches can hide real performance characteristics
3. **Compare apples to apples** - C++ standalone revealed the true CFITSIO overhead
4. **uint8 as control** - Showed that caching overhead is dtype-specific
5. **File handle caching ≠ data caching** - But still has performance implications

---

## Conclusion

**Success:** We've identified that file handle caching is the primary cause of int16 performance degradation (22.90x → 9.17x with cache clearing).

**Remaining Work:** Understand why Python is still 9.17x vs C++'s 6.03x even with cache clearing, then implement the appropriate caching strategy.

**Impact:** Fixing this could bring int16 from 2.83x slower than fitsio to potentially **faster** than fitsio!
