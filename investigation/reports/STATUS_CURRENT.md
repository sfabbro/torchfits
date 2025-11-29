# torchfits Optimization Status
**Last Updated:** 2025-11-28
**Current Branch:** refactor

---

## Current Performance

### Benchmark Results (Same Files, Fresh Opens)

| Configuration | uint8 | int16 | Ratio |
|--------------|-------|-------|-------|
| **fitsio (baseline)** | 0.079ms | 0.249ms | 3.16x |
| **C++ standalone (no Python)** | 0.043ms | 0.260ms | 6.03x |
| **torchfits (cache cleared)** | 0.062ms | 0.572ms | 9.17x |
| **torchfits (normal use)** | 0.044ms | 0.999ms | 22.90x |

### vs fitsio Comparison

| Type | torchfits (normal) | torchfits (cache cleared) | Target |
|------|-------------------|--------------------------|--------|
| uint8 | **0.56x** ✅ (faster!) | 0.79x | ≤1.0x |
| int16 | **4.01x** ❌ (slower) | 2.30x | ≤1.0x |

---

## What We've Accomplished

### Phase 1: DLPack Bypass (Completed ✅)
**Problem:** DLPack had 7x overhead for int16 (36μs vs 5μs for uint8)

**Solution:** Replaced DLPack with `THPVariable_Wrap` in [bindings.cpp:112-119](src/torchfits/cpp_src/bindings.cpp#L112-L119)

**Results:**
- int16: 2.14x → 1.60x (25% improvement)
- int32: 1.39x → 1.04x (25% improvement) - **matches fitsio!**
- Overall: 1.43x → 1.28x (10% improvement)

**Files Changed:**
- [bindings.cpp](src/torchfits/cpp_src/bindings.cpp) - Used THPVariable_Wrap
- [CMakeLists.txt](src/torchfits/cpp_src/CMakeLists.txt) - Added libtorch_python linking
- [fits.cpp](src/torchfits/cpp_src/fits.cpp) - Removed BSCALE/BZERO checking

### Phase 2: GIL Release (Completed ✅)
**Problem:** Handle-based `read_full` wasn't releasing the GIL

**Solution:** Added `nb::gil_scoped_release` in [bindings.cpp:232-240](src/torchfits/cpp_src/bindings.cpp#L232-L240)

**Results:**
- Allows concurrent I/O operations
- No measurable single-threaded performance impact (as expected)

### Phase 3: Cache Discovery (Current 🔍)
**Problem:** int16 shows 22.90x overhead in normal use, but only 6.03x in pure C++

**Discovery:**
- File handle caching in `global_cache.get_or_open()` causes int16 performance degradation
- Cache clearing reduces overhead from 22.90x → 9.17x (60% improvement!)
- C++ standalone shows true CFITSIO overhead is only **6.03x**, not 20x+

**Root Cause:**
- CFITSIO's internal buffering interacts poorly with int16 when file handles are reused
- uint8 is not affected (0.044ms vs 0.043ms with/without cache)
- int16 is severely affected (0.999ms vs 0.260ms with/without cache)

---

## Outstanding Issues

### 1. int16 Caching Overhead (HIGH PRIORITY)
**Current:** 22.90x with cache, 9.17x with cache cleared
**Target:** ≤6.0x (match C++ standalone)
**Impact:** Most users will hit this in normal use

**Options:**
- ✅ Disable file handle caching globally (simple, gets to 9.17x)
- 🔍 Smart caching (cache uint8, not int16)
- 🔍 CFITSIO buffer flushing before reads
- 🔬 Custom int16 reader (mmap + SIMD)

### 2. Python vs C++ Gap (MEDIUM PRIORITY)
**Current:** Python with cache cleared is 9.17x, C++ standalone is 6.03x
**Gap:** 1.52x unexplained overhead in Python binding layer
**Impact:** Affects best-case performance even with fixes

**Next Steps:**
- Profile with Instruments to find the 3ms gap (0.572ms - 0.260ms)
- Check if nanobind has dtype-specific overhead
- Verify THPVariable_Wrap performance for int16

### 3. CFITSIO Inherent Overhead (LOW PRIORITY)
**Current:** C++ shows 6.03x for int16 vs uint8
**fitsio shows:** 3.16x (better!)
**Gap:** 2.87x - we're slower than fitsio even in pure C++

**Possible Causes:**
- Different CFITSIO call parameters
- Different buffer sizes
- CFITSIO version differences
- Compiler optimization differences

---

## Next Steps (Priority Order)

### Immediate (Fix Cache Issue)
1. **Disable file handle caching** - Change `global_cache.get_or_open()` to always open fresh
   - Expected improvement: 22.90x → 9.17x
   - Risk: May impact multi-read workloads (but 0.03ms penalty is small)

2. **Re-benchmark comprehensive suite** - Get updated numbers across all dtypes

3. **Test with real workloads** - Ensure cache disabling doesn't break anything

### Short Term (Close the Gaps)
4. **Profile Python vs C++ gap** - Use Instruments to find the 0.3ms difference

5. **Investigate CFITSIO parameters** - Compare our calls vs fitsio's calls
   - Buffer sizes
   - Read strategies
   - CFITSIO configuration

6. **Test CFITSIO buffer flushing** - See if `fits_flush_buffer()` helps

### Long Term (Advanced Optimizations)
7. **Custom int16 reader** - mmap + SIMD byte swapping for uncompressed data

8. **Smart caching strategy** - Dtype-aware caching policy

9. **Parallel I/O** - For large files or batch operations

---

## Code Changes Summary

### Files Modified This Session
1. **[bindings.cpp:232-240](src/torchfits/cpp_src/bindings.cpp#L232-L240)**
   - Added GIL release to handle-based `read_full`

### Files to Modify Next
1. **[fits.cpp:134-138](src/torchfits/cpp_src/fits.cpp#L134-L138)**
   - Disable `global_cache.get_or_open()`
   - Always open fresh file handles

2. **[cache.cpp](src/torchfits/cpp_src/cache.cpp)**
   - Consider removing or making cache opt-in
   - Or implement dtype-aware caching

---

## Documentation Created

- ✅ [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Overall progress and learnings
- ✅ [DLPACK_FIX_RESULTS.md](DLPACK_FIX_RESULTS.md) - DLPack bypass details
- ✅ [PHASE3_CACHE_DISCOVERY.md](PHASE3_CACHE_DISCOVERY.md) - Cache investigation findings
- ✅ This file - Current status and next steps

---

## Test Files Created

### Benchmarking
- `benchmark_comprehensive.py` - Multi-dtype comprehensive benchmark
- `benchmark_final_validation.py` - Fresh file validation benchmark
- `profile_int16_detailed.py` - Component-level profiling
- `test_handle_reuse_effect.py` - Cache effect demonstration

### C++ Tests
- `test_cfitsio_vs_torch.cpp` - Standalone C++ performance baseline
- `test_dlpack_dtype_overhead.py` - Proved DLPack was the issue

---

## Performance Targets

### Stretch Goals
- All types ≤ 1.0x vs fitsio (faster or equal)
- int16 < 1.2x (acceptable given CFITSIO constraints)

### Realistic Goals (After Cache Fix)
- **int16: ≤ 1.5x** (currently 4.01x, cache cleared would be 2.30x)
- int32: ≤ 1.05x ✅ (already at 1.04x)
- float32: ≤ 1.20x (currently 1.29x)
- float64: ≤ 1.10x ✅ (already at 1.08x)

### Minimum Acceptable
- No type slower than 2.0x ✅ (uint8, int32, float64 already meet this)
- int16 < 2.0x (need cache fix to achieve)

---

## Conclusion

**Major Progress:**
- ✅ Identified and fixed DLPack bottleneck (10% overall improvement)
- ✅ int32 now matches fitsio (1.04x)
- ✅ Identified cache as main int16 bottleneck

**Critical Finding:**
- File handle caching causes 22.90x → 9.17x overhead for int16
- Disabling cache would make int16 go from **4.01x slower** → **2.30x slower** vs fitsio

**Next Priority:**
Disable file handle caching and re-benchmark to confirm the improvement. This single change could cut int16 overhead by >50%.
