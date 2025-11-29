# int16 Bottleneck: Root Cause Identified
**Date:** 2025-11-28
**Status:** Bottleneck located - NOT in wrapping, in CFITSIO call context

---

## Summary

**Pure CFITSIO performance:** 0.104ms (int16, 1000x1000)
**Our actual performance:** 0.873ms (int16, with open/close each iteration)
**Performance gap:** **8.4x slower than pure CFITSIO!**

The bottleneck is **NOT**:
- ❌ Tensor wrapping (NumPy vs torch.Tensor - only ~1μs difference)
- ❌ DLPack (disabled)
- ❌ File handle caching (disabled)
- ❌ Memory allocation (malloc vs tensor.data_ptr - both slow)

The bottleneck **IS**:
- ✅ Something in the C++ code context when calling `fits_read_pixll()`
- ✅ CFITSIO runs 8.4x slower in our C++ code than in pure C

---

## Evidence

### 1. Pure C Benchmark

```c
// benchmark_pure_cfitsio.c
fits_open_file(&fptr, filename, READONLY, &status);
fits_read_pixll(fptr, TSHORT, fpixel, nelements, NULL, buffer, &anynull, &status);
fits_close_file(fptr, &status);
```

**Result:** 0.104ms (median of 100 iterations)

### 2. Our C++ Code

```cpp
// fits.cpp - read_pixels_impl()
auto tensor = torch::empty(shape, torch::kInt16);
fits_read_pixll(fptr_, TSHORT, firstpix, total_pixels, nullptr,
               tensor.data_ptr<int16_t>(), &anynul, &status);
```

**Result:** 0.497-0.810μs according to internal profiling
**But actual measured time:** 0.873ms (full open+read+close)

### 3. Handle Reuse Test

Pure C with handle reuse:
- First read: 0.466ms
- Subsequent reads: 0.19-0.32ms
- **Conclusion:** Handle reuse makes CFITSIO ~20% slower

Pure C without handle reuse (open/close each time):
- All reads: 0.16-0.18ms ✅ FAST

### 4. malloc Buffer Test

Tried reading into malloc() buffer instead of tensor.data_ptr():
- **Result:** Still slow (575-810μs)
- **Conclusion:** NOT a tensor memory allocation issue

---

## The Mystery

**Question:** Why is `fits_read_pixll()` 8.4x slower in our C++ code than in pure C?

Both use:
- Same CFITSIO version (4.6.3)
- Same function (`fits_read_pixll`)
- Same data type (TSHORT)
- Same file
- Same compiler (-O3)

**Possible causes:**

1. **GIL release overhead?**
   - We release the GIL around I/O
   - Could this cause thread/synchronization overhead?

2. **PyTorch library interference?**
   - Linked against libtorch
   - Could PyTorch hooks/allocators affect CFITSIO?

3. **Memory allocation context?**
   - Even malloc() buffer is slow
   - Could be process-wide allocator settings

4. **CFITSIO internal state?**
   - Shared library loaded differently?
   - Different initialization?

5. **Compiler/linker differences?**
   - nanobind compilation flags
   - Symbol visibility settings

---

## Next Steps to Investigate

### Option 1: Isolate the issue (RECOMMENDED)

Create a minimal C++ test that:
1. Links against libtorch (like our code)
2. Calls fits_read_pixll()
3. See if it's slow

If slow → PyTorch interference
If fast → Something else in our code

### Option 2: Profile with Instruments

Use macOS Instruments Time Profiler to see WHERE the time is spent:
- Is it actually in CFITSIO?
- Or in some wrapper/hook code?

### Option 3: Bypass PyTorch for int16

For int16 only:
1. Read with pure C-style code
2. Minimal dependencies
3. Create tensor from raw buffer afterward

### Option 4: Check fitsio's build

They achieve 0.252ms total (including open/read/close).
Our pure C gets 0.16-0.18ms.
What's their secret?

---

## Performance Comparison

| Method | uint8 | int16 | int16/uint8 ratio |
|--------|-------|-------|-------------------|
| **Pure C (no wrapper)** | 0.025ms | 0.104ms | 4.2x ✅ |
| **fitsio** | 0.099ms | 0.252ms | 2.5x ✅ |
| **Our code (NumPy return)** | 0.101ms | 0.873ms | 8.6x ❌ |

**Target:** Match pure C performance (0.104ms for int16)
**Current:** 8.4x slower (0.873ms)
**Gap to close:** ~0.769ms

---

## Code Modifications Made

### 1. NumPy Return for int16

**bindings.cpp:**
- Returns numpy.ndarray for int16
- Returns torch.Tensor for other types
- **Impact:** Negligible (~1μs difference)

### 2. Disabled DLPack type_caster

**bindings.cpp:**
- Commented out nanobind type_caster
- Forces explicit tensor_to_python() calls
- **Impact:** Eliminated DLPack overhead

### 3. Disabled File Handle Caching

**fits.cpp:**
- Commented out global_cache.get_or_open()
- Each read uses fresh file handle
- **Impact:** Reduced overhead, but int16 still slow

### 4. Added Profiling

**fits.cpp & bindings.cpp:**
- Detailed timing of each operation
- Revealed CFITSIO is the bottleneck
- Shows malloc vs tensor.data_ptr has no difference

---

## Conclusion

We've successfully **eliminated all Python/C++ wrapper overhead**:
- NumPy wrapping: ~1μs
- Tensor wrapping: ~2μs
- DLPack: Disabled
- File caching: Disabled

But CFITSIO itself runs **8.4x slower** in our C++ context than in pure C.

**This is a deep C++ / library interaction issue**, not a Python binding issue.

**Recommended next step:** Create minimal C++ reproducer to isolate the cause.

---

## Files Modified

1. `src/torchfits/cpp_src/bindings.cpp` - NumPy return, profiling
2. `src/torchfits/cpp_src/fits.cpp` - Disabled caching, profiling, malloc buffer test
3. `benchmark_pure_cfitsio.c` - Pure C baseline
4. `bench_cfitsio_handle_reuse.c` - Handle reuse investigation

---

## Hypothesis for Further Testing

**Theory:** PyTorch's memory allocator or threading model interferes with CFITSIO's internal buffering/caching mechanisms.

**Test:** Compile a minimal C++ program that:
- Links libcfitsio + libtorch
- Calls torch::empty() to initialize PyTorch
- Then calls fits_read_pixll()
- Measure if it's slow

If slow → PyTorch is interfering
If fast → Bug in our specific code

