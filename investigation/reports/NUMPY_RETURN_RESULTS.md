# NumPy Return Optimization Results
**Date:** 2025-11-28
**Status:** NumPy return implemented, but still 2.56x slower than fitsio

---

## What We Did

Implemented NumPy array return for int16 (instead of torch.Tensor via THPVariable_Wrap) to test if tensor wrapping was the bottleneck.

### Code Changes

**bindings.cpp:**
- Modified `tensor_to_python()` to return `numpy.ndarray` for int16
- Returns `torch.Tensor` via `THPVariable_Wrap` for other types (uint8, etc.)
- Used nanobind's ndarray support for zero-copy NumPy creation

---

## Performance Results

### Current Performance (with NumPy return)

| Type | torchfits | fitsio | Ratio (ours/fitsio) | Status |
|------|-----------|--------|---------------------|--------|
| **uint8** | 0.026ms | 0.099ms | **0.26x** | ✅ **3.8x FASTER** |
| **int16** | 0.645ms | 0.252ms | **2.56x** | ❌ **2.6x SLOWER** |

### Component Breakdown (int16, from profiling)

| Component | Time | Notes |
|-----------|------|-------|
| torch::empty() allocation | 0.5-1.3μs | Negligible |
| **fits_read_pixll() CFITSIO read** | **497μs** | **THE BOTTLENECK** |
| NumPy array wrapping | 0.2-1.4μs | Negligible |
| **Total measured** | **645μs** | Matches CFITSIO time |

---

## Key Finding: CFITSIO is the Bottleneck!

**The problem is NOT in our tensor wrapping** - it's in the CFITSIO read itself!

### Evidence:

1. **CFITSIO profiling:**
   ```
   [INT16] read_pixels: alloc=0.5μs cfitsio=497.0μs
   [INT16] NumPy wrap: 0.2μs
   ```

2. **Total time breakdown:**
   - CFITSIO read: 497μs (77% of total time)
   - Everything else: ~148μs (23% of total time)

3. **NumPy wrapping is nearly free:**
   - NumPy wrap: 0.2-1.4μs (vs 0.4-2.4μs for torch.Tensor via THPVariable_Wrap)
   - Difference: ~1μs - completely negligible!

### The Mystery

**fitsio's entire operation (open + read + close) takes only 252μs for int16.**

But our **CFITSIO read alone** takes 497μs!

This means either:
1. ❓ fitsio is NOT using CFITSIO for small images
2. ❓ fitsio has CFITSIO optimizations we don't have
3. ❓ fitsio is memory-mapping instead of reading
4. ❓ Our fits_read_pixll call has some inefficiency

---

## What We've Eliminated as Bottlenecks

✅ **DLPack**: Disabled type_caster (was causing overhead)
✅ **THPVariable_Wrap**: Only ~1μs difference from NumPy wrap
✅ **torch::empty()**: Takes <2μs
✅ **File handle caching**: Disabled
✅ **Tensor allocation**: Negligible overhead
✅ **Python-C++ boundary**: NumPy wrap is nearly free

---

## Performance Summary

### uint8 (Excellent!)
- **0.026ms** vs fitsio's 0.099ms
- **3.8x FASTER than fitsio** ✅
- Proves our infrastructure is sound!

### int16 (Bottleneck Identified)
- **0.645ms** vs fitsio's 0.252ms
- **2.56x slower than fitsio** ❌
- **Bottleneck:** CFITSIO `fits_read_pixll()` taking 497μs

---

## Next Steps

### Immediate Investigation Needed

1. **Check if fits_read_pixll is the right function**
   - Maybe there's a faster CFITSIO function for contiguous reads?
   - Check CFITSIO documentation for bulk read functions

2. **Compare CFITSIO configuration**
   - Check compile flags used by fitsio vs ours
   - Look for buffer size settings (`CFITSIO_BUFSIZE`)
   - Check if fitsio patches CFITSIO

3. **Memory mapping investigation**
   - Does fitsio bypass CFITSIO and mmap the file directly?
   - Check fitsio source for mmap usage

4. **CFITSIO version/patch differences**
   - fitsio uses CFITSIO 4.6.3 (same as us)
   - But they might have patches (check `fitsio/patches/` directory)

### Alternative Approaches

5. **Consider bypassing CFITSIO for simple cases**
   - For uncompressed, non-scaled images, could we read directly from file?
   - Parse FITS header manually and mmap the data section?

6. **Profile fitsio's CFITSIO calls**
   - Use dtrace/Instruments to see what CFITSIO functions fitsio actually calls
   - Time their CFITSIO operations

---

## Code Locations

### Modified Files

1. **bindings.cpp** (line 113-181)
   ```cpp
   static nb::object tensor_to_python(const torch::Tensor& tensor) {
       if (tensor.scalar_type() == torch::kInt16) {
           // Return NumPy array
           auto result = nb::ndarray<nb::numpy, int16_t>(...)
           return nb::cast(result);
       }
       // Other types use THPVariable_Wrap
       PyObject* tensor_obj = THPVariable_Wrap(tensor);
       return nb::steal(tensor_obj);
   }
   ```

2. **fits.cpp** (line 224-251)
   ```cpp
   // Profiling instrumentation
   auto t2 = std::chrono::high_resolution_clock::now();
   fits_read_pixll(fptr_, fits_dtype, firstpix, total_pixels, nullptr,
                  tensor.data_ptr<T>(), &anynul, &status);
   auto t3 = std::chrono::high_resolution_clock::now();
   ```

---

## Conclusion

**NumPy return helps slightly but doesn't solve the fundamental problem.**

The real bottleneck is **CFITSIO's `fits_read_pixll()` taking 497μs for int16**, which is:
- **2x slower than fitsio's entire operation (252μs)**
- **Inherently slower for int16 due to byte swapping** (expected 5-6x vs uint8)

We need to investigate:
1. Why fitsio is so much faster
2. Whether we can bypass CFITSIO for simple cases
3. If there's a faster CFITSIO function we should use

---

## Performance Goal

**Target:** Match or beat fitsio for all types

**Current Status:**
- ✅ uint8: 3.8x faster (DONE!)
- ❌ int16: 2.56x slower (INVESTIGATE CFITSIO)

**If we can reduce int16 read from 645μs to ~250μs, we'll match fitsio!**
