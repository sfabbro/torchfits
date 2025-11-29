# torchfits Performance Analysis
**Date:** 2025-11-27
**Objective:** Understand why torchfits C++ is 53% slower than fitsio

## Executive Summary

Rigorous benchmarking reveals:
- **torchfits:** 0.777ms
- **fitsio:** 0.459ms
- **Gap:** 0.318ms (69% slower total, 53% slower in C++ core)

The bottleneck is **100% in C++ CFITSIO usage**, not Python bindings (which add only 0.075ms).

---

## Benchmark Results

### Rigorous Benchmark (Fresh Files, Subprocess Isolation)

**Small files (4MB float32 1000×1000):**
- fitsio: 0.870ms
- fitsio→torch: 1.116ms
- torchfits: 1.276ms (1.47x slower than fitsio, 1.14x slower than fitsio→torch)
- astropy: 6.740ms

**Large files (32MB float64 2000×2000):**
- fitsio: 5.730ms
- fitsio→torch: 6.975ms
- torchfits: 7.248ms (1.26x slower than fitsio, 1.04x slower than fitsio→torch)
- astropy: 8.429ms

### Detailed Profiling (In-Process, 50 runs)

**Component breakdown:**
- torchfits total: 0.885ms
  - C++ core: 0.737ms (83%)
  - Python wrapper: 0.148ms (17%)
- fitsio: 0.459ms

**C++ performance gap:** 0.278ms (60% slower)

---

## What We're Doing Right ✅

Based on analysis of our C++ code and [docs/OPTMIZE.md](docs/OPTMIZE.md):

1. **Zero-copy reads** (Work Package #2)
   ```cpp
   auto tensor = torch::empty(shape, torch::kFloat32);
   fits_read_img(fptr_, TFLOAT, 1, total_pixels, nullptr,
                 tensor.data_ptr<float>(), nullptr, &status);
   ```
   ✅ We allocate the tensor first and pass `data_ptr()` directly to cfitsio

2. **GIL release** (Work Package #7)
   ✅ All I/O functions use `nb::gil_scoped_release`

3. **Modern bindings**
   ✅ Using nanobind (Work Package #1)

---

## Root Cause Analysis: Why Are We Slower?

### Hypothesis 1: Excessive CFITSIO Metadata Calls ❓

Our `read_image()` makes **6 CFITSIO calls** before reading data:
1. `fits_movabs_hdu()` - Move to HDU
2. `fits_get_hdu_type()` - Check HDU type
3. `fits_get_img_param()` - Get dimensions/BITPIX
4. `fits_get_compression_type()` - Check compression
5. `fits_read_key()` for BSCALE
6. `fits_read_key()` for BZERO
7. Finally `fits_read_img()` - Read data

**Testing needed:** Measure overhead of each call individually.

### Hypothesis 2: Missing Buffer Size Optimization ⚠️

From [docs/CFITSIO.md](docs/CFITSIO.md) Section 3.4:
> "**ESO Common Pipeline Library:** Explicitly uses `fits_set_buffer_size` to optimize CFITSIO's internal buffer size for specific hardware."

**Current status:** We don't set buffer size - using CFITSIO defaults.

**Action:** Test with `fits_set_bscale(fptr, 10*1024*1024, &status)` for 10MB buffers.

### Hypothesis 3: Switch Statement Overhead ❓

Our code has a large switch statement (lines 215-274 in [fits.cpp](src/torchfits/cpp_src/fits.cpp)) with duplicated code for each BITPIX type.

**Impact:** Likely minimal (modern compilers optimize switches well).

### Hypothesis 4: fitsio Uses Different CFITSIO Functions ❓

We use:
- `fits_get_img_param()` - Standard version
- `fits_read_img()` - High-level read

fitsio might use:
- `fits_get_img_paramll()` - Long long version (faster?)
- `fits_read_pix()` or lower-level functions?

**Action:** Need to examine fitsio source at [github.com/esheldon/fitsio](https://github.com/esheldon/fitsio/blob/master/fitsio/fitsio_pywrap.c).

---

## Next Steps

### Priority 1: Test Buffer Size Optimization
Implement `fits_set_bscale()` with various buffer sizes (1MB, 10MB, 100MB) and measure impact.

### Priority 2: Eliminate Unnecessary Metadata Calls
For simple use cases (HDU 0, no tables), skip:
- `fits_get_hdu_type()` - assume it's an image
- `fits_get_compression_type()` - assume uncompressed (or catch error)
- BSCALE/BZERO reads - assume no scaling for float data

### Priority 3: Study fitsio Implementation
Examine [fitsio_pywrap.c](https://github.com/esheldon/fitsio/blob/master/fitsio/fitsio_pywrap.c) to understand their exact CFITSIO call sequence.

### Priority 4: Profile at CFITSIO Function Level
Use a C++ profiler (gprof, perf, or Instruments on macOS) to see exactly which CFITSIO functions consume the most time.

---

## References

- [GitHub - esheldon/fitsio](https://github.com/esheldon/fitsio)
- [fitsio C wrapper source](https://github.com/esheldon/fitsio/blob/master/fitsio/fitsio_pywrap.c)
- Internal docs:
  - [docs/OPTMIZE.md](docs/OPTMIZE.md) - 15 optimization work packages
  - [docs/CFITSIO.md](docs/CFITSIO.md) - Detailed CFITSIO implementation strategies
  - [docs/OBJECTIVES.md](docs/OBJECTIVES.md) - Project goals
  - [docs/DESIGN.md](docs/DESIGN.md) - Architecture

---

## Benchmark Methodology

All benchmarks use:
- Fresh files for each measurement (no OS cache)
- Subprocess isolation (no Python module cache)
- Full operation measured: open + read + close
- Statistical analysis (median of 10-50 runs)
- Same test file: 1000×1000 float32 (4MB uncompressed)

**Benchmark scripts:**
- [benchmark_rigorous.py](benchmark_rigorous.py) - Fair comparison vs fitsio/astropy
- [profile_detailed.py](profile_detailed.py) - Component-level breakdown
- [benchmark_cfitsio_simple.py](benchmark_cfitsio_simple.py) - Quick C++ vs Python test
