#!/usr/bin/env python3
"""
Direct comparison of our CFITSIO calls vs fitsio
"""
import time
import statistics
import numpy as np
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import fitsio

def create_test_file(dtype_str):
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"compare_detailed_{dtype_str}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)

    astropy_fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark(func, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    # Remove outliers
    times_sorted = sorted(times)
    n_remove = max(1, iterations // 10)
    times_trimmed = times_sorted[n_remove:-n_remove]

    return statistics.median(times_trimmed)

print("="*80)
print("DETAILED CFITSIO COMPARISON")
print("="*80)
print()

# Create test files
uint8_file = create_test_file('uint8')
int16_file = create_test_file('int16')

# Pre-open handles to exclude file opening overhead
uint8_handle = cpp.open_fits_file(uint8_file, 'r')
int16_handle = cpp.open_fits_file(int16_file, 'r')

print("1. Our C++ read (handle pre-opened):")
print("-"*80)

# Warm up the cache
for _ in range(10):
    cpp.read_full(uint8_handle, 0)
    cpp.read_full(int16_handle, 0)

uint8_time = benchmark(lambda: cpp.read_full(uint8_handle, 0), 100)
int16_time = benchmark(lambda: cpp.read_full(int16_handle, 0), 100)

print(f"  uint8:  {uint8_time:.4f}ms")
print(f"  int16:  {int16_time:.4f}ms")
print(f"  Ratio:  {int16_time / uint8_time:.2f}x")
print()

cpp.close_fits_file(uint8_handle)
cpp.close_fits_file(int16_handle)

print("2. fitsio read (includes file open/close):")
print("-"*80)

# Warm up
for _ in range(10):
    fitsio.read(uint8_file)
    fitsio.read(int16_file)

fitsio_uint8_time = benchmark(lambda: fitsio.read(uint8_file), 100)
fitsio_int16_time = benchmark(lambda: fitsio.read(int16_file), 100)

print(f"  uint8:  {fitsio_uint8_time:.4f}ms")
print(f"  int16:  {fitsio_int16_time:.4f}ms")
print(f"  Ratio:  {fitsio_int16_time / fitsio_uint8_time:.2f}x")
print()

print("="*80)
print("ANALYSIS")
print("="*80)
print()

print(f"Our int16 time: {int16_time:.4f}ms")
print(f"fitsio int16 time: {fitsio_int16_time:.4f}ms")
print(f"Ratio (ours/fitsio): {int16_time / fitsio_int16_time:.2f}x")
print()

print("Profiling shows:")
print("  - CFITSIO read: ~497μs (0.497ms)")
print("  - NumPy wrap: ~0.2μs (negligible)")
print("  - Total measured: ~0.559ms")
print()

print("This suggests:")
if int16_time > 0.5:
    print("  ✓ CFITSIO is taking most of the time (as expected)")
    if fitsio_int16_time < 0.2:
        print("  ⚠️  But fitsio is MUCH faster - they must be doing something different!")
        print("     Possible reasons:")
        print("     - Different CFITSIO function")
        print("     - Different buffer settings")
        print("     - Caching we're not seeing")
        print("     - Compiler optimizations")

# Cleanup
Path(uint8_file).unlink()
Path(int16_file).unlink()
