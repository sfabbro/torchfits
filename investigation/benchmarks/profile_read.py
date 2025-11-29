#!/usr/bin/env python3
"""Profile where time is spent in torchfits.read()"""
import torchfits
import torchfits.cpp as cpp
import numpy as np
from astropy.io import fits as astropy_fits
import fitsio
from pathlib import Path
import tempfile
import time

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / "profile_test.fits"
data = np.random.randn(1000, 1000).astype(np.float32)
astropy_fits.writeto(filepath, data, overwrite=True)

print("Profiling torchfits.read() breakdown")
print("=" * 70)

# Clear cache
torchfits.clear_file_cache()

# Time each step
times = {}

# Total Python-level read
start = time.perf_counter()
result, header = torchfits.read(str(filepath))
times['total_python'] = time.perf_counter() - start

# Now measure C++ components directly
torchfits.clear_file_cache()

start = time.perf_counter()
handle = cpp.open_fits_file(str(filepath), "r")
times['open_file'] = time.perf_counter() - start

start = time.perf_counter()
header = cpp.read_header(handle, 0)
times['read_header'] = time.perf_counter() - start

start = time.perf_counter()
data_cpp = cpp.read_full(handle, 0)
times['read_full_cpp'] = time.perf_counter() - start

start = time.perf_counter()
cpp.close_fits_file(handle)
times['close_file'] = time.perf_counter() - start

# Compare to direct fitsio
start = time.perf_counter()
fitsio_data = fitsio.read(str(filepath))
times['fitsio_total'] = time.perf_counter() - start

print("\nTiming breakdown (ms):")
print("-" * 70)
for key, val in times.items():
    print(f"{key:20s}: {val*1000:8.3f}ms")

print("\nAnalysis:")
print("-" * 70)
cpp_overhead = times['total_python'] - times['read_full_cpp']
print(f"C++ read_full:        {times['read_full_cpp']*1000:.3f}ms")
print(f"Python overhead:      {cpp_overhead*1000:.3f}ms")
print(f"fitsio total:         {times['fitsio_total']*1000:.3f}ms")
print()

if times['read_full_cpp'] > times['fitsio_total']:
    slowdown = times['read_full_cpp'] / times['fitsio_total']
    print(f"❌ C++ read is {slowdown:.2f}x slower than fitsio")
    print("   → Need to optimize C++ CFITSIO usage")
else:
    speedup = times['fitsio_total'] / times['read_full_cpp']
    print(f"✅ C++ read is {speedup:.2f}x faster than fitsio")

if cpp_overhead > times['read_full_cpp']:
    print(f"❌ Python overhead ({cpp_overhead*1000:.3f}ms) is larger than C++ read!")
    print("   → Need to optimize Python wrapper code")
