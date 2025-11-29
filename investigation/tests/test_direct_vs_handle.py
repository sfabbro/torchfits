#!/usr/bin/env python3
"""Test direct C++ call vs handle-based API"""
import torchfits.cpp as cpp
import numpy as np
from astropy.io import fits
from pathlib import Path
import tempfile
import time

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / "test_api_overhead.fits"
data = np.random.randn(1000, 1000).astype(np.float32)
fits.writeto(filepath, data, overwrite=True)

print("Testing Python<->C++ crossing overhead")
print("=" * 70)

# Method 1: Handle-based API (current Python code uses this)
times = []
for _ in range(20):
    start = time.perf_counter()
    handle = cpp.open_fits_file(str(filepath), "r")
    header = cpp.read_header(handle, 0)
    hdu_type = cpp.get_hdu_type(handle, 0)
    data = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

handle_based = np.median(times) * 1000

# Method 2: Direct API (single C++ call)
times = []
for _ in range(20):
    start = time.perf_counter()
    data = cpp.read_full(str(filepath), 0)  # Direct call
    elapsed = time.perf_counter() - start
    times.append(elapsed)

direct = np.median(times) * 1000

print(f"Handle-based API (5 C++ calls):  {handle_based:.3f}ms")
print(f"Direct API (1 C++ call):         {direct:.3f}ms")
print(f"Overhead from extra calls:       {handle_based - direct:.3f}ms")
print()

if direct < handle_based:
    speedup = handle_based / direct
    print(f"✅ Direct API is {speedup:.2f}x faster")
    print(f"   Savings: {handle_based - direct:.3f}ms per read")
else:
    print(f"⚠️  No significant difference")
