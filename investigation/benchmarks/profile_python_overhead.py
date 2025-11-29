#!/usr/bin/env python3
"""Identify Python overhead sources in torchfits.read()"""
import torchfits
import torchfits.cpp as cpp
import numpy as np
from astropy.io import fits as astropy_fits
from pathlib import Path
import tempfile
import time
import torch

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / "overhead_test.fits"
data = np.random.randn(1000, 1000).astype(np.float32)
astropy_fits.writeto(filepath, data, overwrite=True)

print("Identifying Python overhead sources")
print("=" * 70)

# Minimal C++ path (just read, no cache, no device transfer)
def minimal_cpp_read(path):
    handle = cpp.open_fits_file(path, "r")
    data = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
    return data

# Test overhead of each Python operation
iterations = 20

# Baseline: minimal C++ read
times = []
for _ in range(iterations):
    torchfits.clear_file_cache()
    start = time.perf_counter()
    data = minimal_cpp_read(str(filepath))
    elapsed = time.perf_counter() - start
    times.append(elapsed)
baseline = np.median(times) * 1000

# With cache key generation
def read_with_cache_key(path):
    cache_key = f"{path}:0:cpu:False:False:None:1:-1"  # Simulate cache key
    handle = cpp.open_fits_file(path, "r")
    data = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
    return data

times = []
for _ in range(iterations):
    torchfits.clear_file_cache()
    start = time.perf_counter()
    data = read_with_cache_key(str(filepath))
    elapsed = time.perf_counter() - start
    times.append(elapsed)
with_cache_key = np.median(times) * 1000

# With header read
def read_with_header(path):
    handle = cpp.open_fits_file(path, "r")
    header = cpp.read_header(handle, 0)  # Add header read
    data = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
    return data, header

times = []
for _ in range(iterations):
    torchfits.clear_file_cache()
    start = time.perf_counter()
    data, header = read_with_header(str(filepath))
    elapsed = time.perf_counter() - start
    times.append(elapsed)
with_header = np.median(times) * 1000

# With HDU type check
def read_with_hdu_check(path):
    handle = cpp.open_fits_file(path, "r")
    hdu_type = cpp.get_hdu_type(handle, 0)  # Add HDU type check
    header = cpp.read_header(handle, 0)
    data = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
    return data, header

times = []
for _ in range(iterations):
    torchfits.clear_file_cache()
    start = time.perf_counter()
    data, header = read_with_hdu_check(str(filepath))
    elapsed = time.perf_counter() - start
    times.append(elapsed)
with_hdu_check = np.median(times) * 1000

# Full torchfits.read() (no cache hits)
times = []
for _ in range(iterations):
    torchfits.clear_file_cache()
    start = time.perf_counter()
    data, header = torchfits.read(str(filepath))
    elapsed = time.perf_counter() - start
    times.append(elapsed)
full_read = np.median(times) * 1000

print("\nTiming breakdown (median of 20 runs):")
print("-" * 70)
print(f"Baseline (minimal C++):      {baseline:.3f}ms")
print(f"+ cache key generation:      {with_cache_key:.3f}ms  (+{with_cache_key-baseline:.3f}ms)")
print(f"+ header read:               {with_header:.3f}ms  (+{with_header-with_cache_key:.3f}ms)")
print(f"+ HDU type check:            {with_hdu_check:.3f}ms  (+{with_hdu_check-with_header:.3f}ms)")
print(f"Full torchfits.read():       {full_read:.3f}ms  (+{full_read-with_hdu_check:.3f}ms)")
print()
print(f"Total Python overhead:       {full_read - baseline:.3f}ms")
print(f"Python overhead percentage:  {(full_read - baseline) / full_read * 100:.1f}%")
