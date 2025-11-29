#!/usr/bin/env python3
"""
Detailed component-level profiling to find the 1.128ms overhead.
Times each operation separately to isolate the bottleneck.
"""
import time
import statistics
import torch
import numpy as np
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import fitsio

def create_test_file(dtype_str):
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"component_breakdown_{dtype_str}_{time.time_ns()}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    astropy_fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def time_operation(func, iterations=100):
    """Time an operation with outlier removal."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - start) * 1000)

    # Remove outliers
    times_sorted = sorted(times)
    n_remove = max(1, iterations // 10)
    times_trimmed = times_sorted[n_remove:-n_remove]

    return statistics.median(times_trimmed), result

def main():
    print("=" * 80)
    print("DETAILED COMPONENT BREAKDOWN")
    print("Isolating the 1.128ms overhead in int16 read path")
    print("=" * 80)
    print()

    # Create test files
    print("Creating test files...")
    uint8_file = create_test_file('uint8')
    int16_file = create_test_file('int16')
    print()

    # Test 1: File opening
    print("1. File opening (cpp.open_fits_file):")
    print("-" * 80)

    def open_uint8():
        handle = cpp.open_fits_file(uint8_file, 'r')
        cpp.close_fits_file(handle)
        return handle

    def open_int16():
        handle = cpp.open_fits_file(int16_file, 'r')
        cpp.close_fits_file(handle)
        return handle

    uint8_open_time, _ = time_operation(open_uint8, 100)
    int16_open_time, _ = time_operation(open_int16, 100)

    print(f"  uint8:  {uint8_open_time:.4f}ms")
    print(f"  int16:  {int16_open_time:.4f}ms")
    print(f"  Diff:   {int16_open_time - uint8_open_time:.4f}ms")
    print()

    # Test 2: Just the read_full call (file already open)
    print("2. Read operation only (cpp.read_full with pre-opened handle):")
    print("-" * 80)

    # Pre-open handles
    uint8_handle = cpp.open_fits_file(uint8_file, 'r')
    int16_handle = cpp.open_fits_file(int16_file, 'r')

    def read_uint8():
        return cpp.read_full(uint8_handle, 0)

    def read_int16():
        return cpp.read_full(int16_handle, 0)

    uint8_read_time, uint8_tensor = time_operation(read_uint8, 100)
    int16_read_time, int16_tensor = time_operation(read_int16, 100)

    print(f"  uint8:  {uint8_read_time:.4f}ms")
    print(f"  int16:  {int16_read_time:.4f}ms")
    print(f"  Diff:   {int16_read_time - uint8_read_time:.4f}ms")
    print(f"  Ratio:  {int16_read_time / uint8_read_time:.2f}x")

    # Verify tensor types
    print(f"  uint8 type: {type(uint8_tensor).__name__}, dtype: {uint8_tensor.dtype}")
    print(f"  int16 type: {type(int16_tensor).__name__}, dtype: {int16_tensor.dtype}")
    print()

    cpp.close_fits_file(uint8_handle)
    cpp.close_fits_file(int16_handle)

    # Test 3: Full path (open + read + close)
    print("3. Full path (open + read + close):")
    print("-" * 80)

    def full_uint8():
        handle = cpp.open_fits_file(uint8_file, 'r')
        tensor = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        return tensor

    def full_int16():
        handle = cpp.open_fits_file(int16_file, 'r')
        tensor = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        return tensor

    uint8_full_time, _ = time_operation(full_uint8, 100)
    int16_full_time, _ = time_operation(full_int16, 100)

    print(f"  uint8:  {uint8_full_time:.4f}ms")
    print(f"  int16:  {int16_full_time:.4f}ms")
    print(f"  Diff:   {int16_full_time - uint8_full_time:.4f}ms")
    print(f"  Ratio:  {int16_full_time / uint8_full_time:.2f}x")
    print()

    # Test 4: Compare with fitsio
    print("4. fitsio baseline:")
    print("-" * 80)

    def fitsio_uint8():
        return fitsio.read(uint8_file)

    def fitsio_int16():
        return fitsio.read(int16_file)

    fitsio_uint8_time, _ = time_operation(fitsio_uint8, 100)
    fitsio_int16_time, _ = time_operation(fitsio_int16, 100)

    print(f"  uint8:  {fitsio_uint8_time:.4f}ms")
    print(f"  int16:  {fitsio_int16_time:.4f}ms")
    print(f"  Diff:   {fitsio_int16_time - fitsio_uint8_time:.4f}ms")
    print(f"  Ratio:  {fitsio_int16_time / fitsio_uint8_time:.2f}x")
    print()

    # Test 5: Pure tensor creation (no I/O)
    print("5. Pure tensor creation (torch.empty, no I/O):")
    print("-" * 80)

    shape = [1000, 1000]

    def create_uint8():
        return torch.empty(shape, dtype=torch.uint8)

    def create_int16():
        return torch.empty(shape, dtype=torch.int16)

    uint8_create_time, _ = time_operation(create_uint8, 1000)
    int16_create_time, _ = time_operation(create_int16, 1000)

    print(f"  uint8:  {uint8_create_time:.4f}ms")
    print(f"  int16:  {int16_create_time:.4f}ms")
    print(f"  Diff:   {int16_create_time - uint8_create_time:.4f}ms")
    print()

    # Analysis
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()

    print("Component breakdown:")
    print(f"  File opening overhead (int16 - uint8):    {int16_open_time - uint8_open_time:.4f}ms")
    print(f"  Read operation overhead (int16 - uint8):   {int16_read_time - uint8_read_time:.4f}ms")
    print(f"  Full path overhead (int16 - uint8):        {int16_full_time - uint8_full_time:.4f}ms")
    print()

    print("Where is the time going?")
    print(f"  Opening:     {int16_open_time:.4f}ms")
    print(f"  Reading:     {int16_read_time:.4f}ms (THE BOTTLENECK)")
    print(f"  Full path:   {int16_full_time:.4f}ms")
    print(f"  Expected:    Open + Read = {int16_open_time + int16_read_time:.4f}ms")
    print(f"  Actual:      {int16_full_time:.4f}ms")
    print(f"  Discrepancy: {int16_full_time - (int16_open_time + int16_read_time):.4f}ms")
    print()

    # The key finding
    extra_overhead = int16_read_time - uint8_read_time
    fitsio_diff = fitsio_int16_time - fitsio_uint8_time
    our_extra_overhead = extra_overhead - fitsio_diff

    print("KEY FINDING:")
    print(f"  fitsio int16 overhead:     {fitsio_diff:.4f}ms (CFITSIO inherent)")
    print(f"  torchfits int16 overhead:  {extra_overhead:.4f}ms")
    print(f"  OUR EXTRA OVERHEAD:        {our_extra_overhead:.4f}ms ← THE PROBLEM")
    print()

    if our_extra_overhead > 0.1:
        print(f"⚠️  We have {our_extra_overhead:.4f}ms extra overhead in cpp.read_full()!")
        print()
        print("This overhead is NOT in:")
        print("  - File opening (measured separately)")
        print("  - torch.empty() (measured as <0.01ms)")
        print(f"  - CFITSIO itself (fitsio has {fitsio_diff:.4f}ms)")
        print()
        print("This overhead MUST BE in:")
        print("  - THPVariable_Wrap (wrapping tensor for Python)")
        print("  - Or some hidden PyTorch overhead for int16")
        print("  - Or GIL interactions")
        print("  - Or memory allocation pattern issues")

    # Cleanup
    Path(uint8_file).unlink()
    Path(int16_file).unlink()

if __name__ == "__main__":
    main()
