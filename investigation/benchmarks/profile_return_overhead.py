#!/usr/bin/env python3
"""
Profile the overhead of our tensor return path vs fitsio's numpy return.
This isolates where the int16 overhead comes from.
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
    filepath = tmpdir / f"profile_return_{dtype_str}_{time.time_ns()}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    astropy_fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_component(func, iterations=100):
    """Benchmark with outlier removal."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - start) * 1000)

    # Remove outliers (top/bottom 10%)
    times_sorted = sorted(times)
    n_remove = max(1, iterations // 10)
    times_trimmed = times_sorted[n_remove:-n_remove]

    return statistics.median(times_trimmed)

def main():
    print("=" * 80)
    print("PROFILING TENSOR RETURN OVERHEAD")
    print("=" * 80)
    print()

    # Create test files
    print("Creating test files...")
    uint8_file = create_test_file('uint8')
    int16_file = create_test_file('int16')
    print()

    # Component 1: Pure CFITSIO read (using cpp module directly)
    print("1. Pure C++ read (CFITSIO + torch::empty + THPVariable_Wrap):")
    print("-" * 80)

    def read_uint8():
        handle = cpp.open_fits_file(uint8_file, 'r')
        tensor = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        return tensor

    def read_int16():
        handle = cpp.open_fits_file(int16_file, 'r')
        tensor = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        return tensor

    uint8_cpp = benchmark_component(read_uint8, 100)
    int16_cpp = benchmark_component(read_int16, 100)

    print(f"  uint8:  {uint8_cpp:.4f}ms")
    print(f"  int16:  {int16_cpp:.4f}ms")
    print(f"  Ratio:  {int16_cpp/uint8_cpp:.2f}x")
    print()

    # Component 2: fitsio read (CFITSIO + numpy array)
    print("2. fitsio read (CFITSIO + numpy array):")
    print("-" * 80)

    def read_fitsio_uint8():
        return fitsio.read(uint8_file)

    def read_fitsio_int16():
        return fitsio.read(int16_file)

    uint8_fitsio = benchmark_component(read_fitsio_uint8, 100)
    int16_fitsio = benchmark_component(read_fitsio_int16, 100)

    print(f"  uint8:  {uint8_fitsio:.4f}ms")
    print(f"  int16:  {int16_fitsio:.4f}ms")
    print(f"  Ratio:  {int16_fitsio/uint8_fitsio:.2f}x")
    print()

    # Component 3: Tensor creation from numpy
    print("3. Converting numpy → torch.Tensor:")
    print("-" * 80)

    # Pre-load data
    numpy_uint8 = fitsio.read(uint8_file)
    numpy_int16 = fitsio.read(int16_file)

    def convert_uint8():
        return torch.from_numpy(numpy_uint8)

    def convert_int16():
        return torch.from_numpy(numpy_int16)

    uint8_convert = benchmark_component(convert_uint8, 1000)
    int16_convert = benchmark_component(convert_int16, 1000)

    print(f"  uint8:  {uint8_convert:.4f}ms")
    print(f"  int16:  {int16_convert:.4f}ms")
    print(f"  Ratio:  {int16_convert/uint8_convert:.2f}x")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    print("Performance Summary:")
    print(f"  torchfits (uint8):  {uint8_cpp:.4f}ms")
    print(f"  torchfits (int16):  {int16_cpp:.4f}ms")
    print(f"  fitsio (uint8):     {uint8_fitsio:.4f}ms")
    print(f"  fitsio (int16):     {int16_fitsio:.4f}ms")
    print()

    print("Ratio Analysis:")
    print(f"  torchfits int16/uint8 ratio: {int16_cpp/uint8_cpp:.2f}x")
    print(f"  fitsio int16/uint8 ratio:    {int16_fitsio/uint8_fitsio:.2f}x")
    print(f"  Extra overhead factor:       {(int16_cpp/uint8_cpp) / (int16_fitsio/uint8_fitsio):.2f}x")
    print()

    print("vs fitsio:")
    print(f"  torchfits/fitsio (uint8): {uint8_cpp/uint8_fitsio:.2f}x")
    print(f"  torchfits/fitsio (int16): {int16_cpp/int16_fitsio:.2f}x")
    print()

    # Calculate where the overhead is
    overhead_uint8 = uint8_cpp - uint8_fitsio
    overhead_int16 = int16_cpp - int16_fitsio
    extra_int16_overhead = overhead_int16 - overhead_uint8

    print("Overhead Breakdown:")
    print(f"  uint8 overhead (torchfits - fitsio): {overhead_uint8:.4f}ms")
    print(f"  int16 overhead (torchfits - fitsio): {overhead_int16:.4f}ms")
    print(f"  Extra int16-specific overhead:       {extra_int16_overhead:.4f}ms")
    print()

    if extra_int16_overhead > 0.1:
        print(f"⚠️  WARNING: int16 has {extra_int16_overhead:.4f}ms extra overhead!")
        print("This suggests dtype-specific overhead in our return path:")
        print("  - THPVariable_Wrap may be slower for int16")
        print("  - torch::Tensor creation may have int16-specific costs")
        print("  - NumPy array → torch.Tensor conversion is fast (see test 3)")
    else:
        print("✅ Overhead is consistent across types.")
        print("The issue may be in CFITSIO configuration or build flags.")

    # Cleanup
    Path(uint8_file).unlink()
    Path(int16_file).unlink()

if __name__ == "__main__":
    main()
