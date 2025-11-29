#!/usr/bin/env python3
"""
Test if handle reuse affects performance differently for int16 vs uint8.
"""
import time
import statistics
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits
from torchfits import cpp

def create_test_file(dtype_str):
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"handle_test_{dtype_str}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_handle_reuse(filepath, iterations=100):
    """Benchmark with handle reuse (open once, read many times)."""
    times = []
    handle = cpp.open_fits_file(filepath, 'r')

    for _ in range(iterations):
        start = time.perf_counter()
        tensor = cpp.read_full(handle, 0)
        times.append((time.perf_counter() - start) * 1000)

    cpp.close_fits_file(handle)
    return statistics.median(times)

def benchmark_fresh_opens(filepath, iterations=100):
    """Benchmark with fresh open/close each time."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        handle = cpp.open_fits_file(filepath, 'r')
        tensor = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times)

def main():
    print("=" * 80)
    print("HANDLE REUSE vs FRESH OPENS COMPARISON")
    print("=" * 80)
    print()

    # Create test files
    print("Creating test files...")
    uint8_file = create_test_file('uint8')
    int16_file = create_test_file('int16')
    print()

    # Test 1: Handle reuse
    print("1. With handle reuse (open once, read 100x):")
    print("-" * 80)
    uint8_reuse = benchmark_handle_reuse(uint8_file)
    int16_reuse = benchmark_handle_reuse(int16_file)
    print(f"   uint8:  {uint8_reuse:.4f}ms")
    print(f"   int16:  {int16_reuse:.4f}ms")
    print(f"   Ratio:  {int16_reuse/uint8_reuse:.2f}x")
    print()

    # Test 2: Fresh opens
    print("2. With fresh opens (open/read/close each iteration):")
    print("-" * 80)
    uint8_fresh = benchmark_fresh_opens(uint8_file)
    int16_fresh = benchmark_fresh_opens(int16_file)
    print(f"   uint8:  {uint8_fresh:.4f}ms")
    print(f"   int16:  {int16_fresh:.4f}ms")
    print(f"   Ratio:  {int16_fresh/uint8_fresh:.2f}x")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    print(f"Handle reuse performance:")
    print(f"  uint8:  {uint8_reuse:.4f}ms")
    print(f"  int16:  {int16_reuse:.4f}ms")
    print(f"  Ratio:  {int16_reuse/uint8_reuse:.2f}x")
    print()

    print(f"Fresh opens performance:")
    print(f"  uint8:  {uint8_fresh:.4f}ms")
    print(f"  int16:  {int16_fresh:.4f}ms")
    print(f"  Ratio:  {int16_fresh/uint8_fresh:.2f}x")
    print()

    print(f"Difference:")
    if abs((int16_reuse/uint8_reuse) - (int16_fresh/uint8_fresh)) > 1.0:
        print(f"  ⚠️  Handle reuse shows different ratio than fresh opens!")
        print(f"  This suggests CFITSIO caching behaves differently for int16 vs uint8")
    else:
        print(f"  ✅ Similar ratios - handle reuse doesn't affect dtype performance")

    # Cleanup
    Path(uint8_file).unlink()
    Path(int16_file).unlink()

if __name__ == "__main__":
    main()
