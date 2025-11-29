#!/usr/bin/env python3
"""
Detailed profiling to understand WHERE the 0.4ms overhead comes from.

Break down torchfits into individual operations and measure each one.
"""
import time
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits
import statistics

def create_test_file():
    """Create a test file."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"profile_detail_{time.time_ns()}.fits"
    data = np.random.randn(1000, 1000).astype(np.float32)
    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def profile_torchfits_components(filepath, num_runs=50):
    """Profile individual components of torchfits.read()."""
    import torchfits
    import torchfits.cpp as cpp

    print("Profiling torchfits components:")
    print("=" * 80)

    results = {
        'cache_check': [],
        'open_file': [],
        'read_header': [],
        'read_data': [],
        'dlpack_conversion': [],
        'close_file': [],
        'total': []
    }

    for _ in range(num_runs):
        torchfits.clear_file_cache()

        # Total time
        start_total = time.perf_counter()

        # Cache check (in real code)
        start = time.perf_counter()
        cache_key = (filepath, 0, 'cpu', False, False, None, 1, -1)
        _ = cache_key in torchfits._file_cache  # noqa
        results['cache_check'].append(time.perf_counter() - start)

        # Open file
        start = time.perf_counter()
        handle = cpp.open_fits_file(filepath, "r")
        results['open_file'].append(time.perf_counter() - start)

        # Read header
        start = time.perf_counter()
        header = cpp.read_header(handle, 0)
        results['read_header'].append(time.perf_counter() - start)

        # Read data (C++ call that returns tensor)
        start = time.perf_counter()
        data = cpp.read_full(handle, 0)
        results['read_data'].append(time.perf_counter() - start)

        # DLPack is already included in read_data, but measure tensor operations
        start = time.perf_counter()
        _ = data.shape  # Accessing tensor properties
        results['dlpack_conversion'].append(time.perf_counter() - start)

        # Close file
        start = time.perf_counter()
        cpp.close_fits_file(handle)
        results['close_file'].append(time.perf_counter() - start)

        results['total'].append(time.perf_counter() - start_total)

    # Print results
    print(f"{'Component':<25} {'Median (μs)':<15} {'% of Total':<12}")
    print("-" * 80)

    total_median = statistics.median(results['total']) * 1e6

    for component in ['cache_check', 'open_file', 'read_header', 'read_data',
                      'dlpack_conversion', 'close_file', 'total']:
        times_us = [t * 1e6 for t in results[component]]
        median_us = statistics.median(times_us)
        pct = (median_us / total_median * 100) if component != 'total' else 100.0
        print(f"{component:<25} {median_us:>12.1f}μs  {pct:>10.1f}%")

    print()
    return results

def profile_fitsio(filepath, num_runs=50):
    """Profile fitsio for comparison."""
    import fitsio

    print("Profiling fitsio:")
    print("=" * 80)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        data = fitsio.read(filepath)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_ms = statistics.median(times) * 1000
    print(f"fitsio median: {median_ms:.3f}ms ({median_ms*1000:.1f}μs)")
    print()
    return times

def profile_raw_cfitsio(filepath, num_runs=50):
    """Profile raw C++ operations without Python overhead."""
    import torchfits.cpp as cpp

    print("Profiling raw C++ operations:")
    print("=" * 80)

    results = {
        'open': [],
        'read': [],
        'close': [],
        'total': []
    }

    for _ in range(num_runs):
        start_total = time.perf_counter()

        start = time.perf_counter()
        handle = cpp.open_fits_file(filepath, "r")
        results['open'].append(time.perf_counter() - start)

        start = time.perf_counter()
        data = cpp.read_full(handle, 0)
        results['read'].append(time.perf_counter() - start)

        start = time.perf_counter()
        cpp.close_fits_file(handle)
        results['close'].append(time.perf_counter() - start)

        results['total'].append(time.perf_counter() - start_total)

    print(f"{'Operation':<15} {'Median (μs)':<15}")
    print("-" * 40)

    for op in ['open', 'read', 'close', 'total']:
        times_us = [t * 1e6 for t in results[op]]
        median_us = statistics.median(times_us)
        print(f"{op:<15} {median_us:>12.1f}μs")

    print()
    return results

def main():
    print("=" * 80)
    print("DETAILED PERFORMANCE PROFILING")
    print("=" * 80)
    print("Objective: Find WHERE the 0.4ms overhead comes from")
    print()

    # Create test file
    filepath = create_test_file()

    # Warmup
    import torchfits
    import fitsio
    for _ in range(5):
        torchfits.clear_file_cache()
        _, _ = torchfits.read(filepath)
        _ = fitsio.read(filepath)

    print()

    # Profile components
    tf_results = profile_torchfits_components(filepath, num_runs=50)
    cpp_results = profile_raw_cfitsio(filepath, num_runs=50)
    fitsio_times = profile_fitsio(filepath, num_runs=50)

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    tf_median_ms = statistics.median(tf_results['total']) * 1000
    cpp_median_ms = statistics.median(cpp_results['total']) * 1000
    fitsio_median_ms = statistics.median(fitsio_times) * 1000

    print(f"torchfits (full Python path):  {tf_median_ms:.3f}ms")
    print(f"Raw C++ (no Python wrapper):    {cpp_median_ms:.3f}ms")
    print(f"fitsio:                         {fitsio_median_ms:.3f}ms")
    print()

    python_overhead = tf_median_ms - cpp_median_ms
    cpp_overhead = cpp_median_ms - fitsio_median_ms

    print(f"Python wrapper overhead:        {python_overhead:.3f}ms ({python_overhead/tf_median_ms*100:.1f}%)")
    print(f"C++ vs fitsio:                  {cpp_overhead:+.3f}ms")
    print()

    if cpp_overhead > 0:
        print(f"⚠️  Our C++ code is {cpp_overhead:.3f}ms slower than fitsio")
        print("   → This is the main optimization target!")
    else:
        print(f"✅ Our C++ code is {-cpp_overhead:.3f}ms faster than fitsio")

    if python_overhead > 0.05:
        print(f"⚠️  Python overhead is {python_overhead:.3f}ms")
        print("   → Consider optimizing Python wrapper")

    # Cleanup
    Path(filepath).unlink()

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The profiling breaks down:")
    print("  1. Which component takes the most time")
    print("  2. How much Python wrapper adds")
    print("  3. How our C++ compares to fitsio's C")
    print("=" * 80)

if __name__ == "__main__":
    main()
