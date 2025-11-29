#!/usr/bin/env python3
"""
Benchmark the overhead of individual CFITSIO calls.

Test hypothesis: Our 276μs performance gap vs fitsio comes from
excessive metadata queries before the actual read.
"""
import time
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits
import statistics
import torchfits.cpp as cpp

def create_test_file():
    """Create simple float32 test file (no scaling, no compression)."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"cfitsio_calls_test_{time.time_ns()}.fits"
    data = np.random.randn(1000, 1000).astype(np.float32)
    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_full_metadata(filepath, num_runs=100):
    """Benchmark our current approach with all metadata calls."""
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()

        handle = cpp.open_fits_file(filepath, "r")

        # All the calls we currently make
        hdu_type = cpp.get_hdu_type(handle, 0)  # Call 1
        shape = cpp.get_shape(handle, 0)         # Call 2 (includes fits_get_img_param)
        header = cpp.read_header(handle, 0)      # Call 3 (includes BSCALE/BZERO reads)
        data = cpp.read_full(handle, 0)          # Call 4 (the actual read)

        cpp.close_fits_file(handle)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_minimal_metadata(filepath, num_runs=100):
    """Benchmark minimal approach: just open→read→close."""
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()

        handle = cpp.open_fits_file(filepath, "r")
        data = cpp.read_full(handle, 0)  # Skip all metadata
        cpp.close_fits_file(handle)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def benchmark_individual_calls(filepath, num_runs=100):
    """Measure cost of each individual CFITSIO call."""
    handle = cpp.open_fits_file(filepath, "r")

    costs = {}

    # get_hdu_type cost
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = cpp.get_hdu_type(handle, 0)
        times.append(time.perf_counter() - start)
    costs['get_hdu_type'] = statistics.median(times) * 1e6

    # get_shape cost
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = cpp.get_shape(handle, 0)
        times.append(time.perf_counter() - start)
    costs['get_shape'] = statistics.median(times) * 1e6

    # read_header cost
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = cpp.read_header(handle, 0)
        times.append(time.perf_counter() - start)
    costs['read_header'] = statistics.median(times) * 1e6

    # read_full cost (the big one)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = cpp.read_full(handle, 0)
        times.append(time.perf_counter() - start)
    costs['read_full'] = statistics.median(times) * 1e6

    cpp.close_fits_file(handle)
    return costs

def benchmark_fitsio_baseline(filepath, num_runs=100):
    """Baseline: how fast is fitsio?"""
    import fitsio

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        data = fitsio.read(filepath)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times

def main():
    print("=" * 80)
    print("CFITSIO CALL OVERHEAD ANALYSIS")
    print("=" * 80)
    print("Hypothesis: Metadata calls (get_hdu_type, read_header, etc.)")
    print("            add significant overhead before the actual read")
    print()

    filepath = create_test_file()

    # Warmup
    import fitsio
    for _ in range(10):
        _ = fitsio.read(filepath)
        handle = cpp.open_fits_file(filepath, "r")
        _ = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)

    print("Running benchmarks...")
    print()

    # Individual call costs
    print("Individual CFITSIO call costs:")
    print("-" * 80)
    costs = benchmark_individual_calls(filepath, num_runs=100)
    total_metadata = sum(v for k, v in costs.items() if k != 'read_full')

    for name, cost_us in costs.items():
        marker = "  ← THE BIG ONE" if name == 'read_full' else ""
        print(f"{name:<20} {cost_us:>8.1f}μs{marker}")

    print(f"{'Total metadata':<20} {total_metadata:>8.1f}μs")
    print()

    # Full vs minimal
    print("Full operation benchmarks:")
    print("-" * 80)

    full_times = benchmark_full_metadata(filepath, num_runs=50)
    minimal_times = benchmark_minimal_metadata(filepath, num_runs=50)
    fitsio_times = benchmark_fitsio_baseline(filepath, num_runs=50)

    full_median = statistics.median(full_times) * 1000
    minimal_median = statistics.median(minimal_times) * 1000
    fitsio_median = statistics.median(fitsio_times) * 1000

    print(f"Full (all metadata):     {full_median:.3f}ms")
    print(f"Minimal (skip metadata): {minimal_median:.3f}ms")
    print(f"fitsio baseline:         {fitsio_median:.3f}ms")
    print()

    metadata_overhead = full_median - minimal_median
    print(f"Metadata overhead:       {metadata_overhead:.3f}ms ({metadata_overhead/full_median*100:.1f}%)")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    gap_full = full_median - fitsio_median
    gap_minimal = minimal_median - fitsio_median

    print(f"Gap (full vs fitsio):    {gap_full:.3f}ms")
    print(f"Gap (minimal vs fitsio): {gap_minimal:.3f}ms")
    print()

    if metadata_overhead > 0.1:
        print(f"⚠️  Metadata calls add {metadata_overhead:.3f}ms overhead!")
        print(f"   Eliminating unnecessary calls could save {metadata_overhead/gap_full*100:.0f}% of the gap")
    else:
        print("✅ Metadata overhead is negligible")

    if gap_minimal > 0.1:
        print(f"⚠️  Even minimal approach is {gap_minimal:.3f}ms slower than fitsio")
        print("   The bottleneck is in the core read operation, not metadata")
    else:
        print("✅ Minimal approach matches fitsio!")

    # Cleanup
    Path(filepath).unlink()

if __name__ == "__main__":
    main()
