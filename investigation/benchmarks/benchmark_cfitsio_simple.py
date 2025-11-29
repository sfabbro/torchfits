#!/usr/bin/env python3
"""Simple benchmark: measure if skipping metadata helps."""
import time
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits
import statistics

def create_test_file():
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"simple_bench_{time.time_ns()}.fits"
    data = np.random.randn(1000, 1000).astype(np.float32)
    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def main():
    import torchfits
    import fitsio

    print("=" * 80)
    print("SIMPLE OVERHEAD TEST")
    print("=" * 80)
    print()

    filepath = create_test_file()

    # Warmup
    for _ in range(10):
        torchfits.clear_file_cache()
        _, _ = torchfits.read(filepath)
        _ = fitsio.read(filepath)

    print("Benchmarking (50 runs each)...")

    # torchfits
    times = []
    for _ in range(50):
        torchfits.clear_file_cache()
        start = time.perf_counter()
        data, header = torchfits.read(filepath)
        times.append(time.perf_counter() - start)
    tf_median = statistics.median(times) * 1000

    # fitsio
    times = []
    for _ in range(50):
        start = time.perf_counter()
        data = fitsio.read(filepath)
        times.append(time.perf_counter() - start)
    fitsio_median = statistics.median(times) * 1000

    print()
    print(f"torchfits:  {tf_median:.3f}ms")
    print(f"fitsio:     {fitsio_median:.3f}ms")
    print(f"Gap:        {tf_median - fitsio_median:.3f}ms ({(tf_median/fitsio_median - 1)*100:.1f}% slower)")
    print()

    # Now test C++ directly
    import torchfits.cpp as cpp

    times = []
    for _ in range(50):
        start = time.perf_counter()
        handle = cpp.open_fits_file(filepath, "r")
        data = cpp.read_full(handle, 0)
        cpp.close_fits_file(handle)
        times.append(time.perf_counter() - start)
    cpp_median = statistics.median(times) * 1000

    print(f"C++ direct: {cpp_median:.3f}ms")
    print(f"Python wrapper overhead: {tf_median - cpp_median:.3f}ms")
    print()

    Path(filepath).unlink()

if __name__ == "__main__":
    main()
