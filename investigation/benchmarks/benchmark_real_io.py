#!/usr/bin/env python3
"""
Real I/O performance benchmark - NO CACHE
Compare torchfits vs astropy vs fitsio
"""
import torchfits
import numpy as np
from astropy.io import fits as astropy_fits
import fitsio
from pathlib import Path
import tempfile
import time
import os

def create_test_files():
    """Create test files with different data types."""
    tmpdir = Path(tempfile.gettempdir()) / "torchfits_benchmark"
    tmpdir.mkdir(exist_ok=True)

    files = {}

    # Float32 1000x1000 (4MB)
    filepath = tmpdir / "test_float32_1k.fits"
    data = np.random.randn(1000, 1000).astype(np.float32)
    astropy_fits.writeto(filepath, data, overwrite=True)
    files['float32_1k'] = filepath

    # Float64 2000x2000 (32MB)
    filepath = tmpdir / "test_float64_2k.fits"
    data = np.random.randn(2000, 2000).astype(np.float64)
    astropy_fits.writeto(filepath, data, overwrite=True)
    files['float64_2k'] = filepath

    # Int16 2000x2000 (8MB)
    filepath = tmpdir / "test_int16_2k.fits"
    data = np.random.randint(-1000, 1000, (2000, 2000), dtype=np.int16)
    astropy_fits.writeto(filepath, data, overwrite=True)
    files['int16_2k'] = filepath

    # Int32 1000x1000 (4MB)
    filepath = tmpdir / "test_int32_1k.fits"
    data = np.random.randint(-1000, 1000, (1000, 1000), dtype=np.int32)
    astropy_fits.writeto(filepath, data, overwrite=True)
    files['int32_1k'] = filepath

    return files

def benchmark_torchfits(filepath, n_runs=10):
    """Benchmark torchfits - clear cache each time."""
    times = []
    for _ in range(n_runs):
        torchfits.clear_file_cache()
        # Force OS to drop disk cache (best effort)
        os.sync()

        start = time.perf_counter()
        data, header = torchfits.read(str(filepath))
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times)

def benchmark_astropy(filepath, n_runs=10):
    """Benchmark astropy."""
    times = []
    for _ in range(n_runs):
        os.sync()

        start = time.perf_counter()
        with astropy_fits.open(filepath) as hdul:
            data = hdul[0].data
            # Force actual read
            _ = data[0, 0]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times)

def benchmark_fitsio(filepath, n_runs=10):
    """Benchmark fitsio."""
    times = []
    for _ in range(n_runs):
        os.sync()

        start = time.perf_counter()
        data = fitsio.read(str(filepath))
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times)

def main():
    print("=" * 80)
    print("Real I/O Performance Benchmark (NO CACHE)")
    print("=" * 80)
    print()

    files = create_test_files()

    results = {}

    for name, filepath in files.items():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\nTesting: {name} ({size_mb:.2f} MB)")
        print("-" * 80)

        # Run benchmarks
        torchfits_times = benchmark_torchfits(filepath, n_runs=5)
        astropy_times = benchmark_astropy(filepath, n_runs=5)
        fitsio_times = benchmark_fitsio(filepath, n_runs=5)

        # Calculate statistics (use median for more stable results)
        torchfits_median = np.median(torchfits_times) * 1000
        astropy_median = np.median(astropy_times) * 1000
        fitsio_median = np.median(fitsio_times) * 1000

        print(f"torchfits: {torchfits_median:.3f}ms (median of {len(torchfits_times)} runs)")
        print(f"astropy:   {astropy_median:.3f}ms (median of {len(astropy_times)} runs)")
        print(f"fitsio:    {fitsio_median:.3f}ms (median of {len(fitsio_times)} runs)")

        # Compare to best competitor
        best_competitor = min(astropy_median, fitsio_median)
        best_name = "astropy" if astropy_median < fitsio_median else "fitsio"

        if torchfits_median < best_competitor:
            speedup = best_competitor / torchfits_median
            print(f"✅ torchfits is {speedup:.2f}x FASTER than {best_name}")
        else:
            slowdown = torchfits_median / best_competitor
            print(f"❌ torchfits is {slowdown:.2f}x SLOWER than {best_name}")

        results[name] = {
            'torchfits': torchfits_median,
            'astropy': astropy_median,
            'fitsio': fitsio_median,
            'best': best_name,
            'ratio': torchfits_median / best_competitor
        }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    wins = sum(1 for r in results.values() if r['ratio'] < 1.0)
    losses = len(results) - wins

    print(f"torchfits wins: {wins}/{len(results)} tests")
    print(f"torchfits losses: {losses}/{len(results)} tests")

    if wins == len(results):
        print("✅ torchfits BEATS all competitors on ALL tests!")
    elif wins > losses:
        print("⚠️  torchfits wins most tests but has some losses")
    else:
        print("❌ torchfits loses most tests - needs optimization")

    # Average performance
    avg_ratio = np.mean([r['ratio'] for r in results.values()])
    if avg_ratio < 1.0:
        print(f"\nAverage speedup: {1/avg_ratio:.2f}x faster than best competitor")
    else:
        print(f"\nAverage slowdown: {avg_ratio:.2f}x slower than best competitor")

if __name__ == "__main__":
    main()
