#!/usr/bin/env python3
"""
Quick validation benchmark - torchfits vs astropy vs fitsio.
Tests the most common operations to verify optimizations.
"""

import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_test_file(size_name, shape, dtype=np.float32):
    """Create a test FITS file."""
    from astropy.io import fits as astropy_fits

    data = np.random.randn(*shape).astype(dtype)
    filepath = Path(tempfile.gettempdir()) / f"test_{size_name}.fits"
    astropy_fits.writeto(filepath, data, overwrite=True)
    return filepath, data

def benchmark_method(func, name, runs=5):
    """Benchmark a method with multiple runs."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del result

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time

def run_comparison(filepath):
    """Run comparison for a single file."""
    import torchfits
    from astropy.io import fits as astropy_fits
    import fitsio

    filepath_str = str(filepath)
    size_mb = filepath.stat().st_size / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Testing: {filepath.name} ({size_mb:.2f} MB)")
    print(f"{'='*60}")

    # Warm up
    _ = torchfits.read(filepath_str)

    # Benchmark torchfits
    tf_mean, tf_std = benchmark_method(
        lambda: torchfits.read(filepath_str),
        "torchfits"
    )
    print(f"torchfits:  {tf_mean:.6f}s ± {tf_std:.6f}s")

    # Benchmark astropy
    astropy_mean, astropy_std = benchmark_method(
        lambda: astropy_fits.getdata(filepath_str),
        "astropy"
    )
    print(f"astropy:    {astropy_mean:.6f}s ± {astropy_std:.6f}s")

    # Benchmark fitsio
    fitsio_mean, fitsio_std = benchmark_method(
        lambda: fitsio.read(filepath_str),
        "fitsio"
    )
    print(f"fitsio:     {fitsio_mean:.6f}s ± {fitsio_std:.6f}s")

    # Benchmark astropy -> torch
    def astropy_to_torch():
        data = astropy_fits.getdata(filepath_str)
        if data.dtype.byteorder not in ('=', '|'):
            data = data.astype(data.dtype.newbyteorder('='))
        return torch.from_numpy(data)

    astropy_torch_mean, astropy_torch_std = benchmark_method(
        astropy_to_torch,
        "astropy+torch"
    )
    print(f"astropy+torch: {astropy_torch_mean:.6f}s ± {astropy_torch_std:.6f}s")

    # Benchmark fitsio -> torch
    def fitsio_to_torch():
        data = fitsio.read(filepath_str)
        return torch.from_numpy(data)

    fitsio_torch_mean, fitsio_torch_std = benchmark_method(
        fitsio_to_torch,
        "fitsio+torch"
    )
    print(f"fitsio+torch:  {fitsio_torch_mean:.6f}s ± {fitsio_torch_std:.6f}s")

    # Analysis
    print(f"\n{'Results':-^60}")

    best_time = min(tf_mean, astropy_mean, fitsio_mean, astropy_torch_mean, fitsio_torch_mean)
    best_method = ["torchfits", "astropy", "fitsio", "astropy+torch", "fitsio+torch"][
        [tf_mean, astropy_mean, fitsio_mean, astropy_torch_mean, fitsio_torch_mean].index(best_time)
    ]

    print(f"Best method: {best_method} ({best_time:.6f}s)")

    if best_method != "torchfits":
        speedup = tf_mean / best_time
        print(f"⚠️  torchfits is {speedup:.2f}x SLOWER than {best_method}")
        print(f"    Need to improve by {(speedup - 1) * 100:.1f}%")
    else:
        second_best = sorted([astropy_mean, fitsio_mean, astropy_torch_mean, fitsio_torch_mean])[0]
        speedup = second_best / tf_mean
        print(f"✅ torchfits is {speedup:.2f}x FASTER than competitors")

    return {
        'torchfits': tf_mean,
        'astropy': astropy_mean,
        'fitsio': fitsio_mean,
        'astropy_torch': astropy_torch_mean,
        'fitsio_torch': fitsio_torch_mean,
        'best': best_method,
        'size_mb': size_mb
    }

def main():
    """Run quick validation benchmarks."""
    print("torchfits Performance Validation")
    print("=" * 60)

    test_cases = [
        ("small_2d", (512, 512), np.float32),
        ("medium_2d", (2048, 2048), np.float32),
        ("large_2d", (4096, 4096), np.float32),
        ("small_1d", (100000,), np.float32),
        ("medium_int16", (1024, 1024), np.int16),
    ]

    results = []
    failures = []

    for size_name, shape, dtype in test_cases:
        try:
            filepath, _ = create_test_file(size_name, shape, dtype)
            result = run_comparison(filepath)
            results.append((size_name, result))

            if result['best'] != 'torchfits':
                failures.append((size_name, result))
        except Exception as e:
            print(f"\n❌ Failed on {size_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if not failures:
        print("✅ torchfits BEATS all competitors on ALL tests!")
    else:
        print(f"⚠️  torchfits FAILED on {len(failures)}/{len(results)} tests:")
        for name, result in failures:
            print(f"   - {name}: {result['best']} was {result[result['best']]/result['torchfits']:.2f}x faster")

        print("\nNeed to investigate and optimize:")
        for name, result in failures:
            print(f"   {name}: torchfits={result['torchfits']:.6f}s, best={result[result['best']]:.6f}s")

    return len(failures) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
