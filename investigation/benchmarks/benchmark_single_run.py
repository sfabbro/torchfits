#!/usr/bin/env python3
"""
Single-run benchmark - create fresh file each time
"""
import subprocess
import sys
from pathlib import Path
import tempfile
import numpy as np
from astropy.io import fits
import time

def create_test_file(size_mb=4):
    """Create a fresh test file."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"single_run_test_{time.time()}.fits"

    if size_mb == 4:
        data = np.random.randn(1000, 1000).astype(np.float32)
    elif size_mb == 32:
        data = np.random.randn(2000, 2000).astype(np.float64)
    else:
        raise ValueError(f"Unknown size: {size_mb}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_torchfits(filepath):
    """Benchmark torchfits in fresh subprocess."""
    code = f"""
import torchfits
import time
torchfits.clear_file_cache()
start = time.perf_counter()
data, header = torchfits.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return float(result.stdout.strip())

def benchmark_fitsio(filepath):
    """Benchmark fitsio in fresh subprocess."""
    code = f"""
import fitsio
import time
start = time.perf_counter()
data = fitsio.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    print("=" * 80)
    print("Single-Run Benchmark (Fresh Files, No Caching)")
    print("=" * 80)
    print()

    for size_mb, label in [(4, "float32_1k"), (32, "float64_2k")]:
        print(f"\nTesting: {label} ({size_mb}MB)")
        print("-" * 80)

        results = {'torchfits': [], 'fitsio': []}

        for run in range(5):
            # Create fresh file
            filepath = create_test_file(size_mb)

            # Test torchfits
            t_time = benchmark_torchfits(filepath)
            results['torchfits'].append(t_time)

            # Test fitsio (reuse same file)
            f_time = benchmark_fitsio(filepath)
            results['fitsio'].append(f_time)

            # Clean up
            Path(filepath).unlink()

            print(f"  Run {run+1}: torchfits={t_time*1000:.3f}ms, fitsio={f_time*1000:.3f}ms")

        # Calculate medians
        t_median = np.median(results['torchfits']) * 1000
        f_median = np.median(results['fitsio']) * 1000

        print()
        print(f"Median: torchfits={t_median:.3f}ms, fitsio={f_median:.3f}ms")

        if t_median < f_median:
            speedup = f_median / t_median
            print(f"✅ torchfits is {speedup:.2f}x FASTER")
        else:
            slowdown = t_median / f_median
            print(f"❌ torchfits is {slowdown:.2f}x SLOWER")

if __name__ == "__main__":
    main()
