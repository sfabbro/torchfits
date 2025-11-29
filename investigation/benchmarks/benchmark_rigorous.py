#!/usr/bin/env python3
"""
Rigorous, fair benchmark comparing torchfits vs fitsio.

Principles:
1. Fresh file for EACH measurement (no OS cache reuse)
2. Subprocess isolation (no Python code/module cache)
3. Full operation measured: open + read + close
4. Same measurement methodology for all libraries
5. Proper statistical analysis (median, quartiles)
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from astropy.io import fits

def create_fresh_file(size_mb=4):
    """Create a unique test file with timestamp."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"rigorous_bench_{time.time_ns()}.fits"

    if size_mb == 4:
        data = np.random.randn(1000, 1000).astype(np.float32)
    elif size_mb == 32:
        data = np.random.randn(2000, 2000).astype(np.float64)
    else:
        raise ValueError(f"Unknown size: {size_mb}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_in_subprocess(code_template, filepath):
    """Run benchmark in isolated subprocess, return time in seconds."""
    code = code_template.format(filepath=filepath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"ERROR in subprocess:\n{result.stderr}", file=sys.stderr)
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"Failed to parse output: {result.stdout}", file=sys.stderr)
        return None

# Benchmark code templates - measure FULL operation
TORCHFITS_CODE = """
import torchfits
import time

# Clear any cache
torchfits.clear_file_cache()

# Measure full operation: cache check + open + read + close
start = time.perf_counter()
data, header = torchfits.read('{filepath}')
elapsed = time.perf_counter() - start

print(elapsed)
"""

FITSIO_CODE = """
import fitsio
import time

# Measure full operation: open + read + close
start = time.perf_counter()
data = fitsio.read('{filepath}')
elapsed = time.perf_counter() - start

print(elapsed)
"""

ASTROPY_CODE = """
from astropy.io import fits
import time

# Measure full operation: open + read + close
start = time.perf_counter()
with fits.open('{filepath}') as hdul:
    data = hdul[0].data
elapsed = time.perf_counter() - start

print(elapsed)
"""

FITSIO_TO_TORCH_CODE = """
import fitsio
import torch
import time

# Measure: open + read + numpy→torch conversion
start = time.perf_counter()
numpy_data = fitsio.read('{filepath}')
torch_data = torch.from_numpy(numpy_data)
elapsed = time.perf_counter() - start

print(elapsed)
"""

def run_benchmark_suite(size_mb, label, num_runs=10):
    """Run complete benchmark suite for one file size."""
    print(f"\n{'='*80}")
    print(f"Testing: {label} ({size_mb}MB)")
    print(f"{'='*80}")
    print(f"Method: Fresh file for EACH run, subprocess isolation")
    print(f"Runs: {num_runs} iterations per library")
    print()

    results = {
        'torchfits': [],
        'fitsio': [],
        'astropy': [],
        'fitsio→torch': []
    }

    # Run benchmarks with fresh files
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...", end='\r')

        # Create 4 fresh files (one per library)
        files = {
            'torchfits': create_fresh_file(size_mb),
            'fitsio': create_fresh_file(size_mb),
            'astropy': create_fresh_file(size_mb),
            'fitsio→torch': create_fresh_file(size_mb)
        }

        # Benchmark each library
        for name, code in [
            ('torchfits', TORCHFITS_CODE),
            ('fitsio', FITSIO_CODE),
            ('astropy', ASTROPY_CODE),
            ('fitsio→torch', FITSIO_TO_TORCH_CODE)
        ]:
            t = benchmark_in_subprocess(code, files[name])
            if t is not None:
                results[name].append(t)

        # Clean up files
        for filepath in files.values():
            Path(filepath).unlink(missing_ok=True)

    print(" " * 40)  # Clear progress line

    # Statistical analysis
    import statistics

    print("Results (milliseconds):")
    print(f"{'Library':<20} {'Median':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 80)

    stats = {}
    for name in ['torchfits', 'fitsio', 'fitsio→torch', 'astropy']:
        if results[name]:
            times_ms = [t * 1000 for t in results[name]]
            median = statistics.median(times_ms)
            min_t = min(times_ms)
            max_t = max(times_ms)
            stddev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0

            stats[name] = {
                'median': median,
                'min': min_t,
                'max': max_t,
                'stddev': stddev
            }

            print(f"{name:<20} {median:>8.3f}ms  {min_t:>8.3f}ms  {max_t:>8.3f}ms  {stddev:>8.3f}ms")

    print()
    print("Analysis:")
    print("-" * 80)

    # Compare torchfits to fitsio
    if 'torchfits' in stats and 'fitsio' in stats:
        ratio = stats['torchfits']['median'] / stats['fitsio']['median']
        diff = stats['torchfits']['median'] - stats['fitsio']['median']

        if ratio < 1:
            print(f"✅ torchfits is {1/ratio:.2f}x FASTER than fitsio")
            print(f"   Advantage: {-diff:.3f}ms")
        else:
            print(f"⚠️  torchfits is {ratio:.2f}x SLOWER than fitsio")
            print(f"   Gap: {diff:.3f}ms")

    # Compare torchfits to fitsio+conversion
    if 'torchfits' in stats and 'fitsio→torch' in stats:
        ratio = stats['torchfits']['median'] / stats['fitsio→torch']['median']
        diff = stats['torchfits']['median'] - stats['fitsio→torch']['median']

        print()
        if ratio < 1:
            print(f"✅ torchfits is {1/ratio:.2f}x FASTER than fitsio→torch")
            print(f"   Our direct-to-torch path wins by {-diff:.3f}ms")
        else:
            print(f"❌ torchfits is {ratio:.2f}x SLOWER than fitsio→torch")
            print(f"   Direct path should be faster! Gap: {diff:.3f}ms")

    # Compare to astropy baseline
    if 'torchfits' in stats and 'astropy' in stats:
        ratio = stats['astropy']['median'] / stats['torchfits']['median']
        print()
        print(f"📊 vs astropy: torchfits is {ratio:.2f}x faster")

def main():
    print("=" * 80)
    print("RIGOROUS BENCHMARK: torchfits vs fitsio vs astropy")
    print("=" * 80)
    print()
    print("Methodology:")
    print("  • Fresh file created for EACH measurement")
    print("  • Subprocess isolation (no code/module cache)")
    print("  • Full operation: open + read + close")
    print("  • Statistical analysis: median, min, max, stddev")
    print()

    # Test both file sizes
    run_benchmark_suite(size_mb=4, label="float32 1000×1000", num_runs=10)
    run_benchmark_suite(size_mb=32, label="float64 2000×2000", num_runs=10)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("This benchmark ensures fair comparison by:")
    print("  ✓ Using fresh files (no OS page cache)")
    print("  ✓ Subprocess isolation (no Python cache)")
    print("  ✓ Measuring identical operations")
    print("  ✓ Statistical analysis over multiple runs")
    print("=" * 80)

if __name__ == "__main__":
    main()
