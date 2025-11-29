#!/usr/bin/env python3
"""
Comprehensive benchmark: Test different data types and sizes.

Tests:
1. Different BITPIX types (int8, int16, int32, float32, float64)
2. Different sizes (small, medium, large)
3. Images vs tables (when implemented)
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from astropy.io import fits
import statistics

def create_test_file(dtype_str, shape):
    """Create FITS file with specific dtype and shape."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"comp_bench_{dtype_str}_{time.time_ns()}.fits"

    # Create data based on dtype
    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, shape, dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, shape, dtype=np.int16)
    elif dtype_str == 'int32':
        data = np.random.randint(-1000, 1000, shape, dtype=np.int32)
    elif dtype_str == 'float32':
        data = np.random.randn(*shape).astype(np.float32)
    elif dtype_str == 'float64':
        data = np.random.randn(*shape).astype(np.float64)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_in_subprocess(code_template, filepath):
    """Benchmark in isolated subprocess."""
    code = code_template.format(filepath=filepath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"ERROR:\n{result.stderr}", file=sys.stderr)
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None

TORCHFITS_CODE = """
import torchfits
import time
torchfits.clear_file_cache()
start = time.perf_counter()
data, header = torchfits.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""

FITSIO_CODE = """
import fitsio
import time
start = time.perf_counter()
data = fitsio.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""

def run_benchmark_suite():
    """Run comprehensive benchmark across data types and sizes."""

    test_cases = [
        # (dtype, shape, size_label, expected_mb)
        ('uint8',   (100, 100),    'tiny_100x100',    0.01),
        ('uint8',   (1000, 1000),  'small_1kx1k',     1),
        ('int16',   (1000, 1000),  'small_1kx1k',     2),
        ('int32',   (1000, 1000),  'small_1kx1k',     4),
        ('float32', (1000, 1000),  'small_1kx1k',     4),
        ('float64', (1000, 1000),  'small_1kx1k',     8),
        ('float32', (2000, 2000),  'medium_2kx2k',    16),
        ('float64', (2000, 2000),  'medium_2kx2k',    32),
        ('float32', (4000, 4000),  'large_4kx4k',     64),
    ]

    print("=" * 90)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 90)
    print()
    print(f"{'Data Type':<12} {'Size':<15} {'MB':<8} {'torchfits':<12} {'fitsio':<12} {'Ratio':<10}")
    print("-" * 90)

    summary = []

    for dtype, shape, size_label, size_mb in test_cases:
        results = {'torchfits': [], 'fitsio': []}

        # Run multiple trials
        for _ in range(5):
            filepath = create_test_file(dtype, shape)

            # Benchmark torchfits
            t = benchmark_in_subprocess(TORCHFITS_CODE, filepath)
            if t: results['torchfits'].append(t)

            # Benchmark fitsio
            t = benchmark_in_subprocess(FITSIO_CODE, filepath)
            if t: results['fitsio'].append(t)

            Path(filepath).unlink()

        if not results['torchfits'] or not results['fitsio']:
            continue

        # Calculate medians
        tf_ms = statistics.median(results['torchfits']) * 1000
        fitsio_ms = statistics.median(results['fitsio']) * 1000
        ratio = tf_ms / fitsio_ms

        status = "✅" if ratio < 1.0 else "⚠️ " if ratio < 1.5 else "❌"
        print(f"{dtype:<12} {size_label:<15} {size_mb:<8.1f} "
              f"{tf_ms:>10.3f}ms  {fitsio_ms:>10.3f}ms  {status} {ratio:>5.2f}x")

        summary.append({
            'dtype': dtype,
            'size_label': size_label,
            'ratio': ratio,
            'tf_ms': tf_ms,
            'fitsio_ms': fitsio_ms
        })

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Group by dtype
    print("\nPerformance by data type:")
    print("-" * 60)

    dtypes = ['uint8', 'int16', 'int32', 'float32', 'float64']
    for dtype in dtypes:
        dtype_results = [s for s in summary if s['dtype'] == dtype]
        if not dtype_results:
            continue

        avg_ratio = statistics.mean([s['ratio'] for s in dtype_results])
        status = "✅ FASTER" if avg_ratio < 1.0 else "⚠️  SLOWER"
        print(f"  {dtype:<10} avg ratio: {avg_ratio:.2f}x  {status}")

    # Overall
    print()
    overall_ratio = statistics.mean([s['ratio'] for s in summary])
    print(f"Overall average: {overall_ratio:.2f}x")
    print()

    if overall_ratio < 1.0:
        print(f"🎉 torchfits is {1/overall_ratio:.2f}x faster than fitsio on average!")
    elif overall_ratio < 1.2:
        print(f"✅ torchfits is within 20% of fitsio (acceptable for Phase 2)")
    elif overall_ratio < 1.5:
        print(f"⚠️  torchfits is {overall_ratio:.2f}x slower (needs optimization)")
    else:
        print(f"❌ torchfits is {overall_ratio:.2f}x slower (significant gap)")

if __name__ == "__main__":
    run_benchmark_suite()
