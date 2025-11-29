#!/usr/bin/env python3
"""
Critical test: Is torchfits direct-to-torch faster than fitsio→numpy→torch?
If not, we have a fundamental problem!
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
    filepath = tmpdir / f"conversion_test_{time.time()}.fits"

    if size_mb == 4:
        data = np.random.randn(1000, 1000).astype(np.float32)
    elif size_mb == 32:
        data = np.random.randn(2000, 2000).astype(np.float64)
    else:
        raise ValueError(f"Unknown size: {size_mb}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def benchmark_method(filepath, code_template):
    """Benchmark a method in fresh subprocess."""
    code = code_template.format(filepath=filepath)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    return float(result.stdout.strip())

TORCHFITS_CODE = """
import torchfits
import time
torchfits.clear_file_cache()
start = time.perf_counter()
data, header = torchfits.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""

FITSIO_TORCH_CODE = """
import fitsio
import torch
import time
start = time.perf_counter()
numpy_data = fitsio.read('{filepath}')
torch_data = torch.from_numpy(numpy_data)
elapsed = time.perf_counter() - start
print(elapsed)
"""

FITSIO_ONLY_CODE = """
import fitsio
import time
start = time.perf_counter()
data = fitsio.read('{filepath}')
elapsed = time.perf_counter() - start
print(elapsed)
"""

NUMPY_TO_TORCH_CODE = """
import numpy as np
import torch
import time
# Create data similar to fitsio output
data = np.random.randn(1000, 1000).astype(np.float32)
start = time.perf_counter()
torch_data = torch.from_numpy(data)
elapsed = time.perf_counter() - start
print(elapsed)
"""

def main():
    print("=" * 80)
    print("CRITICAL TEST: Direct-to-Torch vs fitsio→numpy→torch")
    print("=" * 80)
    print()
    print("If fitsio+conversion is faster, torchfits has a FUNDAMENTAL PROBLEM!")
    print()

    for size_mb, label in [(4, "float32_1k (4MB)"), (32, "float64_2k (32MB)")]:
        print(f"\nTesting: {label}")
        print("-" * 80)

        results = {
            'torchfits': [],
            'fitsio+torch': [],
            'fitsio_only': []
        }

        for run in range(5):
            filepath = create_test_file(size_mb)

            # Test torchfits (direct to torch)
            t = benchmark_method(filepath, TORCHFITS_CODE)
            if t: results['torchfits'].append(t)

            # Test fitsio + conversion (numpy → torch)
            t = benchmark_method(filepath, FITSIO_TORCH_CODE)
            if t: results['fitsio+torch'].append(t)

            # Test fitsio only (for reference)
            t = benchmark_method(filepath, FITSIO_ONLY_CODE)
            if t: results['fitsio_only'].append(t)

            Path(filepath).unlink()

        # Calculate medians
        torchfits_med = np.median(results['torchfits']) * 1000
        fitsio_torch_med = np.median(results['fitsio+torch']) * 1000
        fitsio_only_med = np.median(results['fitsio_only']) * 1000

        # Estimate conversion overhead
        conversion_overhead = fitsio_torch_med - fitsio_only_med

        print()
        print(f"torchfits (direct):      {torchfits_med:.3f}ms")
        print(f"fitsio only:             {fitsio_only_med:.3f}ms")
        print(f"fitsio + numpy→torch:    {fitsio_torch_med:.3f}ms")
        print(f"  (conversion overhead:  {conversion_overhead:.3f}ms)")
        print()

        if torchfits_med < fitsio_torch_med:
            advantage = fitsio_torch_med / torchfits_med
            print(f"✅ GOOD: torchfits is {advantage:.2f}x FASTER than fitsio+conversion")
            print(f"   Direct-to-torch path wins by {fitsio_torch_med - torchfits_med:.3f}ms")
        else:
            problem_ratio = torchfits_med / fitsio_torch_med
            print(f"❌ PROBLEM: torchfits is {problem_ratio:.2f}x SLOWER than fitsio+conversion!")
            print(f"   This means our 'direct' path has extra overhead of {torchfits_med - fitsio_torch_med:.3f}ms")
            print(f"   THIS IS UNACCEPTABLE - we should be faster, not slower!")

        # Compare to just fitsio
        if torchfits_med < fitsio_only_med:
            print(f"✅ torchfits is {fitsio_only_med/torchfits_med:.2f}x faster than fitsio (good!)")
        else:
            print(f"⚠️  torchfits is {torchfits_med/fitsio_only_med:.2f}x slower than fitsio (acceptable for Phase 1 only)")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("The critical question: Should direct-to-torch be faster than numpy→torch?")
    print("Answer: YES! If not, we're doing something fundamentally wrong.")
    print("=" * 80)

if __name__ == "__main__":
    main()
