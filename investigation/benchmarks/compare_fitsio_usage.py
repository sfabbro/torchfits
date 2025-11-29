#!/usr/bin/env python3
"""
Compare exactly how fitsio calls CFITSIO vs how we call it.

Since fitsio is based on CFITSIO and they're faster on int16,
there must be something different in how they use the library.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from astropy.io import fits as astropy_fits
import statistics

def create_test_files():
    tmpdir = Path(tempfile.gettempdir())

    # uint8 file
    uint8_file = tmpdir / f"compare_uint8_{time.time_ns()}.fits"
    data_uint8 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    astropy_fits.writeto(uint8_file, data_uint8, overwrite=True)

    # int16 file
    int16_file = tmpdir / f"compare_int16_{time.time_ns()}.fits"
    data_int16 = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    astropy_fits.writeto(int16_file, data_int16, overwrite=True)

    return str(uint8_file), str(int16_file)

def benchmark_method(code, filepath, iterations=20):
    times = []
    for _ in range(iterations):
        result = subprocess.run(
            [sys.executable, "-c", code.format(filepath=filepath)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            times.append(float(result.stdout.strip()))
    return statistics.median(times) * 1000  # Convert to ms

# Test code for each method
FITSIO_CODE = """
import fitsio, time
start = time.perf_counter()
data = fitsio.read('{filepath}')
print(time.perf_counter() - start)
"""

TORCHFITS_CODE = """
import torchfits, time
torchfits.clear_file_cache()
start = time.perf_counter()
data, header = torchfits.read('{filepath}')
print(time.perf_counter() - start)
"""

ASTROPY_CODE = """
from astropy.io import fits
import torch
import time
start = time.perf_counter()
with fits.open('{filepath}') as hdul:
    data = torch.from_numpy(hdul[0].data.copy())
print(time.perf_counter() - start)
"""

def main():
    print("=" * 80)
    print("Comparing fitsio vs torchfits CFITSIO usage")
    print("=" * 80)
    print()

    uint8_file, int16_file = create_test_files()

    try:
        print("Testing uint8 (1MB)...")
        print("-" * 80)
        fitsio_uint8 = benchmark_method(FITSIO_CODE, uint8_file)
        torchfits_uint8 = benchmark_method(TORCHFITS_CODE, uint8_file)
        astropy_uint8 = benchmark_method(ASTROPY_CODE, uint8_file)

        print(f"  fitsio:     {fitsio_uint8:>8.3f}ms")
        print(f"  torchfits:  {torchfits_uint8:>8.3f}ms")
        print(f"  astropy→torch: {astropy_uint8:>8.3f}ms")
        print()

        print("Testing int16 (2MB)...")
        print("-" * 80)
        fitsio_int16 = benchmark_method(FITSIO_CODE, int16_file)
        torchfits_int16 = benchmark_method(TORCHFITS_CODE, int16_file)
        astropy_int16 = benchmark_method(ASTROPY_CODE, int16_file)

        print(f"  fitsio:     {fitsio_int16:>8.3f}ms")
        print(f"  torchfits:  {torchfits_int16:>8.3f}ms")
        print(f"  astropy→torch: {astropy_int16:>8.3f}ms")
        print()

        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print()

        print("Ratios (int16/uint8):")
        print(f"  fitsio:     {fitsio_int16/fitsio_uint8:.2f}x")
        print(f"  torchfits:  {torchfits_int16/torchfits_uint8:.2f}x")
        print(f"  astropy:    {astropy_int16/astropy_uint8:.2f}x")
        print()

        print("torchfits vs fitsio:")
        print(f"  uint8:  {torchfits_uint8/fitsio_uint8:.2f}x")
        print(f"  int16:  {torchfits_int16/fitsio_int16:.2f}x")
        print()

        if torchfits_uint8 < fitsio_uint8:
            print(f"✅ torchfits WINS on uint8 ({(1 - torchfits_uint8/fitsio_uint8)*100:.1f}% faster)")
        else:
            print(f"❌ torchfits LOSES on uint8 ({(torchfits_uint8/fitsio_uint8 - 1)*100:.1f}% slower)")

        if torchfits_int16 < fitsio_int16:
            print(f"✅ torchfits WINS on int16 ({(1 - torchfits_int16/fitsio_int16)*100:.1f}% faster)")
        else:
            print(f"❌ torchfits LOSES on int16 ({(torchfits_int16/fitsio_int16 - 1)*100:.1f}% slower)")

        print()

        # Check if the ratio difference is the issue
        fitsio_ratio = fitsio_int16 / fitsio_uint8
        torchfits_ratio = torchfits_int16 / torchfits_uint8

        if torchfits_ratio > fitsio_ratio * 1.2:
            print(f"⚠️  torchfits has WORSE int16/uint8 ratio ({torchfits_ratio:.2f}x vs {fitsio_ratio:.2f}x)")
            print("   This suggests int16-specific overhead in our code, not CFITSIO.")
        else:
            print(f"✅ Ratios are similar ({torchfits_ratio:.2f}x vs {fitsio_ratio:.2f}x)")
            print("   The issue is likely in CFITSIO itself.")

    finally:
        # Cleanup
        Path(uint8_file).unlink(missing_ok=True)
        Path(int16_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
