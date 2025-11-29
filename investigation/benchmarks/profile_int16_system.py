#!/usr/bin/env python3
"""
Run int16 benchmark repeatedly for system-level profiling.
Use with: instruments -t "Time Profiler" python3 profile_int16_system.py
"""
import torchfits
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits

# Create test file
tmpdir = Path(tempfile.gettempdir())
uint8_file = tmpdir / "profile_uint8.fits"
int16_file = tmpdir / "profile_int16.fits"

data_uint8 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
data_int16 = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)

fits.writeto(uint8_file, data_uint8, overwrite=True)
fits.writeto(int16_file, data_int16, overwrite=True)

print("=" * 80)
print("System Profiling: uint8 vs int16")
print("=" * 80)
print()
print("Running 1000 iterations of each...")
print()

# Profile uint8
print("Profiling uint8 reads...")
for i in range(1000):
    torchfits.clear_file_cache()
    data, header = torchfits.read(str(uint8_file))
    if i % 100 == 0:
        print(f"  uint8: {i}/1000")

print()
print("Profiling int16 reads...")
for i in range(1000):
    torchfits.clear_file_cache()
    data, header = torchfits.read(str(int16_file))
    if i % 100 == 0:
        print(f"  int16: {i}/1000")

print()
print("Done! Analyze the profile with Instruments.app")

# Cleanup
uint8_file.unlink()
int16_file.unlink()
