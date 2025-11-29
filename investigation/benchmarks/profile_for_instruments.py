#!/usr/bin/env python3
"""
Run int16 reads in a loop for profiling with Instruments.
Usage: instruments -t "Time Profiler" python profile_for_instruments.py
"""
import time
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import numpy as np

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / f"instruments_int16.fits"
data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
astropy_fits.writeto(filepath, data, overwrite=True)

print(f"Created test file: {filepath}")
print("Running 1000 iterations for profiling...")
print("Attach Instruments Time Profiler now!")
time.sleep(3)  # Give time to attach profiler

# Run many iterations
for i in range(1000):
    handle = cpp.open_fits_file(str(filepath), 'r')
    tensor = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)

    if i % 100 == 0:
        print(f"  Iteration {i}/1000...")

print("Done!")
filepath.unlink()
