#!/usr/bin/env python3
"""
Use cProfile to profile the int16 read path.
"""
import cProfile
import pstats
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import numpy as np

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / f"cprofile_int16.fits"
data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
astropy_fits.writeto(filepath, data, overwrite=True)

print("Profiling int16 read operations...")

# Profile the operations
profiler = cProfile.Profile()

profiler.enable()
for _ in range(100):
    handle = cpp.open_fits_file(str(filepath), 'r')
    tensor = cpp.read_full(handle, 0)
    cpp.close_fits_file(handle)
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
print("\nTop 20 functions by cumulative time:")
stats.print_stats(20)

filepath.unlink()
