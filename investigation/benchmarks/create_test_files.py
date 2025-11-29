#!/usr/bin/env python3
"""Create test FITS files for pure CFITSIO benchmark."""
import numpy as np
from astropy.io import fits

# Create uint8 file
data_uint8 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
fits.writeto('test_uint8.fits', data_uint8, overwrite=True)
print("Created test_uint8.fits")

# Create int16 file
data_int16 = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
fits.writeto('test_int16.fits', data_int16, overwrite=True)
print("Created test_int16.fits")
