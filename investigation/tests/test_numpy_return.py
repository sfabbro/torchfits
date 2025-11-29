#!/usr/bin/env python3
"""
Test if int16 is returning NumPy array or torch.Tensor
"""
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import numpy as np

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / f"test_numpy_{np.random.randint(0, 1000000)}.fits"
data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
astropy_fits.writeto(filepath, data, overwrite=True)

print("Testing what type is returned for int16...")
print()

# Test int16
handle = cpp.open_fits_file(str(filepath), 'r')
result = cpp.read_full(handle, 0)
cpp.close_fits_file(handle)

print(f"Result type: {type(result)}")
print(f"Result dtype: {result.dtype}")
print(f"Is numpy: {isinstance(result, np.ndarray)}")
print(f"Is torch: {type(result).__module__ == 'torch'}")
print()

# Test uint8 for comparison
filepath_uint8 = tmpdir / f"test_numpy_uint8_{np.random.randint(0, 1000000)}.fits"
data_uint8 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
astropy_fits.writeto(filepath_uint8, data_uint8, overwrite=True)

handle_uint8 = cpp.open_fits_file(str(filepath_uint8), 'r')
result_uint8 = cpp.read_full(handle_uint8, 0)
cpp.close_fits_file(handle_uint8)

print(f"uint8 result type: {type(result_uint8)}")
print(f"uint8 result dtype: {result_uint8.dtype}")
print(f"uint8 is numpy: {isinstance(result_uint8, np.ndarray)}")
print(f"uint8 is torch: {type(result_uint8).__module__ == 'torch'}")

# Cleanup
filepath.unlink()
filepath_uint8.unlink()
