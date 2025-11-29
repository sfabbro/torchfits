#!/usr/bin/env python3
"""
Test what cpp.read_full returns directly, bypassing torchfits.read()
"""
import sys
from torchfits import cpp
import tempfile
from pathlib import Path
from astropy.io import fits as astropy_fits
import numpy as np

# Create test file
tmpdir = Path(tempfile.gettempdir())
filepath = tmpdir / f"test_direct_{np.random.randint(0, 1000000)}.fits"
data = np.random.randint(-32768, 32767, (100, 100), dtype=np.int16)
astropy_fits.writeto(filepath, data, overwrite=True)

print("Testing direct cpp.read_full call...")
print("="*60)

# Call cpp.read_full directly, redirecting stderr to see profiling
handle = cpp.open_fits_file(str(filepath), 'r')

print(f"About to call cpp.read_full...", file=sys.stderr)
sys.stderr.flush()

result = cpp.read_full(handle, 0)

print(f"Returned from cpp.read_full", file=sys.stderr)
sys.stderr.flush()

cpp.close_fits_file(handle)

print(f"\nResult type: {type(result)}")
print(f"Result type name: {type(result).__name__}")
print(f"Result module: {type(result).__module__}")
print(f"Result dtype: {result.dtype}")
print(f"Is numpy.ndarray: {type(result).__name__ == 'ndarray'}")
print(f"Is torch.Tensor: {type(result).__name__ == 'Tensor'}")

# Also check uint8
filepath_uint8 = tmpdir / f"test_direct_uint8_{np.random.randint(0, 1000000)}.fits"
data_uint8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
astropy_fits.writeto(filepath_uint8, data_uint8, overwrite=True)

handle_uint8 = cpp.open_fits_file(str(filepath_uint8), 'r')
result_uint8 = cpp.read_full(handle_uint8, 0)
cpp.close_fits_file(handle_uint8)

print(f"\nuint8 result type: {type(result_uint8)}")
print(f"uint8 result dtype: {result_uint8.dtype}")

# Cleanup
filepath.unlink()
filepath_uint8.unlink()
