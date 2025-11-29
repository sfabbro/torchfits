import torchfits
import numpy as np
from astropy.io import fits as astropy_fits
from pathlib import Path
import tempfile
import time

# Create test file
filepath = Path(tempfile.gettempdir()) / 'cpp_test.fits'
data = np.random.randn(1000, 1000).astype(np.float32)
astropy_fits.writeto(filepath, data, overwrite=True)

print('Testing full read() function...')
try:
    torchfits.clear_file_cache()
    tensor_data, header = torchfits.read(str(filepath))
    print(f"Success! Got tensor with shape: {tensor_data.shape}, dtype: {tensor_data.dtype}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
