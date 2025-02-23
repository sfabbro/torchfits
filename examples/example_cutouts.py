# examples/example_cutouts.py
import torchfits
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(1024, dtype=np.float32).reshape(32, 32)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CRPIX1'] = 16.0  # Set reference pixel for WCS testing
        hdu.header['CRPIX2'] = 16.0
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = 202.5
        hdu.header['CRVAL2'] = 47.5
        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "cutout_example.fits"
    create_test_file(test_file)

    # Read a cutout using start and shape
    try:
        start = [10, 15]  # 0-based
        shape = [5, 8]
        data, header = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("Cutout (Start/Shape):")
        print(f"  Data shape: {data.shape}")  # Expected: (5, 8)
        # Check that CRPIX is updated correctly
        self.assertAlmostEqual(header['CRPIX1'], 6.0)
        self.assertAlmostEqual(header['CRPIX2'], 1.0)

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a cutout using a CFITSIO string
    try:
        cutout, header = torchfits.read(f"{test_file}[1][11:15,16:23]") # 1-based indexing
        print("\nCutout (CFITSIO String):")
        print(f"  Cutout shape: {cutout.shape}")  # Expected: (5, 8)
        # Check against start/shape result:
        start = [10, 15]  # 0-based
        shape = [5, 8]
        data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        assert np.allclose(data.numpy(), cutout.numpy())
        self.assertAlmostEqual(header['CRPIX1'], 6.0)
        self.assertAlmostEqual(header['CRPIX2'], 1.0)

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read to end of dimension
    try:
        start = [10,15]
        shape = [-1, 5]
        data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("\nCutout (Read to end):")
        print(f"Data shape: {data.shape}")
        print(data.numpy())

    except RuntimeError as e:
        print(f" Error: {e}")
if __name__ == "__main__":
    main()