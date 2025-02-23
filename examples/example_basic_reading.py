# examples/example_basic_reading.py
import torchfits
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = 202.5
        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "basic_example.fits"
    create_test_file(test_file)

    # Read the entire primary HDU
    try:
        data, header = torchfits.read(test_file)
        print("Full Image:")
        print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        print(f"  CRVAL1: {header.get('CRVAL1')}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read the first extension (using HDU number)
    try:
        data, header = torchfits.read(test_file, hdu=1)  # Primary HDU is 1 (not 0)
        print("\nFull Image (HDU=1):")
        print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a cutout using a CFITSIO string
    try:
        cutout, _ = torchfits.read(f"{test_file}[1][2:5,3:7]")  # 1-based indexing
        print("\nCutout (CFITSIO String):")
        print(f"  Cutout shape: {cutout.shape}")  # Expected: (3, 4)
    except RuntimeError as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()
    