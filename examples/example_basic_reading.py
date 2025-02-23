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
        hdu.header['CRVAL2'] = 47.5
        hdu.header['CRPIX1'] = 5.0
        hdu.header['CRPIX2'] = 5.0
        hdu.header['CDELT1'] = -0.001
        hdu.header['CDELT2'] = 0.001
        hdu.header['OBJECT'] = 'Test Object' # Add a non-WCS keyword
        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "basic_example.fits"
    create_test_file(test_file)

    # Read the entire primary HDU
    try:
        data, header = torchfits.read(test_file)
        print("Full Image (Primary HDU):")
        print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        print(f"  CRVAL1: {header.get('CRVAL1')}")
        print(f"  OBJECT: {header.get('OBJECT')}") # Access a non-WCS keyword

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read the first extension (using HDU number)
    try:
        data, header = torchfits.read(test_file, hdu=1)  # Primary HDU is 1 (not 0)
        print("\nFull Image (HDU=1):")
        print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        print(f"  CRVAL1: {header.get('CRVAL1')}")
    except RuntimeError as e:
        print(f"  Error: {e}")



    # Get Header Value
    try:
        object_name = torchfits.get_header_value(test_file, 1, 'OBJECT')
        print(f"\nObject Name: {object_name}")
    except RuntimeError as e:
        print(f" Error: {e}")

    # Get dimensions
    try:
        dims = torchfits.get_dims(test_file, 1)
        print(f"Dimensions: {dims}")
    except RuntimeError as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()