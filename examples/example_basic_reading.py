import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        hdu = fits.PrimaryHDU(data)
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRVAL1"] = 202.5
        hdu.header["CRVAL2"] = 47.5
        hdu.header["CRPIX1"] = 5.0
        hdu.header["CRPIX2"] = 5.0
        hdu.header["CDELT1"] = -0.001
        hdu.header["CDELT2"] = 0.001
        hdu.header["OBJECT"] = "Test Object"  # Add a non-WCS keyword
        hdu.writeto(filename, overwrite=True)


def main():
    test_file = "basic_example.fits"
    create_test_file(test_file)

    # Read the entire primary HDU (HDU 0)
    try:
        data, header = torchfits.read(test_file, hdu=0)  # Primary HDU is 0
        print("Full Image (Primary HDU):")
        print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        print(f"  CRVAL1: {header.get('CRVAL1')}")
        print(f"  OBJECT: {header.get('OBJECT')}")

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Get header using get_header()
    try:
        header = torchfits.get_header(test_file, hdu=0)
        object_name = header.get("OBJECT")
        print(f"\nObject Name: {object_name}")
        print(f"Dimensions: {header.get('NAXIS1')} x {header.get('NAXIS2')}")
    except RuntimeError as e:
        print(f" Error: {e}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 10, 100]:  # Test no cache, small cache, larger cache
        try:
            data, header = torchfits.read(test_file, hdu=0, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            data, header = torchfits.read(test_file, hdu=0, device="cuda")
            print(f"  Data device: {data.device}")  # Should print 'cuda:0'
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")


if __name__ == "__main__":
    main()
