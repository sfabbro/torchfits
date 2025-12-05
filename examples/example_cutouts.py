import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(1024, dtype=np.float32).reshape(32, 32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["CRPIX1"] = 16.0
        hdu.header["CRPIX2"] = 16.0
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRVAL1"] = 202.5
        hdu.header["CRVAL2"] = 47.5
        hdu.header["CDELT1"] = -0.001
        hdu.header["CDELT2"] = 0.001
        hdu.writeto(filename, overwrite=True)


def main():
    test_file = "cutout_example.fits"
    create_test_file(test_file)

    # Read a cutout using read_subset (x1, y1, x2, y2 are 0-based, exclusive end)
    try:
        # Cutout from [10:15, 15:23] (y, x order for FITS)
        x1, y1 = 15, 10
        x2, y2 = 23, 15
        data = torchfits.read_subset(test_file, hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)
        print("Cutout (read_subset):")
        print(f"  Data shape: {data.shape}")  # Expected: (5, 8)
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a cutout using a CFITSIO string
    try:
        cutout, header = torchfits.read(
            f"{test_file}[1][11:15,16:23]"
        )  # 1-based indexing
        print("\nCutout (CFITSIO String):")
        print(f"  Cutout shape: {cutout.shape}")  # Expected: (5, 8)
        # Check that CRPIX is updated correctly:
        print(f"  Updated CRPIX1: {header['CRPIX1']}")
        print(f"  Updated CRPIX2: {header['CRPIX2']}")

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read another cutout
    try:
        x1, y1 = 15, 10
        x2, y2 = 32, 15  # Read to end of x dimension
        data = torchfits.read_subset(test_file, hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)
        print("\nCutout (read to end):")
        print(f"  Data shape: {data.shape}")
    except RuntimeError as e:
        print(f" Error: {e}")

    # --- Test different cache capacities (not applicable to read_subset) ---
    print("\n--- Testing cache with full read ---")
    for capacity in [0, 10]:
        try:
            data, _ = torchfits.read(test_file, hdu=0, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            # Note: read_subset returns tensor on CPU, move to GPU after
            data = torchfits.read_subset(test_file, hdu=0, x1=5, y1=5, x2=15, y2=15)
            data = data.to('cuda')
            print(f"  Data device: {data.device}")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")


if __name__ == "__main__":
    main()
