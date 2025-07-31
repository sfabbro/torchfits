import torchfits
import torch
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(1024, dtype=np.float32).reshape(32, 32)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CRPIX1'] = 16.0
        hdu.header['CRPIX2'] = 16.0
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = 202.5
        hdu.header['CRVAL2'] = 47.5
        hdu.header['CDELT1'] = -0.001
        hdu.header['CDELT2'] = 0.001
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
        #Check CRPIX update
        print(f"  Updated CRPIX1: {header['CRPIX1']}")
        print(f"  Updated CRPIX2: {header['CRPIX2']}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a cutout using a CFITSIO string
    try:
        cutout, header = torchfits.read(f"{test_file}[1][11:15,16:23]") # 1-based indexing
        print("\nCutout (CFITSIO String):")
        print(f"  Cutout shape: {cutout.shape}")  # Expected: (5, 8)
        # Check that CRPIX is updated correctly:
        print(f"  Updated CRPIX1: {header['CRPIX1']}")
        print(f"  Updated CRPIX2: {header['CRPIX2']}")

    except RuntimeError as e:
        print(f"  Error: {e}")

      # Read to end of dimension
    try:
        start = [10,15]
        shape = [5, -1] #Read to the end of the second dimension
        data, header = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("\nCutout (Read to end):")
        print(f"  Data shape: {data.shape}")
        print(f" Updated CRPIX1: {header['CRPIX1']}")
        print(f" Updated CRPIX2: {header['CRPIX2']}")

    except RuntimeError as e:
        print(f" Error: {e}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 10]:
        try:
            start = [5, 5]
            shape = [10, 10]
            data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            start = [5, 5]
            shape = [10, 10]
            data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape, device="cuda")
            print(f"  Data device: {data.device}")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")

if __name__ == "__main__":
    main()