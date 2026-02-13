import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        hdu = fits.PrimaryHDU(data)
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CTYPE3"] = "VELO-LSR"  # Or 'WAVE', 'FREQ', etc.
        hdu.header["CRVAL1"] = 200.0
        hdu.header["CRVAL2"] = 45.0
        hdu.header["CRVAL3"] = 1000.0
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CRPIX2"] = 1.0
        hdu.header["CRPIX3"] = 1.0
        hdu.header["CDELT1"] = -0.01
        hdu.header["CDELT2"] = 0.01
        hdu.header["CDELT3"] = 5.0

        hdu.writeto(filename, overwrite=True)


def main():
    test_file = "cube_example.fits"
    create_test_file(test_file)

    # Read the full 3D cube
    try:
        cube, header = torchfits.read(test_file, hdu=0, return_header=True)
        print("Full 3D Cube:")
        print(f"  Shape: {cube.shape}")  # Expected: (2, 3, 4) - (z, y, x) in FITS order
        print(f"  CTYPE1: {header.get('CTYPE1')}")
        print(f"  CTYPE2: {header.get('CTYPE2')}")
        print(f"  CTYPE3: {header.get('CTYPE3')}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a 2D slice using CFITSIO string syntax
    try:
        # Select 2nd plane along 3rd axis (1-based indexing)
        slice_2d, header = torchfits.read(f"{test_file}[0][*,*,2]", return_header=True)
        print("\n2D Slice (CFITSIO String [*,*,2]):")
        print(f"  Shape: {slice_2d.shape}")  # Expected: (3, 4) - collapsed z dimension
        print(f"  Equivalent to cube[1,:,:] = {cube[1, :, :].shape}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a 1D spectrum using CFITSIO string
    try:
        # Extract spectrum at position (x=2, y=3) - 1-based indexing
        spectrum_1d, header = torchfits.read(
            f"{test_file}[0][2,3,*]", return_header=True
        )
        print("\n1D Spectrum (CFITSIO String [2,3,*]):")
        print(f"  Shape: {spectrum_1d.shape}")  # Expected: (2,) - spectral axis only
        print(f"  Equivalent to cube[:,2,1] = {cube[:, 2, 1].shape}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Manual slicing of the full cube
    print("\n--- Manual Slicing of Full Cube ---")
    print(f"  Single plane: cube[0,:,:] shape = {cube[0, :, :].shape}")
    print(f"  Spectrum at (1,2): cube[:,1,2] shape = {cube[:, 1, 2].shape}")
    print(f"  Sub-cube: cube[:,1:3,2:4] shape = {cube[:, 1:3, 2:4].shape}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 10]:
        try:
            data, _ = torchfits.read(
                test_file, hdu=0, cache_capacity=capacity, return_header=True
            )
            print(f"\nCache Capacity: {capacity}")
            print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            data, _ = torchfits.read(
                test_file, hdu=0, device="cuda", return_header=True
            )
            print(f"  Data device: {data.device}")
            print(f"  Can slice on GPU: data[0,:,:].shape = {data[0, :, :].shape}")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")


if __name__ == "__main__":
    main()
