import torchfits
import torch
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CTYPE3'] = 'VELO-LSR'  # Or 'WAVE', 'FREQ', etc.
        hdu.header['CRVAL1'] = 200.0
        hdu.header['CRVAL2'] = 45.0
        hdu.header['CRVAL3'] = 1000.0
        hdu.header['CRPIX1'] = 1.0
        hdu.header['CRPIX2'] = 1.0
        hdu.header['CRPIX3'] = 1.0
        hdu.header['CDELT1'] = -0.01
        hdu.header['CDELT2'] = 0.01
        hdu.header['CDELT3'] = 5.0

        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "cube_example.fits"
    create_test_file(test_file)

    # Read a 2D slice (RA-DEC plane) using start/shape
    try:
        start = [0, 0, 1]  # Select the 2nd plane (index 1) along the 3rd axis
        shape = [-1, -1, 1]  # Read the entire RA and DEC dimensions, and a single plane
        slice_2d, header = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("2D Slice (Start/Shape):")
        print(f"  Shape: {slice_2d.shape}")  # Expected: (2, 3, 1)
        print(f"  CTYPE3: {header.get('CTYPE3')}")

    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a 2D slice using CFITSIO string
    try:
        slice_2d_cfitsio, header = torchfits.read(f"{test_file}[1][*,*,2]")  # 1-based indexing for CFITSIO
        print("\n2D Slice (CFITSIO String):")
        print(f"  Shape: {slice_2d_cfitsio.shape}")
        print(f"  CTYPE3: {header.get('CTYPE3')}")
         # Verify that the results are the same
        assert np.allclose(slice_2d.squeeze().numpy(), slice_2d_cfitsio.squeeze().numpy())


    except RuntimeError as e:
        print(f"  Error: {e}")


    # Read a 1D spectrum (velocity profile) using start/shape
    try:
        start = [1, 2, 0]  # x, y, z coordinates
        shape = [1, 1, -1]   # Read the entire spectral axis
        spectrum_1d, header = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("\n1D Spectrum (Start/Shape):")
        print(f"  Shape: {spectrum_1d.shape}")  # Expected: (1, 1, 2)
        print(f"  CTYPE3: {header.get('CTYPE3')}")


    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a 1D spectrum using CFITSIO string
    try:
        spectrum_1d_cfitsio, header = torchfits.read(f"{test_file}[1][2,3,*]") # 1-based indexing
        print("\n1D Spectrum (CFITSIO string):")
        print(f"  Shape: {spectrum_1d_cfitsio.shape}")
        # Verify that the results are the same
        assert np.allclose(spectrum_1d.squeeze().numpy(), spectrum_1d_cfitsio.squeeze().numpy())

    except RuntimeError as e:
        print(f" Error: {e}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 10]:
        try:
            start = [0, 1, 0]
            shape = [2, 1, -1]
            data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            print(f"  Data shape: {data.shape}, Data type: {data.dtype}")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            start = [0, 1, 0]
            shape = [2, 1, -1]
            data, _ = torchfits.read(test_file, hdu=1, start=start, shape=shape, device="cuda")
            print(f"  Data device: {data.device}")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")


if __name__ == "__main__":
    main()