# examples/example_datacube.py
import torchfits
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CTYPE3'] = 'VELO-LSR'
        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "cube_example.fits"
    create_test_file(test_file)

    # Read a 2D slice (RA-DEC plane)
    try:
        slice_2d, _ = torchfits.read(test_file, hdu=1, start=[0, 0, 1], shape=[-1, -1, 1])
        print("2D Slice:")
        print(f"  Shape: {slice_2d.shape}")  # Expected: (2, 3, 1)

        # Read a 1D spectrum (velocity profile)
        spectrum_1d, _ = torchfits.read(test_file, hdu=1, start=[1, 2, 0], shape=[1, 1, -1])
        print("\n1D Spectrum:")
        print(f"  Shape: {spectrum_1d.shape}")  # Expected: (1, 1, 2)

        #Using CFITSIO strings (should give same results)
        slice_2d_c, _ = torchfits.read(f"{test_file}[1][*,*,2]")
        print("\n2D Slice (using CFITSIO):")
        print(f"  Shape: {slice_2d.shape}")
        assert np.allclose(slice_2d.numpy(), slice_2d_c.numpy())

        spectrum_1d_c, _ = torchfits.read(f"{test_file}[1][2,3,*]")
        print("\n1D Spectrum (using CFITSIO):")
        print(f"  Shape: {spectrum_1d_c.shape}")
        assert np.allclose(spectrum_1d.numpy().squeeze(), spectrum_1d_c.numpy().squeeze())

    except RuntimeError as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()