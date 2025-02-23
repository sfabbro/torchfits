import torchfits
import numpy as np
import os
from astropy.io import fits

def create_test_file(filename):
    if not os.path.exists(filename):
        data = np.arange(1024, dtype=np.float32).reshape(32, 32)
        hdu = fits.PrimaryHDU(data)
        hdu.header['CRPIX1'] = 16.0
        hdu.header['CRPIX2'] = 16.0
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
        print(f" Updated CRPIX1: {header['CRPIX1']}") # Should be 6.

        #Read to the end
        start = [10, 15]
        shape = [5, -1] #Read to the end of second dimension
        data, header = torchfits.read(test_file, hdu=1, start=start, shape=shape)
        print("\nCutout to end:")
        print(f"  Data shape: {data.shape}")
        print(f" Updated CRPIX1: {header['CRPIX1']}")

    except RuntimeError as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()