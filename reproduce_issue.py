
import torchfits
from astropy.io import fits
import numpy as np
import os

def test_table_read():
    filename = "test_table_repro.fits"
    try:
        # Create a simple table
        c1 = fits.Column(name='a', format='J', array=np.array([1, 2, 3]))
        c2 = fits.Column(name='b', format='E', array=np.array([4.0, 5.0, 6.0]))
        hdu = fits.BinTableHDU.from_columns([c1, c2])
        hdu.writeto(filename, overwrite=True)

        # Read with torchfits
        print(f"Reading {filename}...")
        data, header = torchfits.read(filename, hdu=1, device='cpu')
        
        print(f"Type of data: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
        else:
            print("Data is not a dict!")
            print("Shape:", data.shape)
            print("Dtype:", data.dtype)

    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_table_read()
