# examples/example_mef.py
import torchfits
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

def create_test_file(filename):
    if not os.path.exists(filename):
        primary_hdu = fits.PrimaryHDU()  # Empty primary
        ext1 = fits.ImageHDU(np.arange(100, dtype=np.float32).reshape(10, 10), name='SCI')
        ext2 = fits.ImageHDU(np.random.rand(20, 20), name='ERR')
        # Create some sample table data
        names = ('ra', 'dec', 'flux')
        formats = ('f8', 'f8', 'f4')
        data = [(150.0, 45.0, 10.0), (151.0, 46.0, 12.0), (152.0, 47.0, 15.0)]
        table_data = Table(rows=data, names=names, dtype=formats)
        ext3 = fits.BinTableHDU(table_data, name="CATALOG")

        hdul = fits.HDUList([primary_hdu, ext1, ext2, ext3])
        hdul.writeto(filename, overwrite=True)

def main():
    test_file = "mef_example.fits"
    create_test_file(test_file)

    # Iterate through HDUs
    try:
        num_hdus = torchfits.get_num_hdus(test_file)
        print(f"Number of HDUs: {num_hdus}")

        for i in range(1, num_hdus + 1):  # Iterate through HDUs (1-based)
            print(f"\n--- HDU {i} ---")
            try:
                hdu_type = torchfits.get_hdu_type(test_file, i)
                print(f"  Type: {hdu_type}")

                header = torchfits.get_header(test_file, i)
                print(f"  EXTNAME: {header.get('EXTNAME', 'N/A')}")  # Get EXTNAME, default to 'N/A'

                if hdu_type == "IMAGE":
                    data, _ = torchfits.read(test_file, hdu=i)  # Read by index
                    print(f"  Data shape: {data.shape}")
                elif hdu_type == "BINTABLE":
                    table = torchfits.read(test_file, hdu=i) # Read by index
                    print(f"  Table columns: {list(table.keys())}")

            except RuntimeError as e:
                print(f"  Error reading HDU {i}: {e}")


        # Access by name:
        print("\n--- Accessing HDU by Name ---")
        data, _ = torchfits.read(test_file, hdu="SCI") # String name
        print(f"  SCI data shape: {data.shape}")

        table_data = torchfits.read(test_file, hdu='CATALOG') #String name
        print(f"  CATALOG columns: {list(table_data.keys())}")

        # --- Test different cache capacities ---
        print("\n--- Testing with different cache capacities ---")
        for capacity in [0, 10, 100]:
            try:
                data, _ = torchfits.read(test_file, hdu="SCI", cache_capacity=capacity)
                print(f"\nCache Capacity: {capacity}")
                print(f"  Data shape: {data.shape}")
            except RuntimeError as e:
                print(f"  Error with cache_capacity={capacity}: {e}")


    except RuntimeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()