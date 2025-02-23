import torch
import torchfits
import os

def main():
    # Create a dummy FITS file for the example (if it doesn't exist).
    example_file = "example.fits"
    if not os.path.exists(example_file):
        print(f"Creating example FITS file: {example_file}")
        import numpy as np
        from astropy.io import fits
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
        hdu.writeto(example_file)

        # Create a dummy MEF FITS file for the example.
        mef_file = "example_mef.fits"
        primary_hdu = fits.PrimaryHDU() #Empty primary
        ext1 = fits.ImageHDU(np.arange(100, dtype=np.float32).reshape(10, 10), name='EXT1')
        ext2 = fits.ImageHDU(np.arange(100, 200, dtype=np.float32).reshape(10, 10), name='EXT2')
        hdul = fits.HDUList([primary_hdu, ext1, ext2])
        hdul.writeto(mef_file, overwrite=True)

        # Create example table
        table_file = "example_table.fits"
        from astropy.table import Table
        names = ['ra', 'dec', 'flux', 'id', 'comments']
        formats = ['D', 'D', 'E', 'J', '20A']  # Double, Double, Float, Integer, String
        data = {
            'ra': np.array([200.0, 201.0, 202.0], dtype=np.float64),
            'dec': np.array([45.0, 46.0, 47.0], dtype=np.float64),
            'flux': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'id': np.array([1, 2, 3], dtype=np.int32),
            'comments': np.array(["This is star 1", "This is star 2", "This is star 3"], dtype='U20')
        }
        table = Table(data) #Create astropy table
        hdu = fits.BinTableHDU(table)
        hdu.writeto(table_file, overwrite=True)



    # --- Read Full HDU ---
    print("\n--- Reading Full HDU ---")
    try:
        data, header = torchfits.read(example_file)
        print(f"  Data shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  CRVAL1: {header.get('CRVAL1')}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Read Cutout (CFITSIO string) ---
    print("\n--- Reading Cutout (CFITSIO String) ---")
    try:
        cutout_data, _ = torchfits.read(f"{example_file}[0][2:5,3:7]")
        print(f"  Cutout shape: {cutout_data.shape}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Read Region (start, shape) ---
    print("\n--- Reading Region (Start, Shape) ---")
    try:
        start = [2, 3]
        shape = [3, 4]
        cutout_data, _ = torchfits.read_region(example_file, 1, start, shape)
        print(f"  Cutout shape: {cutout_data.shape}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Read 1D Spectrum ---
    print("\n--- Reading 1D Spectrum ---")
    try:
        spectrum_data, _ = torchfits.read_spectrum(example_file, 1, 2, 4)  # Axis 2, index 4
        print(f"  Spectrum shape: {spectrum_data.shape}")
    except RuntimeError as e:
        print(f"  Error: {e}")
    # --- Read a binary table ---
    print("\n--- Reading Table ---")
    try:
        table_data = torchfits.read("example_table.fits[1]")
        print(f"  Table columns: {list(table_data.keys())}")
        print(f"  'ra' column: {table_data['ra']}")
        print(f"  'comments' column: {table_data['comments']}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Get Header ---
    print("\n--- Getting Header ---")
    try:
        header = torchfits.get_header(example_file, 1)
        print(f"  CRVAL1: {header.get('CRVAL1')}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Get Dimensions ---
    print("\n--- Getting Dimensions ---")
    try:
        dims = torchfits.get_dims(example_file, 1)
        print(f"  Dimensions: {dims}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Get Header Value ---
    print("\n--- Getting Header Value ---")
    try:
        naxis = torchfits.get_header_value(example_file, 1, "NAXIS")
        print(f"  NAXIS: {naxis}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Get HDU Type ---
    print("\n--- Getting HDU Type ---")
    try:
        hdu_type = torchfits.get_hdu_type(example_file, 1)
        print(f"  HDU Type: {hdu_type}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Get Number of HDUs ---
    print("\n--- Getting Number of HDUs ---")
    try:
        num_hdus = torchfits.get_num_hdus("example_mef.fits")
        print(f"  Number of HDUs: {num_hdus}")
    except RuntimeError as e:
        print(f"  Error: {e}")

     # --- Iterate through HDUs in MEF ---
    print("\n--- Iterating through HDUs in MEF ---")
    try:
        num_hdus = torchfits.get_num_hdus("example_mef.fits")
        for hdu_num in range(1, num_hdus + 1):
            print(f"  HDU {hdu_num}:")
            try:
                # Use a try-except block *inside* the loop to handle potential errors
                # with individual HDUs, but continue to the next HDU.
                hdu_type = torchfits.get_hdu_type("example_mef.fits", hdu_num)
                print(f"    Type: {hdu_type}")

                header = torchfits.get_header("example_mef.fits", hdu_num)
                print(f"    EXTNAME: {header.get('EXTNAME', 'N/A')}")  # Handle missing EXTNAME

                # Only try to read data if it's an image HDU
                if hdu_type == "IMAGE":
                  data, _ = torchfits.read(f"example_mef.fits[{hdu_num}]")
                  print(f"    Data shape: {data.shape}")

            except RuntimeError as e:
                print(f"    Error reading HDU {hdu_num}: {e}")

    except RuntimeError as e:
        print(f"  Error: {e}")



if __name__ == "__main__":
    main()
    