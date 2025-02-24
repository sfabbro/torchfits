import torchfits
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

def create_test_file(filename):
    if not os.path.exists(filename):
        names = ['ra', 'dec', 'flux', 'id', 'comments', 'flag']
        formats = ['D', 'D', 'E', 'J', '20A', 'B']  # Added a boolean column
        data = {
            'ra': np.array([200.0, 201.0, 202.0], dtype=np.float64),
            'dec': np.array([45.0, 46.0, 47.0], dtype=np.float64),
            'flux': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'id': np.array([1, 2, 3], dtype=np.int32),
            'comments': np.array(["This is star 1", "This is star 2", "This is star 3"], dtype='U20'),
            'flag': np.array([True, False, True], dtype=bool) #Boolean col

        }
        table = Table(data)
        hdu = fits.BinTableHDU(table, name="MY_TABLE")
        hdu.writeto(filename, overwrite=True)

def main():
    test_file = "table_example.fits"
    create_test_file(test_file)

    # Read the entire table
    try:
        table_data = torchfits.read(test_file, hdu="MY_TABLE")  # Read by name
        print("Table Data:")
        for col_name in table_data:
            print(f"  Column '{col_name}': {table_data[col_name]}")
            print(f"    Data Type: {table_data[col_name].dtype}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read specific columns
    try:
        table_subset = torchfits.read(test_file, hdu=1, columns=['ra', 'id'])
        print("\nSubset of Columns (ra, id):")
        for col_name in table_subset:
            print(f"  Column '{col_name}': {table_subset[col_name]}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a subset of rows
    try:
        table_rows = torchfits.read(test_file, hdu=1, start_row=1, num_rows=2)
        print("\nSubset of Rows (start_row=1, num_rows=2):")
        for col_name in table_rows:
            print(f"  Column '{col_name}': {table_rows[col_name]}")
    except RuntimeError as e:
        print(f" Error: {e}")

    # Read specific columns and rows
    try:
        table_subset = torchfits.read(test_file, hdu=1, columns=['id', 'comments'], start_row=0, num_rows=2)
        print("\nSubset of Columns and Rows:")
        for col_name in table_subset:
            print(f"  Column '{col_name}': {table_subset[col_name]}")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 2, 10]:
        try:
            table_data = torchfits.read(test_file, hdu=1, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            print(f"  Number of Columns: {len(table_data)}") # Just print the number of columns
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            table_data = torchfits.read(test_file, hdu=1, device="cuda")
            print(f"  Data device, first column: {table_data['ra'].device}")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")

if __name__ == "__main__":
    main()