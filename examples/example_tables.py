import os

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table

import torchfits


def create_test_file(filename):
    if not os.path.exists(filename):
        names = ["ra", "dec", "flux", "id", "comments", "flag"]
        formats = ["D", "D", "E", "J", "20A", "B"]  # Added a boolean column
        data = {
            "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
            "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "id": np.array([1, 2, 3], dtype=np.int32),
            "comments": np.array(
                ["This is star 1", "This is star 2", "This is star 3"], dtype="U20"
            ),
            "flag": np.array([True, False, True], dtype=bool),  # Boolean col
        }
        table = Table(data)
        hdu = fits.BinTableHDU(table, name="MY_TABLE")
        hdu.writeto(filename, overwrite=True)


def main():
    test_file = "table_example.fits"
    create_test_file(test_file)

    # Read the entire table
    try:
        table_data, header = torchfits.read(test_file, hdu="MY_TABLE")  # Read by name
        print("Table Data:")
        for col_name in table_data:
            print(f"  Column '{col_name}': {table_data[col_name]}")
            if hasattr(table_data[col_name], "dtype"):
                print(f"    Data Type: {table_data[col_name].dtype}")
            else:
                print(f"    Data Type: string list")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read specific columns
    try:
        table_subset, _ = torchfits.read(test_file, hdu=2, columns=["ra", "id"])
        print("\nSubset of Columns (ra, id):")
        if table_subset is not None:
            for col_name in table_subset:
                print(f"  Column '{col_name}': {table_subset[col_name]}")
        else:
            print("  No data returned")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # Read a subset of rows
    try:
        table_rows, _ = torchfits.read(test_file, hdu=2, start_row=1, num_rows=2)
        print("\nSubset of Rows (start_row=1, num_rows=2):")
        if table_rows is not None:
            for col_name in table_rows:
                print(f"  Column '{col_name}': {table_rows[col_name]}")
        else:
            print("  No data returned")
    except RuntimeError as e:
        print(f" Error: {e}")

    # Read specific columns and rows
    try:
        table_subset, _ = torchfits.read(
            test_file, hdu=2, columns=["id", "flag"], start_row=0, num_rows=2
        )
        print("\nSubset of Columns and Rows:")
        if table_subset is not None:
            for col_name in table_subset:
                print(f"  Column '{col_name}': {table_subset[col_name]}")
        else:
            print("  No data returned")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # --- Test different cache capacities ---
    print("\n--- Testing with different cache capacities ---")
    for capacity in [0, 2, 10]:
        try:
            table_data, _ = torchfits.read(test_file, hdu=2, cache_capacity=capacity)
            print(f"\nCache Capacity: {capacity}")
            if table_data is not None:
                print(
                    f"  Number of Columns: {len(table_data)}"
                )  # Just print the number of columns
            else:
                print("  No data returned")
        except RuntimeError as e:
            print(f"  Error with cache_capacity={capacity}: {e}")

    # --- Test GPU read (if available) ---
    if torch.cuda.is_available():
        print("\n--- Testing GPU Read ---")
        try:
            table_data, _ = torchfits.read(test_file, hdu=2, device="cuda")
            if table_data is not None:
                print(f"  Data device, first column: {table_data['ra'].device}")
            else:
                print("  No data returned")
        except RuntimeError as e:
            print(f"  Error reading to GPU: {e}")
    else:
        print("\n--- CUDA not available, skipping GPU read test ---")


if __name__ == "__main__":
    main()
