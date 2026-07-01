import torch
import torchfits._C as cpp
import traceback

try:
    cpp.read_fits_table_rows_numpy("| echo 'hello'", 1, [], 1, -1, False)
    print("Failed: Should throw security error for table")
except RuntimeError as e:
    print(f"Success for table: Caught {e}")
except Exception as e:
    print(f"Failed for table: Caught unexpected error {e}")
