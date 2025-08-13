import torch
import torchfits
import tempfile
import os

# Minimal table: 2 columns, 5 rows
rows = 5
table = {
    "A": torch.arange(rows, dtype=torch.int32),
    "B": torch.arange(rows, dtype=torch.float32) * 10,
}

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "test_table.fits")
    torchfits.write(path, table, overwrite=True)
    print("Wrote table to", path)
    # Try reading full table
    full = torchfits.read(path, hdu=1, format="table")
    print("Full table:", {k: v.tolist() for k, v in full.data.items()})
    # Try reading a cutout (rows 2:4)
    cut = torchfits.read(path, hdu=1, format="table", start_row=2, num_rows=2)
    print("Cutout rows 2:4:", {k: v.tolist() for k, v in cut.data.items()})
