"""
Example: FITS table I/O — tensor reads, Arrow reads, predicate pushdown, and write.
"""

import os
import tempfile

import numpy as np
import torch
from astropy.table import Table

import torchfits


def _create_test_file(path: str) -> None:
    from astropy.io import fits

    table = Table(
        {
            "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
            "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "id": np.array([1, 2, 3], dtype=np.int32),
            "flag": np.array([True, False, True], dtype=bool),
        }
    )
    fits.BinTableHDU(table, name="MY_TABLE").writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # --- Tensor path: read_table returns dict[str, Tensor] ---
        tensors = torchfits.read_table(path, hdu=1)
        print("read_table columns:", list(tensors.keys()))
        print(f"  ra: {tensors['ra'].tolist()}")

        # Column projection and row slice on the tensor path
        subset = torchfits.read_table_rows(
            path, hdu=1, start_row=1, num_rows=2, columns=["id", "flag"]
        )
        print(
            f"read_table_rows id={subset['id'].tolist()}, flag={subset['flag'].tolist()}"
        )

        # --- Arrow path: table.read returns pyarrow.Table ---
        arrow_table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ra", "dec", "flux"],
            where="flux >= 2.0",
        )
        print(f"table.read (where flux >= 2): {arrow_table.num_rows} rows")
        print(f"  flux values: {arrow_table.column('flux').to_pylist()}")

        # Stream large tables in fixed-size chunks
        chunks = list(torchfits.stream_table(path, hdu=1, chunk_rows=2, columns=["id"]))
        print(
            f"stream_table: {len(chunks)} chunk(s), ids={[c['id'].tolist() for c in chunks]}"
        )

        # --- Write back with table.write ---
        out_path = path.replace(".fits", "_out.fits")
        new_data = {
            "ra": torch.tensor([300.0, 301.0], dtype=torch.float64),
            "dec": torch.tensor([50.0, 51.0], dtype=torch.float64),
        }
        torchfits.table.write(
            out_path,
            new_data,
            header={"EXTNAME": "FILTERED"},
            overwrite=True,
        )
        written = torchfits.read_table(out_path, hdu=1)
        print(f"table.write round-trip: {written['ra'].tolist()}")
        os.unlink(out_path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
