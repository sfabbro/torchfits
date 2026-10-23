"""
Example: FITS tables as dataframes.

Primary path: ``table.read`` → Arrow (portable dataframe). The Arrow columns
are zero-copy views over the native column buffers, and ``table.read`` does not
import PyTorch. Also: ``table.read_torch`` (tensor columns),
``open_table_reader``, streaming, then mutations.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402


def _create_test_file(path: str) -> None:
    torchfits.table.write(
        path,
        {
            "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
            "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "id": np.array([1, 2, 3], dtype=np.int32),
            "flag": np.array([True, False, True], dtype=bool),
        },
        overwrite=True,
        extname="MY_TABLE",
    )


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # --- Primary: dataframe via Arrow ---
        arrow_df = torchfits.table.read(
            path,
            hdu=1,
            columns=["ra", "dec", "flux"],
            where="flux >= 2.0",
        )
        print(f"table.read (where flux >= 2): {arrow_df.num_rows} rows")
        print(f"  flux values: {arrow_df.column('flux').to_pylist()}")

        tensors = torchfits.table.read_torch(path, hdu=1)
        print("read_torch columns:", list(tensors.keys()))
        print(f"  ra: {tensors['ra'].tolist()}")

        with torchfits.open_table_reader(path, hdu=1) as reader:
            chunk = reader.read_torch(columns=["id", "ra"], start_row=1, num_rows=2)
            print(
                f"open_table_reader: {reader.num_rows()} rows, "
                f"first ids={chunk['id'].tolist()}"
            )

        chunks = list(
            torchfits.table.scan_torch(path, hdu=1, batch_size=2, columns=["id"])
        )
        print(
            f"scan_torch: {len(chunks)} chunk(s), "
            f"ids={[c['id'].tolist() for c in chunks]}"
        )

        try:
            pl_df = torchfits.table.read_polars(path, hdu=1)
            print(f"read_polars: {pl_df.shape[0]} rows")
        except ImportError:
            print("read_polars skipped (polars not installed)")

        # --- real file: filter a Chandra events table, when cached ---
        chandra = try_ensure_sample("chandra_events")
        if chandra is not None:
            bright = torchfits.table.read(
                str(chandra), hdu=1, columns=["energy"], where="energy > 5000"
            )
            print(f"chandra_events energy>5000eV: {bright.num_rows} rows")
        else:
            print("chandra_events sample not cached; skipping real-file filter demo")

        # --- mutations ---
        torchfits.table.append_rows(
            path,
            {
                "ra": np.array([203.0], dtype=np.float64),
                "dec": np.array([48.0], dtype=np.float64),
                "flux": np.array([4.0], dtype=np.float32),
                "id": np.array([4], dtype=np.int32),
                "flag": np.array([False], dtype=bool),
            },
            hdu=1,
        )
        torchfits.table.update_rows(
            path,
            {"flux": np.array([9.9, 9.9], dtype=np.float32)},
            row_slice=slice(1, 3),
            hdu=1,
        )
        torchfits.table.insert_column(
            path,
            "quality",
            np.array([100, 100, 100, 100], dtype=np.int16),
            hdu=1,
            format="I",
        )
        torchfits.table.rename_columns(path, {"ra": "right_ascension"}, hdu=1)
        torchfits.table.drop_columns(path, ["flag"], hdu=1)
        modified = torchfits.table.read_torch(path, hdu=1)
        print("after mutations:", list(modified.keys()))
        print(
            f"  flux={modified['flux'].tolist()} quality={modified['quality'].tolist()}"
        )

        out_path = path.replace(".fits", "_out.fits")
        torchfits.table.write(
            out_path,
            {
                "ra": torch.tensor([300.0, 301.0], dtype=torch.float64),
                "dec": torch.tensor([50.0, 51.0], dtype=torch.float64),
            },
            header={"EXTNAME": "FILTERED"},
            overwrite=True,
        )
        written = torchfits.table.read_torch(out_path, hdu=1)
        print(f"table.write round-trip: {written['ra'].tolist()}")
        os.unlink(out_path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
