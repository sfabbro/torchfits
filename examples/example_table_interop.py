"""
Example: variable-length array columns and interop with Pandas, Arrow, and Polars.
"""

import os
import tempfile

import numpy as np
from astropy.table import Table

import torchfits


def _create_table(path: str) -> None:
    vla = np.array([np.array([1, 2]), np.array([3])], dtype=object)
    table = Table(
        {
            "RA": np.array([10.1, 10.2], dtype=np.float64),
            "DEC": np.array([-2.1, -2.2], dtype=np.float64),
            "NAME": np.array(["STAR_A", "STAR_B"], dtype="U8"),
            "VLA": vla,
        }
    )
    table.write(path, format="fits", overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_table(path)

        # Tensor dict path (good for GPU / torch workflows)
        table_dict = torchfits.read_table(path, hdu=1)
        print("read_table columns:", list(table_dict.keys()))

        # Arrow path (good for analytics / interop)
        arrow_table = torchfits.table.read(path, hdu=1, decode_bytes=True)
        print("table.read schema:", arrow_table.schema)

        # Convert in-memory tensor dict to other formats
        try:
            df = torchfits.to_pandas(table_dict, decode_bytes=True, vla_policy="object")
            print("\nto_pandas:")
            print(df)
        except ImportError as exc:
            print(f"Pandas not installed: {exc}")

        try:
            arrow = torchfits.to_arrow(table_dict, decode_bytes=True, vla_policy="list")
            print("\nto_arrow schema:", arrow.schema)
        except ImportError as exc:
            print(f"PyArrow not installed: {exc}")

        try:
            import polars as pl  # noqa: F401

            pl_df = torchfits.to_polars(
                table_dict, decode_bytes=True, vla_policy="list"
            )
            print("\nto_polars:")
            print(pl_df)
        except ImportError:
            print("Polars not installed; skipping to_polars")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
