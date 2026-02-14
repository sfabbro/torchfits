import os
import tempfile

import numpy as np
from astropy.table import Table

import torchfits


def create_table(path: str) -> None:
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


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        filename = f.name

    try:
        create_table(filename)

        table, _ = torchfits.read(filename, hdu=1, return_header=True)

        try:
            df = torchfits.to_pandas(table, decode_bytes=True, vla_policy="object")
            print("Pandas DataFrame")
            print(df)
        except ImportError as exc:
            print(f"Pandas not installed: {exc}")

        try:
            arrow = torchfits.to_arrow(table, decode_bytes=True, vla_policy="list")
            print("PyArrow schema")
            print(arrow.schema)
        except ImportError as exc:
            print(f"PyArrow not installed: {exc}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)
