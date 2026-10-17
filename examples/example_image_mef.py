"""
Example: multi-extension FITS (MEF) files with torchfits.open and read_hdus.
"""

import os
import tempfile

import numpy as np
import torch

import torchfits
from torchfits.hdu import HDUList, TableHDU, TensorHDU


def _create_test_file(path: str) -> None:
    catalog = {
        "ra": np.array([150.0, 151.0, 152.0], dtype=np.float64),
        "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
        "flux": np.array([10.0, 12.0, 15.0], dtype=np.float32),
    }
    HDUList(
        [
            TensorHDU(torch.empty(0)),
            TensorHDU(
                torch.from_numpy(np.arange(100, dtype=np.float32).reshape(10, 10)),
                {"EXTNAME": "SCI"},
            ),
            TensorHDU(
                torch.from_numpy(np.random.rand(20, 20).astype(np.float32)),
                {"EXTNAME": "ERR"},
            ),
            TableHDU(catalog, header={"EXTNAME": "CATALOG"}),
        ]
    ).write(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # Walk HDUs with context manager
        with torchfits.open(path) as hdul:
            print(f"HDU count: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                extname = hdu.header.get("EXTNAME", "PRIMARY")
                if hdu.data is None:
                    print(f"  [{i}] {extname}: empty")
                elif hasattr(hdu.data, "keys"):
                    print(f"  [{i}] {extname}: table, columns={list(hdu.data.keys())}")
                else:
                    print(f"  [{i}] {extname}: image, shape={hdu.data.shape}")

            # Filter table rows in-place on the HDU handle
            filtered = hdul["CATALOG"].filter("flux > 11")
            print(f"filter(flux > 11): {len(filtered.data['ra'])} rows")

        # Batch-read named image extensions
        sci, err = torchfits.read_hdus(path, hdus=["SCI", "ERR"])
        print(f"read_hdus SCI={sci.shape}, ERR={err.shape}")

        # Table by EXTNAME: unified read or table.read_torch both accept str hdu
        catalog = torchfits.read(
            path, hdu="CATALOG", mode="table", columns=["ra", "flux"]
        )
        print(f"read CATALOG: ra={catalog['ra'].tolist()}")
        by_name = torchfits.table.read_torch(path, hdu="CATALOG", columns=["ra"])
        print(f"read_torch CATALOG: ra={by_name['ra'].tolist()}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
