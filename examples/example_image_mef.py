"""
Example: multi-extension FITS (MEF) files with torchfits.open and read_hdus.
"""

import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import Table

import torchfits


def _create_test_file(path: str) -> None:
    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(np.arange(100, dtype=np.float32).reshape(10, 10), name="SCI")
    err = fits.ImageHDU(np.random.rand(20, 20).astype(np.float32), name="ERR")
    catalog = Table(
        rows=[(150.0, 45.0, 10.0), (151.0, 46.0, 12.0), (152.0, 47.0, 15.0)],
        names=("ra", "dec", "flux"),
        dtype=("f8", "f8", "f4"),
    )
    cat_hdu = fits.BinTableHDU(catalog, name="CATALOG")
    fits.HDUList([primary, sci, err, cat_hdu]).writeto(path, overwrite=True)


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

        # Read table by EXTNAME via unified read (read_table requires int hdu)
        catalog = torchfits.read(
            path, hdu="CATALOG", mode="table", columns=["ra", "flux"]
        )
        print(f"read CATALOG: ra={catalog['ra'].tolist()}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
