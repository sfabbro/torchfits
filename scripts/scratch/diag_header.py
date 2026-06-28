#!/usr/bin/env python3
"""Dumps the FITS header to see what TFORM torchfits wrote for NAME.

The on-disk bytes are correct (verified by diag_string_bytes_v2.py), but fitsio
reads dtype='<U21'. The hypothesis: torchfits writes TFORM='21A' (or similar)
when it should be '8A', so fitsio reads 21 bytes per row instead of 8.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torchfits  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="torchfits_diag_hdr_") as tmp:
        path = Path(tmp) / "diag.fits"
        ids = np.array([10, 20], dtype=np.int32)
        flux = np.array([1.5, 2.5], dtype=np.float32)
        bits = np.array(
            [
                [True, False, True, False, True, False, True, False],
                [False, True, False, True, False, True, False, True],
            ],
            dtype=np.bool_,
        )
        names8 = np.array(["alpha", "bravo"], dtype="S8")
        z = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

        torchfits.table.write(
            str(path),
            {"ID": ids, "FLUX": flux, "FLAGS": bits, "NAME": names8, "Z": z},
            schema={
                "ID": {"format": "J"},
                "FLAGS": {"format": "8X"},
                "NAME": {"format": "8A"},
                "Z": {"format": "C"},
            },
            overwrite=True,
        )

        new_names = np.array(["new111", "new222"], dtype="S8")
        torchfits.table.update_rows(
            str(path),
            {"NAME": new_names},
            row_slice=slice(0, 2),
            hdu=1,
            mmap=True,
        )

        # Print the BinTableHDU header.
        from astropy.io import fits
        with fits.open(path, mode="readonly") as hdul:
            print("--- BinTableHDU Header ---")
            print(repr(hdul[1].header))
            print()
            print("--- astropy decode of NAME column ---")
            column = hdul[1].data["NAME"]
            print(f"dtype: {column.dtype}, shape: {column.shape}")
            print(f"content:\n{column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
