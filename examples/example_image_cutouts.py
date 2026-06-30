"""
Example: extract image cutouts with read_subset, CFITSIO strings, and
open_subset_reader for repeated reads.
"""

import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits


def _create_test_file(path: str) -> None:
    data = np.arange(1024, dtype=np.float32).reshape(32, 32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["CRPIX1"] = 16.0
    hdu.header["CRPIX2"] = 16.0
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRVAL1"] = 202.5
    hdu.header["CRVAL2"] = 47.5
    hdu.header["CDELT1"] = -0.001
    hdu.header["CDELT2"] = 0.001
    hdu.writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # read_subset: 0-based pixel bounds (x1, y1, x2, y2), exclusive end
        x1, y1, x2, y2 = 15, 10, 23, 15
        cutout = torchfits.read_subset(path, hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)
        print(f"read_subset shape: {cutout.shape}")  # (5, 8)

        # Compare with the equivalent slice from a full read
        full = torchfits.read_tensor(path, hdu=0)
        manual = full[y1:y2, x1:x2]
        print(f"tensor slice matches read_subset: {torch.equal(cutout, manual)}")

        # open_subset_reader: reuse file handle for many cutouts
        with torchfits.open_subset_reader(path, hdu=0) as reader:
            stamp_a = reader(0, 0, 16, 16)
            stamp_b = reader(16, 16, 32, 32)
            print(f"subset_reader stamps: {stamp_a.shape}, {stamp_b.shape}")

        # Full image via read_tensor (contrast with cutout APIs)
        full = torchfits.read_tensor(path, hdu=0)
        print(f"read_tensor full shape: {full.shape}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
