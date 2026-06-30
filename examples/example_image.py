"""
Example: read and write FITS images with torchfits.
"""

import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits


def _create_test_file(path: str) -> None:
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    hdu = fits.PrimaryHDU(data)
    hdu.header["OBJECT"] = "M31"
    hdu.header["EXPTIME"] = 120.0
    hdu.writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # Tensor-only read (returns torch.Tensor)
        image = torchfits.read_tensor(path, hdu=0)
        print(f"read_tensor: shape={image.shape}, dtype={image.dtype}")

        # Unified read with header
        data, header = torchfits.read(path, hdu=0, return_header=True)
        print(f"read: OBJECT={header['OBJECT']}, EXPTIME={header['EXPTIME']}")

        # Header without loading pixels
        hdr = torchfits.get_header(path, hdu=0)
        print(f"get_header: NAXIS={hdr['NAXIS']}, BITPIX={hdr['BITPIX']}")

        # Write tensor back to FITS
        scaled = data * 2.0
        out_path = path.replace(".fits", "_out.fits")
        torchfits.write_tensor(
            out_path, scaled, header={"OBJECT": "M31 x2"}, overwrite=True
        )
        roundtrip = torchfits.read_tensor(out_path)
        print(
            "write_tensor round-trip:",
            torch.allclose(roundtrip.cpu(), scaled.cpu()),
        )
        os.unlink(out_path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
