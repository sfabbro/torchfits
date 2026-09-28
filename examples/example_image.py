"""
Example: read and write FITS images with torchfits.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402


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

        # Skinny metadata (no full header dump) + full header when needed
        bitpix, shape = torchfits.read_shape(path, hdu=0)
        print(f"read_shape: bitpix={bitpix}, shape={shape}")
        hdr = torchfits.read_header(path, hdu=0)
        print(f"read_header: OBJECT={hdr['OBJECT']}, EXPTIME={hdr['EXPTIME']}")

        # Write tensor back to FITS
        scaled = data * 2.0
        out_path = path.replace(".fits", "_out.fits")
        torchfits.write_tensor(
            out_path, scaled, header={"OBJECT": "M31 x2"}, overwrite=True
        )
        roundtrip = torchfits.read_tensor(out_path)
        # Printed as a bare bool this was a claim the gate could not see: the
        # example printed "write_tensor round-trip: False" on a corrupted
        # write and still exited 0, so the runner reported PASS. Raise instead.
        if not torch.allclose(roundtrip.cpu(), scaled.cpu()):
            raise RuntimeError(f"write_tensor round-trip lost the data at {out_path}")
        print("write_tensor round-trip: True")
        os.unlink(out_path)
    finally:
        os.unlink(path)

    # Real file, when cached (synthetic path above is the CI-stable primary demo)
    horsehead = try_ensure_sample("horsehead")
    if horsehead is not None:
        real = torchfits.read_tensor(str(horsehead), hdu=0)
        print(
            f"horsehead (real sample): shape={real.shape}, "
            f"min={real.min():.1f}, max={real.max():.1f}"
        )
    else:
        print("horsehead sample not cached; skipping real-file demo")


if __name__ == "__main__":
    main()
