"""
Example: read and slice 3D FITS data cubes.
"""

import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits


def _create_test_file(path: str) -> None:
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    hdu = fits.PrimaryHDU(data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CTYPE3"] = "VELO-LSR"
    hdu.header["CRVAL1"] = 200.0
    hdu.header["CRVAL2"] = 45.0
    hdu.header["CRVAL3"] = 1000.0
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CRPIX2"] = 1.0
    hdu.header["CRPIX3"] = 1.0
    hdu.header["CDELT1"] = -0.01
    hdu.header["CDELT2"] = 0.01
    hdu.header["CDELT3"] = 5.0
    hdu.writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # Full cube via read_tensor
        cube = torchfits.read_tensor(path, hdu=0)
        print(f"read_tensor cube shape: {cube.shape}")  # (z, y, x) FITS order

        _, header = torchfits.read(path, hdu=0, return_header=True)
        print(
            f"WCS axes: {header.get('CTYPE1')}, "
            f"{header.get('CTYPE2')}, {header.get('CTYPE3')}"
        )

        # 2D plane: slice the z axis on the in-memory cube
        plane = cube[1, :, :]
        print(f"plane cube[1,:,:] shape: {plane.shape}")

        # 1D spectrum at fixed (x, y)
        spectrum = cube[:, 1, 2]
        print(f"spectrum cube[:,1,2] shape: {spectrum.shape}")

        # Tensor slicing on the full cube
        print(f"Manual plane cube[0,:,:]: {cube[0, :, :].shape}")
        print(f"Manual spectrum cube[:,1,2]: {cube[:, 1, 2].shape}")
        print(f"Manual sub-cube cube[:,1:3,2:4]: {cube[:, 1:3, 2:4].shape}")

        accel = None
        if torch.backends.mps.is_available():
            accel = "mps"
        elif torch.cuda.is_available():
            accel = "cuda"
        if accel:
            gpu_cube = torchfits.read_tensor(path, hdu=0, device=accel)
            print(f"GPU cube device: {gpu_cube.device}")
        else:
            print("No accelerator available; skipping device= read")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
