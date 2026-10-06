"""Example: Read, slice, and write multi-dimensional 3D and 4D FITS data cubes.

Demonstrates:
- Creating and writing a 3D spectral data cube [velocity, y, x].
- Slicing 3D cubes on CPU and GPU.
- Creating and writing a 4D hypercube [polarization, velocity, y, x] representing multi-Stokes channels.
- Multi-dimensional slicing to extract sub-cubes, spatial planes, and 1D spectra.
"""

import os
import tempfile

import numpy as np
import torch

import torchfits


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. 3D FITS Data Cube Example [velocity, y, x]
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix="_cube3d.fits", delete=False) as fh3d:
        path_3d = fh3d.name

    try:
        # Generate synthetic 3D cube: 5 spectral channels, 128x128 pixels
        shape_3d = (5, 128, 128)
        data_3d = torch.arange(np.prod(shape_3d), dtype=torch.float32).reshape(shape_3d)

        # Write 3D tensor to FITS
        torchfits.write(
            path_3d,
            data_3d,
            header={"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CTYPE3": "VELO-LSR"},
            overwrite=True,
        )

        # Read back using read_tensor
        cube_3d = torchfits.read_tensor(path_3d, hdu=0)
        print("--- 3D Data Cube ---")
        print(f"Read 3D cube tensor shape: {cube_3d.shape}")

        # Slicing operations
        plane_2d = cube_3d[2, :, :]  # Spatial plane at channel index 2
        spec_1d = cube_3d[:, 64, 64]  # 1D spectrum at spatial coordinates (64, 64)
        sub_cube = cube_3d[1:4, 32:96, 32:96]  # Sub-cube extraction

        print(f"  Spatial 2D slice shape (z=2): {plane_2d.shape}")
        print(f"  Spectral 1D slice shape (y=64, x=64): {spec_1d.shape}")
        print(f"  Sub-cube shape: {sub_cube.shape}")

        # Check GPU reading if available
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        if device != "cpu":
            gpu_cube = torchfits.read_tensor(path_3d, hdu=0, device=device)
            print(f"  Successfully loaded 3D cube directly to GPU ({gpu_cube.device})")

    finally:
        if os.path.exists(path_3d):
            os.unlink(path_3d)

    print("\n--- 4D Hypercube ---")

    # -------------------------------------------------------------------------
    # 2. 4D FITS Hypercube Example [polarization, velocity, y, x]
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix="_cube4d.fits", delete=False) as fh4d:
        path_4d = fh4d.name

    try:
        # Generate synthetic 4D hypercube:
        # 2 Stokes polarizations (I, Q), 3 velocity channels, 64x64 pixels
        shape_4d = (2, 3, 64, 64)
        data_4d = torch.arange(np.prod(shape_4d), dtype=torch.float32).reshape(shape_4d)

        # Write 4D tensor to FITS
        torchfits.write(
            path_4d,
            data_4d,
            header={
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CTYPE3": "VELO-LSR",
                "CTYPE4": "STOKES",
                "CRVAL4": 1.0,  # Stokes parameters (1=I, 2=Q)
                "CDELT4": 1.0,
            },
            overwrite=True,
        )

        # Read 4D tensor back
        hypercube_4d = torchfits.read_tensor(path_4d, hdu=0)
        print(f"Read 4D hypercube tensor shape: {hypercube_4d.shape}")

        # Extract 3D spectral cube for a specific Stokes polarization (Stokes I, index 0)
        stokes_i_cube = hypercube_4d[0, :, :, :]
        print(f"  Stokes I (pol=0) 3D sub-cube shape: {stokes_i_cube.shape}")

        # Extract a 2D spatial plane for polarization Q (index 1) and velocity channel 1
        plane_4d = hypercube_4d[1, 1, :, :]
        print(f"  Stokes Q (pol=1), vel=1 spatial plane shape: {plane_4d.shape}")

        # Extract a 1D spectrum at spatial index (32, 32) for polarization I (index 0)
        spectrum_4d = hypercube_4d[0, :, 32, 32]
        print(f"  Stokes I (pol=0) 1D spectrum shape: {spectrum_4d.shape}")

    finally:
        if os.path.exists(path_4d):
            os.unlink(path_4d)


if __name__ == "__main__":
    main()
