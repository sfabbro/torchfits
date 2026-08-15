"""Example: Constant-Memory Streaming of 3D Datacubes and 1D Spectra.

This script demonstrates using FitsCubeIterableDataset and FitsSpectrumIterableDataset
for deep learning workloads on integral field units (IFUs) and spectroscopic surveys.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from torch.utils.data import DataLoader

from torchfits.data import FitsCubeIterableDataset, FitsSpectrumIterableDataset


def create_demo_data(root_dir: Path) -> tuple[list[str], list[str]]:
    """Generate sample IFU 3D cubes and multi-arm 1D spectra."""
    cube_paths = []
    for i in range(3):
        path = root_dir / f"ifu_cube_{i:02d}.fits"
        # Simulated 3D IFU datacube: [30 wavelength channels, 64 Y, 64 X]
        data = np.random.randn(30, 64, 64).astype(np.float32)
        fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
        cube_paths.append(str(path))

    spec_paths = []
    for i in range(4):
        path = root_dir / f"multi_arm_spec_{i:02d}.fits"
        hdus = [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.random.randn(100).astype(np.float32), name="BLUE"),
            fits.ImageHDU(np.random.randn(100).astype(np.float32), name="RED"),
        ]
        fits.HDUList(hdus).writeto(str(path), overwrite=True)
        spec_paths.append(str(path))

    return cube_paths, spec_paths


def main() -> None:
    print("--- torchfits Streaming Cubes & Spectra Demo ---")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cube_files, spec_files = create_demo_data(temp_path)

        # 1. Stream 3D IFU cubes with optional channel slicing
        print("\n1. Streaming 3D IFU Datacubes (sliced channel index = 5):")
        cube_dataset = FitsCubeIterableDataset(
            cube_files,
            slice_index=5,  # Extracts channel 5 on the fly
            shuffle=True,
            shuffle_buffer_size=10,
        )
        cube_loader = DataLoader(cube_dataset, batch_size=2)
        for batch_idx, batch in enumerate(cube_loader):
            print(f"  Cube Batch {batch_idx}: shape {batch.shape}")

        # 2. Stream multi-arm 1D spectra (stacked layout)
        print("\n2. Streaming Multi-Arm 1D Spectra (layout='stack'):")
        spec_dataset = FitsSpectrumIterableDataset(
            spec_files,
            hdu=["BLUE", "RED"],
            layout="stack",
            shuffle=True,
        )
        spec_loader = DataLoader(spec_dataset, batch_size=2)
        for batch_idx, batch in enumerate(spec_loader):
            flux = batch["flux"]
            print(f"  Spectra Batch {batch_idx}: flux shape {flux.shape}")


if __name__ == "__main__":
    main()
