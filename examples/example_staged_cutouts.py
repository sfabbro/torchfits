"""Example: Asynchronous Staged Remote Mosaics and Fast Cutout Extraction.

This script demonstrates training pipeline data loading from large survey mosaics
(e.g., CFHT MegaCam, Subaru HSC, Rubin LSST, MegaPipe).

Key features demonstrated:
1. Ephemeral scratch staging (auto-detecting $SLURM_TMPDIR / $TMPDIR).
2. Background prefetching of the next mosaic while sampling cutouts from the current mosaic.
3. Fast in-memory / zero-copy subset reading via open_subset_reader.
4. Automatic cleanup/eviction of consumed files to maintain a bounded local disk footprint.
5. Standard PyTorch DataLoader integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from torch.utils.data import DataLoader

from torchfits.data import FitsStagedCutoutIterableDataset


def create_demo_mosaics(root_dir: Path, count: int = 3) -> list[str]:
    """Create simulated survey mosaic FITS files."""
    paths = []
    for i in range(count):
        path = root_dir / f"survey_mosaic_{i:02d}.fits"
        # Simulated 2000x2000 mosaic image
        data = np.random.randn(2000, 2000).astype(np.float32)
        fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
        paths.append(str(path))
    return paths


def main() -> None:
    print("--- torchfits Staged Cutout Extraction Demo ---")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        mosaics_dir = temp_path / "survey_archive"
        mosaics_dir.mkdir()
        staging_dir = temp_path / "local_scratch"
        staging_dir.mkdir()

        # 1. Create simulated survey mosaics
        mosaic_files = create_demo_mosaics(mosaics_dir, count=4)
        print(f"Created {len(mosaic_files)} simulated survey mosaics.")

        # 2. Instantiate FitsStagedCutoutIterableDataset
        # Each mosaic yields 10 random 128x128 cutout stamps before moving to the next.
        dataset = FitsStagedCutoutIterableDataset(
            paths=mosaic_files,
            cutouts_per_file=10,
            cutout_size=(128, 128),
            staging_dir=staging_dir,
            cleanup=False,  # Keep False for local files; True for remote downloads
            device="cpu",
            shuffle_files=True,
            shuffle_buffer_size=20,  # Mix cutouts across mosaics in-flight
        )

        # 3. Create PyTorch DataLoader
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        total_batches = 0
        total_samples = 0
        for batch_idx, batch in enumerate(loader):
            total_batches += 1
            total_samples += batch.shape[0]
            if batch_idx == 0:
                print(f"Batch 0 tensor shape: {batch.shape} (dtype: {batch.dtype})")

        print(
            f"Successfully streamed {total_samples} cutout stamps across "
            f"{total_batches} batches without loading full mosaics into memory!"
        )


if __name__ == "__main__":
    main()
