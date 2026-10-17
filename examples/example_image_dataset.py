"""Example: FitsImageDataset + make_loader.

Prefer ``read_tensor`` for a single file. Use a Dataset when you need many
files as a PyTorch Dataset (shuffle, workers, epochs). ``make_loader`` is
DataLoader with torchfits cache warm-up defaults — not a separate API.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

import torchfits
from torchfits.data import FitsImageDataset, make_loader


def _create_dummy_fits(
    data_dir: str, num_files: int = 8, size: tuple[int, int] = (64, 64)
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(num_files):
        data = rng.random(size).astype(np.float32)
        torchfits.write(
            os.path.join(data_dir, f"image_{i:03d}.fits"),
            data,
            header={"LABEL": i % 2},
            overwrite=True,
        )


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="torchfits_dataset_")
    try:
        _create_dummy_fits(data_dir)
        file_pattern = os.path.join(data_dir, "*.fits")

        # device="cuda" works the same when a GPU is available
        dataset = FitsImageDataset(
            file_pattern,
            hdu=0,
            label_key="LABEL",
            device="cpu",
        )
        loader = make_loader(
            dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )

        for i, (images, labels) in enumerate(loader):
            print(
                f"batch {i}: images={images.shape} on {images.device}, "
                f"labels={labels.tolist()}"
            )
            if i >= 1:
                break

        # Inference-style batch of paths (no Dataset required)
        batch = torchfits.read_batch(dataset.files[:4], hdu=0)
        print(f"read_batch: {len(batch)} images, first shape={batch[0].shape}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
