"""
Example: PyTorch Dataset pattern using read_tensor and get_header.
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset

import torchfits


def _create_dummy_fits(
    data_dir: str, num_files: int = 10, size: tuple[int, int] = (64, 64)
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_files):
        data = np.random.rand(*size).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["LABEL"] = i % 2
        hdu.writeto(os.path.join(data_dir, f"image_{i:03d}.fits"), overwrite=True)


class FitsImageDataset(Dataset):
    """Minimal Dataset: header for labels, read_tensor for pixels."""

    def __init__(self, data_dir: str, device: str = "cpu") -> None:
        self.device = device
        self.files: list[str] = []
        self.labels: list[int] = []

        for name in sorted(os.listdir(data_dir)):
            if not name.endswith(".fits"):
                continue
            path = os.path.join(data_dir, name)
            header = torchfits.get_header(path, hdu=0)
            self.files.append(path)
            self.labels.append(int(header["LABEL"]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torchfits.read_tensor(self.files[idx], hdu=0, device=self.device)
        if image.ndim == 2:
            image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="torchfits_dataset_")
    try:
        _create_dummy_fits(data_dir, num_files=8)

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            print(f"\n--- device={device} ---")
            dataset = FitsImageDataset(data_dir, device=device)
            # Warm handle/file caches for repeated epoch training (see docs/benchmarks.md).
            torchfits.cache.optimize_for_dataset(
                dataset.files, avg_file_size_mb=(64 * 64 * 4) / (1024 * 1024)
            )
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                # pin_memory only helps CPU tensors; __getitem__ already places on device.
                pin_memory=False,
            )
            for i, (images, labels) in enumerate(loader):
                print(
                    f"  batch {i}: images={images.shape} on {images.device}, "
                    f"labels={labels.tolist()}"
                )
                if i >= 1:
                    break

        # Batch read multiple files at once (useful for inference pipelines)
        batch = torchfits.read_batch(dataset.files[:4], hdu=0)
        print(f"\nread_batch: {len(batch)} images, first shape={batch[0].shape}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
