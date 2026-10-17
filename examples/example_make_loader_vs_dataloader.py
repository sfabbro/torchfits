"""Example: ``make_loader`` vs a plain ``torch.utils.data.DataLoader``.

Both wrap the same ``FitsImageDataset``; ``make_loader`` only adds cache
warm-up (via ``optimize_cache``, which requires the dataset to expose a
``.files`` attribute) and sane defaults on top of ``DataLoader``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from torch.utils.data import DataLoader

import torchfits

from torchfits.data import FitsImageDataset, fits_collate_fn, make_loader


def _write_tiny_files(data_dir: Path, n: int = 4) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        data = rng.random((16, 16)).astype(np.float32)
        torchfits.write(data_dir / f"tiny_{i:02d}.fits", data, overwrite=True)


def main() -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_loader_cmp_"))
    try:
        _write_tiny_files(data_dir)
        dataset = FitsImageDataset(str(data_dir / "*.fits"), hdu=0)
        print(f"dataset: n={len(dataset)} has_files_attr={hasattr(dataset, 'files')}")

        plain = DataLoader(dataset, batch_size=2, collate_fn=fits_collate_fn)
        images, labels = next(iter(plain))
        print(
            f"plain DataLoader: batch images={tuple(images.shape)} labels={labels.tolist()}"
        )

        loader = make_loader(dataset, batch_size=2, optimize_cache=True)
        images, labels = next(iter(loader))
        print(
            f"make_loader: batch images={tuple(images.shape)} labels={labels.tolist()}"
        )
        print(
            "note: optimize_cache=True calls cache.optimize_for_dataset(dataset.files); "
            "datasets without a `.files` list (e.g. custom IterableDataset) should pass "
            "optimize_cache=False."
        )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
