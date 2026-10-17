"""Example: Compose a reversible FITS image transform pipeline.

Use transforms for viz / model preprocess. For Dataset + transform wiring, see
``example_image_dataset.py``. For before/after figures, see ``gallery_images.py``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import torchfits
from torchfits.transforms import (
    ArcsinhStretch,
    BackgroundSubtract,
    Compose,
    ZScaleNormalize,
)


def _create_test_file(path: str) -> None:
    rng = np.random.default_rng(42)
    y, x = np.mgrid[-128:128, -128:128]
    disk = 5000 * np.exp(-np.sqrt(x**2 + y**2) / 50)
    bulge = 20000 * np.exp(-(x**2 + y**2) / (2 * 20**2))
    background = rng.normal(100, 5, (256, 256))
    data = (disk + bulge + background).astype(np.float32)
    torchfits.write(path, data, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)
        image = torchfits.read_tensor(path, hdu=0)
        print(
            f"raw: shape={image.shape} min={image.min().item():.4g} max={image.max().item():.4g}"
        )

        stretch = ArcsinhStretch(a=0.1)
        stretched = stretch(image)
        print(
            f"ArcsinhStretch: max_err={((image - stretch.inverse(stretched)).abs().max().item()):.2e}"
        )

        pipeline = Compose(
            [
                BackgroundSubtract(),
                ArcsinhStretch(a=0.1),
                ZScaleNormalize(contrast=0.3),
            ]
        )
        preprocessed = pipeline(image)
        decoded = pipeline.inverse(preprocessed)
        print(
            f"Compose forward min/max={preprocessed.min().item():.4f}/"
            f"{preprocessed.max().item():.4f}; "
            f"inverse max_err={(image - decoded).abs().max().item():.2e}"
        )
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
