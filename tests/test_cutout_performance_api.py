import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits


def test_read_subset_basic_roundtrip():
    data = (np.arange(256 * 256, dtype=np.float32).reshape(256, 256) * 0.5).astype(
        np.float32
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        name = f.name
    try:
        from astropy.io import fits

        fits.PrimaryHDU(data).writeto(name, overwrite=True)

        # 10x10 cutout at (x=5..14, y=7..16) using torchfits API coords.
        cut = torchfits.read_subset(name, 0, 5, 7, 15, 17)
        assert cut.shape == (10, 10)
        assert np.allclose(cut.numpy(), data[7:17, 5:15])
    finally:
        try:
            os.unlink(name)
        except Exception:
            pass
