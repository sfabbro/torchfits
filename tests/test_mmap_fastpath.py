import os
from pathlib import Path

import pytest
import torch

import torchfits as tf


@pytest.mark.parametrize("example", ["examples/basic_example.fits"])
def test_mmap_parity(example: str):
    p = Path(example)
    if not p.exists():
        pytest.skip("example file missing")

    # Default read
    a, _ = tf.read(str(p), hdu=0)
    # MMAP read (opt-in)
    b, _ = tf.read(str(p), hdu=0, enable_mmap=True)

    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    # Exact bitwise equality for uncompressed example
    assert torch.equal(a, b)
