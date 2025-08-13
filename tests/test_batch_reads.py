import os
import tempfile

import pytest
import torch

import torchfits
from torchfits.dataset import BatchReadSpec, generate_random_cutout_specs, read_batch


def _make_files(td, count=3, shape=(64, 64)):
    paths = []
    for i in range(count):
        t = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(*shape) + i
        path = os.path.join(td, f"im{i}.fits")
        torchfits.write(path, t, overwrite=True)
        paths.append(path)
    return paths


def test_read_batch_sequential_and_parallel():
    with tempfile.TemporaryDirectory() as td:
        paths = _make_files(td, count=4)
        specs = [
            BatchReadSpec(path=p, hdu=0, start=(0, 0), shape=(16, 16)) for p in paths
        ]
        seq = read_batch(specs, parallel=False)
        par = read_batch(specs, parallel=True)
        assert len(seq) == len(par) == len(specs)
        assert torch.allclose(
            seq[0][0] if isinstance(seq[0], tuple) else seq[0],
            par[0][0] if isinstance(par[0], tuple) else par[0],
        )


def test_generate_random_cutout_specs():
    with tempfile.TemporaryDirectory() as td:
        paths = _make_files(td, count=2)
        specs = generate_random_cutout_specs(paths, shape=(8, 8), n=5)
        assert len(specs) == 5
        batch = read_batch(specs, parallel=True)
        assert len(batch) == 5
