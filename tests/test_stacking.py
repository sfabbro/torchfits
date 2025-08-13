import os
import tempfile

import pytest
import torch

import torchfits as tf
from torchfits.dataset import (
    BatchReadSpec,
    FITSCutoutSpec,
    FITSMultiCutoutSpec,
    TableCutoutSpec,
    read_batch,
    read_multi_cutouts,
    read_multi_table_cutouts,
)


def _make_image_file(path, shape=(32, 32), offset=0.0):
    t = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(*shape) + offset
    tf.write(path, t, overwrite=True)


def _make_mef(tmpdir: str, shapes=((32, 32), (32, 32))):
    path = os.path.join(tmpdir, "mef_stack.fits")
    tensors = [
        torch.arange(s[0] * s[1], dtype=torch.float32).reshape(*s) + i
        for i, s in enumerate(shapes)
    ]
    tf.write_mef(path, tensors, overwrite=True)
    return path, tensors


def _make_table(path):
    data = {
        "A": torch.arange(50, dtype=torch.int32),
        "B": torch.arange(50, dtype=torch.float32) * 0.5,
    }
    tf.write(path, data, overwrite=True)


def test_multi_cutouts_stack_tensor():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _make_mef(td, shapes=((32, 32), (32, 32)))
        specs = [
            FITSCutoutSpec(hdu=1, start=(0, 0), shape=(8, 8)),
            FITSCutoutSpec(hdu=2, start=(4, 4), shape=(8, 8)),
            FITSCutoutSpec(hdu=1, start=(8, 8), shape=(8, 8)),
        ]
        multi = FITSMultiCutoutSpec(
            path=path, cutouts=specs, parallel=True, return_dict=False
        )
        stacked = read_multi_cutouts(multi, stack=True)
        assert torch.is_tensor(stacked)
        assert stacked.shape == torch.Size([len(specs), 8, 8])


def test_read_batch_stack_images():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            p = os.path.join(td, f"im{i}.fits")
            _make_image_file(p, shape=(16, 16), offset=float(i))
            paths.append(p)
        specs = [
            BatchReadSpec(path=p, hdu=0, start=(0, 0), shape=(16, 16)) for p in paths
        ]
        out = read_batch(specs, parallel=True, stack=True)
        assert torch.is_tensor(out)
        assert out.shape == torch.Size([len(specs), 16, 16])
        # First plane baseline check
        assert torch.isclose(out[0, 0, 0], torch.tensor(0.0))


def test_read_multi_table_cutouts_stack():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tab.fits")
        _make_table(path)
        specs = [
            TableCutoutSpec(path=path, row_start=0, row_count=10, columns=["A", "B"]),
            TableCutoutSpec(path=path, row_start=10, row_count=10, columns=["A", "B"]),
            TableCutoutSpec(path=path, row_start=20, row_count=10, columns=["A", "B"]),
        ]
        stacked = read_multi_table_cutouts(specs, parallel=True, stack=True)
        # Expect dict of stacked tensors per column
        assert isinstance(stacked, dict)
        assert set(stacked.keys()) == {"A", "B"}
        assert stacked["A"].shape == torch.Size([len(specs), 10])
        assert stacked["B"].shape == torch.Size([len(specs), 10])


def test_read_batch_stack_images_preserve_headers():
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            p = os.path.join(td, f"im{i}.fits")
            _make_image_file(p, shape=(8, 8), offset=float(i))
            paths.append(p)
        specs = [
            BatchReadSpec(path=p, hdu=0, start=(0, 0), shape=(8, 8)) for p in paths
        ]
        stacked, headers = read_batch(
            specs, parallel=False, stack=True, preserve_headers_on_stack=True
        )
        assert torch.is_tensor(stacked)
        assert stacked.shape == torch.Size([len(specs), 8, 8])
        assert isinstance(headers, list) and len(headers) == len(specs)
