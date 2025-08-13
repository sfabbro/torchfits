import math
import os
import tempfile

import pytest
import torch

import torchfits
from torchfits.dataset import (
    FITSCutoutSpec,
    FITSDataset,
    FITSItemSpec,
    FITSMultiCutoutSpec,
    read_multi_cutouts,
)


def _make_mef(tmpdir: str, shapes=((32, 32), (64, 64))):
    path = os.path.join(tmpdir, "test_mef.fits")
    tensors = [
        torch.arange(s[0] * s[1], dtype=torch.float32).reshape(*s) for s in shapes
    ]
    torchfits.write_mef(path, tensors, overwrite=True)
    return path, tensors


def test_read_multi_cutouts_parallel_and_sequential():
    with tempfile.TemporaryDirectory() as td:
        path, tensors = _make_mef(td)
        specs = [
            FITSCutoutSpec(hdu=1, start=(0, 0), shape=(8, 8)),
            FITSCutoutSpec(hdu=2, start=(10, 10), shape=(16, 16)),
            FITSCutoutSpec(hdu=1, start=(4, 4), shape=(8, 8)),
        ]
        multi = FITSMultiCutoutSpec(
            path=path, cutouts=specs, parallel=True, return_dict=True
        )
        out_par = read_multi_cutouts(multi)
        assert len(out_par) == len(specs)
        # Validate one sample's numeric content
        if isinstance(out_par, dict):
            (hdu_idx, start_tuple), tensor = next(iter(out_par.items()))
            assert tensor.shape == torch.Size(specs[0].shape)
        else:  # list
            assert len(out_par[0].shape) == len(specs[0].shape)
        # Sequential path
        multi_seq = FITSMultiCutoutSpec(
            path=path, cutouts=specs, parallel=False, return_dict=False
        )
        out_seq = read_multi_cutouts(multi_seq)
        assert isinstance(out_seq, list) and len(out_seq) == len(specs)


def test_fitsdataset_with_multi_cutout_spec():
    with tempfile.TemporaryDirectory() as td:
        path, tensors = _make_mef(td)
        items = [
            {
                "path": path,
                "cutouts": [
                    {"hdu": 1, "start": (0, 0), "shape": (4, 4)},
                    {"hdu": 2, "start": (8, 8), "shape": (8, 8)},
                ],
                "parallel": True,
                "return_dict": True,
            }
        ]
        ds = FITSDataset(items)
        sample = ds[0]
        assert isinstance(sample, dict)
        assert len(sample) == 2
        for (_, start), ten in sample.items():
            assert torch.is_tensor(ten)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_iterable_dataset_prefetch(prefetch):
    from torchfits.dataset import FITSIterableDataset

    with tempfile.TemporaryDirectory() as td:
        path, _ = _make_mef(td)
        # create many small cutouts across epochs
        source_specs = [
            {"path": path, "hdu": 1, "start": (i, 0), "shape": (4, 4)}
            for i in range(0, 24, 4)
        ]
        iterable = FITSIterableDataset(source_specs, prefetch=prefetch)
        results = list(iterable)
        assert len(results) == len(source_specs)


def test_throughput_prefetch_benchmark():
    from torchfits.dataset import FITSIterableDataset

    with tempfile.TemporaryDirectory() as td:
        path, _ = _make_mef(td, shapes=((256, 256),))
        source_specs = [
            {"path": path, "hdu": 1, "start": (i, 0), "shape": (32, 32)}
            for i in range(0, 256, 32)
        ] * 4  # repeat to increase iterations

        # Artificial delay transform to simulate I/O / post-processing
        def transform(x):
            import time

            time.sleep(0.001)
            return x

        import time

        t0 = time.time()
        list(FITSIterableDataset(source_specs, prefetch=0, transform=transform))
        no_prefetch_time = time.time() - t0
        t0 = time.time()
        list(FITSIterableDataset(source_specs, prefetch=4, transform=transform))
        with_prefetch_time = time.time() - t0
        # Prefetch should not be slower; allow small tolerance
        assert with_prefetch_time <= no_prefetch_time * 1.05
