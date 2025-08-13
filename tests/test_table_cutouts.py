import os
import tempfile

import pytest
import torch

import torchfits
from torchfits.dataset import (
    BatchReadSpec,
    TableCutoutSpec,
    read_batch,
    read_multi_table_cutouts,
    read_table_cutout,
)


def _make_table(path):
    data = {
        "A": torch.arange(100, dtype=torch.int32),
        "B": torch.arange(100, dtype=torch.float32) * 0.5,
        "C": torch.arange(100, dtype=torch.float32) ** 2,
    }
    torchfits.write(path, data, overwrite=True)


def test_single_table_cutout_basic():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tab.fits")
        _make_table(path)
        sub = read_table_cutout(
            path, hdu=1, row_start=10, row_count=5, columns=["A", "C"]
        )
        assert isinstance(sub, dict)
        assert set(sub.keys()) == {"A", "C"}
        assert sub["A"].shape[0] == 5
        assert sub["A"][0].item() == 10


def test_multi_table_cutouts_parallel_and_seq():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tab.fits")
        _make_table(path)
        specs = [
            TableCutoutSpec(path=path, row_start=0, row_count=10, columns=["A"]),
            TableCutoutSpec(path=path, row_start=10, row_count=10, columns=["B"]),
            TableCutoutSpec(path=path, row_start=20, row_count=5, columns=["C"]),
        ]
        par = read_multi_table_cutouts(specs, parallel=True)
        seq = read_multi_table_cutouts(specs, parallel=False)
        assert len(par) == len(seq) == 3
        assert par[0]["A"].shape[0] == 10
        assert seq[2]["C"].shape[0] == 5


def test_batch_read_with_row_aliases():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tab.fits")
        _make_table(path)
        specs = [
            BatchReadSpec(path=path, hdu=1, columns=["A"], row_start=5, row_count=3),
            BatchReadSpec(path=path, hdu=1, columns=["B"], row_start=10, row_count=2),
        ]
        out = read_batch(specs, parallel=False)
        assert len(out) == 2
        assert out[0]["A"][0].item() == 5
    # B column is defined as torch.arange * 0.5, so row_start=10 => 10 * 0.5 == 5
    assert out[1]["B"][0].item() == 5


def test_table_column_slicing_aliases():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tab.fits")
        _make_table(path)
        # Use column slicing instead of explicit names: columns A,B (indices 0,1)
        spec = TableCutoutSpec(
            path=path, col_start=0, col_count=2, row_start=0, row_count=3
        )
        out = read_multi_table_cutouts([spec], parallel=False)[0]
        assert set(out.keys()) == {"A", "B"}
        # BatchReadSpec alias path
        bspec = BatchReadSpec(
            path=path, hdu=1, col_start=1, col_count=1, row_start=5, row_count=2
        )
        bout = read_batch([bspec], parallel=False)[0]
        assert list(bout.keys()) == ["B"]
        assert bout["B"].shape[0] == 2
