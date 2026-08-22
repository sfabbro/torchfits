"""Truncated FITS tables must raise clean errors instead of SIGBUS."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits
import torchfits.table as ttable


@pytest.fixture()
def truncated_table(tmp_path):
    path = tmp_path / "trunc.fits"
    col = afits.Column(name="A", format="J", array=np.arange(50000, dtype="<i4"))
    afits.BinTableHDU.from_columns([col]).writeto(str(path), overwrite=True)
    raw = path.read_bytes()
    # Drop ~15k rows of tail data: header still claims 50000 rows.
    path.write_bytes(raw[: len(raw) - 60000])
    return str(path)


def test_mmap_read_raises_clean_error(truncated_table):
    with pytest.raises(RuntimeError, match="truncat"):
        torchfits.read(truncated_table, hdu=1, mmap=True)


def test_filtered_read_raises_clean_error(truncated_table):
    with pytest.raises(RuntimeError, match="truncat"):
        ttable.read(truncated_table, hdu=1, where="A > 3")


def test_mmap_update_tail_rows_raises_clean_error(truncated_table):
    payload = {"A": np.zeros(5, dtype=np.int32)}
    with pytest.raises(RuntimeError, match="truncat"):
        ttable.update_rows(
            truncated_table, payload, row_slice=(49995, 50000), mmap=True
        )


def test_intact_file_unaffected(tmp_path):
    path = tmp_path / "ok.fits"
    col = afits.Column(name="A", format="J", array=np.arange(1000, dtype="<i4"))
    afits.BinTableHDU.from_columns([col]).writeto(str(path), overwrite=True)
    out = torchfits.read(str(path), hdu=1, mmap=True)
    assert out["A"].shape[0] == 1000
