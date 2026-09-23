"""Batch C++ readers must attribute every failure to its path (r4a-01 C++ half).

``read_images_batch``/``read_hdus_batch`` error vectors must propagate as
``RuntimeError`` naming the offending path (and HDU), never return shrunken or
undefined tensors. The Python pipeline adds len-checks on top (r4a-01); this
suite pins the C++ contract directly, reachable without the Python guard.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
import torchfits._cpp as cpp


@pytest.fixture()
def batch_trio(tmp_path):
    good1 = tmp_path / "good1.fits"
    good2 = tmp_path / "good2.fits"
    corrupt = tmp_path / "corrupt_mid.fits"
    torchfits.write(str(good1), torch.ones(4, 4), overwrite=True)
    torchfits.write(str(good2), torch.full((4, 4), 2.0), overwrite=True)
    corrupt.write_bytes(b"NOTAFITSFILE" + b"\x00" * 2880)
    return [str(good1), str(corrupt), str(good2)]


def test_read_images_batch_corrupt_in_three_raises_naming_path(batch_trio):
    with pytest.raises(RuntimeError) as excinfo:
        cpp.read_images_batch(batch_trio, 0, True)
    assert "corrupt_mid.fits" in str(excinfo.value)


def test_read_images_batch_missing_in_three_raises_naming_path(tmp_path, batch_trio):
    missing = str(tmp_path / "gone_missing.fits")
    paths = [batch_trio[0], missing, batch_trio[2]]
    with pytest.raises(RuntimeError) as excinfo:
        cpp.read_images_batch(paths, 0, True)
    assert "gone_missing.fits" in str(excinfo.value)


def test_read_images_batch_all_good_returns_all(batch_trio):
    out = cpp.read_images_batch([batch_trio[0], batch_trio[2]], 0, True)
    assert len(out) == 2
    assert out[0].shape == (4, 4) and out[1].shape == (4, 4)


def test_read_images_batch_empty_returns_empty():
    assert cpp.read_images_batch([], 0, True) == []


def test_read_hdus_batch_corrupt_path_raises_naming_path(batch_trio):
    with pytest.raises(RuntimeError) as excinfo:
        cpp.read_hdus_batch(batch_trio[1], [0], True)
    assert "corrupt_mid.fits" in str(excinfo.value)


def test_read_hdus_batch_bad_hdu_raises_naming_path_and_hdu(batch_trio):
    """A per-HDU failure must carry the path and HDU index in the exception
    text (error contracts: text INSIDE the exception)."""
    with pytest.raises(RuntimeError) as excinfo:
        cpp.read_hdus_batch(batch_trio[0], [0, 9], True)
    msg = str(excinfo.value)
    assert "good1.fits" in msg
    assert "9" in msg


def test_read_hdus_batch_all_good_returns_all(batch_trio):
    with torchfits.open(batch_trio[0]) as hdul:
        n = len(hdul)
    out = cpp.read_hdus_batch(batch_trio[0], list(range(n)), True)
    assert len(out) == n


def test_read_hdus_sequence_last_empty_raises(batch_trio):
    with pytest.raises(RuntimeError):
        cpp.read_hdus_sequence_last(batch_trio[0], [], True)
