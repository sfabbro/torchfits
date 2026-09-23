"""DataView integer-index boundary contract (A-08).

``data[i]`` keeps its deliberate block semantics (an integer index becomes a
length-1 slice: ``data[0]`` has shape ``(1, n)`` where numpy would give
``(n,)``), but the index bounds follow sequence semantics exactly: at or
beyond ±len raises ``IndexError`` and negative indices wrap.
"""

import numpy as np
import pytest
import torch
from astropy.io import fits as afits

import torchfits


@pytest.fixture()
def dv(tmp_path):
    path = tmp_path / "dvimg.fits"
    afits.PrimaryHDU(np.arange(20, dtype="<f4").reshape(4, 5)).writeto(str(path))
    with torchfits.open(str(path)) as hdul:
        yield hdul[0].data


def test_dataview_int_index_out_of_range_raises(dv):
    rows, cols = dv.shape
    # exactly like sequence semantics: d[len(d)] and d[-len(d)-1] raise
    for bad in (rows, rows + 1, -rows - 1, -rows - 100):
        with pytest.raises(IndexError):
            dv[bad]
    for bad in (cols, cols + 1, -cols - 1, -cols - 100):
        with pytest.raises(IndexError):
            dv[0, bad]


def test_dataview_int_index_wraps_negative(dv):
    rows, cols = dv.shape
    assert torch.equal(dv[-rows], dv[0])
    assert torch.equal(dv[-1], dv[rows - 1])
    assert torch.equal(dv[0, -cols], dv[0, 0])
    assert torch.equal(dv[0, -1], dv[0, cols - 1])
    # block semantics are kept for valid integer indices
    assert tuple(dv[-1].shape) == (1, cols)


def test_dataview_slice_still_clamps(dv):
    """Slices clamp (not raise) out-of-range bounds — slice semantics."""
    rows, cols = dv.shape
    assert tuple(dv[rows : rows + 5, :].shape) == (0, cols)
    assert tuple(dv[-100:2, :].shape) == (2, cols)
