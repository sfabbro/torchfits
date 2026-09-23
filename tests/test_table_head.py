"""TableHDU / TableHDURef.head() composes with existing row slices."""

import numpy as np
import pytest
import torch
from torchfits.hdu import Header, TableHDU, TableHDURef


def test_tablehduref_head():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "x"
    header["TFORM1"] = "1D"
    header["NAXIS2"] = 10

    # We want to test composing head() calls
    ref = TableHDURef(
        header=header, source_path="dummy.fits", source_hdu=1, row_slice=slice(5, 10)
    )
    # the existing slice is [5, 10). Length = 5.

    # If we do head(2), it should give a slice of length 2 within [5, 10), so [5, 7).
    ref2 = ref.head(2)
    assert ref2._row_slice == slice(5, 7), f"Got {ref2._row_slice}"

    # Successive head() narrows within the current window, never replaces it.
    ref3 = ref2.head(3)
    assert ref3._row_slice == slice(5, 7), f"Got {ref3._row_slice}"
    ref4 = ref.head(4).head(2)
    assert ref4._row_slice == slice(5, 7), f"Got {ref4._row_slice}"


def test_tablehduref_head_negative():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "x"
    header["TFORM1"] = "1D"
    header["NAXIS2"] = 10

    ref = TableHDURef(header=header, source_path="dummy.fits", source_hdu=1)
    # If we do head(-2), it should give a slice of length 8 [0, 8).
    ref2 = ref.head(-2)
    assert ref2._row_slice == slice(0, 8), f"Got {ref2._row_slice}"

    # Negative head composes within the current window as well: [0, 4) then
    # all but the last row of that window -> [0, 3); with an offset window
    # [5, 10): [5, 9) then -> [5, 8).
    ref3 = ref.head(4).head(-1)
    assert ref3._row_slice == slice(0, 3), f"Got {ref3._row_slice}"
    windowed = TableHDURef(
        header=header, source_path="dummy.fits", source_hdu=1, row_slice=slice(5, 10)
    )
    ref4 = windowed.head(4).head(-1)
    assert ref4._row_slice == slice(5, 8), f"Got {ref4._row_slice}"


def test_tablehdu_head_negative():
    """In-memory TableHDU.head(-n) is a contract error (n must be >= 0)."""
    data = {"x": torch.zeros(10)}
    hdu = TableHDU(data)

    with pytest.raises(ValueError):
        hdu.head(-2)


def test_tablehdu_head_numpy():
    data = {"x": np.zeros(10)}
    hdu = TableHDU(data)

    hdu2 = hdu.head(3)
    assert hdu2["x"].shape[0] == 3

    with pytest.raises(ValueError):
        hdu.head(-2)


def test_tablehdu_head_composes():
    """Successive head() calls narrow monotonically on the materialized table."""
    hdu = TableHDU({"x": torch.arange(10.0)})
    narrowed = hdu.head(6).head(2)
    assert narrowed["x"].shape[0] == 2
    assert narrowed["x"].tolist() == [0.0, 1.0]
