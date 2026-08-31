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


def test_tablehdu_head_negative():
    data = {"x": torch.zeros(10)}
    hdu = TableHDU(data)

    # Negative head should truncate from the tail like pandas.
    hdu2 = hdu.head(-2)
    assert hdu2["x"].shape[0] == 8, f"Got {hdu2['x'].shape[0]}"


def test_tablehdu_head_numpy():
    import numpy as np

    data = {"x": np.zeros(10)}
    hdu = TableHDU(data)

    hdu2 = hdu.head(3)
    assert hdu2["x"].shape[0] == 3

    hdu3 = hdu.head(-2)
    assert hdu3["x"].shape[0] == 8
