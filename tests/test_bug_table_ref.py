import torch
import numpy as np
from torchfits.hdu import Header, TableHDU, TableHDURef

def test_table_data_accessor_auto_squeeze():
    header = Header()
    header["TFIELDS"] = 2
    header["TTYPE1"] = "x"
    header["TFORM1"] = "1D"
    header["TTYPE2"] = "y"
    header["TFORM2"] = "1D"

    tensor_col = torch.zeros((5, 1))
    numpy_col = np.ones((5, 1))

    hdu = TableHDU({"x": tensor_col, "y": numpy_col}, header=header)
    data = hdu.data

    # Verify auto-squeeze works for tensors and numpy arrays of shape (N, 1)
    assert data["x"].shape == (5,)
    assert data["y"].shape == (5,)

def test_tablehduref_cache_invalidation():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "OLD_NAME"
    header["TFORM1"] = "A10"

    ref = TableHDURef(header=header)
    assert ref.columns == ["OLD_NAME"]

    header["TTYPE1"] = "NEW_NAME"
    assert ref.columns == ["NEW_NAME"]


def test_tablehduref_cache_invalidation_on_del():
    header = Header()
    header["TFIELDS"] = 2
    header["TTYPE1"] = "x"
    header["TFORM1"] = "1D"
    header["TTYPE2"] = "y"
    header["TFORM2"] = "1D"

    ref = TableHDURef(header=header)
    assert ref.columns == ["x", "y"]

    del header["TTYPE2"]
    assert ref.columns == ["x", "COL2"]
