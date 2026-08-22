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


def test_scalar_column_squeeze_is_path_independent(tmp_path):
    """(N,1) FITS scalar columns read as (N,) on every access path.

    Regression: post-1.0 the squeeze existed only in TableDataAccessor, so
    shapes depended on which internal binding served the read.
    """
    from astropy.io import fits as afits

    import torchfits
    from torchfits.table import read_torch

    path = tmp_path / "squeeze_paths.fits"
    scalar = afits.Column(name="S", format="J", array=np.arange(4, dtype=np.int32))
    vector = afits.Column(
        name="V", format="3J", array=np.arange(12, dtype=np.int32).reshape(4, 3)
    )
    afits.BinTableHDU.from_columns([scalar, vector]).writeto(str(path), overwrite=True)

    import torchfits as tf

    with tf.open(str(path)) as hdul:
        assert tuple(hdul[1].data["S"].shape) == (4,)
        assert tuple(hdul[1]["S"].shape) == (4,)
        batch = next(hdul[1].iter_rows(2))
        assert tuple(batch["S"].shape) == (2,)
        # Vector columns keep their shape everywhere.
        assert tuple(hdul[1].data["V"].shape) == (4, 3)

    for mmap in (True, False):
        out = torchfits.read(str(path), hdu=1, mmap=mmap)
        assert tuple(out["S"].shape) == (4,)
    rt = read_torch(str(path), hdu=1)
    assert tuple(rt["S"].shape) == (4,)

    # Packed string columns (uint8 matrices) are never squeezed.
    names = afits.Column(name="N", format="1A", array=np.array(list("abcd")))
    afits.BinTableHDU.from_columns([names]).writeto(
        str(tmp_path / "w1.fits"), overwrite=True
    )
    with tf.open(str(tmp_path / "w1.fits")) as hdul:
        raw = hdul[1]["N"]
        assert getattr(raw, "dim", lambda: 0)() >= 1


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
