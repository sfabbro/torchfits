"""Scalar (N,1) columns squeeze to (N,); TableHDURef columns follow header edits."""

import numpy as np
import pytest
import torch
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


def _write_two_col_table(tmp_path):
    from astropy.io import fits as afits

    path = tmp_path / "refcols.fits"
    afits.BinTableHDU.from_columns(
        [
            afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4")),
            afits.Column(name="B", format="J", array=np.array([3, 4], dtype="<i4")),
        ]
    ).writeto(str(path))
    return path


def test_tablehduref_to_arrow_columns_kwarg(tmp_path):
    """to_arrow(columns=...) duplicates the forwarded projection kwarg (r5c-09).

    Contract: to_arrow/scan_arrow/reader_arrow forward hdu/columns/row_slice
    from the TableHDURef projection and REJECT duplicates in kwargs with a
    clear TypeError naming the duplicated argument. Narrow the projection
    with select()/head() instead.
    """
    import torchfits

    path = _write_two_col_table(tmp_path)

    with torchfits.open(str(path)) as hdul:
        with pytest.raises(TypeError, match=r"duplicate argument 'columns'"):
            hdul[1].to_arrow(columns=["A"])
        # The supported way to project columns:
        out = hdul[1].select(["A"]).to_arrow()
    assert out.column_names == ["A"]


def test_tablehduref_scan_and_reader_arrow_reject_forwarded_kwargs(tmp_path):
    """scan_arrow/reader_arrow reject hdu/columns/row_slice duplicates (r5c-09)."""
    import torchfits

    path = _write_two_col_table(tmp_path)

    with torchfits.open(str(path)) as hdul:
        with pytest.raises(TypeError, match=r"duplicate argument 'hdu'"):
            hdul[1].scan_arrow(hdu=1)
        with pytest.raises(TypeError, match=r"duplicate argument 'row_slice'"):
            hdul[1].reader_arrow(row_slice=slice(0, 1))
        with pytest.raises(TypeError, match=r"duplicate argument 'columns'"):
            hdul[1].scan_arrow(columns=["A"])
        # Non-forwarded keywords still pass through untouched.
        batches = list(hdul[1].scan_arrow(batch_size=1))
    assert len(batches) == 2


def test_tablehduref_columns_follow_replaced_header_on_id_reuse():
    """columns must track the current header even when a fresh Header lands on
    a freed header's id() with the same mutation counter (r6b cache aliasing)."""

    def make(name):
        header = Header()
        header["TFIELDS"] = 1
        header["TTYPE1"] = name
        header["TFORM1"] = "1D"
        header["NAXIS2"] = 10
        return TableHDURef(header=header, source_path="dummy.fits", source_hdu=1)

    ref = make("x")
    assert ref.columns == ["x"]
    first_id = id(ref.header)
    victim = ref.header
    ref.header = Header()
    del victim  # free the cached header so a fresh one can reuse its id()
    cand = None
    for _ in range(200):
        cand = make("y").header
        ref.header = cand
        if id(cand) == first_id:
            break
    assert ref.columns == ["y"]


def test_tablehduref_refresh_preserves_long_string_values(tmp_path):
    """A >68-char string keyword in a table HDU header must survive the
    _refresh_file_view re-read whole (r4b-13 chain reassembly at the raw
    cpp.read_header triples site)."""
    import torchfits
    from astropy.io import fits as afits

    path = tmp_path / "longstr_ref.fits"
    cols = [afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4"))]
    table = afits.BinTableHDU.from_columns(cols)
    long_value = "x" * 80
    table.header["LONGKEY"] = long_value
    table.writeto(str(path))

    with torchfits.open(str(path)) as hdul:
        assert hdul[1].header["LONGKEY"] == long_value
        refreshed = hdul[1].append_rows_file({"A": [3]})
    assert refreshed.header["LONGKEY"] == long_value


def test_tablehduref_read_honors_row_window(tmp_path):
    """ref.read() routes the row window through the shared _normalize_row_slice
    (r5c-15 dedupe) and still returns exactly the window's rows."""
    import torchfits

    path = _write_two_col_table(tmp_path)
    with torchfits.open(str(path)) as hdul:
        first = hdul[1].head(1).read()
        second = hdul[1].read(row_slice=(1, 2))
    assert first["A"].tolist() == [1]
    assert first["B"].tolist() == [3]
    assert second["A"].tolist() == [2]


def test_tablehduref_read_rejects_negative_stop(tmp_path):
    """row_slice with a negative stop raises the shared typed error naming the
    cause, instead of the late generic num_rows error from the old local copy
    (r5c-15 dedupe delta)."""
    import torchfits

    path = _write_two_col_table(tmp_path)
    with torchfits.open(str(path)) as hdul:
        with pytest.raises(ValueError, match="negative stop"):
            hdul[1].read(row_slice=slice(0, -1))
