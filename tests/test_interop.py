import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
from astropy.table import Table

import torchfits


def test_to_arrow_numeric_tensor_shares_buffer():
    pytest.importorskip("pyarrow")

    source = torch.arange(8, dtype=torch.int64)
    view = source[2:6]
    arrow = torchfits.to_arrow({"value": view})

    source[3] = 99
    assert arrow["value"][1].as_py() == 99


def test_to_pandas_decode_bytes():
    pytest.importorskip("pandas")

    table = Table(
        {
            "RA": np.array([10.1, 10.2], dtype=np.float64),
            "NAME": np.array(["STAR_A", "STAR_B"], dtype="U8"),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        table.write(f.name, format="fits", overwrite=True)
        path = f.name

    try:
        data, _ = torchfits.read(path, hdu=1, return_header=True)
        df = torchfits.to_pandas(data, decode_bytes=True)

        assert df.shape[0] == 2
        assert df["NAME"].tolist() == ["STAR_A", "STAR_B"]
        assert np.allclose(df["RA"].to_numpy(), [10.1, 10.2])
    finally:
        os.unlink(path)


def test_to_arrow_vla_list():
    pytest.importorskip("pyarrow")

    vla = np.array([np.array([1, 2]), np.array([3])], dtype=object)
    table = Table(
        {
            "RA": np.array([10.1, 10.2], dtype=np.float64),
            "VLA": vla,
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        table.write(f.name, format="fits", overwrite=True)
        path = f.name

    try:
        data, _ = torchfits.read(path, hdu=1, return_header=True)
        arrow = torchfits.to_arrow(data, vla_policy="list")

        assert arrow.num_rows == 2
        assert "VLA" in arrow.column_names
    finally:
        os.unlink(path)


def test_to_arrow_vla_invalid_policy():
    pytest.importorskip("pyarrow")

    data = {
        "RA": torch.tensor([10.1, 10.2], dtype=torch.float64),
        "VLA": [torch.tensor([1, 2]), torch.tensor([3])],
    }
    with pytest.raises(ValueError, match="vla_policy must be 'list' or 'drop'"):
        torchfits.to_arrow(data, vla_policy="invalid_policy")


def test_to_pandas_missing_pandas():
    with mock.patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(
            ImportError, match="Pandas is required for to_pandas conversion."
        ):
            torchfits.to_pandas({"a": torch.tensor([1, 2, 3])})


def test_read_and_to_astropy():
    pytest.importorskip("astropy")
    pytest.importorskip("pyarrow")

    table = Table(
        {
            "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
            "DEC": np.array([-45.1, 0.0, 30.2], dtype=np.float64),
            "MAG_G": np.array([18.5, 19.2, 21.0], dtype=np.float32),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name

    try:
        torchfits.table.write(path, table, overwrite=True)

        # 1. Read directly as Astropy Table
        at_read = torchfits.table.read_astropy(path, hdu=1)
        assert isinstance(at_read, Table)
        assert len(at_read) == 3
        assert np.allclose(at_read["RA"], [10.1, 10.2, 10.3])
        assert np.allclose(at_read["DEC"], [-45.1, 0.0, 30.2])

        # 2. Filtered read as Astropy Table
        at_filtered = torchfits.table.read_astropy(path, hdu=1, where="MAG_G < 20.0")
        assert len(at_filtered) == 2
        assert np.allclose(at_filtered["MAG_G"], [18.5, 19.2])

        # 3. to_astropy on tensor dict
        tensor_dict = {
            "RA": torch.tensor([10.1, 10.2]),
            "DEC": torch.tensor([-45.1, 0.0]),
        }
        at_from_tensors = torchfits.to_astropy(tensor_dict)
        assert isinstance(at_from_tensors, Table)
        assert len(at_from_tensors) == 2
        assert np.allclose(at_from_tensors["RA"], [10.1, 10.2])
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_table_write_various_dataframe_types():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name

    try:
        # 1. Write PyArrow Table directly
        pa_table = pa.table({"X": [1.0, 2.0, 3.0], "Y": [10, 20, 30]})
        torchfits.table.write(path, pa_table, overwrite=True)
        res_pa = torchfits.table.read(path)
        assert res_pa.num_rows == 3
        assert res_pa.column_names == ["X", "Y"]

        # 2. Write Pandas DataFrame directly
        pd = pytest.importorskip("pandas")
        df_pd = pd.DataFrame({"A": [100, 200], "B": [1.5, 2.5]})
        torchfits.table.write(path, df_pd, overwrite=True)
        res_pd = torchfits.table.read(path)
        assert res_pd.num_rows == 2
        assert res_pd.column_names == ["A", "B"]

        # 3. Write Polars DataFrame directly
        pl = pytest.importorskip("polars")
        df_pl = pl.DataFrame({"C": ["alpha", "beta"], "D": [3.14, 2.71]})
        torchfits.table.write(path, df_pl, overwrite=True)
        res_pl = torchfits.table.read(path)
        assert res_pl.num_rows == 2
        assert res_pl.column_names == ["C", "D"]

        # 4. Root write() with Astropy Table
        at = Table({"FLUX": [10.0, 20.0, 30.0]})
        torchfits.write(path, at, overwrite=True)
        res_root = torchfits.table.read(path)
        assert res_root.num_rows == 3
        assert res_root.column_names == ["FLUX"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_table_to_astropy_fidelity(tmp_path):
    """Path input preserves TUNIT, TNULL masking, and TDIM shapes."""
    from astropy.io import fits as afits

    from torchfits.table import to_astropy

    path = tmp_path / "fidelity.fits"
    mag = afits.Column(
        name="MAG", format="E", array=np.array([1.0, 2.0], dtype="<f4"), unit="mag"
    )
    ident = afits.Column(
        name="ID", format="J", array=np.array([7, 8], dtype="<i4"), null=8
    )
    vec = afits.Column(
        name="VEC", format="3J", array=np.arange(6, dtype="<i4").reshape(2, 3)
    )
    afits.BinTableHDU.from_columns([mag, ident, vec]).writeto(str(path), overwrite=True)

    tbl = to_astropy(str(path))
    assert str(tbl["MAG"].unit) == "mag"
    assert hasattr(tbl["ID"], "mask") and tbl["ID"].mask.tolist() == [False, True]
    assert tbl["VEC"].shape == (2, 3)


def test_arrow_nulls_become_masked_column():
    import pyarrow as pa

    from torchfits.table import to_astropy

    tbl = to_astropy(pa.table({"a": pa.array([1, None, 3], type=pa.int32())}))
    assert tbl["a"].mask.tolist() == [False, True, False]


def test_root_to_astropy_dict_nulls_become_masked():
    """Dict input must not silently turn nulls into NaN / widen int to float."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")

    tbl = torchfits.to_astropy({"a": [1, None, 3], "f": [1.0, None, 3.0]})
    for name in ("a", "f"):
        col = tbl[name]
        assert getattr(col, "mask", None) is not None, name
        assert col.mask.tolist() == [False, True, False], name
    assert tbl["a"].dtype.kind in "iu"
    assert tbl["a"][0] == 1 and tbl["a"][2] == 3
    assert tbl["f"][0] == 1.0 and tbl["f"][2] == 3.0


def test_root_to_astropy_dict_vector_keeps_shape():
    """2-D tensor columns keep their (N, repeat) shape like table.to_astropy."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")

    tbl = torchfits.to_astropy({"v": torch.arange(6).reshape(2, 3)})
    assert tbl["v"].shape == (2, 3)
    assert np.asarray(tbl["v"]).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_root_to_astropy_accepts_pathlike(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    path = tmp_path / "pathlike.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4"))]
    ).writeto(str(path))

    tbl = torchfits.to_astropy(path)
    assert len(tbl) == 2


def _vector_tnull_fits(tmp_path):
    from astropy.io import fits as afits

    path = tmp_path / "vtnull.fits"
    vec = afits.Column(
        name="V",
        format="3J",
        array=np.array([[1, 8, 3], [8, 5, 6], [7, 8, 9], [1, 2, 8]], dtype="<i4"),
        null=8,
    )
    afits.BinTableHDU.from_columns([vec]).writeto(str(path), overwrite=True)
    return path


def test_to_astropy_vector_tnull_matches_astropy(tmp_path):
    """Vector TNULL columns surface MaskedColumn like astropy (r5c-01)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.table import Table as AstropyTable

    from torchfits.table import to_astropy

    path = _vector_tnull_fits(tmp_path)
    gt = AstropyTable.read(str(path))["V"]
    ours = to_astropy(str(path))["V"]

    assert hasattr(ours, "mask"), f"expected MaskedColumn, got {type(ours).__name__}"
    assert ours.dtype.kind == gt.dtype.kind
    assert ours.dtype.itemsize == gt.dtype.itemsize
    assert np.shape(ours) == np.shape(gt) == (4, 3)
    assert np.array_equal(np.asarray(ours.mask), np.asarray(gt.mask))
    m = np.asarray(gt.mask)
    assert np.array_equal(np.asarray(ours)[~m], np.asarray(gt)[~m])


def test_to_astropy_arrow_fsl_nulls_masked_not_object():
    """Arrow fixed-size-list nulls: typed MaskedColumn, not float64/NaN (r5c-01)."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")

    from torchfits.table import to_astropy

    arr = pa.array([[1, None], [None, 3]], type=pa.list_(pa.int32(), 2))
    col = to_astropy(pa.table({"w": arr}))["w"]
    assert hasattr(col, "mask"), f"expected MaskedColumn, got {type(col).__name__}"
    assert col.dtype.kind == "i", col.dtype
    assert col.mask.tolist() == [[False, True], [True, False]]
    assert np.asarray(col)[0][0] == 1 and np.asarray(col)[1][1] == 3


def test_table_to_astropy_accepts_pathlike(tmp_path):
    """torchfits.table.to_astropy(Path) must not raise TypeError (r5c-02, r1b-10)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    from torchfits.table import to_astropy

    path = tmp_path / "pathlike.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4"))]
    ).writeto(str(path))

    tbl = to_astropy(path)
    assert len(tbl) == 2


def test_interop_pathlike_data_inputs(tmp_path):
    """PathLike data reaches read() via os.fspath in every interop entry (r5c-02)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    from astropy.io import fits as afits

    from torchfits.table import to_pandas, write_csv

    path = tmp_path / "pathlike2.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4"))]
    ).writeto(str(path))

    df = to_pandas(path)
    assert df["A"].tolist() == [1, 2]

    out = tmp_path / "out.csv"
    write_csv(str(out), path)
    assert out.exists()
    assert out.read_text().strip().splitlines()[0].replace('"', "") == "A"


def test_to_astropy_rejects_unknown_kwargs(tmp_path):
    """Delegation contract: kwargs go to read(); unusable ones raise (r5c-04)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    import pyarrow as pap
    from astropy.io import fits as afits

    from torchfits.table import to_astropy

    path = tmp_path / "kw.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([1, 2], dtype="<i4"))]
    ).writeto(str(path))

    with pytest.raises(TypeError):
        to_astropy(str(path), bogus_kwarg=1)
    with pytest.raises(TypeError):
        to_astropy(pap.table({"A": [1, 2]}), hdu=1)


def test_to_astropy_empty_reader(tmp_path):
    """Zero-batch RecordBatchReader materializes to an empty Table (r5c-05)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    from torchfits.table import reader as treader, to_astropy

    path = tmp_path / "empty.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([], dtype="<i4"))]
    ).writeto(str(path))

    tbl = to_astropy(treader(str(path)))
    assert tbl.colnames == ["A"]
    assert len(tbl) == 0


def test_stream_write_empty_source_creates_file(tmp_path):
    """Streaming writes of empty sources still produce output files (r5c-06)."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from astropy.io import fits as afits

    from torchfits.table import write_csv, write_ipc, write_parquet

    path = tmp_path / "empty2.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([], dtype="<i4"))]
    ).writeto(str(path))

    csv_out = tmp_path / "o.csv"
    write_csv(str(csv_out), str(path), stream=True)
    assert csv_out.exists()
    assert csv_out.read_text().splitlines()[0].replace('"', "").strip() == "A"

    pq_out = tmp_path / "o.parquet"
    write_parquet(str(pq_out), str(path), stream=True)
    assert pq_out.exists()
    tbl = pq.read_table(str(pq_out))
    assert tbl.num_rows == 0 and tbl.column_names == ["A"]

    ipc_out = tmp_path / "o.arrow"
    write_ipc(str(ipc_out), str(path), stream=True)
    assert ipc_out.exists()


def test_to_pandas_empty_reader_keeps_columns(tmp_path):
    """Empty sources keep their schema columns in pandas output (r5c-06)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    from astropy.io import fits as afits

    from torchfits.table import reader as treader, to_pandas

    path = tmp_path / "empty3.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([], dtype="<i4"))]
    ).writeto(str(path))

    df = to_pandas(treader(str(path)))
    assert list(df.columns) == ["A"]


def test_to_astropy_empty_fixed_list_dtype():
    """Empty fixed-size-list results preserve the Arrow value dtype (r5c-07)."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")

    from torchfits.table import to_astropy

    tab = pa.table({"v": pa.chunked_array([], type=pa.list_(pa.float32(), 2))})
    col = to_astropy(tab)["v"]
    assert np.shape(col) == (0, 2)
    assert col.dtype == np.float32


def test_to_astropy_meta_ioerror_propagates(tmp_path, monkeypatch):
    """Header-meta extraction re-raises IO errors instead of dropping units (r5c-08)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    from torchfits.table import to_astropy

    path = tmp_path / "ioerr.fits"
    afits.BinTableHDU.from_columns(
        [afits.Column(name="A", format="J", array=np.array([1], dtype="<i4"))]
    ).writeto(str(path))

    def boom(*args, **kwargs):
        raise OSError("header read failed")

    monkeypatch.setattr(torchfits, "read_header", boom)
    with pytest.raises(OSError):
        to_astropy(str(path))


def test_to_arrow_keeps_tensor_storage_alive():
    """Arrow arrays keep source storage alive after the tensor is dropped (r5c-11)."""
    pytest.importorskip("pyarrow")
    import gc
    import weakref

    def build():
        t = torch.arange(64, dtype=torch.float32)
        return torchfits.to_arrow({"v": t}), weakref.ref(t.untyped_storage())

    arrow, ref = build()
    gc.collect()
    assert ref() is not None, "Arrow buffer dropped the source tensor storage"
    assert arrow["v"].to_pylist() == [float(x) for x in range(64)]
