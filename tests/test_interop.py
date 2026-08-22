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
