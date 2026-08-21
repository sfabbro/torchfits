import pytest
import numpy as np
from astropy.io import fits


@pytest.fixture
def fits_file(tmp_path):
    path = str(tmp_path / "test_filter.fits")
    n_rows = 1000

    # Create data
    # FLOAT column 'MAG' : 0..100
    # INT column 'ID' : 0..1000
    # STRING column 'LABEL': 'A', 'B' alternating

    mag = np.linspace(0, 100, n_rows, dtype=np.float32)
    ids = np.arange(n_rows, dtype=np.int32)
    short_col = np.arange(n_rows, dtype=np.int16)

    c1 = fits.Column(name="MAG", format="E", array=mag)
    c2 = fits.Column(name="ID", format="J", array=ids)
    c3 = fits.Column(name="SHORT_VAL", format="I", array=short_col)

    # Add a string column if possible, but let's start with numeric

    hdu = fits.BinTableHDU.from_columns([c1, c2, c3])
    hdu.writeto(path)
    return path


def test_filter_lt(fits_file):
    import torchfits.cpp

    # MAG < 50.0. linspace(0,100,1000): i * 100/999 < 50 => i <= 499 (500 rows).
    filters = [("MAG", "<", 50.0)]
    cols = ["ID", "MAG"]

    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)

    assert "ID" in data
    assert "MAG" in data

    ids = data["ID"]
    mags = data["MAG"]

    assert len(ids) == 500
    assert (mags < 50.0).all()
    assert len(mags) == 500


def test_filter_gt(fits_file):
    import torchfits.cpp

    filters = [("ID", ">", 800)]
    cols = ["ID"]

    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)
    ids = data["ID"]

    # 801 to 999 -> 199 items
    assert len(ids) == 199
    assert (ids > 800).all()


def test_table_read_integration(fits_file):
    import torchfits

    # Test integration via torchfits.table.read(where=...)
    # MAG < 50.0 should use fast path

    t = torchfits.table.read(fits_file, where="MAG < 50.0")
    assert len(t) == 500
    mags = t["MAG"].to_numpy()
    assert (mags < 50.0).all()

    # OR falls back to slow path: MAG < 10 (100 rows) OR MAG > 90 (100 rows)
    t_slow = torchfits.table.read(fits_file, where="MAG < 10.0 OR MAG > 90.0")
    assert len(t_slow) == 200
    mags_slow = t_slow["MAG"].to_numpy()
    assert ((mags_slow < 10.0) | (mags_slow > 90.0)).all()


def test_table_read_where_torch_backend(fits_file):
    import torchfits

    # MAG > 10 AND MAG < 20: indices 100..199 -> 100 rows
    t = torchfits.table.read(
        fits_file, where="MAG > 10.0 AND MAG < 20.0", backend="torch"
    )
    mags = t["MAG"].to_numpy()
    assert len(t) == 100
    assert ((mags > 10.0) & (mags < 20.0)).all()


def test_read_torch_where_fused(fits_file):
    import torch

    import torchfits

    data = torchfits.table.read_torch(
        fits_file, columns=["ID", "MAG"], where="MAG > 50.0"
    )
    assert "MAG" in data and "ID" in data
    assert isinstance(data["MAG"], torch.Tensor)
    assert data["MAG"].numel() > 0
    assert bool((data["MAG"] > 50.0).all())
    # Parity with unfiltered + mask
    full = torchfits.table.read_torch(fits_file, columns=["MAG"])
    expected = full["MAG"][full["MAG"] > 50.0]
    assert data["MAG"].numel() == expected.numel()
    assert torch.allclose(data["MAG"].cpu(), expected.cpu())


def test_read_torch_where_zero_match_keeps_schema(fits_file):
    """Zero-match where= must return keyed empty tensors, not {}."""
    import torch

    import torchfits

    data = torchfits.table.read_torch(
        fits_file, columns=["ID", "MAG"], where="MAG > 1000.0"
    )
    assert set(data.keys()) == {"ID", "MAG"}
    assert isinstance(data["ID"], torch.Tensor)
    assert isinstance(data["MAG"], torch.Tensor)
    assert data["ID"].numel() == 0
    assert data["MAG"].numel() == 0
    assert data["ID"].shape[0] == 0
    assert data["MAG"].shape[0] == 0


def test_filter_eq(fits_file):
    import torchfits.cpp

    filters = [("ID", "==", 500)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    assert len(data["ID"]) == 1
    assert data["ID"][0].item() == 500


def test_filter_compound(fits_file):
    import torchfits.cpp

    # ID > 100 AND ID < 200
    filters = [("ID", ">", 100), ("ID", "<", 200)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    ids = data["ID"]

    assert len(ids) == 99  # 101 to 199
    assert ids.min() == 101
    assert ids.max() == 199


def test_where_preserves_tnull_nulls(tmp_path):
    """WHERE-filtered reads must still convert TNULL sentinels to Arrow null.

    Regression: the CPP-pushdown and torch-tensor-mask fast paths taken for
    simple WHERE predicates used to skip TNULL handling entirely, so a
    nullable column read back through where= leaked its raw sentinel value
    instead of Arrow null even with apply_fits_nulls=True (the default).
    """
    import torchfits.table as table

    path = str(tmp_path / "nulls.fits")
    n = 20
    ids = np.arange(n, dtype=np.int32)
    vals = np.arange(n, dtype=np.int32)
    vals[3] = -999
    vals[7] = -999
    c1 = fits.Column(name="ID", format="J", array=ids)
    c2 = fits.Column(name="VAL", format="J", array=vals, null=-999)
    fits.BinTableHDU.from_columns([c1, c2]).writeto(path)

    full = table.read(path, hdu=1, apply_fits_nulls=True)
    assert full.column("VAL").null_count == 2

    filtered = table.read(path, hdu=1, where="ID >= 0", apply_fits_nulls=True)
    assert filtered.column("VAL").null_count == 2
    assert filtered.column("VAL").to_pylist()[3] is None

    # Projected columns (WHERE references a column outside the projection)
    # go through the Arrow-filter fallback and must also preserve nulls.
    projected = table.read(
        path, hdu=1, columns=["VAL"], where="ID >= 0", apply_fits_nulls=True
    )
    assert projected.column("VAL").null_count == 2

    # Explicit opt-out must still be honored.
    disabled = table.read(path, hdu=1, where="ID >= 0", apply_fits_nulls=False)
    assert disabled.column("VAL").null_count == 0


def test_filter_short(fits_file):
    import torchfits.cpp

    # SHORT_VAL is int16. Filter on it.
    filters = [("SHORT_VAL", "==", 10)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["SHORT_VAL"], filters)
    assert len(data["SHORT_VAL"]) == 1
    assert data["SHORT_VAL"][0].item() == 10


def _write_unsigned_fits(path, phys, code, tzero):
    stored = (phys.astype(np.int64) - tzero).astype(
        np.int16 if code == "I" else np.int32
    )
    c = fits.Column(name="U", format=code, array=stored)
    hdu = fits.BinTableHDU.from_columns([c])
    hdu.header["TZERO1"] = tzero
    hdu.writeto(path)


def test_filter_unsigned_int16_pushdown(tmp_path):
    """WHERE on uint16 (TZERO=32768) columns must compare physical values.

    Regression: the C++ mmap pushdown compared the raw signed storage bytes
    and never applied the +32768 offset, so ``U > 40000`` returned wrong rows
    while the Arrow/torch fallback returned the correct result.
    """
    import torchfits.table as table

    path = str(tmp_path / "uint16.fits")
    _write_unsigned_fits(
        path, np.array([100, 200, 40000, 50000, 65535], dtype=np.uint16), "I", 32768
    )

    cases = {
        "U > 40000": [50000, 65535],
        "U >= 50000": [50000, 65535],
        "U < 200": [100],
        "U == 65535": [65535],
        "U != 40000": [100, 200, 50000, 65535],
    }
    for pred, want in cases.items():
        pushdown = sorted(
            table.read(path, hdu=1, where=pred, mmap=True)["U"].to_pylist()
        )
        fallback = sorted(
            table.read(path, hdu=1, where=pred, mmap=False)["U"].to_pylist()
        )
        assert pushdown == want, pred
        assert fallback == want, pred


def test_filter_unsigned_int32_pushdown(tmp_path):
    """WHERE on uint32 (TZERO=2**31) columns must compare physical values."""
    import torchfits.table as table

    path = str(tmp_path / "uint32.fits")
    _write_unsigned_fits(
        path,
        np.array([0, 1, 3000000000, 4000000000, 4294967295], dtype=np.uint32),
        "J",
        2147483648,
    )

    cases = {
        "U > 3000000000": [4000000000, 4294967295],
        "U == 4294967295": [4294967295],
        "U < 2": [0, 1],
        "U >= 4000000000": [4000000000, 4294967295],
    }
    for pred, want in cases.items():
        pushdown = sorted(
            table.read(path, hdu=1, where=pred, mmap=True)["U"].to_pylist()
        )
        fallback = sorted(
            table.read(path, hdu=1, where=pred, mmap=False)["U"].to_pylist()
        )
        assert pushdown == want, pred
        assert fallback == want, pred


def test_filter_literal_out_of_range_int16(tmp_path):
    """WHERE literals wider than the column storage must not be truncated.

    Regression: the C++ pushdown cast the literal to the column's storage type
    (``(int16_t)40000`` == ``-25536``), turning ``S > 40000`` into ``S > -25536``
    and returning nearly every row. The literal must be compared at full width.
    """
    import torchfits.table as table

    path = str(tmp_path / "int16.fits")
    sv = np.array([-30000, -100, 0, 100, 30000], dtype=np.int16)
    c = fits.Column(name="S", format="I", array=sv)
    fits.BinTableHDU.from_columns([c]).writeto(path)

    cases = {
        "S > 40000": [],
        "S < -40000": [],
        "S > -40000": [-30000, -100, 0, 100, 30000],
        "S == 40000": [],
        "S != 40000": [-30000, -100, 0, 100, 30000],
    }
    for pred, want in cases.items():
        pushdown = sorted(
            table.read(path, hdu=1, where=pred, mmap=True)["S"].to_pylist()
        )
        fallback = sorted(
            table.read(path, hdu=1, where=pred, mmap=False)["S"].to_pylist()
        )
        assert pushdown == want, pred
        assert fallback == want, pred
