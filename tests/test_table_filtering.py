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


# ---------------------------------------------------------------------------
# R5 review (r5a): row selection, empty-result schema, error contracts.
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_fits(tmp_path):
    """10-row numeric-only table (all columns scalar, no strings)."""
    path = str(tmp_path / "numeric.fits")
    ids = np.arange(10, dtype=np.int32)
    fits.BinTableHDU.from_columns(
        [fits.Column(name="ID", format="J", array=ids)]
    ).writeto(path)
    return path


def test_rows_out_of_range_raises(numeric_fits):
    """rows= must never silently fall back to the full table (r5a-01).

    Regression: when the scattered row read failed (e.g. an out-of-range
    index), the fallback ignored ``rows`` entirely and returned every row.
    """
    import torchfits.table as table

    with pytest.raises(IndexError, match="999"):
        table.read(numeric_fits, rows=[0, 999])
    with pytest.raises(IndexError, match="999"):
        table.read(numeric_fits, rows=[999])


def test_rows_honored_with_torch_backend(numeric_fits):
    """backend="torch" must select the same rows as the default engine (r5a-01)."""
    import torchfits.table as table

    t = table.read(numeric_fits, rows=[7, 2], backend="torch")
    assert t["ID"].to_pylist() == [7, 2]


def test_rows_honored_raw_decode_flags(numeric_fits):
    """The whole-column fast path must not swallow a rows= selection (r5a-01)."""
    import torchfits.table as table

    t = table.read(
        numeric_fits,
        rows=[7, 2],
        decode_bytes=False,
        apply_fits_nulls=False,
        include_fits_metadata=False,
    )
    assert t["ID"].to_pylist() == [7, 2]


def test_rows_with_where_out_of_range_raises(numeric_fits):
    """rows+where must filter within the requested rows only (r5a-01)."""
    import torchfits.table as table

    with pytest.raises(IndexError, match="999"):
        table.read(numeric_fits, rows=[0, 5, 999], where="ID > 2")


def test_rows_empty_list_preserves_schema(numeric_fits):
    import torchfits.table as table

    t = table.read(numeric_fits, rows=[])
    assert t.column_names == ["ID"]
    assert t.num_rows == 0
    assert str(t.schema.field("ID").type) == "int32"


def test_empty_result_dtype_matches_full_read(tmp_path):
    """Empty results and schema() must report the data dtypes (r5a-02):
    unsigned conventions stay integer (uint16), scaled columns read as
    float64, fixed-width strings read as strings — not list types."""
    import torchfits.table as table

    path = str(tmp_path / "dtypes.fits")
    stored = (np.array([100, 40000], dtype=np.int64) - 32768).astype(np.int16)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="U", format="I", array=stored),
            fits.Column(name="S", format="I", array=np.arange(2, dtype=np.int16)),
            fits.Column(name="STR", format="8A", array=np.array([b"ab", b"cd   e"])),
        ]
    )
    hdu.header["TZERO1"] = 32768
    hdu.header["TSCAL2"] = 0.5
    hdu.header["TZERO2"] = 100.0
    hdu.writeto(path)

    full = table.read(path)
    empty = table.read(path, row_slice=(0, 0))
    schema = table.schema(path)
    for name, want in [("U", "uint16"), ("S", "double"), ("STR", "string")]:
        got_full = str(full.schema.field(name).type)
        assert got_full == want, f"{name}: full read is {got_full}"
        assert str(empty.schema.field(name).type) == want, name
        assert str(schema.field(name).type) == want, name


def test_unknown_column_raises_key_error_everywhere(numeric_fits):
    """Unknown projected columns raise KeyError naming the column for empty
    and non-empty results alike (r5a-03); silent schema drops are gone."""
    import torchfits.table as table

    for kw in (
        {},
        {"row_slice": (0, 0)},
        {"where": "ID > 100"},
        {"where": "ID > 100", "mmap": False},
    ):
        with pytest.raises(KeyError, match="NOPE"):
            table.read(numeric_fits, columns=["ID", "NOPE"], **kw)
    with pytest.raises(KeyError, match="NOPE"):
        list(table.scan(numeric_fits, columns=["ID", "NOPE"]))
    with pytest.raises(KeyError, match="NOPE"):
        table.read_torch(numeric_fits, columns=["ID", "NOPE"])


def test_empty_result_preserves_unnamed_columns(tmp_path):
    """Auto-named (TTYPE-less) columns survive empty reads (r5a-04)."""
    import torchfits.table as table

    path = str(tmp_path / "unnamed.fits")
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.arange(4, dtype=np.int32)),
            fits.Column(
                name="VEC",
                format="5J",
                array=np.arange(20, dtype=np.int32).reshape(4, 5),
            ),
        ]
    )
    hdu.writeto(path)
    with fits.open(path, mode="update") as hd:
        del hd[1].header["TTYPE2"]

    full = table.read(path)
    assert full.column_names == ["ID", "COL2"]
    empty = table.read(path, row_slice=(0, 0))
    assert empty.column_names == ["ID", "COL2"]


def test_capability_check_covers_unnamed_columns(tmp_path):
    """Unnamed vector columns must disqualify the scalar-only fast paths
    (r5a-04); the check used to skip every TTYPE-less column."""
    from torchfits._table._read_schema import _can_use_mmap_row_path_for_full_read

    path = str(tmp_path / "unnamed_vec.fits")
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.arange(4, dtype=np.int32)),
            fits.Column(
                name="VEC",
                format="5J",
                array=np.arange(20, dtype=np.int32).reshape(4, 5),
            ),
        ]
    )
    hdu.writeto(path)
    with fits.open(path, mode="update") as hd:
        del hd[1].header["TTYPE2"]

    assert _can_use_mmap_row_path_for_full_read(path, 1, None) is False


def test_io_errors_propagate_from_read_and_scan(numeric_fits, monkeypatch):
    """A header-level OSError must surface instead of silently degrading the
    read (r5a-05): the decode fallbacks only swallow decode errors."""
    import torchfits
    import torchfits.table as table

    def boom(*args, **kwargs):
        raise OSError("header IO failure")

    monkeypatch.setattr(torchfits, "read_header", boom)
    with pytest.raises(OSError, match="header IO failure"):
        table.read(numeric_fits)
    with pytest.raises(OSError, match="header IO failure"):
        list(table.scan(numeric_fits))


class _StubReader:
    """Minimal TableReader stand-in returning canned per-range segments."""

    def __init__(self, segs):
        self._segs = list(segs)

    def read_rows(self, cols, start, length):
        return self._segs.pop(0)


def test_read_ranges_short_segment_raises():
    """A short segment must raise, never silently splice list buffers (r5a-06)."""
    import torch
    from torchfits._table.engine import _read_ranges_as_chunk

    reader = _StubReader([{"T": torch.tensor([1, 2]), "L": [1, 2]}])
    with pytest.raises(RuntimeError, match="segment"):
        _read_ranges_as_chunk(reader, ["T", "L"], [(0, 3)])


def test_read_ranges_empty_segment_raises():
    """Rows of an empty segment must never surface as zeros/None (r5a-06)."""
    import torch
    from torchfits._table.engine import _read_ranges_as_chunk

    reader = _StubReader([{"T": torch.tensor([1, 2]), "L": [1, 2]}, {}])
    with pytest.raises(RuntimeError, match="segment"):
        _read_ranges_as_chunk(reader, ["T", "L"], [(0, 2), (5, 1)])


def test_read_ranges_zero_length_ranges_ok():
    """Zero-length coalesced ranges are harmless (r5a-06 pin)."""
    import torch
    from torchfits._table.engine import _read_ranges_as_chunk

    reader = _StubReader(
        [{"T": torch.tensor([1, 2]), "L": [1, 2]}, {"T": torch.tensor([5]), "L": [5]}]
    )
    out = _read_ranges_as_chunk(reader, ["T", "L"], [(0, 2), (0, 0), (5, 1)])
    assert out["T"].tolist() == [1, 2, 5]
    assert out["L"] == [1, 2, 5]
    assert _read_ranges_as_chunk(_StubReader([]), ["T"], []) == {}


def test_complex_projection_columns_are_served(tmp_path):
    """Projecting past complex columns must work; unprojectable complex
    columns still raise a clear NotImplementedError (r5a-08)."""
    import torchfits.table as table

    path = str(tmp_path / "complex.fits")
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="CX",
                format="C",
                array=np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            ),
            fits.Column(name="ID", format="J", array=np.arange(2, dtype=np.int32)),
        ]
    )
    hdu.writeto(path)

    t = table.read(path, columns=["ID"])
    assert t.column_names == ["ID"]
    assert t["ID"].to_pylist() == [0, 1]
    with pytest.raises(NotImplementedError, match="complex"):
        table.read(path)
    with pytest.raises(NotImplementedError, match="complex"):
        table.read(path, columns=["ID"], where="CX > 1")


def test_reader_empty_result_schema_matches_hdu(tmp_path):
    """reader() must type an empty result from the requested HDU (r5a-15)."""
    import torchfits.table as table

    path = str(tmp_path / "two_hdus.fits")
    primary = fits.PrimaryHDU()
    one = fits.BinTableHDU.from_columns(
        [fits.Column(name="A", format="J", array=np.arange(2, dtype=np.int32))]
    )
    one.name = "ONE"
    two = fits.BinTableHDU.from_columns(
        [fits.Column(name="B", format="E", array=np.arange(2, dtype=np.float32))]
    )
    two.name = "TWO"
    fits.HDUList([primary, one, two]).writeto(path)

    rdr = table.reader(path, hdu="TWO", row_slice=(0, 0))
    result = rdr.read_all()
    assert result.schema.names == ["B"]
    assert str(result.schema.field("B").type) == "float"


def test_fallback_table_opens_file_once_per_call(tmp_path):
    """A table read must open the file exactly once per call (r5a-11).

    Count proof: per cache-miss call the shared handle is the only open
    (``open_fits_file`` == 1) and no path-based binding runs — those open the
    file a second time inside the extension (the historical "fallback-table
    double-open per call").
    """
    from unittest import mock

    import torchfits
    import torchfits._C as cpp

    path = str(tmp_path / "count.fits")
    fits.BinTableHDU.from_columns(
        [fits.Column(name="ID", format="J", array=np.arange(64, dtype=np.int32))]
    ).writeto(path)

    counts = {"open": 0, "path": 0}
    orig_open = cpp.open_fits_file
    orig_rft = cpp.read_fits_table
    orig_rftr = cpp.read_fits_table_rows

    def c_open(*args, **kwargs):
        counts["open"] += 1
        return orig_open(*args, **kwargs)

    def c_rft(*args, **kwargs):
        counts["path"] += 1
        return orig_rft(*args, **kwargs)

    def c_rftr(*args, **kwargs):
        counts["path"] += 1
        return orig_rftr(*args, **kwargs)

    with (
        mock.patch.object(cpp, "open_fits_file", c_open),
        mock.patch.object(cpp, "read_fits_table", c_rft),
        mock.patch.object(cpp, "read_fits_table_rows", c_rftr),
    ):
        out = torchfits.read(path, 1)

    assert out["ID"].shape == (64,)
    assert counts == {"open": 1, "path": 0}, (
        f"expected one open and zero path-based binding calls, got {counts}"
    )
