"""VLA zero-length rows and table-read edge contracts (R8 slice A pins).

Variable-length-array columns must round-trip rows whose descriptor repeat is
zero (empty rows anywhere in the row set) exactly, on the per-row and the flat
values+offsets representations alike. Scaled VLA columns obey the house
TSCAL/TZERO/TNULL convention: exact float64 physical values with TNULL mapped
to NaN, raw integer TNULL (no TSCAL/TZERO) stays a sentinel. The read edges
pin the error contract: details belong inside the exception, and unknown
column names fail loudly for empty tables too.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits
import torchfits.table as ttable
import torchfits._C as cpp


def _vla_col(name, rows, dtype="<i4"):
    arr = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        arr[i] = np.asarray(r, dtype=dtype)
    return afits.Column(name=name, format="PJ()", array=arr)


def _write(path, cols):
    afits.BinTableHDU.from_columns(cols).writeto(str(path), overwrite=True)
    return str(path)


def test_vla_zero_length_rows_roundtrip_exact(tmp_path):
    rows = [[0, 1, 2], [], [10, 11, 12, 13, 14], [], [99]]
    path = _write(
        tmp_path / "vla.fits",
        [
            _vla_col("V", rows),
            afits.Column(name="N", format="J", array=np.arange(5, dtype="<i4")),
        ],
    )
    out = torchfits.read(path, hdu=1)
    got = [list(np.asarray(v)) for v in out["V"]]
    assert got == [[0, 1, 2], [], [10, 11, 12, 13, 14], [], [99]]
    for v in out["V"]:
        assert np.asarray(v).dtype == np.int32
    assert np.array_equal(np.asarray(out["N"]), np.arange(5))


def test_vla_all_rows_zero_length(tmp_path):
    rows = [[], [], [], [], []]
    path = _write(
        tmp_path / "vla0.fits",
        [
            _vla_col("V", rows),
            afits.Column(name="N", format="J", array=np.arange(5, dtype="<i4")),
        ],
    )
    out = torchfits.read(path, hdu=1)
    assert len(out["V"]) == 5
    assert all(len(np.asarray(v)) == 0 for v in out["V"])
    assert all(np.asarray(v).dtype == np.int32 for v in out["V"])


def test_vla_flat_offsets_with_zero_length(tmp_path):
    rows = [[], [5, 6], [], [50, 51, 52, 53], []]
    path = _write(
        tmp_path / "vlaf.fits",
        [
            _vla_col("V", rows),
            afits.Column(name="N", format="J", array=np.arange(5, dtype="<i4")),
        ],
    )
    out = cpp.read_fits_table_rows_numpy(path, 1, ["V", "N"], 1, -1, False)
    values, offsets = out["V"]
    assert np.array_equal(
        np.asarray(values), np.array([5, 6, 50, 51, 52, 53], dtype="<i4")
    )
    assert np.array_equal(np.asarray(offsets), np.array([0, 0, 2, 2, 6, 6]))


def test_vla_zero_row_window_and_margins(tmp_path):
    rows = [[], [1, 2], [], [4], []]
    path = _write(tmp_path / "vlaw.fits", [_vla_col("V", rows)])
    out = ttable.read_torch(path, hdu=1, start_row=2, num_rows=0)
    assert len(out["V"]) == 0
    out = ttable.read_torch(path, hdu=1, start_row=2, num_rows=2)
    assert [list(np.asarray(v)) for v in out["V"]] == [[1, 2], []]
    out = ttable.read_torch(path, hdu=1, start_row=5, num_rows=1)
    assert [list(np.asarray(v)) for v in out["V"]] == [[]]


def _scaled_vla(path, rows, scale, zero, tnull=None):
    path = _write(path, [_vla_col("V", rows)])
    with afits.open(path, mode="update") as hdul:
        hdul[1].header["TSCAL1"] = scale
        hdul[1].header["TZERO1"] = zero
        if tnull is not None:
            hdul[1].header["TNULL1"] = tnull
    return path


def test_vla_scaled_values_exact_float(tmp_path):
    """Scaled VLA columns decode to exact float64 physical values.

    TSCAL=0.5 on raw [1, 3, 5] must read as [0.5, 1.5, 2.5]; decoding into the
    raw integer tensor truncates the physical values to whole numbers (silent
    data loss).
    """
    rows = [[1, 3, 5], [], [2]]
    path = _scaled_vla(tmp_path / "vsc.fits", rows, scale=0.5, zero=0.0)
    out = torchfits.read(path, hdu=1)
    got = [np.asarray(v) for v in out["V"]]
    assert got[0].dtype == np.float64
    assert np.allclose(got[0], [0.5, 1.5, 2.5], rtol=0, atol=0)
    assert got[1].shape == (0,)
    assert np.allclose(got[2], [1.0], rtol=0, atol=0)


def test_vla_scaled_tnull_becomes_nan(tmp_path):
    rows = [[2, -999, 4], []]
    path = _scaled_vla(tmp_path / "vnul.fits", rows, scale=2.0, zero=0.0, tnull=-999)
    out = torchfits.read(path, hdu=1)
    got = np.asarray(out["V"][0])
    assert got.dtype == np.float64
    assert got[0] == 4.0
    assert np.isnan(got[1])
    assert got[2] == 8.0


def test_vla_raw_tnull_sentinel_stays(tmp_path):
    """Raw integer TNULL without TSCAL/TZERO stays a sentinel (tnull-read-torch)."""
    arr = np.empty(2, dtype=object)
    arr[0] = np.array([3, -999, 4], dtype="<i4")
    arr[1] = np.arange(0, dtype="<i4")
    path = _write(
        tmp_path / "vraw.fits",
        [afits.Column(name="V", format="PJ()", array=arr, null=-999)],
    )
    out = torchfits.read(path, hdu=1)
    got = np.asarray(out["V"][0])
    assert got.dtype == np.int32
    assert np.array_equal(got, [3, -999, 4])


def test_repeat_zero_fixed_column_clean_error(tmp_path):
    """An absurd TFORM (repeat 0) fails loudly at open — never garbage rows."""
    path = _write(
        tmp_path / "rep0.fits",
        [afits.Column(name="A", format="J", array=np.arange(4, dtype="<i4"))],
    )
    with afits.open(path, mode="update") as hdul:
        hdul[1].header["TFORM1"] = "0J"
    with pytest.raises(RuntimeError):
        torchfits.read(path, hdu=1)


def _simple_table(path, nrows=6):
    path = _write(
        path,
        [
            afits.Column(
                name="I32", format="J", array=np.arange(nrows, dtype="<i4") + 100
            ),
            afits.Column(
                name="I16", format="I", array=np.arange(nrows, dtype="<i2") + 7
            ),
        ],
    )
    return path


def test_duplicate_column_request_returns_data(tmp_path):
    """Requesting the same column twice must return its data, not a moved-from
    (None) tensor: the assembly loop used to move the same entry out twice."""
    path = _simple_table(tmp_path / "dupreq.fits")
    expected = np.arange(6, dtype="<i4") + 100
    for call in (
        lambda: cpp.read_fits_table(path, 1, ["I32", "I32"], False),
        lambda: cpp.read_fits_table(path, 1, ["I32", "I32"], True),
        lambda: cpp.read_fits_table_rows(path, 1, ["I32", "I32"], 1, -1, False),
        lambda: cpp.read_fits_table_rows(path, 1, ["I32", "I32"], 1, 4, True),
    ):
        out = call()
        assert out["I32"] is not None
        assert np.array_equal(
            np.asarray(out["I32"])[: len(expected)],
            expected[: len(np.asarray(out["I32"]))],
        )
    out = cpp.read_fits_table_rows(path, 1, ["I16", "I32", "I16"], 1, -1, False)
    assert out["I16"] is not None
    assert out["I32"] is not None


def test_duplicate_ttype_read_keeps_all_columns_valid(tmp_path):
    path = _simple_table(tmp_path / "dupt.fits")
    with afits.open(path, mode="update") as hdul:
        hdul[1].header["TTYPE2"] = "I32"
    out = torchfits.read(path, hdu=1)
    for name, value in out.items():
        assert value is not None, name
    # Name lookups resolve to the first matching column.
    got = cpp.read_fits_table_rows(path, 1, ["I32"], 1, -1, False)
    assert np.array_equal(np.asarray(got["I32"]), np.arange(6, dtype="<i4") + 100)


def test_error_messages_include_details(tmp_path):
    """Diagnostics belong inside the exception, not on stderr (A-07)."""
    path = _simple_table(tmp_path / "msgs.fits")
    with pytest.raises(RuntimeError, match=r"Invalid start row.*99"):
        cpp.read_fits_table_rows(path, 1, [], 99, -1, False)
    with pytest.raises(RuntimeError, match=r"Column not found: nope") as excinfo:
        cpp.read_fits_table_rows(path, 1, ["nope"], 1, -1, False)
    assert "I32" in str(excinfo.value)  # available columns travel with the error


def test_empty_table_unknown_column_raises(tmp_path):
    """Empty tables must not silently drop requested unknown columns."""
    path = _write(
        tmp_path / "empty.fits",
        [afits.Column(name="A", format="J", array=np.arange(0, dtype="<i4"))],
    )
    for call in (
        lambda: cpp.read_fits_table(path, 1, ["nope"], False),
        lambda: cpp.read_fits_table(path, 1, ["nope"], True),
        lambda: cpp.read_fits_table_rows(path, 1, ["nope"], 1, -1, False),
    ):
        with pytest.raises(RuntimeError, match=r"Column not found: nope"):
            call()


def test_empty_table_valid_columns_still_empty(tmp_path):
    path = _write(
        tmp_path / "empty2.fits",
        [afits.Column(name="A", format="J", array=np.arange(0, dtype="<i4"))],
    )
    assert dict(cpp.read_fits_table(path, 1, ["A"], False)) == {}
    assert dict(cpp.read_fits_table(path, 1, [], True)) == {}
