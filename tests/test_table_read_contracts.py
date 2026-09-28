"""read_table error contracts and mmap validation (r4c-10/r4c-11)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import torchfits
import torchfits._C as cpp
from torchfits._io_engine import table_api


def _write_table(path) -> str:
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="ID", format="J", array=np.arange(6, dtype=np.int32))]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(str(path), overwrite=True)
    return str(path)


def test_read_torch_rejects_unknown_mmap_string(tmp_path):
    """Only bool or 'auto' mmap modes are valid; typos must not silently mmap."""
    path = _write_table(tmp_path / "t.fits")
    with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
        torchfits.table.read_torch(path, hdu=1, mmap="sometimes")


def test_read_table_surfaces_unexpected_thin_errors(tmp_path, monkeypatch):
    """Programming errors must not be masked by the read_func fallback."""
    path = _write_table(tmp_path / "t.fits")

    def broken(*_a, **_k):
        raise AttributeError("thin path bug")

    monkeypatch.setattr(table_api, "_thin_read_table_torch", broken)
    with pytest.raises(AttributeError, match="thin path bug"):
        torchfits.table.read_torch(path, hdu=1)


def test_read_table_falls_back_on_read_errors(tmp_path, monkeypatch):
    """Expected read failures keep the documented read_func fallback."""
    import warnings

    path = _write_table(tmp_path / "t.fits")

    def unavailable(*_a, **_k):
        raise RuntimeError("thin reader unavailable")

    monkeypatch.setattr(table_api, "_thin_read_table_torch", unavailable)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = torchfits.table.read_torch(path, hdu=1)
    assert out["ID"].tolist() == [0, 1, 2, 3, 4, 5]
    # The internal fallback must not surface the read() handle_cache_capacity
    # deprecation: the caller never passed the knob.
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []


def test_where_filtered_io_errors_propagate(tmp_path, monkeypatch):
    """IO errors on the filtered gather must surface, not become a generic
    "Failed to apply where" RuntimeError (r4c-10)."""
    path = _write_table(tmp_path / "t.fits")

    def unavailable(*_a, **_k):
        raise RuntimeError("thin read down")

    def io_bomb(*_a, **_k):
        raise OSError("storage gone")

    monkeypatch.setattr(table_api, "_thin_read_table_torch", unavailable)
    monkeypatch.setattr(cpp, "read_fits_table_filtered", io_bomb, raising=False)
    with pytest.raises(OSError, match="storage gone"):
        torchfits.table.read_torch(path, hdu=1, where="ID < 5")


def test_where_filtered_binding_failures_keep_contract(tmp_path, monkeypatch):
    """Binding failures keep the established RuntimeError contract (chained)."""
    path = _write_table(tmp_path / "t.fits")

    def unavailable(*_a, **_k):
        raise RuntimeError("thin read down")

    def cpp_bomb(*_a, **_k):
        raise RuntimeError("bad predicate")

    monkeypatch.setattr(table_api, "_thin_read_table_torch", unavailable)
    monkeypatch.setattr(cpp, "read_fits_table_filtered", cpp_bomb, raising=False)
    with pytest.raises(RuntimeError, match="Failed to apply where"):
        torchfits.table.read_torch(path, hdu=1, where="ID < 5")


def _fits_card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value}".ljust(80)[:80].encode("latin-1")


def _extend_table_header_with_zero_repeat_field(path, tmp_path) -> str:
    """Add a field 4 with TFORM='0J' -- legal, and zero bytes wide per row.

    A FITS header may span several 2880-byte blocks, so this is a well-formed
    header rather than a patched one: the original block loses its END and a new
    block carries TTYPE4/TFORM4/END. The data segment is untouched, which is what
    makes the result a realistic input rather than a corrupt file.
    """
    import os

    raw = bytearray(open(path, "rb").read())
    start = next(
        off
        for off in range(0, len(raw), 2880)
        if bytes(raw[off : off + 8]).strip() == b"XTENSION"
    )
    block = bytearray(raw[start : start + 2880])
    cards = [bytes(block[i : i + 80]) for i in range(0, 2880, 80)]
    end_i = next(i for i, c in enumerate(cards) if c.strip().startswith(b"END"))
    for i, c in enumerate(cards):
        if c.strip().startswith(b"TFIELDS"):
            block[i * 80 : (i + 1) * 80] = _fits_card("TFIELDS", "4")
    without_end = bytes(block[: end_i * 80]).ljust(2880, b" ")
    second = (
        _fits_card("TTYPE4", "'DDDD'") + _fits_card("TFORM4", "'0J'") + b"END".ljust(80)
    ).ljust(2880, b" ")
    assert len(second) == 2880
    out = tmp_path / "zero_repeat.fits"
    out.write_bytes(
        bytes(raw[:start]) + without_end + second + bytes(raw[start + 2880 :])
    )
    return os.fspath(out)


def test_zero_repeat_tform_is_rejected_not_silently_borrowed(tmp_path) -> None:
    """A zero-width column must not be filled with a neighbouring column's bytes.

    Regression: ``analyze_table`` validated the TFORM repeat count for overflow
    and negativity but not for zero, while ``extract_column_data`` already
    rejected ``repeat <= 0``. Because ``col.repeat`` drives the element count, a
    zero-repeat column read a count of 0 out of a row that does hold the other
    columns' bytes, and the value surfaced under this column's name. Measured on
    a file astropy accepts, with three 4J columns of known values:

        astropy   DDDD -> array([], shape=(2, 0))   # zero width, no data
        torchfits DDDD -> [286331154, 0]            # 0x11111112: AAA's row-1 value

    So the answer was not garbage but another column's plausible data. Rejecting
    it at open time also makes the two bounds agree.
    """
    path = tmp_path / "good.fits"
    cols = [
        fits.Column(
            name=name,
            format="4J",
            array=np.array([value, value + 1], dtype=np.int32),
        )
        for name, value in (
            ("AAA", 0x11111111),
            ("BBB", 0x22222222),
            ("CCC", 0x33333333),
        )
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)

    bad = _extend_table_header_with_zero_repeat_field(path, tmp_path)

    # The fixture must be a legal file, or this proves nothing.
    with fits.open(bad) as hdul:
        assert hdul[1].header["TFIELDS"] == 4
        assert hdul[1].header["TFORM4"].strip() == "0J"
        assert hdul[1].columns.names == ["AAA", "BBB", "CCC", "DDDD"]
        assert hdul[1].data["DDDD"].shape == (2, 0)

    with pytest.raises(RuntimeError, match=r"repeat count must be positive.*DDDD"):
        torchfits.table.read(bad, hdu=1)


# ---------------------------------------------------------------------------
# Deep-review unit 10, TS-008: row_slice's negative-stop rejection
# ---------------------------------------------------------------------------
#
# `_normalize_row_slice` rejects a negative `stop` because the total row count
# is unknown at parse time -- and because `-1` is the *internal* "to the end"
# sentinel that `_normalize_row_slice(None)` itself returns. Nothing tested the
# rejection: dropping the guard let 104 tests stay green, and a user writing
# `row_slice=slice(0, -1)` meaning "the first row" would silently get every
# row instead of an error.


def test_row_slice_rejects_a_negative_stop(tmp_path):
    path = _write_table(tmp_path / "t.fits")
    for bad in (slice(0, -1), (0, -1), (0, -2)):
        with pytest.raises(ValueError, match="negative stop"):
            torchfits.table.read(path, hdu=1, row_slice=bad)


def test_row_slice_rejects_a_negative_start_and_a_step(tmp_path):
    path = _write_table(tmp_path / "t.fits")
    for bad in (slice(-1, 3), (-2, 3), slice(-1, None)):
        with pytest.raises(ValueError, match="start must be >= 0"):
            torchfits.table.read(path, hdu=1, row_slice=bad)
    with pytest.raises(ValueError, match="step must be 1"):
        torchfits.table.read(path, hdu=1, row_slice=slice(0, 3, 2))


def test_row_slice_still_accepts_the_supported_shapes(tmp_path):
    """The rejection above must not have narrowed what is accepted."""
    path = _write_table(tmp_path / "t.fits")
    assert torchfits.table.read(path, hdu=1, row_slice=slice(0, 2)).num_rows == 2
    assert torchfits.table.read(path, hdu=1, row_slice=(1, 3)).num_rows == 2
    assert torchfits.table.read(path, hdu=1, row_slice=slice(2, None)).num_rows == 4
    assert torchfits.table.read(path, hdu=1).num_rows == 6


def test_normalize_row_slice_internal_sentinel_is_still_minus_one():
    """`-1` stays the internal all-rows sentinel; only user input is refused."""
    from torchfits._table.utils import _normalize_row_slice

    assert _normalize_row_slice(None) == (1, -1)
    assert _normalize_row_slice(slice(0, 3)) == (1, 3)
    assert _normalize_row_slice((2, 5)) == (3, 3)
