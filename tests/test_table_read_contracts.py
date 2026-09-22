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
