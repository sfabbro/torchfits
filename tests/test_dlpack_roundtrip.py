import importlib
import os

import numpy as np
import pytest

import torch


def _m():
    return importlib.import_module("torchfits._C")


def test_dlpack_roundtrip_cpu():
    # echo_tensor is a nanobind test helper on _C, not the sealed _cpp façade.
    m = importlib.import_module("torchfits._C")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = m.echo_tensor(t)
    # CPU tensors should share the same storage pointer (zero-copy round-trip)
    assert out.data_ptr() == t.data_ptr()


def test_echo_tensor_survives_source_free():
    # Lifetime pin (r5c-11): the echoed tensor must keep the source storage
    # alive after the Python source tensor is garbage-collected.
    import gc
    import weakref

    m = importlib.import_module("torchfits._C")

    def build():
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        return m.echo_tensor(t), weakref.ref(t.untyped_storage())

    out, ref = build()
    gc.collect()
    assert ref() is not None, "echo_tensor result dropped the source storage"
    assert out.data_ptr() != 0
    assert float(out.sum().item()) == float(sum(range(12)))


def test_strided_update_rows_buffered_roundtrip(tmp_path):
    """A strided (t[::2]) column view must write its visible elements (r8b pin).

    The buffered update path materialises strided DLPack payloads through the
    element-stride-aware contiguous copy before fits_write_col.
    """
    from astropy.io import fits

    path = str(tmp_path / "strupd.fits")
    rows = np.arange(8, dtype=np.int32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="ID", format="J", array=rows)]
            ),
        ]
    ).writeto(path, overwrite=True)

    view = np.arange(16, dtype=np.int32)[::2]  # stride-2, 8 visible elements
    assert view.strides == (8,)
    _m().update_fits_table_rows(path, 1, {"ID": view}, 1, 8)

    with fits.open(path, mode="readonly") as hdul:
        got = np.asarray(hdul[1].data["ID"])
    np.testing.assert_array_equal(got, view)


def test_strided_append_rows_roundtrip(tmp_path):
    """Strided payloads through append_rows land the visible elements (r8b pin)."""
    from astropy.io import fits

    path = str(tmp_path / "strapp.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="ID", format="J", array=np.arange(3, dtype=np.int32))]
            ),
        ]
    ).writeto(path, overwrite=True)

    view = np.arange(16, dtype=np.int32)[::2]
    _m().append_fits_table_rows(path, 1, {"ID": view})

    with fits.open(path, mode="readonly") as hdul:
        got = np.asarray(hdul[1].data["ID"])
    np.testing.assert_array_equal(got, np.concatenate([np.arange(3), view]))


def test_append_rows_unaligned_bit_repeat_roundtrip(tmp_path):
    """BIT ('X') appends with repeat % 8 != 0 must keep each row's call (r8b).

    fits_write_col(TBIT) maps a flat element run onto raw data-unit bits and
    ignores per-row byte padding, so a single flat run across rows shifts
    every row after the first. append_rows must write row-by-row exactly like
    populate_rows (update/insert) and the initial table writer.
    """
    from astropy.io import fits

    repeat, n_existing, n_extra = 12, 5, 3
    # Non-periodic pattern so any bit-level shift is visible; the fixture rows
    # carry the pattern too, so any corruption of existing OR appended rows is
    # visible and the comparison below is exact.
    want = np.array(
        [
            [(r * repeat + b) % 3 == 0 for b in range(repeat)]
            for r in range(n_existing + n_extra)
        ],
        dtype=bool,
    )
    path = str(tmp_path / "bit12x.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="FLAGS",
                        format=f"{repeat}X",
                        array=want[:n_existing],
                    )
                ]
            ),
        ]
    ).writeto(path, overwrite=True)

    _m().append_fits_table_rows(path, 1, {"FLAGS": want[n_existing:]})

    with fits.open(path, mode="readonly") as hdul:
        got = np.asarray(hdul[1].data["FLAGS"], dtype=bool)
    np.testing.assert_array_equal(got, want)


def test_update_rows_unaligned_bit_repeat_roundtrip(tmp_path):
    """Companion pin: the buffered update path already writes per-row (r8b)."""
    from astropy.io import fits

    repeat, nrows = 12, 5
    path = str(tmp_path / "bit12x_upd.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="FLAGS",
                        format=f"{repeat}X",
                        array=np.zeros((nrows, repeat), dtype=bool),
                    )
                ]
            ),
        ]
    ).writeto(path, overwrite=True)

    want = np.array(
        [[(r * repeat + b) % 3 == 0 for b in range(repeat)] for r in range(nrows)],
        dtype=bool,
    )
    _m().update_fits_table_rows(path, 1, {"FLAGS": want}, 1, nrows)

    with fits.open(path, mode="readonly") as hdul:
        got = np.asarray(hdul[1].data["FLAGS"], dtype=bool)
    np.testing.assert_array_equal(got, want)


def test_write_table_unknown_type_raises(tmp_path):
    """Unknown table_type must raise, never silently write binary (r8b)."""
    path = str(tmp_path / "tt.fits")
    payload = {"A": np.array([1, 2], dtype=np.int32)}
    with pytest.raises(ValueError):
        _m().write_fits_table(path, payload, {}, True, None, "asccii")

    # Accepted spellings keep working (case-insensitive).
    for kind in ("binary", "ascii", "Binary", "ASCII"):
        _m().write_fits_table(path, payload, {}, True, None, kind)
        assert os.path.exists(path)


def test_zero_length_vla_rows_roundtrip(tmp_path):
    """Zero-length VLA cells round-trip through append and update (r8b pin)."""
    from astropy.io import fits

    path = str(tmp_path / "vla0.fits")
    vla = np.empty(3, dtype=object)
    vla[0] = np.array([1, 2], dtype=np.int16)
    vla[1] = np.array([3], dtype=np.int16)
    vla[2] = np.array([], dtype=np.int16)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="VLA", format="PJ()", array=vla)]
            ),
        ]
    ).writeto(path, overwrite=True)

    _m().append_fits_table_rows(path, 1, {"VLA": [np.array([], dtype=np.int16)]})
    _m().update_fits_table_rows(path, 1, {"VLA": [np.array([], dtype=np.int16)]}, 2, 1)

    with fits.open(path, mode="readonly") as hdul:
        got = [np.asarray(r).tolist() for r in hdul[1].data["VLA"]]
    assert got == [[1, 2], [], [], []], f"zero-length VLA cells corrupted: {got}"
