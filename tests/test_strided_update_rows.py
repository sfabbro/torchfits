"""Strided DLPack payloads and mmap row updates (R8 slice A pins).

`update_fits_table_rows_mmap` receives payloads through the DLPack protocol,
so element offsets must honor the payload's strides: writing through a
non-contiguous view (e.g. ``t[::2]``, a Fortran-order slice, a negative
stride) must land exactly the logical values, for numeric columns as well as
bool/uint8/string. A 0-d payload carries no stride array at all and must be
accepted as a single value.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits as afits

import torch
import torchfits
import torchfits.table as ttable
import torchfits._C as cpp


def _write_table(path, nrows=10):
    cols = [
        afits.Column(name="I32", format="J", array=np.arange(nrows, dtype="<i4") + 100),
        afits.Column(name="I16", format="I", array=np.arange(nrows, dtype="<i2") + 7),
        afits.Column(name="I64", format="K", array=np.arange(nrows, dtype="<i8") + 9),
        afits.Column(
            name="F32", format="E", array=np.arange(nrows, dtype="<f4") + 0.25
        ),
        afits.Column(name="F64", format="D", array=np.arange(nrows, dtype="<f8") + 0.5),
        afits.Column(name="U8", format="B", array=np.arange(nrows, dtype="u1") + 3),
        afits.Column(name="LOG", format="L", array=np.arange(nrows) % 2 == 0),
        afits.Column(
            name="S5", format="5A", array=np.array([f"s{i:04d}" for i in range(nrows)])
        ),
        afits.Column(
            name="V2",
            format="2J",
            array=np.arange(nrows * 2, dtype="<i4").reshape(nrows, 2),
        ),
    ]
    afits.BinTableHDU.from_columns(cols).writeto(str(path), overwrite=True)
    return str(path)


def _read(path):
    out = torchfits.read(path, hdu=1)
    return {k: np.asarray(v) for k, v in out.items()}


def test_strided_1d_payloads_roundtrip_exact(tmp_path):
    path = _write_table(tmp_path / "s1d.fits", nrows=10)
    nrows = 10
    full = {
        "I32": np.arange(2 * nrows, dtype="<i4") + 1000,
        "I16": np.arange(2 * nrows, dtype="<i2") + 55,
        "I64": np.arange(2 * nrows, dtype="<i8") + 77,
        "F32": np.arange(2 * nrows, dtype="<f4") + 1.25,
        "F64": np.arange(2 * nrows, dtype="<f8") + 2.25,
        "U8": np.arange(2 * nrows, dtype="u1") + 66,
        "LOG": (np.arange(2 * nrows) % 3 == 0),
        "S5": np.array(
            [[0x61 + (i % 26)] * 5 for i in range(2 * nrows)], dtype=np.uint8
        ),
    }
    payload = {k: v[::2] for k, v in full.items()}
    # Views into a larger base: flat (unstrided) indexing would write the
    # wrong half of each base array.
    for k, v in payload.items():
        assert v.base is not None, k
    cpp.update_fits_table_rows_mmap(path, 1, payload, 1, nrows)

    got = _read(path)
    for k, v in payload.items():
        exp = np.ascontiguousarray(v)
        if k == "LOG":
            exp = exp.astype(bool)
        assert got[k].shape == exp.shape, k
        assert np.array_equal(got[k], exp), f"{k}: got {got[k]} expected {exp}"


def test_strided_2d_views_roundtrip_exact(tmp_path):
    path = _write_table(tmp_path / "s2d.fits", nrows=10)
    nrows = 10
    base = (np.arange(4 * nrows, dtype="<i4") + 700).reshape(2 * nrows, 2)
    row_strided = base[::2]  # element strides (4, 1)
    assert row_strided.strides == (16, 4)
    wide = np.asfortranarray(
        (np.arange(4 * nrows, dtype="<i4") + 900).reshape(nrows, 4)
    )
    col_strided = wide[:, :2]  # element strides (1, 4)
    assert col_strided.strides == (4, 40)

    cpp.update_fits_table_rows_mmap(path, 1, {"V2": row_strided}, 1, nrows)
    got = _read(path)["V2"]
    assert np.array_equal(got, row_strided)

    cpp.update_fits_table_rows_mmap(path, 1, {"V2": col_strided}, 1, nrows)
    got = _read(path)["V2"]
    assert np.array_equal(got, col_strided)


def test_negative_strided_view_roundtrip_exact(tmp_path):
    path = _write_table(tmp_path / "sneg.fits", nrows=10)
    base = np.arange(20, dtype="<i4") + 300
    view = base[::-2]
    assert view.strides == (-8,)
    cpp.update_fits_table_rows_mmap(path, 1, {"I32": view}, 1, 10)
    got = _read(path)["I32"]
    assert np.array_equal(got, np.ascontiguousarray(view))


def test_torch_strided_view_roundtrip_high_level(tmp_path):
    path = _write_table(tmp_path / "storch.fits", nrows=8)
    big = torch.arange(16, dtype=torch.int32) + 500
    ttable.update_rows(path, {"I32": big[::2]}, row_slice=(0, 8), mmap=True)
    got = _read(path)["I32"]
    exp = (np.arange(16, dtype="<i4") + 500)[::2]
    assert np.array_equal(got, exp)


def test_zero_dim_payload_single_row_update(tmp_path):
    """A 0-d payload has no DLPack strides; it must update one row, not crash.

    The crash is a SIGSEGV inside the extension, which would take the pytest
    process down with it, so the repro runs in a subprocess.
    """
    path = _write_table(tmp_path / "s0d.fits", nrows=3)
    code = (
        "import numpy as np, torchfits, torchfits._C as cpp\n"
        f"cpp.update_fits_table_rows_mmap({path!r}, 1, "
        "{'I32': np.array(42, dtype='<i4')}, 1, 1)\n"
        f"print(int(torchfits.read({path!r}, hdu=1)['I32'][0]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, (
        f"0-dim payload crashed: rc={proc.returncode} {proc.stderr}"
    )
    assert proc.stdout.strip().splitlines()[-1] == "42"


def _make_scaled_column_table(path):
    afits.BinTableHDU.from_columns(
        [afits.Column(name="S", format="I", array=np.arange(5, dtype="<i2"))]
    ).writeto(str(path), overwrite=True)
    with afits.open(str(path), mode="update") as hdul:
        hdul[1].header["TSCAL1"] = 2.0
        hdul[1].header["TZERO1"] = 1.0
    return str(path)


def test_scaled_column_update_roundtrips_physical_values(tmp_path):
    """Update payloads are physical values (CFITSIO inverse-scales on write).

    With TSCAL=2/TZERO=1 a raw cell r reads back as 2r+1. Writing physical
    values through the default (mmap=auto) path must round-trip exactly; a
    raw-bit mmap write silently stores the physical values as raw cells and
    every value comes back doubled.
    """
    path = _make_scaled_column_table(tmp_path / "scaled.fits")
    payload = np.array([3, 5, 7, 9, 11], dtype="<i2")  # raw cells 1..5
    ttable.update_rows(path, {"S": payload}, row_slice=(0, 5))
    got = np.asarray(torchfits.read(path, hdu=1)["S"])
    assert got.dtype == np.float64
    assert np.array_equal(got, payload.astype(np.float64))


def test_scaled_column_forced_mmap_update_raises(tmp_path):
    """Forced mmap updates cannot inverse-scale; they must refuse, not write
    raw bits under a physical-value payload."""
    path = _make_scaled_column_table(tmp_path / "scaled2.fits")
    payload = np.array([3, 5, 7, 9, 11], dtype="<i2")
    with pytest.raises(RuntimeError, match="not supported for mmap"):
        ttable.update_rows(path, {"S": payload}, row_slice=(0, 5), mmap=True)
    # refused before any write: the file still reads as the original raw 0..4
    got = np.asarray(torchfits.read(path, hdu=1)["S"])
    assert np.array_equal(got, np.array([1.0, 3.0, 5.0, 7.0, 9.0]))


def test_duplicate_ttype_update_hits_read_visible_column(tmp_path):
    """Duplicate TTYPE cards: the update writer and the reader must resolve
    the name to the same column (the first), or an update vanishes silently."""
    path = _write_table(tmp_path / "dup.fits", nrows=4)
    with afits.open(path, mode="update") as hdul:
        hdul[1].header["TTYPE2"] = "I32"  # I16 column renamed to duplicate I32
    payload = np.array([90, 91, 92, 93], dtype="<i4")
    ttable.update_rows(path, {"I32": payload}, row_slice=(0, 4), mmap=True)
    got = cpp.read_fits_table(path, 1, ["I32"], True)
    assert np.array_equal(np.asarray(got["I32"]), payload)
