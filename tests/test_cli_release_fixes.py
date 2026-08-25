"""Release-fix regressions for the torchfits CLI (B2, B3, H6, M10).

Covers:
- B2: ``arith`` on integer images must not wrap/truncate silently.
- B3: ``stats`` must work on unsigned-convention images.
- H6: ``diff`` must not report differences between identical NaN-bearing files.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

astropy = pytest.importorskip("astropy")
torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
from astropy.io import fits as afits  # noqa: E402


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_uint16(path):
    """uint16-convention image (BZERO=32768) holding small values."""
    raw = np.arange(10, dtype=np.int16).reshape(2, 5)
    hdu = afits.PrimaryHDU(data=raw)
    hdu.header["BZERO"] = 32768
    hdu.header["BSCALE"] = 1
    hdu.writeto(str(path), overwrite=True)


def _read_u16(path):
    return torchfits.read_tensor(str(path), hdu=0).numpy()


def test_arith_add_does_not_wrap_uint16(tmp_path):
    src = tmp_path / "u16.fits"
    out = tmp_path / "out.fits"
    _write_uint16(src)
    result = _run_cli(
        "arith", str(src), "--op", "add", "--value", "5", "-o", str(out)
    )
    assert result.returncode == 0, result.stderr
    got = _read_u16(out)
    expect = np.clip(_read_u16(src).astype(np.int64) + 5, 0, 65535).astype(np.uint16)
    np.testing.assert_array_equal(got, expect)


def test_arith_mul_saturates_int16(tmp_path):
    src = tmp_path / "i16.fits"
    out = tmp_path / "mul.fits"
    afits.PrimaryHDU(data=np.array([[200, -200]], dtype=np.int16)).writeto(str(src))
    result = _run_cli(
        "arith", str(src), "--op", "mul", "--value", "200", "-o", str(out)
    )
    assert result.returncode == 0, result.stderr
    got = torchfits.read_tensor(str(out), hdu=0).numpy()
    assert got.max() <= np.iinfo(np.int16).max
    assert got.min() >= np.iinfo(np.int16).min


def test_arith_div_produces_float(tmp_path):
    src = tmp_path / "ints.fits"
    out = tmp_path / "div.fits"
    afits.PrimaryHDU(data=np.array([[1, 3]], dtype=np.int32)).writeto(str(src))
    result = _run_cli("arith", str(src), "--op", "div", "--value", "2", "-o", str(out))
    assert result.returncode == 0, result.stderr
    got = torchfits.read_tensor(str(out), hdu=0).numpy()
    assert got.dtype.kind == "f"
    np.testing.assert_allclose(got, [[0.5, 1.5]])


def test_stats_on_unsigned_convention_image(tmp_path):
    src = tmp_path / "u16.fits"
    _write_uint16(src)
    result = _run_cli("stats", str(src))
    assert result.returncode == 0, result.stderr
    assert '"dtype": "uint16"' in result.stdout or "uint16" in result.stdout


def test_diff_identical_nan_files_is_clean(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    data[1, 1] = np.nan
    for p in (a, b):
        afits.PrimaryHDU(data=data.copy()).writeto(str(p))
    result = _run_cli("diff", str(a), str(b))
    assert result.returncode == 0, result.stdout + result.stderr
