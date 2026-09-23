"""``arith`` evaluation edges: integer saturation must not wrap, fractions warn.

The integer accumulation path promises "no silent integer wraparound" (see
``cmds_arith._compute``): out-of-range results saturate to the output dtype
with a ``RuntimeWarning``. Values cast from float arithmetic to an integer
output cannot keep fractions, but the loss must be warned about, not silent.
"""

from __future__ import annotations

import subprocess
import sys

import torch

import torchfits


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_arith_uint32_mul_saturates_instead_of_wrapping(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    big = torch.tensor([[4294967290, 4294967290], [1, 4000000000]], dtype=torch.uint32)
    torchfits.write(str(a), big, overwrite=True)
    torchfits.write(str(b), big, overwrite=True)
    out = tmp_path / "prod.fits"
    result = _run_cli("arith", str(a), str(b), "--op", "mul", "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert "saturated" in result.stderr
    got = torchfits.read_tensor(str(out), hdu=0).to(torch.int64)
    # 4294967290^2 and 4000000000^2 overflow uint32 and must saturate to the
    # type maximum; 1*1 is exact. A wrapping int64 accumulator produces
    # garbage (0s) here instead.
    assert got.flatten().tolist() == [4294967295, 4294967295, 1, 4294967295]


def test_arith_fractional_scalar_on_integer_warns(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(
        str(img), torch.arange(100, 116, dtype=torch.int16).reshape(4, 4), overwrite=True
    )
    out = tmp_path / "out.fits"
    result = _run_cli("arith", str(img), "--op", "add", "--value", "0.5", "-o", str(out))
    assert result.returncode == 0, result.stderr
    # --dtype auto keeps the input dtype: 100 + 0.5 cannot keep the .5 in an
    # int16 output, and the dropped fraction must be warned about.
    assert "fractional" in result.stderr
    got = torchfits.read_tensor(str(out), hdu=0)
    assert got.flatten()[:4].tolist() == [100, 101, 102, 103]


def test_arith_div_by_zero_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.ones(2, 2), overwrite=True)
    zero = tmp_path / "zero.fits"
    torchfits.write(str(zero), torch.zeros(2, 2), overwrite=True)

    result = _run_cli(
        "arith", str(img), "--op", "div", "--value", "0", "-o", str(tmp_path / "o1.fits")
    )
    assert result.returncode == 2, result.stderr
    assert "division by zero" in result.stderr

    result = _run_cli(
        "arith", str(img), str(zero), "--op", "div", "-o", str(tmp_path / "o2.fits")
    )
    assert result.returncode == 2, result.stderr
    assert "division by zero" in result.stderr
