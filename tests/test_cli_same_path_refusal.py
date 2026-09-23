"""OUTPUT == INPUT refusal across the rewrite commands.

The verbatim contract (``cmds_copy.py``): ``reject_same_path`` raises
``UsageError`` (exit 2) with ``refusing same-path rewrite in place: {src}``.
``compress``/``decompress`` refuse via ``resolve_batch_io_pairs(...,
refuse_same_path=True)``; ``convert`` must refuse every INPUT/OUTPUT pair the
same way. ``--split hdu`` generated outputs must never land on another input
of the same batch (re-running a split into the same directory).
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


REFUSAL = "refusing same-path rewrite in place:"


def test_compress_same_path_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), overwrite=True)
    before = img.read_bytes()
    result = _run_cli("compress", str(img), str(img))
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert img.read_bytes() == before


def test_decompress_same_path_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), overwrite=True)
    before = img.read_bytes()
    result = _run_cli("decompress", str(img), str(img))
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert img.read_bytes() == before


def test_convert_table_same_path_refused(tmp_path):
    tab = tmp_path / "tab.fits"
    torchfits.table.write(
        str(tab), {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}, overwrite=True
    )
    before = tab.read_bytes()
    # Without the refusal this silently rewrites the source (dropping column b).
    result = _run_cli("convert", str(tab), str(tab), "--to", "fits", "-c", "a")
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert tab.read_bytes() == before


def test_convert_png_same_path_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), overwrite=True)
    before = img.read_bytes()
    # Without the refusal this replaces the FITS source with PNG bytes.
    result = _run_cli("convert", str(img), "-o", str(img), "--to", "png")
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert img.read_bytes() == before


def test_convert_png_output_equals_one_input_refused(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    torchfits.write(str(a), torch.zeros(2, 2), overwrite=True)
    torchfits.write(str(b), torch.ones(2, 2), overwrite=True)
    before = a.read_bytes()
    result = _run_cli("convert", str(a), str(b), "-o", str(a), "--to", "png")
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert a.read_bytes() == before


def test_compress_split_hdu_output_must_not_clobber_another_input(tmp_path):
    x = tmp_path / "x.fits"
    prior = tmp_path / "x_hdu00.fits"
    torchfits.write(str(x), torch.arange(4, dtype=torch.float32).reshape(2, 2))
    torchfits.write(str(prior), torch.ones(2, 2) * 7)
    before = prior.read_bytes()
    # x's split output is named x_hdu00.fits -- exactly the other input.
    result = _run_cli(
        "compress",
        str(x),
        str(prior),
        "--split",
        "hdu",
        "--out-dir",
        str(tmp_path),
        "-J",
        "1",
    )
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert prior.read_bytes() == before


def test_arith_split_hdu_output_must_not_clobber_another_input(tmp_path):
    x = tmp_path / "x.fits"
    prior = tmp_path / "x_hdu00.fits"
    torchfits.write(str(x), torch.arange(4, dtype=torch.float32).reshape(2, 2))
    torchfits.write(str(prior), torch.ones(2, 2) * 7)
    before = prior.read_bytes()
    result = _run_cli(
        "arith",
        str(x),
        str(prior),
        "--op",
        "add",
        "--value",
        "1",
        "--split",
        "hdu",
        "--out-dir",
        str(tmp_path),
        "-J",
        "1",
    )
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert prior.read_bytes() == before
