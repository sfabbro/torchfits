"""``convert`` edges: honest failures and no silently ignored recipe flags.

An error while inspecting the source must fail the command (exit 3), never
degrade into rendering a wrong-band PNG. Recipe-scoped flags are refused
outright when the selected recipe cannot honor them (silent no-ops would
produce wrongly calibrated or wrongly stretched previews).
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


def test_convert_png_fails_when_hdu_count_read_fails(tmp_path, monkeypatch):
    from torchfits.cli.main import main

    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(4, 4), overwrite=True)
    out = tmp_path / "out.png"

    def boom(path):
        raise RuntimeError("cannot count HDUs")

    monkeypatch.setattr(torchfits, "read_num_hdus", boom)
    rc = main(["convert", str(img), "-o", str(out), "--to", "png"])
    assert rc == 3
    assert not out.exists()


def test_convert_lupton_rejects_auto_recipe_flags(tmp_path):
    r = tmp_path / "r.fits"
    g = tmp_path / "g.fits"
    b = tmp_path / "b.fits"
    torchfits.write(str(r), torch.zeros(2, 2), overwrite=True)
    torchfits.write(str(g), torch.ones(2, 2), overwrite=True)
    torchfits.write(str(b), torch.full((2, 2), 2.0), overwrite=True)
    out = tmp_path / "out.png"

    for extra in (
        ("--zeropoints", "25,25,25"),
        ("--calibrated",),
        ("--brightness", "0.5"),
        ("--saturation", "1.0"),
    ):
        result = _run_cli(
            "convert",
            str(r),
            str(g),
            str(b),
            "-o",
            str(out),
            "--to",
            "png",
            "--recipe",
            "lupton",
            *extra,
        )
        assert result.returncode == 2, (extra, result.stderr)
        assert not out.exists()


def test_convert_auto_rejects_lupton_flags(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), overwrite=True)
    out = tmp_path / "out.png"

    for extra in (("--q", "4.0"), ("--stretch", "0.3")):
        result = _run_cli(
            "convert",
            str(img),
            "-o",
            str(out),
            "--to",
            "png",
            "--recipe",
            "auto",
            *extra,
        )
        assert result.returncode == 2, (extra, result.stderr)
        assert not out.exists()
