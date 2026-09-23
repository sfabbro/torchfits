"""CLI regression tests for ``torchfits diff`` / ``copy`` / ``cutout`` (R7-B).

Covers diff result correctness (float64 means, empty images, type mismatch),
the copy same-path refusal contract (the reference for compress/convert/
decompress), the ``copy-is-binary`` invariant, and ``cutout --box`` edge
validation.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

import torchfits
import torchfits.table as tf_table


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_diff_float64_mean_is_computed_in_float64(tmp_path):
    """Files whose true means differ must not compare equal.

    [1e8, 1.0, -1e8, 0.0] and [1e8, 0.0, -1e8, 0.0] share min/max/shape and
    share a float32-rounded mean of 0.0, but their true (float64) means are
    0.25 and 0.0 — diff must detect the mean difference.
    """
    path_a = tmp_path / "prec_a.fits"
    path_b = tmp_path / "prec_b.fits"
    torchfits.write(
        str(path_a), torch.tensor([1e8, 1.0, -1e8, 0.0], dtype=torch.float64), overwrite=True
    )
    torchfits.write(
        str(path_b), torch.tensor([1e8, 0.0, -1e8, 0.0], dtype=torch.float64), overwrite=True
    )
    result = _run_cli("diff", str(path_a), str(path_b))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "mean" in result.stderr


def test_diff_empty_images_compare_clean(tmp_path):
    """Two identical empty-region images are equal: no crash, exit 0."""
    path = tmp_path / "empty.fits"
    torchfits.write(str(path), torch.zeros((0, 4)), overwrite=True)
    result = _run_cli("diff", str(path), str(path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_diff_type_mismatch_reports_diff_without_crash(tmp_path):
    """An IMAGE-vs-TABLE HDU mismatch is a diff line, not an I/O error."""
    img = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    empty = torch.zeros((0, 0))
    path_a = tmp_path / "mix_a.fits"
    path_b = tmp_path / "mix_b.fits"
    torchfits.write(str(path_a), [empty, img], overwrite=True)
    tf_table.write(str(path_b), {"A": np.array([1.0, 2.0])}, overwrite=True)
    result = _run_cli("diff", str(path_a), str(path_b))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "type" in result.stderr
    assert "Traceback" not in result.stderr
    assert "min():" not in result.stderr


def test_diff_missing_file_is_io_error(tmp_path):
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("diff", str(tmp_path / "nope.fits"), str(path))
    assert result.returncode == 3, result.stdout + result.stderr


def test_copy_same_path_refused_message_and_exit(tmp_path):
    """Same-path refusal contract (reference for compress/convert/decompress)."""
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("copy", str(path), str(path))
    assert result.returncode == 2
    assert result.stderr.strip() == f"refusing same-path rewrite in place: {path}"
    result = _run_cli("copy", str(path), "-o", str(path))
    assert result.returncode == 2
    assert result.stderr.strip() == f"refusing same-path rewrite in place: {path}"


def test_copy_is_byte_identical(tmp_path):
    """copy-is-binary: copy is shutil.copy2, not an HDU rewrite."""
    src = tmp_path / "img.fits"
    dst = tmp_path / "out.fits"
    torchfits.write(
        str(src), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True
    )
    result = _run_cli("copy", str(src), str(dst))
    assert result.returncode == 0, result.stderr
    assert dst.read_bytes() == src.read_bytes()


def test_cutout_inverted_box_is_usage_error(tmp_path):
    src = tmp_path / "img.fits"
    out = tmp_path / "cut.fits"
    torchfits.write(str(src), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True)
    result = _run_cli("cutout", str(src), "-o", str(out), "--box", "3,3,1,1")
    assert result.returncode == 2, result.stdout + result.stderr
    assert not out.exists()


def test_cutout_empty_box_is_usage_error(tmp_path):
    src = tmp_path / "img.fits"
    out = tmp_path / "cut.fits"
    torchfits.write(str(src), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True)
    result = _run_cli("cutout", str(src), "-o", str(out), "--box", "2,2,2,2")
    assert result.returncode == 2, result.stdout + result.stderr
    assert not out.exists()


def test_cutout_negative_origin_is_usage_error(tmp_path):
    src = tmp_path / "img.fits"
    out = tmp_path / "cut.fits"
    torchfits.write(str(src), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True)
    result = _run_cli("cutout", str(src), "-o", str(out), "--box=-2,-2,2,2")
    assert result.returncode == 2, result.stdout + result.stderr
    assert not out.exists()


def test_cutout_box_extends_past_image_clamps(tmp_path):
    """Boxes reaching past the image edge clamp to it (slice-style semantics)."""
    src = tmp_path / "img.fits"
    out = tmp_path / "cut.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(src), data, overwrite=True)
    result = _run_cli("cutout", str(src), "-o", str(out), "--box", "0,0,10,10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert torch.equal(torchfits.read_tensor(str(out)), data)
