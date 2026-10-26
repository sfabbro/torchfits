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


def test_arith_file_output_must_not_clobber_operand_a(tmp_path):
    source = tmp_path / "source.fits"
    torchfits.write(str(source), torch.ones(2, 2), overwrite=True)
    before = source.read_bytes()
    result = _run_cli(
        "arith", str(source), "--op", "add", "--value", "1", "-o", str(source)
    )
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert source.read_bytes() == before


def test_arith_file_output_must_not_clobber_operand_b(tmp_path):
    left = tmp_path / "left.fits"
    right = tmp_path / "right.fits"
    torchfits.write(str(left), torch.ones(2, 2), overwrite=True)
    torchfits.write(str(right), torch.full((2, 2), 2.0), overwrite=True)
    before = right.read_bytes()
    result = _run_cli("arith", str(left), str(right), "--op", "add", "-o", str(right))
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert right.read_bytes() == before


def test_arith_batch_preflights_every_generated_destination(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    left = source_dir / "shared.fits"
    other = source_dir / "other.fits"
    operand = output_dir / "shared.fits"
    torchfits.write(str(left), torch.ones(2, 2), overwrite=True)
    torchfits.write(str(other), torch.ones(2, 2), overwrite=True)
    torchfits.write(str(operand), torch.full((2, 2), 3.0), overwrite=True)
    before = operand.read_bytes()
    result = _run_cli(
        "arith",
        str(left),
        str(other),
        "--operand2",
        str(operand),
        "--op",
        "add",
        "--out-dir",
        str(output_dir),
        "-J",
        "2",
    )
    assert result.returncode == 2, result.stderr
    assert REFUSAL in result.stderr
    assert operand.read_bytes() == before


def _case_insensitive(directory) -> bool:
    probe = directory / "CaseProbe"
    probe.write_text("x")
    insensitive = (directory / "caseprobe").exists()
    probe.unlink()
    return insensitive


def test_arith_case_colliding_file_outputs_follow_filesystem(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    upper = left_dir / "IMAGE.fits"
    lower = right_dir / "image.fits"
    torchfits.write(upper, torch.ones(2, 2))
    torchfits.write(lower, torch.ones(2, 2))
    output = tmp_path / "output"
    result = _run_cli(
        "arith",
        str(upper),
        str(lower),
        "--op",
        "add",
        "--value",
        "1",
        "--out-dir",
        str(output),
        "-J",
        "1",
    )
    if _case_insensitive(tmp_path):
        assert result.returncode == 2, result.stderr
        assert "duplicate basename" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (output / upper.name).exists()
        assert (output / lower.name).exists()


def test_arith_case_colliding_split_outputs_follow_filesystem(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    upper = left_dir / "IMAGE.fits"
    lower = right_dir / "image.fits"
    torchfits.write(upper, torch.ones(2, 2))
    torchfits.write(lower, torch.ones(2, 2))
    output = tmp_path / "output"
    result = _run_cli(
        "arith",
        str(upper),
        str(lower),
        "--op",
        "add",
        "--value",
        "1",
        "--split",
        "hdu",
        "--out-dir",
        str(output),
        "-J",
        "1",
    )
    if _case_insensitive(tmp_path):
        assert result.returncode == 2, result.stderr
        assert "duplicate basename" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (output / "IMAGE_hdu00.fits").exists()
        assert (output / "image_hdu00.fits").exists()


def test_case_variant_same_path_refusal_follows_filesystem(tmp_path):
    source = tmp_path / "IMAGE.fits"
    destination = tmp_path / "image.fits"
    torchfits.write(source, torch.ones(2, 2))
    result = _run_cli(
        "arith", str(source), "--op", "add", "--value", "1", "-o", str(destination)
    )
    if _case_insensitive(tmp_path):
        assert result.returncode == 2, result.stderr
        assert REFUSAL in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert destination.exists()


def test_canonically_equivalent_outputs_follow_filesystem(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    composed = first_dir / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.fits"
    decomposed = second_dir / "cafe\N{COMBINING ACUTE ACCENT}.fits"
    torchfits.write(composed, torch.ones(2, 2))
    torchfits.write(decomposed, torch.ones(2, 2))
    output = tmp_path / "output"
    result = _run_cli(
        "arith",
        str(composed),
        str(decomposed),
        "--op",
        "add",
        "--value",
        "1",
        "--out-dir",
        str(output),
    )
    if sys.platform == "darwin":
        assert result.returncode == 2, result.stderr
        assert "duplicate basename" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
