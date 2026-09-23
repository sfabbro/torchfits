"""CLI regression tests for ``torchfits stats`` / ``torchfits verify`` (R7-B).

Covers the JSON non-finite contract (NaN/±inf serialize as ``null``), empty
region stats, per-command exit codes vs the docs/cli.md exit table, and I/O
error mapping for truncated inputs.
"""

from __future__ import annotations

import json
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


def _write_empty_image(path) -> None:
    torchfits.write(str(path), torch.zeros((0, 4)), overwrite=True)


def test_stats_empty_region_json_yields_null(tmp_path):
    """Empty region stats must parse as JSON with None values (decision 3)."""
    path = tmp_path / "empty.fits"
    _write_empty_image(path)
    result = _run_cli("stats", str(path), "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    for key in ("min", "max", "mean", "std", "median"):
        assert payload[0][key] is None


def test_stats_empty_region_jsonl_yields_null(tmp_path):
    path = tmp_path / "empty.fits"
    _write_empty_image(path)
    result = _run_cli("stats", str(path), "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    for key in ("min", "max", "mean", "std", "median"):
        assert record[key] is None


def test_stats_nonfinite_floats_serialize_as_null(tmp_path):
    """NaN/±inf statistics serialize as null: JSON has no NaN/Infinity."""
    path = tmp_path / "nan.fits"
    torchfits.write(
        str(path),
        torch.tensor([[float("nan"), 1.0], [2.0, float("inf")]]),
        overwrite=True,
    )
    for fmt in ("json", "jsonl"):
        result = _run_cli("stats", str(path), "-f", fmt)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)  # must not contain bare NaN tokens
        record = payload[0] if fmt == "json" else payload
        for key in ("min", "max", "mean", "std", "median"):
            assert record[key] is None


def test_table_json_preview_nonfinite_null(tmp_path):
    """table -f json/jsonl previews serialize non-finite floats as null."""
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, np.nan, np.inf])}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "3", "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    preview = payload[0]["preview"]
    assert preview[0]["A"] == 1.0
    assert preview[1]["A"] is None
    assert preview[2]["A"] is None

    result = _run_cli("table", str(path), "-n", "3", "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["preview"][1]["A"] is None
    assert record["preview"][2]["A"] is None


def test_table_empty_preview_ok(tmp_path):
    """Zero-row tables preview as an empty list, exit 0."""
    path = tmp_path / "emptytbl.fits"
    tf_table.write(str(path), {"A": np.array([], dtype=np.float64)}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "5", "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["nrows"] == 0
    assert payload[0]["preview"] == []


def test_table_text_preview_is_valid_json(tmp_path):
    """The text-mode preview block is json.dumps output and must be valid JSON."""
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, np.nan, np.inf])}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "3")
    assert result.returncode == 0, result.stderr
    marker = "preview:\n"
    assert marker in result.stdout
    preview = json.loads(result.stdout.split(marker, 1)[1])
    assert preview[0]["A"] == 1.0
    assert preview[1]["A"] is None
    assert preview[2]["A"] is None


def test_stats_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("stats", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_table_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, 2.0])}, overwrite=True)
    result = _run_cli("table", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_verify_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("verify", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_stats_truncated_image_is_io_error(tmp_path):
    """Unreadable image data maps to exit 3 (invalid FITS structure), not a crash."""
    path = tmp_path / "trunc.fits"
    full = tmp_path / "full.fits"
    torchfits.write(
        str(full), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True
    )
    raw = full.read_bytes()
    assert len(raw) > 2880
    path.write_bytes(raw[:-2880])
    result = _run_cli("stats", str(path))
    assert result.returncode == 3, result.stderr
    assert "Traceback" not in result.stderr


def test_verify_truncated_checksummed_is_io_error(tmp_path):
    full = tmp_path / "full.fits"
    torchfits.write(
        str(full), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True
    )
    torchfits.write_checksums(str(full), hdu=0)
    raw = full.read_bytes()
    path = tmp_path / "trunc.fits"
    path.write_bytes(raw[:-2880])
    result = _run_cli("verify", str(path))
    assert result.returncode == 3, result.stderr
    assert "Traceback" not in result.stderr


def test_verify_one_corrupt_file_exits_4_under_file_jobs(tmp_path):
    """Any failing HDU across the batch exits 4 even with -J fan-out."""
    good = tmp_path / "good.fits"
    bad = tmp_path / "bad.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(good), data, header={"FOO": 1}, overwrite=True)
    torchfits.write(str(bad), data, header={"FOO": 1}, overwrite=True)
    torchfits.write_checksums(str(good), hdu=0)
    torchfits.write_checksums(str(bad), hdu=0)
    # Corrupt a non-structural keyword (FOO card): the file stays openable but
    # the stored CHECKSUM no longer matches the header bytes.
    raw = bytearray(bad.read_bytes())
    idx = raw.find(b"FOO     =")
    assert idx != -1
    one = raw[idx : idx + 80].find(b"1")
    assert one != -1
    raw[idx + one : idx + one + 1] = b"2"
    bad.write_bytes(raw)
    result = _run_cli("verify", str(good), str(bad), "-J", "2", "-f", "jsonl")
    assert result.returncode == 4, result.stderr
