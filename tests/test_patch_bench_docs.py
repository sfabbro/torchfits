"""Host-scorecard platform labels must come from the data, not the run-id tag."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_bench_docs as pdoc  # noqa: E402


def _run_dir(base: Path, name: str, *, host: str, device: str | None) -> Path:
    """Build a minimal run dir (results.csv) under the test's tmp_path."""
    d = base / name
    d.mkdir()
    if device is None:
        md = ""
        if host == "macbook":
            md = '{"device": "mps"}'
    else:
        md = (
            '{"file_type": "compressed", '
            f'"io_transport": "disk->cpu->{device}", '
            f'"device": "{device}"}}'
        )
    with (d / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["host", "metadata"])
        w.writeheader()
        w.writerow({"host": host, "metadata": md})
    return d


def test_cuda_run_labels_cuda(tmp_path: Path) -> None:
    """A run whose rows ran on cuda must stay CUDA even if named *_cpu_*."""
    d = _run_dir(
        tmp_path, "exhaustive_cpu_20260801_000000", host="node-cuda", device="cuda"
    )
    assert pdoc._host_label(d / "results.csv") == "Linux x86_64 / CUDA"


def test_mps_named_run_without_device_is_cpu(tmp_path: Path) -> None:
    """An exhaustive_mps_* tag alone must not imply MPS: local bench script
    uses that tag on any platform, and a run with no device observed is CPU."""
    d = _run_dir(
        tmp_path, "exhaustive_mps_20260801_000000", host="flexterm", device=None
    )
    assert pdoc._host_label(d / "results.csv") == "Linux x86_64 / CPU"


def test_cpu_host_token_wins_over_mps_tag(tmp_path: Path) -> None:
    """Host column token (CANFAR container names) beats the run-id tag."""
    d = _run_dir(
        tmp_path,
        "exhaustive_mps_20260801_000000",
        host="torchfits-gpu-exhaustive-cpu-20260801_000000",
        device=None,
    )
    assert pdoc._host_label(d / "results.csv") == "Linux x86_64 / CPU"


def test_real_mps_device_still_labels_mps(tmp_path: Path) -> None:
    """A genuine MPS run (device in metadata) keeps the macOS label even when
    the run-id tag does not mention mps."""
    d = _run_dir(
        tmp_path, "exhaustive_cpu_20260801_000000", host="macbook", device="mps"
    )
    assert pdoc._host_label(d / "results.csv") == "macOS arm64 / MPS"


def test_highlights_follow_csv_case_ids(tmp_path: Path) -> None:
    """Image rows use one colon; a dtype_fair sample must not become the time."""
    import render_bench_highlights as highlights

    fields = [
        "case_id",
        "library",
        "method",
        "status",
        "comparable",
        "mmap_target",
        "time_s",
    ]
    rows = [
        ("large_float32_2d:read_full", "torchfits", "torchfits", "True", "on", "0.010"),
        (
            "large_float32_2d:read_full",
            "torchfits",
            "torchfits",
            "True",
            "off",
            "0.050",
        ),
        (
            "large_float32_2d::read_full_gpu",
            "torchfits",
            "torchfits_device",
            "True",
            "on",
            "0.020",
        ),
        (
            "large_float32_2d::read_full_gpu",
            "torchfits",
            "torchfits_dtype_fair_device",
            "False",
            "on",
            "0.001",
        ),
        (
            "large_float32_2d::read_full_gpu",
            "torchfits",
            "torchfits_specialized_device",
            "True",
            "on",
            "0.030",
        ),
        (
            "repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100",
            "torchfits",
            "torchfits",
            "True",
            "n/a",
            "0.040",
        ),
    ]
    with (tmp_path / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case_id, library, method, comparable, mmap_target, time_s in rows:
            writer.writerow(
                {
                    "case_id": case_id,
                    "library": library,
                    "method": method,
                    "status": "OK",
                    "comparable": comparable,
                    "mmap_target": mmap_target,
                    "time_s": time_s,
                }
            )
    text = highlights.render_highlights(tmp_path)
    assert "Large tensor read (Float32 2D, 16.0 MB)" in text
    assert "10.00 ms" in text
    assert "50.00 ms" not in text
    assert "20.00 ms" in text
    assert "30.00 ms" in text
    assert "1.00 ms" not in text
    assert "Repeated cutouts (50x 100x100)" in text
    repeated = next(
        line
        for line in text.splitlines()
        if line.startswith("| Repeated cutouts (50x 100x100) |")
    )
    assert "| — |" in repeated


def test_full_table_joins_cfitsio_on_image_case_stem(tmp_path: Path) -> None:
    """Image case ids are ``name:op``; the direct bench stores ``name`` plus its op."""
    import render_full_benchmarks_table as full

    fields = [
        "domain",
        "case_id",
        "operation",
        "status",
        "comparable",
        "library",
        "method",
        "time_s",
        "size_mb",
        "mmap_target",
        "metadata",
    ]
    image = {
        "domain": "fits",
        "case_id": "large_float32_2d:read_full",
        "operation": "read_full",
        "status": "OK",
        "comparable": "True",
        "library": "torchfits",
        "method": "torchfits",
        "time_s": "0.010",
        "size_mb": "16",
        "mmap_target": "on",
        "metadata": "{}",
    }
    table = {
        "domain": "fitstable",
        "case_id": "mixed_100000::read_full",
        "operation": "read_full",
        "status": "OK",
        "comparable": "True",
        "library": "torchfits",
        "method": "torchfits",
        "time_s": "0.004",
        "size_mb": "1",
        "mmap_target": "on",
        "metadata": "{}",
    }
    with (tmp_path / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(image)
    with (tmp_path / "fitstable_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(table)
    with (tmp_path / "cfitsio_direct.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["case_id", "operation", "status", "time_s"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "large_float32_2d:read_full",
                "operation": "read_full",
                "status": "OK",
                "time_s": "9",
            }
        )
        writer.writerow(
            {
                "case_id": "large_float32_2d",
                "operation": "read_full",
                "status": "OK",
                "time_s": "0.000123",
            }
        )
        writer.writerow(
            {
                "case_id": "mixed_100000",
                "operation": "table_read",
                "status": "OK",
                "time_s": "0.002",
            }
        )
    text = full.render_full_table(tmp_path)
    image_line = next(
        line
        for line in text.splitlines()
        if line.startswith("| tensor | large_float32_2d | read_full |")
    )
    table_line = next(
        line
        for line in text.splitlines()
        if line.startswith("| table | mixed_100000 | read_full |")
    )
    assert "123.0 μs" in image_line
    assert "9.000 s" not in image_line
    assert "2.00 ms" in table_line
