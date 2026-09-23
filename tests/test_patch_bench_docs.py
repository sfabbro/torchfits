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
