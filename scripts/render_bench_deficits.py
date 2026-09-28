#!/usr/bin/env python3
"""Render torchfits_deficits.csv into a markdown section for docs/benchmarks.md."""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path


def _domain_label(raw: str) -> str:
    d = (raw or "").strip().lower()
    if d in {"fits", "tensor", "image"}:
        return "tensor"
    if d in {"fitstable", "table", "dataframe"}:
        return "table"
    return raw or "-"


def _platform_label(host: str, metadata: str = "", case: str = "") -> str:
    """Human host label: OS / arch / accelerator — never raw hostname."""
    blob = f"{metadata} {case} {host}".lower()
    device = "CPU"
    if "cuda" in blob:
        device = "CUDA"
    elif "mps" in blob:
        device = "MPS"

    h = (host or "").lower()
    if "cuda" in h or device == "CUDA":
        return "Linux x86_64 / CUDA"
    if "mps" in h or device == "MPS":
        return "macOS arm64 / MPS"
    if "cpu" in h:
        return "Linux x86_64 / CPU"
    # Scorecard Mac hostname without device token in host field. Derive the
    # arch the same way the fallback below does; hardcoding arm64 published a
    # wrong architecture for any non-arm host matching the pattern.
    if h.startswith("nrc-") or "darwin" in h:
        return f"macOS {platform.machine()} / {device}"
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        return f"macOS {machine} / {device}"
    if system == "Linux":
        return f"Linux {machine} / {device}"
    return f"{system} {machine} / {device}"


def _fmt_time(raw: str) -> str:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return raw or "-"
    if v < 0.001:
        return f"{v * 1e6:.1f} μs"
    if v < 1.0:
        return f"{v * 1000:.2f} ms"
    return f"{v:.3f} s"


def _fmt_rss(raw: str) -> str:
    try:
        return f"{float(raw):.1f}"
    except (TypeError, ValueError):
        return raw or "-"


def _fmt_lag(raw: str) -> str:
    try:
        return f"{float(raw):.2f}×"
    except (TypeError, ValueError):
        return raw or "-"


def render_deficits(csv_path: Path, *, max_rows: int = 40) -> str:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    lines = [
        "## Performance deficits",
        "",
        "Cases where torchfits is **not** first in its comparison family "
        "(CPU and GPU). GPU lags may reflect software or hardware limits — "
        "they are listed, not hidden.",
        "",
    ]
    if not rows:
        lines.append("_No deficits in this run — torchfits won every comparable case._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |",
            "|---|---|---|---|---:|---:|---|---:|",
        ]
    )
    for row in rows[:max_rows]:
        case = row.get("case_label") or row.get("case_id") or "-"
        mmap = row.get("mmap_target") or "-"
        tf = _fmt_time(row.get("torchfits_time_s") or "")
        tf_rss = _fmt_rss(row.get("torchfits_peak_rss_mb") or "")
        winner = f"{row.get('best_library', '-')}/{row.get('best_method', '-')}"
        lag = _fmt_lag(row.get("lag_ratio") or "")
        plat = _platform_label(
            row.get("host") or "",
            row.get("metadata") or "",
            case=str(case),
        )
        domain = _domain_label(row.get("domain", "-"))
        lines.append(
            f"| {plat} | {domain} | {case} | {mmap} | {tf} | "
            f"{tf_rss} | {winner} | {lag} |"
        )
    if len(rows) > max_rows:
        lines.append("")
        lines.append(
            f"_…and {len(rows) - max_rows} more rows in `torchfits_deficits.csv`._"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=40)
    args = parser.parse_args()
    print(render_deficits(args.csv, max_rows=args.max_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
