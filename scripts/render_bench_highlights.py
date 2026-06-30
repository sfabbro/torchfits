#!/usr/bin/env python3
"""Parse benchmark CSV files and render a beautiful Markdown highlights table."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

# Target cases to highlight
TARGET_CASES = [
    # Domain, File, Case ID, Operation, Device, Label
    (
        "fits",
        "results.csv",
        "large_float32_2d::read_full",
        "read_full",
        "CPU",
        "Large Image Read (Float32 2D, 16.0 MB)",
    ),
    (
        "fits",
        "results.csv",
        "large_float32_2d::read_full_gpu",
        "read_full",
        "CUDA",
        "Large Image Read (Float32 2D @ CUDA)",
    ),
    (
        "fits",
        "results.csv",
        "compressed_rice_1::read_full",
        "read_full",
        "CPU",
        "Compressed Image Read (Rice, 1.1 MB)",
    ),
    (
        "fits",
        "results.csv",
        "compressed_rice_1::read_full_gpu",
        "read_full",
        "CUDA",
        "Compressed Image Read (Rice @ CUDA)",
    ),
    (
        "fits",
        "results.csv",
        "repeated_cutouts_50x_100x100::repeated_cutouts_50x_100x100",
        "repeated_cutouts_50x_100x100",
        "CPU",
        "Repeated Cutouts (50x 100x100)",
    ),
    (
        "fits",
        "results.csv",
        "repeated_cutouts_50x_100x100_gpu::repeated_cutouts_50x_100x100",
        "repeated_cutouts_50x_100x100",
        "CUDA",
        "Repeated Cutouts (50x 100x100 @ CUDA)",
    ),
    (
        "fitstable",
        "fitstable_results.csv",
        "mixed_100000::read_full",
        "read_full",
        "CPU",
        "Table Read (100k rows, 8 cols, mixed)",
    ),
    (
        "fitstable",
        "fitstable_results.csv",
        "varlen_100000::read_full",
        "read_full",
        "CPU",
        "Varlen Table Read (100k rows, 3 cols)",
    ),
]


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def format_time(val: str | None) -> str:
    if not val:
        return "—"
    try:
        t = float(val)
        if t < 0.001:
            return f"{t * 1e6:.1f} μs"
        elif t < 1.0:
            return f"{t * 1000:.2f} ms"
        else:
            return f"{t:.3f} s"
    except ValueError:
        return val


def render_highlights(results_dir: Path) -> str:
    results_rows = load_csv(results_dir / "results.csv")
    fitstable_rows = load_csv(results_dir / "fitstable_results.csv")

    all_rows = results_rows + fitstable_rows

    lines = [
        "## Performance Highlights",
        "",
        "The following table showcases median wall-clock execution times of key representative FITS benchmarks.",
        "In almost all core I/O paths, `torchfits` is significantly faster than standard astronomical tools, with extra performance wins from persistent handle caches and direct-to-device transfers.",
        "",
        "| Benchmark Case | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for domain, filename, case_id, op, device, label in TARGET_CASES:
        # Filter rows matching this case_id
        case_rows = [r for r in all_rows if r.get("case_id") == case_id]
        if not case_rows:
            continue

        # Find times for each library/method combination
        tf_time = None
        tf_pers_time = None
        astropy_time = None
        fitsio_time = None

        for row in case_rows:
            status = row.get("status")
            if status != "OK":
                continue
            lib = row.get("library")
            method = row.get("method")
            time_s = row.get("time_s")

            if lib == "torchfits":
                if "specialized" in method:
                    tf_pers_time = time_s
                else:
                    tf_time = time_s
            elif lib == "astropy":
                astropy_time = time_s
            elif lib == "fitsio":
                fitsio_time = time_s

        # Speedups
        astropy_win = "—"
        fitsio_win = "—"

        try:
            # We use the fastest torchfits time as the reference
            best_tf = min(
                [float(t) for t in (tf_time, tf_pers_time) if t is not None],
                default=None,
            )
            if best_tf is not None:
                if astropy_time:
                    astropy_win = f"{float(astropy_time) / best_tf:.2f}x"
                if fitsio_time:
                    fitsio_win = f"{float(fitsio_time) / best_tf:.2f}x"
        except (ValueError, TypeError):
            pass

        tf_str = format_time(tf_time)
        tf_pers_str = format_time(tf_pers_time)
        astropy_str = format_time(astropy_time)
        fitsio_str = format_time(fitsio_time)

        lines.append(
            f"| {label} | {device} | **{tf_str}** | {tf_str if tf_pers_str == '—' else tf_pers_str} | {astropy_str} | {fitsio_str} | **{astropy_win}** | **{fitsio_win}** |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_highlights(args.results_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
