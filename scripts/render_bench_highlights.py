#!/usr/bin/env python3
"""Parse benchmark CSV files and render a beautiful Markdown highlights table."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

# case_id values are the ones bench_fits_io / bench_gpu_transports / bench_fitstable
# actually write. Image rows use a single colon (`name:operation`). GPU full
# reads embed `::` inside the id (`name::read_full_gpu`). Cutout ids do not
# append the operation a second time.
# Tuple: domain, file, case_id, operation, device, mmap_target, label.
TARGET_CASES = [
    (
        "fits",
        "results.csv",
        "large_float32_2d:read_full",
        "read_full",
        "CPU",
        "on",
        "Large tensor read (Float32 2D, 16.0 MB)",
    ),
    (
        "fits",
        "results.csv",
        "large_float32_2d::read_full_gpu",
        "read_full",
        "CUDA",
        "on",
        "Large tensor read (Float32 2D @ CUDA)",
    ),
    (
        "fits",
        "results.csv",
        "compressed_rice_1:read_full",
        "read_full",
        "CPU",
        "on",
        "Compressed tensor read (Rice, 1.1 MB)",
    ),
    (
        "fits",
        "results.csv",
        "compressed_rice_1::read_full_gpu",
        "read_full",
        "CUDA",
        "on",
        "Compressed tensor read (Rice @ CUDA)",
    ),
    (
        "fits",
        "results.csv",
        "repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100",
        "repeated_cutouts_50x_100x100",
        "CPU",
        "n/a",
        "Repeated cutouts (50x 100x100)",
    ),
    (
        "fits",
        "results.csv",
        "repeated_cutouts_50x_100x100_gpu",
        "repeated_cutouts_50x_100x100",
        "CUDA",
        "n/a",
        "Repeated cutouts (50x 100x100 @ CUDA)",
    ),
    (
        "fitstable",
        "fitstable_results.csv",
        "mixed_100000::read_full",
        "read_full",
        "CPU",
        "on",
        "Table read (100k rows, 8 cols, mixed)",
    ),
    (
        "fitstable",
        "fitstable_results.csv",
        "varlen_100000::read_full",
        "read_full",
        "CPU",
        "on",
        "Varlen table read (100k rows, 3 cols)",
    ),
]

_SMART = {"torchfits", "torchfits_device"}
_SPECIALIZED = {"torchfits_specialized", "torchfits_specialized_device"}


def _is_comparable(row: dict[str, str]) -> bool:
    return str(row.get("comparable", "True")).strip().lower() not in {"false", "0"}


def _rows_for_case(
    rows: list[dict[str, str]], case_id: str, mmap_target: str
) -> list[dict[str, str]]:
    """Keep one mmap mode. A matrix CSV must not let the last row win."""
    ok = [
        row
        for row in rows
        if row.get("case_id") == case_id
        and row.get("status") == "OK"
        and _is_comparable(row)
    ]
    preferred = [
        row for row in ok if (row.get("mmap_target") or "") in {mmap_target, ""}
    ]
    if preferred:
        return preferred
    targets = {row.get("mmap_target") for row in ok}
    if len(targets) == 1:
        return ok
    return []


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
        "The following table showcases median wall-clock times for key FITS "
        "tensor and table cases. The **specialized** column is "
        "`torchfits_specialized` (open-once / subset-reader paths); it is "
        "empty when that path was not measured.",
        "",
        "| Benchmark Case | Device | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _domain, _filename, case_id, _op, device, mmap_target, label in TARGET_CASES:
        case_rows = _rows_for_case(all_rows, case_id, mmap_target)
        if not case_rows:
            continue

        # Find times for each library/method combination
        tf_time = None
        tf_pers_time = None
        astropy_time = None
        fitsio_time = None

        for row in case_rows:
            lib = row.get("library")
            method = row.get("method") or ""
            time_s = row.get("time_s")

            if lib == "torchfits" and method in _SMART:
                tf_time = time_s
            elif lib == "torchfits" and method in _SPECIALIZED:
                tf_pers_time = time_s
            elif lib == "astropy" and "specialized" not in method:
                astropy_time = time_s
            elif lib == "fitsio" and "specialized" not in method:
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
            f"| {label} | {device} | **{tf_str}** | {tf_pers_str} | {astropy_str} | {fitsio_str} | **{astropy_win}** | **{fitsio_win}** |"
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
