#!/usr/bin/env python3
"""Parse benchmark CSV files and render the complete, un-cherrypicked benchmarks table."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def format_time(val: float | None) -> str:
    if val is None:
        return "—"
    if val < 0.001:
        return f"{val * 1e6:.1f} μs"
    elif val < 1.0:
        return f"{val * 1000:.2f} ms"
    else:
        return f"{val:.3f} s"


def render_full_table(results_dir: Path) -> str:
    results_rows = load_csv(results_dir / "results.csv")
    fitstable_rows = load_csv(results_dir / "fitstable_results.csv")

    all_rows = results_rows + fitstable_rows

    # Group by (domain, case_id)
    # Each group is a key mapping to a dictionary of methods
    grouped = defaultdict(dict)

    for row in all_rows:
        status = row.get("status")
        if status != "OK":
            continue

        domain = row.get("domain", "")
        case_id = row.get("case_id", "")
        operation = row.get("operation", "")
        size_mb_str = row.get("size_mb", "0.0")
        try:
            size_mb = float(size_mb_str)
        except ValueError:
            size_mb = 0.0

        # Parse metadata for device
        device = "CPU"
        meta_str = row.get("metadata", "{}")
        if meta_str:
            try:
                # Replace single quotes with double quotes for valid JSON
                meta = json.loads(meta_str.replace("'", '"'))
                if "device" in meta:
                    device = str(meta["device"]).upper()
            except Exception:
                pass

        key = (domain, case_id, operation, size_mb, device)

        lib = row.get("library")
        method = row.get("method")
        try:
            time_s = float(row.get("time_s", "0.0"))
        except ValueError:
            continue

        if lib == "torchfits":
            if "specialized" in method:
                grouped[key]["tf_pers"] = time_s
            else:
                grouped[key]["tf"] = time_s
        elif lib == "astropy":
            grouped[key]["astropy"] = time_s
        elif lib == "fitsio":
            grouped[key]["fitsio"] = time_s

    # Sort key order: domain desc (fits first, fitstable next), then case_id
    sorted_keys = sorted(
        grouped.keys(), key=lambda k: (k[0] != "fits", k[4] != "CPU", k[1])
    )

    lines = [
        "## Exhaustive Benchmark Results",
        "",
        "The complete, un-cherrypicked list of all measured benchmark configurations.",
        "",
        "| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key in sorted_keys:
        domain, case_id, operation, size_mb, device = key
        times = grouped[key]

        tf = times.get("tf")
        tf_pers = times.get("tf_pers")
        astropy = times.get("astropy")
        fitsio = times.get("fitsio")

        # Determine reference/fastest torchfits time
        tf_list = [t for t in (tf, tf_pers) if t is not None]
        best_tf = min(tf_list) if tf_list else None

        astropy_win = "—"
        fitsio_win = "—"

        if best_tf is not None:
            if astropy is not None:
                astropy_win = f"{astropy / best_tf:.2f}x"
            if fitsio is not None:
                fitsio_win = f"{fitsio / best_tf:.2f}x"

        # Format display strings
        tf_str = format_time(tf)
        tf_pers_str = format_time(tf_pers)
        astropy_str = format_time(astropy)
        fitsio_str = format_time(fitsio)

        # Clean case name (remove suffix operation)
        case_name = case_id.split("::")[0]

        # Size representation
        size_str = f"{size_mb:.2f} MB" if size_mb > 0.05 else f"{size_mb * 1024:.1f} KB"

        lines.append(
            f"| {domain} | {case_name} | {operation} | {size_str} | {device} | **{tf_str}** | {tf_pers_str} | {astropy_str} | {fitsio_str} | **{astropy_win}** | **{fitsio_win}** |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_full_table(args.results_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
