#!/usr/bin/env python3
"""Parse benchmark CSV files and render the complete, un-cherrypicked benchmarks table."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


_TABLE_CASE_PREFIXES = (
    "narrow_",
    "mixed_",
    "wide_",
    "varlen_",
    "typed_",
    "ascii_",
    "compressed_table",
)


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


def _cfitsio_domain(case_id: str, operation: str) -> str:
    if any(case_id.startswith(p) for p in _TABLE_CASE_PREFIXES):
        return "fitstable"
    if operation in {"projection", "row_slice", "scan_count", "predicate_filter"}:
        return "fitstable"
    return "fits"


def _case_stem(case_id: str) -> str:
    """Drop the operation suffix. Tables use ``::``; image rows use ``:``."""
    if "::" in case_id:
        return case_id.split("::", 1)[0]
    if ":" in case_id:
        return case_id.split(":", 1)[0]
    return case_id


def _cfitsio_operation(domain: str, operation: str) -> str:
    """Scorecard operation name → op column in ``cfitsio_direct.csv``."""
    if domain == "fitstable":
        return {
            "read_full": "table_read",
            "projection": "table_proj",
            "row_slice": "table_slice",
            "scan_count": "table_scan",
            "predicate_filter": "table_pred",
        }.get(operation, operation)
    if operation == "cutout_100x100":
        return "cutout"
    if operation == "random_ext_full_reads_200":
        return "random_ext"
    if operation.startswith("repeated_cutouts_"):
        return "cutout_rep"
    return operation


def _load_cfitsio_times(results_dir: Path) -> dict[tuple[str, str, str], float]:
    """Map (domain, case_id, operation) -> median cfitsio_direct time."""
    csv_path = results_dir / "cfitsio_direct.csv"
    if not csv_path.is_file():
        return {}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in load_csv(csv_path):
        if row.get("status") != "OK":
            continue
        case_id = row.get("case_id", "")
        operation = row.get("operation", "")
        try:
            t = float(row.get("time_s", "0"))
        except ValueError:
            continue
        domain = _cfitsio_domain(case_id, operation)
        grouped[(domain, case_id, operation)].append(t)
    return {k: min(v) for k, v in grouped.items()}


def render_full_table(results_dir: Path) -> str:
    results_rows = load_csv(results_dir / "results.csv")
    fitstable_rows = load_csv(results_dir / "fitstable_results.csv")
    cfitsio_times = _load_cfitsio_times(results_dir)

    all_rows = results_rows + fitstable_rows

    # Group by (domain, case_id)
    # Each group is a key mapping to a dictionary of methods
    grouped = defaultdict(dict)

    for row in all_rows:
        status = row.get("status")
        if status != "OK":
            continue
        comparable = row.get("comparable")
        if comparable in (False, "False", "false", "0", 0):
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

        mmap_target = row.get("mmap_target") or "-"
        key = (domain, case_id, operation, size_mb, device, mmap_target)

        lib = row.get("library")
        method = str(row.get("method") or "")
        try:
            time_s = float(row.get("time_s", "0.0"))
        except ValueError:
            continue

        # Exact method identity — never substring (dtype_fair must not overwrite default).
        if method in {"torchfits", "torchfits_device"}:
            grouped[key]["tf"] = time_s
        elif method in {"torchfits_specialized", "torchfits_specialized_device"}:
            grouped[key]["tf_pers"] = time_s
        elif lib == "astropy" and method in {
            "astropy",
            "astropy_torch",
            "astropy_torch_device",
        }:
            grouped[key]["astropy"] = time_s
        elif lib == "fitsio" and method in {
            "fitsio",
            "fitsio_torch",
            "fitsio_torch_device",
        }:
            grouped[key]["fitsio"] = time_s

        case_name = _case_stem(case_id)
        cf_key = (domain, case_name, _cfitsio_operation(domain, operation))
        if cf_key in cfitsio_times:
            grouped[key]["cfitsio"] = cfitsio_times[cf_key]

    # Sort key order: domain desc (fits first, fitstable next), then case_id
    sorted_keys = sorted(
        grouped.keys(),
        key=lambda k: (k[0] != "fits", k[4] != "CPU", k[5], k[1]),
    )

    lines = [
        "## Exhaustive Benchmark Results",
        "",
        "The complete, un-cherrypicked list of all measured configurations. "
        "Empty cells mean that method was not run for the case (for example "
        "`torchfits_specialized` is only used for open-once / subset-reader paths). "
        "Domain `tensor` = IMAGE HDU payloads (1D–4D); `table` = binary/ASCII tables.",
        "",
        "| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | cfitsio (direct) | Speedup vs Astropy | Speedup vs fitsio |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in sorted_keys:
        domain, case_id, operation, size_mb, device, mmap_target = key
        times = grouped[key]

        tf = times.get("tf")
        tf_pers = times.get("tf_pers")
        astropy = times.get("astropy")
        fitsio = times.get("fitsio")
        cfitsio = times.get("cfitsio")

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
        cfitsio_str = format_time(cfitsio)

        # Clean case name (remove suffix operation)
        case_name = _case_stem(case_id)

        # Size representation
        size_str = f"{size_mb:.2f} MB" if size_mb > 0.05 else f"{size_mb * 1024:.1f} KB"

        domain_label = (
            "tensor"
            if domain == "fits"
            else ("table" if domain == "fitstable" else domain)
        )
        lines.append(
            f"| {domain_label} | {case_name} | {operation} | {size_str} | {device} | {mmap_target} | **{tf_str}** | {tf_pers_str} | {astropy_str} | {fitsio_str} | {cfitsio_str} | **{astropy_win}** | **{fitsio_win}** |"
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
