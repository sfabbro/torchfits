#!/usr/bin/env python3
"""Aggregate and compare CANFAR matrix benchmark results across Python and PyTorch versions.

Reads benchmark results from /arc/home/$USER/torchfits-gpu-bench/ and
benchmarks_results/, computes geometric-mean execution times and throughputs across
the (Python x PyTorch x Device) matrix, determines the champion configurations for
CPU and CUDA, and calculates the relative performance differences.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_result_dirs(search_paths: list[Path]) -> list[Path]:
    """Find all benchmark run directories containing results.csv."""
    found: list[Path] = []
    seen = set()
    for root in search_paths:
        if not root.exists():
            continue
        # Search direct subdirectories and nested benchmarks_results
        for p in root.glob("**/results.csv"):
            run_dir = p.parent
            if run_dir not in seen:
                seen.add(run_dir)
                found.append(run_dir)
    return sorted(found)


def parse_manifest_or_name(run_dir: Path) -> dict[str, str]:
    """Extract metadata (python, torch, cuda, cu_flavor, run_id) from manifest or directory name."""
    meta: dict[str, str] = {
        "run_id": run_dir.name,
        "python": "",
        "torch": "",
        "cuda": "",
        "cu_flavor": "",
    }

    # Check for manifest.txt in run_dir or parent
    manifest_files = [run_dir / "manifest.txt", run_dir.parent / "manifest.txt"]
    for mf in manifest_files:
        if mf.is_file():
            for line in mf.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k == "TORCHFITS_BENCH_PYTHON":
                        meta["python"] = v
                    elif k == "TORCHFITS_BENCH_TORCH":
                        meta["torch"] = v
                    elif k == "TORCHFITS_BENCH_CUDA":
                        meta["cuda"] = v
                    elif k == "TORCHFITS_BENCH_CU_FLAVOR":
                        meta["cu_flavor"] = v
                    elif k == "TORCHFITS_BENCH_RUN_ID":
                        meta["run_id"] = v
            break

    # If missing, parse from run_id tag (e.g. exhaustive_matrix_py311t213cu129_...)
    if not (meta["python"] and meta["torch"]):
        name = run_dir.name
        m = re.search(r"py(\d)(\d+)t(\d)(\d+)(cpu|cu\d+)?", name)
        if m:
            meta["python"] = f"{m.group(1)}.{m.group(2)}"
            meta["torch"] = f"{m.group(3)}.{m.group(4)}"
            flavor = m.group(5) or "cpu"
            meta["cuda"] = "1" if flavor.startswith("cu") else "0"
            meta["cu_flavor"] = flavor

    return meta


def load_run_results(run_dir: Path) -> dict[str, float]:
    """Load case_id -> median_time_ms from a results.csv file for torchfits."""
    csv_file = run_dir / "results.csv"
    if not csv_file.is_file():
        # Check subdirectories
        matches = list(run_dir.glob("**/results.csv"))
        if matches:
            csv_file = matches[0]
        else:
            return {}

    results: dict[str, float] = {}
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lib = row.get("library", "")
            status = row.get("status", "")
            if lib != "torchfits" or status != "OK":
                continue
            case_id = row.get("case_id") or row.get("name") or row.get("case")
            if not case_id:
                continue
            time_s = row.get("time_s")
            time_ms = row.get("median_ms") or row.get("time_ms")
            if time_s is not None:
                try:
                    results[case_id] = float(time_s) * 1000.0
                except ValueError:
                    pass
            elif time_ms is not None:
                try:
                    results[case_id] = float(time_ms)
                except ValueError:
                    pass
    return results


def geom_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    valid = [v for v in values if v > 0]
    if not valid:
        return 0.0
    return math.exp(sum(math.log(v) for v in valid) / len(valid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-dirs",
        nargs="+",
        default=[
            os.path.expanduser("~/torchfits-gpu-bench"),
            os.path.expanduser("/arc/home/sfabbro/torchfits-gpu-bench"),
            str(ROOT / "benchmarks_results"),
        ],
        help="Directories to search for benchmark runs",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON analysis")
    args = parser.parse_args()

    search_paths = [Path(p) for p in args.search_dirs]
    run_dirs = find_result_dirs(search_paths)

    if not run_dirs:
        print(f"No results.csv files found in: {search_paths}")
        return

    print(f"Found {len(run_dirs)} benchmark runs.")

    grid: dict[tuple[str, str, str], dict[str, float]] = {}
    run_metas: dict[tuple[str, str, str], dict[str, str]] = {}
    all_cases: set[str] = set()

    for rd in run_dirs:
        meta = parse_manifest_or_name(rd)
        py = meta.get("python", "unknown")
        torch_ver = meta.get("torch", "unknown")
        cuda = meta.get("cuda", "0")
        dev = "CUDA" if cuda == "1" else "CPU"
        key = (torch_ver, py, dev)

        data = load_run_results(rd)
        if not data:
            continue

        all_cases.update(data.keys())
        if key not in grid:
            grid[key] = data
            run_metas[key] = meta
        else:
            grid[key].update(data)

    print(
        f"Aggregated {len(grid)} distinct (PyTorch x Python x Device) matrix configurations."
    )
    print(f"Total benchmark cases tracked: {len(all_cases)}")

    common_cases = [
        c
        for c in all_cases
        if sum(1 for data in grid.values() if c in data) >= max(1, len(grid) // 2)
    ]

    for dev in ["CPU", "CUDA"]:
        dev_configs = {k: v for k, v in grid.items() if k[2] == dev}
        if not dev_configs:
            continue

        print("\n==================================================")
        print(f"=== {dev} Benchmark Matrix Analysis ===")
        print("==================================================")

        gmeans: dict[tuple[str, str], float] = {}
        for (torch_ver, py, _), data in dev_configs.items():
            times = [data[c] for c in common_cases if c in data]
            if times:
                gmeans[(torch_ver, py)] = geom_mean(times)

        if not gmeans:
            continue

        sorted_configs = sorted(gmeans.items(), key=lambda x: x[1])
        best_cfg, best_time = sorted_configs[0]

        print(
            f"\n🏆 Champion (Best Performance) for {dev}: PyTorch {best_cfg[0]} + Python {best_cfg[1]}"
        )
        print(f"   Baseline Geometric-Mean Case Time: {best_time:.3f} ms\n")

        print(
            "| PyTorch | Python | Device | Geom Mean (ms) | Delta vs Best | Relative Perf |"
        )
        print("|---|---|---|---:|---:|---:|")
        for (torch_ver, py), gm in sorted_configs:
            delta_pct = ((gm - best_time) / best_time) * 100.0 if best_time > 0 else 0.0
            ratio = gm / best_time if best_time > 0 else 1.0
            if delta_pct == 0:
                delta_str = "**Baseline (Best)**"
                ratio_str = "**1.00×**"
            else:
                delta_str = f"+{delta_pct:.1f}% slower"
                ratio_str = f"{ratio:.2f}×"
            print(
                f"| {torch_ver} | {py} | {dev} | {gm:.3f} | {delta_str} | {ratio_str} |"
            )

        py_deltas: dict[str, list[float]] = defaultdict(list)
        for (torch_ver, py), gm in sorted_configs:
            d = ((gm - best_time) / best_time) * 100.0
            py_deltas[py].append(d)

        print(f"\nAverage Performance Delta by Python Version ({dev}):")
        for py in sorted(py_deltas.keys()):
            avg_d = sum(py_deltas[py]) / len(py_deltas[py])
            print(f" - Python {py}: average {avg_d:+.1f}% vs champion baseline")

        torch_deltas: dict[str, list[float]] = defaultdict(list)
        for (torch_ver, py), gm in sorted_configs:
            d = ((gm - best_time) / best_time) * 100.0
            torch_deltas[torch_ver].append(d)

        print(f"\nAverage Performance Delta by PyTorch Version ({dev}):")
        for t in sorted(torch_deltas.keys()):
            avg_t = sum(torch_deltas[t]) / len(torch_deltas[t])
            print(f" - PyTorch {t}: average {avg_t:+.1f}% vs champion baseline")


if __name__ == "__main__":
    main()
