#!/usr/bin/env python3
"""Run multi-profile sphere geometry benchmarks and enforce median ratio gates."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


OPS = (
    "ang2pix_ring",
    "ang2pix_nested",
    "pix2ang_ring",
    "pix2ang_nested",
    "ring2nest",
    "nest2ring",
)
DEFAULT_PROFILES = ("uniform", "boundary", "mixed")


def _parse_ratio_spec(spec: str | None) -> dict[str, float]:
    if spec is None:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    valid = set(OPS)
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid ratio spec segment '{part}', expected op=value")
        op, raw = part.split("=", 1)
        op = op.strip()
        raw = raw.strip()
        if op not in valid:
            raise ValueError(f"Unknown operation '{op}' in ratio spec")
        value = float(raw)
        if value <= 0.0:
            raise ValueError(f"Ratio threshold must be positive for operation '{op}'")
        out[op] = value
    return out


def _ratios_vs_healpy(rows: list[dict[str, Any]], library: str = "torchfits") -> dict[str, float]:
    by_key = {(str(r["library"]), str(r["operation"])): float(r["mpts_s"]) for r in rows}
    ratios: dict[str, float] = {}
    for op in OPS:
        a = by_key.get((library, op))
        b = by_key.get(("healpy", op))
        if a is None or b is None or b <= 0.0:
            continue
        ratios[op] = a / b
    return ratios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=str, default="uniform,boundary,mixed")
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--libraries", type=str, default="torchfits,healpy")
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--allow-nonrelease-distributions", action="store_true")
    parser.add_argument("--max-index-mismatches", type=int, default=0)
    parser.add_argument("--max-pix2ang-dra-deg", type=float, default=1.0e-10)
    parser.add_argument("--max-pix2ang-ddec-deg", type=float, default=1.0e-10)
    parser.add_argument(
        "--min-median-ratio-vs-healpy",
        type=str,
        default=(
            "ang2pix_ring=1.8,ang2pix_nested=1.35,pix2ang_ring=1.15,"
            "pix2ang_nested=1.0,ring2nest=1.05,nest2ring=1.35"
        ),
        help="Comma-separated op=ratio thresholds for median(torchfits/healpy) across profiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench_results/sphere_matrix_latest"),
    )
    args = parser.parse_args()

    profiles = tuple(x.strip() for x in args.profiles.split(",") if x.strip())
    if not profiles:
        print("No profiles specified")
        return 1
    for p in profiles:
        if p not in DEFAULT_PROFILES:
            print(f"Unsupported profile '{p}'")
            return 1

    try:
        min_median_ratios = _parse_ratio_spec(args.min_median_ratio_vs_healpy)
    except ValueError as exc:
        print(f"Invalid --min-median-ratio-vs-healpy: {exc}")
        return 1

    script = Path(__file__).with_name("bench_sphere_geometry.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_profile: dict[str, list[dict[str, Any]]] = {}
    ratios_by_profile: dict[str, dict[str, float]] = {}
    json_paths: dict[str, str] = {}

    for i, profile in enumerate(profiles):
        out_json = args.output_dir / f"profile_{profile}.json"
        cmd = [
            sys.executable,
            str(script),
            "--nside",
            str(args.nside),
            "--n-points",
            str(args.n_points),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--sample-profile",
            profile,
            "--libraries",
            args.libraries,
            "--json-out",
            str(out_json),
            "--max-index-mismatches",
            str(args.max_index_mismatches),
            "--max-pix2ang-dra-deg",
            str(args.max_pix2ang_dra_deg),
            "--max-pix2ang-ddec-deg",
            str(args.max_pix2ang_ddec_deg),
        ]
        if args.strict_missing:
            cmd.append("--strict-missing")
        if args.allow_nonrelease_distributions:
            cmd.append("--allow-nonrelease-distributions")

        print(f"\n[profile={profile}] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"Profile '{profile}' benchmark failed with exit code {proc.returncode}")
            return proc.returncode

        rows = json.loads(out_json.read_text(encoding="utf-8"))
        rows_by_profile[profile] = rows
        ratios = _ratios_vs_healpy(rows, library="torchfits")
        ratios_by_profile[profile] = ratios
        json_paths[profile] = str(out_json)

    median_ratios: dict[str, float] = {}
    for op in OPS:
        vals = [ratios_by_profile[p][op] for p in profiles if op in ratios_by_profile[p]]
        if vals:
            median_ratios[op] = float(statistics.median(vals))

    print("\nMedian TorchFits/healpy ratios across profiles:")
    for op in OPS:
        v = median_ratios.get(op)
        if v is None:
            print(f"  {op:15s} unavailable")
        else:
            print(f"  {op:15s} {v:.3f}x")

    failed = []
    for op, threshold in min_median_ratios.items():
        got = median_ratios.get(op)
        if got is None or got < threshold:
            failed.append((op, threshold, got))

    summary = {
        "profiles": profiles,
        "json_paths": json_paths,
        "median_ratios_torchfits_vs_healpy": median_ratios,
        "min_median_ratio_thresholds": min_median_ratios,
        "failed_thresholds": [
            {"operation": op, "threshold": thr, "observed": got} for op, thr, got in failed
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    if failed:
        print("\nMedian ratio thresholds exceeded:")
        for op, threshold, got in failed:
            if got is None:
                print(f"  {op}: unavailable < {threshold:.3f}")
            else:
                print(f"  {op}: {got:.3f} < {threshold:.3f}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
