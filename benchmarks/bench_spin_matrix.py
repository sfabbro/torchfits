#!/usr/bin/env python3
"""Run spin replay and enforce per-regime ratio/error gates."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


OPS = ("map2alm_spin", "alm2map_spin")
BUCKETS = ("small", "medium", "large")


def _bucket_for(row: dict) -> str:
    nside = int(row["nside"])
    lmax = int(row["lmax"])
    if nside <= 16 and lmax <= 12:
        return "small"
    if nside <= 32 and lmax <= 20:
        return "medium"
    return "large"


def _parse_bucket_ratio_spec(spec: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "=" not in part or ":" not in part:
            raise ValueError(f"invalid segment '{part}', expected bucket:op=value")
        lhs, rhs = part.split("=", 1)
        bucket, op = [x.strip() for x in lhs.split(":", 1)]
        if bucket not in BUCKETS:
            raise ValueError(f"unknown bucket '{bucket}'")
        if op not in OPS:
            raise ValueError(f"unknown operation '{op}'")
        val = float(rhs.strip())
        if val <= 0.0:
            raise ValueError(f"threshold must be positive for {bucket}:{op}")
        out[(bucket, op)] = val
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-set", choices=("default", "extended"), default="extended"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--max-map2alm-rel-error", type=float, default=3.0e-9)
    parser.add_argument("--max-alm2map-rel-error", type=float, default=1.0e-9)
    parser.add_argument(
        "--min-ratio-by-bucket",
        type=str,
        default=(
            "small:map2alm_spin=0.30,small:alm2map_spin=0.25,"
            "medium:map2alm_spin=0.03,medium:alm2map_spin=0.03,"
            "large:map2alm_spin=0.015,large:alm2map_spin=0.02"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_healpy_spin_matrix.json"),
    )
    parser.add_argument(
        "--raw-json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_healpy_spin_extended_raw.json"),
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline summary JSON with median_ratio_by_bucket to enforce regression budget.",
    )
    parser.add_argument(
        "--max-regression-frac",
        type=float,
        default=0.20,
        help="Allowed relative drop vs baseline medians (e.g. 0.20 allows up to 20%% drop).",
    )
    args = parser.parse_args()

    if args.max_regression_frac < 0.0 or args.max_regression_frac >= 1.0:
        print("--max-regression-frac must satisfy 0 <= value < 1")
        return 2

    try:
        ratio_thresholds = _parse_bucket_ratio_spec(args.min_ratio_by_bucket)
    except ValueError as exc:
        print(f"Invalid --min-ratio-by-bucket: {exc}")
        return 2

    replay_script = Path(__file__).with_name("replay_upstream_healpy_spin.py")
    args.raw_json_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(replay_script),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--n-points",
        str(args.n_points),
        "--case-set",
        args.case_set,
        "--json-out",
        str(args.raw_json_out),
        "--disable-gates",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"replay_upstream_healpy_spin failed with exit code {proc.returncode}")
        return proc.returncode

    payload = json.loads(args.raw_json_out.read_text(encoding="utf-8"))
    rows = payload["rows"]

    failures: list[str] = []
    for row in rows:
        op = str(row["operation"])
        rel = float(row["rel_l2_error"])
        if op == "map2alm_spin" and rel > args.max_map2alm_rel_error:
            failures.append(
                f"{op}@nside={row['nside']},lmax={row['lmax']},mmax={row['mmax']},nest={row['nest']}: "
                f"rel_l2 {rel:.3e} > {args.max_map2alm_rel_error:.3e}"
            )
        if op == "alm2map_spin" and rel > args.max_alm2map_rel_error:
            failures.append(
                f"{op}@nside={row['nside']},lmax={row['lmax']},mmax={row['mmax']},nest={row['nest']}: "
                f"rel_l2 {rel:.3e} > {args.max_alm2map_rel_error:.3e}"
            )

    medians: dict[str, dict[str, float]] = {b: {} for b in BUCKETS}
    for bucket in BUCKETS:
        for op in OPS:
            vals = [
                float(r["ratio_vs_healpy"])
                for r in rows
                if _bucket_for(r) == bucket and str(r["operation"]) == op
            ]
            if vals:
                medians[bucket][op] = float(statistics.median(vals))

    print("\nMedian ratio by bucket:")
    for bucket in BUCKETS:
        for op in OPS:
            got = medians[bucket].get(op)
            thr = ratio_thresholds.get((bucket, op))
            if got is None:
                print(f"  {bucket:6s} {op:12s} unavailable")
                if thr is not None:
                    failures.append(f"{bucket}:{op} missing (threshold {thr:.3f})")
                continue
            if thr is None:
                print(f"  {bucket:6s} {op:12s} {got:.3f}x")
                continue
            print(f"  {bucket:6s} {op:12s} {got:.3f}x (threshold {thr:.3f}x)")
            if got < thr:
                failures.append(f"{bucket}:{op} median ratio {got:.3f} < {thr:.3f}")

    baseline_path = args.baseline_json
    baseline_medians: dict[str, dict[str, float]] | None = None
    if baseline_path is not None:
        if not baseline_path.exists():
            print(f"\nBaseline JSON not found: {baseline_path}")
            return 2
        try:
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"\nFailed to read baseline JSON {baseline_path}: {exc}")
            return 2
        baseline_raw = baseline_payload.get("median_ratio_by_bucket")
        if not isinstance(baseline_raw, dict):
            print(
                f"\nInvalid baseline JSON {baseline_path}: missing median_ratio_by_bucket"
            )
            return 2
        baseline_medians = {}
        for bucket in BUCKETS:
            row = baseline_raw.get(bucket, {})
            if not isinstance(row, dict):
                row = {}
            baseline_medians[bucket] = {}
            for op in OPS:
                if op in row:
                    baseline_medians[bucket][op] = float(row[op])

        print("\nBaseline Regression Check:")
        for bucket in BUCKETS:
            for op in OPS:
                base = baseline_medians.get(bucket, {}).get(op)
                got = medians.get(bucket, {}).get(op)
                if base is None:
                    continue
                if got is None:
                    failures.append(
                        f"{bucket}:{op} missing current median for baseline comparison"
                    )
                    print(
                        f"  {bucket:6s} {op:12s} missing current (baseline {base:.3f}x)"
                    )
                    continue
                allowed = base * (1.0 - args.max_regression_frac)
                print(
                    f"  {bucket:6s} {op:12s} {got:.3f}x (baseline {base:.3f}x, min {allowed:.3f}x)"
                )
                if got < allowed:
                    failures.append(
                        f"{bucket}:{op} baseline regression {got:.3f} < {allowed:.3f}"
                    )

    summary = {
        "case_set": args.case_set,
        "runs": args.runs,
        "seed": args.seed,
        "n_points": args.n_points,
        "raw_json_out": str(args.raw_json_out),
        "max_rel_error_thresholds": {
            "map2alm_spin": args.max_map2alm_rel_error,
            "alm2map_spin": args.max_alm2map_rel_error,
        },
        "ratio_thresholds_by_bucket": {
            f"{b}:{o}": v for (b, o), v in ratio_thresholds.items()
        },
        "median_ratio_by_bucket": medians,
        "baseline_json": None if baseline_path is None else str(baseline_path),
        "max_regression_frac": args.max_regression_frac,
        "baseline_median_ratio_by_bucket": baseline_medians,
        "failures": failures,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {args.json_out}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
