#!/usr/bin/env python3
"""4-domain benchmark orchestrator for TorchFits.

Domains:
1) FITS I/O
2) FITS Table I/O
3) WCS
4) Sphere
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from bench_contract import (
    DEFICIT_COLUMNS,
    RESULT_COLUMNS,
    annotate_rankings,
    compute_deficits,
    make_run_id,
    write_csv,
    write_summary,
)
from bench_fits_io import run_fits_domain
from bench_fitstable_io import run_fitstable_domain
from bench_sphere_suite import run_sphere_domain
from bench_wcs_suite import REQUIRED_PROJECTIONS, run_wcs_domain


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["all", "fits", "fitstable", "wcs", "sphere"],
        default="all",
        help="Benchmark scope selector",
    )
    parser.add_argument("--fits-only", action="store_true", help="Alias for --scope fits")
    parser.add_argument(
        "--fitstable-only", action="store_true", help="Alias for --scope fitstable"
    )
    parser.add_argument("--wcs-only", action="store_true", help="Alias for --scope wcs")
    parser.add_argument(
        "--sphere-only", action="store_true", help="Alias for --scope sphere"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks_results"),
        help="Root output directory",
    )
    parser.add_argument("--run-id", type=str, default="", help="Optional run id")

    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--mmap", action="store_true", help="Force mmap on")
    parser.add_argument("--no-mmap", action="store_true", help="Force mmap off")
    parser.add_argument("--filter", type=str, default="", help="Regex case filter (fits domain)")

    parser.add_argument(
        "--legacy-wcs",
        action="store_true",
        help="Include legacy WCS comparators (PyAST/Kapteyn) when available",
    )
    parser.add_argument(
        "--wcs-n-tiers",
        type=str,
        default="1000,10000,100000,1000000,10000000",
        help="Comma-separated WCS point-count tiers",
    )
    parser.add_argument(
        "--wcs-device",
        choices=["cpu", "auto", "cuda"],
        default="cpu",
        help="WCS benchmark device",
    )

    parser.add_argument("--no-gpu", action="store_true", help="Disable optional sphere GPU subsection")

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Reserved: plots are currently disabled in orchestrated mode",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="No plots (default behavior)",
    )

    parser.add_argument("--quick", action="store_true", help="Reduce workload for smoke checks")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary fixture files")
    return parser.parse_args()


def _resolve_scope(args: argparse.Namespace) -> str:
    scope = args.scope
    if args.fits_only:
        scope = "fits"
    elif args.fitstable_only:
        scope = "fitstable"
    elif args.wcs_only:
        scope = "wcs"
    elif args.sphere_only:
        scope = "sphere"
    return scope


def _resolve_use_mmap(args: argparse.Namespace) -> bool:
    use_mmap = True
    if args.no_mmap:
        use_mmap = False
    elif args.mmap:
        use_mmap = True
    return use_mmap


def _scopes_from_scope(scope: str) -> list[str]:
    if scope == "all":
        return ["fits", "fitstable", "wcs", "sphere"]
    return [scope]


def _parse_int_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _print_deficit_summary(deficits: list[dict[str, Any]]) -> None:
    print("\n=== TorchFits Deficits (not first) ===", flush=True)
    if not deficits:
        print("No comparable deficits found.", flush=True)
        return

    print("domain | family | case | torchfits_s | best | lag_x | pct_behind", flush=True)
    for row in deficits:
        best_lbl = f"{row.get('best_library')}:{row.get('best_method')}"
        tf_s = row.get("torchfits_time_s")
        lag_x = row.get("lag_ratio")
        pct = row.get("pct_behind")
        print(
            f"{row.get('domain')} | {row.get('family')} | {row.get('case_label')} | "
            f"{tf_s:.6f} | {best_lbl} | {lag_x:.3f} | {pct:.2f}%",
            flush=True,
        )


def main() -> int:
    # Support `pixi run <task> -- --flag` separator pass-through.
    if "--" in sys.argv[1:]:
        sys.argv = [sys.argv[0], *[a for a in sys.argv[1:] if a != "--"]]
    args = _parse_args()

    scope = _resolve_scope(args)
    scopes = _scopes_from_scope(scope)
    use_mmap = _resolve_use_mmap(args)

    run_id = args.run_id.strip() or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.plots:
        print("[bench-all] --plots requested, but orchestrated mode keeps plots disabled.", flush=True)

    print("Starting benchmark orchestrator", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"scopes={scopes}", flush=True)
    print(f"output={run_dir}", flush=True)
    print(f"mmap={'on' if use_mmap else 'off'}", flush=True)

    all_rows: list[dict[str, Any]] = []

    try:
        if "fits" in scopes:
            all_rows.extend(
                run_fits_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    profile=args.profile,
                    use_mmap=use_mmap,
                    case_filter=args.filter,
                    header_runs=3 if args.quick else 7,
                    header_warmup=1 if args.quick else 2,
                    keep_temp=args.keep_temp,
                )
            )

        if "fitstable" in scopes:
            all_rows.extend(
                run_fitstable_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    use_mmap=use_mmap,
                    profile=args.profile,
                    warmup=0 if args.quick else 1,
                    quick=args.quick,
                    keep_temp=args.keep_temp,
                )
            )

        if "wcs" in scopes:
            tiers = _parse_int_list(args.wcs_n_tiers)
            if args.quick:
                tiers = [1_000, 10_000]
            all_rows.extend(
                run_wcs_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    n_tiers=tiers,
                    projections=list(REQUIRED_PROJECTIONS),
                    device_choice=args.wcs_device,
                    origin=0,
                    sample_profile="mixed",
                    include_legacy=args.legacy_wcs,
                )
            )

        if "sphere" in scopes:
            all_rows.extend(
                run_sphere_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    include_gpu=(not args.no_gpu and not args.quick),
                )
            )

    except Exception as exc:
        print(f"[bench-all] fatal error: {exc}", flush=True)
        # Continue to materialize whatever rows are already available.

    annotate_rankings(all_rows)
    deficits = compute_deficits(all_rows, run_id=run_id)

    results_csv = run_dir / "results.csv"
    deficits_csv = run_dir / "torchfits_deficits.csv"
    summary_md = run_dir / "summary.md"

    write_csv(results_csv, all_rows, RESULT_COLUMNS)
    write_csv(deficits_csv, deficits, DEFICIT_COLUMNS)
    write_summary(summary_md, run_id=run_id, scopes=scopes, rows=all_rows, deficits=deficits)

    print("\nBenchmark run completed", flush=True)
    print(f"- Results CSV: {results_csv}", flush=True)
    print(f"- Deficits CSV: {deficits_csv}", flush=True)
    print(f"- Summary MD: {summary_md}", flush=True)
    _print_deficit_summary(deficits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
