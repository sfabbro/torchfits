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
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from bench_contract import (
    DEFICIT_COLUMNS,
    LARGE_N_THRESHOLD,
    RESULT_COLUMNS,
    SMALL_N_MAX_LAG_RATIO,
    SMALL_N_PERCEIVED_LATENCY_S,
    annotate_rankings,
    compute_deficits,
    make_run_id,
    write_csv,
    write_summary,
)
from bench_fits_io import run_fits_domain
from bench_fitstable_io import run_fitstable_domain
from bench_sphere_suite import run_sphere_domain
from bench_wcs_suite import REQUIRED_PROJECTIONS, WCS_SAMPLE_SEED_BASE, run_wcs_domain


ROOT = Path(__file__).resolve().parents[1]
QUICK_CASES_PER_DOMAIN = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["all", "fits", "fitstable", "wcs", "sphere"],
        default="all",
        help="Benchmark scope selector",
    )
    parser.add_argument(
        "--fits-only", action="store_true", help="Alias for --scope fits"
    )
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
    parser.add_argument(
        "--filter", type=str, default="", help="Regex case filter (fits domain)"
    )

    parser.add_argument(
        "--legacy-wcs",
        action="store_true",
        help="Include legacy WCS comparators (PyAST/Kapteyn) when available",
    )
    parser.add_argument(
        "--no-legacy-wcs-bridge",
        action="store_true",
        help="Disable auto cross-env legacy WCS bridge for wcs-only runs",
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
    parser.add_argument(
        "--wcs-compile",
        action="store_true",
        help="Enable torch.compile() in WCS domain (disabled by default)",
    )
    parser.add_argument(
        "--wcs-replicates",
        type=int,
        default=3,
        help="Number of repeated seeded runs per WCS case/tier (aggregated by median)",
    )

    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable optional sphere GPU subsection"
    )

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

    parser.add_argument(
        "--quick", action="store_true", help="Reduce workload for smoke checks"
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary fixture files"
    )
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

    large_n_deficits = [
        d for d in deficits if (_to_int(d.get("n_points")) or 0) >= LARGE_N_THRESHOLD
    ]
    small_n_visible = [
        d
        for d in deficits
        if (_to_int(d.get("n_points")) or 0) < LARGE_N_THRESHOLD
        and str(d.get("perceived_impact")) in {"visible", "ratio_outlier"}
    ]
    small_n_negligible = [
        d
        for d in deficits
        if (_to_int(d.get("n_points")) or 0) < LARGE_N_THRESHOLD
        and str(d.get("perceived_impact")) == "negligible"
    ]

    print("adoption_policy:", flush=True)
    print(
        f"  large_n_threshold={LARGE_N_THRESHOLD} | small_n_time_s<{SMALL_N_PERCEIVED_LATENCY_S:.6f} | small_n_lag_x<{SMALL_N_MAX_LAG_RATIO:.1f}",
        flush=True,
    )
    print(
        f"  large_n_deficits={len(large_n_deficits)} | small_n_visible={len(small_n_visible)} | small_n_negligible={len(small_n_negligible)}",
        flush=True,
    )

    if large_n_deficits:
        print(
            "large_n: case | n_points | torchfits_s | best | lag_x | pct_behind",
            flush=True,
        )
        for row in large_n_deficits:
            best_lbl = f"{row.get('best_library')}:{row.get('best_method')}"
            tf_s = row.get("torchfits_time_s")
            lag_x = row.get("lag_ratio")
            pct = row.get("pct_behind")
            print(
                f"  {row.get('case_label')} | {row.get('n_points')} | {tf_s:.6f} | "
                f"{best_lbl} | {lag_x:.3f} | {pct:.2f}%",
                flush=True,
            )

    if small_n_visible:
        print(
            "small_n_visible: case | torchfits_s | best | lag_x | pct_behind | impact",
            flush=True,
        )
        for row in small_n_visible:
            best_lbl = f"{row.get('best_library')}:{row.get('best_method')}"
            tf_s = row.get("torchfits_time_s")
            lag_x = row.get("lag_ratio")
            pct = row.get("pct_behind")
            print(
                f"  {row.get('case_label')} | {tf_s:.6f} | {best_lbl} | "
                f"{lag_x:.3f} | {pct:.2f}% | {row.get('perceived_impact')}",
                flush=True,
            )

    print(
        "domain | family | case | torchfits_s | best | lag_x | pct_behind | n_points",
        flush=True,
    )
    for row in deficits:
        best_lbl = f"{row.get('best_library')}:{row.get('best_method')}"
        tf_s = row.get("torchfits_time_s")
        lag_x = row.get("lag_ratio")
        pct = row.get("pct_behind")
        n_points = row.get("n_points", "")
        print(
            f"{row.get('domain')} | {row.get('family')} | {row.get('case_label')} | "
            f"{tf_s:.6f} | {best_lbl} | {lag_x:.3f} | {pct:.2f}% | {n_points}",
            flush=True,
        )


def _domain_failure_row(*, run_id: str, domain: str, error: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": domain,
        "suite": f"{domain}_orchestrator",
        "case_id": f"{domain}::domain_failure",
        "case_label": f"{domain} domain failure",
        "operation": "domain_failure",
        "family": "orchestrator",
        "library": "torchfits",
        "method": "orchestrator",
        "mode": "n/a",
        "status": "FAILED",
        "skip_reason": error,
        "comparable": False,
        "mmap_target": "-",
        "time_s": "",
        "throughput": "",
        "unit": "",
        "size_mb": "",
        "n_points": "",
        "metadata": {"error": error},
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _run_wcs_legacy_bridge(
    *,
    run_id: str,
    run_dir: Path,
    n_tiers: list[int],
    projections: list[str],
    origin: int,
    sample_profile: str,
    replicates: int,
) -> tuple[list[dict[str, Any]], str]:
    bridge_root = run_dir / "_raw" / "wcs_legacy_bridge"
    bridge_run_id = f"{run_id}_legacy"
    bridge_json = bridge_root / f"{bridge_run_id}.json"
    bridge_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pixi",
        "run",
        "-e",
        "bench-legacy",
        "python",
        "benchmarks/bench_wcs_legacy_only.py",
        "--n-tiers",
        ",".join(str(x) for x in n_tiers),
        "--cases",
        ",".join(projections),
        "--sample-profile",
        sample_profile,
        "--origin",
        str(origin),
        "--seed",
        str(WCS_SAMPLE_SEED_BASE),
        "--replicates",
        str(max(1, int(replicates))),
        "--json-out",
        str(bridge_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = bridge_root / "bridge.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        return [], f"legacy_wcs_bridge_failed(code={proc.returncode}, log={log_path})"
    if not bridge_json.exists():
        return [], f"legacy_wcs_bridge_missing_json({bridge_json})"

    try:
        src_rows = json.loads(bridge_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"legacy_wcs_bridge_invalid_json({exc})"
    if not isinstance(src_rows, list):
        return [], "legacy_wcs_bridge_json_not_list"

    out_rows: list[dict[str, Any]] = []
    for src in src_rows:
        if not isinstance(src, dict):
            continue
        library = str(src.get("library", ""))
        if library not in {"pyast", "kapteyn"}:
            continue

        status = str(src.get("status", "SKIPPED"))
        t_val = _to_float(src.get("time_s"))
        n_points = src.get("n_points", "")
        n_int = int(n_points) if str(n_points).strip().isdigit() else 0
        throughput = _to_float(src.get("throughput"))
        if throughput is None and t_val and t_val > 0 and n_int > 0:
            throughput = n_int / t_val / 1e6

        comparable = bool(status == "OK" and t_val is not None and t_val > 0)
        metadata = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
        metadata = dict(metadata)
        metadata["source_env"] = "bench-legacy"
        metadata["cross_env"] = True

        out_rows.append(
            {
                "run_id": run_id,
                "domain": "wcs",
                "suite": str(src.get("suite", "wcs_legacy")),
                "case_id": str(src.get("case_id", "")),
                "case_label": str(src.get("case_label", "")),
                "operation": str(src.get("operation", "forward")),
                # Merge into the smart family table for direct cross-env comparison.
                "family": "smart",
                "library": library,
                "method": library,
                "mode": "legacy_cross_env",
                "status": status,
                "skip_reason": str(src.get("skip_reason", "")),
                "comparable": comparable,
                "mmap_target": "-",
                "time_s": t_val,
                "throughput": throughput,
                "unit": str(src.get("unit", "Mpts/s")),
                "size_mb": src.get("size_mb", ""),
                "n_points": src.get("n_points", ""),
                "metadata": metadata,
            }
        )
    return out_rows, ""


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
        print(
            "[bench-all] --plots requested, but orchestrated mode keeps plots disabled.",
            flush=True,
        )

    print("Starting benchmark orchestrator", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"scopes={scopes}", flush=True)
    print(f"output={run_dir}", flush=True)
    print(f"mmap={'on' if use_mmap else 'off'}", flush=True)

    all_rows: list[dict[str, Any]] = []

    if "fits" in scopes:
        try:
            fits_case_filter = args.filter
            if args.quick and not fits_case_filter:
                fits_case_filter = "^(tiny_int16_2d|mef_small|compressed_rice_1)$"
            all_rows.extend(
                run_fits_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    profile=args.profile,
                    use_mmap=use_mmap,
                    case_filter=fits_case_filter,
                    header_runs=3 if args.quick else 7,
                    header_warmup=1 if args.quick else 2,
                    keep_temp=args.keep_temp,
                )
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[bench-all][fits] failed: {err}", flush=True)
            all_rows.append(
                _domain_failure_row(run_id=run_id, domain="fits", error=err)
            )

    if "fitstable" in scopes:
        try:
            all_rows.extend(
                run_fitstable_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    use_mmap=use_mmap,
                    profile=args.profile,
                    warmup=0 if args.quick else 1,
                    quick=args.quick,
                    max_cases=QUICK_CASES_PER_DOMAIN if args.quick else None,
                    keep_temp=args.keep_temp,
                )
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[bench-all][fitstable] failed: {err}", flush=True)
            all_rows.append(
                _domain_failure_row(run_id=run_id, domain="fitstable", error=err)
            )

    if "wcs" in scopes:
        try:
            tiers = [1_000] if args.quick else _parse_int_list(args.wcs_n_tiers)
            wcs_reps = 1 if args.quick else max(1, int(args.wcs_replicates))
            projections = (
                list(REQUIRED_PROJECTIONS[:QUICK_CASES_PER_DOMAIN])
                if args.quick
                else list(REQUIRED_PROJECTIONS)
            )
            all_rows.extend(
                run_wcs_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    n_tiers=tiers,
                    projections=projections,
                    device_choice=args.wcs_device,
                    origin=0,
                    sample_profile="mixed",
                    include_legacy=args.legacy_wcs,
                    torch_compile=args.wcs_compile,
                    replicates=wcs_reps,
                )
            )
            use_legacy_bridge = not args.no_legacy_wcs_bridge
            if use_legacy_bridge:
                print(
                    "[bench-all][wcs] running cross-env legacy bridge (bench-legacy)...",
                    flush=True,
                )
                bridge_rows, bridge_err = _run_wcs_legacy_bridge(
                    run_id=run_id,
                    run_dir=run_dir,
                    n_tiers=tiers,
                    projections=projections,
                    origin=0,
                    sample_profile="mixed",
                    replicates=wcs_reps,
                )
                if bridge_err:
                    print(
                        f"[bench-all][wcs] legacy bridge failed: {bridge_err}",
                        flush=True,
                    )
                    all_rows.append(
                        _domain_failure_row(
                            run_id=run_id, domain="wcs", error=bridge_err
                        )
                    )
                else:
                    print(
                        f"[bench-all][wcs] legacy bridge rows={len(bridge_rows)} (pyast/kapteyn)",
                        flush=True,
                    )
                    all_rows.extend(bridge_rows)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[bench-all][wcs] failed: {err}", flush=True)
            all_rows.append(_domain_failure_row(run_id=run_id, domain="wcs", error=err))

    if "sphere" in scopes:
        try:
            all_rows.extend(
                run_sphere_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    include_gpu=(not args.no_gpu and not args.quick),
                    quick_cases=QUICK_CASES_PER_DOMAIN if args.quick else None,
                )
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[bench-all][sphere] failed: {err}", flush=True)
            all_rows.append(
                _domain_failure_row(run_id=run_id, domain="sphere", error=err)
            )

    annotate_rankings(all_rows)
    deficits = compute_deficits(all_rows, run_id=run_id)

    results_csv = run_dir / "results.csv"
    deficits_csv = run_dir / "torchfits_deficits.csv"
    summary_md = run_dir / "summary.md"

    write_csv(results_csv, all_rows, RESULT_COLUMNS)
    write_csv(deficits_csv, deficits, DEFICIT_COLUMNS)
    write_summary(
        summary_md, run_id=run_id, scopes=scopes, rows=all_rows, deficits=deficits
    )

    print("\nBenchmark run completed", flush=True)
    print(f"- Results CSV: {results_csv}", flush=True)
    print(f"- Deficits CSV: {deficits_csv}", flush=True)
    print(f"- Summary MD: {summary_md}", flush=True)
    _print_deficit_summary(deficits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
