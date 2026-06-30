#!/usr/bin/env python3
"""FITS benchmark orchestrator for torchfits.

Domains:
1) FITS image I/O
2) FITS table I/O
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from benchmarks.bench_contract import (  # noqa: E402
    DEFICIT_COLUMNS,
    RESULT_COLUMNS,
    annotate_rankings,
    compute_deficits,
    make_run_id,
    write_csv,
    write_summary,
)
from benchmarks.bench_fits_io import run_fits_domain  # noqa: E402
from benchmarks.bench_fitstable_io import run_fitstable_domain  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


QUICK_CASES_PER_DOMAIN = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["all", "fits", "fitstable"],
        default="all",
        help="Benchmark scope selector",
    )
    parser.add_argument(
        "--fits-only", action="store_true", help="Alias for --scope fits"
    )
    parser.add_argument(
        "--fitstable-only",
        action="store_true",
        help="Alias for --scope fitstable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory",
    )
    parser.add_argument("--run-id", type=str, default="", help="Optional run id")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--mmap", action="store_true", help="Force mmap on")
    parser.add_argument("--no-mmap", action="store_true", help="Force mmap off")
    parser.add_argument("--filter", type=str, default="", help="Regex case filter")
    parser.add_argument(
        "--quick", action="store_true", help="Reduce workload for smoke checks"
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary fixture files"
    )
    return parser.parse_args()


def _resolve_scope(args: argparse.Namespace) -> str:
    if args.fits_only:
        return "fits"
    if args.fitstable_only:
        return "fitstable"
    return str(args.scope)


def _resolve_use_mmap(args: argparse.Namespace) -> bool:
    if args.no_mmap:
        return False
    return True


def _scopes_from_scope(scope: str) -> list[str]:
    if scope == "all":
        return ["fits", "fitstable"]
    return [scope]


def _domain_failure_row(*, run_id: str, domain: str, error: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": domain,
        "suite": domain,
        "case_id": f"{domain}::failure",
        "case_label": f"{domain} failure",
        "family": "smart",
        "library": "torchfits",
        "method": "torchfits",
        "status": "ERROR",
        "skip_reason": error,
        "time_s": None,
        "median_s": None,
    }


def _run_fitstable_isolated(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    use_mmap: bool,
) -> list[dict[str, Any]]:
    """Run the table domain in a fresh process after the full image matrix.

    CFITSIO and PyTorch both retain process-global native state. Domain
    isolation also prevents one benchmark's allocator/cache history from
    contaminating the next domain's timings.
    """
    json_path = run_dir / "_fitstable_subprocess_rows.json"
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / "bench_fitstable_io.py"),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--run-id",
        run_id,
        "--profile",
        args.profile,
        "--warmup",
        "0" if args.quick else "1",
        "--json-out",
        str(json_path.resolve()),
        "--mmap" if use_mmap else "--no-mmap",
    ]
    if args.quick:
        command.extend(["--quick", "--max-cases", str(QUICK_CASES_PER_DOMAIN)])
    if args.keep_temp:
        command.append("--keep-temp")

    try:
        subprocess.run(command, cwd=ROOT, check=True)
        with json_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise RuntimeError("isolated table benchmark returned invalid JSON")
        return loaded
    finally:
        json_path.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    scope = _resolve_scope(args)
    scopes = _scopes_from_scope(scope)
    use_mmap = _resolve_use_mmap(args)
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting torchfits benchmark orchestrator", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"scopes={scopes}", flush=True)
    print(f"output={run_dir}", flush=True)
    print(f"mmap={'on' if use_mmap else 'off'}", flush=True)

    rows: list[dict[str, Any]] = []

    if "fits" in scopes:
        try:
            case_filter = args.filter
            if args.quick and not case_filter:
                case_filter = "^(tiny_int16_2d|mef_small|compressed_rice_1)$"
            rows.extend(
                run_fits_domain(
                    run_id=run_id,
                    output_dir=run_dir,
                    profile=args.profile,
                    use_mmap=use_mmap,
                    case_filter=case_filter,
                    header_runs=3 if args.quick else 7,
                    header_warmup=1 if args.quick else 2,
                    keep_temp=args.keep_temp,
                )
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[bench-all][fits] failed: {err}", flush=True)
            rows.append(_domain_failure_row(run_id=run_id, domain="fits", error=err))

    if "fitstable" in scopes:
        try:
            if "fits" in scopes:
                rows.extend(
                    _run_fitstable_isolated(
                        args=args,
                        run_id=run_id,
                        run_dir=run_dir,
                        use_mmap=use_mmap,
                    )
                )
            else:
                rows.extend(
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
            rows.append(
                _domain_failure_row(run_id=run_id, domain="fitstable", error=err)
            )

    try:
        from benchmarks.bench_gpu_transports import run_gpu_transport_rows

        iterations = 7 if args.profile == "lab" else 3
        warmup = 2 if args.profile == "lab" else 1
        if args.quick:
            iterations = 1
            warmup = 0
        gpu_rows = run_gpu_transport_rows(
            run_id=run_id,
            iterations=iterations,
            warmup=warmup,
            quick=args.quick,
        )
        if gpu_rows:
            rows.extend(gpu_rows)
            print(f"Added {len(gpu_rows)} GPU transport rows", flush=True)
    except Exception as exc:
        print(f"[bench-all][gpu] failed: {type(exc).__name__}: {exc}", flush=True)

    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id=run_id)

    write_csv(run_dir / "results.csv", rows, RESULT_COLUMNS)
    write_csv(run_dir / "torchfits_deficits.csv", deficits, DEFICIT_COLUMNS)
    write_summary(
        run_dir / "summary.md",
        run_id=run_id,
        scopes=scopes,
        rows=rows,
        deficits=deficits,
    )

    print(f"Wrote {len(rows)} benchmark rows to {run_dir / 'results.csv'}", flush=True)
    print(
        f"Wrote {len(deficits)} deficit rows to {run_dir / 'torchfits_deficits.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
