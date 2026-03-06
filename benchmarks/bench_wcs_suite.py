#!/usr/bin/env python3
"""Authoritative WCS benchmark sweep runner."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench_contract import RESULT_COLUMNS, annotate_rankings, write_csv
from bench_wcs import CASES, _bench_case, _resolve_device


REQUIRED_PROJECTIONS = [
    "TAN",
    "SIN",
    "ARC",
    "ZEA",
    "STG",
    "ZPN",
    "AIT",
    "MOL",
    "HPX",
    "CEA",
    "MER",
    "CAR",
    "SFL",
    "TAN_SIP",
    "TPV",
]

WCS_SAMPLE_SEED_BASE = 12345


def _legacy_runner():
    """Load same-env legacy comparators only on compatible interpreters."""
    if sys.version_info >= (3, 14):
        return None
    try:
        from bench_competitors import run_case as run_legacy_case
    except Exception:
        return None
    return run_legacy_case


def _runs_for_points(n_points: int) -> int:
    if n_points <= 10_000:
        return 9
    if n_points <= 100_000:
        return 7
    if n_points <= 1_000_000:
        return 5
    return 3


def _ok_time(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def _sample_seed(i_case: int, i_tier: int, i_rep: int) -> int:
    # Shared seed contract with bench_wcs_legacy_only.py for cross-env parity.
    return int(WCS_SAMPLE_SEED_BASE + i_case * 100_000 + i_tier * 1_000 + i_rep)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _base_row(
    *,
    run_id: str,
    case_name: str,
    projection: str,
    n_points: int,
    op_name: str,
    family: str,
    library: str,
    method: str,
    mode: str,
    time_s: float | None,
    throughput: float | None,
    status: str,
    skip_reason: str,
    comparable: bool,
    sample_profile: str,
    device: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": "wcs",
        "suite": "wcs_sweep",
        "case_id": f"{case_name}::n{n_points}::{op_name}",
        "case_label": f"{case_name} n={n_points} [{op_name}]",
        "operation": op_name,
        "family": family,
        "library": library,
        "method": method,
        "mode": mode,
        "status": status,
        "skip_reason": skip_reason,
        "comparable": comparable,
        "mmap_target": "-",
        "time_s": time_s,
        "throughput": throughput,
        "unit": "Mpts/s",
        "size_mb": "",
        "n_points": n_points,
        "metadata": {
            "projection": projection,
            "sample_profile": sample_profile,
            "device": device,
            **metadata,
        },
    }


def run_wcs_domain(
    *,
    run_id: str,
    output_dir: Path,
    n_tiers: list[int] | None = None,
    projections: list[str] | None = None,
    device_choice: str = "cpu",
    origin: int = 0,
    sample_profile: str = "mixed",
    include_legacy: bool = False,
    torch_compile: bool = False,
    replicates: int = 1,
) -> list[dict[str, Any]]:
    _ = output_dir
    tiers = n_tiers or [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    requested = projections or list(REQUIRED_PROJECTIONS)
    run_legacy_case = _legacy_runner() if include_legacy else None

    device = _resolve_device(device_choice)
    rows: list[dict[str, Any]] = []

    for i_case, case_name in enumerate(requested):
        case = CASES.get(case_name)
        if case is None:
            # Explicitly report unsupported projections as non-fatal skips.
            for n in tiers:
                rows.append(
                    _base_row(
                        run_id=run_id,
                        case_name=case_name,
                        projection=case_name,
                        n_points=n,
                        op_name="forward",
                        family="smart",
                        library="torchfits",
                        method="torchfits",
                        mode="smart",
                        time_s=None,
                        throughput=None,
                        status="SKIPPED",
                        skip_reason="projection_not_supported",
                        comparable=False,
                        sample_profile=sample_profile,
                        device=device.type,
                        metadata={},
                    )
                )
                rows.append(
                    _base_row(
                        run_id=run_id,
                        case_name=case_name,
                        projection=case_name,
                        n_points=n,
                        op_name="forward",
                        family="smart",
                        library="astropy",
                        method="astropy_torch",
                        mode="smart",
                        time_s=None,
                        throughput=None,
                        status="SKIPPED",
                        skip_reason="projection_not_supported",
                        comparable=False,
                        sample_profile=sample_profile,
                        device="cpu",
                        metadata={},
                    )
                )
            continue

        for i_tier, n in enumerate(tiers):
            runs = _runs_for_points(n)
            print(
                f"[wcs] case={case_name} n={n} runs={runs} reps={max(1, int(replicates))} device={device.type}",
                flush=True,
            )
            rep_results: list[dict[str, Any]] = []
            rep_seeds: list[int] = []
            rep_errors: list[str] = []
            rep_count = max(1, int(replicates))
            for i_rep in range(rep_count):
                sample_seed = _sample_seed(i_case, i_tier, i_rep)
                rep_seeds.append(sample_seed)
                rng = np.random.default_rng(sample_seed)
                try:
                    result = _bench_case(
                        case,
                        n_points=n,
                        runs=runs,
                        device=device,
                        origin=origin,
                        torch_compile=bool(torch_compile),
                        profile=sample_profile,
                        rng=rng,
                    )
                except Exception as exc:
                    rep_errors.append(
                        f"seed={sample_seed}: {type(exc).__name__}: {exc}"
                    )
                    continue
                rep_results.append(result)

            if not rep_results:
                reason = "case_failed_all_replicates"
                if rep_errors:
                    reason += f": {rep_errors[0]}"
                rows.append(
                    _base_row(
                        run_id=run_id,
                        case_name=case_name,
                        projection=case.projection,
                        n_points=n,
                        op_name="forward",
                        family="smart",
                        library="torchfits",
                        method="torchfits",
                        mode="smart",
                        time_s=None,
                        throughput=None,
                        status="FAILED",
                        skip_reason=reason,
                        comparable=False,
                        sample_profile=sample_profile,
                        device=device.type,
                        metadata={},
                    )
                )
                rows.append(
                    _base_row(
                        run_id=run_id,
                        case_name=case_name,
                        projection=case.projection,
                        n_points=n,
                        op_name="forward",
                        family="smart",
                        library="astropy",
                        method="astropy_torch",
                        mode="smart",
                        time_s=None,
                        throughput=None,
                        status="FAILED",
                        skip_reason=reason,
                        comparable=False,
                        sample_profile=sample_profile,
                        device="cpu",
                        metadata={},
                    )
                )
                continue

            tf_fw = _median(
                [
                    t
                    for t in (
                        _ok_time(r.get("torch_forward_ms", 0.0) / 1000.0)
                        for r in rep_results
                    )
                    if t is not None
                ]
            )
            as_fw = _median(
                [
                    t
                    for t in (
                        _ok_time(r.get("astropy_forward_ms", 0.0) / 1000.0)
                        for r in rep_results
                    )
                    if t is not None
                ]
            )
            tf_inv = _median(
                [
                    t
                    for t in (
                        _ok_time(r.get("torch_inverse_ms", 0.0) / 1000.0)
                        for r in rep_results
                    )
                    if t is not None
                ]
            )
            as_inv = _median(
                [
                    t
                    for t in (
                        _ok_time(r.get("astropy_inverse_ms", 0.0) / 1000.0)
                        for r in rep_results
                    )
                    if t is not None
                ]
            )

            n_valid_vals = [
                int(v)
                for v in (r.get("n_valid") for r in rep_results)
                if isinstance(v, (int, np.integer))
            ]
            n_valid_agg = min(n_valid_vals) if n_valid_vals else 0
            inv_refs = [r.get("inverse_reference") for r in rep_results]
            if any(ref == "astropy" for ref in inv_refs):
                inv_ref_agg = "astropy"
            elif any(ref == "roundtrip" for ref in inv_refs):
                inv_ref_agg = "roundtrip"
            else:
                inv_ref_agg = "none"

            def _max_finite(key: str) -> float | None:
                vals = []
                for r in rep_results:
                    v = _ok_time(r.get(key))
                    if v is not None:
                        vals.append(v)
                if not vals:
                    return None
                return float(max(vals))

            meta = {
                "n_valid": n_valid_agg,
                "inverse_reference": inv_ref_agg,
                "sample_seed_base": WCS_SAMPLE_SEED_BASE,
                "sample_seeds": rep_seeds,
                "replicates": rep_count,
                "replicates_ok": len(rep_results),
                "replicates_failed": len(rep_errors),
                "max_angular_error_arcsec": _max_finite("max_angular_error_arcsec"),
                "p99_angular_error_arcsec": _max_finite("p99_angular_error_arcsec"),
                "max_inverse_pixel_error": _max_finite("max_inverse_pixel_error"),
                "p99_inverse_pixel_error": _max_finite("p99_inverse_pixel_error"),
            }

            rows.append(
                _base_row(
                    run_id=run_id,
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n,
                    op_name="forward",
                    family="smart",
                    library="torchfits",
                    method="torchfits",
                    mode="smart",
                    time_s=tf_fw,
                    throughput=(n / tf_fw / 1e6) if tf_fw else None,
                    status="OK" if tf_fw is not None else "FAILED",
                    skip_reason="",
                    comparable=tf_fw is not None,
                    sample_profile=sample_profile,
                    device=device.type,
                    metadata=meta,
                )
            )
            rows.append(
                _base_row(
                    run_id=run_id,
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n,
                    op_name="forward",
                    family="smart",
                    library="astropy",
                    method="astropy_torch",
                    mode="smart",
                    time_s=as_fw,
                    throughput=(n / as_fw / 1e6) if as_fw else None,
                    status="OK" if as_fw is not None else "FAILED",
                    skip_reason="",
                    comparable=as_fw is not None,
                    sample_profile=sample_profile,
                    device="cpu",
                    metadata=meta,
                )
            )

            inv_ok = inv_ref_agg not in {"none", None}
            astropy_inv_skip_reason = ""
            if not inv_ok:
                astropy_inv_skip_reason = "inverse_not_available"
            elif as_inv is None:
                astropy_inv_skip_reason = "astropy_inverse_not_available_for_case"
            rows.append(
                _base_row(
                    run_id=run_id,
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n,
                    op_name="inverse",
                    family="smart",
                    library="torchfits",
                    method="torchfits",
                    mode="smart",
                    time_s=tf_inv if inv_ok else None,
                    throughput=(n / tf_inv / 1e6) if (inv_ok and tf_inv) else None,
                    status="OK" if (inv_ok and tf_inv is not None) else "SKIPPED",
                    skip_reason="inverse_not_available" if not inv_ok else "",
                    comparable=bool(inv_ok and tf_inv is not None),
                    sample_profile=sample_profile,
                    device=device.type,
                    metadata=meta,
                )
            )
            rows.append(
                _base_row(
                    run_id=run_id,
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n,
                    op_name="inverse",
                    family="smart",
                    library="astropy",
                    method="astropy_torch",
                    mode="smart",
                    time_s=as_inv if inv_ok else None,
                    throughput=(n / as_inv / 1e6) if (inv_ok and as_inv) else None,
                    status="OK" if (inv_ok and as_inv is not None) else "SKIPPED",
                    skip_reason=astropy_inv_skip_reason,
                    comparable=bool(inv_ok and as_inv is not None),
                    sample_profile=sample_profile,
                    device="cpu",
                    metadata=meta,
                )
            )

            if include_legacy and run_legacy_case is not None:
                legacy_times: dict[str, list[float]] = {"pyast": [], "kapteyn": []}
                for i_rep in range(rep_count):
                    rep_seed = _sample_seed(i_case, i_tier, i_rep)
                    try:
                        leg = run_legacy_case(
                            case_name, n, origin, rep_seed, sample_profile
                        )
                    except Exception:
                        leg = {}
                    if not isinstance(leg, dict):
                        leg = {}
                    for lib_key in ("pyast", "kapteyn"):
                        ms = leg.get(f"{lib_key}_ms")
                        t = _ok_time((float(ms) / 1000.0) if ms is not None else None)
                        if t is not None:
                            legacy_times[lib_key].append(t)
                for lib_key, mode in (
                    ("pyast", "specialized"),
                    ("kapteyn", "specialized"),
                ):
                    t = _median(legacy_times[lib_key])
                    status = "OK" if t is not None else "SKIPPED"
                    rows.append(
                        _base_row(
                            run_id=run_id,
                            case_name=case_name,
                            projection=case.projection,
                            n_points=n,
                            op_name="forward",
                            family="specialized_legacy",
                            library=lib_key,
                            method=lib_key,
                            mode=mode,
                            time_s=t,
                            throughput=(n / t / 1e6) if t else None,
                            status=status,
                            skip_reason="not_installed_or_failed"
                            if status != "OK"
                            else "",
                            comparable=False,
                            sample_profile=sample_profile,
                            device="cpu",
                            metadata={
                                "legacy": True,
                                "sample_seed_base": WCS_SAMPLE_SEED_BASE,
                                "replicates": rep_count,
                                "replicates_ok": len(legacy_times[lib_key]),
                            },
                        )
                    )

    annotate_rankings(rows)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks_results"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--origin", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--sample-profile", choices=["interior", "boundary", "mixed"], default="mixed"
    )
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile() for TorchFits WCS transforms",
    )
    parser.add_argument(
        "--n-tiers", type=str, default="1000,10000,100000,1000000,10000000"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of repeated seeded runs per case/tier (aggregated by median)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id

    n_tiers = [int(x.strip()) for x in args.n_tiers.split(",") if x.strip()]

    rows = run_wcs_domain(
        run_id=run_id,
        output_dir=run_dir,
        n_tiers=n_tiers,
        projections=list(REQUIRED_PROJECTIONS),
        device_choice=args.device,
        origin=args.origin,
        sample_profile=args.sample_profile,
        include_legacy=args.include_legacy,
        torch_compile=args.torch_compile,
        replicates=max(1, int(args.replicates)),
    )

    out_csv = run_dir / "wcs_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"[wcs] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
