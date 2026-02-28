#!/usr/bin/env python3
"""Authoritative WCS benchmark sweep runner."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench_contract import RESULT_COLUMNS, annotate_rankings, write_csv
from bench_wcs import CASES, _bench_case, _resolve_device

try:
    from bench_competitors import run_case as run_legacy_case
except Exception:
    run_legacy_case = None


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
) -> list[dict[str, Any]]:
    _ = output_dir
    tiers = n_tiers or [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    requested = projections or list(REQUIRED_PROJECTIONS)

    device = _resolve_device(device_choice)
    rows: list[dict[str, Any]] = []

    for case_name in requested:
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

        for i, n in enumerate(tiers):
            runs = _runs_for_points(n)
            rng = np.random.default_rng(12345 + i)
            print(
                f"[wcs] case={case_name} n={n} runs={runs} device={device.type}",
                flush=True,
            )
            try:
                result = _bench_case(
                    case,
                    n_points=n,
                    runs=runs,
                    device=device,
                    origin=origin,
                    torch_compile=False,
                    profile=sample_profile,
                    rng=rng,
                )
            except Exception as exc:
                reason = f"case_failed: {exc}"
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

            tf_fw = _ok_time(result.get("torch_forward_ms", 0.0) / 1000.0)
            as_fw = _ok_time(result.get("astropy_forward_ms", 0.0) / 1000.0)
            tf_inv = _ok_time(result.get("torch_inverse_ms", 0.0) / 1000.0)
            as_inv = _ok_time(result.get("astropy_inverse_ms", 0.0) / 1000.0)

            meta = {
                "n_valid": result.get("n_valid"),
                "inverse_reference": result.get("inverse_reference"),
                "max_angular_error_arcsec": result.get("max_angular_error_arcsec"),
                "p99_angular_error_arcsec": result.get("p99_angular_error_arcsec"),
                "max_inverse_pixel_error": result.get("max_inverse_pixel_error"),
                "p99_inverse_pixel_error": result.get("p99_inverse_pixel_error"),
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
                    throughput=float(result.get("torch_forward_mpts_s", 0.0)),
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
                    throughput=float(result.get("astropy_forward_mpts_s", 0.0)),
                    status="OK" if as_fw is not None else "FAILED",
                    skip_reason="",
                    comparable=as_fw is not None,
                    sample_profile=sample_profile,
                    device="cpu",
                    metadata=meta,
                )
            )

            inv_ok = result.get("inverse_reference") not in {"none", None}
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
                    skip_reason="inverse_not_available" if not inv_ok else "",
                    comparable=bool(inv_ok and as_inv is not None),
                    sample_profile=sample_profile,
                    device="cpu",
                    metadata=meta,
                )
            )

            if include_legacy and run_legacy_case is not None:
                try:
                    leg = run_legacy_case(case_name, n, origin, 123 + i, sample_profile)
                except Exception as exc:
                    leg = {"error": str(exc)}
                for lib_key, mode in (("pyast", "specialized"), ("kapteyn", "specialized")):
                    ms = leg.get(f"{lib_key}_ms") if isinstance(leg, dict) else None
                    t = _ok_time((float(ms) / 1000.0) if ms is not None else None)
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
                            skip_reason="not_installed_or_failed" if status != "OK" else "",
                            comparable=False,
                            sample_profile=sample_profile,
                            device="cpu",
                            metadata={"legacy": True},
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
    parser.add_argument("--sample-profile", choices=["interior", "boundary", "mixed"], default="mixed")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--n-tiers", type=str, default="1000,10000,100000,1000000,10000000")
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
    )

    out_csv = run_dir / "wcs_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"[wcs] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
