#!/usr/bin/env python3
"""Legacy WCS-only benchmark runner (PyAST/Kapteyn) for cross-env aggregation."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

try:
    import starlink.Ast as Ast  # type: ignore

    HAS_PYAST = True
except Exception:
    Ast = None
    HAS_PYAST = False

try:
    from kapteyn import wcs as kwcs  # type: ignore

    HAS_KAPTEYN = True
except Exception:
    kwcs = None
    HAS_KAPTEYN = False


@dataclass(frozen=True)
class CaseSpec:
    name: str
    projection: str
    sip: bool = False
    tpv: bool = False

    @property
    def is_allsky(self) -> bool:
        return self.projection in {"AIT", "MOL", "HPX"}

    @property
    def is_cylindrical(self) -> bool:
        return self.projection in {"CEA", "MER", "CAR", "SFL"}


CASES: dict[str, CaseSpec] = {
    "TAN": CaseSpec("TAN", "TAN"),
    "SIN": CaseSpec("SIN", "SIN"),
    "ARC": CaseSpec("ARC", "ARC"),
    "ZEA": CaseSpec("ZEA", "ZEA"),
    "STG": CaseSpec("STG", "STG"),
    "ZPN": CaseSpec("ZPN", "ZPN"),
    "AIT": CaseSpec("AIT", "AIT"),
    "MOL": CaseSpec("MOL", "MOL"),
    "HPX": CaseSpec("HPX", "HPX"),
    "CEA": CaseSpec("CEA", "CEA"),
    "MER": CaseSpec("MER", "MER"),
    "CAR": CaseSpec("CAR", "CAR"),
    "SFL": CaseSpec("SFL", "SFL"),
    "TAN_SIP": CaseSpec("TAN_SIP", "TAN", sip=True),
    "TPV": CaseSpec("TPV", "TPV", tpv=True),
}


REQUIRED_PROJECTIONS = list(CASES.keys())
WCS_SAMPLE_SEED_BASE = 12345


def _runs_for_points(n_points: int) -> int:
    if n_points <= 10_000:
        return 9
    if n_points <= 100_000:
        return 7
    if n_points <= 1_000_000:
        return 5
    return 3


def _time_median(fn, *, runs: int, warmup: int = 1) -> float:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    min_total_s = 0.20 if runs <= 7 else 0.0
    max_runs = max(int(runs), int(runs) * 8)
    elapsed_total = 0.0
    n_done = 0
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        while True:
            if n_done >= max(1, runs) and elapsed_total >= min_total_s:
                break
            if n_done >= max_runs:
                break
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            samples.append(dt)
            elapsed_total += dt
            n_done += 1
    finally:
        if gc_enabled:
            gc.enable()
    return float(np.median(samples))


def _make_header(case: CaseSpec) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4096
    header["NAXIS2"] = 4096
    header["CRPIX1"] = 2048.0
    header["CRPIX2"] = 2048.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 0.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    ctype1 = f"RA---{case.projection}"
    ctype2 = f"DEC--{case.projection}"
    if case.sip:
        ctype1 += "-SIP"
        ctype2 += "-SIP"
    header["CTYPE1"] = ctype1
    header["CTYPE2"] = ctype2

    if case.is_allsky:
        header["CRVAL1"] = 0.0
        header["CRVAL2"] = 0.0
        header["CD1_1"] = -1.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0
    elif case.is_cylindrical:
        header["CD1_1"] = -0.5
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.5
        if case.projection == "CEA":
            header["PV2_1"] = 1.0
    else:
        header["CD1_1"] = -2.8e-4
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 2.8e-4
        if case.projection == "ZPN":
            header["PV2_1"] = 1.0

    if case.sip:
        header["A_ORDER"] = 3
        header["B_ORDER"] = 3
        header["A_2_0"] = 3.0e-6
        header["A_1_1"] = -2.0e-6
        header["A_0_2"] = 1.0e-6
        header["B_2_0"] = -2.5e-6
        header["B_1_1"] = 2.0e-6
        header["B_0_2"] = 2.5e-6

    if case.tpv:
        header["PV1_1"] = 1.0
        header["PV1_4"] = 2.0e-4
        header["PV1_5"] = -3.0e-4
        header["PV1_7"] = 2.0e-6
        header["PV1_39"] = 2.0e-11
        header["PV2_2"] = 1.0
        header["PV2_4"] = -1.0e-4
        header["PV2_5"] = 2.5e-4
        header["PV2_7"] = -1.5e-6
        header["PV2_39"] = -1.0e-11
    return header


def _sample_pixels(
    header: fits.Header,
    n_points: int,
    rng: np.random.Generator,
    case: CaseSpec,
    profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    if profile == "mixed":
        n0 = n_points // 2
        n1 = n_points - n0
        x0, y0 = _sample_pixels(header, n0, rng, case, "interior")
        x1, y1 = _sample_pixels(header, n1, rng, case, "boundary")
        return np.concatenate([x0, x1]), np.concatenate([y0, y1])

    cx = float(header["CRPIX1"] - 1.0)
    cy = float(header["CRPIX2"] - 1.0)
    if case.is_allsky:
        span_x, span_y = (60.0, 40.0) if profile == "interior" else (140.0, 90.0)
    elif case.is_cylindrical:
        span_x, span_y = (180.0, 90.0) if profile == "interior" else (320.0, 160.0)
    else:
        span_x, span_y = (900.0, 900.0) if profile == "interior" else (1600.0, 1600.0)

    x = cx + rng.uniform(-span_x, span_x, size=n_points)
    y = cy + rng.uniform(-span_y, span_y, size=n_points)
    x = np.clip(x, 0.0, float(header["NAXIS1"] - 1.0)).astype(np.float64)
    y = np.clip(y, 0.0, float(header["NAXIS2"] - 1.0)).astype(np.float64)
    return x, y


def _sample_allsky_pixels_from_world(
    awcs: AstropyWCS,
    header: fits.Header,
    case: CaseSpec,
    n_points: int,
    rng: np.random.Generator,
    profile: str,
    origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    if profile == "mixed":
        n0 = n_points // 2
        n1 = n_points - n0
        x0, y0 = _sample_allsky_pixels_from_world(
            awcs, header, case, n0, rng, "interior", origin
        )
        x1, y1 = _sample_allsky_pixels_from_world(
            awcs, header, case, n1, rng, "boundary", origin
        )
        return np.concatenate([x0, x1]), np.concatenate([y0, y1])

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total = 0

    min_x = 0.0 if origin == 0 else 1.0
    min_y = 0.0 if origin == 0 else 1.0
    max_x = float(header["NAXIS1"] - 1.0) if origin == 0 else float(header["NAXIS1"])
    max_y = float(header["NAXIS2"] - 1.0) if origin == 0 else float(header["NAXIS2"])

    for _ in range(12):
        if total >= n_points:
            break
        batch = max((n_points - total) * 4, 8192)
        ra = rng.uniform(0.0, 360.0, size=batch)

        if profile == "interior":
            u = rng.uniform(
                -np.sin(np.deg2rad(60.0)), np.sin(np.deg2rad(60.0)), size=batch
            )
            dec = np.rad2deg(np.arcsin(u))
        else:
            sign = np.where(rng.random(size=batch) < 0.5, -1.0, 1.0)
            dec = sign * rng.uniform(60.0, 89.9, size=batch)

        x, y = awcs.all_world2pix(ra, dec, origin)
        valid = np.isfinite(x) & np.isfinite(y)
        valid &= (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

        if np.any(valid):
            x_parts.append(np.asarray(x[valid], dtype=np.float64))
            y_parts.append(np.asarray(y[valid], dtype=np.float64))
            total += int(np.sum(valid))

    if total < n_points:
        return _sample_pixels(header, n_points, rng, case, profile)

    x_all = np.concatenate(x_parts)[:n_points]
    y_all = np.concatenate(y_parts)[:n_points]
    return x_all, y_all


def _sample_seed(i_case: int, i_tier: int, i_rep: int, base_seed: int) -> int:
    if base_seed == WCS_SAMPLE_SEED_BASE:
        return int(WCS_SAMPLE_SEED_BASE + i_case * 100_000 + i_tier * 1_000 + i_rep)
    return int(base_seed + i_case * 100_000 + i_tier * 1_000 + i_rep)


def _row(
    *,
    case_name: str,
    projection: str,
    n_points: int,
    library: str,
    status: str,
    skip_reason: str,
    time_s: float | None,
) -> dict[str, Any]:
    throughput = (
        (n_points / time_s / 1e6) if (time_s is not None and time_s > 0) else None
    )
    return {
        "suite": "wcs_legacy",
        "case_id": f"{case_name}::n{n_points}::forward",
        "case_label": f"{case_name} n={n_points} [forward]",
        "operation": "forward",
        "library": library,
        "method": library,
        "status": status,
        "skip_reason": skip_reason,
        "time_s": time_s,
        "throughput": throughput,
        "unit": "Mpts/s",
        "n_points": n_points,
        "metadata": {"projection": projection, "legacy": True},
    }


def _time_pyast(
    header: dict[str, Any], x: np.ndarray, y: np.ndarray, runs: int
) -> float | None:
    if not HAS_PYAST:
        return None
    fc = Ast.FitsChan()
    for k, v in header.items():
        if isinstance(v, str):
            fc.putfits(f"{k:8}= '{v}'", False)
        else:
            fc.putfits(f"{k:8}= {v}", False)
    w = fc.read()
    if w is None:
        return None
    coords = np.stack([x, y])
    return _time_median(lambda: w.tran(coords, True), runs=runs)


def _time_kapteyn(
    header: dict[str, Any], x: np.ndarray, y: np.ndarray, runs: int
) -> float | None:
    if not HAS_KAPTEYN:
        return None
    h = {str(k): v for k, v in header.items()}
    h.setdefault("NAXIS", 2)
    h.setdefault("NAXIS1", 4096)
    h.setdefault("NAXIS2", 4096)
    wh = kwcs.WrappedHeader(h, "")
    proj = kwcs.Projection(wh)
    coords = np.stack([x, y], axis=-1)
    return _time_median(lambda: proj.toworld(coords), runs=runs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-tiers", type=str, default="1000,10000,100000,1000000,10000000"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated projection cases to run (default: all required)",
    )
    parser.add_argument(
        "--sample-profile", choices=["interior", "boundary", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of repeated seeded runs per case/tier (aggregated by median)",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--origin", type=int, default=0, choices=[0, 1])
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    n_tiers = [int(x.strip()) for x in args.n_tiers.split(",") if x.strip()]
    requested = (
        [x.strip() for x in args.cases.split(",") if x.strip()]
        if args.cases.strip()
        else list(REQUIRED_PROJECTIONS)
    )
    unknown = [name for name in requested if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown WCS legacy cases: {', '.join(unknown)}")

    rows: list[dict[str, Any]] = []

    for i_case, case_name in enumerate(requested):
        case = CASES[case_name]
        for i_tier, n_points in enumerate(n_tiers):
            runs = _runs_for_points(n_points)
            rep_count = max(1, int(args.replicates))
            pyast_times: list[float] = []
            kapteyn_times: list[float] = []
            pyast_err = ""
            kapteyn_err = ""

            print(
                f"[wcs-legacy] case={case_name} n={n_points} runs={runs} reps={rep_count}",
                flush=True,
            )

            for i_rep in range(rep_count):
                sample_seed = _sample_seed(i_case, i_tier, i_rep, args.seed)
                rng = np.random.default_rng(sample_seed)
                header = _make_header(case)
                awcs = AstropyWCS(header)
                if case.is_allsky:
                    x, y = _sample_allsky_pixels_from_world(
                        awcs,
                        header,
                        case,
                        n_points,
                        rng,
                        args.sample_profile,
                        args.origin,
                    )
                else:
                    x, y = _sample_pixels(
                        header, n_points, rng, case, args.sample_profile
                    )

                header_dict = dict(header)
                try:
                    pyast_t = _time_pyast(header_dict, x, y, runs=runs)
                    if pyast_t is not None:
                        pyast_times.append(pyast_t)
                except Exception as exc:
                    if not pyast_err:
                        pyast_err = f"pyast_failed: {type(exc).__name__}: {exc}"

                try:
                    kapteyn_t = _time_kapteyn(header_dict, x, y, runs=runs)
                    if kapteyn_t is not None:
                        kapteyn_times.append(kapteyn_t)
                except Exception as exc:
                    if not kapteyn_err:
                        kapteyn_err = f"kapteyn_failed: {type(exc).__name__}: {exc}"

            pyast_med = float(np.median(pyast_times)) if pyast_times else None
            rows.append(
                _row(
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n_points,
                    library="pyast",
                    status="OK" if pyast_med is not None else "SKIPPED",
                    skip_reason=(
                        ""
                        if pyast_med is not None
                        else (pyast_err or "pyast_not_available")
                    ),
                    time_s=pyast_med,
                )
            )

            kapteyn_med = float(np.median(kapteyn_times)) if kapteyn_times else None
            rows.append(
                _row(
                    case_name=case_name,
                    projection=case.projection,
                    n_points=n_points,
                    library="kapteyn",
                    status="OK" if kapteyn_med is not None else "SKIPPED",
                    skip_reason=(
                        ""
                        if kapteyn_med is not None
                        else (kapteyn_err or "kapteyn_not_available")
                    ),
                    time_s=kapteyn_med,
                )
            )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[wcs-legacy] wrote {len(rows)} rows to {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
