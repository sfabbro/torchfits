#!/usr/bin/env python3
"""Legacy WCS-only benchmark runner (PyAST/Kapteyn) for cross-env aggregation."""

from __future__ import annotations

import argparse
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
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
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
        span_x, span_y = ((60.0, 40.0) if profile == "interior" else (140.0, 90.0))
    elif case.is_cylindrical:
        span_x, span_y = ((180.0, 90.0) if profile == "interior" else (320.0, 160.0))
    else:
        span_x, span_y = ((900.0, 900.0) if profile == "interior" else (1600.0, 1600.0))

    x = cx + rng.uniform(-span_x, span_x, size=n_points)
    y = cy + rng.uniform(-span_y, span_y, size=n_points)
    x = np.clip(x, 0.0, float(header["NAXIS1"] - 1.0)).astype(np.float64)
    y = np.clip(y, 0.0, float(header["NAXIS2"] - 1.0)).astype(np.float64)
    return x, y


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
    throughput = (n_points / time_s / 1e6) if (time_s is not None and time_s > 0) else None
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


def _time_pyast(header: dict[str, Any], x: np.ndarray, y: np.ndarray, runs: int) -> float | None:
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


def _time_kapteyn(header: dict[str, Any], x: np.ndarray, y: np.ndarray, runs: int) -> float | None:
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
    parser.add_argument("--n-tiers", type=str, default="1000,10000,100000,1000000,10000000")
    parser.add_argument("--sample-profile", choices=["interior", "boundary", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    n_tiers = [int(x.strip()) for x in args.n_tiers.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    for i_case, case_name in enumerate(REQUIRED_PROJECTIONS):
        case = CASES[case_name]
        for i_tier, n_points in enumerate(n_tiers):
            runs = _runs_for_points(n_points)
            rng = np.random.default_rng(args.seed + i_case * 100 + i_tier)
            header = _make_header(case)
            # Validate the WCS payload and ensure parity with the main sweep inputs.
            _ = AstropyWCS(header)
            x, y = _sample_pixels(header, n_points, rng, case, args.sample_profile)

            print(
                f"[wcs-legacy] case={case_name} n={n_points} runs={runs}",
                flush=True,
            )

            header_dict = dict(header)
            try:
                pyast_t = _time_pyast(header_dict, x, y, runs=runs)
                if pyast_t is None:
                    rows.append(
                        _row(
                            case_name=case_name,
                            projection=case.projection,
                            n_points=n_points,
                            library="pyast",
                            status="SKIPPED",
                            skip_reason="pyast_not_available",
                            time_s=None,
                        )
                    )
                else:
                    rows.append(
                        _row(
                            case_name=case_name,
                            projection=case.projection,
                            n_points=n_points,
                            library="pyast",
                            status="OK",
                            skip_reason="",
                            time_s=pyast_t,
                        )
                    )
            except Exception as exc:
                rows.append(
                    _row(
                        case_name=case_name,
                        projection=case.projection,
                        n_points=n_points,
                        library="pyast",
                        status="SKIPPED",
                        skip_reason=f"pyast_failed: {type(exc).__name__}: {exc}",
                        time_s=None,
                    )
                )

            try:
                kapteyn_t = _time_kapteyn(header_dict, x, y, runs=runs)
                if kapteyn_t is None:
                    rows.append(
                        _row(
                            case_name=case_name,
                            projection=case.projection,
                            n_points=n_points,
                            library="kapteyn",
                            status="SKIPPED",
                            skip_reason="kapteyn_not_available",
                            time_s=None,
                        )
                    )
                else:
                    rows.append(
                        _row(
                            case_name=case_name,
                            projection=case.projection,
                            n_points=n_points,
                            library="kapteyn",
                            status="OK",
                            skip_reason="",
                            time_s=kapteyn_t,
                        )
                    )
            except Exception as exc:
                rows.append(
                    _row(
                        case_name=case_name,
                        projection=case.projection,
                        n_points=n_points,
                        library="kapteyn",
                        status="SKIPPED",
                        skip_reason=f"kapteyn_failed: {type(exc).__name__}: {exc}",
                        time_s=None,
                    )
                )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[wcs-legacy] wrote {len(rows)} rows to {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
