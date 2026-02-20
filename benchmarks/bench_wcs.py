#!/usr/bin/env python3
"""Benchmark TorchFits WCS performance and parity against Astropy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchfits.wcs.core import WCS as TorchWCS  # noqa: E402


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
        return self.projection in {"CEA", "MER"}


CASES = {
    "TAN": CaseSpec("TAN", "TAN"),
    "SIN": CaseSpec("SIN", "SIN"),
    "ARC": CaseSpec("ARC", "ARC"),
    "AIT": CaseSpec("AIT", "AIT"),
    "MOL": CaseSpec("MOL", "MOL"),
    "HPX": CaseSpec("HPX", "HPX"),
    "CEA": CaseSpec("CEA", "CEA"),
    "MER": CaseSpec("MER", "MER"),
    "TAN_SIP": CaseSpec("TAN_SIP", "TAN", sip=True),
    "TPV": CaseSpec("TPV", "TPV", tpv=True),
}


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
        if profile == "interior":
            span_x, span_y = 60.0, 40.0
        else:
            span_x, span_y = 140.0, 90.0
    elif case.is_cylindrical:
        if profile == "interior":
            span_x, span_y = 180.0, 90.0
        else:
            span_x, span_y = 320.0, 160.0
    else:
        if profile == "interior":
            span_x, span_y = 900.0, 900.0
        else:
            span_x, span_y = 1600.0, 1600.0

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
        x0, y0 = _sample_allsky_pixels_from_world(awcs, header, case, n0, rng, "interior", origin)
        x1, y1 = _sample_allsky_pixels_from_world(awcs, header, case, n1, rng, "boundary", origin)
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
            u = rng.uniform(-np.sin(np.deg2rad(60.0)), np.sin(np.deg2rad(60.0)), size=batch)
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _time_many(fn, runs: int, sync_device: torch.device | None = None) -> float:
    fn()
    if sync_device is not None:
        _sync(sync_device)

    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_device is not None:
            _sync(sync_device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _angular_sep_deg(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    r1 = np.deg2rad(ra1)
    d1 = np.deg2rad(dec1)
    r2 = np.deg2rad(ra2)
    d2 = np.deg2rad(dec2)
    dr = r1 - r2
    dd = d1 - d2
    a = np.sin(dd / 2.0) ** 2 + np.cos(d1) * np.cos(d2) * np.sin(dr / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _bench_case(
    case: CaseSpec,
    n_points: int,
    runs: int,
    device: torch.device,
    origin: int,
    torch_compile: bool,
    profile: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    header = _make_header(case)
    awcs = AstropyWCS(header)
    if case.is_allsky:
        x, y = _sample_allsky_pixels_from_world(
            awcs,
            header,
            case,
            n_points,
            rng,
            profile,
            origin,
        )
    else:
        x, y = _sample_pixels(header, n_points, rng, case, profile)

    twcs = TorchWCS(dict(header)).to(device)
    if torch_compile and hasattr(twcs, "compile"):
        twcs.compile(mode="reduce-overhead")

    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float64)
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.float64)

    with torch.no_grad():
        ast_fw_s = _time_many(lambda: awcs.all_pix2world(x, y, origin), runs)
        torch_fw_s = _time_many(lambda: twcs.pixel_to_world(x_t, y_t, origin=origin), runs, sync_device=device)

        ra_a, dec_a = awcs.all_pix2world(x, y, origin)
        ra_t, dec_t = twcs.pixel_to_world(x_t, y_t, origin=origin)

    ra_t_np = ra_t.detach().cpu().numpy()
    dec_t_np = dec_t.detach().cpu().numpy()

    valid = np.isfinite(ra_a) & np.isfinite(dec_a)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)

    if np.any(valid):
        ang_err_deg = _angular_sep_deg(ra_t_np[valid], dec_t_np[valid], ra_a[valid], dec_a[valid])
        max_ang_deg = float(np.max(ang_err_deg))
        med_ang_deg = float(np.median(ang_err_deg))
        p90_ang_deg, p99_ang_deg = [float(v) for v in np.quantile(ang_err_deg, [0.9, 0.99])]

        ra_valid = ra_a[valid]
        dec_valid = dec_a[valid]
        x_valid = x[valid]
        y_valid = y[valid]

        ra_v_t = torch.from_numpy(ra_valid).to(device=device, dtype=torch.float64)
        dec_v_t = torch.from_numpy(dec_valid).to(device=device, dtype=torch.float64)

        with torch.no_grad():
            torch_inv_s = _time_many(
                lambda: twcs.world_to_pixel(ra_v_t, dec_v_t, origin=origin),
                runs,
                sync_device=device,
            )
            x_t_inv, y_t_inv = twcs.world_to_pixel(ra_v_t, dec_v_t, origin=origin)

        x_t_inv_np = x_t_inv.detach().cpu().numpy()
        y_t_inv_np = y_t_inv.detach().cpu().numpy()

        # Always report internal round-trip accuracy for inverse stability.
        roundtrip_err = np.hypot(x_t_inv_np - x_valid, y_t_inv_np - y_valid)
        max_roundtrip_pix_err = float(np.max(roundtrip_err))
        med_roundtrip_pix_err = float(np.median(roundtrip_err))
        p90_roundtrip_pix_err, p99_roundtrip_pix_err = [
            float(v) for v in np.quantile(roundtrip_err, [0.9, 0.99])
        ]

        # Prefer Astropy inverse parity when available; TPV/all-sky can fail to invert
        # for some valid forward points and should not crash the benchmark.
        try:
            with torch.no_grad():
                ast_inv_s = _time_many(
                    lambda: awcs.all_world2pix(ra_valid, dec_valid, origin),
                    runs,
                )
            x_a_inv, y_a_inv = awcs.all_world2pix(ra_valid, dec_valid, origin)
            pix_err = np.hypot(x_t_inv_np - x_a_inv, y_t_inv_np - y_a_inv)
            max_pix_err = float(np.max(pix_err))
            med_pix_err = float(np.median(pix_err))
            p90_pix_err, p99_pix_err = [float(v) for v in np.quantile(pix_err, [0.9, 0.99])]
            inverse_reference = "astropy"
        except Exception:
            ast_inv_s = float("nan")
            max_pix_err = max_roundtrip_pix_err
            med_pix_err = med_roundtrip_pix_err
            p90_pix_err = p90_roundtrip_pix_err
            p99_pix_err = p99_roundtrip_pix_err
            inverse_reference = "roundtrip"

        valid_count = int(np.sum(valid))
    else:
        valid_count = 0
        max_ang_deg = float("nan")
        med_ang_deg = float("nan")
        p90_ang_deg = float("nan")
        p99_ang_deg = float("nan")
        ast_inv_s = float("nan")
        torch_inv_s = float("nan")
        max_pix_err = float("nan")
        med_pix_err = float("nan")
        p90_pix_err = float("nan")
        p99_pix_err = float("nan")
        max_roundtrip_pix_err = float("nan")
        med_roundtrip_pix_err = float("nan")
        p90_roundtrip_pix_err = float("nan")
        p99_roundtrip_pix_err = float("nan")
        inverse_reference = "none"

    result = {
        "case": case.name,
        "projection": case.projection,
        "sample_profile": profile,
        "device": device.type,
        "origin": origin,
        "n_points": int(n_points),
        "n_valid": valid_count,
        "astropy_forward_ms": ast_fw_s * 1000.0,
        "torch_forward_ms": torch_fw_s * 1000.0,
        "forward_speedup": ast_fw_s / torch_fw_s if torch_fw_s > 0 else float("nan"),
        "astropy_inverse_ms": ast_inv_s * 1000.0,
        "torch_inverse_ms": torch_inv_s * 1000.0,
        "inverse_speedup": ast_inv_s / torch_inv_s if torch_inv_s > 0 else float("nan"),
        "inverse_reference": inverse_reference,
        "torch_forward_mpts_s": (n_points / torch_fw_s) / 1e6,
        "astropy_forward_mpts_s": (n_points / ast_fw_s) / 1e6,
        "max_angular_error_arcsec": max_ang_deg * 3600.0,
        "median_angular_error_arcsec": med_ang_deg * 3600.0,
        "p90_angular_error_arcsec": p90_ang_deg * 3600.0,
        "p99_angular_error_arcsec": p99_ang_deg * 3600.0,
        "max_inverse_pixel_error": max_pix_err,
        "median_inverse_pixel_error": med_pix_err,
        "p90_inverse_pixel_error": p90_pix_err,
        "p99_inverse_pixel_error": p99_pix_err,
        "max_roundtrip_pixel_error": max_roundtrip_pix_err,
        "median_roundtrip_pixel_error": med_roundtrip_pix_err,
        "p90_roundtrip_pixel_error": p90_roundtrip_pix_err,
        "p99_roundtrip_pixel_error": p99_roundtrip_pix_err,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=str,
        default="TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV",
        help=f"Comma-separated case names. Available: {','.join(CASES.keys())}",
    )
    parser.add_argument("--n-points", type=int, default=200_000, help="Number of pixel samples per case")
    parser.add_argument("--runs", type=int, default=5, help="Timing repetitions per operation")
    parser.add_argument("--origin", type=int, default=0, choices=[0, 1], help="Pixel origin convention")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--sample-profile",
        choices=["interior", "boundary", "mixed"],
        default="mixed",
        help="Pixel sampling profile: interior, boundary, or mixed",
    )
    parser.add_argument("--torch-compile", action="store_true", help="Benchmark torch.compile()d transform")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--max-angular-error-arcsec",
        type=float,
        default=None,
        help="Optional fail threshold for max angular error",
    )
    parser.add_argument(
        "--max-inverse-pixel-error",
        type=float,
        default=None,
        help="Optional fail threshold for max inverse pixel error",
    )
    parser.add_argument(
        "--max-roundtrip-pixel-error",
        type=float,
        default=None,
        help="Optional fail threshold for max internal roundtrip pixel error",
    )
    parser.add_argument(
        "--p99-angular-error-arcsec",
        type=float,
        default=None,
        help="Optional fail threshold for p99 angular error",
    )
    parser.add_argument(
        "--p99-inverse-pixel-error",
        type=float,
        default=None,
        help="Optional fail threshold for p99 inverse pixel error",
    )
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(choice)


def _print_results(results: list[dict[str, Any]]) -> None:
    header = (
        "case",
        "profile",
        "valid",
        "ast_fw_ms",
        "torch_fw_ms",
        "fw_x",
        "ast_inv_ms",
        "torch_inv_ms",
        "inv_x",
        "max_ang_asec",
        "p99_ang_asec",
        "max_pix",
        "p99_pix",
    )
    print(" ".join(f"{col:>12s}" for col in header))
    for r in results:
        print(
            f"{r['case']:>12s}"
            f"{r['sample_profile']:>12s}"
            f"{r['n_valid']:12d}"
            f"{r['astropy_forward_ms']:12.3f}"
            f"{r['torch_forward_ms']:12.3f}"
            f"{r['forward_speedup']:12.2f}"
            f"{r['astropy_inverse_ms']:12.3f}"
            f"{r['torch_inverse_ms']:12.3f}"
            f"{r['inverse_speedup']:12.2f}"
            f"{r['max_angular_error_arcsec']:12.4g}"
            f"{r['p99_angular_error_arcsec']:12.4g}"
            f"{r['max_inverse_pixel_error']:12.4g}"
            f"{r['p99_inverse_pixel_error']:12.4g}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)

    requested = [x.strip() for x in args.cases.split(",") if x.strip()]
    unknown = [name for name in requested if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(unknown)}")

    rng = np.random.default_rng(args.seed)
    results = [
        _bench_case(
            CASES[name],
            n_points=args.n_points,
            runs=args.runs,
            device=device,
            origin=args.origin,
            torch_compile=args.torch_compile,
            profile=args.sample_profile,
            rng=rng,
        )
        for name in requested
    ]

    _print_results(results)

    if args.csv_out is not None:
        _write_csv(args.csv_out, results)
    if args.json_out is not None:
        _write_json(args.json_out, results)

    if args.max_angular_error_arcsec is not None:
        too_high = [
            r
            for r in results
            if np.isfinite(r["max_angular_error_arcsec"]) and r["max_angular_error_arcsec"] > args.max_angular_error_arcsec
        ]
        if too_high:
            print("\nError threshold exceeded for angular parity:")
            for row in too_high:
                print(f"  {row['case']}: {row['max_angular_error_arcsec']:.6g} arcsec")
            return 1

    if args.max_inverse_pixel_error is not None:
        too_high = [
            r
            for r in results
            if np.isfinite(r["max_inverse_pixel_error"]) and r["max_inverse_pixel_error"] > args.max_inverse_pixel_error
        ]
        if too_high:
            print("\nError threshold exceeded for inverse parity:")
            for row in too_high:
                print(f"  {row['case']}: {row['max_inverse_pixel_error']:.6g} px")
            return 1

    if args.max_roundtrip_pixel_error is not None:
        too_high = [
            r
            for r in results
            if np.isfinite(r["max_roundtrip_pixel_error"])
            and r["max_roundtrip_pixel_error"] > args.max_roundtrip_pixel_error
        ]
        if too_high:
            print("\nError threshold exceeded for internal inverse roundtrip:")
            for row in too_high:
                print(f"  {row['case']}: {row['max_roundtrip_pixel_error']:.6g} px")
            return 1

    if args.p99_angular_error_arcsec is not None:
        too_high = [
            r
            for r in results
            if np.isfinite(r["p99_angular_error_arcsec"]) and r["p99_angular_error_arcsec"] > args.p99_angular_error_arcsec
        ]
        if too_high:
            print("\nError threshold exceeded for p99 angular parity:")
            for row in too_high:
                print(f"  {row['case']}: {row['p99_angular_error_arcsec']:.6g} arcsec")
            return 1

    if args.p99_inverse_pixel_error is not None:
        too_high = [
            r
            for r in results
            if np.isfinite(r["p99_inverse_pixel_error"]) and r["p99_inverse_pixel_error"] > args.p99_inverse_pixel_error
        ]
        if too_high:
            print("\nError threshold exceeded for p99 inverse parity:")
            for row in too_high:
                print(f"  {row['case']}: {row['p99_inverse_pixel_error']:.6g} px")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
