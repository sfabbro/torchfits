#!/usr/bin/env python3
"""Optional cross-library WCS benchmark (Astropy, TorchFits, PyAST, Kapteyn)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchfits.wcs.core import WCS as TorchWCS
from bench_wcs import CASES, _make_header, _sample_pixels

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


def _time_one(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _angular_sep_deg(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    r1 = np.deg2rad(ra1)
    d1 = np.deg2rad(dec1)
    r2 = np.deg2rad(ra2)
    d2 = np.deg2rad(dec2)
    dr = r1 - r2
    dd = d1 - d2
    a = np.sin(dd / 2.0) ** 2 + np.cos(d1) * np.cos(d2) * np.sin(dr / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _run_pyast(header: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    if not HAS_PYAST:
        return None

    try:
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
        dt = _time_one(lambda: w.tran(coords, True))
        out = w.tran(coords, True)
        # PyAST returns radians for celestial outputs.
        ra = np.rad2deg(out[0]) % 360.0
        dec = np.rad2deg(out[1])
        return dt, ra, dec
    except Exception:
        return None


def _run_kapteyn(header: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    if not HAS_KAPTEYN:
        return None

    try:
        h = {str(k): v for k, v in header.items()}
        h.setdefault("NAXIS", 2)
        h.setdefault("NAXIS1", 2000)
        h.setdefault("NAXIS2", 2000)
        wh = kwcs.WrappedHeader(h, "")
        proj = kwcs.Projection(wh)
        coords = np.stack([x, y], axis=-1)

        dt = _time_one(lambda: proj.toworld(coords))
        out = proj.toworld(coords)
        return dt, out[:, 0], out[:, 1]
    except Exception:
        return None


def run_case(case_name: str, n_points: int, origin: int, seed: int, profile: str) -> dict[str, Any]:
    case = CASES[case_name]
    header = _make_header(case)
    header_dict = dict(header)

    rng = np.random.default_rng(seed)
    x, y = _sample_pixels(header, n_points, rng, case, profile)

    awcs = AstropyWCS(header)
    twcs = TorchWCS(header_dict)

    ast_dt = _time_one(lambda: awcs.all_pix2world(x, y, origin))
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    torch_dt = _time_one(lambda: twcs.pixel_to_world(x_t, y_t, origin=origin))
    ra_t, dec_t = twcs.pixel_to_world(x_t, y_t, origin=origin)

    ra_t_np = ra_t.detach().cpu().numpy()
    dec_t_np = dec_t.detach().cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)

    result: dict[str, Any] = {
        "case": case_name,
        "sample_profile": profile,
        "n_points": int(n_points),
        "n_valid": int(np.sum(valid)),
        "astropy_ms": ast_dt * 1000.0,
        "torchfits_ms": torch_dt * 1000.0,
        "torchfits_speedup": ast_dt / torch_dt,
        "torchfits_max_angular_error_arcsec": float(np.max(_angular_sep_deg(ra_t_np[valid], dec_t_np[valid], ra_ast[valid], dec_ast[valid])) * 3600.0),
    }

    pyast = _run_pyast(header_dict, x, y)
    if pyast is not None:
        pyast_dt, ra_p, dec_p = pyast
        valid_p = valid & np.isfinite(ra_p) & np.isfinite(dec_p)
        result["pyast_ms"] = pyast_dt * 1000.0
        result["pyast_speedup"] = ast_dt / pyast_dt
        result["pyast_max_angular_error_arcsec"] = float(
            np.max(_angular_sep_deg(ra_p[valid_p], dec_p[valid_p], ra_ast[valid_p], dec_ast[valid_p])) * 3600.0
        )
    else:
        result["pyast_ms"] = float("nan")
        result["pyast_speedup"] = float("nan")
        result["pyast_max_angular_error_arcsec"] = float("nan")

    kapteyn = _run_kapteyn(header_dict, x, y)
    if kapteyn is not None:
        kap_dt, ra_k, dec_k = kapteyn
        valid_k = valid & np.isfinite(ra_k) & np.isfinite(dec_k)
        result["kapteyn_ms"] = kap_dt * 1000.0
        result["kapteyn_speedup"] = ast_dt / kap_dt
        result["kapteyn_max_angular_error_arcsec"] = float(
            np.max(_angular_sep_deg(ra_k[valid_k], dec_k[valid_k], ra_ast[valid_k], dec_ast[valid_k])) * 3600.0
        )
    else:
        result["kapteyn_ms"] = float("nan")
        result["kapteyn_speedup"] = float("nan")
        result["kapteyn_max_angular_error_arcsec"] = float("nan")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=str, default="TAN,TAN_SIP,TPV,AIT,HPX")
    parser.add_argument("--n-points", type=int, default=100_000)
    parser.add_argument("--origin", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample-profile",
        choices=["interior", "boundary", "mixed"],
        default="mixed",
        help="Pixel sampling profile: interior, boundary, or mixed",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = [x.strip() for x in args.cases.split(",") if x.strip()]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(unknown)}")

    results = [
        run_case(name, args.n_points, args.origin, args.seed + i, args.sample_profile)
        for i, name in enumerate(names)
    ]

    print(
        f"PyAST={'yes' if HAS_PYAST else 'no'} Kapteyn={'yes' if HAS_KAPTEYN else 'no'}"
    )
    cols = [
        "case",
        "astropy_ms",
        "torchfits_ms",
        "torchfits_speedup",
        "torchfits_max_angular_error_arcsec",
        "pyast_ms",
        "kapteyn_ms",
    ]
    print(" ".join(f"{c:>16s}" for c in cols))
    for r in results:
        print(
            f"{r['case']:>16s}"
            f"{r['astropy_ms']:16.3f}"
            f"{r['torchfits_ms']:16.3f}"
            f"{r['torchfits_speedup']:16.2f}"
            f"{r['torchfits_max_angular_error_arcsec']:16.4g}"
            f"{r['pyast_ms']:16.3f}"
            f"{r['kapteyn_ms']:16.3f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
