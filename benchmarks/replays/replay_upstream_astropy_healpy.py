#!/usr/bin/env python3
"""Replay astropy-healpix test-style HEALPix parity/throughput against TorchFits."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import healpy as hp
import torch

from torchfits.wcs.healpix import (
    ang2pix_nested as tf_ang2pix_nested,
    ang2pix_ring as tf_ang2pix_ring,
    nest2ring as tf_nest2ring,
    pix2ang_nested as tf_pix2ang_nested,
    pix2ang_ring as tf_pix2ang_ring,
    ring2nest as tf_ring2nest,
)


NSIDE_VALUES = [
    2**n for n in range(1, 6)
]  # Mirrors astropy_healpix/tests/test_healpy.py
EXAMPLE_LON_LAT = [
    (1.0000000028043134e-05, -41.81031451395941),
    (1.0000000028043134e-05, 1.000000000805912e-05),
    (359.9999986588955, 41.81031489577861),
    (359.9999922886491, -41.81031470486902),
    (1.6345238095238293, 69.42254649458224),
]
FRACS = [0.0, 0.125, 0.1666666694606345, 2.0 / 3.0, 0.999999999]


@dataclass
class OpResult:
    operation: str
    nside: int
    points: int
    mismatches: int
    max_dra_deg: float | None
    max_ddec_deg: float | None
    torchfits_mpts_s: float
    healpy_mpts_s: float
    ratio_vs_healpy: float


def _ra_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _time_many(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _build_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = rng.uniform(-89.8, 89.8, size=n)
    extra_lon = np.array([x for x, _ in EXAMPLE_LON_LAT], dtype=np.float64)
    extra_lat = np.array([y for _, y in EXAMPLE_LON_LAT], dtype=np.float64)
    lon[: extra_lon.size] = extra_lon
    lat[: extra_lat.size] = extra_lat
    return lon, lat


def _run_ang2pix(
    op: str, nside: int, lon: np.ndarray, lat: np.ndarray, nest: bool, runs: int
) -> OpResult:
    lon_t = torch.from_numpy(lon).to(torch.float64)
    lat_t = torch.from_numpy(lat).to(torch.float64)
    n = lon.size

    if nest:

        def tf_fn() -> np.ndarray:
            return tf_ang2pix_nested(nside, lon_t, lat_t).cpu().numpy()

        def hp_fn() -> np.ndarray:
            return hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)
    else:

        def tf_fn() -> np.ndarray:
            return tf_ang2pix_ring(nside, lon_t, lat_t).cpu().numpy()

        def hp_fn() -> np.ndarray:
            return hp.ang2pix(nside, lon, lat, nest=False, lonlat=True)

    t_tf = _time_many(tf_fn, runs=runs)
    t_hp = _time_many(hp_fn, runs=runs)
    got = tf_fn()
    ref = hp_fn()
    mismatches = int(np.sum(got != ref))
    return OpResult(
        operation=op,
        nside=nside,
        points=n,
        mismatches=mismatches,
        max_dra_deg=None,
        max_ddec_deg=None,
        torchfits_mpts_s=(n / t_tf) / 1e6,
        healpy_mpts_s=(n / t_hp) / 1e6,
        ratio_vs_healpy=t_hp / t_tf,
    )


def _run_pix2ang(
    op: str, nside: int, pix: np.ndarray, nest: bool, runs: int
) -> OpResult:
    pix_t = torch.from_numpy(pix)
    n = pix.size

    if nest:

        def tf_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return tf_pix2ang_nested(nside, pix_t)

        def hp_fn() -> tuple[np.ndarray, np.ndarray]:
            return hp.pix2ang(nside, pix, nest=True, lonlat=True)
    else:

        def tf_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return tf_pix2ang_ring(nside, pix_t)

        def hp_fn() -> tuple[np.ndarray, np.ndarray]:
            return hp.pix2ang(nside, pix, nest=False, lonlat=True)

    t_tf = _time_many(tf_fn, runs=runs)
    t_hp = _time_many(hp_fn, runs=runs)
    ra_tf, dec_tf = tf_fn()
    ra_hp, dec_hp = hp_fn()
    ra_np = ra_tf.cpu().numpy()
    dec_np = dec_tf.cpu().numpy()
    dra = _ra_delta_deg(ra_np, np.asarray(ra_hp))
    ddec = np.abs(dec_np - np.asarray(dec_hp))
    eps = 1.0e-10
    mismatches = int(np.sum((dra > eps) | (ddec > eps)))
    return OpResult(
        operation=op,
        nside=nside,
        points=n,
        mismatches=mismatches,
        max_dra_deg=float(dra.max(initial=0.0)),
        max_ddec_deg=float(ddec.max(initial=0.0)),
        torchfits_mpts_s=(n / t_tf) / 1e6,
        healpy_mpts_s=(n / t_hp) / 1e6,
        ratio_vs_healpy=t_hp / t_tf,
    )


def _run_index_conv(
    op: str, nside: int, pix: np.ndarray, nest_to_ring: bool, runs: int
) -> OpResult:
    pix_t = torch.from_numpy(pix)
    n = pix.size
    if nest_to_ring:

        def tf_fn() -> np.ndarray:
            return tf_nest2ring(nside, pix_t).cpu().numpy()

        def hp_fn() -> np.ndarray:
            return hp.nest2ring(nside, pix)
    else:

        def tf_fn() -> np.ndarray:
            return tf_ring2nest(nside, pix_t).cpu().numpy()

        def hp_fn() -> np.ndarray:
            return hp.ring2nest(nside, pix)

    t_tf = _time_many(tf_fn, runs=runs)
    t_hp = _time_many(hp_fn, runs=runs)
    got = tf_fn()
    ref = hp_fn()
    mismatches = int(np.sum(got != ref))
    return OpResult(
        operation=op,
        nside=nside,
        points=n,
        mismatches=mismatches,
        max_dra_deg=None,
        max_ddec_deg=None,
        torchfits_mpts_s=(n / t_tf) / 1e6,
        healpy_mpts_s=(n / t_hp) / 1e6,
        ratio_vs_healpy=t_hp / t_tf,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--max-index-mismatches", type=int, default=0)
    parser.add_argument("--max-pix2ang-dra-deg", type=float, default=1.0e-10)
    parser.add_argument("--max-pix2ang-ddec-deg", type=float, default=1.0e-10)
    parser.add_argument(
        "--min-ratio-vs-healpy",
        type=str,
        default="ang2pix_ring=1.5,ang2pix_nested=1.3,pix2ang_ring=1.1,pix2ang_nested=1.0,ring2nest=1.05,nest2ring=1.3",
    )
    args = parser.parse_args()

    ratio_thresholds: dict[str, float] = {}
    for part in [p.strip() for p in args.min_ratio_vs_healpy.split(",") if p.strip()]:
        op, raw = part.split("=", 1)
        ratio_thresholds[op.strip()] = float(raw.strip())

    rows: list[OpResult] = []
    for i, nside in enumerate(NSIDE_VALUES):
        lon, lat = _build_lonlat(args.n_points, seed=args.seed + i)
        npix = 12 * nside * nside
        pix = np.random.default_rng(args.seed + 100 + i).integers(
            0, npix, size=args.n_points, dtype=np.int64
        )
        frac_cases = np.array([int(f * npix) for f in FRACS], dtype=np.int64)
        pix[: frac_cases.size] = frac_cases

        rows.append(
            _run_ang2pix("ang2pix_ring", nside, lon, lat, nest=False, runs=args.runs)
        )
        rows.append(
            _run_ang2pix("ang2pix_nested", nside, lon, lat, nest=True, runs=args.runs)
        )
        rows.append(
            _run_pix2ang("pix2ang_ring", nside, pix, nest=False, runs=args.runs)
        )
        rows.append(
            _run_pix2ang("pix2ang_nested", nside, pix, nest=True, runs=args.runs)
        )
        rows.append(
            _run_index_conv("ring2nest", nside, pix, nest_to_ring=False, runs=args.runs)
        )
        rows.append(
            _run_index_conv("nest2ring", nside, pix, nest_to_ring=True, runs=args.runs)
        )

    agg: dict[str, dict[str, float]] = {}
    for op in ratio_thresholds:
        vals = [r.ratio_vs_healpy for r in rows if r.operation == op]
        agg[op] = {"median_ratio_vs_healpy": float(np.median(vals))}

    print(
        "operation         nside  mpts/s(tf)  mpts/s(hp)   ratio   mismatches   max_dra       max_ddec"
    )
    failures: list[str] = []
    for r in rows:
        print(
            f"{r.operation:16s} {r.nside:5d} {r.torchfits_mpts_s:10.2f} {r.healpy_mpts_s:10.2f}"
            f" {r.ratio_vs_healpy:7.3f} {r.mismatches:11d}"
            f" {r.max_dra_deg if r.max_dra_deg is not None else float('nan'):11.3e}"
            f" {r.max_ddec_deg if r.max_ddec_deg is not None else float('nan'):11.3e}"
        )

        if (
            r.operation in {"ang2pix_ring", "ang2pix_nested", "ring2nest", "nest2ring"}
            and r.mismatches > args.max_index_mismatches
        ):
            failures.append(
                f"{r.operation}@nside={r.nside}: mismatches {r.mismatches} > {args.max_index_mismatches}"
            )
        if r.operation in {"pix2ang_ring", "pix2ang_nested"}:
            if (r.max_dra_deg or 0.0) > args.max_pix2ang_dra_deg:
                failures.append(
                    f"{r.operation}@nside={r.nside}: max_dra {r.max_dra_deg:.3e} > {args.max_pix2ang_dra_deg:.3e}"
                )
            if (r.max_ddec_deg or 0.0) > args.max_pix2ang_ddec_deg:
                failures.append(
                    f"{r.operation}@nside={r.nside}: max_ddec {r.max_ddec_deg:.3e} > {args.max_pix2ang_ddec_deg:.3e}"
                )

    print("\nMedian ratio vs healpy across NSIDE_VALUES:")
    for op, info in agg.items():
        med = info["median_ratio_vs_healpy"]
        thr = ratio_thresholds[op]
        print(f"  {op:15s} {med:.3f}x (threshold {thr:.3f}x)")
        if med < thr:
            failures.append(f"{op}: median ratio {med:.3f} < {thr:.3f}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nside_values": NSIDE_VALUES,
            "n_points": args.n_points,
            "runs": args.runs,
            "rows": [asdict(r) for r in rows],
            "median_ratio_vs_healpy": agg,
            "thresholds": {
                "max_index_mismatches": args.max_index_mismatches,
                "max_pix2ang_dra_deg": args.max_pix2ang_dra_deg,
                "max_pix2ang_ddec_deg": args.max_pix2ang_ddec_deg,
                "min_ratio_vs_healpy": ratio_thresholds,
            },
            "failures": failures,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
