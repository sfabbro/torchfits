#!/usr/bin/env python3
"""Replay interpolation edge-case parity/perf against official healpy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import healpy
import numpy as np
import torch

from torchfits.wcs.healpix import get_interp_val, get_interp_weights


@dataclass
class Row:
    operation: str
    nside: int
    nest: bool
    n_points: int
    mismatches: int
    max_abs_err: float
    torchfits_mpts_s: float
    healpy_mpts_s: float
    ratio_vs_healpy: float


def _time_many(fn, runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _edge_lonlat() -> tuple[np.ndarray, np.ndarray]:
    lon_base = np.array(
        [
            -720.0,
            -360.0,
            -180.0,
            -1.0e-12,
            0.0,
            1.0e-12,
            45.0,
            180.0,
            359.999999999999,
            360.0,
            720.0,
        ],
        dtype=np.float64,
    )
    lat_base = np.array(
        [-90.0, -89.999999999, -60.0, -1.0e-12, 0.0, 1.0e-12, 60.0, 89.999999999, 90.0],
        dtype=np.float64,
    )
    lon_grid, lat_grid = np.meshgrid(lon_base, lat_base, indexing="ij")
    return lon_grid.reshape(-1), lat_grid.reshape(-1)


def _edge_pixels(nside: int) -> np.ndarray:
    npix = 12 * nside * nside
    return np.array(
        sorted(
            {
                0,
                1,
                2,
                max(npix // 2 - 1, 0),
                npix // 2,
                min(npix // 2 + 1, npix - 1),
                npix - 3,
                npix - 2,
                npix - 1,
            }
        ),
        dtype=np.int64,
    )


def _interp_mismatch_count(
    pix_tf: np.ndarray,
    w_tf: np.ndarray,
    pix_hp: np.ndarray,
    w_hp: np.ndarray,
    weight_floor: float = 1.0e-8,
    weight_tol: float = 1.0e-6,
) -> int:
    keep_tf = w_tf > weight_floor
    keep_hp = w_hp > weight_floor
    p_tf = np.where(keep_tf, pix_tf, -1)
    p_hp = np.where(keep_hp, pix_hp, -1)
    v_tf = np.where(keep_tf, w_tf, 0.0)
    v_hp = np.where(keep_hp, w_hp, 0.0)
    ord_tf = np.argsort(p_tf, axis=0)
    ord_hp = np.argsort(p_hp, axis=0)
    p_tf_s = np.take_along_axis(p_tf, ord_tf, axis=0)
    p_hp_s = np.take_along_axis(p_hp, ord_hp, axis=0)
    v_tf_s = np.take_along_axis(v_tf, ord_tf, axis=0)
    v_hp_s = np.take_along_axis(v_hp, ord_hp, axis=0)
    ok_pix = np.all(p_tf_s == p_hp_s, axis=0)
    ok_w = np.all(np.abs(v_tf_s - v_hp_s) <= weight_tol, axis=0)
    return int(np.sum(~(ok_pix & ok_w)))


def _parse_ratio_spec(spec: str | None, allowed_ops: set[str]) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"invalid ratio spec segment '{part}', expected op=value")
        op, raw = part.split("=", 1)
        op = op.strip()
        if op not in allowed_ops:
            raise ValueError(f"unknown operation '{op}' in ratio spec")
        val = float(raw.strip())
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"ratio threshold must be positive for operation '{op}'")
        out[op] = val
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside-values", type=str, default="1,8,64")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--perf-repeat", type=int, default=2048)
    parser.add_argument("--max-weight-mismatches", type=int, default=0)
    parser.add_argument("--max-val-max-abs", type=float, default=1e-9)
    parser.add_argument("--min-ratio-vs-healpy", type=str, default="")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_healpy_interp_edges.json"),
    )
    args = parser.parse_args()

    nside_values = [int(x.strip()) for x in args.nside_values.split(",") if x.strip()]
    rows: list[Row] = []
    lon, lat = _edge_lonlat()
    lon_t = torch.from_numpy(lon).to(torch.float64)
    lat_t = torch.from_numpy(lat).to(torch.float64)
    if args.perf_repeat <= 0:
        raise ValueError("--perf-repeat must be positive")
    lon_perf = np.tile(lon, int(args.perf_repeat))
    lat_perf = np.tile(lat, int(args.perf_repeat))
    lon_perf_t = torch.from_numpy(lon_perf).to(torch.float64)
    lat_perf_t = torch.from_numpy(lat_perf).to(torch.float64)

    for nside in nside_values:
        npix = 12 * nside * nside
        rng = np.random.default_rng(9900 + nside)
        m = rng.normal(size=npix).astype(np.float64)
        m_t = torch.from_numpy(m).to(torch.float64)
        pix = _edge_pixels(nside)
        pix_t = torch.from_numpy(pix)
        pix_perf = np.tile(pix, int(args.perf_repeat))
        pix_perf_t = torch.from_numpy(pix_perf)

        for nest in (False, True):

            def tf_w_perf() -> tuple[np.ndarray, np.ndarray]:
                return tuple(
                    x.cpu().numpy()
                    for x in get_interp_weights(
                        nside, lon_perf_t, lat_perf_t, nest=nest, lonlat=True
                    )
                )

            def hp_w_perf() -> tuple[np.ndarray, np.ndarray]:
                return healpy.get_interp_weights(
                    nside, lon_perf, lat_perf, nest=nest, lonlat=True
                )

            def tf_w_base() -> tuple[np.ndarray, np.ndarray]:
                return tuple(
                    x.cpu().numpy()
                    for x in get_interp_weights(
                        nside, lon_t, lat_t, nest=nest, lonlat=True
                    )
                )

            def hp_w_base() -> tuple[np.ndarray, np.ndarray]:
                return healpy.get_interp_weights(
                    nside, lon, lat, nest=nest, lonlat=True
                )

            t_tf = _time_many(tf_w_perf, runs=args.runs)
            t_hp = _time_many(hp_w_perf, runs=args.runs)
            out_tf = tf_w_base()
            out_hp = hp_w_base()
            mm = _interp_mismatch_count(out_tf[0], out_tf[1], out_hp[0], out_hp[1])
            rows.append(
                Row(
                    operation="interp_weights_edge",
                    nside=nside,
                    nest=nest,
                    n_points=int(lon_perf.size),
                    mismatches=mm,
                    max_abs_err=float("nan"),
                    torchfits_mpts_s=(lon_perf.size / t_tf) / 1e6,
                    healpy_mpts_s=(lon_perf.size / t_hp) / 1e6,
                    ratio_vs_healpy=t_hp / t_tf,
                )
            )

            def tf_v_perf() -> np.ndarray:
                return (
                    get_interp_val(m_t, lon_perf_t, lat_perf_t, nest=nest, lonlat=True)
                    .cpu()
                    .numpy()
                )

            def hp_v_perf() -> np.ndarray:
                return healpy.get_interp_val(
                    m, lon_perf, lat_perf, nest=nest, lonlat=True
                )

            def tf_v_base() -> np.ndarray:
                return (
                    get_interp_val(m_t, lon_t, lat_t, nest=nest, lonlat=True)
                    .cpu()
                    .numpy()
                )

            def hp_v_base() -> np.ndarray:
                return healpy.get_interp_val(m, lon, lat, nest=nest, lonlat=True)

            t_tf = _time_many(tf_v_perf, runs=args.runs)
            t_hp = _time_many(hp_v_perf, runs=args.runs)
            out_tf_v = tf_v_base()
            out_hp_v = hp_v_base()
            max_abs = float(np.max(np.abs(out_tf_v - out_hp_v)))
            rows.append(
                Row(
                    operation="interp_val_edge",
                    nside=nside,
                    nest=nest,
                    n_points=int(lon_perf.size),
                    mismatches=0,
                    max_abs_err=max_abs,
                    torchfits_mpts_s=(lon_perf.size / t_tf) / 1e6,
                    healpy_mpts_s=(lon_perf.size / t_hp) / 1e6,
                    ratio_vs_healpy=t_hp / t_tf,
                )
            )

            def tf_wp_perf() -> tuple[np.ndarray, np.ndarray]:
                return tuple(
                    x.cpu().numpy()
                    for x in get_interp_weights(nside, pix_perf_t, nest=nest)
                )

            def hp_wp_perf() -> tuple[np.ndarray, np.ndarray]:
                return healpy.get_interp_weights(nside, pix_perf, nest=nest)

            def tf_wp_base() -> tuple[np.ndarray, np.ndarray]:
                return tuple(
                    x.cpu().numpy() for x in get_interp_weights(nside, pix_t, nest=nest)
                )

            def hp_wp_base() -> tuple[np.ndarray, np.ndarray]:
                return healpy.get_interp_weights(nside, pix, nest=nest)

            t_tf = _time_many(tf_wp_perf, runs=args.runs)
            t_hp = _time_many(hp_wp_perf, runs=args.runs)
            out_tf_p = tf_wp_base()
            out_hp_p = hp_wp_base()
            mm_p = _interp_mismatch_count(
                out_tf_p[0], out_tf_p[1], out_hp_p[0], out_hp_p[1]
            )
            rows.append(
                Row(
                    operation="interp_weights_pix_edge",
                    nside=nside,
                    nest=nest,
                    n_points=int(pix_perf.size),
                    mismatches=mm_p,
                    max_abs_err=float("nan"),
                    torchfits_mpts_s=(pix_perf.size / t_tf) / 1e6,
                    healpy_mpts_s=(pix_perf.size / t_hp) / 1e6,
                    ratio_vs_healpy=t_hp / t_tf,
                )
            )

    print(
        "operation                 nside  nest  npts  mismatches   max_abs_err   tf_mpts/s  hp_mpts/s  ratio"
    )
    for r in rows:
        print(
            f"{r.operation:24s} {r.nside:5d} {str(r.nest):5s} {r.n_points:5d}"
            f" {r.mismatches:11d} {r.max_abs_err:12.3e}"
            f" {r.torchfits_mpts_s:10.3f} {r.healpy_mpts_s:10.3f} {r.ratio_vs_healpy:7.3f}"
        )

    failures: list[str] = []
    for r in rows:
        if (
            r.operation in {"interp_weights_edge", "interp_weights_pix_edge"}
            and r.mismatches > args.max_weight_mismatches
        ):
            failures.append(
                f"{r.operation}@nside={r.nside},nest={r.nest}: mismatches {r.mismatches} > {args.max_weight_mismatches}"
            )
        if r.operation == "interp_val_edge" and (
            not np.isfinite(r.max_abs_err) or r.max_abs_err > args.max_val_max_abs
        ):
            failures.append(
                f"{r.operation}@nside={r.nside},nest={r.nest}: max_abs {r.max_abs_err:.3e} > {args.max_val_max_abs:.3e}"
            )

    allowed_ops = {r.operation for r in rows}
    try:
        ratio_thresholds = _parse_ratio_spec(args.min_ratio_vs_healpy, allowed_ops)
    except ValueError as exc:
        print(f"Invalid --min-ratio-vs-healpy: {exc}")
        return 2
    for op, thr in ratio_thresholds.items():
        vals = [r.ratio_vs_healpy for r in rows if r.operation == op]
        med = float(np.median(vals)) if vals else float("nan")
        print(f"median ratio {op:24s}: {med:.3f}x (threshold {thr:.3f}x)")
        if not np.isfinite(med) or med < thr:
            failures.append(f"{op}: median ratio {med:.3f} < {thr:.3f}")

    payload = {
        "rows": [asdict(r) for r in rows],
        "gates": {
            "max_weight_mismatches": args.max_weight_mismatches,
            "max_val_max_abs": args.max_val_max_abs,
            "min_ratio_vs_healpy": ratio_thresholds,
        },
        "failures": failures,
    }
    _write_json(args.json_out, payload)
    print(f"JSON written to: {args.json_out}")

    if failures:
        print("\nGate failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
