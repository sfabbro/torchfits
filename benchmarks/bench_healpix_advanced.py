#!/usr/bin/env python3
"""Benchmark advanced HEALPix primitives (neighbors and interpolation) vs healpy."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import healpy
import numpy as np
import torch

from torchfits.wcs.healpix import get_all_neighbours, get_interp_val, get_interp_weights


def _parse_ratio_spec(spec: str | None, allowed_ops: set[str]) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid ratio spec segment '{part}', expected op=value")
        op, raw = part.split("=", 1)
        op = op.strip()
        if op not in allowed_ops:
            raise ValueError(f"Unknown operation '{op}' in ratio spec")
        val = float(raw.strip())
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"Ratio threshold must be positive for operation '{op}'")
        out[op] = val
    return out


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if choice == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but no MPS device is available")
    return torch.device(choice)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_many(fn: Callable[[], Any], runs: int, sync_device: torch.device | None = None) -> float:
    fn()
    if sync_device is not None:
        _sync(sync_device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_device is not None:
            _sync(sync_device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _sample_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return lon, lat


def _sample_pix(nside: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    return rng.integers(0, npix, size=n, dtype=np.int64)


def _interp_mismatch_count(
    pix_tf: np.ndarray,
    w_tf: np.ndarray,
    pix_hp: np.ndarray,
    w_hp: np.ndarray,
    weight_floor: float = 1.0e-6,
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


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--min-ratio-vs-healpy",
        type=str,
        default="",
        help="Comma-separated op=ratio thresholds for torchfits/healpy.",
    )
    parser.add_argument("--json-out", type=Path, default=Path("bench_results/healpix_advanced.json"))
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    nside = args.nside
    n = args.n_points
    npix = 12 * nside * nside
    device = _resolve_device(args.device)

    lon, lat = _sample_lonlat(n, args.seed)
    pix_ring = _sample_pix(nside, n, args.seed + 1)
    pix_nest = _sample_pix(nside, n, args.seed + 2)

    lon_t = torch.from_numpy(lon).to(device=device, dtype=torch.float32 if device.type == "mps" else torch.float64)
    lat_t = torch.from_numpy(lat).to(device=device, dtype=torch.float32 if device.type == "mps" else torch.float64)
    pix_ring_t = torch.from_numpy(pix_ring).to(device=device, dtype=torch.int64)
    pix_nest_t = torch.from_numpy(pix_nest).to(device=device, dtype=torch.int64)

    rng = np.random.default_rng(args.seed + 3)
    m_ring = rng.normal(size=npix).astype(np.float64)
    m_nest = healpy.reorder(m_ring, r2n=True)
    m_ring_t = torch.from_numpy(m_ring).to(device=device, dtype=torch.float32 if device.type == "mps" else torch.float64)
    m_nest_t = torch.from_numpy(m_nest).to(device=device, dtype=torch.float32 if device.type == "mps" else torch.float64)

    rows: list[dict[str, Any]] = []

    ops: list[tuple[str, Callable[[], Any], Callable[[], Any], Callable[[Any, Any], tuple[int, float]]]] = [
        (
            "neighbors_ring",
            lambda: get_all_neighbours(nside, pix_ring_t, nest=False).cpu().numpy(),
            lambda: healpy.get_all_neighbours(nside, pix_ring, nest=False),
            lambda a, b: (int(np.sum(np.asarray(a) != np.asarray(b))), float("nan")),
        ),
        (
            "neighbors_nested",
            lambda: get_all_neighbours(nside, pix_nest_t, nest=True).cpu().numpy(),
            lambda: healpy.get_all_neighbours(nside, pix_nest, nest=True),
            lambda a, b: (int(np.sum(np.asarray(a) != np.asarray(b))), float("nan")),
        ),
        (
            "interp_weights_ring",
            lambda: tuple(x.cpu().numpy() for x in get_interp_weights(nside, lon_t, lat_t, nest=False, lonlat=True)),
            lambda: healpy.get_interp_weights(nside, lon, lat, nest=False, lonlat=True),
            lambda a, b: (_interp_mismatch_count(a[0], a[1], b[0], b[1]), float("nan")),
        ),
        (
            "interp_weights_nested",
            lambda: tuple(x.cpu().numpy() for x in get_interp_weights(nside, lon_t, lat_t, nest=True, lonlat=True)),
            lambda: healpy.get_interp_weights(nside, lon, lat, nest=True, lonlat=True),
            lambda a, b: (_interp_mismatch_count(a[0], a[1], b[0], b[1]), float("nan")),
        ),
        (
            "interp_val_ring",
            lambda: get_interp_val(m_ring_t, lon_t, lat_t, nest=False, lonlat=True).cpu().numpy(),
            lambda: healpy.get_interp_val(m_ring, lon, lat, nest=False, lonlat=True),
            lambda a, b: (int(np.sum(~np.isclose(np.asarray(a), np.asarray(b), atol=1e-10, rtol=1e-10))), float(np.max(np.abs(np.asarray(a) - np.asarray(b))))),
        ),
        (
            "interp_val_nested",
            lambda: get_interp_val(m_nest_t, lon_t, lat_t, nest=True, lonlat=True).cpu().numpy(),
            lambda: healpy.get_interp_val(m_nest, lon, lat, nest=True, lonlat=True),
            lambda a, b: (int(np.sum(~np.isclose(np.asarray(a), np.asarray(b), atol=1e-10, rtol=1e-10))), float(np.max(np.abs(np.asarray(a) - np.asarray(b))))),
        ),
    ]
    allowed_ops = {name for name, _, _, _ in ops}
    try:
        min_ratios = _parse_ratio_spec(args.min_ratio_vs_healpy, allowed_ops)
    except ValueError as exc:
        print(f"Invalid --min-ratio-vs-healpy: {exc}")
        return 2

    for name, tf_fn, hp_fn, metric_fn in ops:
        t_tf = _time_many(tf_fn, runs=args.runs, sync_device=device)
        t_hp = _time_many(hp_fn, runs=args.runs, sync_device=None)
        out_tf = tf_fn()
        out_hp = hp_fn()
        mismatches, max_abs_err = metric_fn(out_tf, out_hp)

        rows.append(
            {
                "operation": name,
                "nside": nside,
                "n_points": n,
                "device": device.type,
                "mpts_s_torchfits": (n / t_tf) / 1e6,
                "mpts_s_healpy": (n / t_hp) / 1e6,
                "ratio_torchfits_vs_healpy": t_hp / t_tf,
                "mismatches": mismatches,
                "max_abs_err": max_abs_err,
            }
        )

    print(
        f"{'operation':>22s} {'mpts/s(tf)':>12s} {'mpts/s(hp)':>12s} {'ratio':>8s} {'mismatch':>10s} {'max_abs_err':>12s}"
    )
    for r in rows:
        print(
            f"{r['operation']:>22s} {r['mpts_s_torchfits']:12.2f} {r['mpts_s_healpy']:12.2f} {r['ratio_torchfits_vs_healpy']:8.3f} {int(r['mismatches']):10d} {r['max_abs_err']:12.3e}"
        )

    _write_json(args.json_out, rows)
    print(f"\nJSON: {args.json_out}")
    if args.csv_out is not None:
        _write_csv(args.csv_out, rows)
        print(f"CSV: {args.csv_out}")

    if min_ratios:
        by_op = {str(r["operation"]): float(r["ratio_torchfits_vs_healpy"]) for r in rows}
        failed: list[tuple[str, float, float]] = []
        for op, threshold in min_ratios.items():
            got = by_op.get(op, float("nan"))
            if not np.isfinite(got) or got < threshold:
                failed.append((op, threshold, got))
        if failed:
            print("\nTorchFits/healpy ratio threshold exceeded:")
            for op, threshold, got in failed:
                print(f"  {op}: {got:.3f} < {threshold:.3f}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
