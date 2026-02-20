#!/usr/bin/env python3
"""Benchmark spherical polygon primitives and optional comparator parity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from torchfits.sphere.geom import (
    SphericalPolygon,
    query_polygon_general,
    spherical_polygon_contains,
)

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    SGP = None


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _star_polygon(
    center_lon: float, center_lat: float, n: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.where((np.arange(n) % 2) == 0, 12.0, 5.0)
    lon = center_lon + radii * np.cos(angles) / max(
        np.cos(np.deg2rad(center_lat)), 0.25
    )
    lat = center_lat + radii * np.sin(angles)
    lon = (lon + 360.0) % 360.0
    lat = np.clip(lat, -85.0, 85.0)
    return lon.astype(np.float64), lat.astype(np.float64)


def _random_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon, lat


def _sg_polygon_from_lonlat(lon: np.ndarray, lat: np.ndarray):
    if SGP is None:
        return None
    if hasattr(SGP, "from_lonlat"):
        return SGP.from_lonlat(lon, lat, degrees=True)
    if hasattr(SGP, "from_radec"):
        return SGP.from_radec(lon, lat, degrees=True)
    return None


def _sg_contains(poly: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if poly is None:
        raise RuntimeError("spherical_geometry comparator unavailable")
    if hasattr(poly, "contains_lonlat"):
        try:
            out = np.asarray(poly.contains_lonlat(lon, lat, degrees=True), dtype=bool)
            if out.shape == lon.shape:
                return out
        except Exception:
            pass
        return np.asarray(
            [
                bool(poly.contains_lonlat(float(lo), float(la), degrees=True))
                for lo, la in zip(lon, lat, strict=False)
            ],
            dtype=bool,
        )
    if hasattr(poly, "contains_radec"):
        try:
            out = np.asarray(poly.contains_radec(lon, lat, degrees=True), dtype=bool)
            if out.shape == lon.shape:
                return out
        except Exception:
            pass
        return np.asarray(
            [
                bool(poly.contains_radec(float(lo), float(la), degrees=True))
                for lo, la in zip(lon, lat, strict=False)
            ],
            dtype=bool,
        )
    raise RuntimeError("unsupported spherical_geometry API for contains")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-out", type=Path, default=Path("bench_results/sphere_polygons.json")
    )
    args = parser.parse_args()

    lon_poly, lat_poly = _star_polygon(center_lon=120.0, center_lat=-20.0, n=9)
    lon_q, lat_q = _random_lonlat(args.n_points, args.seed)

    lon_q_t = torch.from_numpy(lon_q).to(torch.float64)
    lat_q_t = torch.from_numpy(lat_q).to(torch.float64)
    lon_poly_t = torch.from_numpy(lon_poly).to(torch.float64)
    lat_poly_t = torch.from_numpy(lat_poly).to(torch.float64)

    poly = SphericalPolygon(lon_poly_t, lat_poly_t)

    rows: list[dict[str, Any]] = []

    t_contains = _time_many(
        lambda: spherical_polygon_contains(
            lon_q_t,
            lat_q_t,
            lon_poly_t,
            lat_poly_t,
            inclusive=False,
        ),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "contains_nonconvex",
            "library": "torchfits",
            "n_points": args.n_points,
            "points_s": args.n_points / t_contains,
        }
    )

    t_pixel_query = _time_many(
        lambda: query_polygon_general(args.nside, lon_poly_t, lat_poly_t, nest=False),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "query_polygon_nonconvex",
            "library": "torchfits",
            "nside": args.nside,
            "queries_s": 1.0 / t_pixel_query,
        }
    )

    t_area = _time_many(lambda: poly.area(degrees=False), runs=args.runs)
    rows.append(
        {
            "operation": "area_nonconvex",
            "library": "torchfits",
            "calls_s": 1.0 / t_area,
            "area_sr": float(poly.area(degrees=False)),
        }
    )

    sg_poly = _sg_polygon_from_lonlat(lon_poly, lat_poly)
    if sg_poly is not None:
        t_sg_contains = _time_many(
            lambda: _sg_contains(sg_poly, lon_q, lat_q), runs=max(2, args.runs // 2)
        )
        tf_contains = (
            spherical_polygon_contains(
                lon_q_t,
                lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=False,
            )
            .cpu()
            .numpy()
        )
        tf_contains_inclusive = (
            spherical_polygon_contains(
                lon_q_t,
                lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=True,
                atol_deg=1e-5,
            )
            .cpu()
            .numpy()
        )
        sg_contains = _sg_contains(sg_poly, lon_q, lat_q)
        mismatches = int(np.sum(tf_contains != sg_contains))
        boundary_like = tf_contains_inclusive != tf_contains
        mismatches_nonboundary = int(
            np.sum((tf_contains != sg_contains) & (~boundary_like))
        )

        sg_area = float(sg_poly.area()) if hasattr(sg_poly, "area") else float("nan")
        rel_area_err = abs(float(poly.area(degrees=False)) - sg_area) / max(
            abs(sg_area), 1e-15
        )

        rows.append(
            {
                "operation": "contains_nonconvex",
                "library": "spherical-geometry",
                "n_points": args.n_points,
                "points_s": args.n_points / t_sg_contains,
                "ratio_torchfits_vs_spherical_geometry": t_sg_contains / t_contains,
                "mismatches_vs_spherical_geometry": mismatches,
                "mismatches_vs_spherical_geometry_nonboundary": mismatches_nonboundary,
            }
        )
        rows.append(
            {
                "operation": "area_nonconvex",
                "library": "spherical-geometry",
                "area_sr": sg_area,
                "relative_error_vs_torchfits": rel_area_err,
            }
        )

    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items()))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
