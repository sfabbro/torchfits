#!/usr/bin/env python3
"""Benchmark sparse HEALPix map primitives and optional perf gates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from torchfits.sphere.sparse import SparseHealpixMap
from torchfits.wcs import healpix as hp


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _sample_lonlat(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=n)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))).to(torch.float64)
    return lon, lat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--coverage-frac", type=float, default=0.1)
    parser.add_argument("--n-queries", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-out", type=Path, default=Path("bench_results/sphere_sparse.json"))
    parser.add_argument("--min-ratio-sparse-udgrade-vs-dense", type=float, default=1.0)
    args = parser.parse_args()

    if not (0.0 < args.coverage_frac <= 1.0):
        raise ValueError("coverage-frac must be in (0, 1]")

    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(args.seed)

    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    n_keep = max(1, int(round(npix * args.coverage_frac)))
    keep = torch.from_numpy(rng.choice(npix, size=n_keep, replace=False)).to(torch.int64)
    dense[keep] = torch.from_numpy(rng.normal(size=n_keep)).to(torch.float64)

    sparse = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    lon, lat = _sample_lonlat(int(args.n_queries), args.seed + 1)

    rows: list[dict[str, Any]] = []

    t_sparse_nearest = _time_many(lambda: sparse.interpolate(lon, lat, method="nearest"), args.runs)
    rows.append(
        {
            "operation": "sparse_interpolate_nearest",
            "nside": nside,
            "coverage_frac": sparse.coverage_fraction,
            "n_queries": int(args.n_queries),
            "qps": float(args.n_queries) / t_sparse_nearest,
        }
    )

    t_sparse_bilinear = _time_many(lambda: sparse.interpolate(lon, lat, method="bilinear"), args.runs)
    rows.append(
        {
            "operation": "sparse_interpolate_bilinear",
            "nside": nside,
            "coverage_frac": sparse.coverage_fraction,
            "n_queries": int(args.n_queries),
            "qps": float(args.n_queries) / t_sparse_bilinear,
        }
    )

    nside_out = max(1, nside // 2)
    t_sparse_ud = _time_many(lambda: sparse.ud_grade(nside_out, pess=False), args.runs)
    t_dense_ud = _time_many(
        lambda: hp.ud_grade(
            dense,
            nside_out,
            pess=False,
            badval=float(hp.UNSEEN),
            order_in="RING",
            order_out="RING",
        ),
        args.runs,
    )
    ratio_ud = t_dense_ud / t_sparse_ud
    rows.append(
        {
            "operation": "ud_grade_sparse_vs_dense",
            "nside_in": nside,
            "nside_out": nside_out,
            "coverage_frac": sparse.coverage_fraction,
            "sparse_s": t_sparse_ud,
            "dense_s": t_dense_ud,
            "ratio_sparse_vs_dense": ratio_ud,
        }
    )

    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items()))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")

    if ratio_ud < float(args.min_ratio_sparse_udgrade_vs_dense):
        print(
            "Sparse ud_grade ratio threshold exceeded: "
            f"{ratio_ud:.3f} < {float(args.min_ratio_sparse_udgrade_vs_dense):.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

