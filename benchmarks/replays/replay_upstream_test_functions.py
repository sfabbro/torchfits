#!/usr/bin/env python3
"""Replay selected upstream test functions against TorchFits adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import healpy as hp
import numpy as np
import torch

from torchfits.wcs.healpix import (
    ang2vec as tf_ang2vec,
    boundaries as tf_boundaries,
    get_interp_val as tf_get_interp_val,
    get_interp_weights as tf_get_interp_weights,
    nside2npix as tf_nside2npix,
    nside2pixarea as tf_nside2pixarea,
    nside2resol as tf_nside2resol,
    npix2nside as tf_npix2nside,
    order2nside as tf_order2nside,
    ang2pix_nested as tf_ang2pix_nested,
    ang2pix_ring as tf_ang2pix_ring,
    nest2ring as tf_nest2ring,
    pix2vec as tf_pix2vec,
    pix2ang_nested as tf_pix2ang_nested,
    pix2ang_ring as tf_pix2ang_ring,
    ring2nest as tf_ring2nest,
    vec2ang as tf_vec2ang,
    vec2pix as tf_vec2pix,
)


LON_LAT_CASES = [
    (1.0000000028043134e-05, -41.81031451395941),
    (1.0000000028043134e-05, 1.000000000805912e-05),
    (359.9999986588955, 41.81031489577861),
    (359.9999922886491, -41.81031470486902),
    (1.6345238095238293, 69.42254649458224),
]
FRACS = [0.0, 0.125, 0.1666666694606345, 2.0 / 3.0, 0.999999999]
NSIDE_POW_CASES = [0, 2, 5, 8]
INTERP_VAL_NSIDE_POW_CASES = [0, 1, 2, 3]
VEC_CASES = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
]


@dataclass
class ReplayResult:
    name: str
    passed: bool
    error: str | None = None


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def _maybe_scalar(x: np.ndarray | tuple[np.ndarray, ...]) -> Any:
    if isinstance(x, tuple):
        if all(np.ndim(v) == 0 for v in x):
            return tuple(float(v) for v in x)
        return x
    if np.ndim(x) == 0:
        return x.item()
    return x


class TorchFitsHealpyCompat:
    """Subset of astropy_healpix.healpy compatibility API backed by TorchFits."""

    @staticmethod
    def nside2pixarea(nside: int, degrees: bool = False) -> float:
        return float(tf_nside2pixarea(nside, degrees=degrees))

    @staticmethod
    def nside2resol(nside: int, arcmin: bool = False) -> float:
        return float(tf_nside2resol(nside, arcmin=arcmin))

    @staticmethod
    def nside2npix(nside: int) -> int:
        return int(tf_nside2npix(nside))

    @staticmethod
    def order2nside(order: int) -> int:
        return int(tf_order2nside(order))

    @staticmethod
    def npix2nside(npix: int) -> int:
        return int(tf_npix2nside(npix))

    @staticmethod
    def ang2pix(
        nside: int, theta: Any, phi: Any, nest: bool = False, lonlat: bool = False
    ) -> Any:
        theta_np = _to_numpy(theta)
        phi_np = _to_numpy(phi)
        if lonlat:
            lon = theta_np.astype(np.float64, copy=False)
            lat = phi_np.astype(np.float64, copy=False)
        else:
            lon = np.degrees(phi_np.astype(np.float64, copy=False))
            lat = 90.0 - np.degrees(theta_np.astype(np.float64, copy=False))

        lon_t = torch.from_numpy(np.ascontiguousarray(lon))
        lat_t = torch.from_numpy(np.ascontiguousarray(lat))
        if nest:
            out = tf_ang2pix_nested(nside, lon_t, lat_t).cpu().numpy()
        else:
            out = tf_ang2pix_ring(nside, lon_t, lat_t).cpu().numpy()
        return _maybe_scalar(out)

    @staticmethod
    def pix2ang(nside: int, ipix: Any, nest: bool = False, lonlat: bool = False) -> Any:
        pix_np = _to_numpy(ipix).astype(np.int64, copy=False)
        scalar_input = np.ndim(pix_np) == 0
        pix_t = torch.from_numpy(np.ascontiguousarray(pix_np))
        if nest:
            lon_t, lat_t = tf_pix2ang_nested(nside, pix_t)
        else:
            lon_t, lat_t = tf_pix2ang_ring(nside, pix_t)

        lon = lon_t.cpu().numpy()
        lat = lat_t.cpu().numpy()
        if scalar_input and lon.size == 1 and lat.size == 1:
            lon_s = float(lon.reshape(-1)[0])
            lat_s = float(lat.reshape(-1)[0])
            if lonlat:
                return lon_s, lat_s
            return float(np.radians(90.0 - lat_s)), float(np.radians(lon_s))
        if lonlat:
            return _maybe_scalar((lon, lat))

        theta = np.radians(90.0 - lat)
        phi = np.radians(lon)
        return _maybe_scalar((theta, phi))

    @staticmethod
    def nest2ring(nside: int, ipix: Any) -> Any:
        pix_np = _to_numpy(ipix).astype(np.int64, copy=False)
        out = (
            tf_nest2ring(nside, torch.from_numpy(np.ascontiguousarray(pix_np)))
            .cpu()
            .numpy()
        )
        return _maybe_scalar(out)

    @staticmethod
    def ring2nest(nside: int, ipix: Any) -> Any:
        pix_np = _to_numpy(ipix).astype(np.int64, copy=False)
        out = (
            tf_ring2nest(nside, torch.from_numpy(np.ascontiguousarray(pix_np)))
            .cpu()
            .numpy()
        )
        return _maybe_scalar(out)

    @staticmethod
    def vec2pix(nside: int, x: Any, y: Any, z: Any, nest: bool = False) -> Any:
        x_np = _to_numpy(x).astype(np.float64, copy=False)
        y_np = _to_numpy(y).astype(np.float64, copy=False)
        z_np = _to_numpy(z).astype(np.float64, copy=False)
        out = (
            tf_vec2pix(
                nside,
                torch.from_numpy(np.ascontiguousarray(x_np)),
                torch.from_numpy(np.ascontiguousarray(y_np)),
                torch.from_numpy(np.ascontiguousarray(z_np)),
                nest=nest,
            )
            .cpu()
            .numpy()
        )
        return _maybe_scalar(out)

    @staticmethod
    def pix2vec(nside: int, ipix: Any, nest: bool = False) -> Any:
        pix_np = _to_numpy(ipix).astype(np.int64, copy=False)
        scalar_input = np.ndim(pix_np) == 0
        x_t, y_t, z_t = tf_pix2vec(
            nside, torch.from_numpy(np.ascontiguousarray(pix_np)), nest=nest
        )
        x = x_t.cpu().numpy()
        y = y_t.cpu().numpy()
        z = z_t.cpu().numpy()
        if scalar_input and x.size == 1 and y.size == 1 and z.size == 1:
            return (
                float(x.reshape(-1)[0]),
                float(y.reshape(-1)[0]),
                float(z.reshape(-1)[0]),
            )
        return x, y, z

    @staticmethod
    def boundaries(nside: int, pix: Any, step: int = 1, nest: bool = False) -> Any:
        pix_np = _to_numpy(pix).astype(np.int64, copy=False)
        scalar_input = np.ndim(pix_np) == 0
        if scalar_input:
            pix_t = torch.tensor(int(pix_np), dtype=torch.int64)
        else:
            pix_t = torch.from_numpy(np.ascontiguousarray(pix_np))
        out = tf_boundaries(nside, pix_t, step=step, nest=nest).cpu().numpy()
        if scalar_input and out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        return out

    @staticmethod
    def vec2ang(vectors: Any, lonlat: bool = False) -> Any:
        v_np = np.array(_to_numpy(vectors), dtype=np.float64, copy=True)
        lon_t, lat_t = tf_vec2ang(torch.as_tensor(v_np), lonlat=True)
        lon = lon_t.cpu().numpy()
        lat = lat_t.cpu().numpy()
        if lon.ndim == 0:
            lon = lon.reshape(1)
            lat = lat.reshape(1)
        if lonlat:
            return lon, lat
        theta = np.radians(90.0 - lat)
        phi = np.radians(lon)
        return theta, phi

    @staticmethod
    def ang2vec(theta: Any, phi: Any, lonlat: bool = False) -> Any:
        theta_np = np.array(_to_numpy(theta), dtype=np.float64, copy=True)
        phi_np = np.array(_to_numpy(phi), dtype=np.float64, copy=True)
        if lonlat:
            lon = theta_np
            lat = phi_np
            vec = tf_ang2vec(
                torch.as_tensor(lon),
                torch.as_tensor(lat),
                lonlat=True,
            )
        else:
            vec = tf_ang2vec(
                torch.as_tensor(theta_np),
                torch.as_tensor(phi_np),
                lonlat=False,
            )
        out = vec.cpu().numpy()
        if (
            theta_np.ndim == 0
            and phi_np.ndim == 0
            and out.ndim == 2
            and out.shape[0] == 1
        ):
            out = out[0]
        return out

    @staticmethod
    def get_interp_weights(
        nside: int,
        theta: Any,
        phi: Any = None,
        nest: bool = False,
        lonlat: bool = False,
    ) -> Any:
        if phi is None:
            theta_np = _to_numpy(theta).astype(np.int64, copy=False)
            pix_t = torch.from_numpy(np.ascontiguousarray(theta_np))
            p_t, w_t = tf_get_interp_weights(nside, pix_t, nest=nest, lonlat=lonlat)
        else:
            theta_np = _to_numpy(theta).astype(np.float64, copy=False)
            phi_np = _to_numpy(phi).astype(np.float64, copy=False)
            p_t, w_t = tf_get_interp_weights(
                nside,
                torch.from_numpy(np.ascontiguousarray(theta_np)),
                torch.from_numpy(np.ascontiguousarray(phi_np)),
                nest=nest,
                lonlat=lonlat,
            )
        return p_t.cpu().numpy(), w_t.cpu().numpy()

    @staticmethod
    def get_interp_val(
        m: Any, theta: Any, phi: Any, nest: bool = False, lonlat: bool = False
    ) -> Any:
        m_np = _to_numpy(m)
        theta_np = _to_numpy(theta).astype(np.float64, copy=False)
        phi_np = _to_numpy(phi).astype(np.float64, copy=False)
        out = (
            tf_get_interp_val(
                torch.from_numpy(np.ascontiguousarray(m_np)),
                torch.from_numpy(np.ascontiguousarray(theta_np)),
                torch.from_numpy(np.ascontiguousarray(phi_np)),
                nest=nest,
                lonlat=lonlat,
            )
            .cpu()
            .numpy()
        )
        return _maybe_scalar(out)


def _run_named(name: str, fn: Callable[[], None]) -> ReplayResult:
    try:
        fn()
        return ReplayResult(name=name, passed=True, error=None)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return ReplayResult(
            name=name, passed=False, error=f"{type(exc).__name__}: {exc}"
        )


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _microbench(nside: int, n_points: int, runs: int) -> dict[str, float]:
    rng = np.random.default_rng(123)
    lon = rng.uniform(0.0, 360.0, size=n_points).astype(np.float64)
    lat = rng.uniform(-89.8, 89.8, size=n_points).astype(np.float64)
    pix = rng.integers(0, 12 * nside * nside, size=n_points, dtype=np.int64)

    adapter = TorchFitsHealpyCompat()
    out: dict[str, float] = {}

    for op, tf_fn, hp_fn in [
        (
            "ang2pix_ring",
            lambda: adapter.ang2pix(nside, lon, lat, nest=False, lonlat=True),
            lambda: hp.ang2pix(nside, lon, lat, nest=False, lonlat=True),
        ),
        (
            "ang2pix_nested",
            lambda: adapter.ang2pix(nside, lon, lat, nest=True, lonlat=True),
            lambda: hp.ang2pix(nside, lon, lat, nest=True, lonlat=True),
        ),
        (
            "pix2ang_ring",
            lambda: adapter.pix2ang(nside, pix, nest=False, lonlat=True),
            lambda: hp.pix2ang(nside, pix, nest=False, lonlat=True),
        ),
        (
            "pix2ang_nested",
            lambda: adapter.pix2ang(nside, pix, nest=True, lonlat=True),
            lambda: hp.pix2ang(nside, pix, nest=True, lonlat=True),
        ),
        (
            "ring2nest",
            lambda: adapter.ring2nest(nside, pix),
            lambda: hp.ring2nest(nside, pix),
        ),
        (
            "nest2ring",
            lambda: adapter.nest2ring(nside, pix),
            lambda: hp.nest2ring(nside, pix),
        ),
    ]:
        t_tf = _time_many(tf_fn, runs=runs)
        t_hp = _time_many(hp_fn, runs=runs)
        out[f"{op}_ratio_vs_healpy"] = t_hp / t_tf
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("bench_results/upstream_fixtures/sources/astropy-healpix-1.1.3"),
    )
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--bench-nside", type=int, default=1024)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_test_functions.json"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    mod = importlib.import_module("astropy_healpix.tests.test_healpy")
    mod.hp_compat = TorchFitsHealpyCompat()

    results: list[ReplayResult] = []

    # Parametrized deterministic tests
    for nside in mod.NSIDE_VALUES:
        for degrees in (False, True):
            results.append(
                _run_named(
                    f"test_nside2pixarea(nside={nside},degrees={degrees})",
                    lambda nside=nside, degrees=degrees: mod.test_nside2pixarea(
                        nside, degrees
                    ),
                )
            )
    for nside in mod.NSIDE_VALUES:
        for arcmin in (False, True):
            results.append(
                _run_named(
                    f"test_nside2resol(nside={nside},arcmin={arcmin})",
                    lambda nside=nside, arcmin=arcmin: mod.test_nside2resol(
                        nside, arcmin
                    ),
                )
            )
    for nside in mod.NSIDE_VALUES:
        results.append(
            _run_named(
                f"test_nside2npix(nside={nside})",
                lambda nside=nside: mod.test_nside2npix(nside),
            )
        )
    for level in [0, 3, 7]:
        results.append(
            _run_named(
                f"test_order2nside(level={level})",
                lambda level=level: mod.test_order2nside(level),
            )
        )
    for npix in [12 * 2 ** (2 * n) for n in range(1, 6)]:
        results.append(
            _run_named(
                f"test_npix2nside(npix={npix})",
                lambda npix=npix: mod.test_npix2nside(npix),
            )
        )

    # Shape tests (non-hypothesis)
    results.append(_run_named("test_ang2pix_shape()", mod.test_ang2pix_shape))
    results.append(_run_named("test_pix2ang_shape()", mod.test_pix2ang_shape))
    results.append(_run_named("test_vec2pix_shape()", mod.test_vec2pix_shape))
    results.append(_run_named("test_pix2vec_shape()", mod.test_pix2vec_shape))
    results.append(_run_named("test_boundaries_shape()", mod.test_boundaries_shape))

    # Replay hypothesis-decorated bodies with upstream-style explicit inputs.
    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            for lonlat in (False, True):
                for lon, lat in LON_LAT_CASES:
                    name = f"test_ang2pix.inner(nside_pow={nside_pow},nest={nest},lonlat={lonlat},lon={lon},lat={lat})"
                    results.append(
                        _run_named(
                            name,
                            lambda nside_pow=nside_pow, lon=lon, lat=lat, nest=nest, lonlat=lonlat: (
                                mod.test_ang2pix.hypothesis.inner_test(
                                    nside_pow=nside_pow,
                                    lon=lon,
                                    lat=lat,
                                    nest=nest,
                                    lonlat=lonlat,
                                )
                            ),
                        )
                    )

    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            for lonlat in (False, True):
                for frac in FRACS:
                    name = f"test_pix2ang.inner(nside_pow={nside_pow},nest={nest},lonlat={lonlat},frac={frac})"
                    results.append(
                        _run_named(
                            name,
                            lambda nside_pow=nside_pow, frac=frac, nest=nest, lonlat=lonlat: (
                                mod.test_pix2ang.hypothesis.inner_test(
                                    nside_pow=nside_pow,
                                    frac=frac,
                                    nest=nest,
                                    lonlat=lonlat,
                                )
                            ),
                        )
                    )

    for nside_pow in NSIDE_POW_CASES:
        for frac in FRACS:
            results.append(
                _run_named(
                    f"test_ring2nest.inner(nside_pow={nside_pow},frac={frac})",
                    lambda nside_pow=nside_pow, frac=frac: (
                        mod.test_ring2nest.hypothesis.inner_test(
                            nside_pow=nside_pow, frac=frac
                        )
                    ),
                )
            )
            results.append(
                _run_named(
                    f"test_nest2ring.inner(nside_pow={nside_pow},frac={frac})",
                    lambda nside_pow=nside_pow, frac=frac: (
                        mod.test_nest2ring.hypothesis.inner_test(
                            nside_pow=nside_pow, frac=frac
                        )
                    ),
                )
            )

    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            nside = 2**nside_pow
            for frac in FRACS:
                ipix = int(frac * 12 * nside * nside)
                x, y, z = hp.pix2vec(nside, ipix, nest=nest)
                vec_args = (nside_pow, nest, float(x), float(y), float(z))
                results.append(
                    _run_named(
                        f"test_vec2pix.inner(nside_pow={nside_pow},nest={nest},frac={frac})",
                        lambda vec_args=vec_args: (
                            mod.test_vec2pix.hypothesis.inner_test(args=vec_args)
                        ),
                    )
                )

    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            for frac in FRACS:
                results.append(
                    _run_named(
                        f"test_pix2vec.inner(nside_pow={nside_pow},nest={nest},frac={frac})",
                        lambda nside_pow=nside_pow, frac=frac, nest=nest: (
                            mod.test_pix2vec.hypothesis.inner_test(
                                nside_pow=nside_pow, frac=frac, nest=nest
                            )
                        ),
                    )
                )

    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            for step in (1, 4, 8):
                for frac in FRACS:
                    results.append(
                        _run_named(
                            f"test_boundaries.inner(nside_pow={nside_pow},nest={nest},step={step},frac={frac})",
                            lambda nside_pow=nside_pow, frac=frac, step=step, nest=nest: (
                                mod.test_boundaries.hypothesis.inner_test(
                                    nside_pow=nside_pow, frac=frac, step=step, nest=nest
                                )
                            ),
                        )
                    )

    for lonlat in (False, True):
        for ndim in (0, 2):
            for vx, vy, vz in VEC_CASES:
                vec = np.array([vx, vy, vz], dtype=np.float64)
                name = f"test_vec2ang.inner(lonlat={lonlat},ndim={ndim},vec={vec.tolist()})"
                results.append(
                    _run_named(
                        name,
                        lambda lonlat=lonlat, ndim=ndim, vec=vec: (
                            mod.test_vec2ang.hypothesis.inner_test(
                                vectors=vec, lonlat=lonlat, ndim=ndim
                            )
                        ),
                    )
                )

    for lon, lat in LON_LAT_CASES:
        for lonlat in (False, True):
            results.append(
                _run_named(
                    f"test_ang2vec.inner(lon={lon},lat={lat},lonlat={lonlat})",
                    lambda lon=lon, lat=lat, lonlat=lonlat: (
                        mod.test_ang2vec.hypothesis.inner_test(
                            lon=lon, lat=lat, lonlat=lonlat
                        )
                    ),
                )
            )

    for nside_pow in NSIDE_POW_CASES:
        for nest in (False, True):
            for lonlat in (False, True):
                for lon, lat in LON_LAT_CASES:
                    name = f"test_interp_weights.inner(nside_pow={nside_pow},nest={nest},lonlat={lonlat},lon={lon},lat={lat})"
                    results.append(
                        _run_named(
                            name,
                            lambda nside_pow=nside_pow, lon=lon, lat=lat, nest=nest, lonlat=lonlat: (
                                mod.test_interp_weights.hypothesis.inner_test(
                                    nside_pow=nside_pow,
                                    lon=lon,
                                    lat=lat,
                                    nest=nest,
                                    lonlat=lonlat,
                                )
                            ),
                        )
                    )

    for nside_pow in INTERP_VAL_NSIDE_POW_CASES:
        for nest in (False, True):
            for lonlat in (False, True):
                for lon, lat in LON_LAT_CASES:
                    name = f"test_interp_val.inner(nside_pow={nside_pow},nest={nest},lonlat={lonlat},lon={lon},lat={lat})"
                    results.append(
                        _run_named(
                            name,
                            lambda nside_pow=nside_pow, lon=lon, lat=lat, nest=nest, lonlat=lonlat: (
                                mod.test_interp_val.hypothesis.inner_test(
                                    nside_pow=nside_pow,
                                    lon=lon,
                                    lat=lat,
                                    nest=nest,
                                    lonlat=lonlat,
                                )
                            ),
                        )
                    )

    total = len(results)
    failed = [r for r in results if not r.passed]

    bench = _microbench(nside=args.bench_nside, n_points=args.n_points, runs=args.runs)

    print(f"Replayed {total} upstream test-function cases")
    print(f"  passed: {total - len(failed)}")
    print(f"  failed: {len(failed)}")
    if failed:
        for r in failed[:20]:
            print(f"    - {r.name}: {r.error}")

    print("Median throughput ratio vs healpy (TorchFits faster if >1):")
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}x")

    payload = {
        "source_root": str(args.source_root),
        "total_cases": total,
        "failed_cases": len(failed),
        "failures": [r.__dict__ for r in failed],
        "results": [r.__dict__ for r in results],
        "bench_ratios_vs_healpy": bench,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON: {args.json_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
