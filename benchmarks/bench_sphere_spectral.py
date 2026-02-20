#!/usr/bin/env python3
"""Benchmark sphere spectral primitives (scalar + spin transforms)."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from torchfits.sphere.spectral import (
    alm2map,
    alm2cl,
    alm2map_spin,
    alm_index,
    alm_size,
    almxfl,
    anafast,
    map2alm,
    map2alm_lsq,
    map2alm_spin,
)
from torchfits.wcs import healpix as hp


@dataclass
class Row:
    op: str
    backend: str
    nside: int
    lmax: int
    n: int
    time_s: float
    mops_s: float


def _time_many(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _random_spin_alms(lmax: int, spin: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    a_e = np.zeros(alm_size(lmax), dtype=np.complex128)
    a_b = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            if ell < spin:
                continue
            idx = alm_index(ell, m, lmax)
            scale = 1.0 / ((ell + 1) ** 2)
            if m == 0:
                a_e[idx] = scale * rng.normal()
                a_b[idx] = scale * rng.normal()
            else:
                a_e[idx] = scale * (rng.normal() + 1j * rng.normal())
                a_b[idx] = scale * (rng.normal() + 1j * rng.normal())
    return torch.from_numpy(a_e).to(torch.complex128), torch.from_numpy(a_b).to(torch.complex128)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nside", type=int, default=32)
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--spin", type=int, default=2)
    p.add_argument("--lsq-maxiter", type=int, default=3)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-torch-cpp-variant", action="store_true")
    p.add_argument("--json-out", type=Path, default=Path("bench_results/sphere_spectral.json"))
    args = p.parse_args()

    # Keep spin spectral benches on the known-fast torch path unless explicitly overridden.
    os.environ.setdefault("TORCHFITS_RING_FOURIER_CPP", "0")
    os.environ.setdefault("TORCHFITS_SPIN_MAP2ALM_RECURRENCE_CPP", "1")
    os.environ.setdefault("TORCHFITS_SPIN_ALM2MAP_RECURRENCE_CPP", "1")
    os.environ.setdefault("TORCHFITS_SPIN_MAP2ALM_RING_TORCH", "auto")
    os.environ.setdefault("TORCHFITS_SPIN_ALM2MAP_RING_TORCH", "auto")

    rng = np.random.default_rng(args.seed)
    npix = hp.nside2npix(args.nside)
    map_vals = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    nalm = alm_size(args.lmax)
    alm_vals = torch.from_numpy(rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).to(torch.complex128)
    qu_vals = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    spin_alm_e, spin_alm_b = _random_spin_alms(args.lmax, args.spin, args.seed + 17)

    backends = ["torch"]
    cpp_available = False
    try:
        import torchfits.cpp as _cpp  # type: ignore

        cpp_available = hasattr(_cpp, "_healpix_scalar_alm2map_direct_cpu")
    except Exception:
        cpp_available = False
    if cpp_available and args.include_torch_cpp_variant:
        backends.append("torch_cpp")
    try:
        import healpy as _hp  # type: ignore  # noqa: F401

        backends.append("healpy")
    except Exception:
        pass

    rows: list[Row] = []
    print("op            backend  nside  lmax   mops/s   time(s)", flush=True)
    for backend in backends:
        if backend == "torch" and args.nside >= 128 and os.environ.get("TORCHFITS_SPIN_MAP2ALM_RING_TORCH") != "force":
            print(f"Skipping {backend} for nside={args.nside} (matrix path) to avoid OOM")
            continue
        backend_impl = "torch" if backend == "torch_cpp" else backend
        prev_cpp_env = os.environ.get("TORCHFITS_SCALAR_ALM2MAP_CPP")
        if backend == "torch_cpp":
            os.environ["TORCHFITS_SCALAR_ALM2MAP_CPP"] = "1"
        elif prev_cpp_env is not None:
            os.environ["TORCHFITS_SCALAR_ALM2MAP_CPP"] = "0"
        try:
            if False: # Skip scalar for now to avoid OOM in matrix path
                t = _time_many(lambda: map2alm(map_vals, nside=args.nside, lmax=args.lmax, backend=backend_impl), runs=args.runs)
                rows.append(Row("map2alm", backend, args.nside, args.lmax, npix, t, npix / t / 1e6))

                t = _time_many(
                    lambda: map2alm_lsq(
                        map_vals,
                        lmax=args.lmax,
                        mmax=args.lmax,
                        nside=args.nside,
                        pol=False,
                        maxiter=args.lsq_maxiter,
                        backend=backend_impl,
                    ),
                    runs=args.runs,
                )
                rows.append(Row("map2alm_lsq", backend, args.nside, args.lmax, npix, t, npix / t / 1e6))

                t = _time_many(lambda: alm2map(alm_vals, nside=args.nside, lmax=args.lmax, backend=backend_impl), runs=args.runs)
                rows.append(Row("alm2map", backend, args.nside, args.lmax, npix, t, npix / t / 1e6))

                fl = torch.ones((args.lmax + 1,), dtype=torch.float64)
                t = _time_many(lambda: almxfl(alm_vals, fl), runs=args.runs)
                rows.append(Row("almxfl", backend, args.nside, args.lmax, nalm, t, nalm / t / 1e6))

                t = _time_many(lambda: alm2cl(alm_vals, lmax=args.lmax, mmax=args.lmax), runs=args.runs)
                rows.append(Row("alm2cl", backend, args.nside, args.lmax, args.lmax + 1, t, (args.lmax + 1) / t / 1e6))

                t = _time_many(lambda: anafast(map_vals, nside=args.nside, lmax=args.lmax, backend=backend_impl), runs=args.runs)
                rows.append(Row("anafast", backend, args.nside, args.lmax, npix, t, npix / t / 1e6))

            t = _time_many(
                lambda: map2alm_spin(qu_vals, spin=args.spin, nside=args.nside, lmax=args.lmax, backend=backend_impl),
                runs=args.runs,
            )
            r = Row("map2alm_spin", backend, args.nside, args.lmax, npix, t, npix / t / 1e6)
            rows.append(r)
            print(f"{r.op:13s} {r.backend:8s} {r.nside:5d} {r.lmax:5d} {r.mops_s:8.3f} {r.time_s:8.4f}", flush=True)

            t = _time_many(
                lambda: alm2map_spin(
                    (spin_alm_e, spin_alm_b),
                    nside=args.nside,
                    spin=args.spin,
                    lmax=args.lmax,
                    backend=backend_impl,
                ),
                runs=args.runs,
            )
            r = Row("alm2map_spin", backend, args.nside, args.lmax, npix, t, npix / t / 1e6)
            rows.append(r)
            print(f"{r.op:13s} {r.backend:8s} {r.nside:5d} {r.lmax:5d} {r.mops_s:8.3f} {r.time_s:8.4f}", flush=True)
        finally:
            if prev_cpp_env is None:
                os.environ.pop("TORCHFITS_SCALAR_ALM2MAP_CPP", None)
            else:
                os.environ["TORCHFITS_SCALAR_ALM2MAP_CPP"] = prev_cpp_env

    print("op            backend  nside  lmax   mops/s   time(s)")
    for r in rows:
        print(f"{r.op:13s} {r.backend:8s} {r.nside:5d} {r.lmax:5d} {r.mops_s:8.3f} {r.time_s:8.4f}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
