#!/usr/bin/env python3
"""Benchmark TorchFits HEALPix kernels against healpy on CPU/CUDA."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import importlib.metadata as importlib_metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("healpy is required for bench_healpix.py") from exc

try:
    from torchfits.wcs.healpix import (
        ang2pix_nested,
        ang2pix_ring,
        nest2ring,
        pix2ang_nested,
        pix2ang_ring,
        ring2nest,
    )
except Exception:
    # Allow benchmarks in envs where torchfits package/extensions are not installed.
    local_mod = (
        Path(__file__).resolve().parents[1] / "src" / "torchfits" / "wcs" / "healpix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "torchfits_wcs_healpix_local", local_mod
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load local HEALPix module from {local_mod}")
    healpix_local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(healpix_local)
    ang2pix_nested = healpix_local.ang2pix_nested
    ang2pix_ring = healpix_local.ang2pix_ring
    nest2ring = healpix_local.nest2ring
    pix2ang_nested = healpix_local.pix2ang_nested
    pix2ang_ring = healpix_local.pix2ang_ring
    ring2nest = healpix_local.ring2nest


def _healpy_provenance() -> dict[str, Any] | None:
    try:
        dist = importlib_metadata.distribution("healpy")
    except importlib_metadata.PackageNotFoundError:
        return None

    meta: dict[str, Any] = {
        "version": dist.version,
        "release_like": True,
        "source": "site-packages",
        "installer": (dist.read_text("INSTALLER") or "unknown").strip().lower()
        or "unknown",
        "reason": "installed from index/conda/wheel/sdist",
    }
    if meta["installer"] == "conda":
        meta["reason"] = "conda package"
        return meta

    raw_direct = dist.read_text("direct_url.json")
    if not raw_direct:
        return meta

    try:
        direct = json.loads(raw_direct)
    except json.JSONDecodeError:
        return meta

    url = str(direct.get("url", "")).strip()
    if url:
        meta["source"] = url
    if direct.get("vcs_info") is not None:
        meta["release_like"] = False
        meta["reason"] = "VCS install"
    elif direct.get("dir_info") is not None and direct.get("archive_info") is None:
        meta["release_like"] = False
        meta["reason"] = "local path/editable install"
    return meta


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_many(fn, runs: int, sync_device: torch.device | None = None) -> float:
    fn()
    if sync_device is not None:
        _sync(sync_device)
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_device is not None:
            _sync(sync_device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def _sample_lonlat(n: int, seed: int, profile: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if profile == "uniform":
        ra = rng.uniform(0.0, 360.0, n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
        return ra, dec

    transition = np.degrees(np.arcsin(2.0 / 3.0))
    if profile == "boundary":
        ra0 = (rng.integers(0, 8, size=n) * 45.0).astype(np.float64)
        ra = np.mod(ra0 + rng.normal(0.0, 1.0e-5, size=n), 360.0)

        block = np.array(
            [transition, -transition, 0.0, 89.9999, -89.9999],
            dtype=np.float64,
        )
        idx = rng.integers(0, block.size, size=n)
        dec = block[idx] + rng.normal(0.0, 1.0e-5, size=n)
        dec = np.clip(dec, -90.0, 90.0)
        return ra, dec

    n0 = n // 2
    n1 = n - n0
    ra0, dec0 = _sample_lonlat(n0, seed, "uniform")
    ra1, dec1 = _sample_lonlat(n1, seed + 1, "boundary")
    return np.concatenate([ra0, ra1]), np.concatenate([dec0, dec1])


def _sample_pix(nside: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    base = rng.integers(0, npix, size=n, dtype=np.int64)
    if n >= 32:
        base[:16] = np.arange(16, dtype=np.int64)
        base[16:32] = np.arange(npix - 16, npix, dtype=np.int64)
    return base


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if choice == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but no MPS device is available")
    return torch.device(choice)


def _run_benchmark(
    nside: int,
    n: int,
    runs: int,
    seed: int,
    profile: str,
    device: torch.device,
    compare_cpu: bool,
) -> list[dict[str, Any]]:
    ra, dec = _sample_lonlat(n, seed, profile)
    ring = _sample_pix(nside, n, seed + 1)
    nest = _sample_pix(nside, n, seed + 2)

    float_dtype = torch.float32 if device.type == "mps" else torch.float64
    ra_t = torch.from_numpy(ra).to(device=device, dtype=float_dtype)
    dec_t = torch.from_numpy(dec).to(device=device, dtype=float_dtype)
    ring_t = torch.from_numpy(ring).to(device=device, dtype=torch.int64)
    nest_t = torch.from_numpy(nest).to(device=device, dtype=torch.int64)

    cpu_ops: dict[str, float] | None = None
    if compare_cpu and device.type in {"cuda", "mps"}:
        ra_tc = ra_t.cpu()
        dec_tc = dec_t.cpu()
        ring_tc = ring_t.cpu()
        nest_tc = nest_t.cpu()
        cpu_device = torch.device("cpu")
        cpu_ops = {
            "ang2pix_ring": _time_many(
                lambda: ang2pix_ring(nside, ra_tc, dec_tc), runs, sync_device=cpu_device
            ),
            "ang2pix_nested": _time_many(
                lambda: ang2pix_nested(nside, ra_tc, dec_tc),
                runs,
                sync_device=cpu_device,
            ),
            "pix2ang_ring": _time_many(
                lambda: pix2ang_ring(nside, ring_tc), runs, sync_device=cpu_device
            ),
            "pix2ang_nested": _time_many(
                lambda: pix2ang_nested(nside, nest_tc), runs, sync_device=cpu_device
            ),
            "ring2nest": _time_many(
                lambda: ring2nest(nside, ring_tc), runs, sync_device=cpu_device
            ),
            "nest2ring": _time_many(
                lambda: nest2ring(nside, nest_tc), runs, sync_device=cpu_device
            ),
        }

    rows: list[dict[str, Any]] = []

    t_torch = _time_many(
        lambda: ang2pix_ring(nside, ra_t, dec_t), runs, sync_device=device
    )
    t_hp = _time_many(lambda: hp.ang2pix(nside, ra, dec, lonlat=True, nest=False), runs)
    p_t = ang2pix_ring(nside, ra_t, dec_t).cpu().numpy()
    p_h = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
    pix2ang_eps = 1.0e-10 if float_dtype == torch.float64 else 1.0e-4

    rows.append(
        {
            "operation": "ang2pix_ring",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch * 1000.0,
            "healpy_ms": t_hp * 1000.0,
            "torch_mpts_s": (n / t_torch) / 1e6,
            "healpy_mpts_s": (n / t_hp) / 1e6,
            "speedup_vs_healpy": t_hp / t_torch if t_torch > 0 else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["ang2pix_ring"] / t_torch)
            if cpu_ops is not None and t_torch > 0
            else float("nan"),
            "mismatches": int(np.sum(p_t != p_h)),
            "max_dra_deg": float("nan"),
            "p99_dra_deg": float("nan"),
            "max_ddec_deg": float("nan"),
            "p99_ddec_deg": float("nan"),
        }
    )

    t_torch_n = _time_many(
        lambda: ang2pix_nested(nside, ra_t, dec_t), runs, sync_device=device
    )
    t_hp_n = _time_many(
        lambda: hp.ang2pix(nside, ra, dec, lonlat=True, nest=True), runs
    )
    pn_t = ang2pix_nested(nside, ra_t, dec_t).cpu().numpy()
    pn_h = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    rows.append(
        {
            "operation": "ang2pix_nested",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch_n * 1000.0,
            "healpy_ms": t_hp_n * 1000.0,
            "torch_mpts_s": (n / t_torch_n) / 1e6,
            "healpy_mpts_s": (n / t_hp_n) / 1e6,
            "speedup_vs_healpy": t_hp_n / t_torch_n if t_torch_n > 0 else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["ang2pix_nested"] / t_torch_n)
            if cpu_ops is not None and t_torch_n > 0
            else float("nan"),
            "mismatches": int(np.sum(pn_t != pn_h)),
            "max_dra_deg": float("nan"),
            "p99_dra_deg": float("nan"),
            "max_ddec_deg": float("nan"),
            "p99_ddec_deg": float("nan"),
        }
    )

    t_torch_p2a_r = _time_many(
        lambda: pix2ang_ring(nside, ring_t), runs, sync_device=device
    )
    t_hp_p2a_r = _time_many(
        lambda: hp.pix2ang(nside, ring, nest=False, lonlat=True), runs
    )
    ra_r_t, dec_r_t = pix2ang_ring(nside, ring_t)
    ra_r_h, dec_r_h = hp.pix2ang(nside, ring, nest=False, lonlat=True)
    dra_r = np.abs(_ra_delta(ra_r_t.cpu().numpy(), ra_r_h))
    ddec_r = np.abs(dec_r_t.cpu().numpy() - dec_r_h)
    rows.append(
        {
            "operation": "pix2ang_ring",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch_p2a_r * 1000.0,
            "healpy_ms": t_hp_p2a_r * 1000.0,
            "torch_mpts_s": (n / t_torch_p2a_r) / 1e6,
            "healpy_mpts_s": (n / t_hp_p2a_r) / 1e6,
            "speedup_vs_healpy": t_hp_p2a_r / t_torch_p2a_r
            if t_torch_p2a_r > 0
            else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["pix2ang_ring"] / t_torch_p2a_r)
            if cpu_ops is not None and t_torch_p2a_r > 0
            else float("nan"),
            "mismatches": int(np.sum((dra_r > pix2ang_eps) | (ddec_r > pix2ang_eps))),
            "max_dra_deg": float(dra_r.max()),
            "p99_dra_deg": float(np.quantile(dra_r, 0.99)),
            "max_ddec_deg": float(ddec_r.max()),
            "p99_ddec_deg": float(np.quantile(ddec_r, 0.99)),
        }
    )

    t_torch_p2a_n = _time_many(
        lambda: pix2ang_nested(nside, nest_t), runs, sync_device=device
    )
    t_hp_p2a_n = _time_many(
        lambda: hp.pix2ang(nside, nest, nest=True, lonlat=True), runs
    )
    ra_n_t, dec_n_t = pix2ang_nested(nside, nest_t)
    ra_n_h, dec_n_h = hp.pix2ang(nside, nest, nest=True, lonlat=True)
    dra_n = np.abs(_ra_delta(ra_n_t.cpu().numpy(), ra_n_h))
    ddec_n = np.abs(dec_n_t.cpu().numpy() - dec_n_h)
    rows.append(
        {
            "operation": "pix2ang_nested",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch_p2a_n * 1000.0,
            "healpy_ms": t_hp_p2a_n * 1000.0,
            "torch_mpts_s": (n / t_torch_p2a_n) / 1e6,
            "healpy_mpts_s": (n / t_hp_p2a_n) / 1e6,
            "speedup_vs_healpy": t_hp_p2a_n / t_torch_p2a_n
            if t_torch_p2a_n > 0
            else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["pix2ang_nested"] / t_torch_p2a_n)
            if cpu_ops is not None and t_torch_p2a_n > 0
            else float("nan"),
            "mismatches": int(np.sum((dra_n > pix2ang_eps) | (ddec_n > pix2ang_eps))),
            "max_dra_deg": float(dra_n.max()),
            "p99_dra_deg": float(np.quantile(dra_n, 0.99)),
            "max_ddec_deg": float(ddec_n.max()),
            "p99_ddec_deg": float(np.quantile(ddec_n, 0.99)),
        }
    )

    t_torch_r2n = _time_many(lambda: ring2nest(nside, ring_t), runs, sync_device=device)
    t_hp_r2n = _time_many(lambda: hp.ring2nest(nside, ring), runs)
    r2n_t = ring2nest(nside, ring_t).cpu().numpy()
    r2n_h = hp.ring2nest(nside, ring)
    rows.append(
        {
            "operation": "ring2nest",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch_r2n * 1000.0,
            "healpy_ms": t_hp_r2n * 1000.0,
            "torch_mpts_s": (n / t_torch_r2n) / 1e6,
            "healpy_mpts_s": (n / t_hp_r2n) / 1e6,
            "speedup_vs_healpy": t_hp_r2n / t_torch_r2n
            if t_torch_r2n > 0
            else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["ring2nest"] / t_torch_r2n)
            if cpu_ops is not None and t_torch_r2n > 0
            else float("nan"),
            "mismatches": int(np.sum(r2n_t != r2n_h)),
            "max_dra_deg": float("nan"),
            "p99_dra_deg": float("nan"),
            "max_ddec_deg": float("nan"),
            "p99_ddec_deg": float("nan"),
        }
    )

    t_torch_n2r = _time_many(lambda: nest2ring(nside, nest_t), runs, sync_device=device)
    t_hp_n2r = _time_many(lambda: hp.nest2ring(nside, nest), runs)
    n2r_t = nest2ring(nside, nest_t).cpu().numpy()
    n2r_h = hp.nest2ring(nside, nest)
    rows.append(
        {
            "operation": "nest2ring",
            "nside": nside,
            "n_points": n,
            "sample_profile": profile,
            "device": device.type,
            "torch_ms": t_torch_n2r * 1000.0,
            "healpy_ms": t_hp_n2r * 1000.0,
            "torch_mpts_s": (n / t_torch_n2r) / 1e6,
            "healpy_mpts_s": (n / t_hp_n2r) / 1e6,
            "speedup_vs_healpy": t_hp_n2r / t_torch_n2r
            if t_torch_n2r > 0
            else float("nan"),
            "speedup_vs_torch_cpu": (cpu_ops["nest2ring"] / t_torch_n2r)
            if cpu_ops is not None and t_torch_n2r > 0
            else float("nan"),
            "mismatches": int(np.sum(n2r_t != n2r_h)),
            "max_dra_deg": float("nan"),
            "p99_dra_deg": float("nan"),
            "max_ddec_deg": float("nan"),
            "p99_ddec_deg": float("nan"),
        }
    )

    return rows


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print(
        " ".join(
            f"{c:>14s}"
            for c in (
                "operation",
                "torch_mpts/s",
                "healpy_mpts/s",
                "vs_healpy_x",
                "vs_cpu_x",
                "mismatches",
                "max_dra",
                "max_ddec",
            )
        )
    )
    for row in rows:
        print(
            f"{row['operation']:>14s} "
            f"{row['torch_mpts_s']:14.2f} "
            f"{row['healpy_mpts_s']:14.2f} "
            f"{row['speedup_vs_healpy']:14.2f} "
            f"{row['speedup_vs_torch_cpu']:14.2f} "
            f"{row['mismatches']:14d} "
            f"{row['max_dra_deg']:14.3e} "
            f"{row['max_ddec_deg']:14.3e}"
        )


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--sample-profile", choices=["uniform", "boundary", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="When running on CUDA/MPS, also benchmark Torch CPU and report accelerator speedup",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Optional JSON output path"
    )
    parser.add_argument(
        "--csv-out", type=Path, default=None, help="Optional CSV output path"
    )
    parser.add_argument(
        "--max-index-mismatches",
        type=int,
        default=0,
        help="Fail if any index-producing op exceeds this mismatch count",
    )
    parser.add_argument(
        "--max-pix2ang-dra-deg",
        type=float,
        default=1e-10,
        help="Fail if pix2ang max delta-RA exceeds this threshold",
    )
    parser.add_argument(
        "--max-pix2ang-ddec-deg",
        type=float,
        default=1e-10,
        help="Fail if pix2ang max delta-Dec exceeds this threshold",
    )
    parser.add_argument(
        "--min-cuda-speedup-vs-cpu",
        type=float,
        default=None,
        help="Optional fail threshold for minimum accelerator/CPU speedup (when --compare-cpu is active)",
    )
    parser.add_argument(
        "--allow-nonrelease-healpy",
        action="store_true",
        help="Allow local/editable/VCS healpy install for comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    provenance = _healpy_provenance()
    if provenance is None:
        print("healpy distribution metadata not found")
        if not args.allow_nonrelease_healpy:
            return 1
    else:
        status = "release-like" if provenance["release_like"] else "non-release"
        print(
            f"healpy version={provenance['version']} "
            f"[{status}] installer={provenance['installer']} source={provenance['source']}"
        )
        if not args.allow_nonrelease_healpy and not provenance["release_like"]:
            print(f"Refusing non-release healpy comparator: {provenance['reason']}")
            return 1

    rows = _run_benchmark(
        nside=args.nside,
        n=args.n_points,
        runs=args.runs,
        seed=args.seed,
        profile=args.sample_profile,
        device=device,
        compare_cpu=args.compare_cpu,
    )

    print(
        f"NSIDE={args.nside} N={args.n_points} runs={args.runs} device={device.type} profile={args.sample_profile}"
    )
    _print_rows(rows)

    if args.json_out is not None:
        _write_json(args.json_out, rows)
    if args.csv_out is not None:
        _write_csv(args.csv_out, rows)

    index_ops = {"ang2pix_ring", "ang2pix_nested", "ring2nest", "nest2ring"}
    bad_index = [
        r
        for r in rows
        if r["operation"] in index_ops and r["mismatches"] > args.max_index_mismatches
    ]
    if bad_index:
        print("\nMismatch threshold exceeded:")
        for row in bad_index:
            print(f"  {row['operation']}: mismatches={row['mismatches']}")
        return 1

    pix2ang_ops = {"pix2ang_ring", "pix2ang_nested"}
    bad_pix2ang = [
        r
        for r in rows
        if r["operation"] in pix2ang_ops
        and (
            r["max_dra_deg"] > args.max_pix2ang_dra_deg
            or r["max_ddec_deg"] > args.max_pix2ang_ddec_deg
        )
    ]
    if bad_pix2ang:
        print("\npix2ang error threshold exceeded:")
        for row in bad_pix2ang:
            print(
                "  "
                f"{row['operation']}: max_dra={row['max_dra_deg']:.3e} "
                f"max_ddec={row['max_ddec_deg']:.3e}"
            )
        return 1

    if (
        args.min_cuda_speedup_vs_cpu is not None
        and device.type in {"cuda", "mps"}
        and args.compare_cpu
    ):
        bad_speed = [
            r
            for r in rows
            if np.isfinite(r["speedup_vs_torch_cpu"])
            and r["speedup_vs_torch_cpu"] < args.min_cuda_speedup_vs_cpu
        ]
        if bad_speed:
            print("\nAccelerator speedup threshold not met:")
            for row in bad_speed:
                print(
                    "  "
                    f"{row['operation']}: accel/cpu={row['speedup_vs_torch_cpu']:.3f} "
                    f"(required {args.min_cuda_speedup_vs_cpu:.3f})"
                )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
