#!/usr/bin/env python3
"""Benchmark spherical geometry primitives across HEALPix ecosystem libraries."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import importlib.metadata as importlib_metadata
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from torchfits.wcs.healpix import (
        ang2pix_nested as tf_ang2pix_nested,
        ang2pix_ring as tf_ang2pix_ring,
        nest2ring as tf_nest2ring,
        pix2ang_nested as tf_pix2ang_nested,
        pix2ang_ring as tf_pix2ang_ring,
        ring2nest as tf_ring2nest,
    )
except Exception:
    # Allow benchmarks in envs where torchfits package/extensions are not installed.
    local_mod = Path(__file__).resolve().parents[1] / "src" / "torchfits" / "wcs" / "healpix.py"
    spec = importlib.util.spec_from_file_location("torchfits_wcs_healpix_local", local_mod)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load local HEALPix module from {local_mod}")
    healpix_local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(healpix_local)
    tf_ang2pix_nested = healpix_local.ang2pix_nested
    tf_ang2pix_ring = healpix_local.ang2pix_ring
    tf_nest2ring = healpix_local.nest2ring
    tf_pix2ang_nested = healpix_local.pix2ang_nested
    tf_pix2ang_ring = healpix_local.pix2ang_ring
    tf_ring2nest = healpix_local.ring2nest


IndexFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
IndexPixFn = Callable[[np.ndarray], np.ndarray]
AngleFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
OPS = (
    "ang2pix_ring",
    "ang2pix_nested",
    "pix2ang_ring",
    "pix2ang_nested",
    "ring2nest",
    "nest2ring",
)

COMPARATOR_DISTS: dict[str, str] = {
    "healpy": "healpy",
    "hpgeom": "hpgeom",
    "astropy-healpix": "astropy-healpix",
    "healpix": "healpix",
    "mhealpy": "mhealpy",
}


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
        block = np.array([transition, -transition, 0.0, 89.9999, -89.9999], dtype=np.float64)
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
    out = rng.integers(0, npix, size=n, dtype=np.int64)
    if n >= 32:
        out[:16] = np.arange(16, dtype=np.int64)
        out[16:32] = np.arange(npix - 16, npix, dtype=np.int64)
    return out


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def _index_mismatch(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def _angle_metrics(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_ref: np.ndarray,
    dec_ref: np.ndarray,
    eps: float,
) -> tuple[int, float, float, float, float]:
    dra = np.abs(_ra_delta(ra, ra_ref))
    ddec = np.abs(dec - dec_ref)
    mismatches = int(np.sum((dra > eps) | (ddec > eps)))
    return mismatches, float(dra.max()), float(np.quantile(dra, 0.99)), float(ddec.max()), float(np.quantile(ddec, 0.99))


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _parse_ratio_spec(spec: str | None) -> dict[str, float]:
    if spec is None:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return out
    valid = set(OPS)
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid ratio spec segment '{part}', expected op=value")
        op, raw = part.split("=", 1)
        op = op.strip()
        raw = raw.strip()
        if op not in valid:
            raise ValueError(f"Unknown operation '{op}' in ratio spec")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid float ratio '{raw}' for operation '{op}'") from exc
        if value <= 0.0:
            raise ValueError(f"Ratio threshold must be positive for operation '{op}'")
        out[op] = value
    return out


def _ratios_vs_healpy(rows: list[dict[str, Any]], library: str) -> dict[str, float]:
    by_key = {(str(r["library"]), str(r["operation"])): float(r["mpts_s"]) for r in rows}
    ratios: dict[str, float] = {}
    for op in OPS:
        a = by_key.get((library, op))
        b = by_key.get(("healpy", op))
        if a is None or b is None or b <= 0.0:
            continue
        ratios[op] = a / b
    return ratios


def _import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _distribution_provenance(dist_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    meta: dict[str, Any] = {
        "name": dist.metadata.get("Name", dist_name),
        "version": dist.version,
        "release_like": True,
        "source": "site-packages",
        "installer": (dist.read_text("INSTALLER") or "unknown").strip().lower() or "unknown",
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


def _make_torchfits_adapter(
    nside: int,
    ra: np.ndarray,
    dec: np.ndarray,
    ring: np.ndarray,
    nest: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Callable[..., Any]], float]:
    float_dtype = torch.float32 if device.type == "mps" else torch.float64
    ra_t = torch.from_numpy(ra).to(device=device, dtype=float_dtype)
    dec_t = torch.from_numpy(dec).to(device=device, dtype=float_dtype)
    ring_t = torch.from_numpy(ring).to(device=device, dtype=torch.int64)
    nest_t = torch.from_numpy(nest).to(device=device, dtype=torch.int64)

    def ang2pix_ring(_: np.ndarray, __: np.ndarray) -> np.ndarray:
        return tf_ang2pix_ring(nside, ra_t, dec_t).cpu().numpy()

    def ang2pix_nested(_: np.ndarray, __: np.ndarray) -> np.ndarray:
        return tf_ang2pix_nested(nside, ra_t, dec_t).cpu().numpy()

    def pix2ang_ring(_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ra_o, dec_o = tf_pix2ang_ring(nside, ring_t)
        return ra_o.cpu().numpy(), dec_o.cpu().numpy()

    def pix2ang_nested(_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ra_o, dec_o = tf_pix2ang_nested(nside, nest_t)
        return ra_o.cpu().numpy(), dec_o.cpu().numpy()

    def ring2nest(_: np.ndarray) -> np.ndarray:
        return tf_ring2nest(nside, ring_t).cpu().numpy()

    def nest2ring(_: np.ndarray) -> np.ndarray:
        return tf_nest2ring(nside, nest_t).cpu().numpy()

    eps = 1.0e-4 if device.type == "mps" else 1.0e-10
    return {
        "ang2pix_ring": ang2pix_ring,
        "ang2pix_nested": ang2pix_nested,
        "pix2ang_ring": pix2ang_ring,
        "pix2ang_nested": pix2ang_nested,
        "ring2nest": ring2nest,
        "nest2ring": nest2ring,
    }, eps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--sample-profile", choices=["uniform", "boundary", "mixed"], default="mixed")
    parser.add_argument(
        "--libraries",
        type=str,
        default="torchfits,healpy,hpgeom,astropy-healpix,healpix,mhealpy",
        help="Comma-separated library aliases",
    )
    parser.add_argument(
        "--allow-nonrelease-distributions",
        action="store_true",
        help="Allow comparator libraries installed from local/editable/VCS sources",
    )
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any requested library is unavailable")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--max-index-mismatches", type=int, default=None)
    parser.add_argument("--max-pix2ang-dra-deg", type=float, default=None)
    parser.add_argument("--max-pix2ang-ddec-deg", type=float, default=None)
    parser.add_argument(
        "--min-ratio-vs-healpy",
        type=str,
        default=None,
        help="Comma-separated op=ratio thresholds for torchfits/healpy",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    nside = args.nside
    n = args.n_points

    ra, dec = _sample_lonlat(n, args.seed, args.sample_profile)
    ring = _sample_pix(nside, n, args.seed + 1)
    nest = _sample_pix(nside, n, args.seed + 2)

    requested = [x.strip().lower() for x in args.libraries.split(",") if x.strip()]
    missing: list[str] = []

    healpy = _import_optional("healpy")
    hpgeom_compat = _import_optional("hpgeom.healpy_compat")
    astropy_healpy = _import_optional("astropy_healpix.healpy")
    healpix_mod = _import_optional("healpix")
    mhealpy_single = _import_optional("mhealpy.pixelfunc.single")

    adapters: dict[str, dict[str, Callable[..., Any]]] = {}
    eps_by_lib: dict[str, float] = {}
    sync_by_lib: dict[str, torch.device | None] = {}

    tf_adapter, tf_eps = _make_torchfits_adapter(nside, ra, dec, ring, nest, device)
    adapters["torchfits"] = tf_adapter
    eps_by_lib["torchfits"] = tf_eps
    sync_by_lib["torchfits"] = device

    if healpy is not None:
        adapters["healpy"] = {
            "ang2pix_ring": lambda lon, lat: healpy.ang2pix(nside, lon, lat, lonlat=True, nest=False),
            "ang2pix_nested": lambda lon, lat: healpy.ang2pix(nside, lon, lat, lonlat=True, nest=True),
            "pix2ang_ring": lambda pix: healpy.pix2ang(nside, pix, lonlat=True, nest=False),
            "pix2ang_nested": lambda pix: healpy.pix2ang(nside, pix, lonlat=True, nest=True),
            "ring2nest": lambda pix: healpy.ring2nest(nside, pix),
            "nest2ring": lambda pix: healpy.nest2ring(nside, pix),
        }
        eps_by_lib["healpy"] = 1.0e-10
        sync_by_lib["healpy"] = None

    if hpgeom_compat is not None:
        adapters["hpgeom"] = {
            "ang2pix_ring": lambda lon, lat: hpgeom_compat.ang2pix(nside, lon, lat, lonlat=True, nest=False),
            "ang2pix_nested": lambda lon, lat: hpgeom_compat.ang2pix(nside, lon, lat, lonlat=True, nest=True),
            "pix2ang_ring": lambda pix: hpgeom_compat.pix2ang(nside, pix, lonlat=True, nest=False),
            "pix2ang_nested": lambda pix: hpgeom_compat.pix2ang(nside, pix, lonlat=True, nest=True),
            "ring2nest": lambda pix: hpgeom_compat.ring2nest(nside, pix),
            "nest2ring": lambda pix: hpgeom_compat.nest2ring(nside, pix),
        }
        eps_by_lib["hpgeom"] = 1.0e-10
        sync_by_lib["hpgeom"] = None

    if astropy_healpy is not None:
        adapters["astropy-healpix"] = {
            "ang2pix_ring": lambda lon, lat: astropy_healpy.ang2pix(nside, lon, lat, lonlat=True, nest=False),
            "ang2pix_nested": lambda lon, lat: astropy_healpy.ang2pix(nside, lon, lat, lonlat=True, nest=True),
            "pix2ang_ring": lambda pix: astropy_healpy.pix2ang(nside, pix, lonlat=True, nest=False),
            "pix2ang_nested": lambda pix: astropy_healpy.pix2ang(nside, pix, lonlat=True, nest=True),
            "ring2nest": lambda pix: astropy_healpy.ring2nest(nside, pix),
            "nest2ring": lambda pix: astropy_healpy.nest2ring(nside, pix),
        }
        eps_by_lib["astropy-healpix"] = 1.0e-10
        sync_by_lib["astropy-healpix"] = None

    if healpix_mod is not None:
        adapters["healpix"] = {
            "ang2pix_ring": lambda lon, lat: healpix_mod.ang2pix(nside, lon, lat, lonlat=True, nest=False),
            "ang2pix_nested": lambda lon, lat: healpix_mod.ang2pix(nside, lon, lat, lonlat=True, nest=True),
            "pix2ang_ring": lambda pix: healpix_mod.pix2ang(nside, pix, lonlat=True, nest=False),
            "pix2ang_nested": lambda pix: healpix_mod.pix2ang(nside, pix, lonlat=True, nest=True),
            "ring2nest": lambda pix: healpix_mod.ring2nest(nside, pix),
            "nest2ring": lambda pix: healpix_mod.nest2ring(nside, pix),
        }
        eps_by_lib["healpix"] = 1.0e-10
        sync_by_lib["healpix"] = None

    if mhealpy_single is not None:
        adapters["mhealpy"] = {
            "ang2pix_ring": lambda lon, lat: mhealpy_single.ang2pix(nside, lon, lat, lonlat=True, nest=False),
            "ang2pix_nested": lambda lon, lat: mhealpy_single.ang2pix(nside, lon, lat, lonlat=True, nest=True),
            "pix2ang_ring": lambda pix: mhealpy_single.pix2ang(nside, pix, lonlat=True, nest=False),
            "pix2ang_nested": lambda pix: mhealpy_single.pix2ang(nside, pix, lonlat=True, nest=True),
            "ring2nest": lambda pix: mhealpy_single.ring2nest(nside, pix),
            "nest2ring": lambda pix: mhealpy_single.nest2ring(nside, pix),
        }
        eps_by_lib["mhealpy"] = 1.0e-10
        sync_by_lib["mhealpy"] = None

    for lib in requested:
        if lib not in adapters:
            missing.append(lib)

    comparator_provenance: list[tuple[str, dict[str, Any]]] = []
    provenance_violations: list[str] = []
    for lib in requested:
        if lib == "torchfits" or lib not in adapters:
            continue
        dist_name = COMPARATOR_DISTS.get(lib)
        if dist_name is None:
            continue
        provenance = _distribution_provenance(dist_name)
        if provenance is None:
            provenance_violations.append(f"{lib}: distribution metadata not found")
            continue
        comparator_provenance.append((lib, provenance))
        if not args.allow_nonrelease_distributions and not provenance["release_like"]:
            provenance_violations.append(
                f"{lib}: {provenance['reason']} ({provenance['source']})"
            )

    if comparator_provenance:
        print("Comparator package provenance:")
        for lib, provenance in comparator_provenance:
            status = "release-like" if provenance["release_like"] else "non-release"
            print(
                f"  {lib}: {provenance['version']} "
                f"[{status}] installer={provenance['installer']} source={provenance['source']}"
            )

    if provenance_violations:
        print("\nNon-release comparator distributions detected:")
        for line in provenance_violations:
            print(f"  - {line}")
        if not args.allow_nonrelease_distributions:
            return 1

    if args.strict_missing and missing:
        print(f"Missing requested libraries: {', '.join(missing)}")
        return 1

    ref = adapters.get("healpy")
    ref_outputs: dict[str, Any] = {}
    if ref is not None:
        ref_outputs["ang2pix_ring"] = ref["ang2pix_ring"](ra, dec)
        ref_outputs["ang2pix_nested"] = ref["ang2pix_nested"](ra, dec)
        ref_outputs["pix2ang_ring"] = ref["pix2ang_ring"](ring)
        ref_outputs["pix2ang_nested"] = ref["pix2ang_nested"](nest)
        ref_outputs["ring2nest"] = ref["ring2nest"](ring)
        ref_outputs["nest2ring"] = ref["nest2ring"](nest)

    # On MPS, torch inputs are float32; compare against float32-quantized lon/lat
    # to avoid attributing input quantization deltas to kernel correctness.
    ref_outputs_torchfits_mps: dict[str, Any] | None = None
    if ref is not None and device.type == "mps":
        ra_mps_ref = ra.astype(np.float32).astype(np.float64)
        dec_mps_ref = dec.astype(np.float32).astype(np.float64)
        ref_outputs_torchfits_mps = dict(ref_outputs)
        ref_outputs_torchfits_mps["ang2pix_ring"] = ref["ang2pix_ring"](ra_mps_ref, dec_mps_ref)
        ref_outputs_torchfits_mps["ang2pix_nested"] = ref["ang2pix_nested"](ra_mps_ref, dec_mps_ref)

    rows: list[dict[str, Any]] = []
    try:
        min_ratios = _parse_ratio_spec(args.min_ratio_vs_healpy)
    except ValueError as exc:
        print(f"Invalid --min-ratio-vs-healpy: {exc}")
        return 1

    ops = list(OPS)

    for lib in requested:
        adapter = adapters.get(lib)
        if adapter is None:
            continue
        for op in ops:
            fn = adapter[op]
            sync_device = sync_by_lib[lib]

            if op in {"ang2pix_ring", "ang2pix_nested"}:
                elapsed = _time_many(lambda: fn(ra, dec), args.runs, sync_device=sync_device)
                out = fn(ra, dec)
            elif op in {"ring2nest"}:
                elapsed = _time_many(lambda: fn(ring), args.runs, sync_device=sync_device)
                out = fn(ring)
            elif op in {"nest2ring"}:
                elapsed = _time_many(lambda: fn(nest), args.runs, sync_device=sync_device)
                out = fn(nest)
            elif op == "pix2ang_ring":
                elapsed = _time_many(lambda: fn(ring), args.runs, sync_device=sync_device)
                out = fn(ring)
            else:
                elapsed = _time_many(lambda: fn(nest), args.runs, sync_device=sync_device)
                out = fn(nest)

            row: dict[str, Any] = {
                "library": lib,
                "operation": op,
                "nside": nside,
                "n_points": n,
                "sample_profile": args.sample_profile,
                "device": device.type if lib == "torchfits" else "cpu",
                "ms": elapsed * 1000.0,
                "mpts_s": (n / elapsed) / 1e6,
                "mismatches": float("nan"),
                "max_dra_deg": float("nan"),
                "p99_dra_deg": float("nan"),
                "max_ddec_deg": float("nan"),
                "p99_ddec_deg": float("nan"),
            }

            if ref is not None and lib != "healpy":
                ref_view = ref_outputs_torchfits_mps if (lib == "torchfits" and ref_outputs_torchfits_mps is not None) else ref_outputs
                if op in {"ang2pix_ring", "ang2pix_nested", "ring2nest", "nest2ring"}:
                    row["mismatches"] = _index_mismatch(np.asarray(out), np.asarray(ref_view[op]))
                else:
                    eps = eps_by_lib[lib]
                    ra_o, dec_o = out
                    ra_ref, dec_ref = ref_view[op]
                    mism, max_dra, p99_dra, max_ddec, p99_ddec = _angle_metrics(
                        np.asarray(ra_o),
                        np.asarray(dec_o),
                        np.asarray(ra_ref),
                        np.asarray(dec_ref),
                        eps=eps,
                    )
                    row["mismatches"] = mism
                    row["max_dra_deg"] = max_dra
                    row["p99_dra_deg"] = p99_dra
                    row["max_ddec_deg"] = max_ddec
                    row["p99_ddec_deg"] = p99_ddec

            rows.append(row)

    print(
        " ".join(
            f"{c:>14s}"
            for c in ("library", "operation", "mpts/s", "mismatch", "max_dra", "max_ddec")
        )
    )
    for row in rows:
        print(
            f"{row['library']:>14s} "
            f"{row['operation']:>14s} "
            f"{row['mpts_s']:14.2f} "
            f"{int(row['mismatches']) if np.isfinite(row['mismatches']) else -1:14d} "
            f"{row['max_dra_deg']:14.3e} "
            f"{row['max_ddec_deg']:14.3e}"
        )

    if missing:
        print(f"\nUnavailable libraries: {', '.join(missing)}")

    if args.csv_out is not None and rows:
        _write_csv(args.csv_out, rows)
    if args.json_out is not None and rows:
        _write_json(args.json_out, rows)

    if args.max_index_mismatches is not None:
        bad = [
            r
            for r in rows
            if r["operation"] in {"ang2pix_ring", "ang2pix_nested", "ring2nest", "nest2ring"}
            and np.isfinite(r["mismatches"])
            and int(r["mismatches"]) > args.max_index_mismatches
        ]
        if bad:
            print("\nIndex mismatch threshold exceeded:")
            for row in bad:
                print(f"  {row['library']} {row['operation']}: {int(row['mismatches'])}")
            return 1

    if args.max_pix2ang_dra_deg is not None:
        bad = [
            r
            for r in rows
            if r["operation"] in {"pix2ang_ring", "pix2ang_nested"}
            and np.isfinite(r["max_dra_deg"])
            and r["max_dra_deg"] > args.max_pix2ang_dra_deg
        ]
        if bad:
            print("\nRA delta threshold exceeded:")
            for row in bad:
                print(f"  {row['library']} {row['operation']}: {row['max_dra_deg']:.3e}")
            return 1

    if args.max_pix2ang_ddec_deg is not None:
        bad = [
            r
            for r in rows
            if r["operation"] in {"pix2ang_ring", "pix2ang_nested"}
            and np.isfinite(r["max_ddec_deg"])
            and r["max_ddec_deg"] > args.max_pix2ang_ddec_deg
        ]
        if bad:
            print("\nDec delta threshold exceeded:")
            for row in bad:
                print(f"  {row['library']} {row['operation']}: {row['max_ddec_deg']:.3e}")
            return 1

    if min_ratios and "torchfits" in requested and "healpy" in adapters:
        ratios = _ratios_vs_healpy(rows, "torchfits")
        bad_ratios = []
        for op, threshold in min_ratios.items():
            got = ratios.get(op)
            if got is None:
                bad_ratios.append((op, threshold, float("nan")))
            elif got < threshold:
                bad_ratios.append((op, threshold, got))
        if bad_ratios:
            print("\nTorchFits/healpy ratio threshold exceeded:")
            for op, threshold, got in bad_ratios:
                if np.isfinite(got):
                    print(f"  {op}: {got:.3f} < {threshold:.3f}")
                else:
                    print(f"  {op}: unavailable < {threshold:.3f}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
