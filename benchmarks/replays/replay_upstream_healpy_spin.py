#!/usr/bin/env python3
"""Replay spin transform parity/throughput against official healpy releases."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import torch

from torchfits.sphere.spectral import alm2map_spin, alm_index, alm_size, map2alm_spin


@dataclass
class Case:
    nside: int
    lmax: int
    mmax: int | None
    spin: int
    nest: bool


@dataclass
class Row:
    operation: str
    nside: int
    lmax: int
    mmax: int
    spin: int
    nest: bool
    points: int
    rel_l2_error: float
    max_abs_error: float
    torchfits_mpts_s: float
    healpy_mpts_s: float
    ratio_vs_healpy: float


DEFAULT_CASES = [
    Case(nside=8, lmax=8, mmax=None, spin=1, nest=False),
    Case(nside=8, lmax=8, mmax=None, spin=2, nest=False),
    Case(nside=16, lmax=8, mmax=6, spin=3, nest=False),
    Case(nside=16, lmax=8, mmax=6, spin=2, nest=True),
    Case(nside=16, lmax=12, mmax=None, spin=2, nest=False),
    Case(nside=16, lmax=12, mmax=8, spin=2, nest=True),
    Case(nside=32, lmax=12, mmax=10, spin=2, nest=False),
    Case(nside=32, lmax=16, mmax=12, spin=2, nest=False),
]
EXTENDED_CASES = DEFAULT_CASES + [
    Case(nside=32, lmax=20, mmax=16, spin=3, nest=False),
    Case(nside=32, lmax=20, mmax=16, spin=2, nest=False),
    Case(nside=64, lmax=24, mmax=20, spin=2, nest=True),
]


def _time_many(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _random_spin_alms(
    lmax: int, mmax: int, spin: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nalm = alm_size(lmax, mmax)
    a_e = np.zeros(nalm, dtype=np.complex128)
    a_b = np.zeros(nalm, dtype=np.complex128)
    for m in range(mmax + 1):
        for ell in range(m, lmax + 1):
            if ell < spin:
                continue
            idx = alm_index(ell, m, lmax, mmax)
            scale = 1.0 / ((ell + 1) ** 2)
            if m == 0:
                a_e[idx] = scale * rng.normal()
                a_b[idx] = scale * rng.normal()
            else:
                a_e[idx] = scale * (rng.normal() + 1j * rng.normal())
                a_b[idx] = scale * (rng.normal() + 1j * rng.normal())
    return a_e, a_b


def _map2alm_case(case: Case, n_points: int, runs: int, seed: int) -> Row:
    rng = np.random.default_rng(seed)
    npix = 12 * case.nside * case.nside
    mm = case.lmax if case.mmax is None else int(case.mmax)

    qu_ring = rng.normal(size=(2, npix))
    if case.nest:
        qu_in = np.stack(
            [hp.reorder(qu_ring[0], r2n=True), hp.reorder(qu_ring[1], r2n=True)], axis=0
        )
    else:
        qu_in = qu_ring

    qu_t = torch.from_numpy(qu_in).to(torch.float64)

    def tf_fn():
        return map2alm_spin(
            qu_t,
            spin=case.spin,
            nside=case.nside,
            lmax=case.lmax,
            mmax=mm,
            nest=case.nest,
            backend="torch",
        )

    def hp_fn():
        arr = qu_in
        if case.nest:
            arr = np.stack(
                [hp.reorder(arr[0], n2r=True), hp.reorder(arr[1], n2r=True)], axis=0
            )
        return hp.map2alm_spin(arr, spin=case.spin, lmax=case.lmax, mmax=mm)

    t_tf = _time_many(tf_fn, runs=runs)
    t_hp = _time_many(hp_fn, runs=runs)

    tf_ae, tf_ab = tf_fn()
    hp_ae, hp_ab = hp_fn()
    hp_ae_t = torch.from_numpy(hp_ae).to(torch.complex128)
    hp_ab_t = torch.from_numpy(hp_ab).to(torch.complex128)

    tf_cat = torch.cat([tf_ae, tf_ab], dim=0)
    hp_cat = torch.cat([hp_ae_t, hp_ab_t], dim=0)
    rel = float(
        (
            torch.linalg.norm(tf_cat - hp_cat)
            / torch.linalg.norm(hp_cat).clamp_min(1e-15)
        ).item()
    )
    max_abs = float(torch.max(torch.abs(tf_cat - hp_cat)).item())
    n = min(int(n_points), npix)

    return Row(
        operation="map2alm_spin",
        nside=case.nside,
        lmax=case.lmax,
        mmax=mm,
        spin=case.spin,
        nest=case.nest,
        points=n,
        rel_l2_error=rel,
        max_abs_error=max_abs,
        torchfits_mpts_s=(n / t_tf) / 1e6,
        healpy_mpts_s=(n / t_hp) / 1e6,
        ratio_vs_healpy=t_hp / t_tf,
    )


def _alm2map_case(case: Case, n_points: int, runs: int, seed: int) -> Row:
    npix = 12 * case.nside * case.nside
    mm = case.lmax if case.mmax is None else int(case.mmax)
    ae_np, ab_np = _random_spin_alms(case.lmax, mm, case.spin, seed)
    ae_t = torch.from_numpy(ae_np).to(torch.complex128)
    ab_t = torch.from_numpy(ab_np).to(torch.complex128)

    def tf_fn():
        return alm2map_spin(
            (ae_t, ab_t),
            nside=case.nside,
            spin=case.spin,
            lmax=case.lmax,
            mmax=mm,
            nest=case.nest,
            backend="torch",
        )

    def hp_fn():
        out = hp.alm2map_spin(
            [ae_np, ab_np], nside=case.nside, spin=case.spin, lmax=case.lmax, mmax=mm
        )
        if case.nest:
            out = np.stack(
                [hp.reorder(out[0], r2n=True), hp.reorder(out[1], r2n=True)], axis=0
            )
        return out

    t_tf = _time_many(tf_fn, runs=runs)
    t_hp = _time_many(hp_fn, runs=runs)

    tf_qu = tf_fn()
    hp_qu = hp_fn()
    hp_qu_t = torch.from_numpy(np.asarray(hp_qu)).to(torch.float64)

    rel = float(
        (
            torch.linalg.norm(tf_qu - hp_qu_t)
            / torch.linalg.norm(hp_qu_t).clamp_min(1e-15)
        ).item()
    )
    max_abs = float(torch.max(torch.abs(tf_qu - hp_qu_t)).item())
    n = min(int(n_points), npix)

    return Row(
        operation="alm2map_spin",
        nside=case.nside,
        lmax=case.lmax,
        mmax=mm,
        spin=case.spin,
        nest=case.nest,
        points=n,
        rel_l2_error=rel,
        max_abs_error=max_abs,
        torchfits_mpts_s=(n / t_tf) / 1e6,
        healpy_mpts_s=(n / t_hp) / 1e6,
        ratio_vs_healpy=t_hp / t_tf,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument(
        "--case-set", choices=("default", "extended"), default="default"
    )
    parser.add_argument("--max-map2alm-rel-error", type=float, default=1.0e-9)
    parser.add_argument("--max-alm2map-rel-error", type=float, default=1.0e-9)
    parser.add_argument(
        "--min-ratio-vs-healpy",
        type=str,
        default="map2alm_spin=0.20,alm2map_spin=0.15",
        help="Comma-separated op=ratio thresholds for median(torchfits/healpy).",
    )
    parser.add_argument(
        "--disable-gates",
        action="store_true",
        help="Report replay metrics without enforcing thresholds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_healpy_spin.json"),
    )
    args = parser.parse_args()

    # Default to the validated torch ring profile for reproducible spin replay.
    os.environ.setdefault("TORCHFITS_RING_FOURIER_CPP", "1")
    os.environ.setdefault("TORCHFITS_SPIN_MAP2ALM_RECURRENCE_CPP", "0")
    os.environ.setdefault("TORCHFITS_SPIN_ALM2MAP_RECURRENCE_CPP", "0")
    os.environ.setdefault("TORCHFITS_SPIN_MAP2ALM_RING_TORCH", "auto")
    os.environ.setdefault("TORCHFITS_SPIN_ALM2MAP_RING_TORCH", "auto")
    os.environ.setdefault("TORCHFITS_SPIN_RING_AUTO_MIN_BYTES", str(32 * 1024 * 1024))

    ratio_thresholds: dict[str, float] = {}
    for part in [p.strip() for p in args.min_ratio_vs_healpy.split(",") if p.strip()]:
        op, raw = part.split("=", 1)
        ratio_thresholds[op.strip()] = float(raw.strip())

    cases = DEFAULT_CASES if args.case_set == "default" else EXTENDED_CASES

    rows: list[Row] = []
    for i, case in enumerate(cases):
        rows.append(
            _map2alm_case(
                case, n_points=args.n_points, runs=args.runs, seed=args.seed + 10 * i
            )
        )
        rows.append(
            _alm2map_case(
                case,
                n_points=args.n_points,
                runs=args.runs,
                seed=args.seed + 10 * i + 1,
            )
        )

    print(
        "operation      nside  lmax  mmax  nest  rel_l2       max_abs      tf_mpts/s  hp_mpts/s  ratio"
    )
    failures: list[str] = []
    for r in rows:
        print(
            f"{r.operation:13s} {r.nside:5d} {r.lmax:5d} {r.mmax:5d} {str(r.nest):5s}"
            f" {r.rel_l2_error:11.3e} {r.max_abs_error:11.3e}"
            f" {r.torchfits_mpts_s:10.3f} {r.healpy_mpts_s:10.3f} {r.ratio_vs_healpy:7.3f}"
        )
        if (
            r.operation == "map2alm_spin"
            and r.rel_l2_error > args.max_map2alm_rel_error
        ):
            failures.append(
                f"{r.operation}@nside={r.nside},lmax={r.lmax},mmax={r.mmax},nest={r.nest}:"
                f" rel_l2 {r.rel_l2_error:.3e} > {args.max_map2alm_rel_error:.3e}"
            )
        if (
            r.operation == "alm2map_spin"
            and r.rel_l2_error > args.max_alm2map_rel_error
        ):
            failures.append(
                f"{r.operation}@nside={r.nside},lmax={r.lmax},mmax={r.mmax},nest={r.nest}:"
                f" rel_l2 {r.rel_l2_error:.3e} > {args.max_alm2map_rel_error:.3e}"
            )

    med: dict[str, float] = {}
    for op in ("map2alm_spin", "alm2map_spin"):
        vals = [r.ratio_vs_healpy for r in rows if r.operation == op]
        med[op] = float(np.median(vals))
        thr = ratio_thresholds.get(op, 0.0)
        print(f"median ratio {op:13s}: {med[op]:.3f}x (threshold {thr:.3f}x)")
        if med[op] < thr:
            failures.append(f"{op}: median ratio {med[op]:.3f} < {thr:.3f}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "cases": [asdict(c) for c in cases],
                "case_set": args.case_set,
                "runs": args.runs,
                "rows": [asdict(r) for r in rows],
                "median_ratio_vs_healpy": med,
                "thresholds": {
                    "max_map2alm_rel_error": args.max_map2alm_rel_error,
                    "max_alm2map_rel_error": args.max_alm2map_rel_error,
                    "min_ratio_vs_healpy": ratio_thresholds,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"JSON written to: {args.json_out}")

    if args.disable_gates:
        return 0
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
