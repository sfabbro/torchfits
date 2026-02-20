#!/usr/bin/env python3
"""End-to-end benchmark: table predicate pushdown + spherical HEALPix primitives."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

import torchfits
import torchfits.cpp as cpp
from torchfits.wcs import healpix as hp


@dataclass
class Row:
    operation: str
    backend: str
    n_rows_total: int
    n_rows_selected: int
    nside: int
    time_s: float
    mrows_s: float
    checksum: int


def _time_many(fn, runs: int) -> tuple[float, tuple[int, int]]:
    # warm
    warm = fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals)), warm


def _write_synth_table(path: Path, n_rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n_rows).astype(np.float64)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n_rows))).astype(np.float64)
    mag = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)
    band = rng.integers(0, 6, size=n_rows, dtype=np.int16)
    flux = (rng.normal(loc=1.0, scale=0.2, size=n_rows) * (1.0 + 0.1 * band)).astype(np.float32)

    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="DEC", format="D", array=dec),
            fits.Column(name="MAG", format="E", array=mag),
            fits.Column(name="BAND", format="I", array=band),
            fits.Column(name="FLUX", format="E", array=flux),
        ]
    )
    hdu.writeto(path, overwrite=True)


def _arrow_col_to_numpy(table, name: str) -> np.ndarray:
    col = table.column(name)
    if hasattr(col, "combine_chunks"):
        col = col.combine_chunks()
    return col.to_numpy(zero_copy_only=False)


def _sphere_reduce(ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int) -> int:
    # Copy to avoid non-writable NumPy views warning when converting to torch.
    ra_t = torch.from_numpy(np.asarray(ra_deg).copy()).to(torch.float64)
    dec_t = torch.from_numpy(np.asarray(dec_deg).copy()).to(torch.float64)
    pix = hp.ang2pix_ring(nside, ra_t, dec_t)
    npix = hp.nside2npix(nside)
    counts = torch.bincount(pix, minlength=npix)

    # Include neighborhood topology work in the integrated pipeline.
    take = min(int(pix.numel()), 4096)
    if take > 0:
        neigh = hp.get_all_neighbours(nside, pix[:take], nest=False)
        neigh_term = int((neigh >= 0).sum().item())
    else:
        neigh_term = 0
    checksum = int(counts[: min(npix, 2048)].sum().item()) + neigh_term
    return checksum


def _apply_filter_mask(mag: np.ndarray, band: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return (mag > 0.25) & (mag < 0.75) & (band > 0) & (band < 4) & (dec > -30.0) & (dec < 30.0)


def _run_read_where(path: Path, where: str, backend: str, runs: int) -> Row:
    cols = ["RA", "DEC", "MAG", "BAND"]
    filters = [
        ("MAG", ">", 0.25),
        ("MAG", "<", 0.75),
        ("BAND", ">", 0),
        ("BAND", "<", 4),
        ("DEC", ">", -30.0),
        ("DEC", "<", 30.0),
    ]

    def _fn() -> tuple[int, int]:
        if backend == "torch":
            table = torchfits.table.read(str(path), hdu=1, columns=cols, where=None, backend="torch")
            mag = _arrow_col_to_numpy(table, "MAG")
            band = _arrow_col_to_numpy(table, "BAND")
            dec = _arrow_col_to_numpy(table, "DEC")
            mask = _apply_filter_mask(mag, band, dec)
            n_sel = int(mask.sum())
            checksum = int(band[mask][: min(n_sel, 1024)].sum()) if n_sel else 0
        else:
            data = cpp.read_fits_table_filtered(str(path), 1, cols, filters)
            band_t = torch.as_tensor(data["BAND"])
            n_sel = int(band_t.numel())
            checksum = int(band_t[: min(n_sel, 1024)].sum().item()) if n_sel else 0
        return n_sel, checksum

    t, (n_sel, checksum) = _time_many(_fn, runs=runs)
    return Row(
        operation="read_where",
        backend=backend,
        n_rows_total=_n_rows_in_table(path),
        n_rows_selected=n_sel,
        nside=0,
        time_s=t,
        mrows_s=(n_sel / t) / 1e6 if t > 0 else float("nan"),
        checksum=checksum,
    )


def _run_pipeline_where(path: Path, where: str, backend: str, nside: int, runs: int) -> Row:
    cols = ["RA", "DEC", "MAG", "BAND"]
    filters = [
        ("MAG", ">", 0.25),
        ("MAG", "<", 0.75),
        ("BAND", ">", 0),
        ("BAND", "<", 4),
        ("DEC", ">", -30.0),
        ("DEC", "<", 30.0),
    ]

    def _fn() -> tuple[int, int]:
        if backend == "torch":
            table = torchfits.table.read(str(path), hdu=1, columns=cols, where=None, backend="torch")
            ra_all = _arrow_col_to_numpy(table, "RA")
            dec_all = _arrow_col_to_numpy(table, "DEC")
            mag = _arrow_col_to_numpy(table, "MAG")
            band = _arrow_col_to_numpy(table, "BAND")
            mask = _apply_filter_mask(mag, band, dec_all)
            ra = ra_all[mask]
            dec = dec_all[mask]
            n_sel = int(mask.sum())
        else:
            data = cpp.read_fits_table_filtered(str(path), 1, cols, filters)
            ra = torch.as_tensor(data["RA"]).detach().cpu().numpy()
            dec = torch.as_tensor(data["DEC"]).detach().cpu().numpy()
            n_sel = int(ra.shape[0])
        return n_sel, _sphere_reduce(ra, dec, nside=nside)

    t, (n_sel, checksum) = _time_many(_fn, runs=runs)
    return Row(
        operation="pipeline_where",
        backend=backend,
        n_rows_total=_n_rows_in_table(path),
        n_rows_selected=n_sel,
        nside=nside,
        time_s=t,
        mrows_s=(n_sel / t) / 1e6 if t > 0 else float("nan"),
        checksum=checksum,
    )


def _run_pipeline_no_pushdown(path: Path, nside: int, runs: int) -> Row:
    cols = ["RA", "DEC", "MAG", "BAND"]

    def _fn() -> tuple[int, int]:
        table = torchfits.table.read(str(path), hdu=1, columns=cols, where=None, backend="cpp_numpy")
        ra = _arrow_col_to_numpy(table, "RA")
        dec = _arrow_col_to_numpy(table, "DEC")
        mag = _arrow_col_to_numpy(table, "MAG")
        band = _arrow_col_to_numpy(table, "BAND")
        mask = _apply_filter_mask(mag, band, dec)
        ra_sel = ra[mask]
        dec_sel = dec[mask]
        return int(mask.sum()), _sphere_reduce(ra_sel, dec_sel, nside=nside)

    t, (n_sel, checksum) = _time_many(_fn, runs=runs)
    return Row(
        operation="pipeline_no_pushdown",
        backend="cpp_numpy",
        n_rows_total=_n_rows_in_table(path),
        n_rows_selected=n_sel,
        nside=nside,
        time_s=t,
        mrows_s=(n_sel / t) / 1e6 if t > 0 else float("nan"),
        checksum=checksum,
    )


def _n_rows_in_table(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        return int(len(hdul[1].data))


def _run_all_rows(table_path: Path, where: str, nside: int, runs: int) -> list[Row]:
    rows: list[Row] = []
    rows.append(_run_read_where(table_path, where=where, backend="cpp_filtered", runs=runs))
    rows.append(_run_read_where(table_path, where=where, backend="torch", runs=runs))
    rows.append(_run_pipeline_where(table_path, where=where, backend="cpp_filtered", nside=nside, runs=runs))
    rows.append(_run_pipeline_where(table_path, where=where, backend="torch", nside=nside, runs=runs))
    rows.append(_run_pipeline_no_pushdown(table_path, nside=nside, runs=runs))
    return rows


def _compute_ratios_and_failures(
    rows: list[Row],
    *,
    min_read_ratio_filtered_vs_torch: float,
    min_pipeline_ratio_filtered_vs_no_pushdown: float,
    min_pipeline_ratio_torch_vs_no_pushdown: float,
) -> tuple[dict[str, float], list[str]]:
    read_cpp = next(r for r in rows if r.operation == "read_where" and r.backend == "cpp_filtered")
    read_torch = next(r for r in rows if r.operation == "read_where" and r.backend == "torch")
    pipe_cpp = next(r for r in rows if r.operation == "pipeline_where" and r.backend == "cpp_filtered")
    pipe_torch = next(r for r in rows if r.operation == "pipeline_where" and r.backend == "torch")
    pipe_no = next(r for r in rows if r.operation == "pipeline_no_pushdown")

    if read_cpp.n_rows_selected != read_torch.n_rows_selected:
        raise RuntimeError("row-count mismatch between cpp_numpy and torch where paths")
    if pipe_cpp.checksum != pipe_torch.checksum:
        raise RuntimeError("pipeline checksum mismatch between cpp_numpy and torch where paths")

    read_ratio = (read_cpp.mrows_s / read_torch.mrows_s) if read_torch.mrows_s > 0 else float("nan")
    pipe_cpp_ratio = (pipe_cpp.mrows_s / pipe_no.mrows_s) if pipe_no.mrows_s > 0 else float("nan")
    pipe_torch_ratio = (pipe_torch.mrows_s / pipe_no.mrows_s) if pipe_no.mrows_s > 0 else float("nan")

    failures: list[str] = []
    if read_ratio < min_read_ratio_filtered_vs_torch:
        failures.append(
            f"read_where cpp_filtered/torch ratio {read_ratio:.3f} < {min_read_ratio_filtered_vs_torch:.3f}"
        )
    if pipe_cpp_ratio < min_pipeline_ratio_filtered_vs_no_pushdown:
        failures.append(
            "pipeline_where cpp_filtered/no_pushdown ratio "
            f"{pipe_cpp_ratio:.3f} < {min_pipeline_ratio_filtered_vs_no_pushdown:.3f}"
        )
    if pipe_torch_ratio < min_pipeline_ratio_torch_vs_no_pushdown:
        failures.append(
            f"pipeline_where torch/no_pushdown ratio {pipe_torch_ratio:.3f} < {min_pipeline_ratio_torch_vs_no_pushdown:.3f}"
        )
    return (
        {
            "read_filtered_vs_torch": read_ratio,
            "pipeline_filtered_vs_no_pushdown": pipe_cpp_ratio,
            "pipeline_torch_vs_no_pushdown": pipe_torch_ratio,
        },
        failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-path", type=Path, default=Path("bench_results/pipeline_table_sphere.fits"))
    parser.add_argument("--n-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--regen", action="store_true")
    parser.add_argument("--min-read-ratio-filtered-vs-torch", type=float, default=0.2)
    parser.add_argument("--min-pipeline-ratio-filtered-vs-no-pushdown", type=float, default=0.7)
    parser.add_argument("--min-pipeline-ratio-torch-vs-no-pushdown", type=float, default=0.7)
    parser.add_argument(
        "--retry-borderline-frac",
        type=float,
        default=0.01,
        help="If failing ratio is within this fractional margin of threshold, auto-retry with --retry-runs.",
    )
    parser.add_argument(
        "--retry-runs",
        type=int,
        default=7,
        help="Runs used for automatic borderline recheck.",
    )
    parser.add_argument("--json-out", type=Path, default=Path("bench_results/pipeline_table_sphere.json"))
    args = parser.parse_args()

    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    must_regen = args.regen or not args.table_path.exists()
    if not must_regen and _n_rows_in_table(args.table_path) != args.n_rows:
        must_regen = True
    if must_regen:
        _write_synth_table(args.table_path, n_rows=args.n_rows, seed=args.seed)

    where = "MAG > 0.25 AND MAG < 0.75 AND BAND > 0 AND BAND < 4 AND DEC > -30 AND DEC < 30"

    rows = _run_all_rows(args.table_path, where=where, nside=args.nside, runs=args.runs)
    ratios, failures = _compute_ratios_and_failures(
        rows,
        min_read_ratio_filtered_vs_torch=args.min_read_ratio_filtered_vs_torch,
        min_pipeline_ratio_filtered_vs_no_pushdown=args.min_pipeline_ratio_filtered_vs_no_pushdown,
        min_pipeline_ratio_torch_vs_no_pushdown=args.min_pipeline_ratio_torch_vs_no_pushdown,
    )

    # Borderline auto-recheck for noisy timing misses.
    if failures and args.retry_runs > args.runs and args.retry_borderline_frac > 0.0:
        thresholds = {
            "read_filtered_vs_torch": args.min_read_ratio_filtered_vs_torch,
            "pipeline_filtered_vs_no_pushdown": args.min_pipeline_ratio_filtered_vs_no_pushdown,
            "pipeline_torch_vs_no_pushdown": args.min_pipeline_ratio_torch_vs_no_pushdown,
        }
        borderline = True
        for key, thr in thresholds.items():
            got = ratios[key]
            if got >= thr:
                continue
            if (thr - got) > (args.retry_borderline_frac * thr):
                borderline = False
                break
        if borderline:
            print(
                "\nBorderline gate miss detected; re-running with "
                f"{args.retry_runs} runs for stability check..."
            )
            rows = _run_all_rows(args.table_path, where=where, nside=args.nside, runs=args.retry_runs)
            ratios, failures = _compute_ratios_and_failures(
                rows,
                min_read_ratio_filtered_vs_torch=args.min_read_ratio_filtered_vs_torch,
                min_pipeline_ratio_filtered_vs_no_pushdown=args.min_pipeline_ratio_filtered_vs_no_pushdown,
                min_pipeline_ratio_torch_vs_no_pushdown=args.min_pipeline_ratio_torch_vs_no_pushdown,
            )

    print("operation              backend    selected_rows   mrows/s   time(s)   checksum")
    for r in rows:
        print(
            f"{r.operation:21s} {r.backend:9s} {r.n_rows_selected:13d}"
            f" {r.mrows_s:8.3f} {r.time_s:8.4f} {r.checksum:10d}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "table_path": str(args.table_path),
                "n_rows": _n_rows_in_table(args.table_path),
                "where": where,
                "rows": [asdict(r) for r in rows],
                "ratios": ratios,
                "thresholds": {
                    "min_read_ratio_filtered_vs_torch": args.min_read_ratio_filtered_vs_torch,
                    "min_pipeline_ratio_filtered_vs_no_pushdown": args.min_pipeline_ratio_filtered_vs_no_pushdown,
                    "min_pipeline_ratio_torch_vs_no_pushdown": args.min_pipeline_ratio_torch_vs_no_pushdown,
                    "retry_borderline_frac": args.retry_borderline_frac,
                    "retry_runs": args.retry_runs,
                },
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.json_out}")
    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
