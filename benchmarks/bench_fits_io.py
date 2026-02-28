#!/usr/bin/env python3
"""Authoritative FITS I/O benchmark domain runner (images + MEFs + cutouts + headers)."""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitsio
import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits

from bench_contract import RESULT_COLUMNS, annotate_rankings, write_csv
from bench_legacy_all import ExhaustiveBenchmarkSuite


SMART_METHODS = [
    ("torchfits", "torchfits", "torchfits"),
    ("astropy_torch", "astropy", "astropy_torch"),
    ("fitsio_torch", "fitsio", "fitsio_torch"),
]

SPECIALIZED_METHODS = [
    ("torchfits_specialized", "torchfits", "torchfits_specialized"),
    ("astropy", "astropy", "astropy"),
    ("fitsio", "fitsio", "fitsio"),
]


def _hdu_for_file_type(file_type: str) -> int:
    if file_type in {"compressed", "mef", "multi_mef"}:
        return 1
    return 0


def _time_median(fn, *, runs: int, warmup: int) -> tuple[float | None, str | None]:
    for _ in range(max(0, warmup)):
        try:
            _ = fn()
        except Exception as exc:
            return None, str(exc)

    times: list[float] = []
    for _ in range(max(1, runs)):
        gc.collect()
        t0 = time.perf_counter()
        try:
            _ = fn()
        except Exception as exc:
            return None, str(exc)
        times.append(time.perf_counter() - t0)
    if not times:
        return None, "no_samples"
    return float(np.median(times)), None


def _strict_patch_astropy(suite: ExhaustiveBenchmarkSuite) -> None:
    fallback_paths: set[str] = set()

    @contextmanager
    def _astropy_open_strict(filepath: Path, memmap: bool):
        hdul = None
        try:
            hdul = astropy_fits.open(filepath, memmap=bool(memmap))
        except Exception:
            if bool(memmap):
                # Astropy cannot memmap some scaled/byte-packed payloads; read via
                # non-mmap fallback but mark these rows non-comparable later.
                hdul = astropy_fits.open(filepath, memmap=False)
                fallback_paths.add(str(filepath))
            else:
                raise
        try:
            yield hdul
        finally:
            if hdul is not None:
                hdul.close()

    def _astropy_read_strict(filepath: Path, hdu_num: int):
        try:
            with _astropy_open_strict(filepath, suite.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        return suite._table_to_numpy_dict(hdu.data)
                    return suite._ensure_native_endian_numpy(np.array(hdu.data, copy=True))
                return None
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(filepath))
                with astropy_fits.open(filepath, memmap=False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            return suite._table_to_numpy_dict(hdu.data)
                        return suite._ensure_native_endian_numpy(np.array(hdu.data, copy=True))
                    return None
            raise

    def _astropy_to_torch_strict(filepath: Path, hdu_num: int):
        try:
            with _astropy_open_strict(filepath, suite.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        out = {}
                        data = hdu.data
                        for col in data.names:
                            col_data = np.ascontiguousarray(np.asarray(data[col]))
                            if col_data.dtype.byteorder not in ("=", "|"):
                                col_data = col_data.astype(col_data.dtype.newbyteorder("="))
                            if col_data.dtype.kind in {"S", "U"}:
                                if col_data.dtype.kind == "U":
                                    col_data = np.char.encode(col_data, "ascii")
                                col_data = np.ascontiguousarray(col_data).view("uint8").reshape(
                                    len(col_data), -1
                                )
                            elif col_data.dtype.kind == "b":
                                col_data = col_data.astype(bool)
                            out[col] = torch.from_numpy(col_data)
                        return out
                    arr = np.array(hdu.data, copy=True)
                    if arr.dtype.byteorder not in ("=", "|"):
                        arr = arr.astype(arr.dtype.newbyteorder("="))
                    return torch.from_numpy(arr)
                return None
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(filepath))
                with astropy_fits.open(filepath, memmap=False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            out = {}
                            data = hdu.data
                            for col in data.names:
                                col_data = np.ascontiguousarray(np.asarray(data[col]))
                                if col_data.dtype.byteorder not in ("=", "|"):
                                    col_data = col_data.astype(col_data.dtype.newbyteorder("="))
                                if col_data.dtype.kind in {"S", "U"}:
                                    if col_data.dtype.kind == "U":
                                        col_data = np.char.encode(col_data, "ascii")
                                    col_data = np.ascontiguousarray(col_data).view(
                                        "uint8"
                                    ).reshape(len(col_data), -1)
                                elif col_data.dtype.kind == "b":
                                    col_data = col_data.astype(bool)
                                out[col] = torch.from_numpy(col_data)
                            return out
                        arr = np.array(hdu.data, copy=True)
                        if arr.dtype.byteorder not in ("=", "|"):
                            arr = arr.astype(arr.dtype.newbyteorder("="))
                        return torch.from_numpy(arr)
                    return None
            raise

    def _astropy_cutout_strict(path, hdu, x1, y1, x2, y2):
        try:
            with _astropy_open_strict(path, suite.use_mmap) as hdul:
                return hdul[hdu].section[y1:y2, x1:x2]
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(path))
                with astropy_fits.open(path, memmap=False) as hdul:
                    return hdul[hdu].section[y1:y2, x1:x2]
            raise

    suite._astropy_open = _astropy_open_strict  # type: ignore[attr-defined]
    suite._astropy_read = _astropy_read_strict  # type: ignore[attr-defined]
    suite._astropy_to_torch = _astropy_to_torch_strict  # type: ignore[attr-defined]
    suite._astropy_cutout = _astropy_cutout_strict  # type: ignore[attr-defined]
    suite._astropy_memmap_fallback_paths = fallback_paths  # type: ignore[attr-defined]


def _normalize_legacy_rows(
    *,
    run_id: str,
    raw_rows: list[dict[str, Any]],
    mmap_target: str,
    files: dict[str, Path],
    astropy_fallback_paths: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        domain_file_type = str(raw.get("file_type", ""))
        if domain_file_type in {"table", "wcs", "sphere"}:
            continue

        case_id = f"{raw.get('filename')}::{raw.get('operation', 'read_full')}"
        case_label = f"{raw.get('filename')} [{raw.get('operation', 'read_full')}]"
        metadata = {
            "file_type": raw.get("file_type"),
            "data_type": raw.get("data_type"),
            "dimensions": raw.get("dimensions"),
            "compression": raw.get("compression"),
        }

        for family, methods in (("smart", SMART_METHODS), ("specialized", SPECIALIZED_METHODS)):
            for method_key, library, method_label in methods:
                t_col = f"{method_key}_median"
                tp_col = f"{method_key}_mb_s"
                t_val = raw.get(t_col)
                tp_val = raw.get(tp_col)

                status = "OK" if t_val is not None else "FAILED"
                comparable = status == "OK"
                skip_reason = ""

                if library == "fitsio":
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = "strict_mmap_fairness: comparator mmap mode is not controllable"
                    t_val = None
                    tp_val = None
                elif library == "astropy" and status != "OK":
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = "strict_mmap_fairness: astropy mmap mode unavailable for this case"
                    t_val = None
                    tp_val = None
                elif (
                    library == "astropy"
                    and mmap_target == "on"
                    and str(files.get(str(raw.get("filename")), "")) in astropy_fallback_paths
                ):
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = (
                        "strict_mmap_fairness: astropy required memmap=False fallback "
                        "for this payload"
                    )
                    t_val = None
                    tp_val = None

                row = {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "fits_io",
                    "case_id": case_id,
                    "case_label": case_label,
                    "operation": raw.get("operation", "read_full"),
                    "family": family,
                    "library": library,
                    "method": method_label,
                    "mode": "smart" if family == "smart" else "specialized",
                    "status": status,
                    "skip_reason": skip_reason,
                    "comparable": comparable,
                    "mmap_target": mmap_target,
                    "time_s": t_val,
                    "throughput": tp_val,
                    "unit": "MB/s",
                    "size_mb": raw.get("size_mb"),
                    "n_points": "",
                    "metadata": metadata,
                }
                rows.append(row)

    return rows


def _benchmark_headers(
    *,
    run_id: str,
    files: dict[str, Path],
    suite: ExhaustiveBenchmarkSuite,
    mmap_target: str,
    runs: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_memmap = mmap_target == "on"

    for name, path in sorted(files.items()):
        file_type = suite._get_file_type(name)
        if file_type in {"table", "wcs", "sphere"}:
            continue
        hdu = _hdu_for_file_type(file_type)
        case_id = f"{name}::header_read"
        case_label = f"{name} [header_read]"

        def _tf_header():
            return torchfits.get_header(str(path), hdu=hdu)

        def _astropy_header():
            with astropy_fits.open(path, memmap=target_memmap) as hdul:
                return dict(hdul[hdu].header)

        def _fitsio_header():
            return fitsio.read_header(str(path), ext=hdu)

        methods = [
            ("torchfits", "torchfits_specialized", _tf_header),
            ("astropy", "astropy", _astropy_header),
            ("fitsio", "fitsio", _fitsio_header),
        ]

        for library, method_label, fn in methods:
            t_val, err = _time_median(fn, runs=runs, warmup=warmup)
            status = "OK" if t_val is not None else "FAILED"
            comparable = status == "OK"
            skip_reason = ""
            if library == "fitsio":
                status = "SKIPPED"
                comparable = False
                skip_reason = "strict_mmap_fairness: comparator mmap mode is not controllable"
                t_val = None

            if library == "astropy" and err and target_memmap:
                status = "SKIPPED"
                comparable = False
                skip_reason = f"strict_mmap_fairness: astropy memmap={target_memmap} unavailable ({err})"
                t_val = None

            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "fits_io",
                    "case_id": case_id,
                    "case_label": case_label,
                    "operation": "header_read",
                    "family": "specialized",
                    "library": library,
                    "method": method_label,
                    "mode": "specialized",
                    "status": status,
                    "skip_reason": skip_reason,
                    "comparable": comparable,
                    "mmap_target": mmap_target,
                    "time_s": t_val,
                    "throughput": "",
                    "unit": "ops/s",
                    "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                    "n_points": "",
                    "metadata": {
                        "file_type": file_type,
                        "data_type": "header",
                        "dimensions": "n/a",
                        "compression": suite._get_compression_type(name),
                    },
                }
            )

    return rows


def run_fits_domain(
    *,
    run_id: str,
    output_dir: Path,
    profile: str = "user",
    use_mmap: bool = True,
    case_filter: str = "",
    header_runs: int = 7,
    header_warmup: int = 2,
    keep_temp: bool = False,
) -> list[dict[str, Any]]:
    mmap_target = "on" if use_mmap else "off"
    raw_dir = output_dir / "_raw" / "fits"
    raw_dir.mkdir(parents=True, exist_ok=True)

    suite = ExhaustiveBenchmarkSuite(
        output_dir=raw_dir,
        use_mmap=use_mmap,
        include_tables=False,
        include_wcs=False,
        include_sphere=False,
        profile=profile,
    )
    _strict_patch_astropy(suite)

    try:
        files = suite.create_test_files()
        files = {
            k: v
            for k, v in files.items()
            if (not k.startswith("table_")) and ("wcs" not in k.lower())
        }
        if case_filter:
            rx = re.compile(case_filter)
            files = {k: v for k, v in files.items() if rx.search(k)}

        print(f"[fits] cases={len(files)} mmap={mmap_target} profile={profile}", flush=True)

        raw_rows = suite.run_exhaustive_benchmarks(files)
        astropy_fallback_paths = set(
            getattr(suite, "_astropy_memmap_fallback_paths", set())  # type: ignore[attr-defined]
        )
        rows = _normalize_legacy_rows(
            run_id=run_id,
            raw_rows=raw_rows,
            mmap_target=mmap_target,
            files=files,
            astropy_fallback_paths=astropy_fallback_paths,
        )
        rows.extend(
            _benchmark_headers(
                run_id=run_id,
                files=files,
                suite=suite,
                mmap_target=mmap_target,
                runs=header_runs,
                warmup=header_warmup,
            )
        )

        annotate_rankings(rows)
        return rows
    finally:
        if keep_temp:
            print(f"[fits] temp files kept: {suite.temp_dir}", flush=True)
        else:
            suite.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks_results"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--header-runs", type=int, default=7)
    parser.add_argument("--header-warmup", type=int, default=2)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_mmap = not args.no_mmap
    if args.mmap:
        use_mmap = True

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    rows = run_fits_domain(
        run_id=run_id,
        output_dir=run_dir,
        profile=args.profile,
        use_mmap=use_mmap,
        case_filter=args.filter,
        header_runs=args.header_runs,
        header_warmup=args.header_warmup,
        keep_temp=args.keep_temp,
    )

    out_csv = run_dir / "fits_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"[fits] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
