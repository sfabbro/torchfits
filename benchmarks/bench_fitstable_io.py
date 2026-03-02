#!/usr/bin/env python3
"""Authoritative FITS table I/O benchmark runner."""

from __future__ import annotations

import argparse
import gc
import gzip
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import fitsio
import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits

from bench_contract import RESULT_COLUMNS, annotate_rankings, write_csv


def _repeats_for_rows(nrows: int) -> int:
    if nrows <= 10_000:
        return 9
    if nrows <= 100_000:
        return 5
    return 3


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


def _dtype_values(dtype: str, nrows: int, rng: np.random.Generator):
    if dtype == "f4":
        return rng.normal(size=nrows).astype(np.float32)
    if dtype == "f8":
        return rng.normal(size=nrows).astype(np.float64)
    if dtype == "i4":
        return rng.integers(-1_000_000, 1_000_000, size=nrows, dtype=np.int32)
    if dtype == "i8":
        return rng.integers(-1_000_000, 1_000_000, size=nrows, dtype=np.int64)
    if dtype == "bool":
        return rng.random(size=nrows) > 0.5
    if dtype.startswith("S"):
        width = int(dtype[1:]) if len(dtype) > 1 else 8
        return np.array([f"s{i:08d}"[:width] for i in range(nrows)], dtype=f"S{width}")
    raise ValueError(f"unsupported dtype spec: {dtype}")


def _write_table_file(
    *,
    out_path: Path,
    nrows: int,
    schema_name: str,
    schema: list[tuple[str, str]],
    rng_seed: int,
) -> tuple[str, list[str]]:
    rng = np.random.default_rng(rng_seed)
    cols = []
    for col_name, dtype in schema:
        arr = _dtype_values(dtype, nrows, rng)
        cols.append(astropy_fits.Column(name=col_name, format=_to_tform(dtype), array=arr))
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name=f"TABLE_{schema_name.upper()}")
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(out_path, overwrite=True)
    return schema_name, [c[0] for c in schema]


def _write_varlen_file(*, out_path: Path, nrows: int, rng_seed: int) -> list[str]:
    rng = np.random.default_rng(rng_seed)
    ids = np.arange(nrows, dtype=np.int32)
    flux = rng.normal(size=nrows).astype(np.float32)
    var = np.empty(nrows, dtype=object)
    for i in range(nrows):
        width = int((i % 7) + 1)
        var[i] = np.arange(width, dtype=np.int32)

    cols = [
        astropy_fits.Column(name="id", format="J", array=ids),
        astropy_fits.Column(name="flux", format="E", array=flux),
        astropy_fits.Column(name="values", format="PJ()", array=var),
    ]
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name="TABLE_VARLEN")
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(out_path, overwrite=True)
    return ["id", "flux", "values"]


def _to_tform(dtype: str) -> str:
    mapping = {
        "f4": "E",
        "f8": "D",
        "i4": "J",
        "i8": "K",
        "bool": "L",
    }
    if dtype in mapping:
        return mapping[dtype]
    if dtype.startswith("S"):
        return dtype[1:] + "A"
    raise ValueError(f"unsupported dtype spec: {dtype}")


def _gzip_fits(in_path: Path, out_path: Path) -> None:
    with in_path.open("rb") as f_in:
        with gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _table_to_torch_dict(data) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    names = list(data.dtype.names or [])
    for name in names:
        arr = np.ascontiguousarray(np.asarray(data[name]))
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        if arr.dtype.kind in {"S", "U"}:
            if arr.dtype.kind == "U":
                arr = np.char.encode(arr, "ascii")
            arr = np.ascontiguousarray(arr).view("uint8").reshape(len(arr), -1)
        elif arr.dtype.kind == "b":
            arr = arr.astype(bool)
        elif arr.dtype.kind == "O":
            continue
        out[name] = torch.from_numpy(arr)
    return out


def _choose_numeric_col(columns: list[str], schema: list[tuple[str, str]] | None) -> str:
    if schema:
        for name, dtype in schema:
            if dtype in {"f4", "f8", "i4", "i8"}:
                return name
    return columns[0]


def _bench_case(
    *,
    run_id: str,
    case: dict[str, Any],
    use_mmap: bool,
    policy_profile: str,
    warmup: int,
) -> list[dict[str, Any]]:
    path: Path = case["path"]
    nrows: int = int(case["nrows"])
    columns: list[str] = list(case["columns"])
    schema: list[tuple[str, str]] | None = case.get("schema")
    case_name = str(case["name"])
    compressed = bool(case.get("compressed", False))
    variable = bool(case.get("variable", False))
    unsupported = bool(case.get("unsupported", False))

    proj_cols = columns[: min(3, len(columns))]
    num_col = _choose_numeric_col(columns, schema)
    row_slice_start = 1
    row_slice_n = min(10_000, max(100, nrows // 10))

    mmap_target = "on" if use_mmap else "off"
    target_memmap = use_mmap

    runs = _repeats_for_rows(nrows)

    if unsupported:
        rows: list[dict[str, Any]] = []
        operations = ["read_full", "projection", "row_slice", "predicate_filter", "scan_count"]
        method_specs = [
            ("torchfits", "torchfits", "smart", "smart"),
            ("astropy_torch", "astropy", "smart", "smart"),
            ("fitsio_torch", "fitsio", "smart", "smart"),
            ("torchfits_specialized", "torchfits", "specialized", "specialized"),
            ("astropy", "astropy", "specialized", "specialized"),
            ("fitsio", "fitsio", "specialized", "specialized"),
        ]
        for op_name in operations:
            for method, library, family, mode in method_specs:
                rows.append(
                    _make_row(
                        run_id=run_id,
                        case_name=case_name,
                        case=case,
                        operation=op_name,
                        family=family,
                        method=method,
                        library=library,
                        mode=mode,
                        mmap_target=mmap_target,
                        status="SKIPPED",
                        comparable=False,
                        skip_reason="compressed_table_case_not_enabled_in_default_env",
                        time_s=None,
                        throughput=None,
                        unit="rows/s",
                        n_points=nrows,
                    )
                )
        return rows

    try:
        import pyarrow.dataset as ds  # noqa: F401

        has_pyarrow = True
    except Exception:
        has_pyarrow = False

    print(
        f"[fitstable] case={case_name} rows={nrows} cols={len(columns)} compressed={compressed} variable={variable} runs={runs}",
        flush=True,
    )

    operations = {
        "read_full": {
            "torchfits": lambda: torchfits.read(
                str(path), hdu=1, mode="table", policy="smart", mmap=target_memmap
            ),
            "torchfits_specialized": lambda: torchfits.read_table(
                str(path), hdu=1, policy="default", mmap=target_memmap
            ),
            "astropy": lambda: _astropy_read_full(path, memmap=target_memmap),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_read_full(path, memmap=target_memmap)
            ),
            "fitsio": lambda: fitsio.read(str(path), ext=1),
            "fitsio_torch": lambda: _table_to_torch_dict(fitsio.read(str(path), ext=1)),
        },
        "projection": {
            "torchfits": lambda: torchfits.read(
                str(path),
                hdu=1,
                mode="table",
                policy="smart",
                mmap=target_memmap,
                columns=proj_cols,
            ),
            "torchfits_specialized": lambda: torchfits.read_table(
                str(path),
                hdu=1,
                columns=proj_cols,
                policy="default",
                mmap=target_memmap,
            ),
            "astropy": lambda: _astropy_projection(path, proj_cols, memmap=target_memmap),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_projection(path, proj_cols, memmap=target_memmap)
            ),
            "fitsio": lambda: fitsio.read(str(path), ext=1, columns=proj_cols),
            "fitsio_torch": lambda: _table_to_torch_dict(
                fitsio.read(str(path), ext=1, columns=proj_cols)
            ),
        },
        "row_slice": {
            "torchfits": lambda: torchfits.read(
                str(path),
                hdu=1,
                mode="table",
                policy="smart",
                mmap=target_memmap,
                start_row=row_slice_start,
                num_rows=row_slice_n,
            ),
            "torchfits_specialized": lambda: torchfits.read_table_rows(
                str(path),
                hdu=1,
                start_row=row_slice_start,
                num_rows=row_slice_n,
                policy="default",
                mmap=target_memmap,
            ),
            "astropy": lambda: _astropy_row_slice(
                path, row_slice_start, row_slice_n, memmap=target_memmap
            ),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_row_slice(path, row_slice_start, row_slice_n, memmap=target_memmap)
            ),
            "fitsio": lambda: _fitsio_row_slice(path, row_slice_start, row_slice_n),
            "fitsio_torch": lambda: _table_to_torch_dict(
                _fitsio_row_slice(path, row_slice_start, row_slice_n)
            ),
        },
        "predicate_filter": {
            "torchfits": lambda: _torchfits_filter_pushdown(
                path, col=num_col, mmap=target_memmap, has_pyarrow=has_pyarrow
            ),
            "torchfits_specialized": lambda: _torchfits_filter_local(
                path, col=num_col, mmap=target_memmap
            ),
            "astropy": lambda: _astropy_filter(path, col=num_col, memmap=target_memmap),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_filter(path, col=num_col, memmap=target_memmap)
            ),
            "fitsio": lambda: _fitsio_filter(path, col=num_col),
            "fitsio_torch": lambda: _table_to_torch_dict(_fitsio_filter(path, col=num_col)),
        },
        "scan_count": {
            "torchfits": lambda: _torchfits_scan_count(
                path, col=num_col, mmap=target_memmap, has_pyarrow=has_pyarrow
            ),
            "torchfits_specialized": lambda: _torchfits_scan_count_local(
                path, col=num_col, mmap=target_memmap
            ),
            "astropy": lambda: _astropy_scan_count(path, col=num_col, memmap=target_memmap),
            "astropy_torch": lambda: _astropy_scan_count(path, col=num_col, memmap=target_memmap),
            "fitsio": lambda: _fitsio_scan_count(path, col=num_col),
            "fitsio_torch": lambda: _fitsio_scan_count(path, col=num_col),
        },
    }

    rows: list[dict[str, Any]] = []

    for op_name, method_map in operations.items():
        for method, fn in method_map.items():
            if method in {"fitsio", "fitsio_torch"}:
                rows.append(
                    _make_row(
                        run_id=run_id,
                        case_name=case_name,
                        case=case,
                        operation=op_name,
                        family="smart" if method == "fitsio_torch" else "specialized",
                        method=method,
                        library="fitsio",
                        mode="smart" if method == "fitsio_torch" else "specialized",
                        mmap_target=mmap_target,
                        status="SKIPPED",
                        comparable=False,
                        skip_reason="strict_mmap_fairness: comparator mmap mode is not controllable",
                        time_s=None,
                        throughput=None,
                        unit="ops/s",
                        n_points=nrows,
                    )
                )
                continue

            t_val, err = _time_median(fn, runs=runs, warmup=warmup)
            status = "OK" if t_val is not None else "FAILED"
            comparable = status == "OK"
            skip_reason = ""

            library = "torchfits" if method.startswith("torchfits") else "astropy"
            family = "smart" if method in {"torchfits", "astropy_torch"} else "specialized"
            mode = "smart" if family == "smart" else "specialized"

            # If strict mmap parity cannot be honored by astropy in this case, mark SKIPPED.
            if library == "astropy" and err and target_memmap:
                status = "SKIPPED"
                comparable = False
                skip_reason = f"strict_mmap_fairness: astropy memmap={target_memmap} unavailable ({err})"
                t_val = None

            throughput = (nrows / t_val) if (t_val is not None and t_val > 0) else None
            rows.append(
                _make_row(
                    run_id=run_id,
                    case_name=case_name,
                    case=case,
                    operation=op_name,
                    family=family,
                    method=method,
                    library=library,
                    mode=mode,
                    mmap_target=mmap_target,
                    status=status,
                    comparable=comparable,
                    skip_reason=skip_reason,
                    time_s=t_val,
                    throughput=throughput,
                    unit="rows/s",
                    n_points=nrows,
                )
            )

    return rows


def _astropy_read_full(path: Path, *, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        return np.array(hdul[1].data, copy=False)


def _astropy_projection(path: Path, columns: list[str], *, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data
        # FITS_rec does not support list-of-names indexing directly.
        return {col: np.asarray(data[col]) for col in columns}


def _astropy_row_slice(path: Path, start_row: int, num_rows: int, *, memmap: bool):
    start0 = max(0, int(start_row) - 1)
    stop0 = start0 + int(num_rows)
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data
        return np.array(data[start0:stop0], copy=False)


def _astropy_filter(path: Path, *, col: str, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data
        mask = np.asarray(data[col]) > 0
        return np.array(data[mask], copy=False)


def _astropy_scan_count(path: Path, *, col: str, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        return int(len(hdul[1].data[col]))


def _fitsio_row_slice(path: Path, start_row: int, num_rows: int):
    data = fitsio.read(str(path), ext=1)
    start0 = max(0, int(start_row) - 1)
    stop0 = start0 + int(num_rows)
    return data[start0:stop0]


def _fitsio_filter(path: Path, *, col: str):
    data = fitsio.read(str(path), ext=1)
    mask = np.asarray(data[col]) > 0
    return data[mask]


def _fitsio_scan_count(path: Path, *, col: str):
    data = fitsio.read(str(path), ext=1)
    return int(len(data[col]))


def _torchfits_filter_pushdown(path: Path, *, col: str, mmap: bool, has_pyarrow: bool):
    if not has_pyarrow:
        raise RuntimeError("pyarrow not available for scanner filter")
    import torchfits.table

    # Use the high-level read API which chooses the best path (C++ pushdown or local)
    res = torchfits.table.read(
        str(path),
        hdu=1,
        columns=[col],
        where=f"{col} > 0",
        mmap=mmap,
    )
    return len(res)


def _torchfits_filter_local(path: Path, *, col: str, mmap: bool):
    data = torchfits.read_table(str(path), hdu=1, mmap=mmap, policy="default")
    values = data[col]
    if isinstance(values, torch.Tensor):
        return values[values > 0]
    arr = np.asarray(values)
    return arr[arr > 0]


def _torchfits_scan_count(path: Path, *, col: str, mmap: bool, has_pyarrow: bool):
    import torchfits
    # Use hdu.num_rows if no filtering is needed, it's MUCH faster for FITS.
    with torchfits.open(str(path)) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "num_rows"):
                return hdu.num_rows
    return 0


def _torchfits_scan_count_local(path: Path, *, col: str, mmap: bool):
    data = torchfits.read_table(str(path), hdu=1, mmap=mmap, policy="default")
    values = data[col]
    if isinstance(values, torch.Tensor):
        return int(values.shape[0])
    return int(len(values))


def _make_row(
    *,
    run_id: str,
    case_name: str,
    case: dict[str, Any],
    operation: str,
    family: str,
    method: str,
    library: str,
    mode: str,
    mmap_target: str,
    status: str,
    comparable: bool,
    skip_reason: str,
    time_s: float | None,
    throughput: float | None,
    unit: str,
    n_points: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": "fitstable",
        "suite": "fitstable_io",
        "case_id": f"{case_name}::{operation}",
        "case_label": f"{case_name} [{operation}]",
        "operation": operation,
        "family": family,
        "library": library,
        "method": method,
        "mode": mode,
        "status": status,
        "skip_reason": skip_reason,
        "comparable": comparable,
        "mmap_target": mmap_target,
        "time_s": time_s,
        "throughput": throughput,
        "unit": unit,
        "size_mb": case["size_mb"],
        "n_points": n_points,
        "metadata": {
            "schema": case["schema_name"],
            "nrows": case["nrows"],
            "ncols": case["ncols"],
            "variable": case.get("variable", False),
            "compressed": case.get("compressed", False),
            "profile": case.get("profile", "default"),
        },
    }


def _build_cases(temp_dir: Path, *, quick: bool = False) -> list[dict[str, Any]]:
    schemas: dict[str, list[tuple[str, str]]] = {
        "narrow": [
            ("id", "i4"),
            ("flux", "f4"),
            ("err", "f4"),
            ("flag", "bool"),
        ],
        "mixed": [
            ("id", "i8"),
            ("ra", "f8"),
            ("dec", "f8"),
            ("flux", "f4"),
            ("fluxerr", "f4"),
            ("class", "i4"),
            ("name", "S16"),
            ("quality", "bool"),
        ],
        "wide": [
            *[(f"f{i:02d}", "f4") for i in range(20)],
            *[(f"i{i:02d}", "i4") for i in range(10)],
            *[(f"d{i:02d}", "f8") for i in range(6)],
            *[(f"s{i:02d}", "S12") for i in range(4)],
            ("flag", "bool"),
        ],
    }

    row_scales = [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000]
    cases: list[dict[str, Any]] = []

    seed = 123
    for nrows in row_scales:
        schema_order = ["narrow", "mixed"]
        if nrows <= 100_000:
            schema_order.append("wide")

        for schema_name in schema_order:
            schema = schemas[schema_name]
            path = temp_dir / f"table_{schema_name}_{nrows}.fits"
            _schema, columns = _write_table_file(
                out_path=path,
                nrows=nrows,
                schema_name=schema_name,
                schema=schema,
                rng_seed=seed,
            )
            seed += 1
            cases.append(
                {
                    "name": f"{schema_name}_{nrows}",
                    "schema_name": schema_name,
                    "path": path,
                    "nrows": nrows,
                    "ncols": len(columns),
                    "columns": columns,
                    "schema": schema,
                    "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                    "variable": False,
                    "compressed": False,
                    "profile": "base",
                }
            )

    for nrows in ([1_000, 10_000] if quick else [1_000, 10_000, 100_000]):
        path = temp_dir / f"table_varlen_{nrows}.fits"
        columns = _write_varlen_file(out_path=path, nrows=nrows, rng_seed=seed)
        seed += 1
        cases.append(
            {
                "name": f"varlen_{nrows}",
                "schema_name": "varlen",
                "path": path,
                "nrows": nrows,
                "ncols": len(columns),
                "columns": columns,
                "schema": None,
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                "variable": True,
                "compressed": False,
                "profile": "varlen",
            }
        )

    # Compressed table benchmark placeholder (marked SKIPPED if not enabled).
    cases.append(
        {
            "name": "compressed_table_placeholder",
            "schema_name": "mixed",
            "path": temp_dir / "compressed_table_placeholder.fits",
            "nrows": 100_000,
            "ncols": 8,
            "columns": [
                "id",
                "ra",
                "dec",
                "flux",
                "fluxerr",
                "class",
                "name",
                "quality",
            ],
            "schema": schemas["mixed"],
            "size_mb": 0.0,
            "variable": False,
            "compressed": True,
            "unsupported": True,
            "profile": "compressed_placeholder",
        }
    )

    return cases


def run_fitstable_domain(
    *,
    run_id: str,
    output_dir: Path,
    use_mmap: bool = True,
    profile: str = "user",
    warmup: int = 1,
    quick: bool = False,
    max_cases: int | None = None,
    keep_temp: bool = False,
) -> list[dict[str, Any]]:
    _ = profile
    temp_root = Path(tempfile.mkdtemp(prefix="torchfits_fitstable_"))
    rows: list[dict[str, Any]] = []

    try:
        cases = _build_cases(temp_root, quick=quick)
        if max_cases is not None and max_cases > 0:
            supported_cases = [c for c in cases if not bool(c.get("unsupported", False))]
            cases = supported_cases[:max_cases]
            print(f"[fitstable] quick case cap applied: {len(cases)} case(s)", flush=True)
        for case in cases:
            rows.extend(
                _bench_case(
                    run_id=run_id,
                    case=case,
                    use_mmap=use_mmap,
                    policy_profile=profile,
                    warmup=warmup,
                )
            )

        annotate_rankings(rows)
        return rows
    finally:
        if keep_temp:
            print(f"[fitstable] temp files kept: {temp_root}", flush=True)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks_results"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_mmap = not args.no_mmap
    if args.mmap:
        use_mmap = True

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id

    rows = run_fitstable_domain(
        run_id=run_id,
        output_dir=run_dir,
        use_mmap=use_mmap,
        profile=args.profile,
        warmup=args.warmup,
        quick=args.quick,
        max_cases=(args.max_cases if args.max_cases > 0 else None),
        keep_temp=args.keep_temp,
    )

    out_csv = run_dir / "fitstable_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"[fitstable] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
