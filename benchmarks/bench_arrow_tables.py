#!/usr/bin/env python3
"""
Focused benchmark for Arrow-native FITS table workflows.

Compares:
- torchfits.table.read / scan
- fitsio.read (raw + decoded-string variants)
- astropy.table.Table.read
- Arrow -> pandas / polars conversion
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from astropy.table import Table
import pyarrow as pa
import torch

import fitsio
import torchfits


def _build_table(path: Path, rows: int) -> None:
    rng = np.random.default_rng(0)
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "MAG_G": rng.normal(20.0, 1.5, rows).astype(np.float32),
        "OBJID": rng.integers(0, 2**31 - 1, rows, dtype=np.int64),
        "FLAG": rng.integers(0, 2, rows, dtype=np.int16),
        "NAME": np.array([f"SRC_{i:08d}" for i in range(rows)], dtype="U16"),
    }
    Table(data).write(path, format="fits", overwrite=True)


def _time(fn, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


def _to_native_endian(arr: np.ndarray) -> np.ndarray:
    byteorder = arr.dtype.byteorder
    if byteorder in ("=", "|"):
        return arr
    if (byteorder == "<" and np.little_endian) or (
        byteorder == ">" and not np.little_endian
    ):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _fitsio_record_to_arrow(rec: np.ndarray):
    pydict: dict[str, object] = {}
    for name in rec.dtype.names or ():
        col = _to_native_endian(rec[name])
        if col.ndim <= 1:
            pydict[name] = pa.array(col)
        elif col.ndim == 2:
            flat = _to_native_endian(col.reshape(-1))
            values = pa.array(flat)
            pydict[name] = pa.FixedSizeListArray.from_arrays(values, int(col.shape[1]))
        else:
            pydict[name] = pa.array(col.tolist())
    return pa.table(pydict)


def _fitsio_record_to_torch(rec: np.ndarray):
    tensors: dict[str, object] = {}
    for name in rec.dtype.names or ():
        col = _to_native_endian(rec[name])
        if not isinstance(col, np.ndarray):
            continue
        if col.dtype.kind in {"U", "S"}:
            continue
        if col.dtype == np.object_:
            converted = []
            for item in col.tolist():
                if isinstance(item, np.ndarray):
                    arr = _to_native_endian(np.ascontiguousarray(item))
                    converted.append(torch.from_numpy(arr))
                else:
                    converted.append(item)
            tensors[name] = converted  # type: ignore[assignment]
            continue
        arr = np.ascontiguousarray(col)
        tensors[name] = torch.from_numpy(arr)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrow table benchmark")
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--output", type=str, default="bench_results/table_arrow_results.csv"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_table_arrow_"))
    file_path = data_dir / "table_arrow_bench.fits"
    _build_table(file_path, args.rows)
    print(f"Dataset: {file_path} ({args.rows} rows)")

    methods = []

    methods.append(
        (
            "torchfits_arrow_read_raw",
            lambda: torchfits.table.read(
                str(file_path),
                hdu=1,
                decode_bytes=False,
                apply_fits_nulls=False,
                batch_size=args.batch_size,
                backend="cpp_numpy",
            ),
        )
    )
    methods.append(
        (
            "torchfits_arrow_read_decoded",
            lambda: torchfits.table.read(
                str(file_path),
                hdu=1,
                decode_bytes=True,
                apply_fits_nulls=False,
                batch_size=args.batch_size,
                backend="cpp_numpy",
            ),
        )
    )
    methods.append(("fitsio_read_raw", lambda: fitsio.read(str(file_path), ext=1)))
    methods.append(
        (
            "fitsio_read_raw_arrow",
            lambda: _fitsio_record_to_arrow(fitsio.read(str(file_path), ext=1)),
        )
    )
    methods.append(
        (
            "fitsio_read_raw_torch",
            lambda: _fitsio_record_to_torch(fitsio.read(str(file_path), ext=1)),
        )
    )
    methods.append(
        (
            "fitsio_read_decoded",
            lambda: fitsio.read(str(file_path), ext=1)["NAME"].astype("U16"),
        )
    )
    methods.append(("astropy_table_read", lambda: Table.read(str(file_path), hdu=1)))

    proj_cols = ["RA", "MAG_G"]
    proj_start = args.rows // 5
    proj_stop = min(args.rows, proj_start + min(50_000, max(1, args.rows // 2)))
    proj_rows = np.arange(proj_start, proj_stop, dtype=np.int64)
    methods.append(
        (
            "torchfits_arrow_read_projected",
            lambda: torchfits.table.read(
                str(file_path),
                hdu=1,
                columns=proj_cols,
                row_slice=(int(proj_start), int(proj_stop)),
                decode_bytes=False,
                apply_fits_nulls=False,
                batch_size=args.batch_size,
                backend="cpp_numpy",
            ),
        )
    )
    methods.append(
        (
            "fitsio_read_projected",
            lambda: fitsio.read(
                str(file_path), ext=1, columns=proj_cols, rows=proj_rows
            ),
        )
    )
    methods.append(
        (
            "fitsio_read_projected_arrow",
            lambda: _fitsio_record_to_arrow(
                fitsio.read(str(file_path), ext=1, columns=proj_cols, rows=proj_rows)
            ),
        )
    )

    def _torchfits_arrow_scan_sum():
        total = 0.0
        for batch in torchfits.table.scan(
            str(file_path),
            hdu=1,
            decode_bytes=False,
            apply_fits_nulls=False,
            batch_size=args.batch_size,
        ):
            total += float(np.asarray(batch.column("RA")).sum())
        return total

    methods.append(("torchfits_arrow_scan_sum", _torchfits_arrow_scan_sum))

    def _torchfits_arrow_scan_sum_cpp_numpy():
        total = 0.0
        for batch in torchfits.table.scan(
            str(file_path),
            hdu=1,
            decode_bytes=False,
            apply_fits_nulls=False,
            batch_size=args.batch_size,
            backend="cpp_numpy",
        ):
            total += float(np.asarray(batch.column("RA")).sum())
        return total

    methods.append(
        ("torchfits_arrow_scan_sum_cpp_numpy", _torchfits_arrow_scan_sum_cpp_numpy)
    )

    def _torchfits_scan_torch_sum_cpu():
        total = 0.0
        for chunk in torchfits.table.scan_torch(
            str(file_path),
            hdu=1,
            batch_size=args.batch_size,
            device="cpu",
        ):
            total += float(chunk["RA"].sum().item())
        return total

    methods.append(("torchfits_scan_torch_sum_cpu", _torchfits_scan_torch_sum_cpu))

    results: list[dict[str, str | float | int]] = []
    arrow_tbl = None
    for name, fn in methods:
        m, s = _time(fn, args.warmup, args.iterations)
        print(f"{name:24s}: {m:.6f}s ± {s:.6f}s")
        results.append({"method": name, "mean_s": m, "std_s": s, "rows": args.rows})
        if name == "torchfits_arrow_read_raw":
            arrow_tbl = fn()

    if arrow_tbl is not None:
        try:
            import pandas as pd  # noqa: F401

            m, s = _time(
                lambda: torchfits.table.to_pandas(arrow_tbl), 1, args.iterations
            )
            print(f"{'arrow_to_pandas':24s}: {m:.6f}s ± {s:.6f}s")
            results.append(
                {
                    "method": "arrow_to_pandas",
                    "mean_s": m,
                    "std_s": s,
                    "rows": args.rows,
                }
            )
        except ImportError:
            pass

        try:
            import polars as pl  # noqa: F401

            m, s = _time(
                lambda: torchfits.table.to_polars(arrow_tbl), 1, args.iterations
            )
            print(f"{'arrow_to_polars':24s}: {m:.6f}s ± {s:.6f}s")
            results.append(
                {
                    "method": "arrow_to_polars",
                    "mean_s": m,
                    "std_s": s,
                    "rows": args.rows,
                }
            )
        except ImportError:
            pass

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_s", "std_s", "rows"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
