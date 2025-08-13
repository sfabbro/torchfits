"""Micro-benchmark for table column selection bulk read path.

Now compares torchfits native read vs astropy->numpy->torch and fitsio->numpy->torch
and prints a compact table. Focuses on multiple scalar numeric columns.

Run:
    python benchmarks/table_bulk_read_micro.py
"""
from __future__ import annotations
import os, time, statistics, tempfile, pathlib, sys
import numpy as np
import torch
from torchfits import write_table, read

# Local import for shared utils when running as a script
sys.path.append(os.path.dirname(__file__))
from bench_utils import format_table, try_import  # type: ignore
from bench_utils import numpy_to_torch

N_ROWS = 200_000
N_COLS = 8
REPEATS = 5
COLUMNS = [f"C{i}" for i in range(N_COLS)]


def make_table(path: str):
    data = {c: torch.randint(0, 1000, (N_ROWS,), dtype=torch.int32) for c in COLUMNS}
    write_table(path, data, header={"EXTNAME": "TAB"}, overwrite=True)


def time_call(fn, *a, **k):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    fn(*a, **k)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) * 1000


def run():
    tmp = pathlib.Path(tempfile.gettempdir()) / "bulk_read_bench.fits"
    make_table(str(tmp))
    cols = COLUMNS[:4]
    # Warmup torchfits
    read(str(tmp), hdu=1, columns=cols, format="tensor")

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    times = []
    for _ in range(REPEATS):
        ms = time_call(read, str(tmp), hdu=1, columns=cols, format="tensor")
        times.append(ms)
    rows.append(["torchfits", "read(columns)->tensor", f"{statistics.mean(times):.2f}", f"{statistics.pstdev(times):.2f}", "native tensor dict"])

    if astropy is not None:
        vals = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            with astropy.open(str(tmp)) as hdul:  # type: ignore[attr-defined]
                t = hdul[1].data
                _ = {c: numpy_to_torch(t[c]) for c in cols}
            vals.append((time.perf_counter() - t0) * 1000.0)
        rows.append(["astropy", "numpy->torch (cols)", f"{statistics.mean(vals):.2f}", f"{statistics.pstdev(vals):.2f}", "np->torch"])
    else:
        rows.append(["astropy", "numpy->torch (cols)", "n/a", "", "missing module"])

    if fitsio is not None:
        vals = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            with fitsio.FITS(str(tmp)) as f:  # type: ignore[attr-defined]
                T = f[1]
                _ = {c: numpy_to_torch(T[c][:]) for c in cols}
            vals.append((time.perf_counter() - t0) * 1000.0)
        rows.append(["fitsio", "numpy->torch (cols)", f"{statistics.mean(vals):.2f}", f"{statistics.pstdev(vals):.2f}", "np->torch"])
    else:
        rows.append(["fitsio", "numpy->torch (cols)", "n/a", "", "missing module"])

    print("\n== Table bulk column read (", N_ROWS, " rows, ", len(cols), " cols) ==", sep="")
    print(format_table(rows, headers=headers))

if __name__ == "__main__":
    run()
