import argparse
import concurrent.futures
import os
import random
import time
from typing import Any, Callable, Dict, List

import fitsio
import numpy as np
import torch

import torchfits
import torchfits.cpp as cpp


def _time_callable(func: Callable[[], Any], runs: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        _ = func()

    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = func()
        samples.append(time.perf_counter() - t0)

    arr = np.array(samples, dtype=np.float64)
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    median = float(np.median(arr))
    return {
        "median_s": median,
        "p10_s": p10,
        "p90_s": p90,
        "spread_s": p90 - p10,
    }


def bench_parallel(args: argparse.Namespace) -> None:
    n_files = args.n_files
    filename_base = args.filename_base
    shape = (1024, 1024)
    dtype = np.float32

    print("[diagnostic] Parallel I/O microbenchmark (non-gating)")
    print("[diagnostic] This script is for local diagnostics, not release gating.")

    files: List[str] = []
    for i in range(n_files):
        fname = f"{filename_base}_{i}.fits"
        files.append(fname)
        if args.recreate and os.path.exists(fname):
            os.remove(fname)
        if not os.path.exists(fname):
            data = np.random.randn(*shape).astype(dtype)
            fitsio.write(fname, data, clobber=True)

    print(f"Benchmarking parallel read of {n_files} files...")

    def read_serial_torchfits() -> List[torch.Tensor]:
        return [torchfits.read(f, hdu=0, policy="smart") for f in files]

    def read_parallel_torchfits() -> List[torch.Tensor]:
        return cpp.read_images_batch(files, 0)

    def read_parallel_python_threads() -> List[torch.Tensor]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(torchfits.read, f, 0, policy="smart") for f in files
            ]
            return [f.result() for f in futures]

    serial_stats = _time_callable(
        read_serial_torchfits, runs=args.runs, warmup=args.warmup
    )
    cpp_stats = _time_callable(
        read_parallel_torchfits, runs=args.runs, warmup=args.warmup
    )
    py_stats = _time_callable(
        read_parallel_python_threads, runs=args.runs, warmup=args.warmup
    )

    print(
        "Serial (Python): "
        f"median={serial_stats['median_s'] * 1000:.2f} ms "
        f"(p10={serial_stats['p10_s'] * 1000:.2f}, p90={serial_stats['p90_s'] * 1000:.2f})"
    )
    print(
        "Parallel (Python Threads): "
        f"median={py_stats['median_s'] * 1000:.2f} ms "
        f"(p10={py_stats['p10_s'] * 1000:.2f}, p90={py_stats['p90_s'] * 1000:.2f})"
    )
    print(
        "Parallel (C++ Batch): "
        f"median={cpp_stats['median_s'] * 1000:.2f} ms "
        f"(p10={cpp_stats['p10_s'] * 1000:.2f}, p90={cpp_stats['p90_s'] * 1000:.2f})"
    )

    speedup_cpp = serial_stats["median_s"] / cpp_stats["median_s"]
    speedup_py = serial_stats["median_s"] / py_stats["median_s"]

    print(f"Speedup (C++ median): {speedup_cpp:.2f}x")
    print(f"Speedup (Py median):  {speedup_py:.2f}x")

    res_cpp = read_parallel_torchfits()
    res_serial = read_serial_torchfits()

    assert len(res_cpp) == len(res_serial)
    for i in range(n_files):
        assert torch.allclose(res_cpp[i], res_serial[i])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel I/O diagnostic benchmark (informational, non-gating)."
    )
    parser.add_argument("--filename-base", type=str, default="parallel_test")
    parser.add_argument("--n-files", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)
    bench_parallel(cli_args)
