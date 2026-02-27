import argparse
import gc
import os
import random
import time
from typing import Any, Callable, Dict

import fitsio
import numpy as np
import psutil

import torchfits


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


def get_open_fds() -> int:
    process = psutil.Process()
    return process.num_fds()


def bench_mmap(args: argparse.Namespace) -> None:
    filename = args.filename
    shape = (4096, 4096)
    dtype = np.float32

    print("[diagnostic] MMap/safety microbenchmark (non-gating)")
    print("[diagnostic] This script is for local diagnostics, not release gating.")

    if args.recreate and os.path.exists(filename):
        os.remove(filename)

    if not os.path.exists(filename):
        print(f"Creating {filename} with shape {shape}...")
        data = np.random.randn(*shape).astype(dtype)
        fitsio.write(filename, data, clobber=True)
    else:
        print(f"Reusing existing {filename}")

    print("Benchmarking mmap read...")

    initial_fds = get_open_fds()
    print(f"Initial open FDs: {initial_fds}")

    def read_torchfits():
        return torchfits.read(
            filename,
            mmap=True,
            policy="smart",
            cache_capacity=args.cache_capacity,
        )

    def read_fitsio():
        return fitsio.read(filename)

    tf_stats = _time_callable(read_torchfits, runs=args.runs, warmup=args.warmup)
    fi_stats = _time_callable(read_fitsio, runs=args.runs, warmup=args.warmup)

    print(
        "TorchFits (mmap): "
        f"median={tf_stats['median_s'] * 1000:.2f} ms "
        f"(p10={tf_stats['p10_s'] * 1000:.2f}, p90={tf_stats['p90_s'] * 1000:.2f}, "
        f"spread={tf_stats['spread_s'] * 1000:.2f})"
    )
    print(
        "Fitsio:           "
        f"median={fi_stats['median_s'] * 1000:.2f} ms "
        f"(p10={fi_stats['p10_s'] * 1000:.2f}, p90={fi_stats['p90_s'] * 1000:.2f}, "
        f"spread={fi_stats['spread_s'] * 1000:.2f})"
    )
    print(f"Speedup (median):  {fi_stats['median_s'] / tf_stats['median_s']:.2f}x")

    torchfits.clear_file_cache()
    gc.collect()
    final_fds = get_open_fds()
    print(f"Final open FDs: {final_fds}")

    if final_fds > initial_fds:
        print(f"[diagnostic] WARN: potential FD leak ({final_fds - initial_fds} extra FDs)")
    else:
        print("[diagnostic] PASS: no FD leak detected.")

    print("\nTesting Error Handling...")
    try:
        torchfits.open("non_existent_file.fits")
    except FileNotFoundError:
        print("[diagnostic] PASS: correctly caught FileNotFoundError")
    except Exception as e:
        print(f"[diagnostic] WARN: unexpected error for missing file: {type(e).__name__}: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMap/safety diagnostic benchmark (informational, non-gating)."
    )
    parser.add_argument("--filename", type=str, default="large_mmap.fits")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache-capacity", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)
    bench_mmap(cli_args)
