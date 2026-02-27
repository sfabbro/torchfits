import argparse
import os
import random
import time
from typing import Any, Callable, Dict

import fitsio
import numpy as np

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
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean_s": mean,
        "median_s": median,
        "std_s": std,
        "p10_s": p10,
        "p90_s": p90,
        "spread_s": p90 - p10,
    }


def bench_scaled(args: argparse.Namespace) -> None:
    filename = args.filename
    shape = (4096, 4096)
    dtype = np.int16
    bscale = 1.5
    bzero = 32768.0

    print("[diagnostic] Scaled-path microbenchmark (non-gating)")
    print("[diagnostic] This script is for local diagnostics, not release gating.")

    if args.recreate and os.path.exists(filename):
        os.remove(filename)

    if not os.path.exists(filename):
        print(f"Creating {filename} with shape {shape} and scaling...")
        data = np.random.randint(-32768, 32767, size=shape, dtype=dtype)
        fitsio.write(filename, data, clobber=True)
        with fitsio.FITS(filename, "rw") as f:
            f[0].write_key("BSCALE", bscale)
            f[0].write_key("BZERO", bzero)
    else:
        print(f"Reusing existing {filename}")

    def read_torchfits():
        return torchfits.read(filename, policy="smart", scale_on_device=True)

    def read_fitsio():
        return fitsio.read(filename)

    tf_stats = _time_callable(read_torchfits, runs=args.runs, warmup=args.warmup)
    fi_stats = _time_callable(read_fitsio, runs=args.runs, warmup=args.warmup)

    print("Benchmarking scaled read...")
    print(
        "TorchFits: "
        f"median={tf_stats['median_s'] * 1000:.2f} ms "
        f"(p10={tf_stats['p10_s'] * 1000:.2f}, p90={tf_stats['p90_s'] * 1000:.2f}, "
        f"spread={tf_stats['spread_s'] * 1000:.2f})"
    )
    print(
        "Fitsio:    "
        f"median={fi_stats['median_s'] * 1000:.2f} ms "
        f"(p10={fi_stats['p10_s'] * 1000:.2f}, p90={fi_stats['p90_s'] * 1000:.2f}, "
        f"spread={fi_stats['spread_s'] * 1000:.2f})"
    )
    print(f"Speedup (median): {fi_stats['median_s'] / tf_stats['median_s']:.2f}x")

    tf_data = read_torchfits()
    fitsio_data = read_fitsio()
    diff = np.abs(tf_data.numpy() - fitsio_data).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaled-path diagnostic benchmark (informational, non-gating)."
    )
    parser.add_argument("--filename", type=str, default="scaled.fits")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)
    bench_scaled(cli_args)
