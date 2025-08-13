#!/usr/bin/env python3
"""
Prefetch micro-benchmark for FITSIterableDataset.

Measures wall time with prefetch disabled vs enabled on an I/O-bound loop
with a small transform delay. Emits JSON with timings and speedup.

Usage:
  python benchmarks/prefetch_iterable_bench.py --output artifacts/validation/prefetch_bench.json
  python benchmarks/prefetch_iterable_bench.py --reps 3 --sleep 0.002
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torchfits as tf
from torchfits.dataset import FITSIterableDataset


def _make_mef(tmpdir: str, shape=(256, 256)) -> str:
    path = os.path.join(tmpdir, "bench_mef.fits")
    ten = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(*shape)
    tf.write_mef(path, [ten], overwrite=True)
    return path


def run_benchmark(reps: int = 3, sleep_s: float = 0.001) -> dict:
    results = []
    with tempfile.TemporaryDirectory() as td:
        path = _make_mef(td, shape=(256, 256))
        # Prepare many small cutouts
        source_specs = [
            {"path": path, "hdu": 1, "start": (i, 0), "shape": (32, 32)}
            for i in range(0, 256, 32)
        ] * 8  # increase iterations

        def transform(x):
            time.sleep(sleep_s)
            return x

        for _ in range(reps):
            t0 = time.time()
            list(FITSIterableDataset(source_specs, prefetch=0, transform=transform))
            t_no_prefetch = time.time() - t0

            t0 = time.time()
            list(FITSIterableDataset(source_specs, prefetch=4, transform=transform))
            t_prefetch = time.time() - t0

            results.append({
                "no_prefetch_s": t_no_prefetch,
                "prefetch_s": t_prefetch,
                "speedup": (t_no_prefetch / t_prefetch) if t_prefetch > 0 else None,
            })

    # Aggregate
    no_prefetch = [r["no_prefetch_s"] for r in results]
    prefetch = [r["prefetch_s"] for r in results]
    speedups = [r["speedup"] for r in results if r["speedup"]]
    summary = {
        "reps": reps,
        "sleep_s": sleep_s,
        "no_prefetch_mean_s": sum(no_prefetch) / len(no_prefetch),
        "prefetch_mean_s": sum(prefetch) / len(prefetch),
        "speedup_mean": sum(speedups) / len(speedups) if speedups else None,
        "runs": results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="Number of repetitions")
    ap.add_argument(
        "--sleep", type=float, default=0.001, help="Transform sleep seconds per sample"
    )
    ap.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = ap.parse_args()

    summary = run_benchmark(reps=args.reps, sleep_s=args.sleep)
    print(json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
