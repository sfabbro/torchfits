#!/usr/bin/env python3
"""Quick test to verify scaled fast-path usage."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in sys.path so we can import benchmarks.config
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import time
from statistics import median
from pathlib import Path

import torch
import torchfits

from bench_all import ExhaustiveBenchmarkSuite


def _time(fn, iters: int = 20) -> float:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)


def main() -> None:
    suite = ExhaustiveBenchmarkSuite(output_dir=DEFAULT_OUTPUT_DIR)
    files = suite.create_test_files()
    path = files.get("scaled_small")
    if not path:
        print("scaled_small not found")
        return

    def read_scaled():
        return torchfits.read(str(path), hdu=0, mmap=True, scale_on_device=True)

    def read_direct():
        data, scaled, bscale, bzero = torchfits.cpp.read_full_raw_with_scale(
            str(path), 0, True
        )
        if scaled:
            data = data.to(dtype=torch.float32)
            if bscale != 1.0:
                data.mul_(bscale)
            if bzero != 0.0:
                data.add_(bzero)
        return data

    t_scaled = _time(read_scaled)
    t_direct = _time(read_direct)

    print(f"scaled_small read(): {t_scaled:.6f}s")
    print(f"scaled_small direct: {t_direct:.6f}s")
    if t_scaled <= t_direct * 1.2:
        print("OK: scaled path matches direct fast path.")
    else:
        print("WARN: scaled path is slower than direct fast path.")


if __name__ == "__main__":
    main()
