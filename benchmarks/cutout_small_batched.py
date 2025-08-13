#!/usr/bin/env python3
"""Micro-benchmark for [P1]: small cutouts batched vs per-cutout loop.

Measures two methods on a single image and uniform small cutout size:
  - loop: call torchfits.read(start,shape) for each cutout
  - batched: use torchfits.dataset.read_multi_cutouts (sequential) which triggers
             the new C++ batched small-cutout path when applicable

Outputs a compact table and appends JSONL results to artifacts/benchmarks/dev.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from typing import List, Tuple

import numpy as np
import torch
import torchfits as tf
from torchfits.dataset import FITSCutoutSpec, FITSMultiCutoutSpec, read_multi_cutouts

import sys
sys.path.append(os.path.dirname(__file__))
from bench_utils import format_table, time_repeat  # type: ignore


def _make_image(path: str, shape: Tuple[int, int]) -> None:
    arr = (np.random.rand(*shape).astype(np.float32) * 1000).astype(np.float32)
    tf.write(path, torch.from_numpy(arr), overwrite=True)


def _random_coords(n: int, shape: Tuple[int, int], hw: int) -> List[Tuple[int, int]]:
    h, w = shape
    coords = []
    for _ in range(n):
        y = random.randint(0, max(0, h - hw))
        x = random.randint(0, max(0, w - hw))
        coords.append((y, x))
    return coords


def run_once(size: int, cutouts: int, hw: int, reps: int) -> list[list[str]]:
    rows: list[list[str]] = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "img.fits")
        _make_image(path, (size, size))
        coords = _random_coords(cutouts, (size, size), hw)

        # loop method
        def _loop():
            for (y, x) in coords:
                _ = tf.read(path, start=[y, x], shape=[hw, hw])[0]

        m, s, _ = time_repeat(_loop, reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", "loop tf.read", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{hw}^2"]) 

        # batched method (sequential)
        def _batched():
            specs = [FITSCutoutSpec(hdu=0, start=(y, x), shape=(hw, hw), device="cpu") for (y, x) in coords]
            out = read_multi_cutouts(FITSMultiCutoutSpec(path=path, cutouts=specs, parallel=False, return_dict=False))
            # consume tensors
            for t in out:
                if isinstance(t, tuple):
                    t = t[0]
                if torch.is_tensor(t):
                    _ = t.numel()

        m, s, _ = time_repeat(_batched, reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", "batched multi-cutout", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{hw}^2"]) 

    # print results
    print(f"\n== Small cutouts batched vs loop (size={size}, hw={hw}, N={cutouts}) ==")
    print(format_table(rows, headers=headers))

    # append JSONL
    os.makedirs("artifacts/benchmarks", exist_ok=True)
    out_path = "artifacts/benchmarks/dev.jsonl"
    for r in rows:
        if r[2] == "n/a":
            continue
        rec = {
            "scenario": "cutouts_small_batched",
            "size": size,
            "cutouts": cutouts,
            "cut_hw": hw,
            "impl": r[0],
            "api": r[1],
            "mean_ms": float(r[2]),
            "stdev_ms": float(r[3]) if r[3] else None,
            "notes": r[4],
            "reps": reps,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cutouts", type=int, default=200)
    ap.add_argument("--hw", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    random.seed(0)
    run_once(args.size, args.cutouts, args.hw, args.reps)


if __name__ == "__main__":
    main()
