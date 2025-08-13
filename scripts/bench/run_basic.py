#!/usr/bin/env python
"""Basic benchmark skeleton producing JSONL metrics.
Focus: image read, table read (if sample files present), header access.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torchfits as tf
import torch

CASES = []

# Discover example FITS files
EX_DIR = Path("examples")
if (EX_DIR / "basic_example.fits").exists():
    CASES.append({"name": "read_image_basic", "path": str(EX_DIR / "basic_example.fits"), "hdu": 0, "kind": "image"})
if (EX_DIR / "table_example.fits").exists():
    CASES.append({"name": "read_table_basic", "path": str(EX_DIR / "table_example.fits"), "hdu": 1, "kind": "table"})


def bench_case(case):
    t0 = time.perf_counter()
    if case["kind"] == "image":
        data, header = tf.read(case["path"], hdu=case["hdu"], format="tensor")
        size = int(data.numel())
    else:
        table = tf.read(case["path"], hdu=case["hdu"], format="table")
        # Sum tensor elements only; skip string columns and count VLA elements if present
        size = 0
        for v in table.data.values():
            if isinstance(v, torch.Tensor):
                size += int(v.numel())
            elif isinstance(v, (list, tuple)):
                # Variable-length array columns as list[Tensor]
                for item in v:
                    if isinstance(item, torch.Tensor):
                        size += int(item.numel())
    dt = (time.perf_counter() - t0) * 1000
    return {
        "benchmark": "basic_read",
        "case": case["name"],
        "wall_time_ms": round(dt, 3),
        "elements": size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in CASES:
            result = bench_case(c)
            f.write(json.dumps(result) + "\n")
            print(result)
    print(f"Wrote benchmarks to {args.output}")

if __name__ == "__main__":
    main()
