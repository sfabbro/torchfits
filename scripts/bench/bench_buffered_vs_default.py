#!/usr/bin/env python
"""Micro-benchmark for default vs buffered/tiled image read paths.
Outputs JSONL with two entries per file: default and buffered.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torchfits as tf


def time_read(path: str, enable_buffered: bool | None) -> float:
    t0 = time.perf_counter()
    tf.read(path, hdu=0, enable_buffered=enable_buffered)
    return (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=False,
                    help="FITS image files to benchmark (defaults to examples)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = args.input or []
    if not files:
        ex = Path("examples")
        defaults = [ex / "basic_example.fits"]
        files = [p for p in defaults if p.exists()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for p in files:
            path = str(p)
            t_def = time_read(path, enable_buffered=None)
            f.write(json.dumps({
                "benchmark": "buffered_vs_default",
                "mode": "default",
                "file": path,
                "time_ms": round(t_def, 3),
            }) + "\n")
            t_buf = time_read(path, enable_buffered=True)
            f.write(json.dumps({
                "benchmark": "buffered_vs_default",
                "mode": "buffered",
                "file": path,
                "time_ms": round(t_buf, 3),
            }) + "\n")
            print({"file": path, "default_ms": round(t_def, 3), "buffered_ms": round(t_buf, 3)})

    print(f"Wrote micro-benchmarks to {args.output}")


if __name__ == "__main__":
    main()
