#!/usr/bin/env python
"""Micro-benchmark for default vs memory-mapped image read paths.
Outputs JSONL with two entries per file: default and mmap.
Only applies to uncompressed images; compressed images will fall back.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torchfits as tf


def time_read(path: str, enable_mmap: bool | None) -> tuple[float, dict | None]:
    t0 = time.perf_counter()
    tf.read(path, hdu=0, enable_mmap=enable_mmap)
    ms = (time.perf_counter() - t0) * 1000.0
    info = None
    try:
        info = tf.get_last_read_info()
    except Exception:
        pass
    return ms, info


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
            t_def, i_def = time_read(path, enable_mmap=None)
            f.write(json.dumps({
                "benchmark": "mmap_vs_default",
                "mode": "default",
                "file": path,
                "time_ms": round(t_def, 3),
                "path_used": (i_def or {}).get("path_used"),
                "used_mmap": bool((i_def or {}).get("used_mmap", False)),
            }) + "\n")
            t_mmap, i_mmap = time_read(path, enable_mmap=True)
            f.write(json.dumps({
                "benchmark": "mmap_vs_default",
                "mode": "mmap",
                "file": path,
                "time_ms": round(t_mmap, 3),
                "path_used": (i_mmap or {}).get("path_used"),
                "used_mmap": bool((i_mmap or {}).get("used_mmap", False)),
            }) + "\n")
            print({"file": path, "default_ms": round(t_def, 3), "mmap_ms": round(t_mmap, 3), "default_path": (i_def or {}).get("path_used"), "mmap_path": (i_mmap or {}).get("path_used")})

    print(f"Wrote micro-benchmarks to {args.output}")


if __name__ == "__main__":
    main()
