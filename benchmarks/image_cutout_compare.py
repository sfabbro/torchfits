#!/usr/bin/env python3
"""Image cutout comparison micro-benchmark.

Reads N random cutouts from a generated image and compares:
  - torchfits read(start,shape)
  - astropy numpy slice (if available)
  - fitsio numpy slice (if available)

Also times numpy->torch conversion explicitly to make pipeline costs visible.
"""
from __future__ import annotations

import argparse
import os
import random
import tempfile
from typing import List, Tuple

import numpy as np
import torch
import torchfits as tf

import sys
sys.path.append(os.path.dirname(__file__))
from bench_utils import format_table, time_repeat, try_import, numpy_to_torch  # type: ignore


def _make_image(path: str, shape: Tuple[int, int]) -> None:
    arr = (np.random.rand(*shape).astype(np.float32) * 1000).astype(np.float32)
    tf.write(path, torch.from_numpy(arr), overwrite=True)


def _coords(n: int, shape: Tuple[int, int], hw: int) -> List[Tuple[int, int]]:
    h, w = shape
    coords: List[Tuple[int, int]] = []
    for _ in range(n):
        y = random.randint(0, max(0, h - hw))
        x = random.randint(0, max(0, w - hw))
        coords.append((y, x))
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cutouts", type=int, default=10)
    ap.add_argument("--cutout-size", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    random.seed(0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "img.fits")
        _make_image(path, (args.size, args.size))
        coords = _coords(args.cutouts, (args.size, args.size), args.cutout_size)

        astropy = try_import("astropy.io.fits")
        fitsio = try_import("fitsio")

        rows = []
        headers = ["Impl", "API", "mean ms", "stdev", "notes"]

        # torchfits
        def _tf():
            for (y, x) in coords:
                _ = tf.read(path, start=[y, x], shape=[args.cutout_size, args.cutout_size])[0]

        m, s, _ = time_repeat(_tf, reps=args.reps)
        rows.append(["torchfits", "read(start,shape)", f"{m:.2f}", f"{s:.2f}", f"{args.cutouts}x{args.cutout_size}^2"])

        # astropy
        if astropy is not None:
            def _ap():
                with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                    data = hdul[0].data
                    for (y, x) in coords:
                        cut = data[y:y+args.cutout_size, x:x+args.cutout_size]
                        _ = numpy_to_torch(cut)

            m, s, _ = time_repeat(_ap, reps=args.reps)
            rows.append(["astropy", "slice + np->torch", f"{m:.2f}", f"{s:.2f}", "np->torch per cutout"])
        else:
            rows.append(["astropy", "slice + np->torch", "n/a", "", "missing module"])

        # fitsio
        if fitsio is not None:
            def _fi():
                with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                    data = f[0].read()
                    for (y, x) in coords:
                        cut = data[y:y+args.cutout_size, x:x+args.cutout_size]
                        _ = numpy_to_torch(cut)

            m, s, _ = time_repeat(_fi, reps=args.reps)
            rows.append(["fitsio", "slice + np->torch", f"{m:.2f}", f"{s:.2f}", "np->torch per cutout"])
        else:
            rows.append(["fitsio", "slice + np->torch", "n/a", "", "missing module"])

        print("\n== Cutout comparison ==")
        print(format_table(rows, headers=headers))


if __name__ == "__main__":
    main()
