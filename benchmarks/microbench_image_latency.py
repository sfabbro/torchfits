#!/usr/bin/env python3
"""Microbench: image read latency hot vs cold, torchfits vs fitsio.

This is intentionally small and single-purpose so we can iterate quickly while
optimizing the C++ read paths.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torchfits

try:
    import fitsio
except ImportError:  # pragma: no cover
    fitsio = None

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None


def _time_many(fn, *, iters: int, warmup: int, clear_each: callable | None = None):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if clear_each is not None:
            clear_each()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "mean_s": mean(times),
        "median_s": median(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def _mk_fits(path: Path, arr: np.ndarray):
    if fits is None:
        raise RuntimeError("astropy is required for this microbench")
    fits.PrimaryHDU(arr).writeto(path, overwrite=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="torchfits_microbench_img_"))
    paths = {
        "tiny_int8_2d": tmp / "tiny_int8_2d.fits",
        "small_int8_2d": tmp / "small_int8_2d.fits",
        "medium_int8_2d": tmp / "medium_int8_2d.fits",
        "medium_int16_2d": tmp / "medium_int16_2d.fits",
    }

    rng = np.random.default_rng(0)
    _mk_fits(paths["tiny_int8_2d"], rng.integers(0, 255, size=(64, 64), dtype=np.uint8))
    _mk_fits(
        paths["small_int8_2d"], rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    )
    _mk_fits(
        paths["medium_int8_2d"], rng.integers(0, 255, size=(1024, 1024), dtype=np.uint8)
    )
    _mk_fits(
        paths["medium_int16_2d"],
        rng.integers(-1000, 1000, size=(1024, 1024), dtype=np.int16),
    )

    def clear_torchfits():
        try:
            torchfits.clear_file_cache()
        except Exception:
            pass

    for name, path in paths.items():
        print(f"\n== {name} ==")

        for mmap in (True, False):
            hot = _time_many(
                lambda: torchfits.read(str(path), hdu=0, mmap=mmap, cache_capacity=10),
                iters=args.iters,
                warmup=args.warmup,
                clear_each=None,
            )
            cold = _time_many(
                lambda: torchfits.read(str(path), hdu=0, mmap=mmap, cache_capacity=0),
                iters=args.iters,
                warmup=0,
                clear_each=clear_torchfits,
            )
            print(
                f"torchfits mmap={mmap} hot  : median={hot['median_s'] * 1e3:.3f}ms mean={hot['mean_s'] * 1e3:.3f}ms"
            )
            print(
                f"torchfits mmap={mmap} cold : median={cold['median_s'] * 1e3:.3f}ms mean={cold['mean_s'] * 1e3:.3f}ms"
            )

        if fitsio is not None:
            rec = _time_many(
                lambda: fitsio.read(str(path), ext=0),
                iters=args.iters,
                warmup=args.warmup,
                clear_each=None,
            )
            print(
                f"fitsio read      : median={rec['median_s'] * 1e3:.3f}ms mean={rec['mean_s'] * 1e3:.3f}ms"
            )


if __name__ == "__main__":
    main()
