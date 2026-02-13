#!/usr/bin/env python3
"""
Quick compressed-image sanity benchmark for torchfits vs fitsio.

This is intentionally small so we can iterate without running the full suite.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import fitsio
import numpy as np
import torch
from astropy.io import fits
from astropy.io.fits import CompImageHDU

import torchfits


def _create_compressed(path: Path, compression_type: str, shape=(1024, 1024)) -> None:
    data = (np.random.randn(*shape) * 100.0).astype(np.float32)
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), CompImageHDU(data, compression_type=compression_type)]
    )
    hdul.writeto(path, overwrite=True)


def _time(fn, warmup: int = 12, iterations: int = 80) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - t0) / iterations


def _bench_one(path: Path, name: str) -> None:
    tf = _time(
        lambda: torchfits.read(
            str(path),
            hdu=1,
            return_header=False,
            cache_capacity=10,
            scale_on_device=True,
        )
    )
    fi = _time(lambda: fitsio.read(str(path), ext=1))
    ratio = tf / fi if fi > 0 else float("inf")
    print(f"{name}: torchfits={tf:.6f}s fitsio={fi:.6f}s ratio={ratio:.3f}x")


def main() -> None:
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)

    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_compressed_debug_"))
    rice_path = data_dir / "rice.fits"
    hcomp_path = data_dir / "hcompress.fits"

    _create_compressed(rice_path, "RICE_1")
    _create_compressed(hcomp_path, "HCOMPRESS_1")

    _bench_one(rice_path, "RICE_1")
    _bench_one(hcomp_path, "HCOMPRESS_1")


if __name__ == "__main__":
    main()
