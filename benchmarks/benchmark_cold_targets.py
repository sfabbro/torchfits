#!/usr/bin/env python3
"""
Focused cold-cache benchmark for known regression-prone file types.
"""

import argparse
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

import fitsio
import numpy as np
from astropy.io import fits as astropy_fits

import torchfits


def _time(fn, warmup: int, runs: int):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


def _write_file(path: Path, shape, dtype, with_wcs: bool = False):
    if np.issubdtype(dtype, np.integer):
        if dtype == np.int8:
            data = np.random.randint(-100, 100, size=shape, dtype=dtype)
        elif dtype == np.int16:
            data = np.random.randint(-1000, 1000, size=shape, dtype=dtype)
        else:
            data = np.random.randint(-10000, 10000, size=shape, dtype=dtype)
    else:
        data = np.random.randn(*shape).astype(dtype)

    hdu = astropy_fits.PrimaryHDU(data)
    if with_wcs:
        hdu.header["CRPIX1"] = shape[-1] / 2
        hdu.header["CRPIX2"] = shape[0] / 2
        hdu.header["CRVAL1"] = 180.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CDELT1"] = -0.0001
        hdu.header["CDELT2"] = 0.0001
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.writeto(path, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Cold-cache target benchmark")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    base = Path(tempfile.mkdtemp(prefix="torchfits_cold_targets_"))
    files = {
        "wcs_image": ("wcs_image.fits", (1024, 1024), np.float32, True),
        "medium_int8_2d": ("medium_int8_2d.fits", (1024, 1024), np.int8, False),
        "medium_float32_2d": ("medium_float32_2d.fits", (1024, 1024), np.float32, False),
        "large_int32_1d": ("large_int32_1d.fits", (1_000_000,), np.int32, False),
        "large_int8_1d": ("large_int8_1d.fits", (1_000_000,), np.int8, False),
        "large_float32_2d": ("large_float32_2d.fits", (2048, 2048), np.float32, False),
        "medium_int16_2d": ("medium_int16_2d.fits", (1024, 1024), np.int16, False),
    }

    print(f"Cold target files in {base}")
    for name, (fname, shape, dtype, with_wcs) in files.items():
        path = base / fname
        _write_file(path, shape, dtype, with_wcs=with_wcs)

        print(f"\n{name}")
        print("-" * 80)

        methods = {
            "torchfits_read_cold": lambda p=path: torchfits.read(
                str(p), mmap=True, cache_capacity=0
            ),
            "torchfits_read_cold_nommap": lambda p=path: torchfits.read(
                str(p), mmap=False, cache_capacity=0
            ),
            "torchfits_cpp_read_full": lambda p=path: torchfits.cpp.read_full(
                str(p), 0, True
            ),
            "torchfits_cpp_read_full_nocache": lambda p=path: torchfits.cpp.read_full_nocache(
                str(p), 0, True
            ),
            "fitsio_read": lambda p=path: fitsio.read(str(p)),
        }

        for mname, fn in methods.items():
            m, s = _time(fn, warmup=args.warmup, runs=args.runs)
            print(f"{mname:32s}: {m:.6f}s ± {s:.6f}s")

    # Dedicated MEF medium-style benchmark (ext=1 image read)
    print("\nmef_medium (ext=1)")
    print("-" * 80)
    mef_path = base / "mef_medium.fits"
    hdus = [astropy_fits.PrimaryHDU(np.random.randn(256, 256).astype(np.float32))]
    for i in range(8):
        hdus.append(astropy_fits.ImageHDU(np.random.randn(256, 256).astype(np.float32) + i))
    astropy_fits.HDUList(hdus).writeto(mef_path, overwrite=True)

    mef_methods = {
        "torchfits_read_cold_ext1": lambda p=mef_path: torchfits.read(
            str(p), hdu=1, mmap=True, cache_capacity=0
        ),
        "torchfits_cpp_read_full_ext1": lambda p=mef_path: torchfits.cpp.read_full(
            str(p), 1, True
        ),
        "torchfits_cpp_read_nocache_ext1": lambda p=mef_path: torchfits.cpp.read_full_nocache(
            str(p), 1, True
        ),
        "fitsio_read_ext1": lambda p=mef_path: fitsio.read(str(p), ext=1),
    }
    for mname, fn in mef_methods.items():
        m, s = _time(fn, warmup=args.warmup, runs=args.runs)
        print(f"{mname:32s}: {m:.6f}s ± {s:.6f}s")


if __name__ == "__main__":
    main()
