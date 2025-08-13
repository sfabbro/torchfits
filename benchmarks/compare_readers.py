#!/usr/bin/env python3
"""Comparison benchmarks across torchfits, astropy, and fitsio.

Scenarios:
  - Image full read to CPU tensor
  - Random cutouts (10 samples) with WCS usage
  - Table read with diversified columns + dtype conversions
  - numpy->torch pipelines for astropy and fitsio

Console output prints a compact comparison table per scenario.

Usage:
    python benchmarks/compare_readers.py --size 1024 --cutouts 10 --reps 5 \
        --mef-hdus 3 --files 3 --sky-cutouts 10 --sky-radius-arcsec 30
"""
from __future__ import annotations

import argparse
import os
import random
import tempfile
import platform
import time
import socket
from dataclasses import dataclass
import json
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
import torchfits as tf

import sys
sys.path.append(os.path.dirname(__file__))
from bench_utils import format_table, time_repeat, try_import, numpy_to_torch, force_consume  # type: ignore


np.random.seed(0)
random.seed(0)


def _make_image(path: str, shape: Tuple[int, int] = (1024, 1024)) -> None:
    data = (np.random.rand(*shape).astype(np.float32) * 1000).astype(np.float32)
    # Write via torchfits to keep it simple
    tf.write(path, torch.from_numpy(data), overwrite=True)


def _make_table(path: str, rows: int = 200_000) -> None:
    # diversified columns: ints, floats, strings, bools
    data = {
        "I32": torch.randint(0, 10_000, (rows,), dtype=torch.int32),
        "I64": torch.randint(0, 10_000, (rows,), dtype=torch.int64),
        "F32": torch.randn(rows, dtype=torch.float32),
        "F64": torch.randn(rows, dtype=torch.float64),
    "B": torch.randint(0, 2, (rows,), dtype=torch.bool),
    # For strings, torchfits accepts Python list[str]
    "S": [f"id_{i:06d}" for i in range(rows)],
    }
    tf.write_table(path, data, overwrite=True)


def _random_cutouts(n: int, shape: Tuple[int, int], cutout_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
    h, w = shape
    ch, cw = cutout_hw
    coords = []
    for _ in range(n):
        y = random.randint(0, max(0, h - ch))
        x = random.randint(0, max(0, w - cw))
        coords.append((y, x))
    return coords


def _make_mef(path: str, hdu_count: int = 3, size: int = 512) -> bool:
    """Create a simple MEF with multiple image HDUs using astropy.

    Returns True if created, False if astropy missing.
    """
    apfits = try_import("astropy.io.fits")
    if apfits is None:
        return False
    rng = np.random.default_rng(0)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdus = [apfits.PrimaryHDU()]  # type: ignore[attr-defined]
    for i in range(hdu_count):
        arr = (rng.random((size, size), dtype=np.float32) * 1000).astype(np.float32)
        hdu = apfits.ImageHDU(arr, name=f"SCI{i+1}")  # type: ignore[attr-defined]
        hdus.append(hdu)
    apfits.HDUList(hdus).writeto(path, overwrite=True)  # type: ignore[attr-defined]
    return True


def _make_wcs_image(path: str, size: int = 1024) -> bool:
    """Create a WCS-bearing FITS image for sky cutout tests via astropy."""
    apfits = try_import("astropy.io.fits")
    if apfits is None:
        return False
    data = (np.random.rand(size, size).astype(np.float32) * 1000).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu = apfits.PrimaryHDU(data)  # type: ignore[attr-defined]
    hdr = hdu.header
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 200.0
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX1"] = size / 2
    hdr["CRPIX2"] = size / 2
    hdr["CDELT1"] = -0.0002777777778  # ~1 arcsec/pixel
    hdr["CDELT2"] = 0.0002777777778
    hdu.writeto(path, overwrite=True)
    return True


@dataclass
class Scenario:
    name: str
    run: Callable[[], None]


def _tf_opts(enable_mmap: Optional[bool], cache_capacity: int, enable_buffered: Optional[bool] = None) -> dict:
    return {
        "enable_mmap": enable_mmap,
        "enable_buffered": enable_buffered,
        "cache_capacity": cache_capacity,
    }


def _tf_auto_candidates(allow_buffered: bool = True, shuffle: bool = True):
    # Try mmap on/off and optionally buffered on (non-mmap) as candidates
    cands: list[tuple[bool, Optional[bool], str]] = [
        (True, None, "mmap=on"),
        (False, None, "mmap=off"),
    ]
    if allow_buffered:
        cands.append((False, True, "buffered=on"))
    if shuffle:
        random.shuffle(cands)
    return cands


def _tf_auto_candidates_size_aware(
    *,
    op: str,
    allow_buffered_default: bool = True,
    size: Optional[int] = None,
    cut_hw: Optional[int] = None,
    files: Optional[int] = None,
    hdus: Optional[int] = None,
    half_hw: Optional[int] = None,
    shuffle: bool = True,
):
    """Heuristic gating of auto candidates based on workload size.

    Returns a list of (enable_mmap, enable_buffered, label) tuples.
    """
    def c(enable_mmap: bool, enable_buffered: Optional[bool], label: str):
        return (enable_mmap, enable_buffered, label)

    cands: list[tuple[bool, Optional[bool], str]] = []

    if op == "image_full":
        # Small images: prefer mmap paths; large: allow buffered
        sz = size or 0
        if sz < 384:
            cands = [c(True, None, "mmap=on"), c(False, None, "mmap=off")]
        else:
            if allow_buffered_default:
                cands = [c(False, True, "buffered=on"), c(False, None, "mmap=off"), c(True, None, "mmap=on")]
            else:
                cands = [c(False, None, "mmap=off"), c(True, None, "mmap=on")]
    elif op in ("cutouts", "mef_cutouts", "multifile_cutouts"):
        hw = cut_hw or 0
        # Tiny windows: avoid buffered; larger windows: include buffered
        tiny = hw <= 32
        many = (files or 0) >= 3 or (hdus or 0) >= 3
        if tiny and not many:
            cands = [c(True, None, "mmap=on"), c(False, None, "mmap=off")]
        else:
            if allow_buffered_default:
                cands = [c(False, True, "buffered=on"), c(True, None, "mmap=on"), c(False, None, "mmap=off")]
            else:
                cands = [c(True, None, "mmap=on"), c(False, None, "mmap=off")]
    elif op == "sky_cutouts":
        # Use half_hw (half window in pixels)
        hh = half_hw or 0
        if hh <= 16:
            cands = [c(True, None, "mmap=on"), c(False, None, "mmap=off")]
        else:
            if allow_buffered_default:
                cands = [c(False, True, "buffered=on"), c(False, None, "mmap=off"), c(True, None, "mmap=on")]
            else:
                cands = [c(False, None, "mmap=off"), c(True, None, "mmap=on")]
    else:
        # Fallback: generic behavior
        cands = _tf_auto_candidates(allow_buffered_default, shuffle=False)

    if shuffle:
        random.shuffle(cands)
    return cands


def _collect_meta_record() -> dict:
    """Collect environment and library version info for JSONL."""
    meta: dict = {
        "scenario": "meta",
        "timestamp": int(time.time()),
        "python": sys.version.split(" ")[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "hostname": socket.gethostname(),
        },
        "versions": {},
    }
    # library versions
    try:
        meta["versions"]["numpy"] = np.__version__
    except Exception:
        pass
    try:
        meta["versions"]["torch"] = torch.__version__
    except Exception:
        pass
    try:
        meta["versions"]["torchfits"] = getattr(tf, "__version__", "unknown")
    except Exception:
        pass
    apfits = try_import("astropy")
    if apfits is not None:
        try:
            meta["versions"]["astropy"] = getattr(apfits, "__version__", "unknown")
        except Exception:
            pass
    fi = try_import("fitsio")
    if fi is not None:
        try:
            meta["versions"]["fitsio"] = getattr(fi, "__version__", "unknown")
        except Exception:
            pass
    return meta


def bench_image_full(tmp: str, size: int, reps: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    path = os.path.join(tmp, "img.fits")
    _make_image(path, (size, size))

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "to", "mean ms", "stdev", "notes"]

    # torchfits -> torch (native), pick best of mmap/buffered in auto
    def _tf_read(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        return tf.read(path, **_tf_opts(mmap_flag, cache_capacity, buffered_flag))[0]

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, label in _tf_auto_candidates_size_aware(op="image_full", allow_buffered_default=True, size=size):
            try:
                def _runner(mm: Optional[bool] = mm, buf: Optional[bool] = buf):
                    return _tf_read(mm, buf)
                m, s, _ = time_repeat(_runner, reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or m < best[0]:
                best = (m, s, label)
        if best is None:
            # fallback mmap off
            def _runner_off():
                return _tf_read(False, None)
            m, s, _ = time_repeat(_runner_off, reps=reps, warmup=1, use_median=True)
            best = (m, s, "mmap=off")
        rows.append(["torchfits", f"torch ({best[2]})", f"{best[0]:.2f}", f"{best[1]:.2f}", "native"])
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        def _runner_flag():
            return _tf_read(mm_flag, None)
        m, s, _ = time_repeat(_runner_flag, reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", f"torch (mmap={'auto' if mm_flag is None else 'on' if mm_flag else 'off'})", f"{m:.2f}", f"{s:.2f}", "native"])

    # competitors: build actions, then run in randomized order
    competitors: list[tuple[str, str, Callable[[], None]]] = []
    if astropy is not None:
        def _ap_np():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                arr = hdul[0].data
                force_consume(arr)
        competitors.append(("astropy", "numpy", _ap_np))
        def _ap():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                arr = hdul[0].data
                t = numpy_to_torch(arr)
                force_consume(t)
        competitors.append(("astropy", "numpy->torch", _ap))
    else:
        rows.append(["astropy", "numpy", "n/a", "", "missing module"])
        rows.append(["astropy", "numpy->torch", "n/a", "", "missing module"])
    if fitsio is not None:
        def _fi_np():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                arr = f[0].read()
                force_consume(arr)
        competitors.append(("fitsio", "numpy", _fi_np))
        def _fi():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                arr = f[0].read()
                t = numpy_to_torch(arr)
                force_consume(t)
        competitors.append(("fitsio", "numpy->torch", _fi))
    else:
        rows.append(["fitsio", "numpy", "n/a", "", "missing module"])
        rows.append(["fitsio", "numpy->torch", "n/a", "", "missing module"])
    random.shuffle(competitors)
    for name, api, fn in competitors:
        m, s, _ = time_repeat(fn, reps=reps, warmup=1, use_median=True)
        rows.append([name, api, f"{m:.2f}", f"{s:.2f}", "np only" if api=="numpy" else "np->torch"])

    print("\n== Image full read (", size, "x", size, ") ==", sep="")
    # shuffle display order to reduce any subtle order bias in presentation
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "image_full",
                "size": size,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_cutouts(tmp: str, size: int, cutouts: int, cut_hw: int, reps: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    path = os.path.join(tmp, "img_cut.fits")
    _make_image(path, (size, size))
    coords = _random_cutouts(cutouts, (size, size), (cut_hw, cut_hw))

    astropy = try_import("astropy.io.fits")
    wcsmod = try_import("astropy.wcs")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # torchfits cutouts (best-of mmap/buffered if auto)
    def _tf_cut_with(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        def _run():
            for (y, x) in coords:
                _ = tf.read(path, start=[y, x], shape=[cut_hw, cut_hw], **_tf_opts(mmap_flag, cache_capacity, buffered_flag))[0]
        return _run

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, label in _tf_auto_candidates_size_aware(op="cutouts", allow_buffered_default=True, cut_hw=cut_hw):
            try:
                m, s, _ = time_repeat(_tf_cut_with(mm, buf), reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or m < best[0]:
                best = (m, s, label)
        if best is None:
            m, s, _ = time_repeat(_tf_cut_with(False, None), reps=reps, warmup=1, use_median=True)
            best = (m, s, "mmap=off")
        rows.append(["torchfits", f"read(start,shape) ({best[2]})", f"{best[0]:.2f}", f"{best[1]:.2f}", f"{cutouts}x{cut_hw}^2"])
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        m, s, _ = time_repeat(_tf_cut_with(mm_flag, None), reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", f"read(start,shape) (mmap={'auto' if mm_flag is None else 'on' if mm_flag else 'off'})", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{cut_hw}^2"])

    # astropy cutouts + WCS transform (basic usage)
    if astropy is not None and wcsmod is not None:
        def _ap_cut():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                data = hdul[0].data
                hdr = hdul[0].header
                _ = wcsmod.WCS(hdr)  # type: ignore[attr-defined]
                for (y, x) in coords:
                    sl = data[y : y + cut_hw, x : x + cut_hw]
                    force_consume(sl)

        m, s, _ = time_repeat(_ap_cut, reps=reps, warmup=1, use_median=True)
        rows.append(["astropy", "slice + WCS", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{cut_hw}^2"])
    else:
        rows.append(["astropy", "slice + WCS", "n/a", "", "missing module"])

    # fitsio cutouts (numpy slicing)
    if fitsio is not None:
        def _fi_cut():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                data = f[0].read()
                for (y, x) in coords:
                    sl = data[y : y + cut_hw, x : x + cut_hw]
                    force_consume(sl)

        m, s, _ = time_repeat(_fi_cut, reps=reps, warmup=1, use_median=True)
        rows.append(["fitsio", "slice", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{cut_hw}^2"])
    else:
        rows.append(["fitsio", "slice", "n/a", "", "missing module"])

    print("\n== Random cutouts (", cutouts, " x ", cut_hw, "x", cut_hw, ") ==", sep="")
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "cutouts_random",
                "size": size,
                "cutouts": cutouts,
                "cut_hw": cut_hw,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_mef_cutouts(tmp: str, hdus: int, cutouts: int, cut_hw: int, reps: int, size: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    path = os.path.join(tmp, "mef.fits")
    created = _make_mef(path, hdu_count=hdus, size=size)
    if not created:
        print("\n== MEF cutouts (skipped: astropy missing) ==")
        return

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # Randomly pick (hdu index, y, x) for each cutout
    coords: List[Tuple[int, int, int]] = []
    for _ in range(cutouts):
        h = np.random.randint(1, hdus + 1)
        y = np.random.randint(0, size - cut_hw)
        x = np.random.randint(0, size - cut_hw)
        coords.append((h, y, x))

    def _tf_with(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        def _run():
            for (h, y, x) in coords:
                _ = tf.read(path, hdu=h, start=[y, x], shape=[cut_hw, cut_hw], **_tf_opts(mmap_flag, cache_capacity, buffered_flag))[0]
        return _run

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, label in _tf_auto_candidates_size_aware(op="mef_cutouts", allow_buffered_default=True, cut_hw=cut_hw, hdus=hdus):
            try:
                m, s, _ = time_repeat(_tf_with(mm, buf), reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or m < best[0]:
                best = (m, s, label)
        if best is None:
            m, s, _ = time_repeat(_tf_with(False, None), reps=reps, warmup=1, use_median=True)
            best = (m, s, "mmap=off")
        rows.append(["torchfits", f"read(hdu,start,shape) ({best[2]})", f"{best[0]:.2f}", f"{best[1]:.2f}", f"{cutouts}x{cut_hw}^2 @ {hdus} HDUs"])
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        m, s, _ = time_repeat(_tf_with(mm_flag, None), reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", f"read(hdu,start,shape) (mmap={'auto' if mm_flag is None else 'on' if mm_flag else 'off'})", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{cut_hw}^2 @ {hdus} HDUs"])

    comp_rows: list[tuple[str, str, Callable[[], None]]] = []
    if astropy is not None:
        def _ap():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                for (h, y, x) in coords:
                    data = hdul[h].data
                    sl = data[y:y+cut_hw, x:x+cut_hw]
                    force_consume(sl)
        comp_rows.append(("astropy", "slice per HDU", _ap))
    else:
        rows.append(["astropy", "slice per HDU", "n/a", "", "missing module"]) 

    if fitsio is not None:
        def _fi():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                for (h, y, x) in coords:
                    data = f[h].read()
                    sl = data[y:y+cut_hw, x:x+cut_hw]
                    force_consume(sl)
        comp_rows.append(("fitsio", "slice per HDU", _fi))
    else:
        rows.append(["fitsio", "slice per HDU", "n/a", "", "missing module"]) 

    random.shuffle(comp_rows)
    for name, api, fn in comp_rows:
        m, s, _ = time_repeat(fn, reps=reps, warmup=1, use_median=True)
        rows.append([name, api, f"{m:.2f}", f"{s:.2f}", ""]) 

    print("\n== MEF cutouts (", cutouts, " x ", cut_hw, "x", cut_hw, ") ==", sep="")
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "mef_cutouts",
                "size": size,
                "hdus": hdus,
                "cutouts": cutouts,
                "cut_hw": cut_hw,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_multifile_cutouts(tmp: str, files: int, cutouts: int, size: int, cut_hw: int, reps: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    paths = []
    for i in range(files):
        p = os.path.join(tmp, f"img_{i}.fits")
        _make_image(p, (size, size))
        paths.append(p)

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # Prepare random coordinates across files
    picks: List[Tuple[str, int, int]] = []
    for _ in range(cutouts):
        path = random.choice(paths)
        y = random.randint(0, size - cut_hw)
        x = random.randint(0, size - cut_hw)
        picks.append((path, y, x))

    def _tf_with(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        def _run():
            for (p, y, x) in picks:
                _ = tf.read(p, start=[y, x], shape=[cut_hw, cut_hw], **_tf_opts(mmap_flag, cache_capacity, buffered_flag))[0]
        return _run

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, label in _tf_auto_candidates_size_aware(op="multifile_cutouts", allow_buffered_default=True, cut_hw=cut_hw, files=files):
            try:
                m, s, _ = time_repeat(_tf_with(mm, buf), reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or m < best[0]:
                best = (m, s, label)
        if best is None:
            m, s, _ = time_repeat(_tf_with(False, None), reps=reps, warmup=1, use_median=True)
            best = (m, s, "mmap=off")
        rows.append(["torchfits", f"read(start,shape) ({best[2]})", f"{best[0]:.2f}", f"{best[1]:.2f}", f"{cutouts}x{cut_hw}^2 @ {files} files"])
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        m, s, _ = time_repeat(_tf_with(mm_flag, None), reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", f"read(start,shape) (mmap={'auto' if mm_flag is None else 'on' if mm_flag else 'off'})", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x{cut_hw}^2 @ {files} files"])

    comp_rows: list[tuple[str, str, Callable[[], None]]] = []
    if astropy is not None:
        def _ap():
            for (p, y, x) in picks:
                with astropy.open(p) as hdul:  # type: ignore[attr-defined]
                    data = hdul[0].data
                    sl = data[y:y+cut_hw, x:x+cut_hw]
                    force_consume(sl)
        comp_rows.append(("astropy", "slice per file", _ap))
    else:
        rows.append(["astropy", "slice per file", "n/a", "", "missing module"]) 

    if fitsio is not None:
        def _fi():
            for (p, y, x) in picks:
                with fitsio.FITS(p) as f:  # type: ignore[attr-defined]
                    data = f[0].read()
                    sl = data[y:y+cut_hw, x:x+cut_hw]
                    force_consume(sl)
        comp_rows.append(("fitsio", "slice per file", _fi))
    else:
        rows.append(["fitsio", "slice per file", "n/a", "", "missing module"]) 

    random.shuffle(comp_rows)
    for name, api, fn in comp_rows:
        m, s, _ = time_repeat(fn, reps=reps, warmup=1, use_median=True)
        rows.append([name, api, f"{m:.2f}", f"{s:.2f}", "open+slice each"]) 

    print("\n== Multi-file cutouts (", cutouts, " x ", cut_hw, "x", cut_hw, ") ==", sep="")
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "multifile_cutouts",
                "size": size,
                "files": files,
                "cutouts": cutouts,
                "cut_hw": cut_hw,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_table_frameworks(tmp: str, rows_n: int, reps: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    path = os.path.join(tmp, "tab_fw.fits")
    _make_table(path, rows=rows_n)

    ap_table = try_import("astropy.table")
    apfits = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "format", "mean ms", "stdev", "notes"]

    # torchfits native table
    def _tf_table(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        return tf.read(path, hdu=1, format="table", **_tf_opts(mmap_flag, cache_capacity, buffered_flag))

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, _label in _tf_auto_candidates(False):
            try:
                def _runner_tbl(mm: Optional[bool] = mm, buf: Optional[bool] = buf):
                    return _tf_table(mm, buf)
                mtry, stry, _ = time_repeat(_runner_tbl, reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or mtry < best[0]:
                best = (mtry, stry)
        if best is None:
            def _runner_tbl_off():
                return _tf_table(False, None)
            m, s, _ = time_repeat(_runner_tbl_off, reps=reps, warmup=1, use_median=True)
        else:
            m, s = best
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        def _runner_tbl_flag():
            return _tf_table(mm_flag, None)
        m, s, _ = time_repeat(_runner_tbl_flag, reps=reps, warmup=1, use_median=True)
    rows.append(["torchfits", "table", f"{m:.2f}", f"{s:.2f}", "native table"])

    # torchfits torch-frame
    try:
        def _tf_df(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
            return tf.read(path, hdu=1, format="dataframe", **_tf_opts(mmap_flag, cache_capacity, buffered_flag))
        if tf_mmap_mode == "auto":
            bestdf = None
            for mm, buf, _label in _tf_auto_candidates(False):
                try:
                    def _runner_df(mm: Optional[bool] = mm, buf: Optional[bool] = buf):
                        return _tf_df(mm, buf)
                    dm, ds, _ = time_repeat(_runner_df, reps=reps, warmup=1, use_median=True)
                except Exception:
                    continue
                if bestdf is None or dm < bestdf[0]:
                    bestdf = (dm, ds)
            if bestdf is None:
                def _runner_df_off():
                    return _tf_df(False, None)
                m2, s2, _ = time_repeat(_runner_df_off, reps=reps, warmup=1, use_median=True)
            else:
                m2, s2 = bestdf
        else:
            mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
            def _runner_df_flag():
                return _tf_df(mm_flag, None)
            m2, s2, _ = time_repeat(_runner_df_flag, reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", "dataframe", f"{m2:.2f}", f"{s2:.2f}", "torch-frame"])
    except Exception:
        rows.append(["torchfits", "dataframe", "n/a", "", "torch-frame missing"])

    # astropy.table
    if ap_table is not None:
        def _ap():
            tbl = ap_table.Table.read(path, format="fits")  # type: ignore[attr-defined]
            # force materialization by touching numeric columns
            try:
                for name in tbl.colnames:
                    col = tbl[name]
                    dt = getattr(getattr(col, 'dtype', None), 'kind', None)
                    if dt in ('i', 'u', 'f', 'b'):
                        _ = float(np.asarray(col).sum())
            except Exception:
                pass
            return tbl
        m, s, _ = time_repeat(_ap, reps=reps, warmup=1, use_median=True)
        rows.append(["astropy", "Table.read", f"{m:.2f}", f"{s:.2f}", "fits format"])
    else:
        rows.append(["astropy", "Table.read", "n/a", "", "missing module"])

    # fitsio table -> numpy structured array
    if fitsio is not None:
        def _fi():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                arr = f[1].read()
                force_consume(arr)
                return arr
        m, s, _ = time_repeat(_fi, reps=reps, warmup=1, use_median=True)
        rows.append(["fitsio", "FITS[1].read", f"{m:.2f}", f"{s:.2f}", "structured numpy"]) 
    else:
        rows.append(["fitsio", "FITS[1].read", "n/a", "", "missing module"]) 

    print(f"\n== Table frameworks (rows={rows_n}) ==")
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "table_frameworks",
                "rows": rows_n,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_sky_cutouts(tmp: str, size: int, cutouts: int, radius_arcsec: float, reps: int, collector: list | None = None, tf_mmap_mode: str = "auto", cache_capacity: int = 0) -> None:
    path = os.path.join(tmp, "wcs_img.fits")
    created = _make_wcs_image(path, size=size)
    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")
    if not created:
        print("\n== Sky cutouts (skipped: astropy missing for image make) ==")
        return

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # Use torchfits WCS utilities (wcslib via C++; no astropy.wcs in torchfits path)
    hdr = tf.get_header(path, hdu=0)
    # define random sky positions within ~center region
    # center sky coordinate from pixel center
    world_center, _ = tf.pixel_to_world([[size/2, size/2]], hdr)
    ra0, dec0 = float(world_center[0][0]), float(world_center[0][1])
    # compute pixels per deg from CDELT2 (deg/pix)
    try:
        cdelt2 = float(hdr.get('CDELT2', 1.0))
    except Exception:
        cdelt2 = 1.0
    pix_per_deg = 1.0 / abs(cdelt2)
    half_hw = int(max(1, round((radius_arcsec / 3600.0) * pix_per_deg)))
    # sample positions within a small box around center
    rng = np.random.default_rng(0)
    sky_points = []
    for _ in range(cutouts):
        dra = (rng.random() - 0.5) * 0.05  # +/-0.025 deg
        ddec = (rng.random() - 0.5) * 0.05
        sky_points.append((ra0 + dra, dec0 + ddec))

    # torchfits path: sky->pixel via torchfits.wcs_utils (wcslib), then torchfits read(start,shape)
    # Precompute pixel coordinates once to avoid per-cutout WCS overhead
    pix_points, _ = tf.world_to_pixel([[ra, dec] for (ra, dec) in sky_points], hdr)

    def _tf_with(mmap_flag: Optional[bool], buffered_flag: Optional[bool]):
        def _run():
            for i in range(len(sky_points)):
                x, y = float(pix_points[i][0]), float(pix_points[i][1])
                ys = max(0, int(round(y)) - half_hw)
                xs = max(0, int(round(x)) - half_hw)
                _ = tf.read(path, start=[ys, xs], shape=[2*half_hw, 2*half_hw], **_tf_opts(mmap_flag, cache_capacity, buffered_flag))[0]
        return _run

    if tf_mmap_mode == "auto":
        best = None
        for mm, buf, label in _tf_auto_candidates_size_aware(op="sky_cutouts", allow_buffered_default=True, half_hw=half_hw):
            try:
                m, s, _ = time_repeat(_tf_with(mm, buf), reps=reps, warmup=1, use_median=True)
            except Exception:
                continue
            if best is None or m < best[0]:
                best = (m, s, label)
        if best is None:
            m, s, _ = time_repeat(_tf_with(False, None), reps=reps, warmup=1, use_median=True)
            best = (m, s, "mmap=off")
        rows.append(["torchfits", f"WCS->read(start,shape) ({best[2]})", f"{best[0]:.2f}", f"{best[1]:.2f}", f"{cutouts}x({2*half_hw})^2 ~{radius_arcsec}" ])
    else:
        mm_flag = True if tf_mmap_mode == "true" else False if tf_mmap_mode == "false" else None
        m, s, _ = time_repeat(_tf_with(mm_flag, None), reps=reps, warmup=1, use_median=True)
        rows.append(["torchfits", f"WCS->read(start,shape) (mmap={'auto' if mm_flag is None else 'on' if mm_flag else 'off'})", f"{m:.2f}", f"{s:.2f}", f"{cutouts}x({2*half_hw})^2 ~{radius_arcsec}" ])

    # astropy slice path (WCS conversion still via torchfits so transforms are consistent)
    comp_rows: list[tuple[str, str, Callable[[], None]]] = []
    if astropy is not None:
        def _ap():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                data = hdul[0].data
                for i in range(len(sky_points)):
                    x, y = float(pix_points[i][0]), float(pix_points[i][1])
                    ys = max(0, int(round(y)) - half_hw)
                    xs = max(0, int(round(x)) - half_hw)
                    sl = data[ys:ys+2*half_hw, xs:xs+2*half_hw]
                    force_consume(sl)
        comp_rows.append(("astropy", "WCS->slice", _ap))
    else:
        rows.append(["astropy", "WCS->slice", "n/a", "", "missing module"]) 

    # fitsio slice path
    if fitsio is not None:
        def _fi():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                data = f[0].read()
                for i in range(len(sky_points)):
                    x, y = float(pix_points[i][0]), float(pix_points[i][1])
                    ys = max(0, int(round(y)) - half_hw)
                    xs = max(0, int(round(x)) - half_hw)
                    sl = data[ys:ys+2*half_hw, xs:xs+2*half_hw]
                    force_consume(sl)
        comp_rows.append(("fitsio", "WCS->slice", _fi))
    else:
        rows.append(["fitsio", "WCS->slice", "n/a", "", "missing module"]) 

    random.shuffle(comp_rows)
    for name, api, fn in comp_rows:
        m, s, _ = time_repeat(fn, reps=reps, warmup=1, use_median=True)
        rows.append([name, api, f"{m:.2f}", f"{s:.2f}", ""]) 

    print("\n== Sky-position cutouts (radius arcsec=", radius_arcsec, ") ==", sep="")
    rows_disp = rows[:]
    random.shuffle(rows_disp)
    print(format_table(rows_disp, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "sky_cutouts",
                "size": size,
                "radius_arcsec": radius_arcsec,
                "cutouts": cutouts,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_write_image(tmp: str, size: int, reps: int, collector: list | None = None) -> None:
    path_base = os.path.join(tmp, "write_img")
    os.makedirs(tmp, exist_ok=True)
    data_np = (np.random.rand(size, size).astype(np.float32) * 1000).astype(np.float32)
    data_t = torch.from_numpy(data_np)

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # torchfits write (torch tensor)
    def _tf_write():
        p = f"{path_base}_tf.fits"
        tf.write(p, data_t, overwrite=True)
        try:
            os.remove(p)
        except Exception:
            pass
    m, s, _ = time_repeat(_tf_write, reps=reps)
    rows.append(["torchfits", "write(tensor)", f"{m:.2f}", f"{s:.2f}", "torch->fits"])

    # astropy write (numpy)
    if astropy is not None:
        def _ap_write():
            p = f"{path_base}_ap.fits"
            hdu = astropy.PrimaryHDU(data_np)  # type: ignore[attr-defined]
            hdu.writeto(p, overwrite=True)
            try:
                os.remove(p)
            except Exception:
                pass
        m, s, _ = time_repeat(_ap_write, reps=reps)
        rows.append(["astropy", "PrimaryHDU.writeto", f"{m:.2f}", f"{s:.2f}", "numpy->fits"])
    else:
        rows.append(["astropy", "PrimaryHDU.writeto", "n/a", "", "missing module"])

    # fitsio write (numpy)
    if fitsio is not None:
        def _fi_write():
            p = f"{path_base}_fi.fits"
            fitsio.write(p, data_np, clobber=True)  # type: ignore[attr-defined]
            try:
                os.remove(p)
            except Exception:
                pass
        m, s, _ = time_repeat(_fi_write, reps=reps)
        rows.append(["fitsio", "write", f"{m:.2f}", f"{s:.2f}", "numpy->fits"])
    else:
        rows.append(["fitsio", "write", "n/a", "", "missing module"])

    print("\n== Image write (", size, "x", size, ") ==", sep="")
    print(format_table(rows, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "image_write",
                "size": size,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def bench_write_table_numeric(tmp: str, rows_n: int, reps: int, collector: list | None = None) -> None:
    path_base = os.path.join(tmp, "write_tab")
    rng = np.random.default_rng(0)
    # numeric-only columns for cross-lib parity
    t_data_np = {
        "I32": rng.integers(0, 10_000, size=rows_n, dtype=np.int32),
        "I64": rng.integers(0, 10_000, size=rows_n, dtype=np.int64),
        "F32": rng.standard_normal(rows_n).astype(np.float32),
        "F64": rng.standard_normal(rows_n).astype(np.float64),
    }
    t_data_t = {k: torch.from_numpy(v) for k, v in t_data_np.items()}

    ap_table = try_import("astropy.table")
    fitsio = try_import("fitsio")

    rows = []
    headers = ["Impl", "API", "mean ms", "stdev", "notes"]

    # torchfits write_table
    def _tf_wt():
        p = f"{path_base}_tf.fits"
        tf.write_table(p, t_data_t, overwrite=True)
        try:
            os.remove(p)
        except Exception:
            pass
    m, s, _ = time_repeat(_tf_wt, reps=reps)
    rows.append(["torchfits", "write_table(tensors)", f"{m:.2f}", f"{s:.2f}", "numeric cols"])

    # astropy table write
    if ap_table is not None:
        def _ap_wt():
            p = f"{path_base}_ap.fits"
            tbl = ap_table.Table(t_data_np)  # type: ignore[attr-defined]
            tbl.write(p, overwrite=True, format="fits")
            try:
                os.remove(p)
            except Exception:
                pass
        m, s, _ = time_repeat(_ap_wt, reps=reps)
        rows.append(["astropy", "Table.write", f"{m:.2f}", f"{s:.2f}", "numeric cols"])
    else:
        rows.append(["astropy", "Table.write", "n/a", "", "missing module"])

    # fitsio table write using recarray
    if fitsio is not None:
        def _fi_wt():
            p = f"{path_base}_fi.fits"
            names = list(t_data_np.keys())
            dtype = np.dtype([(n, t_data_np[n].dtype.str) for n in names])
            arr = np.rec.fromarrays([t_data_np[n] for n in names], dtype=dtype)
            fitsio.write(p, arr, clobber=True)  # type: ignore[attr-defined]
            try:
                os.remove(p)
            except Exception:
                pass
        m, s, _ = time_repeat(_fi_wt, reps=reps)
        rows.append(["fitsio", "write(recarray)", f"{m:.2f}", f"{s:.2f}", "numeric cols"])
    else:
        rows.append(["fitsio", "write(recarray)", "n/a", "", "missing module"])

    print(f"\n== Table write (rows={rows_n}, numeric-only) ==")
    print(format_table(rows, headers=headers))
    if collector is not None:
        for r in rows:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "table_write_numeric",
                "rows": rows_n,
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })

    


def bench_table(tmp: str, rows_n: int, reps: int, collector: list | None = None) -> None:
    path = os.path.join(tmp, "tab.fits")
    _make_table(path, rows=rows_n)

    astropy = try_import("astropy.io.fits")
    fitsio = try_import("fitsio")

    # columns subset to stress selection, includes dtypes and string
    cols = ["I32", "F32", "F64", "S"]

    rpt: List[List[str]] = []
    headers = ["Impl", "API", "mean ms", "stdev", "Notes"]

    # torchfits -> dict of tensors
    m, s, _ = time_repeat(lambda: tf.read(path, hdu=1, columns=cols, format="tensor"), reps=reps, warmup=1, use_median=True)
    rpt.append(["torchfits", "read(columns)->tensor", f"{m:.2f}", f"{s:.2f}", "tensor dict"])

    # torchfits -> dataframe (if available)
    try:
        m2, s2, _ = time_repeat(lambda: tf.read(path, hdu=1, columns=cols, format="dataframe"), reps=reps, warmup=1, use_median=True)
        rpt.append(["torchfits", "read(columns)->dataframe", f"{m2:.2f}", f"{s2:.2f}", "torch-frame"])
    except Exception:
        rpt.append(["torchfits", "read(columns)->dataframe", "n/a", "", "torch-frame missing"])

    # astropy -> numpy -> torch (selected columns)
    if astropy is not None:
        def _ap_tab():
            with astropy.open(path) as hdul:  # type: ignore[attr-defined]
                t = hdul[1].data
                out = {}
                for c in cols:
                    arr = t[c]
                    if getattr(arr, 'dtype', None) is not None and arr.dtype.kind in ('i', 'u', 'f', 'b'):
                        out[c] = numpy_to_torch(arr)
                return out

        m, s, _ = time_repeat(_ap_tab, reps=reps, warmup=1, use_median=True)
        rpt.append(["astropy", "numpy->torch (cols)", f"{m:.2f}", f"{s:.2f}", "numeric only"])
    else:
        rpt.append(["astropy", "numpy->torch (cols)", "n/a", "", "missing module"])

    # fitsio -> numpy -> torch (selected columns)
    if fitsio is not None:
        def _fi_tab():
            with fitsio.FITS(path) as f:  # type: ignore[attr-defined]
                t = f[1]
                out = {}
                for c in cols:
                    arr = t[c][:]
                    if getattr(arr, 'dtype', None) is not None and arr.dtype.kind in ('i', 'u', 'f', 'b'):
                        out[c] = numpy_to_torch(arr)
                return out

        m, s, _ = time_repeat(_fi_tab, reps=reps, warmup=1, use_median=True)
        rpt.append(["fitsio", "numpy->torch (cols)", f"{m:.2f}", f"{s:.2f}", "numeric only"])
    else:
        rpt.append(["fitsio", "numpy->torch (cols)", "n/a", "", "missing module"])

    print(f"\n== Table column subset (rows={rows_n}, cols={len(cols)}) ==")
    rpt_disp = rpt[:]
    random.shuffle(rpt_disp)
    print(format_table(rpt_disp, headers=headers))
    if collector is not None:
        for r in rpt:
            if r[2] == 'n/a':
                continue
            collector.append({
                "scenario": "table_cols",
                "rows": rows_n,
                "cols": len(cols),
                "impl": r[0],
                "api": r[1],
                "mean_ms": float(r[2]),
                "stdev_ms": float(r[3]) if r[3] else None,
                "notes": r[4],
                "reps": reps,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024, help="image ax size")
    ap.add_argument("--cutouts", type=int, default=10, help="# random cutouts")
    ap.add_argument("--cutout-size", type=int, default=64, help="square cutout hw")
    ap.add_argument("--table-rows", type=int, default=200_000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mef-hdus", type=int, default=3, help="# image HDUs in MEF for MEF benchmark")
    ap.add_argument("--files", type=int, default=3, help="# of files for multi-file cutouts benchmark")
    ap.add_argument("--sky-cutouts", type=int, default=10, help="# of sky-position cutouts")
    ap.add_argument("--sky-radius-arcsec", type=float, default=30.0, help="radius in arcsec for sky cutouts")
    ap.add_argument("--jsonl", type=str, default=None, help="Path to append JSONL results")
    ap.add_argument("--tf-mmap", type=str, choices=["auto", "true", "false"], default="auto", help="torchfits mmap mode for reads")
    ap.add_argument("--cache-mb", type=int, default=0, help="torchfits read cache capacity in MB")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        results: list = []
        # Insert a meta record first (sys/lib versions) for this run
        results.append(_collect_meta_record())
        bench_image_full(td, args.size, args.reps, results, args.tf_mmap, args.cache_mb)
        bench_cutouts(td, args.size, args.cutouts, args.cutout_size, args.reps, results, args.tf_mmap, args.cache_mb)
        bench_table(td, args.table_rows, args.reps, results)
        bench_mef_cutouts(td, args.mef_hdus, args.cutouts, args.cutout_size, args.reps, args.size, results, args.tf_mmap, args.cache_mb)
        bench_multifile_cutouts(td, args.files, args.cutouts, args.size, args.cutout_size, args.reps, results, args.tf_mmap, args.cache_mb)
        bench_table_frameworks(td, args.table_rows, args.reps, results, args.tf_mmap, args.cache_mb)
        bench_sky_cutouts(td, args.size, args.sky_cutouts, args.sky_radius_arcsec, args.reps, results, args.tf_mmap, args.cache_mb)
        # write benchmarks
        bench_write_image(td, args.size, args.reps, results)
        bench_write_table_numeric(td, args.table_rows, args.reps, results)
        if args.jsonl:
            os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
            with open(args.jsonl, 'a') as f:
                for rec in results:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
