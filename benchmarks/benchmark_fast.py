#!/usr/bin/env python3
"""
Focused benchmark suite for fast iteration on I/O changes.

Targets a small set of representative files to reduce turnaround time.
"""

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fitsio
from astropy.io import fits as astropy_fits
from astropy.io.fits import CompImageHDU

import torchfits


def _generate_data(shape, dtype):
    return (np.random.randn(*shape) * 100).astype(dtype)


def _write_primary_image(path: Path, data: np.ndarray):
    astropy_fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _create_scaled_file(path: Path, shape):
    float_data = np.random.randn(*shape).astype(np.float32) * 1000 + 32768
    hdu = astropy_fits.PrimaryHDU()
    hdu.data = float_data.astype(np.int16)
    hdu.header["BSCALE"] = 0.1
    hdu.header["BZERO"] = 32768
    hdu.header["COMMENT"] = "Scaled data test"
    hdu.writeto(path, overwrite=True)


def _create_wcs_file(path: Path, shape=(512, 512)):
    data = _generate_data(shape, np.float32)
    hdu = astropy_fits.PrimaryHDU(data)
    hdu.header["CRPIX1"] = shape[1] / 2
    hdu.header["CRPIX2"] = shape[0] / 2
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT1"] = -0.0001
    hdu.header["CDELT2"] = 0.0001
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CUNIT2"] = "deg"
    hdu.writeto(path, overwrite=True)


def _create_compressed_file(path: Path, compression_type: str, shape=(1024, 1024), tile_size=None):
    data = _generate_data(shape, np.float32)
    if tile_size is not None:
        comp_hdu = CompImageHDU(data, compression_type=compression_type, tile_shape=tile_size)
    else:
        comp_hdu = CompImageHDU(data, compression_type=compression_type)
    hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), comp_hdu])
    hdul.writeto(path, overwrite=True)


def _create_timeseries_files(dir_path: Path, shape=(256, 256), count=5, key_prefix="timeseries_frame"):
    files = {}
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        data = _generate_data(shape, np.float32) + i * 100
        filename = dir_path / f"frame_{i:03d}.fits"
        astropy_fits.PrimaryHDU(data).writeto(filename, overwrite=True)
        files[f"{key_prefix}_{i:03d}"] = filename
    return files


def _create_mef_file(path: Path, shape=(256, 256), count=10):
    hdus = [astropy_fits.PrimaryHDU(_generate_data(shape, np.float32))]
    for i in range(count):
        data = _generate_data(shape, np.float32) + i
        hdus.append(astropy_fits.ImageHDU(data))
    astropy_fits.HDUList(hdus).writeto(path, overwrite=True)


def _time_method(fn, warmup, iterations, sync_fn=None):
    for _ in range(warmup):
        fn()
        if sync_fn is not None:
            sync_fn()
    times = []
    for _ in range(iterations):
        if sync_fn is not None:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn is not None:
            sync_fn()
        times.append(time.perf_counter() - t0)
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


def _time_method_repeat(fn, warmup, iterations, repeats, sync_fn=None):
    means = []
    stds = []
    for _ in range(repeats):
        m, s = _time_method(fn, warmup, iterations, sync_fn=sync_fn)
        means.append(m)
        stds.append(s)
    means_sorted = sorted(means)
    median = means_sorted[len(means_sorted) // 2]
    return median, mean(stds)


class _TorchThreadGuard:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.prev_threads = None

    def __enter__(self):
        if hasattr(torch, "get_num_threads"):
            self.prev_threads = torch.get_num_threads()
            torch.set_num_threads(self.num_threads)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.prev_threads is not None:
            torch.set_num_threads(self.prev_threads)
        return False


def build_dataset(data_dir: Path):
    files = {}

    # Tiny set
    files["tiny_int8_1d"] = data_dir / "tiny_int8_1d.fits"
    _write_primary_image(files["tiny_int8_1d"], _generate_data((1000,), np.int8))

    files["tiny_int16_1d"] = data_dir / "tiny_int16_1d.fits"
    _write_primary_image(files["tiny_int16_1d"], _generate_data((1000,), np.int16))

    files["tiny_int32_1d"] = data_dir / "tiny_int32_1d.fits"
    _write_primary_image(files["tiny_int32_1d"], _generate_data((1000,), np.int32))

    files["tiny_float64_2d"] = data_dir / "tiny_float64_2d.fits"
    _write_primary_image(files["tiny_float64_2d"], _generate_data((64, 64), np.float64))

    # Small/medium/large
    files["small_int32_3d"] = data_dir / "small_int32_3d.fits"
    _write_primary_image(files["small_int32_3d"], _generate_data((10, 128, 128), np.int32))

    files["medium_int16_2d"] = data_dir / "medium_int16_2d.fits"
    _write_primary_image(files["medium_int16_2d"], _generate_data((1024, 1024), np.int16))

    files["medium_int32_3d"] = data_dir / "medium_int32_3d.fits"
    _write_primary_image(files["medium_int32_3d"], _generate_data((25, 256, 256), np.int32))

    files["medium_float32_3d"] = data_dir / "medium_float32_3d.fits"
    _write_primary_image(files["medium_float32_3d"], _generate_data((25, 256, 256), np.float32))

    files["large_int32_1d"] = data_dir / "large_int32_1d.fits"
    _write_primary_image(files["large_int32_1d"], _generate_data((1000000,), np.int32))

    # Scaled
    files["scaled_small"] = data_dir / "scaled_small.fits"
    _create_scaled_file(files["scaled_small"], (256, 256))
    files["scaled_medium"] = data_dir / "scaled_medium.fits"
    _create_scaled_file(files["scaled_medium"], (1024, 1024))

    # WCS
    files["wcs_image"] = data_dir / "wcs_image.fits"
    _create_wcs_file(files["wcs_image"])

    # Compressed
    files["compressed_rice_1"] = data_dir / "compressed_rice_1.fits"
    _create_compressed_file(files["compressed_rice_1"], "RICE_1")
    files["compressed_hcompress_1"] = data_dir / "compressed_hcompress_1.fits"
    _create_compressed_file(files["compressed_hcompress_1"], "HCOMPRESS_1")
    files["compressed_rice_1_large"] = data_dir / "compressed_rice_1_large.fits"
    _create_compressed_file(files["compressed_rice_1_large"], "RICE_1", shape=(4096, 4096), tile_size=(1024, 1024))
    files["compressed_hcompress_1_large"] = data_dir / "compressed_hcompress_1_large.fits"
    _create_compressed_file(files["compressed_hcompress_1_large"], "HCOMPRESS_1", shape=(4096, 4096), tile_size=(1024, 1024))

    # Timeseries
    files.update(_create_timeseries_files(data_dir / "timeseries"))
    files.update(_create_timeseries_files(data_dir / "timeseries_long", shape=(64, 64), count=32, key_prefix="timeseries_long"))

    # Multi-MEF
    files["multi_mef_10ext"] = data_dir / "multi_mef_10ext.fits"
    _create_mef_file(files["multi_mef_10ext"], shape=(256, 256), count=10)

    return files


def main():
    parser = argparse.ArgumentParser(description="Focused fast benchmark suite")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set PyTorch intra-op threads for CPU benchmarks (0 keeps current setting).",
    )
    parser.add_argument(
        "--compressed-repeats",
        type=int,
        default=1,
        help="Override repeats for compressed cases only (1 disables override).",
    )
    args = parser.parse_args()

    np.random.seed(0)
    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        print(f"Using torch threads: {args.torch_threads}")

    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = Path(tempfile.mkdtemp(prefix="torchfits_bench_fast_"))

    files = build_dataset(data_dir)

    device = args.device
    device_available = True
    if device == "cuda":
        device_available = torch.cuda.is_available()
    elif device == "mps":
        device_available = torch.backends.mps.is_available()

    def sync_device():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    results = []
    for name, path in files.items():
        if name.startswith("timeseries_long_") or name.endswith("_large"):
            continue
        hdu = 1 if name.startswith("compressed_") else 0
        print(f"Benchmarking {name} (hdu={hdu})")

        def torchfits_read():
            return torchfits.read(
                str(path),
                hdu=hdu,
                return_header=False,
                cache_capacity=10,
                scale_on_device=True,
            )

        def torchfits_read_device():
            return torchfits.read(
                str(path),
                hdu=hdu,
                return_header=False,
                cache_capacity=10,
                device=device,
                scale_on_device=True,
            )

        def torchfits_cpp():
            return torchfits.cpp.read_full(str(path), hdu, True)

        def fitsio_read():
            return fitsio.read(str(path), ext=hdu)

        def fitsio_torch():
            return torch.from_numpy(fitsio.read(str(path), ext=hdu))

        def fitsio_torch_device():
            return torch.from_numpy(fitsio.read(str(path), ext=hdu)).to(device)

        methods = {
            "torchfits": torchfits_read,
            "fitsio": fitsio_read,
            "fitsio_torch": fitsio_torch,
        }
        if hdu == 0:
            methods["torchfits_cpp"] = torchfits_cpp
        if device != "cpu" and device_available:
            methods["torchfits_device"] = torchfits_read_device
            methods["fitsio_torch_device"] = fitsio_torch_device

        if name.startswith("scaled_"):
            def torchfits_raw():
                return torchfits.read(str(path), hdu=hdu, return_header=False, raw_scale=True, cache_capacity=10)
            methods["torchfits_raw"] = torchfits_raw

            header = astropy_fits.getheader(str(path))
            bscale = float(header.get("BSCALE", 1.0))
            bzero = float(header.get("BZERO", 0.0))

            def torchfits_raw_scale_cpu():
                data = torchfits.read(str(path), hdu=hdu, return_header=False, raw_scale=True, cache_capacity=10)
                data = data.to(torch.float32)
                if bscale != 1.0:
                    data.mul_(bscale)
                if bzero != 0.0:
                    data.add_(bzero)
                return data

            methods["torchfits_raw_scale_cpu"] = torchfits_raw_scale_cpu

        row = {"name": name, "path": str(path), "hdu": hdu}
        iter_count = args.iterations
        warmup_count = args.warmup
        if name.startswith("compressed_"):
            iter_count = max(iter_count, 50)
            warmup_count = max(warmup_count, 10)
        for mname, fn in methods.items():
            try:
                sync = sync_device if mname.endswith("_device") else None
                repeat_count = args.repeats
                if name.startswith("compressed_") and args.compressed_repeats > 1:
                    repeat_count = args.compressed_repeats
                if repeat_count > 1:
                    m, s = _time_method_repeat(fn, warmup_count, iter_count, repeat_count, sync_fn=sync)
                else:
                    m, s = _time_method(fn, warmup_count, iter_count, sync_fn=sync)
                row[f"{mname}_mean"] = m
                row[f"{mname}_std"] = s
                print(f"  {mname:14s}: {m:.6f}s ± {s:.6f}s")
            except Exception as e:
                row[f"{mname}_mean"] = None
                row[f"{mname}_std"] = None
                print(f"  {mname:14s}: FAILED ({e})")
        results.append(row)

    # Summary
    def safe(v):
        return v if isinstance(v, (int, float)) else math.inf

    print("\nSummary (torchfits best vs comparable best):")
    for row in sorted(results, key=lambda r: safe(r.get("torchfits_mean"))):
        torchfits_candidates = [
            row.get("torchfits_mean"),
            row.get("torchfits_cpp_mean"),
        ]
        torchfits_best = min(
            [v for v in torchfits_candidates if isinstance(v, (int, float))],
            default=None,
        )
        comparable = [
            v
            for k, v in row.items()
            if k.endswith("_mean")
            and isinstance(v, (int, float))
            and "raw" not in k
            and "device" not in k
        ]
        best = min(comparable, default=None)
        ratio = (torchfits_best / best) if (torchfits_best is not None and best is not None and best > 0) else None
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
        tf_str = f"{torchfits_best:.6f}s" if torchfits_best is not None else "n/a"
        best_str = f"{best:.6f}s" if best is not None else "n/a"
        print(f"{row['name']:24s} torchfits_best={tf_str} best={best_str} ratio={ratio_str}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            headers = [
                "name",
                "path",
                "hdu",
                "torchfits_mean",
                "torchfits_std",
                "torchfits_cpp_mean",
                "torchfits_cpp_std",
                "torchfits_raw_mean",
                "torchfits_raw_std",
                "torchfits_raw_scale_cpu_mean",
                "torchfits_raw_scale_cpu_std",
                "torchfits_device_mean",
                "torchfits_device_std",
                "fitsio_mean",
                "fitsio_std",
                "fitsio_torch_mean",
                "fitsio_torch_std",
                "fitsio_torch_device_mean",
                "fitsio_torch_device_std",
            ]
            f.write(",".join(headers) + "\n")
            for row in results:
                values = [str(row.get(h, "")) for h in headers]
                f.write(",".join(values) + "\n")

    # Microbenches
    print("\nMicrobench: scaled CPU path (scaled_medium)")
    scaled_path = files.get("scaled_medium")
    if scaled_path:
        header = astropy_fits.getheader(str(scaled_path))
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))

        def tf_scaled():
            return torchfits.read(
                str(scaled_path),
                hdu=0,
                return_header=False,
                cache_capacity=10,
                scale_on_device=True,
            )

        def tf_raw():
            return torchfits.read(str(scaled_path), hdu=0, return_header=False, raw_scale=True, cache_capacity=10)

        def tf_raw_scale():
            data = torchfits.read(str(scaled_path), hdu=0, return_header=False, raw_scale=True, cache_capacity=10)
            data = data.to(torch.float32)
            if bscale != 1.0:
                data.mul_(bscale)
            if bzero != 0.0:
                data.add_(bzero)
            return data

        tests = [("torchfits_scaled", tf_scaled)]
        if hasattr(torchfits, "_read_scaled_cpu_fast"):
            tests.append(
                (
                    "torchfits_scaled_fast",
                    lambda: torchfits._read_scaled_cpu_fast(str(scaled_path)),
                )
            )
        tests.extend(
            [
                ("torchfits_raw", tf_raw),
                ("torchfits_raw_scale_cpu", tf_raw_scale),
            ]
        )

        for label, fn in tests:
            if args.repeats > 1:
                m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
            else:
                m, s = _time_method(fn, warmup_count, iter_count)
            print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")

    print("\nMicrobench: compressed_rice_1 (tile read)")
    comp_path = files.get("compressed_rice_1")
    if comp_path:
        def tf_compressed():
            return torchfits.read(str(comp_path), hdu=1, return_header=False, cache_capacity=0)

        def fitsio_compressed():
            return fitsio.read(str(comp_path), ext=1)

        for label, fn in [
            ("torchfits", tf_compressed),
            ("fitsio", fitsio_compressed),
        ]:
            if args.repeats > 1:
                m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
            else:
                m, s = _time_method(fn, warmup_count, iter_count)
            print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")

        print("\nMicrobench: compressed_rice_1 (single-thread)")
        with _TorchThreadGuard(1):
            for label, fn in [
                ("torchfits", tf_compressed),
                ("fitsio", fitsio_compressed),
            ]:
                if args.repeats > 1:
                    m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
                else:
                    m, s = _time_method(fn, warmup_count, iter_count)
                print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")

        print("\nMicrobench: compressed_rice_1 (open-once, read-many)")
        try:
            file_handle = torchfits.cpp.open_fits_file(str(comp_path), "r")
        except Exception as e:
            file_handle = None
            print(f"  torchfits_open_once: FAILED ({e})")
        if file_handle is not None:
            def tf_open_once():
                return torchfits.cpp.read_full(file_handle, 1, True)

            if args.repeats > 1:
                m, s = _time_method_repeat(tf_open_once, warmup_count, iter_count, args.repeats)
            else:
                m, s = _time_method(tf_open_once, warmup_count, iter_count)
            print(f"  torchfits_open_once: {m:.6f}s ± {s:.6f}s")
            try:
                torchfits.cpp.close_fits_file(file_handle)
            except Exception:
                pass

        print("\nMicrobench: compressed_rice_1 (cached handle)")
        def tf_compressed_cached():
            return torchfits.read(str(comp_path), hdu=1, return_header=False, cache_capacity=1)

        if args.repeats > 1:
            m, s = _time_method_repeat(tf_compressed_cached, warmup_count, iter_count, args.repeats)
        else:
            m, s = _time_method(tf_compressed_cached, warmup_count, iter_count)
        print(f"  torchfits_cached     : {m:.6f}s ± {s:.6f}s")

    print("\nMicrobench: compressed large tiles (full image)")
    large_warmup = 1
    large_iters = 3
    comp_large = [
        ("compressed_rice_1_large", files.get("compressed_rice_1_large")),
        ("compressed_hcompress_1_large", files.get("compressed_hcompress_1_large")),
    ]
    for name, path in comp_large:
        if not path:
            continue

        def tf_compressed_large(p=str(path)):
            return torchfits.read(p, hdu=1, return_header=False, cache_capacity=0)

        def fitsio_compressed_large(p=str(path)):
            return fitsio.read(p, ext=1)

        print(f"  {name}:")
        for label, fn in [
            ("torchfits", tf_compressed_large),
            ("fitsio", fitsio_compressed_large),
        ]:
            if args.repeats > 1:
                m, s = _time_method_repeat(fn, large_warmup, large_iters, args.repeats)
            else:
                m, s = _time_method(fn, large_warmup, large_iters)
            print(f"    {label:20s}: {m:.6f}s ± {s:.6f}s")

    print("\nMicrobench: compressed_hcompress_1 (single-thread)")
    comp_hc_path = files.get("compressed_hcompress_1")
    if comp_hc_path:
        def tf_compressed_hc():
            return torchfits.read(str(comp_hc_path), hdu=1, return_header=False, cache_capacity=0)

        def fitsio_compressed_hc():
            return fitsio.read(str(comp_hc_path), ext=1)

        with _TorchThreadGuard(1):
            for label, fn in [
                ("torchfits", tf_compressed_hc),
                ("fitsio", fitsio_compressed_hc),
            ]:
                if args.repeats > 1:
                    m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
                else:
                    m, s = _time_method(fn, warmup_count, iter_count)
                print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")

        print("\nMicrobench: compressed_hcompress_1 (open-once, read-many)")
        try:
            file_handle_hc = torchfits.cpp.open_fits_file(str(comp_hc_path), "r")
        except Exception as e:
            file_handle_hc = None
            print(f"  torchfits_open_once: FAILED ({e})")
        if file_handle_hc is not None:
            def tf_open_once_hc():
                return torchfits.cpp.read_full(file_handle_hc, 1, True)

            if args.repeats > 1:
                m, s = _time_method_repeat(tf_open_once_hc, warmup_count, iter_count, args.repeats)
            else:
                m, s = _time_method(tf_open_once_hc, warmup_count, iter_count)
            print(f"  torchfits_open_once: {m:.6f}s ± {s:.6f}s")
            try:
                torchfits.cpp.close_fits_file(file_handle_hc)
            except Exception:
                pass

        print("\nMicrobench: compressed_hcompress_1 (cached handle)")
        def tf_compressed_hc_cached():
            return torchfits.read(str(comp_hc_path), hdu=1, return_header=False, cache_capacity=1)

        if args.repeats > 1:
            m, s = _time_method_repeat(tf_compressed_hc_cached, warmup_count, iter_count, args.repeats)
        else:
            m, s = _time_method(tf_compressed_hc_cached, warmup_count, iter_count)
        print(f"  torchfits_cached     : {m:.6f}s ± {s:.6f}s")

    print("\nMicrobench: timeseries batch (single HDU across many files)")
    ts_paths = []
    for i in range(32):
        key = f"timeseries_long_{i:03d}"
        if key in files:
            ts_paths.append(str(files[key]))
    if ts_paths:
        def tf_loop():
            return [torchfits.read(p, hdu=0, return_header=False, cache_capacity=0) for p in ts_paths]

        def tf_batch():
            return torchfits.read(ts_paths, hdu=0, return_header=False, cache_capacity=0)

        def fitsio_loop():
            return [fitsio.read(p, ext=0) for p in ts_paths]

        for label, fn in [
            ("torchfits_loop", tf_loop),
            ("torchfits_batch", tf_batch),
            ("fitsio_loop", fitsio_loop),
        ]:
            if args.repeats > 1:
                m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
            else:
                m, s = _time_method(fn, warmup_count, iter_count)
            print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")

    print("\nMicrobench: multi-MEF batch (multiple HDUs from one file)")
    mef_path = files.get("multi_mef_10ext")
    if mef_path:
        try:
            with astropy_fits.open(str(mef_path)) as hdul:
                hdus = [
                    idx
                    for idx, hdu in enumerate(hdul)
                    if getattr(hdu, "is_image", False) and hdu.data is not None
                ]
        except Exception:
            hdus = list(range(11))

        def tf_loop_mef():
            return [torchfits.read(str(mef_path), hdu=h, return_header=False, cache_capacity=0) for h in hdus]

        def tf_batch_mef():
            return torchfits.read(str(mef_path), hdu=hdus, return_header=False, cache_capacity=0)

        def fitsio_loop_mef():
            return [fitsio.read(str(mef_path), ext=h) for h in hdus]

        for label, fn in [
            ("torchfits_loop", tf_loop_mef),
            ("torchfits_batch", tf_batch_mef),
            ("fitsio_loop", fitsio_loop_mef),
        ]:
            try:
                if args.repeats > 1:
                    m, s = _time_method_repeat(fn, warmup_count, iter_count, args.repeats)
                else:
                    m, s = _time_method(fn, warmup_count, iter_count)
                print(f"  {label:22s}: {m:.6f}s ± {s:.6f}s")
            except Exception as e:
                print(f"  {label:22s}: FAILED ({e})")


if __name__ == "__main__":
    main()
