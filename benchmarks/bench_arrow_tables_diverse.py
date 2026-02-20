#!/usr/bin/env python3
"""
Diverse Arrow-table benchmark suite.

This script targets different table shapes/workloads to diagnose where time is spent:
- numeric-only tables
- mixed tables with NUL-padded strings
- mixed tables with trailing-space strings
- wide vector-valued columns
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import tempfile
import threading
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from astropy.io import fits
from astropy.table import Table
import pyarrow as pa
import torch

import fitsio
import torchfits
import torchfits.cpp as torchfits_cpp

try:
    import psutil
except ImportError:  # pragma: no cover - optional at runtime
    psutil = None


def _time(fn, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


def _clear_torchfits_runtime_caches() -> None:
    """Best-effort cache clear between cold measurements."""
    try:
        torchfits.clear_file_cache()
    except Exception:
        pass
    try:
        torchfits.table._close_all_cached_handles()
    except Exception:
        pass


def _measure_method_once(
    fn: callable, sample_interval_s: float
) -> tuple[float, float, float]:
    """
    Return (elapsed_s, rss_delta_mb, rss_peak_delta_mb) for one method call.
    """
    if psutil is None:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        return elapsed, math.nan, math.nan

    process = psutil.Process()
    rss_before = process.memory_info().rss
    rss_peak = rss_before
    stop_event = threading.Event()

    def _sampler() -> None:
        nonlocal rss_peak
        while not stop_event.wait(sample_interval_s):
            try:
                rss_now = process.memory_info().rss
                if rss_now > rss_peak:
                    rss_peak = rss_now
            except Exception:
                return

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        elapsed = time.perf_counter() - t0
        stop_event.set()
        sampler_thread.join(timeout=0.1)

    rss_after = process.memory_info().rss
    if rss_after > rss_peak:
        rss_peak = rss_after

    to_mb = 1024.0 * 1024.0
    rss_delta_mb = (rss_after - rss_before) / to_mb
    rss_peak_delta_mb = (rss_peak - rss_before) / to_mb
    return elapsed, rss_delta_mb, rss_peak_delta_mb


def _profile_methods_round_robin(
    methods: list[tuple[str, callable]], warmup: int, iterations: int
) -> dict[str, dict[str, float]]:
    # Warm each method once (or more) before timed rounds.
    for _ in range(warmup):
        for _, fn in methods:
            fn()

    timings: dict[str, list[float]] = {name: [] for name, _ in methods}
    rss_deltas: dict[str, list[float]] = {name: [] for name, _ in methods}
    rss_peak_deltas: dict[str, list[float]] = {name: [] for name, _ in methods}
    n = len(methods)
    if n == 0:
        return {}

    sample_interval_s = 0.001
    for i in range(iterations):
        # Rotate start index each round to avoid fixed-order cache bias.
        for j in range(n):
            idx = (i + j) % n
            name, fn = methods[idx]
            elapsed_s, rss_delta_mb, rss_peak_delta_mb = _measure_method_once(
                fn, sample_interval_s=sample_interval_s
            )
            timings[name].append(elapsed_s)
            rss_deltas[name].append(rss_delta_mb)
            rss_peak_deltas[name].append(rss_peak_delta_mb)

    out: dict[str, dict[str, float]] = {}
    for name, vals in timings.items():
        deltas = rss_deltas[name]
        peaks = rss_peak_deltas[name]
        out[name] = {
            "mean_s": mean(vals),
            "std_s": stdev(vals) if len(vals) > 1 else 0.0,
            "mean_rss_delta_mb": mean(deltas),
            "std_rss_delta_mb": stdev(deltas) if len(deltas) > 1 else 0.0,
            "mean_peak_rss_delta_mb": mean(peaks),
            "std_peak_rss_delta_mb": stdev(peaks) if len(peaks) > 1 else 0.0,
            "samples": float(len(vals)),
        }
    return out


def _profile_methods_cold_once(
    methods: list[tuple[str, callable]],
) -> dict[str, dict[str, float]]:
    """
    Profile one cold-ish call per method with cache clears between calls.
    """
    out: dict[str, dict[str, float]] = {}
    sample_interval_s = 0.001
    for name, fn in methods:
        _clear_torchfits_runtime_caches()
        gc.collect()
        elapsed_s, rss_delta_mb, rss_peak_delta_mb = _measure_method_once(
            fn, sample_interval_s=sample_interval_s
        )
        out[name] = {
            "mean_s": elapsed_s,
            "std_s": 0.0,
            "mean_rss_delta_mb": rss_delta_mb,
            "std_rss_delta_mb": 0.0,
            "mean_peak_rss_delta_mb": rss_peak_delta_mb,
            "std_peak_rss_delta_mb": 0.0,
            "samples": 1.0,
        }
    return out


def _to_native_endian(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        return arr
    byteorder = arr.dtype.byteorder
    if byteorder in ("=", "|"):
        return arr
    if (byteorder == "<" and np.little_endian) or (
        byteorder == ">" and not np.little_endian
    ):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _fitsio_record_to_arrow(rec: np.ndarray):
    pydict: dict[str, object] = {}
    for name in rec.dtype.names or ():
        col = _to_native_endian(rec[name])
        if col.ndim <= 1:
            pydict[name] = pa.array(col)
        elif col.ndim == 2:
            flat = _to_native_endian(col.reshape(-1))
            values = pa.array(flat)
            pydict[name] = pa.FixedSizeListArray.from_arrays(values, int(col.shape[1]))
        else:
            pydict[name] = pa.array(col.tolist())
    return pa.table(pydict)


def _fitsio_record_to_torch(rec: np.ndarray):
    tensors: dict[str, object] = {}
    for name in rec.dtype.names or ():
        col = _to_native_endian(rec[name])
        if not isinstance(col, np.ndarray):
            continue
        if col.dtype.kind in {"U", "S"}:
            continue
        if col.dtype == np.object_:
            converted = []
            for item in col.tolist():
                if isinstance(item, np.ndarray):
                    arr = _to_native_endian(np.ascontiguousarray(item))
                    converted.append(torch.from_numpy(arr))
                else:
                    converted.append(item)
            tensors[name] = converted
            continue
        arr = np.ascontiguousarray(col)
        tensors[name] = torch.from_numpy(arr)
    return tensors


def _build_numeric_only(rows: int, rng: np.random.Generator):
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "MAG_G": rng.normal(20.0, 1.5, rows).astype(np.float32),
        "OBJID": rng.integers(0, 2**31 - 1, rows, dtype=np.int64),
        "FLAG": rng.integers(0, 2, rows, dtype=np.int16),
    }
    return data, [], {}, {}


def _build_mixed_null_strings(rows: int, rng: np.random.Generator):
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "OBJID": rng.integers(0, 2**31 - 1, rows, dtype=np.int64),
        "NAME": np.array([f"SRC_{i:08d}" for i in range(rows)], dtype="U16"),
        "BAND": np.array([f"B{(i % 5)}" for i in range(rows)], dtype="U4"),
    }
    return data, ["NAME", "BAND"], {}, {}


def _build_mixed_space_strings(rows: int, rng: np.random.Generator):
    name = np.array(
        [(f"SRC_{i:08d}" + " " * 4).encode("ascii") for i in range(rows)],
        dtype="S16",
    )
    band = np.array(
        [(f"B{(i % 5)}" + " " * 3).encode("ascii") for i in range(rows)],
        dtype="S4",
    )
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "OBJID": rng.integers(0, 2**31 - 1, rows, dtype=np.int64),
        "NAME": name,
        "BAND": band,
    }
    return data, ["NAME", "BAND"], {}, {}


def _build_wide_vectors(rows: int, rng: np.random.Generator):
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "FLUX8": rng.random((rows, 8), dtype=np.float32),
        "ERR8": rng.random((rows, 8), dtype=np.float32),
        "FLAG8": rng.integers(0, 3, size=(rows, 8), dtype=np.int16),
    }
    return data, [], {}, {}


def _build_null_int(rows: int, rng: np.random.Generator):
    sentinel = -32768
    quality = rng.integers(-1000, 1000, size=rows, dtype=np.int16)
    # Inject deterministic null sentinels.
    quality[::17] = sentinel
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "QUALITY": quality,
    }
    return data, [], {"QUALITY": sentinel}, {}


def _build_scaled_int(rows: int, rng: np.random.Generator):
    quality = rng.integers(-1000, 1000, size=rows, dtype=np.int16)
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "QUALITY": quality,
    }
    # Fractional scaling to validate physical-value path.
    return data, [], {}, {"QUALITY": (0.5, 1.25)}


def _build_bit_vla(rows: int, rng: np.random.Generator):
    rows = max(3, rows)
    bit = rng.integers(0, 2, size=(rows, 8), dtype=np.uint8)
    vla = np.empty(rows, dtype=object)
    for i in range(rows):
        n = int(rng.integers(1, 8))
        vla[i] = rng.integers(0, 1000, size=n, dtype=np.int32)

    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "BITS": bit,
        "VLA": vla,
    }
    return data, [], {}, {}


def _write_bit_vla_table(path: Path, data: dict[str, np.ndarray]) -> None:
    cols = [
        fits.Column(name="RA", format="D", array=data["RA"]),
        fits.Column(name="DEC", format="D", array=data["DEC"]),
        fits.Column(name="BITS", format="8X", array=data["BITS"]),
        fits.Column(name="VLA", format="PJ()", array=data["VLA"]),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


def _build_bit_only(rows: int, rng: np.random.Generator):
    rows = max(3, rows)
    bit = rng.integers(0, 2, size=(rows, 8), dtype=np.uint8)
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "BITS": bit,
    }
    return data, [], {}, {}


def _write_bit_only_table(path: Path, data: dict[str, np.ndarray]) -> None:
    cols = [
        fits.Column(name="RA", format="D", array=data["RA"]),
        fits.Column(name="DEC", format="D", array=data["DEC"]),
        fits.Column(name="BITS", format="8X", array=data["BITS"]),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


def _build_vla_only(rows: int, rng: np.random.Generator):
    rows = max(3, rows)
    vla = np.empty(rows, dtype=object)
    for i in range(rows):
        n = int(rng.integers(1, 8))
        vla[i] = rng.integers(0, 1000, size=n, dtype=np.int32)
    data = {
        "RA": rng.uniform(0.0, 360.0, rows).astype(np.float64),
        "DEC": rng.uniform(-90.0, 90.0, rows).astype(np.float64),
        "VLA": vla,
    }
    return data, [], {}, {}


def _write_vla_only_table(path: Path, data: dict[str, np.ndarray]) -> None:
    cols = [
        fits.Column(name="RA", format="D", array=data["RA"]),
        fits.Column(name="DEC", format="D", array=data["DEC"]),
        fits.Column(name="VLA", format="PJ()", array=data["VLA"]),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


SCENARIOS = {
    "numeric_only": _build_numeric_only,
    "mixed_null_strings": _build_mixed_null_strings,
    "mixed_space_strings": _build_mixed_space_strings,
    "wide_vectors": _build_wide_vectors,
    "null_int": _build_null_int,
    "scaled_int": _build_scaled_int,
    "bit_vla": _build_bit_vla,
    "bit_only": _build_bit_only,
    "vla_only": _build_vla_only,
}


def _run_scenario(
    scenario: str,
    file_path: Path,
    rows: int,
    string_cols: list[str],
    null_cols: dict[str, int],
    scale_cols: dict[str, tuple[float, float]],
    batch_size: int,
    warmup: int,
    iterations: int,
):
    result_rows: list[dict[str, str | float | int]] = []
    col_list = []
    start_row = 1
    num_rows = rows

    def _torchfits_cpp_numpy_chunk():
        return torchfits_cpp.read_fits_table_rows_numpy(
            str(file_path), 1, col_list, start_row, num_rows, False
        )

    def _torchfits_numpy_chunk_to_arrow(chunk: dict[str, object]):
        pydict: dict[str, object] = {}
        for name, value in chunk.items():
            if isinstance(value, np.ndarray):
                col = _to_native_endian(value)
                if col.ndim <= 1:
                    pydict[name] = pa.array(col)
                elif col.ndim == 2:
                    flat = _to_native_endian(col.reshape(-1))
                    values = pa.array(flat)
                    pydict[name] = pa.FixedSizeListArray.from_arrays(
                        values, int(col.shape[1])
                    )
                else:
                    pydict[name] = pa.array(col.tolist())
            elif (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], np.ndarray)
            ):
                flat = np.ascontiguousarray(value[0]).reshape(-1)
                offsets = np.ascontiguousarray(value[1], dtype=np.int64).reshape(-1)
                if offsets.size == 0:
                    pydict[name] = pa.array([])
                elif int(offsets[-1]) <= np.iinfo(np.int32).max:
                    pydict[name] = pa.ListArray.from_arrays(
                        pa.array(offsets.astype(np.int32, copy=False)), pa.array(flat)
                    )
                else:
                    pydict[name] = pa.LargeListArray.from_arrays(
                        pa.array(offsets), pa.array(flat)
                    )
            else:
                pydict[name] = value
        return pa.table(pydict)

    methods = [
        ("torchfits_cpp_numpy_raw", _torchfits_cpp_numpy_chunk),
        (
            "torchfits_cpp_numpy_to_arrow",
            lambda: _torchfits_numpy_chunk_to_arrow(_torchfits_cpp_numpy_chunk()),
        ),
        (
            "torchfits_read_raw",
            lambda: torchfits.table.read(
                str(file_path),
                hdu=1,
                decode_bytes=False,
                apply_fits_nulls=False,
                backend="cpp_numpy",
                batch_size=batch_size,
            ),
        ),
        ("fitsio_read_raw", lambda: fitsio.read(str(file_path), ext=1)),
        (
            "fitsio_read_raw_arrow",
            lambda: _fitsio_record_to_arrow(fitsio.read(str(file_path), ext=1)),
        ),
        (
            "fitsio_read_raw_torch",
            lambda: _fitsio_record_to_torch(fitsio.read(str(file_path), ext=1)),
        ),
        ("astropy_table_read", lambda: Table.read(str(file_path), hdu=1)),
        (
            "torchfits_scan_sum_raw",
            lambda: sum(
                float(np.asarray(batch.column("RA")).sum())
                for batch in torchfits.table.scan(
                    str(file_path),
                    hdu=1,
                    decode_bytes=False,
                    apply_fits_nulls=False,
                    backend="cpp_numpy",
                    batch_size=batch_size,
                )
            ),
        ),
        (
            "torchfits_scan_torch_sum_cpu",
            lambda: sum(
                float(chunk["RA"].sum().item())
                for chunk in torchfits.table.scan_torch(
                    str(file_path),
                    hdu=1,
                    batch_size=batch_size,
                    device="cpu",
                )
            ),
        ),
    ]

    if null_cols:
        methods.append(
            (
                "torchfits_read_raw_with_nulls",
                lambda: torchfits.table.read(
                    str(file_path),
                    hdu=1,
                    decode_bytes=False,
                    apply_fits_nulls=True,
                    backend="cpp_numpy",
                    batch_size=batch_size,
                ),
            )
        )

    if scale_cols:
        methods.append(
            (
                "torchfits_read_scaled",
                lambda: torchfits.table.read(
                    str(file_path),
                    hdu=1,
                    decode_bytes=False,
                    apply_fits_nulls=False,
                    backend="cpp_numpy",
                    batch_size=batch_size,
                ),
            )
        )

    if string_cols:
        methods.extend(
            [
                (
                    "torchfits_read_decoded",
                    lambda: torchfits.table.read(
                        str(file_path),
                        hdu=1,
                        decode_bytes=True,
                        apply_fits_nulls=False,
                        backend="cpp_numpy",
                        batch_size=batch_size,
                    ),
                ),
                (
                    "fitsio_read_decoded",
                    lambda: {
                        col: fitsio.read(str(file_path), ext=1)[col].astype("U")
                        for col in string_cols
                    },
                ),
            ]
        )

    proj_cols = ["RA", "DEC"]
    # Keep projected benchmark stable by using a fixed-size contiguous window.
    proj_start = max(0, rows // 4)
    proj_len = min(20_000, max(4_096, rows // 8), max(1, rows - proj_start))
    proj_stop = proj_start + proj_len
    proj_rows = np.arange(proj_start, proj_stop, dtype=np.int64)
    methods.extend(
        [
            (
                "torchfits_read_projected",
                lambda: torchfits.table.read(
                    str(file_path),
                    hdu=1,
                    columns=proj_cols,
                    row_slice=(int(proj_rows[0]), int(proj_rows[-1]) + 1),
                    decode_bytes=False,
                    apply_fits_nulls=False,
                    backend="cpp_numpy",
                    batch_size=batch_size,
                ),
            ),
            (
                "fitsio_read_projected",
                lambda: fitsio.read(
                    str(file_path), ext=1, columns=proj_cols, rows=proj_rows
                ),
            ),
            (
                "fitsio_read_projected_arrow",
                lambda: _fitsio_record_to_arrow(
                    fitsio.read(
                        str(file_path), ext=1, columns=proj_cols, rows=proj_rows
                    )
                ),
            ),
        ]
    )

    warm_profile = _profile_methods_round_robin(
        methods, warmup=warmup, iterations=iterations
    )
    cold_profile = _profile_methods_cold_once(methods)
    for method, _ in methods:
        warm = warm_profile[method]
        cold = cold_profile[method]
        print(
            f"{scenario:20s}  {method:24s} [warm]: {warm['mean_s']:.6f}s ± {warm['std_s']:.6f}s"
            f" | rssΔ={warm['mean_rss_delta_mb']:.3f}MB peakΔ={warm['mean_peak_rss_delta_mb']:.3f}MB"
        )
        print(
            f"{scenario:20s}  {method:24s} [cold]: {cold['mean_s']:.6f}s"
            f" | rssΔ={cold['mean_rss_delta_mb']:.3f}MB peakΔ={cold['mean_peak_rss_delta_mb']:.3f}MB"
        )
        result_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "phase": "warm_round_robin",
                "mean_s": warm["mean_s"],
                "std_s": warm["std_s"],
                "mean_rss_delta_mb": warm["mean_rss_delta_mb"],
                "std_rss_delta_mb": warm["std_rss_delta_mb"],
                "mean_peak_rss_delta_mb": warm["mean_peak_rss_delta_mb"],
                "std_peak_rss_delta_mb": warm["std_peak_rss_delta_mb"],
                "samples": int(warm["samples"]),
                "warmup": warmup,
                "rows": rows,
            }
        )
        result_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "phase": "cold_once",
                "mean_s": cold["mean_s"],
                "std_s": cold["std_s"],
                "mean_rss_delta_mb": cold["mean_rss_delta_mb"],
                "std_rss_delta_mb": cold["std_rss_delta_mb"],
                "mean_peak_rss_delta_mb": cold["mean_peak_rss_delta_mb"],
                "std_peak_rss_delta_mb": cold["std_peak_rss_delta_mb"],
                "samples": int(cold["samples"]),
                "warmup": 0,
                "rows": rows,
            }
        )

    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diverse Arrow table benchmark")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="numeric_only,mixed_null_strings,mixed_space_strings,wide_vectors,null_int,scaled_int,bit_vla,bit_only,vla_only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench_results/table_arrow_diverse_results.csv",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SCENARIOS]
    if unknown:
        raise ValueError(
            f"Unknown scenarios: {unknown}. Available: {sorted(SCENARIOS)}"
        )

    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_table_arrow_diverse_"))
    rng = np.random.default_rng(0)
    all_results: list[dict[str, str | float | int]] = []

    for scenario in selected:
        builder = SCENARIOS[scenario]
        data, string_cols, null_cols, scale_cols = builder(args.rows, rng)
        file_path = data_dir / f"{scenario}.fits"
        if scenario == "bit_vla":
            _write_bit_vla_table(file_path, data)
        elif scenario == "bit_only":
            _write_bit_only_table(file_path, data)
        elif scenario == "vla_only":
            _write_vla_only_table(file_path, data)
        else:
            Table(data).write(file_path, format="fits", overwrite=True)
        if null_cols or scale_cols:
            with fits.open(file_path, mode="update") as hdul:
                header = hdul[1].header
                for idx in range(1, int(header.get("TFIELDS", 0)) + 1):
                    name = header.get(f"TTYPE{idx}")
                    if name in null_cols:
                        header[f"TNULL{idx}"] = int(null_cols[name])
                    if name in scale_cols:
                        bscale, bzero = scale_cols[name]
                        header[f"TSCAL{idx}"] = float(bscale)
                        header[f"TZERO{idx}"] = float(bzero)
        print(f"\nScenario: {scenario} -> {file_path}")
        all_results.extend(
            _run_scenario(
                scenario=scenario,
                file_path=file_path,
                rows=args.rows,
                string_cols=string_cols,
                null_cols=null_cols,
                scale_cols=scale_cols,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "method",
                "phase",
                "mean_s",
                "std_s",
                "mean_rss_delta_mb",
                "std_rss_delta_mb",
                "mean_peak_rss_delta_mb",
                "std_peak_rss_delta_mb",
                "samples",
                "warmup",
                "rows",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
