#!/usr/bin/env python3
"""GPU I/O transport rows for bench-all (MPS on macOS, CUDA on Linux).

Emits normalized rows with metadata.io_transport set so
scripts/render_bench_iopath_table.py can fill disk→GPU columns.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitsio  # noqa: E402
import torchfits  # noqa: E402
from benchmarks.bench_fast import build_dataset  # noqa: E402


def default_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _median_time(fn, warmup: int, iters: int, device: str) -> float | None:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) if samples else None


def run_gpu_transport_rows(
    *,
    run_id: str,
    device: str | None = None,
    iterations: int = 10,
    warmup: int = 3,
) -> list[dict[str, Any]]:
    device = device or default_device()
    if device == "cpu":
        return []

    if device == "cuda" and not torch.cuda.is_available():
        return []
    if device == "mps" and not torch.backends.mps.is_available():
        return []

    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_bench_gpu_"))
    files = build_dataset(data_dir)
    rows: list[dict[str, Any]] = []
    transport = "disk\u2192RAM\u2192GPU"

    for name, path in sorted(files.items()):
        if name.startswith("timeseries_long_") or name.endswith("_large"):
            continue
        hdu = 1 if name.startswith("compressed_") else 0
        case_id = f"{name}::read_full_gpu"
        size_mb = path.stat().st_size / (1024 * 1024)

        read_kwargs: dict = {
            "hdu": hdu,
            "device": device,
            "scale_on_device": True,
            "cache_capacity": 10,
        }
        if device == "mps":
            read_kwargs["fp16"] = True

        def tf_read():
            return torchfits.read(str(path), **read_kwargs)

        def fitsio_torch():
            arr = fitsio.read(str(path), ext=hdu)
            t = torch.from_numpy(arr)
            if device == "mps" and t.dtype == torch.float64:
                t = t.to(torch.float32)
            return t.to(device)

        for library, method, fn in (
            ("torchfits", "torchfits_device", tf_read),
            ("fitsio", "fitsio_torch_device", fitsio_torch),
        ):
            try:
                t = _median_time(fn, warmup, iterations, device)
            except Exception as exc:
                print(f"[bench-gpu] skip {name} {library}: {exc}", flush=True)
                continue
            if t is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "fits_gpu",
                    "case_id": case_id,
                    "case_label": f"{name} [read_full @ {device}]",
                    "operation": "read_full",
                    "family": "smart",
                    "library": library,
                    "method": method,
                    "mode": "smart",
                    "status": "OK",
                    "skip_reason": "",
                    "comparable": True,
                    "mmap_target": "on",
                    "time_s": t,
                    "throughput": "",
                    "unit": "MB/s",
                    "size_mb": size_mb,
                    "n_points": "",
                    "metadata": json.dumps(
                        {"device": device, "io_transport": transport}
                    ),
                }
            )

    return rows


def main() -> int:
    import argparse

    from benchmarks.bench_contract import RESULT_COLUMNS, write_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()
    rows = run_gpu_transport_rows(run_id=args.run_id, device=args.device)
    write_csv(args.output, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} GPU transport rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
