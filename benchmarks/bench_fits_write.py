#!/usr/bin/env python3
"""Write→compress benchmarks: torchfits vs astropy CompImageHDU."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow standalone import (python -m benchmarks.bench_fits_write / direct run)
# even though bench_all.py normally provides the repo-root sys.path entry.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits

from benchmarks.bench_contract import annotate_rankings
from benchmarks.bench_timing import time_medians_interleaved

_BENCH_HOST = socket.gethostname()

_SHAPE = (1024, 1024)
_COMPRESSIONS: tuple[tuple[str, str], ...] = (
    ("RICE_1", "rice"),
    ("HCOMPRESS_1", "hcompress"),
)


def run_write_compress_rows(
    *,
    run_id: str,
    output_dir: Path,
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
) -> list[dict[str, Any]]:
    """Time compressed FITS writes for a float32 2D tensor."""
    if runs is None:
        runs = 3 if profile == "user" else 7
    if warmup is None:
        warmup = 1 if profile == "user" else 2

    rng = np.random.default_rng(20260318)
    tensor = torch.from_numpy(rng.normal(size=_SHAPE).astype(np.float32))
    payload_mb = tensor.numel() * 4 / (1024.0 * 1024.0)
    temp_root = Path(tempfile.mkdtemp(prefix="torchfits_bench_write_"))
    rows: list[dict[str, Any]] = []

    try:
        for algo, label in _COMPRESSIONS:
            case_id = f"write_compress_{label}_medium_float32_2d"
            case_label = f"write_compress [{label}] medium float32 2D"
            tf_path = temp_root / f"torchfits_{label}.fits"
            ap_path = temp_root / f"astropy_{label}.fits"

            def torchfits_write(p=tf_path, a=algo) -> None:
                if p.exists():
                    p.unlink()
                torchfits.write_tensor(str(p), tensor, overwrite=True, compress=a)

            def astropy_write(p=ap_path, a=algo) -> None:
                if p.exists():
                    p.unlink()
                astropy_fits.HDUList(
                    [
                        astropy_fits.PrimaryHDU(),
                        astropy_fits.CompImageHDU(
                            tensor.numpy(),
                            compression_type=a,
                        ),
                    ]
                ).writeto(p, overwrite=True)

            print(f"[fits-write] case={case_id} runs={runs}", flush=True)
            timed = time_medians_interleaved(
                {"torchfits": torchfits_write, "astropy": astropy_write},
                runs=runs,
                warmup=warmup,
            )
            for library, method in (("torchfits", "torchfits"), ("astropy", "astropy")):
                t_val, peak_rss, peak_cuda, err = timed[method]
                status = "OK" if t_val is not None else "FAILED"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "fits_write",
                        "case_id": case_id,
                        "case_label": case_label,
                        "operation": "write_compress",
                        "family": "smart",
                        "library": library,
                        "method": method,
                        "mode": "smart",
                        "status": status,
                        "skip_reason": "" if err is None else str(err),
                        "comparable": status == "OK",
                        "mmap_target": "n/a",
                        "host": _BENCH_HOST,
                        "time_s": t_val,
                        "peak_rss_mb": peak_rss,
                        "peak_cuda_alloc_mb": peak_cuda,
                        "throughput": (payload_mb / t_val) if t_val else None,
                        "unit": "MB/s",
                        "size_mb": payload_mb,
                        "n_points": "",
                        "metadata": json.dumps(
                            {
                                "compression": label,
                                "algorithm": algo,
                                "dimensions": "2d",
                            }
                        ),
                    }
                )
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)

    annotate_rankings(rows)
    out_csv = output_dir / "fits_write_results.csv"
    from benchmarks.bench_contract import RESULT_COLUMNS, write_csv

    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} write-compress rows to {out_csv}", flush=True)
    return rows
