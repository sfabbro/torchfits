#!/usr/bin/env python3
"""HTTP cutout benchmark: URL-to-memory reads, no disk.

Workload: N cutouts from MEF extensions served over HTTPS (CADC Direct Data),
read straight into memory. Cases:
- ``torchfits_http_reader``: one ``open_subset_reader`` over the URL, all
  cutouts from the same reader handle (comparable).
- ``torchfits_http_naive``: one fresh ``read_subset`` per cutout
  (non-comparable baseline: pays per-call connection/parse overhead).
- ``astropy_http``: ``fits.open(url, memmap=False)`` + ``section`` slices
  (lazy reads from the remote file).
- ``astropy_http_materialize``: whole-plane ``.data`` then in-memory slices
  (fetches the entire HDU once).

Timing via the interleaved harness; per-case metadata carries the probe
dtype/itemsize, cutout size and seed so payload accounting is exact.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402

from benchmarks.bench_contract import (  # noqa: E402
    RESULT_COLUMNS,
    annotate_rankings,
    make_run_id,
    write_csv,
)
from benchmarks.bench_fixtures import (  # noqa: E402
    cadc_direct_urls,
    discover_fits_names,
)
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

_BENCH_HOST = socket.gethostname()

_DEFAULT_DATA = ROOT / "benchmarks_data" / "cfht_megacam"

_CUTOUT_SIZE = 256
_CUTOUTS_PER_HDU = 4


def _cutout_coords(
    naxis1: int, naxis2: int, *, n: int, seed: int
) -> list[tuple[int, int, int, int]]:
    """Deterministic non-overlapping cutout boxes (x1, y1, x2, y2)."""
    rng = np.random.default_rng(seed)
    coords: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        x1 = int(rng.integers(0, max(1, naxis1 - _CUTOUT_SIZE)))
        y1 = int(rng.integers(0, max(1, naxis2 - _CUTOUT_SIZE)))
        coords.append((x1, y1, x1 + _CUTOUT_SIZE, y1 + _CUTOUT_SIZE))
    return coords


def _cutout_payload_mb(
    coords: list[tuple[int, int, int, int]], *, itemsize: int
) -> float:
    pixels = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in coords)
    return pixels * itemsize / (1024.0 * 1024.0)


def run_http_stream_rows(
    *,
    run_id: str,
    urls: list[str],
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
    max_hdus: int = 4,
) -> list[dict[str, Any]]:
    if runs is None:
        runs = 3 if profile == "user" else 5
    if warmup is None:
        warmup = 1 if profile == "user" else 2
    rows: list[dict[str, Any]] = []
    hdus = list(range(1, max_hdus + 1))

    for url in urls:
        for hdu in hdus:
            # Shape from the HTTP reader meta (header peek, outside timed path).
            try:
                with torchfits.open_subset_reader(url, hdu=hdu) as reader:
                    naxis2, naxis1 = reader.shape
            except Exception as exc:
                print(f"[http_stream] skip {url} hdu{hdu}: {exc}", flush=True)
                continue
            if naxis1 < 16 or naxis2 < 16:
                continue
            coords = _cutout_coords(
                naxis1, naxis2, n=_CUTOUTS_PER_HDU, seed=hdu + len(url)
            )
            # Probe dtype once for payload accounting (outside timed path).
            probe = torchfits.read_subset(url, hdu=hdu, x1=0, y1=0, x2=8, y2=8)
            itemsize = probe.dtype.itemsize
            size_mb = _cutout_payload_mb(coords, itemsize=itemsize)
            case_id = f"{Path(url).name}::hdu{hdu}::http_stream"
            case_label = f"{Path(url).name} ext {hdu} [{len(coords)} url cutouts]"

            def torchfits_reader(
                u=url,
                h=hdu,
                c=coords,
            ) -> list[Any]:
                with torchfits.open_subset_reader(u, hdu=h) as reader:
                    return [reader.read_subset(x1, y1, x2, y2) for x1, y1, x2, y2 in c]

            def torchfits_naive(u=url, h=hdu, c=coords) -> list[Any]:
                return [
                    torchfits.read_subset(u, hdu=h, x1=x1, y1=y1, x2=x2, y2=y2)
                    for x1, y1, x2, y2 in c
                ]

            def astropy_http(u=url, h=hdu, c=coords) -> list[Any]:
                from astropy.io import fits

                with fits.open(u, memmap=False) as hdul:
                    ext = hdul[h]
                    return [
                        np.asarray(ext.section[y1:y2, x1:x2]) for x1, y1, x2, y2 in c
                    ]

            def astropy_materialize(u=url, h=hdu, c=coords) -> list[Any]:
                from astropy.io import fits

                with fits.open(u, memmap=False) as hdul:
                    data = np.asarray(hdul[h].data)
                    return [data[y1:y2, x1:x2] for x1, y1, x2, y2 in c]

            print(
                f"[http_stream] case={case_id} runs={runs} "
                f"payload={size_mb:.3f}MB itemsize={itemsize}",
                flush=True,
            )
            timed = time_medians_interleaved(
                {
                    "torchfits_http_reader": torchfits_reader,
                    "torchfits_http_naive": torchfits_naive,
                    "astropy_http": astropy_http,
                    "astropy_http_materialize": astropy_materialize,
                },
                runs=runs,
                warmup=warmup,
            )
            method_meta = (
                ("torchfits_http_reader", "torchfits"),
                ("torchfits_http_naive", "torchfits"),
                ("astropy_http", "astropy"),
                ("astropy_http_materialize", "astropy"),
            )
            for method, library in method_meta:
                if method not in timed:
                    continue
                t_val, peak_rss, peak_cuda, err = timed[method]
                status = "OK" if t_val is not None else "FAILED"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "http_stream",
                        "case_id": case_id,
                        "case_label": case_label,
                        "operation": "http_range_cutouts",
                        "family": "pipeline",
                        "library": library,
                        "method": method,
                        "mode": "pipeline",
                        "status": status,
                        "skip_reason": "" if err is None else str(err),
                        "comparable": status == "OK",
                        "mmap_target": "n/a",
                        "host": _BENCH_HOST,
                        "time_s": t_val,
                        "peak_rss_mb": peak_rss,
                        "peak_cuda_alloc_mb": peak_cuda,
                        "throughput": (size_mb / t_val) if t_val else None,
                        "unit": "MB/s",
                        "size_mb": size_mb,
                        "n_points": len(coords),
                        "metadata": json.dumps(
                            {
                                "hdu": hdu,
                                "url": url,
                                "itemsize": itemsize,
                                "cutout_size": _CUTOUT_SIZE,
                                "n_cutouts": len(coords),
                                "seed": hdu + len(url),
                            }
                        ),
                    }
                )
    annotate_rankings(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--urls", default="", help="comma-separated URL list override")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-hdus", type=int, default=4)
    args = parser.parse_args()
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.urls:
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    else:
        names = discover_fits_names(args.data_dir, max_files=args.max_files)
        urls = cadc_direct_urls(names)
        if not urls:
            print(
                f"[http_stream] no FITS names under {args.data_dir} to derive URLs; "
                "pass --urls or run scripts/fetch_cfht_megacam_sample.sh",
                flush=True,
            )
            return 1
    rows = run_http_stream_rows(
        run_id=run_id,
        urls=urls,
        profile=args.profile,
        max_hdus=args.max_hdus,
    )
    out_csv = run_dir / "http_stream_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} http_stream rows to {out_csv}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
