#!/usr/bin/env python3
"""MEF median stack benchmark: torchfits API paths + CLI, no astropy in the timed path.

Workload: median of N CFHT MegaCam MEFs (40 CCD HDUs each, 4644x2112 f32),
sharded slice-by-slice so peak RAM ~= N * slice_rows * row_bytes.

One parametrized slice-streaming kernel drives every method: only the read
backend (torchfits ``open_subset_reader`` / fitsio) and the order-statistic
(torch / numpy) differ, so the fitsio-vs-torchfits comparison is airtight.

Timed cases, three families:
- ``io_read``: the read engines alone (no median): torchfits.read_batch
  vs fitsio whole-plane reads.
- ``io_compute``: the order-statistic alone (stack pre-loaded once):
  torch (sort/select) vs numpy.
- ``pipeline``: end-to-end median stacks: torchfits read_batch,
  slice-stream local (threaded), torchfits DataLoader (FitsImageDataset +
  make_loader, one batch per hdu), slice-stream URL, fitsio+numpy slice
  (classic single-threaded fitsio, threads=1).

Resource metrics per method: peak RSS from the timing harness; median process
CPU-seconds (ru_utime+ru_stime, threads included) and a cpu/wall ratio
(=1.0 pure single-core; >1.0 shows parallelism); LOC of the timed function
as a code-complexity proxy.

Untimed pipeline (uses more torchfits API): read_batch_info schema probe,
write() for the output MEF, write_checksums(). The CLI-diff reference is
computed with an independent read path (fitsio for local, torchfits readers
for URL mode) so validation does not share the artifact's code.

Timed CLI cases (subprocess ``torchfits``) on the outputs:
info, stats, verify, diff, cutout (+ probe for URL mode).
"""

from __future__ import annotations

import argparse
import inspect
import json
import resource
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchfits  # noqa: E402
from torchfits.data import FitsImageDataset, make_loader  # noqa: E402

from benchmarks.bench_contract import (  # noqa: E402
    RESULT_COLUMNS,
    annotate_rankings,
    make_run_id,
    write_csv,
)
from benchmarks.bench_fixtures import (  # noqa: E402
    discover_local_paths,
)
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

_BENCH_HOST = socket.gethostname()
_DEFAULT_DATA = ROOT / "benchmarks_data" / "cfht_megacam"

_ITEMSIZE = 4  # f32

_CUTOUT_BOX = (0, 0, 256, 256)


def _median_axis0(stack: torch.Tensor) -> torch.Tensor:
    """Median over dim 0 matching numpy: even N averages the two middle values.

    ``torch.sort`` is used: on the torch builds measured here a full sort of
    the N-element column is faster than two ``kthvalue`` introselects (both
    O(N) class for the selection, unlike numpy's C partition which wins on
    CPU constants). Keep the numpy path on ``np.median`` (its own partition).
    """
    n = stack.shape[0]
    if n % 2 == 1:
        return stack.median(dim=0).values
    ordered, _ = torch.sort(stack, dim=0)
    return (ordered[n // 2 - 1] + ordered[n // 2]) * 0.5


class _FitsioExt:
    """Adapter giving fitsio an ``open_subset_reader``-shaped surface."""

    def __init__(self, path: str, hdu: int) -> None:
        import fitsio

        self._handle = fitsio.FITS(path)
        self._ext = self._handle[hdu]
        height, width = self._ext.get_dims()
        self.shape = (int(height), int(width))

    def read_subset(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        # Normalize like torchfits (BSCALE/BZERO to f32): fitsio returns raw
        # scaled ints (e.g. u16 from BITPIX 8 + BZERO), and np.median upcasts
        # integer inputs to float64, breaking cli_diff dtype parity.
        return np.asarray(self._ext[y1:y2, x1:x2], dtype=np.float32)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_FitsioExt":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _median_torch(col: list[Any]) -> torch.Tensor:
    return _median_axis0(torch.stack([torch.as_tensor(a) for a in col], dim=0))


def _median_numpy(col: list[Any]) -> np.ndarray:
    return np.median(np.stack([np.asarray(a) for a in col], axis=0), axis=0)


def _slice_median_stack(
    paths: list[str],
    hdu: int,
    *,
    slice_rows: int,
    threads: int,
    open_reader: Callable[[str, int], Any],
    median_fn: Callable[[list[Any]], Any],
) -> Any:
    """Slice-wise median of N files: one loop, injectable read + median.

    Peak RAM ~= N * slice_rows * row_bytes regardless of N. *open_reader*
    yields a reader with ``.shape`` and ``read_subset(x1, y1, x2, y2)``;
    *median_fn* maps one column (N stacked slices) to the output slice.
    """
    out: Any = None
    with ExitStack() as stack:
        readers = [stack.enter_context(open_reader(p, hdu)) for p in paths]
        height, width = readers[0].shape
        fan_out = threads > 1 and len(readers) > 1
        for y0 in range(0, height, slice_rows):
            y1 = min(y0 + slice_rows, height)
            if fan_out:
                with ThreadPoolExecutor(max_workers=threads) as ex:
                    col = [
                        f.result()
                        for f in [
                            ex.submit(r.read_subset, 0, y0, width, y1) for r in readers
                        ]
                    ]
            else:
                col = [r.read_subset(0, y0, width, y1) for r in readers]
            med = median_fn(col)
            if out is None:
                out = (
                    torch.empty(height, width, dtype=med.dtype)
                    if isinstance(med, torch.Tensor)
                    else np.empty((height, width), dtype=med.dtype)
                )
            out[y0:y1] = med
    return out


def _read_median_slices(
    paths: list[str], hdu: int, *, slice_rows: int, threads: int
) -> torch.Tensor:
    return _slice_median_stack(
        paths,
        hdu,
        slice_rows=slice_rows,
        threads=threads,
        open_reader=torchfits.open_subset_reader,
        median_fn=_median_torch,
    )


def _read_median_fitsio(paths: list[str], hdu: int, *, slice_rows: int) -> np.ndarray:
    # Classic fitsio+numpy: single-threaded by design (cpu/wall ~1.0).
    return _slice_median_stack(
        paths,
        hdu,
        slice_rows=slice_rows,
        threads=1,
        open_reader=_FitsioExt,
        median_fn=_median_numpy,
    )


def _read_median_batch(paths: list[str], hdu: int) -> torch.Tensor:
    tensors = torchfits.read_batch(paths, hdu=hdu, strict=True)
    return _median_axis0(torch.stack(tensors, dim=0))


def _read_median_dataloader(
    paths: list[str], hdu: int, *, num_workers: int
) -> torch.Tensor:
    """Median via torchfits FitsImageDataset + torch DataLoader.

    One batch of ``len(paths)`` planes per hdu, median over dim 0.
    ``make_loader(shuffle=False, optimize_cache=False)``: deterministic file
    order and no cache pre-warm, so this measures the plain DataLoader path
    (dataset machinery, collate, iteration) on equal footing with the direct
    reads.

    num_workers > 0 is blocked on hosts with a small ``/dev/shm`` tmpfs:
    torch shares worker tensors via ``shm_open`` under **both** sharing
    strategies, so a ~390MB batch exceeds a 64MB tmpfs (ENOSPC). Use
    num_workers=0 here; on hosts with adequate shm the same code works with
    ``num_workers=8``.
    """
    ds = FitsImageDataset(paths, hdu=hdu, add_channel_dim=False, mmap=False)
    loader = make_loader(
        ds,
        batch_size=len(paths),
        shuffle=False,
        num_workers=num_workers,
        optimize_cache=False,
    )
    batch, _labels = next(iter(loader))
    return _median_axis0(batch)


def _read_batch_only(paths: list[str], hdu: int) -> list[Any]:
    return torchfits.read_batch(paths, hdu=hdu, strict=True)


def _read_fitsio_only(paths: list[str], hdu: int) -> list[np.ndarray]:
    import fitsio

    out = []
    for p in paths:
        with fitsio.FITS(p) as handle:
            out.append(np.asarray(handle[hdu].read()))
    return out


def _median_compute_only(stack: torch.Tensor, *, use_torch: bool) -> np.ndarray:
    if use_torch:
        return _median_axis0(stack).numpy()
    return np.median(stack.numpy(), axis=0)


def _write_median_mef(path: str, planes: list[Any]) -> None:
    torchfits.write(
        path,
        [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in planes],
        overwrite=True,
    )


def _run_cli(args: list[str], *, timeout: float = 300.0) -> tuple[float, int, str]:
    import shutil

    cli = shutil.which("torchfits")
    if not cli:
        return 0.0, 127, "torchfits CLI not on PATH"
    t0 = time.monotonic()
    proc = subprocess.run(
        [cli, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dt = time.monotonic() - t0
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
    return dt, proc.returncode, " | ".join(tail)


def _payload_mb(n_files: int, n_hdus: int, plane_pixels: int) -> float:
    return n_files * n_hdus * plane_pixels * _ITEMSIZE / (1024.0 * 1024.0)


def _cpu_time() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _cpu_wrap(fn: Any, tracker: dict[str, list[float]], name: str) -> Any:
    def wrapped() -> Any:
        t0 = _cpu_time()
        out = fn()
        tracker.setdefault(name, []).append(_cpu_time() - t0)
        return out

    return wrapped


def _loc(fn: Any) -> int:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return 0
    return sum(
        1
        for line in src.splitlines()
        if line.strip() and not line.strip().startswith(("#", "def ", "@"))
    )


def run_median_stack_rows(
    *,
    run_id: str,
    paths: list[str],
    urls: list[str] | None = None,
    out_dir: Path,
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
    max_hdus: int = 2,
    slice_rows: int = 128,
    threads: int = 8,
) -> list[dict[str, Any]]:
    if runs is None:
        runs = 3 if profile == "user" else 5
    if warmup is None:
        warmup = 1 if profile == "user" else 2
    urls = urls or []

    if not paths:
        print(
            f"[median_stack] no FITS under {_DEFAULT_DATA}; "
            "run scripts/fetch_cfht_megacam_sample.sh",
            flush=True,
        )
        return []

    # Schema probe (untimed): read_batch_info + read_keys from every file.
    try:
        info = torchfits.read_batch_info(paths)
    except Exception as exc:
        print(f"[median_stack] read_batch_info failed: {exc}", flush=True)
        info = {}
    hdus = list(range(1, max_hdus + 1))
    with torchfits.open_subset_reader(paths[0], hdu=hdus[0]) as _probe:
        height, width = _probe.shape
    plane_pixels = int(height) * int(width)
    size_mb = _payload_mb(len(paths), 1, plane_pixels)
    rows: list[dict[str, Any]] = []

    for hdu in hdus:
        case_id = f"median_hdu{hdu}::n{len(paths)}"
        print(
            f"[median_stack] case={case_id} runs={runs} payload={size_mb:.1f}MB",
            flush=True,
        )
        cpu_track: dict[str, list[float]] = {}
        timed = time_medians_interleaved(
            {
                "batch_whole_plane": _cpu_wrap(
                    lambda h=hdu: _read_median_batch(paths, h),
                    cpu_track,
                    "batch_whole_plane",
                ),
                "slice_stream_local": _cpu_wrap(
                    lambda h=hdu: _read_median_slices(
                        paths, h, slice_rows=slice_rows, threads=threads
                    ),
                    cpu_track,
                    "slice_stream_local",
                ),
                "dataloader_stack": _cpu_wrap(
                    # num_workers=0: this host's /dev/shm is 64MB, too small
                    # for torch's shm_open-based worker tensor sharing.
                    lambda h=hdu: _read_median_dataloader(paths, h, num_workers=0),
                    cpu_track,
                    "dataloader_stack",
                ),
                "fitsio_numpy": _cpu_wrap(
                    lambda h=hdu: _read_median_fitsio(paths, h, slice_rows=slice_rows),
                    cpu_track,
                    "fitsio_numpy",
                ),
                "read_batch": _cpu_wrap(
                    lambda h=hdu: _read_batch_only(paths, h),
                    cpu_track,
                    "read_batch",
                ),
                "read_fitsio": _cpu_wrap(
                    lambda h=hdu: _read_fitsio_only(paths, h),
                    cpu_track,
                    "read_fitsio",
                ),
            },
            runs=runs,
            warmup=warmup,
        )
        # Compute-only: stack loaded once outside the timed window (the read
        # engine is not under test here), then just the order-statistic.
        stack_once = torch.stack(torchfits.read_batch(paths, hdu=hdu, strict=True))
        timed.update(
            time_medians_interleaved(
                {
                    "compute_torch": _cpu_wrap(
                        lambda s=stack_once: _median_compute_only(s, use_torch=True),
                        cpu_track,
                        "compute_torch",
                    ),
                    "compute_numpy": _cpu_wrap(
                        lambda s=stack_once: _median_compute_only(s, use_torch=False),
                        cpu_track,
                        "compute_numpy",
                    ),
                },
                runs=runs,
                warmup=warmup,
            )
        )
        if urls:
            timed.update(
                time_medians_interleaved(
                    {
                        "slice_stream_url": _cpu_wrap(
                            lambda h=hdu: _read_median_slices(
                                urls, h, slice_rows=slice_rows, threads=threads
                            ),
                            cpu_track,
                            "slice_stream_url",
                        ),
                    },
                    runs=runs,
                    warmup=warmup,
                )
            )
        method_meta = (
            ("batch_whole_plane", "torchfits", "pipeline"),
            ("slice_stream_local", "torchfits", "pipeline"),
            ("dataloader_stack", "torchfits", "pipeline"),
            ("fitsio_numpy", "fitsio", "pipeline"),
            ("slice_stream_url", "torchfits", "pipeline"),
            ("read_batch", "torchfits", "io_read"),
            ("read_fitsio", "fitsio", "io_read"),
            ("compute_torch", "torch", "io_compute"),
            ("compute_numpy", "numpy", "io_compute"),
        )
        for method, library, family in method_meta:
            if method not in timed:
                continue
            t_val, peak_rss, peak_cuda, err = timed[method]
            status = "OK" if t_val is not None else "FAILED"
            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "median_stack",
                    "case_id": case_id,
                    "case_label": f"median {len(paths)} MEFs hdu {hdu}",
                    "operation": "median_stack",
                    "family": family,
                    "library": library,
                    "method": method,
                    "mode": family,
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
                    "n_points": len(paths),
                    "metadata": json.dumps(
                        {
                            "hdu": hdu,
                            "n_files": len(paths),
                            "slice_rows": slice_rows,
                            "threads": threads,
                            "plane_pixels": plane_pixels,
                            "cpu_s": float(np.median(cpu_track[method]))
                            if cpu_track.get(method)
                            else None,
                            "cpu_over_wall": (
                                float(np.median(cpu_track[method]) / t_val)
                                if cpu_track.get(method) and t_val
                                else None
                            ),
                            "loc": _loc(
                                {
                                    "batch_whole_plane": _read_median_batch,
                                    "slice_stream_local": _read_median_slices,
                                    "dataloader_stack": _read_median_dataloader,
                                    "fitsio_numpy": _read_median_fitsio,
                                    "slice_stream_url": _read_median_slices,
                                    "read_batch": _read_batch_only,
                                    "read_fitsio": _read_fitsio_only,
                                    "compute_torch": _median_compute_only,
                                    "compute_numpy": _median_compute_only,
                                }[method]
                            ),
                            "batch_info": {str(k): v for k, v in (info or {}).items()},
                        }
                    ),
                }
            )

    # --- pipeline artifacts (untimed API writes) ---
    # Artifacts use one consistent source: URLs when given, else local files.
    src_paths = urls if urls else paths
    out_mef = out_dir / "median_out.fits"
    planes = [
        _read_median_slices(src_paths, h, slice_rows=slice_rows, threads=threads)
        for h in hdus
    ]
    _write_median_mef(str(out_mef), planes)
    torchfits.write_checksums(str(out_mef))
    # Reference for CLI diff: independent read path so validation shares no
    # code with the artifact: fitsio for local files, torchfits readers with
    # a numpy median for URL mode. Written back through torchfits so headers
    # match the out MEF exactly (cli diff requires identical headers).
    ref = out_dir / "median_ref.fits"
    if urls:
        ref_planes = [
            _slice_median_stack(
                urls,
                h,
                slice_rows=slice_rows,
                threads=threads,
                open_reader=torchfits.open_subset_reader,
                median_fn=_median_numpy,
            )
            for h in hdus
        ]
    else:
        ref_planes = [
            _read_median_fitsio(paths, h, slice_rows=slice_rows) for h in hdus
        ]
    _write_median_mef(str(ref), ref_planes)
    del ref_planes

    # --- timed CLI cases on the artifacts ---
    cli_cases: list[tuple[str, list[str], str]] = [
        ("cli_info", ["info", str(out_mef)], "info"),
        (
            "cli_stats",
            ["stats", str(out_mef), "--hdu", "0", "-j", "4"],
            "stats",
        ),
        ("cli_verify", ["verify", str(out_mef)], "verify"),
        ("cli_diff", ["diff", str(ref), str(out_mef)], "diff"),
        (
            "cli_cutout",
            [
                "cutout",
                str(out_mef),
                "--hdu",
                "0",
                "--box",
                f"{_CUTOUT_BOX[0]},{_CUTOUT_BOX[1]},{_CUTOUT_BOX[2]},{_CUTOUT_BOX[3]}",
                "-o",
                str(out_dir / "median_cutout.fits"),
            ],
            "cutout",
        ),
    ]
    if urls:
        cli_cases.append(("cli_probe", ["probe", urls[0]], "probe"))
    for method, argv, _op in cli_cases:
        t_val, rc, tail = _run_cli(argv)
        status = "OK" if rc == 0 else "FAILED"
        err = "" if rc == 0 else f"exit {rc}: {tail}"
        rows.append(
            {
                "run_id": run_id,
                "domain": "fits",
                "suite": "median_stack",
                "case_id": method,
                "case_label": f"torchfits cli {method.replace('cli_', '')} on median MEF",
                "operation": method,
                "family": "cli",
                "library": "cli",
                "method": method,
                "mode": "cli",
                "status": status,
                "skip_reason": err,
                "comparable": status == "OK",
                "mmap_target": "n/a",
                "host": _BENCH_HOST,
                "time_s": t_val,
                "peak_rss_mb": None,
                "peak_cuda_alloc_mb": None,
                "throughput": None,
                "unit": "",
                "size_mb": 0.0,
                "n_points": 1,
                "metadata": json.dumps({"artifact": out_mef.name, "tail": tail}),
            }
        )
    annotate_rankings(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--urls", default="", help="comma-separated https URL list")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--max-files", type=int, default=6)
    parser.add_argument("--max-hdus", type=int, default=2)
    parser.add_argument("--slice-rows", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    # Inherited envs (e.g. OMP_NUM_THREADS=1) silently cap torch ops to one
    # OMP thread; the compute paths must use the same thread count as the
    # read fan-out or the io_compute/pipeline numbers are misleading.
    torch.set_num_threads(max(1, args.threads))
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_local_paths(args.data_dir, max_files=args.max_files)
    urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    rows = run_median_stack_rows(
        run_id=run_id,
        paths=paths,
        urls=urls or None,
        out_dir=run_dir,
        profile=args.profile,
        max_hdus=args.max_hdus,
        slice_rows=args.slice_rows,
        threads=args.threads,
    )
    out_csv = run_dir / "median_stack_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} median_stack rows to {out_csv}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
