#!/usr/bin/env python3
"""Exemplary science pipeline: robust coadd + ML cutout serving, torchfits vs astropy.io.fits.

The story: N CFHT MegaCam exposures (MEF, 40 extensions each) are reduced
into one robust coadd — per-pixel sigma-clipped mean over the stack, which
rejects cosmic rays / satellite trails — and the coadd plus per-exposure
cutouts are served as ML-ready batches. Each stage composes the public
torchfits API; the astropy.io.fits equivalent implements the same work with
numpy glue.

Stages (timed, interleaved, per HDU):

- ``select_meta``     header-only metadata loop (``read_keys`` vs ``getheader``)
- ``stack_read``      one C++ batch pass (``read_batch``) vs per-file ``getdata``
- ``reject_clip``     ``transforms.SigmaClip`` over dim 0 vs ``astropy.stats.sigma_clip``
- ``coadd_write``     median coadd + ``write`` + ``write_checksums`` vs ``writeto``
- ``serve_cutouts``   ``FitsCutoutDataset`` + ``Compose`` + ``make_loader`` vs
                      manual ``section`` loop; a reader-level variant
                      (``open_subset_reader``) is measured alongside
- ``serve_http``      same serving over CADC URLs: range cutouts vs whole-file
                      download (``--http`` only)

Code-lines rows (``operation=code_lines``) record the implementation size of
each library function (source lines minus blanks/comments), so the
"easy to code" claim is measured, not asserted.

Composability audit (friction found while writing this file):

1. ``read_batch`` returns a *list* of tensors — the stack stage needs an
   explicit ``torch.stack``. One line, acceptable; a ``stack=True`` kwarg
   would remove it.
2. ``open_subset_reader`` exposes ``.shape`` but no ``.dtype`` — payload
   accounting forced a probe read (``read_subset(0, 0, 8, 8)``). A
   ``.dtype`` property would remove the probe.
3. ``FitsCutoutDataset`` re-opens the file per row (documented); the
   reader-level variant measures the no-reopen path. Both compose with the
   same ``Compose`` transform — good.
4. dtype normalization is a real ease-of-use win: torchfits returns f32
   (BZERO/BSCALE applied) uniformly, while fitsio/astropy return raw u16 /
   f64 section views, forcing casts in the numpy paths.

Everything else (``read_keys``/``read_shape`` meta, ``SigmaClip(dim=(0,))``,
``write([...])`` multi-plane, ``Compose`` + ``MinMaxNormalize``,
``make_loader``) composed with zero friction.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402

from benchmarks.bench_contract import (  # noqa: E402
    RESULT_COLUMNS,
    annotate_rankings,
    code_lines,
    make_run_id,
    write_csv,
)
from benchmarks.bench_fixtures import (  # noqa: E402
    cadc_direct_urls,
    discover_fits_names,
    discover_local_paths,
)
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

_BENCH_HOST = socket.gethostname()

_DEFAULT_DATA = ROOT / "benchmarks_data" / "cfht_megacam"

_CUTOUT_SIZE = 128
_CUTOUTS_PER_FILE = 4
_CLIP_SIGMA = 3.0
_CLIP_MAX_ITER = 5


# ---------------------------------------------------------------------------
# Stage implementations — module level so code-lines counting is stable.
# ---------------------------------------------------------------------------


def _tf_meta(paths: list[str], hdu: int) -> int:
    return sum(
        len(torchfits.read_keys(p, ["DATE-OBS", "EXPTIME", "OBJECT"], hdu=hdu))
        for p in paths
    )


def _ap_meta(paths: list[str], hdu: int) -> int:
    from astropy.io import fits

    n = 0
    for p in paths:
        h = fits.getheader(p, hdu)
        n += sum(1 for k in ("DATE-OBS", "EXPTIME", "OBJECT") if k in h)
    return n


def _tf_stack(paths: list[str], hdu: int) -> torch.Tensor:
    return torch.stack(torchfits.read_batch(paths, hdu=hdu, strict=True))


def _ap_stack(paths: list[str], hdu: int) -> np.ndarray:
    from astropy.io import fits

    planes = []
    for p in paths:
        planes.append(np.asarray(fits.getdata(p, hdu, memmap=False), dtype=np.float32))
    return np.stack(planes, axis=0)


def _tf_clip(stack: torch.Tensor) -> torch.Tensor:
    return torchfits.transforms.SigmaClip(
        n_sigma=_CLIP_SIGMA, max_iter=_CLIP_MAX_ITER, dim=(0,), fill="mean"
    )(stack)


def _ap_clip(stack: np.ndarray) -> np.ndarray:
    from astropy.stats import sigma_clip

    clipped = sigma_clip(stack, sigma=_CLIP_SIGMA, maxiters=_CLIP_MAX_ITER, axis=0)
    return np.where(clipped.mask, clipped.mean(axis=0), clipped.data)


def _tf_coadd_write(stack: torch.Tensor, out_path: str) -> int:
    coadd = stack.median(dim=0).values
    torchfits.write(out_path, [coadd], overwrite=True)
    torchfits.write_checksums(out_path)
    return int(coadd.numel())


def _ap_coadd_write(stack: np.ndarray, out_path: str) -> int:
    from astropy.io import fits

    coadd = np.median(stack, axis=0)
    fits.writeto(out_path, coadd, overwrite=True)
    return int(coadd.size)


def _tf_serve(cutouts: list[Any], batch_size: int, transform: Any) -> int:
    ds = torchfits.data.FitsCutoutDataset(cutouts, transform=transform)
    loader = torchfits.data.make_loader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, optimize_cache=False
    )
    return sum(int(batch.shape[0]) for batch in loader)


def _tf_serve_reader(cutouts: list[Any], transform: Any) -> int:
    seen = 0
    handles: dict[str, Any] = {}
    try:
        for path, hdu, x, y, s in cutouts:
            reader = handles.get(path)
            if reader is None:
                reader = torchfits.open_subset_reader(path, hdu=hdu)
                handles[path] = reader
            transform(reader.read_subset(x, y, x + s, y + s))
            seen += 1
    finally:
        for reader in handles.values():
            reader.close()
    return seen


def _ap_serve(cutouts: list[Any]) -> int:
    from astropy.io import fits

    seen = 0
    handles: dict[str, Any] = {}
    try:
        for path, hdu, x, y, s in cutouts:
            handle = handles.get(path)
            if handle is None:
                # memmap=False: BZERO-scaled fz files cannot be section-mmap'd,
                # so astropy loads the whole plane — the honest cost.
                handle = fits.open(path, memmap=False)
                handles[path] = handle
            d = np.asarray(handle[hdu].section[y : y + s, x : x + s], dtype=np.float32)
            vmin, vmax = float(d.min()), float(d.max())
            _ = (d - vmin) / (vmax - vmin + 1e-6)
            seen += 1
    finally:
        for handle in handles.values():
            handle.close()
    return seen


def _tf_serve_http(cutouts: list[Any], transform: Any) -> int:
    return _tf_serve(cutouts, batch_size=8, transform=transform)


def _ap_serve_http(cutouts: list[Any]) -> int:
    from astropy.io import fits

    seen = 0
    handles: dict[str, Any] = {}
    try:
        for path, hdu, x, y, s in cutouts:
            handle = handles.get(path)
            if handle is None:
                handle = fits.open(path, memmap=False)
                handles[path] = handle
            d = np.asarray(handle[hdu].section[y : y + s, x : x + s], dtype=np.float32)
            vmin, vmax = float(d.min()), float(d.max())
            _ = (d - vmin) / (vmax - vmin + 1e-6)
            seen += 1
    finally:
        for handle in handles.values():
            handle.close()
    return seen


# ---------------------------------------------------------------------------
# Helpers (not timed, not code-counted)
# ---------------------------------------------------------------------------


def _build_cutouts(
    paths: list[str], hdu: int, *, per_file: int, size: int
) -> list[tuple[str, int, int, int, int]]:
    out: list[tuple[str, int, int, int, int]] = []
    for p in paths:
        _ndim, shape = torchfits.read_shape(p, hdu)
        naxis1, naxis2 = int(shape[-1]), int(shape[-2])
        rng = np.random.default_rng(hdu + len(p))
        for _ in range(per_file):
            x = int(rng.integers(0, max(1, naxis1 - size)))
            y = int(rng.integers(0, max(1, naxis2 - size)))
            out.append((p, hdu, x, y, size))
    return out


def _code_lines(fn: Any) -> tuple[int, list[str]]:
    return code_lines(fn)


# Module-level stage implementations for the code-lines rows (the stage specs
# carry the timed closures; LOC must count the real functions).
_LOC_FNS: dict[str, dict[str, Any]] = {
    "select_meta": {"tf": _tf_meta, "ap": _ap_meta},
    "stack_read": {"tf": _tf_stack, "ap": _ap_stack},
    "reject_clip": {"tf": _tf_clip, "ap": _ap_clip},
    "coadd_write": {"tf": _tf_coadd_write, "ap": _ap_coadd_write},
    "serve_cutouts": {
        "tf": _tf_serve,
        "tf_reader": _tf_serve_reader,
        "ap": _ap_serve,
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_science_pipeline_rows(
    *,
    run_id: str,
    paths: list[str],
    urls: list[str],
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
    max_hdus: int = 1,
    max_rows: int = 0,
    cutouts_per_file: int = _CUTOUTS_PER_FILE,
    http: bool = False,
) -> list[dict[str, Any]]:
    if runs is None:
        runs = 3 if profile == "user" else 5
    if warmup is None:
        warmup = 1 if profile == "user" else 2
    rows: list[dict[str, Any]] = []
    hdus = list(range(1, max_hdus + 1))
    transform = torchfits.transforms.Compose(
        [torchfits.transforms.MinMaxNormalize(dim=(-2, -1))]
    )

    for hdu in hdus:
        plane_pixels = _plane_pixels(paths, hdu)
        stack_pixels = plane_pixels * len(paths)
        stack_mb = stack_pixels * 4 / (1024.0 * 1024.0)
        case_id = f"pipeline_coadd_hdu{hdu}::n{len(paths)}"
        cutouts = _build_cutouts(
            paths, hdu, per_file=cutouts_per_file, size=_CUTOUT_SIZE
        )
        cutout_pixels = len(cutouts) * _CUTOUT_SIZE * _CUTOUT_SIZE
        cutout_mb = cutout_pixels * 4 / (1024.0 * 1024.0)
        out_mef = ROOT / "benchmarks_results" / run_id / f"coadd_hdu{hdu}.fits"

        stages: list[tuple[str, str, dict[str, Any]]] = [
            (
                "select_meta",
                f"{case_id}::select_meta",
                {
                    "tf": (lambda: _tf_meta(paths, hdu), "torchfits", "read_keys"),
                    "ap": (lambda: _ap_meta(paths, hdu), "astropy", "getheader"),
                    "size_mb": 0.0,
                    "n_points": len(paths),
                },
            ),
            (
                "stack_read",
                f"{case_id}::stack_read",
                {
                    "tf": (lambda: _tf_stack(paths, hdu), "torchfits", "read_batch"),
                    "ap": (lambda: _ap_stack(paths, hdu), "astropy", "getdata"),
                    "size_mb": stack_mb,
                    "n_points": stack_pixels,
                },
            ),
            (
                "reject_clip",
                f"{case_id}::reject_clip",
                {
                    "tf": (
                        lambda: _tf_clip(_tf_stack(paths, hdu)),
                        "torchfits",
                        "SigmaClip",
                    ),
                    "ap": (
                        lambda: _ap_clip(_ap_stack(paths, hdu)),
                        "astropy",
                        "sigma_clip",
                    ),
                    "size_mb": stack_mb,
                    "n_points": stack_pixels,
                },
            ),
            (
                "coadd_write",
                f"{case_id}::coadd_write",
                {
                    "tf": (
                        lambda: _tf_coadd_write(_tf_stack(paths, hdu), str(out_mef)),
                        "torchfits",
                        "write",
                    ),
                    "ap": (
                        lambda: _ap_coadd_write(_ap_stack(paths, hdu), str(out_mef)),
                        "astropy",
                        "writeto",
                    ),
                    "size_mb": plane_pixels * 4 / (1024.0 * 1024.0),
                    "n_points": plane_pixels,
                },
            ),
            (
                "serve_cutouts",
                f"{case_id}::serve_cutouts",
                {
                    "tf": (
                        lambda: _tf_serve(cutouts, batch_size=8, transform=transform),
                        "torchfits",
                        "dataset",
                    ),
                    "tf_reader": (
                        lambda: _tf_serve_reader(cutouts, transform),
                        "torchfits",
                        "reader",
                    ),
                    "ap": (lambda: _ap_serve(cutouts), "astropy", "section"),
                    "size_mb": cutout_mb,
                    "n_points": len(cutouts),
                },
            ),
        ]

        for op, op_case_id, spec in stages:
            print(
                f"[science_pipeline] case={op_case_id} runs={runs} "
                f"size={spec['size_mb']:.2f}MB n_points={spec['n_points']}",
                flush=True,
            )
            methods: dict[str, Any] = {}
            for key, value in spec.items():
                if not isinstance(value, tuple):
                    continue
                fn, _lib, _api = value
                methods[key] = fn
            timed = time_medians_interleaved(methods, runs=runs, warmup=warmup)
            for key, value in spec.items():
                if not isinstance(value, tuple):
                    continue
                fn, lib, api = value
                if key not in timed:
                    continue
                t_val, peak_rss, peak_cuda, err = timed[key]
                status = "OK" if t_val is not None else "FAILED"
                method = f"{lib}_{api}"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "science_pipeline",
                        "case_id": op_case_id,
                        "case_label": f"hdu {hdu} {op} on {len(paths)} files",
                        "operation": op,
                        "family": "pipeline",
                        "library": lib,
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
                        "throughput": (spec["size_mb"] / t_val)
                        if t_val and spec["size_mb"]
                        else None,
                        "unit": "MB/s",
                        "size_mb": spec["size_mb"],
                        "n_points": spec["n_points"],
                        "metadata": json.dumps(
                            {
                                "hdu": hdu,
                                "api": api,
                                "n_files": len(paths),
                                "threads": torch.get_num_threads(),
                            }
                        ),
                    }
                )

        # Code-lines rows: static implementation size per library per stage.
        for op, op_case_id, spec in stages:
            for lib, fn in sorted(_LOC_FNS[op].items()):
                lines, imports = _code_lines(fn)
                lib_name = "torchfits" if lib.startswith("tf") else "astropy"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "science_pipeline",
                        "case_id": op_case_id,
                        "case_label": f"hdu {hdu} {op} on {len(paths)} files",
                        "operation": "code_lines",
                        "family": "pipeline",
                        "library": lib_name,
                        "method": f"lo_{lib_name}_{fn.__name__}",
                        "mode": "static",
                        "status": "OK",
                        "skip_reason": "",
                        "comparable": False,
                        "mmap_target": "n/a",
                        "host": _BENCH_HOST,
                        "time_s": None,
                        "peak_rss_mb": None,
                        "peak_cuda_alloc_mb": None,
                        "throughput": None,
                        "unit": "lines",
                        "size_mb": None,
                        "n_points": None,
                        "metadata": json.dumps({"lines": lines, "imports": imports}),
                    }
                )

        if http and urls:
            op_case_id = f"{case_id}::serve_http"
            http_cutouts: list[Any] = []
            for u in urls:
                _ndim, shape = torchfits.read_shape(u, hdu)
                naxis1, naxis2 = int(shape[-1]), int(shape[-2])
                rng = np.random.default_rng(hdu + len(u))
                for _ in range(cutouts_per_file):
                    x = int(rng.integers(0, max(1, naxis1 - _CUTOUT_SIZE)))
                    y = int(rng.integers(0, max(1, naxis2 - _CUTOUT_SIZE)))
                    http_cutouts.append((u, hdu, x, y, _CUTOUT_SIZE))
            print(
                f"[science_pipeline] case={op_case_id} runs={runs} "
                f"size={cutout_mb:.2f}MB n_points={len(http_cutouts)}",
                flush=True,
            )
            timed = time_medians_interleaved(
                {
                    "tf": lambda: _tf_serve_http(http_cutouts, transform),
                    "ap": lambda: _ap_serve_http(http_cutouts),
                },
                runs=runs,
                warmup=warmup,
            )
            for key, fn, lib, api, method in (
                ("tf", None, "torchfits", "dataset_http", "torchfits_dataset_http"),
                ("ap", None, "astropy", "section_http", "astropy_section_http"),
            ):
                if key not in timed:
                    continue
                t_val, peak_rss, peak_cuda, err = timed[key]
                status = "OK" if t_val is not None else "FAILED"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "science_pipeline",
                        "case_id": op_case_id,
                        "case_label": f"hdu {hdu} http cutouts on {len(urls)} urls",
                        "operation": "serve_http",
                        "family": "pipeline",
                        "library": lib,
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
                        "throughput": (cutout_mb / t_val) if t_val else None,
                        "unit": "MB/s",
                        "size_mb": cutout_mb,
                        "n_points": len(http_cutouts),
                        "metadata": json.dumps(
                            {
                                "hdu": hdu,
                                "api": api,
                                "n_urls": len(urls),
                                "threads": torch.get_num_threads(),
                            }
                        ),
                    }
                )

    annotate_rankings(rows)
    return rows


def _plane_pixels(paths: list[str], hdu: int) -> int:
    _ndim, shape = torchfits.read_shape(paths[0], hdu)
    return int(np.prod(shape))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-hdus", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full planes")
    parser.add_argument("--cutouts-per-file", type=int, default=_CUTOUTS_PER_FILE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--http", action="store_true", help="also run serve_http")
    args = parser.parse_args()
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    paths = discover_local_paths(args.data_dir, max_files=args.max_files)
    if not paths:
        print(
            f"[science_pipeline] no FITS files under {args.data_dir}; "
            "run scripts/fetch_cfht_megacam_sample.sh",
            flush=True,
        )
        return 1
    urls = cadc_direct_urls(
        discover_fits_names(args.data_dir, max_files=args.max_files)
    )
    if args.max_rows:
        # Rebind module-level fns so code-lines + timing see the capped stack.
        global _tf_stack  # noqa: PLW0603
        global _ap_stack  # noqa: PLW0603
        _original_tf_stack = _tf_stack
        _original_ap_stack = _ap_stack

        def _tf_stack_rows(paths: list[str], hdu: int) -> torch.Tensor:
            return _original_tf_stack(paths, hdu)[:, : args.max_rows, :]

        def _ap_stack_rows(paths: list[str], hdu: int) -> np.ndarray:
            return _original_ap_stack(paths, hdu)[:, : args.max_rows, :]

        _tf_stack = _tf_stack_rows
        _ap_stack = _ap_stack_rows

    rows = run_science_pipeline_rows(
        run_id=run_id,
        paths=paths,
        urls=urls,
        profile=args.profile,
        max_hdus=args.max_hdus,
        cutouts_per_file=args.cutouts_per_file,
        http=args.http,
    )
    out_csv = run_dir / "science_pipeline_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} science_pipeline rows to {out_csv}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
