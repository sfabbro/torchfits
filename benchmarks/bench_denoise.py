#!/usr/bin/env python3
"""Cosmic-ray filtering by learning: Noise2Noise, dark->blank, real MegaCam.

Training inputs are real CFHT MegaCam DARK calibration frames (zero exposure
— a zero-valued field IS the perfect Noise2Noise twin: every dark is an
independent noisy draw of the same blank, so pairs are free and never share
pixels). The net learns ``dark -> 0`` (L1 against a blank target): cosmic-ray
hits and detector noise are not in the target, so the optimal prediction
ignores them.

Evaluation is on real SCIENCE frames: CR-like outlier suppression (compact
bright pixels, before vs after) is reported per patch. There is no paired
clean target on science — a PSNR against the input would be meaningless here
(the honest science metrics, incl. star flux and CR-removal at known
positions, live in ``examples/example_megacam_cr_denoise.py`` and
``docs/denoise-pipeline.md``).

Stages (timed, interleaved):

- ``pair_meta``    calibration-pair discovery by header keys (``read_keys``
                   vs ``getheader``)
- ``pair_read``    both members in one ``read_batch`` vs per-file ``getdata``
- ``train_epoch``  one training epoch over dark patches: torchfits composes
                   ``FitsCutoutDataset`` (dark j) + ``FitsCutoutDataset``
                   (dark k) + ``torch.utils.data.StackDataset`` +
                   ``make_loader`` (default ``fits_collate_fn``) + a
                   pair-global median/MAD normalization transform (the
                   standard Noise2Noise setup — per-patch min/max would let
                   CR spikes set the scale); astropy does the same work with
                   ``section`` reads and numpy glue. Same tiny net, same
                   step schedule — only the data path differs.
- ``infer_cr``     denoise held-out patches of a real science frame with the
                   trained net; report the CR-like outlier fraction (pixels
                   > 5 sigma above a local box background) before vs after —
                   a diagnostic only: on a live field there is no clean
                   target, and the trained net can amplify bright star cores,
                   so this fraction is not a CR-suppression guarantee. The
                   honest CR metrics (removal at known CR positions, star
                   flux, background drift) live in
                   ``examples/example_megacam_cr_denoise.py``.

Extras:
- ``operation=code_lines`` rows record implementation size per library.
- Data layout: darks live under ``calib/darks/`` (fetch via
  ``scripts/fetch_cfht_calib_frames.sh``); science frames at the top level
  (``scripts/fetch_cfht_megacam_sample.sh``).

Composability notes (measured while writing this file):

- ``FitsCutoutDataset`` accepts ``(path, hdu, x, y, size)`` specs — patch
  grids are one comprehension.
- ``StackDataset(FitsCutoutDataset(j), FitsCutoutDataset(k))`` collates with
  the default ``fits_collate_fn`` ((Tensor, Tensor) branch) — no custom
  collate needed.
- Known cost, kept honest: ``FitsCutoutDataset`` re-opens the file per row
  (documented); the epoch case measures it. A pre-read tensor dataset is
  shown separately in ``pair_read``.
- ``open_subset_reader`` still lacks ``.dtype`` (payload probes needed) —
  same finding as ``bench_science_pipeline`` / ``bench_http_stream``.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from benchmarks.bench_fixtures import discover_local_paths  # noqa: E402
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

_BENCH_HOST = socket.gethostname()

_DEFAULT_DATA = ROOT / "benchmarks_data" / "cfht_megacam"

_PATCH = 128
_BATCH = 8
_EPOCHS = 1
_LR = 1e-3
_CR_SIGMA = 5.0
_CR_BOX = 7


class _DenoiseNet(nn.Module):
    """Tiny 4-conv denoiser: demo grade, torch-only, no new dependencies.

    Downsample x2 (stride-2 convs), then symmetric upsampling; the last
    layer maps back to the input domain. ~25k parameters. Swap for a full
    U-Net without touching the bench structure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 16, 3, padding=1)
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        h3 = F.relu(self.enc3(h2))
        d3 = F.relu(self.dec3(F.interpolate(h3, scale_factor=2, mode="nearest")))
        d2 = F.relu(self.dec2(F.interpolate(d3, scale_factor=2, mode="nearest")))
        return self.head(d2 + h1)


# ---------------------------------------------------------------------------
# Stage implementations — module level so code-lines counting is stable.
# ---------------------------------------------------------------------------


def _tf_pair_meta(pair: list[str], hdu: int) -> int:
    return sum(
        len(torchfits.read_keys(p, ["OBJECT", "EXPTIME", "DATE-OBS"], hdu=hdu))
        for p in pair
    )


def _ap_pair_meta(pair: list[str], hdu: int) -> int:
    from astropy.io import fits

    n = 0
    for p in pair:
        h = fits.getheader(p, hdu)
        n += sum(1 for k in ("OBJECT", "EXPTIME", "DATE-OBS") if k in h)
    return n


def _tf_pair_read(pair: list[str], hdu: int) -> torch.Tensor:
    return torch.stack(torchfits.read_batch(pair, hdu=hdu, strict=True))


def _ap_pair_read(pair: list[str], hdu: int) -> np.ndarray:
    from astropy.io import fits

    return np.stack(
        [np.asarray(fits.getdata(p, hdu, memmap=False), dtype=np.float32) for p in pair]
    )


def _tf_train_epoch(spec: dict[str, Any], net: nn.Module, batch_size: int) -> int:
    o_ds = torchfits.data.FitsCutoutDataset(
        [(spec["o_path"], spec["hdu"], x, y, s) for x, y, s in spec["patches"]],
        transform=spec["transform"],
    )
    p_ds = torchfits.data.FitsCutoutDataset(
        [(spec["p_path"], spec["hdu"], x, y, s) for x, y, s in spec["patches"]],
        transform=spec["transform"],
    )
    loader = torchfits.data.make_loader(
        torch.utils.data.StackDataset(o_ds, p_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        optimize_cache=False,
    )
    optim = torch.optim.Adam(net.parameters(), lr=_LR)
    steps = 0
    for xs, _ys in loader:
        # dark -> blank: the target is a zero (blank) field; CRs and noise
        # are absent from the target, so the optimum ignores them.
        target = torch.zeros_like(xs)
        optim.zero_grad()
        loss = F.l1_loss(net(xs), target)
        loss.backward()
        optim.step()
        steps += 1
    return steps


def _ap_train_epoch(spec: dict[str, Any], net: nn.Module, batch_size: int) -> int:
    from astropy.io import fits

    # memmap=False: BZERO-scaled fz files cannot be section-mmap'd, so astropy
    # loads the whole plane — the honest cost on this data.
    with (
        fits.open(spec["o_path"], memmap=False) as o_h,
        fits.open(spec["p_path"], memmap=False) as p_h,
    ):
        o_data = o_h[spec["hdu"]].data
        p_data = p_h[spec["hdu"]].data
        med, mad = spec["med"], spec["mad"]
        pairs = []
        for x, y, s in spec["patches"]:
            o_patch = np.asarray(o_data[y : y + s, x : x + s], dtype=np.float32)
            p_patch = np.asarray(p_data[y : y + s, x : x + s], dtype=np.float32)
            o_patch = (o_patch - med) / mad
            p_patch = (p_patch - med) / mad
            pairs.append((o_patch[None], p_patch[None]))
    optim = torch.optim.Adam(net.parameters(), lr=_LR)
    steps = 0
    for i in range(0, len(pairs), batch_size):
        xs = torch.as_tensor(np.stack([p[0] for p in pairs[i : i + batch_size]]))
        optim.zero_grad()
        loss = F.l1_loss(net(xs), torch.zeros_like(xs))
        loss.backward()
        optim.step()
        steps += 1
    return steps


def _cr_stats(patch: torch.Tensor) -> tuple[float, float]:
    """CR-like outlier fraction: isolated bright pixels, not structure.

    A CR is a compact spike against the local background: ``x`` must exceed
    the 7x7 box background by > 5 sigma (MAD of the residual) AND exceed its
    own 3x3 neighbourhood median by > 5 sigma — real sources/edges pass the
    first test but not the second (their neighbours are bright too).
    """
    bg = F.avg_pool2d(patch, _CR_BOX, stride=1, padding=_CR_BOX // 2)
    resid = patch - bg
    _b, _c, _h, _w = resid.shape
    sigma = (
        resid.abs()
        .reshape(_b, _c, _h * _w)
        .median(dim=-1, keepdim=True)
        .values.reshape(_b, _c, 1, 1)
        .clamp_min(1e-9)
    )
    neighbors = (
        F.unfold(patch, kernel_size=3, padding=1).median(dim=1).values.view_as(patch)
    )
    is_cr = (resid > _CR_SIGMA * sigma) & (patch > neighbors + _CR_SIGMA * sigma)
    return (
        float(is_cr.float().mean().item()),
        float(sigma.mean().item()),
    )


def _tf_eval_cr(spec: dict[str, Any], net: nn.Module) -> dict[str, float]:
    """CR-like outlier suppression on a real science frame (no clean target)."""
    ds = torchfits.data.FitsCutoutDataset(
        [(spec["s_path"], spec["hdu"], x, y, s) for x, y, s in spec["patches"]],
        transform=spec["transform"],
    )
    loader = torchfits.data.make_loader(
        ds,
        batch_size=_BATCH,
        shuffle=False,
        num_workers=0,
        optimize_cache=False,
    )
    with torch.no_grad():
        before = 0.0
        after = 0.0
        n = 0
        for xs in loader:
            b0, _ = _cr_stats(xs)
            out = net(xs)
            b1, _ = _cr_stats(out)
            before += b0 * xs.shape[0]
            after += b1 * xs.shape[0]
            n += xs.shape[0]
    return {
        "before_cr_frac": before / n,
        "after_cr_frac": after / n,
    }


def _ap_eval_cr(spec: dict[str, Any], net: nn.Module) -> dict[str, float]:
    from astropy.io import fits

    # memmap=False: same BZERO-scaled fz limitation as _ap_train_epoch.
    with fits.open(spec["s_path"], memmap=False) as s_h:
        s_data = s_h[spec["hdu"]].data
        med, mad = spec["med"], spec["mad"]
        with torch.no_grad():
            before = 0.0
            after = 0.0
            n = 0
            for x, y, s in spec["patches"]:
                s_patch = np.asarray(s_data[y : y + s, x : x + s], dtype=np.float32)
                s_patch = (s_patch - med) / mad
                xs = torch.as_tensor(s_patch[None, None])
                b0, _ = _cr_stats(xs)
                out = net(xs)
                b1, _ = _cr_stats(out)
                before += b0
                after += b1
                n += 1
    return {
        "before_cr_frac": before / n,
        "after_cr_frac": after / n,
    }


_LOC_FNS: dict[str, dict[str, Any]] = {
    "pair_meta": {"torchfits": _tf_pair_meta, "astropy": _ap_pair_meta},
    "pair_read": {"torchfits": _tf_pair_read, "astropy": _ap_pair_read},
    "train_epoch": {"torchfits": _tf_train_epoch, "astropy": _ap_train_epoch},
    "infer_cr": {"torchfits": _tf_eval_cr, "astropy": _ap_eval_cr},
}


# ---------------------------------------------------------------------------
# Helpers (not timed, not code-counted)
# ---------------------------------------------------------------------------


def _pair_robust_stats(o_path: str, p_path: str, hdu: int) -> tuple[float, float]:
    planes = torch.stack(torchfits.read_batch([o_path, p_path], hdu)).float()
    med = float(planes.median())
    mad = float((planes - med).abs().median())
    return med, mad


class _GlobalNorm(torchfits.transforms.FITSTransform):
    """Pair-global median/MAD normalization (the Noise2Noise convention)."""

    def __init__(self, med: float, mad: float) -> None:
        super().__init__()
        self.med = med
        self.mad = mad

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del mask  # unused: normalization is global, not mask-aware
        return (x.float() - self.med) / max(self.mad, 1e-6)

    def __repr__(self) -> str:
        return f"_GlobalNorm(med={self.med:.1f}, mad={self.mad:.1f})"


def _patch_grid(
    naxis1: int, naxis2: int, size: int, *, seed: int
) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, max(1, naxis1 - size), size=24)
    ys = rng.integers(0, max(1, naxis2 - size), size=24)
    return [(int(x), int(y), size) for x, y in zip(xs, ys)]


def _discover_pairs(paths: list[str]) -> list[list[str]]:
    """Two darks are a training pair: each is an independent noisy draw of
    the same blank field (dark->blank Noise2Noise)."""
    darks = [p for p in paths if "calib/darks" in str(Path(p).parent.as_posix())]
    if len(darks) < 2:
        return []
    return [sorted(darks[:2])]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_denoise_rows(
    *,
    run_id: str,
    paths: list[str],
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
    max_hdus: int = 1,
    patch: int = _PATCH,
    epochs: int = _EPOCHS,
    batch: int = _BATCH,
) -> list[dict[str, Any]]:
    if runs is None:
        runs = 3 if profile == "user" else 5
    if warmup is None:
        warmup = 1 if profile == "user" else 2
    rows: list[dict[str, Any]] = []
    pairs = _discover_pairs(paths)
    sciences = [p for p in paths if "calib/darks" not in str(Path(p).parent.as_posix())]
    if len(pairs) < 1:
        print(
            "[denoise] need >=2 dark frames under calib/darks/; "
            "run scripts/fetch_cfht_calib_frames.sh",
            flush=True,
        )
        return rows
    if not sciences:
        print(
            "[denoise] need >=1 science frame for CR evaluation; "
            "run scripts/fetch_cfht_megacam_sample.sh",
            flush=True,
        )
        return rows
    train_pair = pairs[0]
    eval_pair = [sciences[0]]
    for hdu in range(1, max_hdus + 1):
        for tag, pair in (("train", train_pair),):
            _ndim, shape = torchfits.read_shape(pair[0], hdu)
            naxis1, naxis2 = int(shape[-1]), int(shape[-2])
            patches = _patch_grid(naxis1, naxis2, patch, seed=hdu + len(pair[0]))
            med, mad = _pair_robust_stats(pair[0], pair[1], hdu)
            spec = {
                "o_path": pair[0],
                "p_path": pair[1],
                "hdu": hdu,
                "patches": patches,
                "med": med,
                "mad": mad,
                "transform": _GlobalNorm(med, mad),
            }
            pair_mb = 2 * naxis1 * naxis2 * 4 / (1024.0 * 1024.0)
            patch_mb = len(patches) * patch * patch * 4 / (1024.0 * 1024.0)
            label = f"{Path(pair[0]).name} x {Path(pair[1]).name}"
            case_base = f"denoise_{tag}_hdu{hdu}::n{len(patches)}"

            stages: list[tuple[str, dict[str, Any]]] = [
                (
                    "pair_meta",
                    {
                        "tf": (
                            lambda: _tf_pair_meta(pair, hdu),
                            "torchfits",
                            "read_keys",
                        ),
                        "ap": (
                            lambda: _ap_pair_meta(pair, hdu),
                            "astropy",
                            "getheader",
                        ),
                        "size_mb": 0.0,
                        "n_points": 2,
                    },
                ),
                (
                    "pair_read",
                    {
                        "tf": (
                            lambda: _tf_pair_read(pair, hdu),
                            "torchfits",
                            "read_batch",
                        ),
                        "ap": (
                            lambda: _ap_pair_read(pair, hdu),
                            "astropy",
                            "getdata",
                        ),
                        "size_mb": pair_mb,
                        "n_points": naxis1 * naxis2,
                    },
                ),
                (
                    "train_epoch",
                    {
                        "tf": (
                            lambda: _tf_train_epoch(spec, _DenoiseNet(), batch),
                            "torchfits",
                            "StackDataset",
                        ),
                        "ap": (
                            lambda: _ap_train_epoch(spec, _DenoiseNet(), batch),
                            "astropy",
                            "section",
                        ),
                        "size_mb": patch_mb,
                        "n_points": len(patches),
                    },
                ),
            ]
            for op, case_spec in stages:
                print(
                    f"[denoise] case={case_base}::{op} runs={runs} "
                    f"size={case_spec['size_mb']:.2f}MB n_points={case_spec['n_points']}",
                    flush=True,
                )
                methods = {
                    key: fn
                    for key, value in case_spec.items()
                    if isinstance(value, tuple) and value[0] is not None
                    for fn in (value[0],)
                }
                timed = time_medians_interleaved(methods, runs=runs, warmup=warmup)
                for key, value in case_spec.items():
                    if not isinstance(value, tuple):
                        continue
                    fn, lib, api = value
                    if key not in timed:
                        continue
                    t_val, peak_rss, peak_cuda, err = timed[key]
                    status = "OK" if t_val is not None else "FAILED"
                    rows.append(
                        {
                            "run_id": run_id,
                            "domain": "fits",
                            "suite": "denoise",
                            "case_id": f"{case_base}::{op}",
                            "case_label": f"{label} {op} (hdu {hdu})",
                            "operation": op,
                            "family": "pipeline",
                            "library": lib,
                            "method": f"{lib}_{key}_{api}",
                            "mode": "pipeline",
                            "status": status,
                            "skip_reason": "" if err is None else str(err),
                            "comparable": status == "OK",
                            "mmap_target": "n/a",
                            "host": _BENCH_HOST,
                            "time_s": t_val,
                            "peak_rss_mb": peak_rss,
                            "peak_cuda_alloc_mb": peak_cuda,
                            "throughput": (case_spec["size_mb"] / t_val)
                            if t_val and case_spec["size_mb"]
                            else None,
                            "unit": "MB/s",
                            "size_mb": case_spec["size_mb"],
                            "n_points": case_spec["n_points"],
                            "metadata": json.dumps(
                                {
                                    "hdu": hdu,
                                    "api": api,
                                    "tag": tag,
                                    "epochs": epochs,
                                    "patch": patch,
                                    "threads": torch.get_num_threads(),
                                }
                            ),
                        }
                    )

        # code-lines rows per stage per library (module-level impls, not the
        # timing closures) — static, so once per hdu, not once per tag.
        for op, case_spec in stages:
            for lib, fn in sorted(_LOC_FNS[op].items()):
                lines, imports = code_lines(fn)
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "denoise",
                        "case_id": f"{case_base}::{op}",
                        "case_label": f"{label} {op} (hdu {hdu})",
                        "operation": "code_lines",
                        "family": "pipeline",
                        "library": lib,
                        "method": f"lo_{lib}_{op}",
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

        # CR filtering on the science frame with a net trained on darks.
        train_dims = torchfits.read_shape(train_pair[0], hdu)[1]
        eval_dims = torchfits.read_shape(eval_pair[0], hdu)[1]
        train_spec = {
            "o_path": train_pair[0],
            "p_path": train_pair[1],
            "hdu": hdu,
            "patches": _patch_grid(
                int(train_dims[-1]), int(train_dims[-2]), patch, seed=hdu
            ),
        }
        train_med, train_mad = _pair_robust_stats(train_pair[0], train_pair[1], hdu)
        train_spec.update(
            med=train_med, mad=train_mad, transform=_GlobalNorm(train_med, train_mad)
        )
        eval_spec = {
            "s_path": eval_pair[0],
            "hdu": hdu,
            "patches": _patch_grid(
                int(eval_dims[-1]), int(eval_dims[-2]), patch, seed=hdu
            ),
        }
        eval_med, eval_mad = _pair_robust_stats(train_pair[0], train_pair[1], hdu)
        eval_spec.update(
            med=eval_med, mad=eval_mad, transform=_GlobalNorm(eval_med, eval_mad)
        )
        print(
            f"[denoise] case=denoise_infer_cr_hdu{hdu} runs=1 (train on "
            f"{Path(train_pair[0]).name}, eval on {Path(eval_pair[0]).name})",
            flush=True,
        )
        for key, eval_fn, lib, api in (
            ("tf", _tf_eval_cr, "torchfits", "dataset"),
            ("ap", _ap_eval_cr, "astropy", "section"),
        ):
            net = _DenoiseNet()
            for _ in range(epochs):
                _tf_train_epoch(train_spec, net, batch)
            t0 = time.monotonic()
            metrics = eval_fn(eval_spec, net)
            t_val = time.monotonic() - t0
            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "denoise",
                    "case_id": f"denoise_infer_cr_hdu{hdu}::n{len(eval_spec['patches'])}",
                    "case_label": (
                        f"CR filter: train {Path(train_pair[0]).name}, "
                        f"eval {Path(eval_pair[0]).name} (hdu {hdu})"
                    ),
                    "operation": "infer_cr",
                    "family": "pipeline",
                    "library": lib,
                    "method": f"{lib}_{key}_{api}",
                    "mode": "inference",
                    "status": "OK",
                    "skip_reason": "",
                    "comparable": False,
                    "mmap_target": "n/a",
                    "host": _BENCH_HOST,
                    "time_s": t_val,
                    "peak_rss_mb": None,
                    "peak_cuda_alloc_mb": None,
                    "throughput": None,
                    "unit": "n/a",
                    "size_mb": None,
                    "n_points": len(eval_spec["patches"]),
                    "metadata": json.dumps(
                        {
                            "hdu": hdu,
                            "api": api,
                            "epochs": epochs,
                            **metrics,
                        }
                    ),
                }
            )

        for lib, fn in sorted(_LOC_FNS["infer_cr"].items()):
            lines, imports = code_lines(fn)
            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "denoise",
                    "case_id": f"denoise_infer_cr_hdu{hdu}",
                    "case_label": "CR filter stage (hdu {hdu})".format(hdu=hdu),
                    "operation": "code_lines",
                    "family": "pipeline",
                    "library": lib,
                    "method": f"lo_{lib}_infer_cr",
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

    annotate_rankings(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-hdus", type=int, default=1)
    parser.add_argument("--patch", type=int, default=_PATCH)
    parser.add_argument("--epochs", type=int, default=_EPOCHS)
    parser.add_argument("--batch", type=int, default=_BATCH)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    paths = discover_local_paths(args.data_dir, max_files=args.max_files)
    paths += discover_local_paths(args.data_dir / "calib" / "darks", max_files=10)
    if not paths:
        print(
            f"[denoise] no FITS files under {args.data_dir}; "
            "run scripts/fetch_cfht_calib_frames.sh (darks) and "
            "scripts/fetch_cfht_megacam_sample.sh (science)",
            flush=True,
        )
        return 1
    rows = run_denoise_rows(
        run_id=run_id,
        paths=paths,
        profile=args.profile,
        max_hdus=args.max_hdus,
        patch=args.patch,
        epochs=args.epochs,
        batch=args.batch,
    )
    out_csv = run_dir / "denoise_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} denoise rows to {out_csv}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
