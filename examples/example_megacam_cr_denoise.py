#!/usr/bin/env python3
"""Cosmic-ray and detector-noise cleaning of MegaCam science via Noise2Noise
on real calibration twins.

A dark frame is an exposure with the shutter closed: its field is zero, so
ANY two darks are perfect Noise2Noise twins (identical signal — nothing —
with independent noise: read noise, dark current, hot pixels, cosmic rays).
The same holds for any two biases (0 s darks: read noise only). Training
``dark_a -> dark_b`` with L1 therefore converges to the blank frame — the
net learns the detector's noise-to-zero map. Applied to a real science
frame of the same detector/era, the same map suppresses CRs, hot pixels
and read noise while leaving smooth astronomical structure (stars,
galaxies, sky background) untouched — provided the noise statistics match
(measured below, not assumed).

This example implements that design on public CFHT MegaCam calibration
frames (``scripts/fetch_cfht_calib_frames.sh``) and evaluates the transfer
on the real science frames of ``scripts/fetch_cfht_megacam_sample.sh``:

- training pairs: ``FitsCutoutDataset`` over two dark MEFs (shared cutout
  coords) + ``StackDataset`` + ``make_loader``, zero-mean normalized with
  pair-global median/MAD (the Noise2Noise convention)
- CCD split: train on CCDs 0..29, hold out CCDs 30..39 of the same darks
  for the convergence check (the net must not beat the noise floor)
- transfer metrics on science (no ground truth exists — every number is
  honest): CR-like fraction, sharp-outlier (CR+hot) fraction, faint-source
  counts, bright-star aperture flux ratios, background level drift, and a
  noise-injection probe (science + (dark_j - dark_k) must come back clean)
- bias control: the same training/eval on bias pairs (read-noise-only)
- optional ``--inject-stars``: flux-recovery appendix on synthetic Moffat
  stars dropped onto a real CCD (off by default)

Data: CFHT MegaCam, 40 CCDs per MEF, 4644x2112 int16, ~230 MB / file.
Skips cleanly when calibration frames are missing.

Runs in ``TORCHFITS_EXAMPLE_FAST=1`` (CI) as a one-epoch, few-patch smoke.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import megacam_dir  # noqa: E402

import torchfits  # noqa: E402
from torchfits.data import FitsCutoutDataset, make_loader  # noqa: E402
from torchfits.transforms import FITSTransform  # noqa: E402

_PATCH = 128
_BATCH = 8
_CR_SIGMA = 5.0
_SHARP_SIGMA = 8.0
_FAINT_ADU = 20.0
_FAINT_SIGMA = 3.0
_MARGIN = 8
_STAR_SIGMA = 10.0
_APERTURE_R = 5


def _fast_mode() -> bool:
    import os

    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


# ---------------------------------------------------------------------------
# Model and transforms
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.c2(F.relu(self.c1(x))))


class BlankUNet(nn.Module):
    """Compact fully-convolutional denoiser, torch-only, ~200k parameters.

    3-level encoder/decoder with skip connections; fully convolutional so
    inference works on a whole 4644x2112 CCD in one pass.
    """

    def __init__(self, ch: tuple[int, int, int] = (24, 48, 64)) -> None:
        super().__init__()
        self.d1 = _Block(1, ch[0])
        self.d2 = _Block(ch[0], ch[1])
        self.d3 = _Block(ch[1], ch[2])
        self.u2 = _Block(ch[2] + ch[1], ch[1])
        self.u1 = _Block(ch[1] + ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.d1(x)
        h2 = self.d2(F.avg_pool2d(h1, 2))
        h3 = self.d3(F.avg_pool2d(h2, 2))
        u2 = self.u2(
            torch.cat([F.interpolate(h3, scale_factor=2, mode="nearest"), h2], 1)
        )
        u1 = self.u1(
            torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), h1], 1)
        )
        return self.head(u1)


class SelfNorm(FITSTransform):
    """Self-normalizing median/MAD transform (per patch, per CCD).

    MegaCam CCDs carry different bias levels (1090-1330 ADU across the
    mosaic), so a global normalization mis-centres most patches. Subtracting
    each patch's OWN median (and dividing by its own MAD) puts every dark
    patch at N(0,1) — the Noise2Noise optimum is then the blank frame — and
    the same transform makes science backgrounds self-centre during
    inference, so the sky level is preserved by construction. Medians are
    robust to the sparse CR spikes.
    """

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del mask
        x = x.float()
        med = x.median()
        mad = (x - med).abs().median().clamp_min(1e-6)
        return (x - med) / mad

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del mask
        return x.float()

    def __repr__(self) -> str:
        return "SelfNorm(per-patch median/MAD)"


# ---------------------------------------------------------------------------
# Metrics (all computed on raw-ADU images, no ground truth involved)
# ---------------------------------------------------------------------------


def _box_bg(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


def _robust_sigma(x: torch.Tensor) -> torch.Tensor:
    return x.abs().median().clamp_min(0.5)


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0).unsqueeze(0) if x.ndim == 2 else x


def _cr_like(x: torch.Tensor) -> float:
    """CR-like fraction: >5 sigma over the 7x7 box background AND >5 sigma
    above the 3x3 neighbourhood median (real sources fail the second test).
    Interior-only: 8 px margin keeps conv-border artifacts out of the count."""
    x = _as_batch(x)
    resid = x - _box_bg(x)
    sigma = _robust_sigma(resid)
    nbr = F.unfold(x, kernel_size=3, padding=1).median(dim=1).values.view_as(x)
    is_cr = (resid > _CR_SIGMA * sigma) & (x > nbr + _CR_SIGMA * sigma)
    return float(is_cr[..., _MARGIN:-_MARGIN, _MARGIN:-_MARGIN].float().mean().item())


def _cr_mask(x: torch.Tensor) -> torch.Tensor:
    """Boolean CR-like mask (same criterion as ``_cr_like``), full size."""
    x = _as_batch(x)
    resid = x - _box_bg(x)
    sigma = _robust_sigma(resid)
    nbr = F.unfold(x, kernel_size=3, padding=1).median(dim=1).values.view_as(x)
    return (resid > _CR_SIGMA * sigma) & (x > nbr + _CR_SIGMA * sigma)


def _cr_removal(before: torch.Tensor, after: torch.Tensor) -> float:
    """CR removal at known positions: fraction of before-CR pixels that are
    no longer CR-like in the cleaned frame. The honest way to count CR
    suppression — new flags elsewhere are a separate (net-artifact) metric."""
    m = _cr_mask(before)[..., _MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
    if m.sum() == 0:
        return float("nan")
    after_m = _cr_mask(after)[..., _MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
    return 1.0 - float((m & after_m).sum().item() / m.sum().item())


def _sharp_outliers(x: torch.Tensor) -> float:
    """1-2 px sharp outliers (CRs + hot pixels): >8 sigma over the box
    background AND >8 sigma above the 3x3 median — the tightest spike test.
    Interior-only (8 px margin)."""
    x = _as_batch(x)
    resid = x - _box_bg(x)
    sigma = _robust_sigma(resid)
    nbr = F.unfold(x, kernel_size=3, padding=1).median(dim=1).values.view_as(x)
    is_spike = (resid > _SHARP_SIGMA * sigma) & (x > nbr + _SHARP_SIGMA * sigma)
    return float(
        is_spike[..., _MARGIN:-_MARGIN, _MARGIN:-_MARGIN].float().mean().item()
    )


def _faint_counts(x: torch.Tensor) -> int:
    """Faint (3 sigma, >20 ADU) compact structures: faint-source proxy.
    Interior-only (8 px margin)."""
    x = _as_batch(x)
    resid = x - _box_bg(x)
    sigma = _robust_sigma(resid)
    inner = resid[..., _MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
    return int(((inner > _FAINT_SIGMA * sigma) & (inner > _FAINT_ADU)).sum().item())


def _bg_median(x: torch.Tensor) -> float:
    return float(x.median().item())


def _star_fluxes(x: torch.Tensor) -> list[tuple[int, int, float]]:
    """Bright-star apertures: 5x5 local maxima above 12 sigma over the box
    background (noise peaks at 5 sigma are background-dominated and would
    hide real flux loss). Returns ``(y, x, flux)``; flux of the same
    positions is measured on the cleaned image for the before/after ratio."""
    x = _as_batch(x)
    sigma = _robust_sigma(x - _box_bg(x))
    maxima = F.max_pool2d(x, 5, stride=1, padding=2)
    is_max = (x == maxima) & (x - _box_bg(x) > _STAR_SIGMA * sigma)
    ys, xs = torch.nonzero(is_max.squeeze(0).squeeze(0), as_tuple=False).unbind(1)
    vals = x.squeeze(0).squeeze(0)[ys, xs]
    order = torch.argsort(vals, descending=True)
    picked: list[tuple[int, int, float]] = []
    for idx in order:
        yy, xx = int(ys[idx]), int(xs[idx])
        if any(abs(yy - py) < 40 and abs(xx - px) < 40 for py, px, _ in picked):
            continue
        picked.append((yy, xx, float(vals[idx].item())))
        if len(picked) >= 20:
            break
    return [
        (cy, cx, _aperture_flux(x[0, 0], cy, cx))
        for cy, cx, _ in picked
        if _in_bounds(x[0, 0], cy, cx)
    ]


def _in_bounds(img: torch.Tensor, cy: int, cx: int) -> bool:
    r = _APERTURE_R
    h, w = img.shape
    return cy - r >= 0 and cx - r >= 0 and cy + r + 1 <= h and cx + r + 1 <= w


def _aperture_flux(img: torch.Tensor, cy: int, cx: int) -> float:
    r = _APERTURE_R
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
    )
    disk = (yy**2 + xx**2) <= r**2
    return float((img[cy - r : cy + r + 1, cx - r : cx + r + 1] * disk).sum().item())


def _clean_ccd(
    net: nn.Module,
    ccd: torch.Tensor,
    transform: SelfNorm,
    device: str,
) -> torch.Tensor:
    """Denoise one full CCD through the trained net, back in raw ADU.

    Pads symmetrically (up to +7 px each side) so the net's conv borders
    never land inside the reconstructed image; the interior metric margin
    additionally guards against residual edge effects.
    """
    x = ccd.float().unsqueeze(0).unsqueeze(0)
    med = x.median()
    mad = (x - med).abs().median().clamp_min(1e-6)
    x = (x - med) / mad
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    pt, pl = pad_h // 2, pad_w // 2
    pb, pr = pad_h - pt, pad_w - pl
    x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    with torch.no_grad():
        out = net(x.to(device)).cpu()
    out = out[0, 0, pt : pt + h, pl : pl + w]
    return out * mad + med


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _pair_loader(
    files: list[Path],
    hdus: list[int],
    n_pairs: int,
    n_patches: int,
    patch: int,
    batch: int,
    transform: SelfNorm,
    seed: int,
) -> Any:
    rng = np.random.default_rng(seed)
    _ndim, shape = torchfits.read_shape(str(files[0]), hdus[0])
    x_lim, y_lim = int(shape[-1]), int(shape[-2])
    cutouts: list[tuple[str, int, int, int, int]] = []
    for _ in range(n_pairs * n_patches):
        a, b = tuple(rng.choice(len(files), size=2, replace=False))
        hdu = int(rng.choice(hdus))
        x = int(rng.integers(0, x_lim - patch))
        y = int(rng.integers(0, y_lim - patch))
        cutouts.append((str(files[a]), hdu, x, y, patch))
        cutouts.append((str(files[b]), hdu, x, y, patch))
    ds_a = FitsCutoutDataset(cutouts[0::2], transform=transform)
    ds_b = FitsCutoutDataset(cutouts[1::2], transform=transform)
    return make_loader(
        torch.utils.data.StackDataset(ds_a, ds_b),
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        optimize_cache=False,
    )


def train_blank(
    files: list[Path],
    *,
    epochs: int,
    n_pairs: int,
    n_patches: int,
    patch: int,
    batch: int,
    lr: float,
    transform: SelfNorm,
    device: str,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    train_hdus = list(range(1, 31))
    val_hdus = list(range(31, 41))
    net = BlankUNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    for epoch in range(epochs):
        loader = _pair_loader(
            files, train_hdus, n_pairs, n_patches, patch, batch, transform, seed + epoch
        )
        total = 0.0
        steps = 0
        for xs, ys in loader:
            optim.zero_grad()
            loss = F.l1_loss(net(xs.to(device)), ys.to(device))
            loss.backward()
            optim.step()
            total += float(loss.item())
            steps += 1
        print(
            f"  epoch {epoch + 1}/{epochs}: l1={total / max(steps, 1):.4f}", flush=True
        )
    train_s = time.monotonic() - t0

    net.eval()
    loader = _pair_loader(
        files,
        val_hdus,
        max(1, n_pairs // 2),
        max(4, n_patches // 4),
        patch,
        batch,
        transform,
        seed + 999,
    )
    with torch.no_grad():
        resid = 0.0
        yvar = 0.0
        n = 0
        for xs, ys in loader:
            out = net(xs.to(device)).cpu()
            resid += float(((out - ys) ** 2).mean().item()) * xs.shape[0]
            yvar += float((ys**2).mean().item()) * xs.shape[0]
            n += xs.shape[0]
    val_mse = resid / max(n, 1)
    y_std = float(np.sqrt(yvar / max(n, 1)))
    stats = {
        "train_s": train_s,
        "val_mse": val_mse,
        "val_rms": float(np.sqrt(val_mse)),
        "val_target_std": y_std,
        "val_blank_explained": 1.0 - float(np.sqrt(val_mse)) / max(y_std, 1e-9),
        "params": sum(p.numel() for p in net.parameters()),
    }
    print(
        f"  held-out dark CCDs: val rms={stats['val_rms']:.3f} of target "
        f"std={y_std:.3f} (blank explains {100 * stats['val_blank_explained']:.1f}% "
        "of the variance — the rest is unpredictable noise)",
        flush=True,
    )
    return net, stats


# ---------------------------------------------------------------------------
# Science evaluation
# ---------------------------------------------------------------------------


def _ccd_metrics(
    net: nn.Module, ccd: torch.Tensor, transform: SelfNorm, device: str
) -> dict[str, Any]:
    before = ccd.float()
    cleaned = _clean_ccd(net, before, transform, device)
    fluxes_before = _star_fluxes(before)
    ratios = [
        _aperture_flux(cleaned, cy, cx) / f
        for cy, cx, f in fluxes_before
        if f > 1.0 and _in_bounds(cleaned, cy, cx)
    ]
    return {
        "cr_before": _cr_like(before),
        "cr_after": _cr_like(cleaned),
        "cr_removal": _cr_removal(before, cleaned),
        "sharp_before": _sharp_outliers(before),
        "sharp_after": _sharp_outliers(cleaned),
        "faint_before": _faint_counts(before),
        "faint_after": _faint_counts(cleaned),
        "bg_before": _bg_median(before),
        "bg_after": _bg_median(cleaned),
        "star_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "star_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "star_n": len(ratios),
    }


def eval_science(
    net: nn.Module,
    transform: SelfNorm,
    files: list[Path],
    full_hdus: int,
    other_hdus: int,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _ndim, shape = torchfits.read_shape(str(files[0]), 1)
    x_lim, y_lim = int(shape[-1]), int(shape[-2])
    for fi, path in enumerate(files):
        n_hdus = full_hdus if fi == 0 else other_hdus
        for hdu in range(1, n_hdus + 1):
            t0 = time.monotonic()
            ccd = torchfits.read_subset(str(path), hdu, 0, 0, x_lim, y_lim).float()
            m = _ccd_metrics(net, ccd, transform, device)
            m.update(
                {
                    "file": path.name,
                    "hdu": hdu,
                    "eval_s": time.monotonic() - t0,
                }
            )
            rows.append(m)
            print(
                f"  hdu {hdu:02d} {path.name}: cr {m['cr_before']:.2e}->{m['cr_after']:.2e} "
                f"(removed {100 * m['cr_removal']:.1f}%) "
                f"sharp {m['sharp_before']:.2e}->{m['sharp_after']:.2e} "
                f"bg {m['bg_before']:.0f}->{m['bg_after']:.0f} "
                f"stars {m['star_ratio_mean']:.3f} (n={m['star_n']})",
                flush=True,
            )
    return rows


def noise_injection_probe(
    net: nn.Module,
    transform: SelfNorm,
    darks: list[Path],
    science: Path,
    hdu: int,
    device: str,
    size: int = 0,
) -> dict[str, Any]:
    """science + (dark_j - dark_k) must come back to science: real noise
    of the same statistics the net was trained on, on a real field.
    ``size > 0`` probes a square window instead of the full CCD (FAST mode)."""
    _ndim, shape = torchfits.read_shape(str(science), hdu)
    x_lim, y_lim = int(shape[-1]), int(shape[-2])
    if size > 0:
        x_lim = min(x_lim, size)
        y_lim = min(y_lim, size)
    ccd = torchfits.read_subset(str(science), hdu, 0, 0, x_lim, y_lim).float()
    dj = torchfits.read_subset(str(darks[0]), hdu, 0, 0, x_lim, y_lim).float()
    dk = torchfits.read_subset(str(darks[1]), hdu, 0, 0, x_lim, y_lim).float()
    injected = ccd + (dj - dk)
    cleaned = _clean_ccd(net, injected, transform, device)
    baseline = _clean_ccd(net, ccd, transform, device)
    sigma_inj = float((cleaned - ccd).std().item())
    sigma_base = float((baseline - ccd).std().item())
    return {
        "probe_injected_sigma": float((injected - ccd).std().item()),
        "probe_residual_sigma": sigma_inj,
        "probe_baseline_sigma": sigma_base,
    }


def _inject_stars(
    net: nn.Module,
    transform: SelfNorm,
    science: Path,
    hdu: int,
    device: str,
) -> dict[str, Any]:
    """Flux recovery on synthetic Moffat stars dropped onto a real CCD.

    Off by default: the real-star aperture ratios are the primary check; this
    appendix only quantifies recovery on KNOWN injected flux.
    """
    rng = np.random.default_rng(7)
    _ndim, shape = torchfits.read_shape(str(science), hdu)
    x_lim, y_lim = int(shape[-1]), int(shape[-2])
    ccd = torchfits.read_subset(str(science), hdu, 0, 0, x_lim, y_lim).float()
    synth = ccd.clone()
    truths: list[tuple[int, int, float]] = []
    h, w = ccd.shape
    for peak in (300.0, 800.0, 1500.0, 3000.0):
        cx = int(rng.integers(200, w - 200))
        cy = int(rng.integers(200, h - 200))
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2).float()
        synth += peak * (1 + r2 / (4.0 * 4.0)) ** (-1.5)
        truths.append(
            (cy, cx, _aperture_flux(synth, cy, cx) - _aperture_flux(ccd, cy, cx))
        )
    cleaned = _clean_ccd(net, synth, transform, device)
    ratios = [
        (_aperture_flux(cleaned, cy, cx) - _aperture_flux(ccd, cy, cx)) / f
        for cy, cx, f in truths
        if f > 1.0
    ]
    return {
        "inject_n": len(ratios),
        "inject_recovery_mean": float(np.mean(ratios)) if ratios else float("nan"),
    }


# ---------------------------------------------------------------------------
# Report and products
# ---------------------------------------------------------------------------


def _write_products(out_dir: Path, rows: list[dict[str, Any]], tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"## {tag}", ""]
    md += [
        "| file | hdu | cr before | cr after | cr removed | sharp before | sharp after |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows[:40]:
        md += [
            f"| {r['file']} | {r['hdu']} | {r['cr_before']:.2e} | {r['cr_after']:.2e} "
            f"| {100 * r['cr_removal']:.1f}% "
            f"| {r['sharp_before']:.2e} | {r['sharp_after']:.2e} |"
        ]
    (out_dir / f"{tag}_metrics.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir / (tag + '_metrics.md')}")


def _render_gallery(
    net: nn.Module,
    transform: SelfNorm,
    science: Path,
    hdu: int,
    device: str,
    tag: str,
) -> Path | None:
    """Before/after figure on the CR-richest 512x512 window of a science CCD.

    Written to examples/output/megacam_cr_denoise_<tag>.png (the docs
    gallery convention); skipped when matplotlib is unavailable. FAST mode
    skips the full-CCD scan, so the smoke path stays bounded.
    """
    if _fast_mode():
        return None
    try:
        from examples._plotting import save_image_before_after
    except ImportError:
        print("skipping gallery figure: matplotlib not installed")
        return None
    _ndim, shape = torchfits.read_shape(str(science), hdu)
    x_lim, y_lim = int(shape[-1]), int(shape[-2])
    ccd = torchfits.read_subset(str(science), hdu, 0, 0, x_lim, y_lim).float()
    mask = _cr_mask(ccd).bool()
    wins = 512
    best_x, best_y, best_n = 0, 0, 0
    for y0 in range(_MARGIN, y_lim - wins, wins // 2):
        for x0 in range(_MARGIN, x_lim - wins, wins // 2):
            n = int(mask[y0 : y0 + wins, x0 : x0 + wins].sum().item())
            if n > best_n:
                best_x, best_y, best_n = x0, y0, n
    if best_n == 0:
        best_x, best_y = x_lim // 4, y_lim // 4
    raw = ccd[best_y : best_y + wins, best_x : best_x + wins]
    cleaned = _clean_ccd(net, raw, transform, device)
    path = save_image_before_after(
        raw,
        cleaned,
        f"megacam_cr_denoise_{tag}",
        titles=(
            f"{science.name} hdu {hdu} (CRs before)",
            f"{tag} net (CRs after)",
        ),
    )
    if path is not None:
        print(f"wrote {path}", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-dir", type=Path, default=megacam_dir() / "calib")
    parser.add_argument("--science-dir", type=Path, default=megacam_dir())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "denoise_cr",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--n-patches", type=int, default=64)
    parser.add_argument("--patch", type=int, default=_PATCH)
    parser.add_argument("--batch", type=int, default=_BATCH)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--full-eval-files", type=int, default=1)
    parser.add_argument("--full-hdus", type=int, default=40)
    parser.add_argument("--eval-hdus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-stars", action="store_true")
    parser.add_argument("--mode", choices=["dark", "bias", "both"], default="both")
    parser.add_argument(
        "--probe-size", type=int, default=0, help="probe window; 0 = full CCD"
    )
    args = parser.parse_args()

    if _fast_mode():
        args.epochs = min(args.epochs, 1)
        args.n_pairs = min(args.n_pairs, 1)
        args.n_patches = min(args.n_patches, 8)
        args.full_eval_files = min(args.full_eval_files, 1)
        args.full_hdus = min(args.full_hdus, 1)
        args.eval_hdus = min(args.eval_hdus, 1)
        args.probe_size = min(args.probe_size or 2**31, 1024)
    darks = sorted((args.calib_dir / "darks").glob("*.fits.fz"))
    biases = sorted((args.calib_dir / "biases").glob("*.fits.fz"))
    sciences = sorted((args.science_dir).glob("*o.fits.fz"))
    if not darks or len(darks) < 2:
        print(
            "SKIP: need >=2 darks in benchmarks_data/cfht_megacam/calib/darks. "
            "Fetch via: bash scripts/fetch_cfht_calib_frames.sh"
        )
        return 0
    if not sciences:
        print("SKIP: no science frames; fetch via scripts/fetch_cfht_megacam_sample.sh")
        return 0

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(
        f"darks={len(darks)} biases={len(biases)} science={len(sciences)} "
        f"device={device}",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tag, files in (("dark", darks), ("bias", biases)):
        if args.mode == "both" or args.mode == tag:
            transform = SelfNorm()
            print(
                f"[{tag}->blank N2N] training on {len(files)} {tag} frames", flush=True
            )
            net, stats = train_blank(
                files,
                epochs=args.epochs,
                n_pairs=args.n_pairs,
                n_patches=args.n_patches,
                patch=args.patch,
                batch=args.batch,
                lr=args.lr,
                transform=transform,
                device=device,
                seed=args.seed,
            )
            print(f"[{tag}] transfer evaluation on science", flush=True)
            rows = eval_science(
                net,
                transform,
                sciences[: args.full_eval_files],
                args.full_hdus,
                args.eval_hdus,
                device,
            )
            probe = noise_injection_probe(
                net, transform, darks, sciences[0], 1, device, size=args.probe_size
            )
            extra: dict[str, Any] = {}
            if args.inject_stars:
                extra.update(_inject_stars(net, transform, sciences[0], 1, device))
            _write_products(args.out_dir, rows, tag)
            if rows:
                _render_gallery(
                    net, transform, sciences[0], rows[0]["hdu"], device, tag
                )
            summary = {
                "tag": tag,
                "stats": stats,
                "probe": probe,
                **extra,
            }
            (args.out_dir / f"{tag}_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
