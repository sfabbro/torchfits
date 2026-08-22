"""Lupton asinh RGB plus auto-adaptive multi-band ``rgb()``.

``lupton_rgb`` matches Astropy's ``make_lupton_rgb`` / ``RGBImageMappingLupton``:
stretch intensity with ``LuptonAsinhStretch``, colour = band * f(I)/I, then
per-pixel peak clip when max(R,G,B) > 1. Never divide by the field-wide max
(that crushed midtones to near-black whenever one star saturated).

``rgb`` is the pretty default: fitspng-style MAD auto-scale, scarlet N-band
mix, optional per-band MAD equalize, coupled asinh, saturation, sRGB.
"""

from __future__ import annotations

import binascii
import math
import struct
import zlib
from typing import Any, Final, Sequence

import torch

from .helpers import _quantile, estimate_background

# Astropy softens near-zero Q to this floor so asinh(frac*Q) stays finite.
# The threshold is float32 machine epsilon (2**-23): Q values below it are
# treated as zero and replaced by _LUPTON_Q_FLOOR.
_LUPTON_Q_EPS: Final = 1.0 / 2**23
_LUPTON_Q_FLOOR: Final = 0.1


def lupton_rgb(
    r: Any,
    g: Any,
    b: Any,
    *,
    Q: float = 8.0,
    stretch: float = 0.5,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Return float RGB tensor with shape ``(H, W, 3)`` in ``[0, 1]``.

    Parameters follow Astropy's Lupton asinh convention. ``stretch`` is the
    linear intensity scale (smaller → brighter preview); Astropy's default is
    ``5``, while ``0.5`` suits typical survey cutout previews.
    """
    if Q < 0:
        raise ValueError(f"Q must be non-negative, got {Q}")
    if stretch <= 0:
        raise ValueError(f"stretch must be > 0, got {stretch}")

    # Match Astropy's Q floor for near-zero softening.
    q = float(Q)
    if abs(q) < _LUPTON_Q_EPS:
        q = _LUPTON_Q_FLOOR

    r_t = torch.as_tensor(r)
    work_dtype = torch.float32 if r_t.device.type == "mps" else torch.float64
    red = torch.as_tensor(r, dtype=work_dtype) - float(minimum)
    green = torch.as_tensor(g, dtype=work_dtype) - float(minimum)
    blue = torch.as_tensor(b, dtype=work_dtype) - float(minimum)
    intensity = (red + green + blue) / 3.0

    # LuptonAsinhStretch: asinh(Q*I/stretch) * (frac / asinh(frac*Q)), frac=0.1
    soften = q / float(stretch)
    frac = 0.1
    slope = frac / math.asinh(frac * q)
    f_intensity = torch.asinh(intensity * soften) * slope
    fac = torch.where(
        intensity > 0,
        f_intensity / intensity,
        torch.zeros_like(intensity),
    )
    channels = torch.stack((red * fac, green * fac, blue * fac), dim=-1)
    channels = torch.clamp(channels, min=0.0)

    # Per-pixel peak clip (not field-wide). Preserves colour on bright stars.
    if channels.numel() > 0:
        peak = channels.amax(dim=-1, keepdim=True)
        channels = torch.where(peak > 1.0, channels / peak, channels)
    return torch.clamp(channels, 0.0, 1.0)


# Scarlet ``channels_to_rgb`` matrices, bands short → long; row 0 = R.
# https://github.com/pmelchior/scarlet/blob/master/scarlet/display.py
_SCARLET_MAPS: Final[dict[int, tuple[tuple[float, ...], ...]]] = {
    1: ((1.0,), (1.0,), (1.0,)),
    2: (
        (0.0, 1.0),
        (0.333 / 0.667, 0.333 / 0.667),
        (1.0, 0.0),
    ),
    3: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    4: (
        (0.0, 0.0, 0.333 / 1.333, 1.0 / 1.333),
        (0.0, 0.667 / 1.333, 0.667 / 1.333, 0.0),
        (1.0 / 1.333, 0.333 / 1.333, 0.0, 0.0),
    ),
    5: (
        (0.0, 0.0, 0.0, 0.667 / 1.667, 1.0 / 1.667),
        (0.0, 0.333 / 1.667, 1.0 / 1.667, 0.333 / 1.667, 0.0),
        (1.0 / 1.667, 0.667 / 1.667, 0.0, 0.0, 0.0),
    ),
    6: (
        (0.0, 0.0, 0.0, 0.333 / 2.0, 0.667 / 2.0, 1.0 / 2.0),
        (0.0, 0.333 / 2.0, 0.667 / 2.0, 0.667 / 2.0, 0.333 / 2.0, 0.0),
        (1.0 / 2.0, 0.667 / 2.0, 0.333 / 2.0, 0.0, 0.0, 0.0),
    ),
    # 7-band extension of scarlet's scheme, rows normalized to sum 1 like
    # every other width (the reddest band contributes warm + a little grey).
    7: (
        (0.0, 0.0, 0.0, 1.0 / 8.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0),
        (0.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 8.0, 0.0, 1.0 / 4.0),
        (3.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 1.0 / 4.0),
    ),
}

_RGB_Q: Final = 8.0
_RGB_BLACK_MAD: Final = 3.0
_RGB_FILLED_FRAC: Final = 0.22
# A bright extended object (planet disk, nearby galaxy core) towers above
# its noise: after MAD equalize its p90 sits orders of magnitude past a
# few sigma, while star-rich/deep fields top out near single digits.
# Classify those as filled so the faint-feature stretch cannot blow the
# object out (measured: Jupiter disk p90 ~= 1200 sigma vs <= 6 elsewhere).
_RGB_OBJECT_SPAN_SIGMAS: Final = 50.0
_RGB_FILLED_TARGET: Final = 0.72
_RGB_NMGY_ZP: Final = 22.5
_MAD_SCALE: Final = 1.4826  # estimate_background returns MAD × this


def _work_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _as_band_stack(*bands: Any) -> torch.Tensor:
    """Return ``(C, H, W)`` float stack, C in 1..7. Shortest-λ first."""
    if not bands:
        raise ValueError("rgb() needs at least one band")
    if len(bands) == 1:
        tensor = torch.as_tensor(bands[0])
        if tensor.ndim == 2:
            stack = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            stack = tensor
        else:
            raise ValueError(
                f"single-argument rgb() expects (H, W) or (C, H, W), got {tuple(tensor.shape)}"
            )
    else:
        tensors = [torch.as_tensor(band) for band in bands]
        if any(item.ndim != 2 for item in tensors):
            raise ValueError("each rgb() band must be a 2-D image")
        shape = tuple(tensors[0].shape)
        if any(tuple(item.shape) != shape for item in tensors):
            raise ValueError("rgb() bands must share the same spatial shape")
        stack = torch.stack(tensors, dim=0)
    channels = int(stack.shape[0])
    if not 1 <= channels <= 7:
        raise ValueError(f"rgb() supports 1–7 bands, got C={channels}")
    if stack.ndim != 3:
        raise ValueError(f"rgb() stack must be (C, H, W), got {tuple(stack.shape)}")
    dtype = _work_dtype(stack.device)
    return stack.to(dtype=dtype)


def _subsample(image: torch.Tensor) -> torch.Tensor:
    if image.numel() == 0:
        return image
    height, width = int(image.shape[-2]), int(image.shape[-1])
    stride = 10 if min(height, width) >= 32 else 1
    return image[..., ::stride, ::stride]


def _scalar_stats(image: torch.Tensor) -> tuple[float, float, float, float]:
    """Median, raw MAD, p0.5, p99.5 of a 2-D image (NaNs ignored)."""
    sample = _subsample(image)
    if sample.numel() == 0:
        return 0.0, 0.0, 0.0, 1.0
    med_t, std_t = estimate_background(sample, dim=(-2, -1))
    lo_t = _quantile(sample, 0.005, dim=(-2, -1))
    hi_t = _quantile(sample, 0.995, dim=(-2, -1))
    med = float(med_t.reshape(()).item())
    std = float(std_t.reshape(()).item())
    mad = std / _MAD_SCALE if math.isfinite(std) else 0.0
    lo = float(lo_t.reshape(()).item())
    hi = float(hi_t.reshape(()).item())
    if not math.isfinite(med):
        med = 0.0
    if not math.isfinite(mad) or mad < 0.0:
        mad = 0.0
    if not math.isfinite(lo):
        lo = med
    if not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return med, mad, lo, hi


def _equalize_bands(stack: torch.Tensor) -> torch.Tensor:
    """Sky-subtract and divide by MAD so ADU scale / pedestal do not paint colour."""
    out = stack.clone()
    for index in range(int(stack.shape[0])):
        med, mad, _lo, _hi = _scalar_stats(stack[index])
        band = stack[index] - med
        if mad > 0.0:
            band = band / mad
        out[index] = band
    return out


def _mix_to_rgb(stack: torch.Tensor, weights: Any | None) -> torch.Tensor:
    channels, height, width = (int(n) for n in stack.shape)
    if weights is None:
        matrix = torch.tensor(
            _SCARLET_MAPS[channels],
            dtype=stack.dtype,
            device=stack.device,
        )
    else:
        matrix = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device)
        if tuple(matrix.shape) != (3, channels):
            raise ValueError(
                f"weights must have shape (3, {channels}), got {tuple(matrix.shape)}"
            )
    mixed = matrix @ stack.reshape(channels, -1)
    return mixed.T.reshape(height, width, 3)


def _stretch_for_target(i_ref: float, target: float, q: float) -> float:
    """Lupton stretch so ``f(i_ref) ≈ target`` (``f`` = asinh mapping)."""
    frac = 0.1
    slope = frac / math.asinh(frac * q)
    goal = min(max(target, 1e-4), 0.99)
    denom = math.sinh(goal / slope)
    if denom <= 0.0 or i_ref <= 0.0 or not math.isfinite(i_ref):
        return 1.0
    stretch = q * i_ref / denom
    if not math.isfinite(stretch) or stretch <= 0.0:
        return 1.0
    return stretch


def _srgb_oetf(linear: torch.Tensor) -> torch.Tensor:
    linear = torch.clamp(linear, 0.0, 1.0)
    low = linear * 12.92
    high = 1.055 * torch.pow(linear.clamp(min=0.0), 1.0 / 2.4) - 0.055
    return torch.where(linear <= 0.0031308, low, high)


def _apply_saturation(rgb: torch.Tensor, saturation: float) -> torch.Tensor:
    if saturation == 1.0:
        return rgb
    luma = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).unsqueeze(-1)
    return luma + float(saturation) * (rgb - luma)


def rgb(
    *bands: Any,
    brightness: float = 0.15,
    saturation: float = 2.0,
    scene: str = "auto",
    weights: Any | None = None,
    calibrated: bool = False,
    zeropoints: Sequence[float] | None = None,
) -> torch.Tensor:
    """Auto RGB from 1–7 aligned bands (shortest wavelength first).

    Parameters
    ----------
    bands :
        One ``(H, W)`` image, one ``(C, H, W)`` cube with ``C`` in 1..7, or
        1–7 separate ``(H, W)`` tensors. Order is **blue → red**
        (``rgb(g, r, i)``), unlike :func:`lupton_rgb` which is reddest first.
    brightness :
        Display value of sky + 1σ on empty fields (default ``0.15``).
    saturation :
        STIFF-style chroma boost around luma. ``1`` is photometric.
    scene :
        ``"empty"``, ``"filled"``, or ``"auto"`` (median in the 0.5–99.5
        percentile span; filled when that fraction exceeds 0.22).
    weights :
        Optional ``(3, C)`` mix (rows R, G, B). Default is scarlet's map.
    calibrated :
        If true, bands are already on one flux scale; skip per-band
        sky-median subtract and MAD equalize. Implied when ``zeropoints``
        is set.
    zeropoints :
        AB magnitude of 1 count per band. Converted to nanomaggies with
        ``counts * 10**(-0.4*(zp - 22.5))``.

    Returns
    -------
    Tensor
        Display-referred ``(H, W, 3)`` in ``[0, 1]`` after sRGB OETF.
    """
    if brightness <= 0.0 or brightness >= 1.0:
        raise ValueError(f"brightness must be in (0, 1), got {brightness}")
    if saturation < 0.0:
        raise ValueError(f"saturation must be >= 0, got {saturation}")
    if scene not in ("auto", "empty", "filled"):
        raise ValueError(f"scene must be 'auto', 'empty', or 'filled', got {scene!r}")

    stack = _as_band_stack(*bands)
    # Keep NaN through stats / equalize so mosaic holes are not a fake sky
    # floor (zero-fill made HiPS footprints look filled and washed-out).
    stack = torch.where(
        torch.isfinite(stack), stack, torch.full_like(stack, float("nan"))
    )
    n_band = int(stack.shape[0])
    use_calibrated = bool(calibrated) or zeropoints is not None
    if zeropoints is not None:
        zps = [float(z) for z in zeropoints]
        if len(zps) != n_band:
            raise ValueError(
                f"zeropoints length must equal number of bands ({n_band}), got {len(zps)}"
            )
        scales = torch.tensor(
            [10.0 ** (-0.4 * (zp - _RGB_NMGY_ZP)) for zp in zps],
            dtype=stack.dtype,
            device=stack.device,
        )
        stack = stack * scales.reshape(n_band, 1, 1)

    if not use_calibrated:
        stack = _equalize_bands(stack)

    mixed = _mix_to_rgb(stack, weights)
    if mixed.numel() == 0:
        return mixed.clamp(0.0, 1.0)

    luma = mixed.mean(dim=-1)
    med, mad, p_lo, p_hi = _scalar_stats(luma)
    span = p_hi - p_lo
    frac = (med - p_lo) / span if span > 0.0 else 1.0

    finite_luma = luma[torch.isfinite(luma)]
    p90_sigma = (
        float(_quantile(finite_luma, 0.90, dim=(-1,)).reshape(()))
        if finite_luma.numel() > 0
        else 0.0
    )
    object_span_sigmas = (p90_sigma - med) / mad if mad > 0.0 else 0.0
    if scene == "filled":
        filled = True
    elif scene == "empty":
        filled = False
    else:
        filled = frac > _RGB_FILLED_FRAC or object_span_sigmas > _RGB_OBJECT_SPAN_SIGMAS

    if filled:
        # Anchor the stretch near the bright structure (p90), not the top
        # quantile of the whole range: for extreme dynamic ranges (Jupiter
        # disks, saturated cores) p99.5 sits deep in the asinh-saturated
        # zone and every extended pixel clips.
        black = p_lo
        anchor = max(p90_sigma - max(black, 0.0), mad, 1e-12)
        stretch = _stretch_for_target(anchor, _RGB_FILLED_TARGET, _RGB_Q)
    else:
        black = med - _RGB_BLACK_MAD * mad
        i_ref = (_RGB_BLACK_MAD * mad) + (_MAD_SCALE * mad)
        stretch = _stretch_for_target(max(i_ref, 1e-12), float(brightness), _RGB_Q)

    shifted = torch.clamp(mixed - black, min=0.0)
    shifted = torch.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
    mapped = lupton_rgb(
        shifted[..., 0],
        shifted[..., 1],
        shifted[..., 2],
        Q=_RGB_Q,
        stretch=stretch,
        minimum=0.0,
    )
    mapped = _apply_saturation(mapped, float(saturation))
    mapped = torch.clamp(mapped, 0.0, 1.0)
    return _srgb_oetf(mapped)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_rgb_image(path: str, rgb: torch.Tensor) -> None:
    """Write RGB float image as PNG (stdlib only; no Pillow / NumPy import)."""
    if rgb.dim() != 3 or int(rgb.shape[-1]) != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    height, width, _ = map(int, rgb.shape)
    flat = (
        torch.clamp(rgb, 0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .contiguous()
        .reshape(-1)
    )
    row_bytes = width * 3
    rows = flat.reshape(height, row_bytes)
    scanlines = torch.cat(
        (torch.zeros(height, 1, dtype=torch.uint8), rows), dim=1
    ).contiguous()
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines.numpy().tobytes(), level=6))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)
