"""Lupton asinh RGB (1.0 surface); richer multi-band RGB → 1.1.

Matches Astropy's ``make_lupton_rgb`` / ``RGBImageMappingLupton`` path:
stretch intensity with ``LuptonAsinhStretch``, colour = band * f(I)/I, then
per-pixel peak clip when max(R,G,B) > 1. Never divide by the field-wide max
(that crushed midtones to near-black whenever one star saturated).
"""

from __future__ import annotations

import binascii
import math
import struct
import zlib
from typing import Any, Final

import torch

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
    raw = flat.numpy().tobytes()
    row_bytes = width * 3
    scanlines = bytearray((row_bytes + 1) * height)
    for row in range(height):
        offset = row * (row_bytes + 1)
        scanlines[offset] = 0
        start = row * row_bytes
        scanlines[offset + 1 : offset + 1 + row_bytes] = raw[start : start + row_bytes]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)
