"""Heuristics for choosing read fast paths (mmap vs buffered).

These are lightweight, best-effort decisions that set enable_mmap/enable_buffered
for the C++ backend. The backend will validate and safely fall back if a choice
is not applicable (e.g., mmap below size threshold, or not a full-image read).
"""
from __future__ import annotations

import os
from typing import Optional

from .fits_reader import get_header, get_dims


def detect_compressed_from_header(header: dict) -> bool:
    """Best-effort compressed image detection via header.

    CFITSIO marks compressed image extensions with ZIMAGE/T keys, ZCMPTYPE, etc.
    """
    try:
        zimage = str(header.get("ZIMAGE", "F")).strip().upper() in ("T", "TRUE", "1")
        if zimage:
            return True
        if "ZCMPTYPE" in header:
            return True
    except Exception:
        pass
    return False


def choose_flags(
    *,
    is_full_image: bool,
    is_compressed: bool,
    file_size_mb: Optional[float],
    image_dim: Optional[int] = None,
) -> tuple[Optional[bool], Optional[bool], str]:
    """Choose (enable_mmap, enable_buffered, reason) for a read.

    - Only considers full-image reads; for cutouts returns (None, None).
    - If compressed: prefer buffered.
    - If uncompressed: prefer mmap only when size above threshold.
    """
    if not is_full_image:
        return None, None, "cutout-or-partial"

    if is_compressed:
        return None, True, "compressed->buffered"

    # uncompressed full image: consider mmap for larger files
    try:
        min_mb_env = os.environ.get("TORCHFITS_MMAP_MIN_MB")
        min_mb = float(min_mb_env) if min_mb_env else 50.0
    except Exception:
        min_mb = 50.0

    if file_size_mb is not None and file_size_mb >= min_mb:
        return True, None, f"mmap>= {min_mb:g}MB"

    # small images: default path
    return None, None, "standard"


def choose_read_mode_for_image(path: str, hdu: int | str = 0) -> dict:
    """Decide flags for a full-image read on a local path.

    Returns dict: { enable_mmap: Optional[bool], enable_buffered: Optional[bool], reason: str }
    """
    header = get_header(path, hdu=hdu)
    is_comp = detect_compressed_from_header(header)
    file_size_mb: Optional[float] = None
    try:
        st = os.stat(path)
        file_size_mb = st.st_size / (1024.0 * 1024.0)
    except Exception:
        pass
    # We could use dims if desired; not strictly needed for current rules
    try:
        # get_dims expects int hdu; if name is provided, dims not critical for decision
        dims = get_dims(path, hdu if isinstance(hdu, int) else 0)
        dim0 = int(dims[-1]) if isinstance(dims, (list, tuple)) and len(dims) else None
    except Exception:
        dim0 = None
    em, eb, reason = choose_flags(
        is_full_image=True,
        is_compressed=is_comp,
        file_size_mb=file_size_mb,
        image_dim=dim0,
    )
    return {"enable_mmap": em, "enable_buffered": eb, "reason": reason}
