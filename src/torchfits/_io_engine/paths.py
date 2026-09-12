"""Path helpers for CFITSIO extended filenames."""

from __future__ import annotations

import os
from typing import Any

from torchfits.http_util import guard_cfitsio_remote_path, is_cfitsio_network_url

__all__ = [
    "cfitsio_base_path",
    "coerce_fits_path",
    "guard_fits_path",
    "has_bz2_support",
    "has_cfitsio_filter",
    "is_cfitsio_network_url",
]


def has_bz2_support() -> bool:
    """True when this build can read bzip2-wrapped (``.bz2``) FITS files.

    CFITSIO decompresses whole-file ``.bz2`` transparently on open, but only
    when compiled with ``HAVE_BZIP2`` (libbz2 linked). The vendored build
    enables this whenever libbz2 is found; the flag mirrors the compiled-in
    capability so input guards can be honest instead of blanket-rejecting.
    """
    try:
        import torchfits._C as _C

        return bool(getattr(_C, "HAS_BZIP2", False))
    except Exception:
        return False


def require_bz2_support(path: str) -> None:
    """Raise the actionable error for ``.bz2`` inputs on incapable builds."""
    if path.lower().endswith(".bz2") and not has_bz2_support():
        raise ValueError(
            "This torchfits build was compiled without bzip2 support, so "
            "'.bz2' FITS files cannot be opened. Decompress the file first, "
            "or install a bzip2-enabled wheel/build."
        )


def coerce_fits_path(path: Any) -> Any:
    """Normalize an ``os.PathLike`` path to ``str`` at the read boundary.

    Writes have always called ``os.fspath`` (see ``write_api``), but the read
    side rejected ``pathlib.Path`` outright -- ``torchfits.read(Path(...))``
    raised ``ValueError`` while ``torchfits.write(Path(...))`` worked. That is a
    sharp edge for the common ``for p in root.glob("*.fits")`` loop.

    Strings (including CFITSIO network URLs and ``file.fits[1]`` filters) pass
    through untouched, and non-path values are returned as-is so the existing
    argument validation still produces its own error.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    if isinstance(path, (list, tuple)):
        return type(path)(coerce_fits_path(item) for item in path)
    return path


def guard_fits_path(path: str) -> str:
    """SSRF-check CFITSIO network URLs; return *path* unchanged for CFITSIO."""
    guard_cfitsio_remote_path(path)
    return path


def cfitsio_base_path(path: str) -> str:
    """Return the on-disk path, stripping a CFITSIO ``[...]`` filter if present.

    CFITSIO extended filenames look like ``image.fits[10:100,20:200]`` or
    ``file.fits[1]``. Existence checks must use the base file, not the filter.
    Brackets in a directory component (``/tmp/[data]/file.fits``) are not
    filters: only ``[`` after the last ``/`` that closes with a trailing ``]``.
    """
    if not path.endswith("]"):
        return path
    last_slash = path.rfind("/")
    search_start = last_slash + 1 if last_slash >= 0 else 0
    bracket = path.find("[", search_start)
    if bracket < 0:
        return path
    return path[:bracket]


def has_cfitsio_filter(path: str) -> bool:
    """True when the final path component has a CFITSIO ``[...]`` section."""
    if not path.endswith("]"):
        return False
    last_slash = path.rfind("/")
    search_start = last_slash + 1 if last_slash >= 0 else 0
    return "[" in path[search_start:]
