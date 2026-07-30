"""Path helpers for CFITSIO extended filenames."""

from __future__ import annotations

from torchfits.http_util import guard_cfitsio_remote_path, is_cfitsio_network_url

__all__ = [
    "cfitsio_base_path",
    "guard_fits_path",
    "has_cfitsio_filter",
    "is_cfitsio_network_url",
]


def guard_fits_path(path: str) -> str:
    """SSRF-check CFITSIO network URLs; return *path* unchanged for CFITSIO."""
    guard_cfitsio_remote_path(path)
    return path


def cfitsio_base_path(path: str) -> str:
    """Return the on-disk path, stripping a CFITSIO ``[...]`` filter if present.

    CFITSIO extended filenames look like ``image.fits[10:100,20:200]`` or
    ``file.fits[1]``. Existence checks must use the base file, not the filter.
    """
    bracket = path.find("[")
    if bracket < 0:
        return path
    return path[:bracket]


def has_cfitsio_filter(path: str) -> bool:
    """True when ``path`` includes any CFITSIO ``[...]`` extended-filename bracket.

    This is a bracket presence test, not an image-section detector: HDU selectors
    like ``file.fits[1]`` / ``[EVENTS]`` also match. Prefer ``hdu=`` / EXTNAME for
    those; only image pixel sections are a smoke-tested torchfits surface today.
    """
    return "[" in path
