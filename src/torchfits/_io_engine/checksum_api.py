"""FITS checksum helpers for the torchfits I/O engine."""

from __future__ import annotations

from typing import Any, Dict

from ..core import ChecksumVerifier


def write_checksums(path: str, hdu: int = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    ChecksumVerifier.write_hdu_checksums(path, hdu)


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    return dict(ChecksumVerifier.verify_hdu_checksums(path, hdu))
