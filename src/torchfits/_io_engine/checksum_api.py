"""FITS checksum helpers for the torchfits I/O engine."""

from __future__ import annotations

from typing import Any, Dict

import torchfits._C as cpp


def _validate_hdu(hdu: int) -> int:
    if isinstance(hdu, bool) or not isinstance(hdu, int):
        raise TypeError("hdu must be a non-negative integer")
    if hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    return int(hdu)


def write_checksums(path: str, hdu: int = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    from .paths import guard_fits_path

    guard_fits_path(path)
    cpp.write_hdu_checksums(str(path), _validate_hdu(hdu))


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    CFITSIO ``ffvcks`` status codes:
    - ``0`` — checksum keywords absent (nothing to verify)
    - ``1`` — checksum present and correct
    - ``-1`` — checksum present but incorrect (corrupt)

    Returns a dict with ``datastatus``, ``hdustatus``, ``ok``, ``present``,
    and ``status`` (``"ok"``, ``"no_checksums"``, or ``"fail"``).
    ``present`` is False when CFITSIO reports no checksum keywords.
    """
    from .paths import guard_fits_path

    guard_fits_path(path)
    datastatus, hdustatus = cpp.verify_hdu_checksums(str(path), _validate_hdu(hdu))
    data_i = int(datastatus)
    hdu_i = int(hdustatus)

    if data_i == 0 and hdu_i == 0:
        status_str = "no_checksums"
        ok = True
        present = False
    elif data_i == 1 and hdu_i == 1:
        status_str = "ok"
        ok = True
        present = True
    else:
        status_str = "fail"
        ok = False
        present = True

    return {
        "datastatus": data_i,
        "hdustatus": hdu_i,
        "ok": ok,
        "present": present,
        "status": status_str,
    }
