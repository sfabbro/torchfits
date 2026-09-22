"""FITS checksum helpers for the torchfits I/O engine."""

from __future__ import annotations

import operator
from typing import Any, Dict


def _cpp() -> Any:
    """Resolve the native extension lazily.

    Importing it maps libtorch and imports ``torch``, so checksums -- pure byte
    arithmetic that never touches a tensor -- must not do it at module scope.
    """
    import torchfits._C as cpp

    return cpp


def _validate_hdu(hdu: int) -> int:
    if isinstance(hdu, bool):
        raise TypeError("hdu must be a non-negative integer")
    try:
        hdu = operator.index(hdu)
    except TypeError:
        raise TypeError("hdu must be a non-negative integer") from None
    if hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    return int(hdu)


def write_checksums(path: str, hdu: int = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    from .paths import coerce_fits_path, guard_fits_path

    path = coerce_fits_path(path)
    guard_fits_path(path)
    _cpp().write_hdu_checksums(str(path), _validate_hdu(hdu))


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    CFITSIO ``ffvcks`` status codes (``datastatus`` / ``hdustatus``):
    - ``0`` — checksum keywords absent (nothing to verify)
    - ``1`` — checksum present and correct
    - ``-1`` — checksum present but incorrect (corrupt)

    Returns a dict with ``datastatus``, ``hdustatus``, ``ok``, ``present``,
    and ``status`` (``"ok"``, ``"no_checksums"``, or ``"fail"``).
    ``status`` is ``"fail"`` only when a present checksum is incorrect; a
    correct DATASUM without a CHECKSUM keyword (or vice versa) is ``"ok"``.
    ``present`` is False when CFITSIO reports no checksum keywords at all.
    """
    from .paths import coerce_fits_path, guard_fits_path

    path = coerce_fits_path(path)
    guard_fits_path(path)
    datastatus, hdustatus = _cpp().verify_hdu_checksums(str(path), _validate_hdu(hdu))
    data_i = int(datastatus)
    hdu_i = int(hdustatus)

    if data_i < 0 or hdu_i < 0:
        status_str = "fail"
        ok = False
        present = True
    elif data_i == 0 and hdu_i == 0:
        status_str = "no_checksums"
        ok = True
        present = False
    else:
        status_str = "ok"
        ok = True
        present = True

    return {
        "datastatus": data_i,
        "hdustatus": hdu_i,
        "ok": ok,
        "present": present,
        "status": status_str,
    }
