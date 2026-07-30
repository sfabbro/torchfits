"""Private per-call C++ FITS handles/readers (CFITSIO §4 Option A).

Sharing a single ``fitsfile*`` across threads corrupts CFITSIO's internal
position state, so every read opens a fresh, privately-owned handle. There is
deliberately no cross-thread handle/reader cache here.
"""

from __future__ import annotations

from typing import Any


def _acquire_cpp_handle(path: str, cpp: Any) -> Any:
    """Open a fresh, privately-owned CFITSIO handle. The caller must close it."""
    return cpp.open_fits_file(path, "r")


def _acquire_cpp_reader(path: str, hdu: int, cpp: Any) -> Any:
    """Open a fresh ``TableReader`` with its own private handle (never shared).

    The filename constructor opens and owns a per-instance ``fitsfile*`` that is
    closed when the reader is garbage-collected, so distinct HDU readers on
    distinct threads never touch the same underlying handle.
    """
    return cpp.TableReader(path, int(hdu))
