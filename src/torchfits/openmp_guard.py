"""OpenMP runtime duplication guard for torchfits.

Detects multiple libomp / OpenMP runtimes loaded and provides
clear remediation instructions rather than relying on the unsafe
KMP_DUPLICATE_LIB_OK workaround.

This does NOT attempt to fix the condition automatically (which
must be solved by aligning the environment so only one OpenMP
runtime is present). It instead:
  * Exposes `detect_duplicate_openmp()` returning diagnostics
  * Exposes `raise_if_duplicate_openmp()` raising a RuntimeError with guidance
  * Tests can mark certain cases xfail when duplication is present.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any


def _list_loaded_libraries() -> list[str]:
    """Return list of currently loaded dynamic libraries (best-effort)."""
    libs: list[str] = []
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["/usr/bin/vmmap", str(os.getpid())], text=True
            )
            for line in out.splitlines():
                if ".dylib" in line:
                    libs.append(line.strip())
        except Exception:
            pass
    return libs


def detect_duplicate_openmp() -> dict[str, Any]:
    """Detect duplicate OpenMP runtimes.

    Returns dict with keys:
      duplicate (bool)
      candidates (list[str])
      message (str)
    """
    candidates = []
    for name in list(sys.modules):
        if "omp" in name.lower():
            candidates.append(name)
    loaded = _list_loaded_libraries()
    pattern = re.compile(r"libi?omp\\.dylib", re.IGNORECASE)
    lib_hits = [lib for lib in loaded if pattern.search(lib)]
    candidates.extend(lib_hits)
    uniq = sorted(set(candidates))
    duplicate = len(lib_hits) > 1
    if duplicate:
        msg = "Multiple OpenMP runtimes detected"
    else:
        msg = (
            "OpenMP runtime presence uncertain"
            if uniq
            else "No OpenMP indicators found"
        )
    return {"duplicate": duplicate, "candidates": uniq, "message": msg}


def raise_if_duplicate_openmp():
    """Raise a helpful error if multiple OpenMP runtimes are detected."""
    info = detect_duplicate_openmp()
    if info["duplicate"]:
        details = "\n".join(info["candidates"])
        raise RuntimeError(
            "Duplicate OpenMP runtimes detected.\n" + details + "\nResolution steps:\n"
            "  1. Ensure only one provider (e.g. conda-forge libomp) is used.\n"
            "  2. Avoid mixing pip wheels and conda packages that ship their own libomp.\n"
            "  3. Reinstall conflicting packages from a consistent channel.\n"
            "  4. Do NOT rely on KMP_DUPLICATE_LIB_OK in production."
        )


__all__ = ["detect_duplicate_openmp", "raise_if_duplicate_openmp"]
