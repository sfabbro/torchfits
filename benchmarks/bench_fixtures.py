"""Shared fixture discovery for the CFHT MegaCam benchmark suites."""

from __future__ import annotations

import re
from pathlib import Path

CADC_CFHT_DIRECT_BASE = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"

_FITS_RE = re.compile(r"\.fits(\.fz|\.gz)?$", re.I)


def discover_fits_names(data_dir: Path, *, max_files: int) -> list[str]:
    """Sorted FITS file names under *data_dir* (names only, not paths)."""
    if not data_dir.is_dir():
        return []
    names = sorted(
        p.name for p in data_dir.iterdir() if p.is_file() and _FITS_RE.search(p.name)
    )
    return names[:max_files]


def discover_local_paths(data_dir: Path, *, max_files: int) -> list[str]:
    return [
        str(data_dir / name)
        for name in discover_fits_names(data_dir, max_files=max_files)
    ]


def cadc_direct_urls(names: list[str]) -> list[str]:
    return [f"{CADC_CFHT_DIRECT_BASE}/{name}" for name in names]
