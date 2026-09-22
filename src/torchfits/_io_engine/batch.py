"""Batch FITS image read helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from torch import Tensor

from .device import batch_to_device, validate_device
from .paths import coerce_fits_path


def read_batch(
    read_func: Callable[..., Tensor],
    read_exc_types: tuple[type[BaseException], ...],
    log: logging.Logger,
    file_paths: list[str],
    hdu: int = 0,
    device: str = "cpu",
    *,
    strict: bool = False,
) -> list[Tensor]:
    """Read multiple FITS files in batch."""
    file_paths = coerce_fits_path(file_paths)
    if not file_paths:
        return []

    validate_device(device)

    from .paths import guard_fits_path

    # Fail closed before the C++ batch open so private URLs never hit CFITSIO.
    for path in file_paths:
        guard_fits_path(path)

    try:
        if isinstance(hdu, int) and hdu >= 0:
            import torchfits._C as cpp

            tensors = cpp.read_images_batch(list(file_paths), hdu)
            if len(tensors) != len(file_paths):
                # Contract enforcement: the C++ batch reader must not silently
                # shrink or misalign the result list (r4a-01). Fall through to
                # per-file reads so any bad path raises naming that path.
                raise RuntimeError(
                    f"read_images_batch returned {len(tensors)} of "
                    f"{len(file_paths)} results for {file_paths!r}"
                )
            if str(device) != "cpu":
                tensors = batch_to_device(tensors, device)
            return tensors
    except read_exc_types as exc:
        if strict:
            raise
        log.debug("read_batch: C++ batch path failed, falling back per file: %s", exc)

    results = []
    for path in file_paths:
        try:
            tensor = read_func(path, hdu=hdu, device=device, return_header=False)
            results.append(tensor)
        except read_exc_types as exc:
            if strict:
                raise
            log.debug("read_batch: %r failed: %s", path, exc, exc_info=True)
            raise RuntimeError(
                f"read_batch: failed to read {path!r} ({len(results)} of "
                f"{len(file_paths)} files read before the failure): {exc}"
            ) from exc
    return results


def get_batch_info(file_paths: list[str]) -> dict[str, Any]:
    """Get information about a batch of FITS files.

    ``existing_files`` counts paths present on disk (``os.path.exists``); it does
    not open or validate FITS structure. Network URLs are never counted as
    existing (CFITSIO opens them separately). Private/loopback network URLs are
    rejected before the exists scan.
    """
    from .paths import guard_fits_path

    file_paths = coerce_fits_path(file_paths)
    existing_files = 0
    for path in file_paths:
        guard_fits_path(path)
        try:
            if os.path.exists(path):
                existing_files += 1
        except OSError:
            continue

    return {"num_files": len(file_paths), "existing_files": existing_files}
