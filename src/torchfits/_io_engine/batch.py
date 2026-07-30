"""Batch FITS image read helpers."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, cast

from torch import Tensor

from .image import batch_to_device


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
    if not file_paths:
        return []

    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    from .paths import guard_fits_path

    # Fail closed before the C++ batch open so private URLs never hit CFITSIO.
    for path in file_paths:
        guard_fits_path(path)

    try:
        if isinstance(hdu, int) and hdu >= 0:
            import torchfits._C as cpp

            tensors = cpp.read_images_batch(list(file_paths), hdu)
            if device != "cpu":
                tensors = batch_to_device(tensors, device)
            return cast(list[Tensor], tensors)
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
            warnings.warn(
                f"read_batch: skipped {path!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            log.debug("read_batch: skipped %r: %s", path, exc, exc_info=True)
            continue
    return results


def get_batch_info(file_paths: list[str]) -> dict[str, Any]:
    """Get information about a batch of FITS files.

    ``existing_files`` counts paths present on disk (``os.path.exists``); it does
    not open or validate FITS structure. Network URLs are never counted as
    existing (CFITSIO opens them separately). Private/loopback network URLs are
    rejected before the exists scan.
    """
    from .paths import guard_fits_path

    existing_files = 0
    for path in file_paths:
        guard_fits_path(path)
        try:
            if os.path.exists(path):
                existing_files += 1
        except Exception:
            continue

    return {"num_files": len(file_paths), "existing_files": existing_files}
