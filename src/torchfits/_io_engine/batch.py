"""Batch FITS image read helpers."""

from __future__ import annotations

import os
from typing import Any, Callable

from torch import Tensor

from .image import batch_to_device


def read_batch(
    read_func: Callable[..., Tensor],
    read_exc_types: tuple[type[BaseException], ...],
    log,
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

    try:
        if isinstance(hdu, int) and hdu >= 0:
            import torchfits._C as cpp

            tensors = cpp.read_images_batch(list(file_paths), hdu)
            if device != "cpu":
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
            log.debug("read_batch: skipped %r: %s", path, exc, exc_info=True)
            continue
    return results


def get_batch_info(file_paths: list[str]) -> dict[str, Any]:
    """Get information about a batch of FITS files."""
    valid_files = 0
    for path in file_paths:
        try:
            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue

    return {"num_files": len(file_paths), "valid_files": valid_files}
