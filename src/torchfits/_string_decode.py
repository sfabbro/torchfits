"""Numpy-free string decode for 2-D uint8 FITS byte columns.

FITS stores fixed-width string columns as ``uint8`` tensors of shape
``(N, width)``.  This module provides a single helper, :func:`decode_byte_tensor`,
that decodes such a tensor into a ``list[str]`` using only the Python buffer
protocol — no numpy dependency.
"""

from __future__ import annotations

import ctypes

import torch


def decode_byte_tensor(
    tensor: torch.Tensor,
    encoding: str = "ascii",
    strip: bool = True,
) -> list[str]:
    """Decode a 2-D uint8 tensor ``(N, width)`` into a list of strings.

    Replaces the previous ``np.char.decode`` / ``np.char.rstrip`` path with
    pure-Python byte decoding.  Handles CPU transfer, contiguity, and
    storage offset correctly.

    Args:
        tensor: 2-D ``torch.uint8`` tensor of shape ``(N, width)``.
        encoding: Character encoding for ``bytes.decode``.
        strip: When True, trailing spaces and NULs are stripped.

    Returns:
        List of ``N`` decoded strings.
    """
    # detach so we never keep an autograd graph alive for raw byte access
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    n_rows, width = tensor.shape
    if width == 0:
        return [""] * n_rows

    # memcpy the view's window straight from the storage pointer. The obvious
    # ``bytes(tensor.untyped_storage())`` copies the whole backing storage
    # (a small window of a large column would copy the entire column) and
    # ``bytes(UntypedStorage)`` additionally walks elements in Python.
    raw = ctypes.string_at(tensor.data_ptr(), n_rows * width)

    result: list[str] = []
    for i in range(n_rows):
        s = raw[i * width : (i + 1) * width].decode(encoding, errors="ignore")
        if strip:
            s = s.rstrip(" \x00")
        result.append(s)
    return result
