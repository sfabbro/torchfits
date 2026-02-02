"""Minimal I/O helpers for fast image reads."""

from __future__ import annotations

from typing import Union

import torch

import torchfits.cpp as cpp


def read(
    path: Union[str, list[str], tuple[str, ...]],
    hdu: Union[int, list[int], tuple[int, ...]] = 0,
    mmap: bool = True,
    device: str = "cpu",
    fp16: bool = False,
    bf16: bool = False,
    use_cache: bool = True,
    raw_scale: bool = False,
    scale_on_device: bool = True,
):
    """Fast image read with minimal overhead."""
    if not path:
        raise ValueError("Path must be a non-empty string")

    if isinstance(path, (list, tuple)):
        if not isinstance(hdu, int):
            raise ValueError("Batch read requires a single integer HDU")
        data_list = cpp.read_images_batch(list(path), hdu)
        if device != "cpu":
            data_list = [t.to(device) for t in data_list]
        return data_list

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("HDU index must be a non-negative integer")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    if isinstance(hdu, (list, tuple)) and hasattr(cpp, "read_hdus_batch"):
        data = cpp.read_hdus_batch(path, list(hdu))
        if device != "cpu":
            data = [t.to(device) for t in data]
        return data

    if (
        scale_on_device
        and not raw_scale
        and device == "cpu"
        and hasattr(cpp, "read_full_raw_with_scale")
    ):
        data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
        if scaled:
            data = data.to(dtype=torch.float32)
            if bscale != 1.0:
                data.mul_(bscale)
            if bzero != 0.0:
                data.add_(bzero)
        return data
    if scale_on_device and not raw_scale:
        if device == "cpu" and hasattr(cpp, "read_full_raw_with_scale"):
            data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
            if scaled:
                data = data.to(dtype=torch.float32)
                if bscale != 1.0:
                    data.mul_(bscale)
                if bzero != 0.0:
                    data.add_(bzero)
        elif hasattr(cpp, "read_full_raw_with_scale"):
            data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
            if scaled:
                data = data.to(device=device, dtype=torch.float32)
                if bscale != 1.0:
                    data.mul_(bscale)
                if bzero != 0.0:
                    data.add_(bzero)
            else:
                data = data.to(device)
    elif use_cache:
        if raw_scale and hasattr(cpp, "read_full_raw"):
            data = cpp.read_full_raw(path, hdu, mmap)
        else:
            data = cpp.read_full(path, hdu, mmap)
    else:
        if raw_scale and hasattr(cpp, "read_full_unmapped_raw"):
            if not mmap:
                data = cpp.read_full_unmapped_raw(path, hdu)
            elif hasattr(cpp, "read_full_raw"):
                data = cpp.read_full_raw(path, hdu, mmap)
            else:
                data = cpp.read_full(path, hdu, mmap)
        else:
            if not mmap and hasattr(cpp, "read_full_unmapped"):
                data = cpp.read_full_unmapped(path, hdu)
            elif hasattr(cpp, "read_full_nocache"):
                data = cpp.read_full_nocache(path, hdu, mmap)
            else:
                data = cpp.read_full(path, hdu, mmap)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu" and data.device.type == "cpu":
        data = data.to(device)

    return data
