"""Minimal I/O helpers for fast image reads."""

from typing import Union

import torch

try:
    import torchfits.cpp as cpp
except ImportError:
    cpp = None

_HAS_READ_HDUS_BATCH = hasattr(cpp, "read_hdus_batch")
_HAS_READ_FULL_RAW_WITH_SCALE = hasattr(cpp, "read_full_raw_with_scale")
_HAS_READ_FULL_RAW = hasattr(cpp, "read_full_raw")
_HAS_READ_FULL_UNMAPPED_RAW = hasattr(cpp, "read_full_unmapped_raw")
_HAS_READ_FULL_UNMAPPED = hasattr(cpp, "read_full_unmapped")
_HAS_READ_FULL_NOCACHE = hasattr(cpp, "read_full_nocache")


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

    is_cpu = device == "cpu"
    target_device = None if is_cpu else device
    target_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    if isinstance(path, (list, tuple)):
        if not all(isinstance(p, str) for p in path):
            raise ValueError("Path must be a string or list of strings")
        if not isinstance(hdu, int):
            raise ValueError("Batch read requires a single integer HDU")
        data_list = cpp.read_images_batch(list(path), hdu)
        if not is_cpu or target_dtype is not None:
            data_list = [
                t.to(device=target_device, dtype=target_dtype) for t in data_list
            ]
        return data_list

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    if not isinstance(hdu, (int, list, tuple)):
        raise ValueError("HDU index must be a non-negative integer")
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be a non-negative integer")
    if isinstance(hdu, (list, tuple)) and any(not isinstance(h, int) or h < 0 for h in hdu):
        raise ValueError("HDU index must be a non-negative integer")
    if not is_cpu and device not in ["cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    if isinstance(hdu, (list, tuple)) and _HAS_READ_HDUS_BATCH:
        data = cpp.read_hdus_batch(path, list(hdu))
        if not is_cpu or target_dtype is not None:
            data = [t.to(device=target_device, dtype=target_dtype) for t in data]
        return data

    if scale_on_device and not raw_scale and _HAS_READ_FULL_RAW_WITH_SCALE:
        data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
        if scaled:
            data = data.to(device=target_device, dtype=torch.float32)
            if bscale != 1.0:
                data.mul_(bscale)
            if bzero != 0.0:
                data.add_(bzero)

            if target_dtype is not None:
                data = data.to(dtype=target_dtype)
            return data

        if not is_cpu or target_dtype is not None:
            data = data.to(device=target_device, dtype=target_dtype)
        return data

    if use_cache:
        if raw_scale and _HAS_READ_FULL_RAW:
            data = cpp.read_full_raw(path, hdu, mmap)
        else:
            data = cpp.read_full(path, hdu, mmap)
    else:
        if raw_scale and _HAS_READ_FULL_UNMAPPED_RAW:
            if not mmap:
                data = cpp.read_full_unmapped_raw(path, hdu)
            elif _HAS_READ_FULL_RAW:
                data = cpp.read_full_raw(path, hdu, mmap)
            else:
                data = cpp.read_full(path, hdu, mmap)
        else:
            if not mmap and _HAS_READ_FULL_UNMAPPED:
                data = cpp.read_full_unmapped(path, hdu)
            elif _HAS_READ_FULL_NOCACHE:
                data = cpp.read_full_nocache(path, hdu, mmap)
            else:
                data = cpp.read_full(path, hdu, mmap)

    if target_dtype is not None or (not is_cpu and data.device.type == "cpu"):
        data = data.to(device=target_device, dtype=target_dtype)

    return data
