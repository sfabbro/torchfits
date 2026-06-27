"""Low-level deterministic FITS image reads."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..hdu import Header

from .read_dispatch import _read_unsigned_image_if_needed


def batch_to_device(
    tensors: list[torch.Tensor], device: str | torch.device
) -> list[torch.Tensor]:
    """Move a list of tensors to a device, stacking when shapes match."""
    if not tensors:
        return []
    if len(tensors) == 1:
        return [tensors[0].to(device, non_blocking=True)]

    first = tensors[0]
    shape = first.shape
    dtype = first.dtype

    if all(t.shape == shape and t.dtype == dtype for t in tensors):
        return list(torch.stack(tensors).to(device, non_blocking=True).unbind(0))
    return [t.to(device, non_blocking=True) for t in tensors]


def validate_read_image_args(
    path: str, hdu: int, mmap: bool, handle_cache: bool, device: str
) -> None:
    """Validate arguments for low-level read_image."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    if not isinstance(mmap, bool):
        raise ValueError("read_image requires explicit mmap=True/False")
    if not isinstance(handle_cache, bool):
        raise ValueError("handle_cache must be bool")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")


def dispatch_read_image_cpp(
    cpp, path: str, hdu: int, mmap: bool, handle_cache: bool, raw_scale: bool
) -> Tensor:
    """Dispatch the correct C++ function for low-level image reading."""
    if raw_scale:
        if not mmap and hasattr(cpp, "read_full_unmapped_raw"):
            return cpp.read_full_unmapped_raw(path, hdu)
        elif hasattr(cpp, "read_full_raw"):
            return cpp.read_full_raw(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)
    else:
        if handle_cache and hasattr(cpp, "read_full_cached"):
            return cpp.read_full_cached(path, hdu, mmap)
        elif not mmap and hasattr(cpp, "read_full_unmapped"):
            return cpp.read_full_unmapped(path, hdu)
        elif hasattr(cpp, "read_full_nocache"):
            return cpp.read_full_nocache(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)


def read_image(
    path: str,
    hdu: int = 0,
    device: str = "cpu",
    mmap: bool = True,
    handle_cache: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    return_header: bool = False,
    fallback_get_header: Callable[[str, int], Header] | None = None,
) -> Union[Tensor, Tuple[Tensor, Header]]:
    """Read image data through a direct low-level path."""
    import torchfits._C as cpp

    validate_read_image_args(path, hdu, mmap, handle_cache, device)

    data = None
    if not raw_scale and not (fp16 or bf16):
        try:
            header = Header(cpp.read_header_dict(path, hdu))
        except Exception:
            header = None
        data = _read_unsigned_image_if_needed(
            cpp_module=cpp,
            path=path,
            hdu_num=hdu,
            effective_mmap=mmap,
            header=header,
        )
    if data is None:
        data = dispatch_read_image_cpp(cpp, path, hdu, mmap, handle_cache, raw_scale)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu" and data.device.type == "cpu":
        data = data.to(device)

    if return_header:
        try:
            return data, Header(cpp.read_header_dict(path, hdu))
        except Exception:
            if fallback_get_header is None:
                raise
            return data, fallback_get_header(path, hdu)
    return data


def read_hdus(
    path: str,
    hdus: Sequence[Union[int, str]],
    *,
    device: str = "cpu",
    mmap: bool = True,
    return_header: bool = False,
):
    """Read multiple image HDUs from one file using a direct one-handle path."""
    import torchfits._C as cpp

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(hdus, (list, tuple)) or len(hdus) == 0:
        raise ValueError("hdus must be a non-empty list/tuple of HDU indices or names")
    if device not in ["cpu", "cuda", "mps"] and not str(device).startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    if not isinstance(mmap, bool):
        raise ValueError("mmap must be a bool for read_hdus")

    resolved_hdus: list[int] = []
    for hdu in hdus:
        if isinstance(hdu, int):
            if hdu < 0:
                raise ValueError("HDU index must be non-negative")
            resolved_hdus.append(int(hdu))
            continue
        if isinstance(hdu, str):
            if hasattr(cpp, "resolve_hdu_name_cached"):
                resolved_hdus.append(int(cpp.resolve_hdu_name_cached(path, hdu)))
                continue
            raise ValueError("named HDUs require resolve_hdu_name_cached support")
        raise ValueError("each item in hdus must be an int or str")

    data = cpp.read_hdus_batch(path, resolved_hdus, mmap)
    for idx, hdu_num in enumerate(resolved_hdus):
        try:
            header = Header(cpp.read_header_dict(path, hdu_num))
        except Exception:
            header = None
        unsigned = _read_unsigned_image_if_needed(
            cpp_module=cpp,
            path=path,
            hdu_num=hdu_num,
            effective_mmap=mmap,
            header=header,
        )
        if unsigned is not None:
            data[idx] = unsigned
    if device != "cpu":
        data = batch_to_device(data, device)

    if not return_header:
        return data

    headers = [Header(cpp.read_header_dict(path, hdu_num)) for hdu_num in resolved_hdus]
    return data, headers
