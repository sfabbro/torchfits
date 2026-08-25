"""Low-level deterministic FITS image reads."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union, cast

import torch
from torch import Tensor

import torchfits._C as _cpp

from ..hdu import Header
from .device import (
    batch_to_device as batch_to_device,
    to_device as to_device,
    validate_device as validate_device,
)
from .paths import guard_fits_path, require_bz2_support


def validate_read_image_args(
    path: str, hdu: int | str, mmap: bool, device: str
) -> None:
    """Validate arguments for low-level read_image."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    require_bz2_support(path)
    if not isinstance(hdu, (int, str)):
        raise ValueError("hdu must be an integer or string")
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    if isinstance(hdu, str) and hdu.strip().lower() == "auto":
        # "auto" is the autodetect sentinel accepted by read()/read_header();
        # read_image()/read_tensor() require an explicit HDU index or name.
        raise ValueError(
            "read_tensor requires an explicit non-negative integer HDU index "
            "or a named EXTNAME (got hdu='auto'); use read() for autodetection"
        )
    if not isinstance(mmap, bool):
        raise ValueError("read_image requires explicit mmap=True/False")
    validate_device(device)


def dispatch_read_image_cpp(
    cpp: Any, path: str, hdu: int, mmap: bool, raw_scale: bool
) -> Tensor:
    """Dispatch the correct C++ function for low-level image reading."""
    if raw_scale:
        if not mmap and hasattr(cpp, "read_full_unmapped_raw"):
            return cast(Tensor, cpp.read_full_unmapped_raw(path, hdu))
        if hasattr(cpp, "read_full_raw"):
            return cast(Tensor, cpp.read_full_raw(path, hdu, mmap))
        return cast(Tensor, cpp.read_full(path, hdu, mmap))
    return cast(Tensor, cpp.read_full(path, hdu, mmap))


def read_image(
    path: str,
    hdu: int | str = 0,
    device: str = "cpu",
    mmap: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    return_header: bool = False,
    fallback_get_header: Callable[[str, int], Header] | None = None,
) -> Union[Tensor, Tuple[Tensor, Header]]:
    """Read image data through a direct low-level path."""
    validate_read_image_args(path, hdu, mmap, device)
    guard_fits_path(path)

    if isinstance(hdu, str):
        if hasattr(_cpp, "resolve_hdu_name_cached"):
            hdu = int(_cpp.resolve_hdu_name_cached(path, hdu))
        else:
            raise ValueError("named HDUs require resolve_hdu_name_cached support")

    data = dispatch_read_image_cpp(_cpp, path, hdu, mmap, raw_scale)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if str(device) != "cpu" and data.device.type == "cpu":
        data = to_device(data, device)

    if return_header:
        try:
            return data, Header(_cpp.read_header_dict(path, hdu))
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
) -> Any:
    """Read multiple image HDUs from one file using a direct one-handle path."""
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    require_bz2_support(path)
    guard_fits_path(path)
    if not isinstance(hdus, (list, tuple)) or len(hdus) == 0:
        raise ValueError("hdus must be a non-empty list/tuple of HDU indices or names")
    validate_device(device)
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
            if hasattr(_cpp, "resolve_hdu_name_cached"):
                resolved_hdus.append(int(_cpp.resolve_hdu_name_cached(path, hdu)))
                continue
            raise ValueError("named HDUs require resolve_hdu_name_cached support")
        raise ValueError("each item in hdus must be an int or str")

    data = _cpp.read_hdus_batch(path, resolved_hdus, mmap)
    if str(device) != "cpu":
        data = batch_to_device(data, device)

    if not return_header:
        return data

    headers = [
        Header(_cpp.read_header_dict(path, hdu_num)) for hdu_num in resolved_hdus
    ]
    return data, headers
