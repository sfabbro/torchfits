"""Table-shaped wrappers over unified FITS reads."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch


def read_table(
    read_func: Callable[..., Any],
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
):
    """Read a table HDU as a dictionary of tensors/lists."""
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")

    out = read_func(
        path=path,
        hdu=hdu,
        mode="table",
        device=device,
        mmap=mmap,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
    )
    data = out[0] if return_header else out
    if isinstance(data, torch.Tensor):
        raise ValueError(
            f"HDU {hdu!r} is an image HDU. Use read_image(...) or read(...)."
        )
    return out
