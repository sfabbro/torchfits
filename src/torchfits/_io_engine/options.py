"""Read option types for root FITS I/O dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict


@dataclass
class ReadOptions:
    """Configuration options for reading FITS files."""

    fp16: bool = False
    bf16: bool = False
    raw_scale: bool = False
    scale_on_device: bool = True
    use_cache: Optional[bool] = None
    columns: Optional[list[str]] = None
    start_row: int = 1
    num_rows: int = -1
    cache_capacity: int = 10
    handle_cache_capacity: int = 16
    fast_header: bool = True
    mode: str = "auto"
    policy: str = "default"


class ReadOptionsKwargs(TypedDict, total=False):
    fp16: bool
    bf16: bool
    raw_scale: bool
    scale_on_device: bool
    use_cache: Optional[bool]
    columns: Optional[list[str]]
    start_row: int
    num_rows: int
    cache_capacity: int
    handle_cache_capacity: int
    fast_header: bool
    mode: str
    policy: str
