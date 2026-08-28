"""Read option types for root FITS I/O dispatch."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


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
    # Deprecated: per-path handle caching was removed (see caches.get_cached_handle).
    # The value is ignored; kept only so existing callers/signatures do not churn.
    handle_cache_capacity: int = 16
    fast_header: bool = True
    mode: str = "auto"

    def __post_init__(self) -> None:
        if self.handle_cache_capacity != 16:
            warnings.warn(
                "ReadOptions.handle_cache_capacity is deprecated and ignored; handles are no longer cached",
                DeprecationWarning,
                stacklevel=2,
            )
