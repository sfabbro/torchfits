"""Shared C++ table read dispatch engine.

Provides :func:`_read_ranges_as_chunk` — reads multiple row ranges from a
C++ TableReader and assembles them into a single torch-backed dict.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .arrow_convert import _is_vla_tuple


def _segment_row_count(value: Any) -> Optional[int]:
    """Row count carried by one column segment, or None for scalar broadcast."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if isinstance(value, list):
        return len(value)
    if _is_vla_tuple(value):
        return int(value[1].reshape(-1).shape[0]) - 1
    return None


def _read_ranges_as_chunk(
    reader: Any,
    col_list: list[str],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Read multiple row ranges from a TableReader and assemble into one chunk.

    Each ``(start0, length)`` range triggers one ``reader.read_rows`` round-trip
    to CFITSIO, so callers MUST pass *coalesced* ranges (adjacent/contiguous
    rows merged into a single range) to avoid N small reads for scattered row
    lists. Zero-length ranges are skipped without a round-trip.

    A segment that is empty or short for a non-empty range raises instead of
    fabricating rows: unreadable rows used to surface as zeros (tensors) or
    ``None`` (lists) and short list segments silently spliced the buffer
    (r5a-06, A-14).
    """
    import numpy as np

    out_sorted: dict[str, Any] = {}
    n_total = sum(length for _, length in ranges)
    if n_total == 0:
        return {}

    expected: Optional[set[str]] = set(col_list) if col_list else None
    cursor = 0
    for start0, length in ranges:
        if length <= 0:
            continue
        seg = reader.read_rows(col_list, start0 + 1, length)
        row_lo, row_hi = start0, start0 + length
        if not seg:
            raise RuntimeError(
                f"empty row segment for rows [{row_lo}, {row_hi}): "
                "refusing to fabricate rows"
            )
        if expected is None:
            expected = set(seg.keys())
        missing = expected - set(seg.keys())
        if missing:
            raise RuntimeError(
                f"row segment for rows [{row_lo}, {row_hi}) is missing "
                f"columns {sorted(missing)}"
            )
        for name, value in seg.items():
            buf: Any = out_sorted.get(name)
            if buf is None:
                if isinstance(value, torch.Tensor):
                    # Zero-initialized: an empty reader segment must surface
                    # as zeros, never uninitialized memory. Segments are
                    # length-validated below, so every row slot is filled.
                    buf = torch.zeros(
                        (n_total,) + tuple(value.shape[1:]), dtype=value.dtype
                    )
                else:
                    buf = [None] * n_total
                out_sorted[name] = buf

            n_got = _segment_row_count(value)
            if n_got is not None and n_got != length:
                raise RuntimeError(
                    f"short row segment for rows [{row_lo}, {row_hi}): "
                    f"column {name!r} returned {n_got} of {length} rows"
                )
            if isinstance(value, torch.Tensor):
                buf[cursor : cursor + length] = value
            elif isinstance(value, list):
                buf[cursor : cursor + length] = value
            elif _is_vla_tuple(value):
                fixed, offsets = value
                fixed = np.asarray(fixed)
                offsets = np.asarray(offsets)
                items = []
                for i in range(length):
                    a = int(offsets[i])
                    b = int(offsets[i + 1])
                    items.append(fixed[a:b])
                buf[cursor : cursor + length] = items
            else:
                buf[cursor : cursor + length] = [value] * length
        cursor += length
    return out_sorted
