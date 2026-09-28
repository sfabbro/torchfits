"""Shared C++ table read dispatch engine.

Provides :func:`_read_ranges_as_chunk` — reads multiple row ranges from a
C++ TableReader and assembles them into a single torch-backed dict.
"""

from __future__ import annotations

from typing import Any, Optional

from .arrow_convert import (
    _is_raw_column,
    _is_torch_tensor,
    _is_vla_tuple,
    _raw_column_to_numpy,
)


def _segment_row_count(value: Any) -> Optional[int]:
    """Row count carried by one column segment, or None for scalar broadcast."""
    if _is_raw_column(value):
        return int(value["shape"][0])
    if _is_torch_tensor(value):
        return int(value.shape[0])
    if isinstance(value, list):
        return len(value)
    if _is_vla_tuple(value):
        return int(value[1].reshape(-1).shape[0]) - 1
    return None


def _read_raw_ranges_as_chunk(
    reader: Any,
    col_list: list[str],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Assemble native raw ranges into owned NumPy columns for Arrow conversion."""
    import numpy as np

    n_total = sum(length for _, length in ranges)
    if n_total == 0:
        return {}

    expected: Optional[set[str]] = set(col_list) if col_list else None
    fixed_parts: dict[str, list[Any]] = {}
    vla_parts: dict[str, list[tuple[Any, Any]]] = {}

    for start0, length in ranges:
        if length <= 0:
            continue
        seg = reader.read_rows_raw(col_list, start0 + 1, length)
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
            if not _is_raw_column(value):
                raise TypeError(
                    f"raw table reader returned {type(value).__name__} for {name!r}"
                )
            n_got = _segment_row_count(value)
            if n_got != length:
                raise RuntimeError(
                    f"short row segment for rows [{row_lo}, {row_hi}): "
                    f"column {name!r} returned {n_got} of {length} rows"
                )
            materialized = _raw_column_to_numpy(value)
            if value["kind"] == "vla":
                vla_parts.setdefault(name, []).append(materialized)
            else:
                fixed_parts.setdefault(name, []).append(materialized)

    out: dict[str, Any] = {}
    for name, parts in fixed_parts.items():
        out[name] = np.concatenate(parts, axis=0)
    for name, parts in vla_parts.items():
        values = np.concatenate([part[0] for part in parts], axis=0)
        offsets = [parts[0][1]]
        running = int(parts[0][1][-1])
        for _values, next_offsets in parts[1:]:
            base = running
            running += int(next_offsets[-1])
            offsets.append(np.asarray(next_offsets[1:], dtype=np.int64) + base)
        out[name] = (
            values,
            np.concatenate(offsets).astype(np.int64, copy=False),
        )
    return out


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
    if hasattr(reader, "read_rows_raw"):
        return _read_raw_ranges_as_chunk(reader, col_list, ranges)

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
                if _is_torch_tensor(value):
                    import torch

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
            if _is_torch_tensor(value):
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
