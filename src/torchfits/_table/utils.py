"""Private helpers for table I/O."""

from __future__ import annotations

from typing import Any, Optional

from .. import fits_schema


_TABLE_IO_KEYS = {
    "hdu",
    "columns",
    "row_slice",
    "rows",
    "where",
    "batch_size",
    "mmap",
    "decode_bytes",
    "encoding",
    "strip",
    "include_fits_metadata",
    "apply_fits_nulls",
    "backend",
}


def _normalize_cpp_table_data(data: Any) -> Any:
    from torchfits.io import _normalize_cpp_table_data as normalize

    return normalize(data)


def _write_header_cards_if_supported(path: str, hdu: int, hdr: Any) -> None:
    from torchfits.io import _write_header_cards_if_supported as write_hdr

    write_hdr(path, hdu, hdr)


def _parse_tform(tform: str) -> tuple[bool, str, int]:
    info = fits_schema.parse_tform(tform)
    return info.vla, info.code or "", info.repeat


def _column_tnull_map(header_map: dict[str, Any]) -> dict[str, Any]:
    return fits_schema.column_tnull_map(header_map)


def _naxis2_row_count(header_map: dict[str, Any], path: str) -> int:
    """Return the table row count from the NAXIS2 card.

    A missing NAXIS2 legitimately means zero rows.  A present-but-malformed
    value is a corrupt header and raises instead of being silently coerced to
    zero rows (which would turn row mutations into silent no-ops).
    """
    try:
        return int(header_map.get("NAXIS2", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot parse NAXIS2 row count {header_map.get('NAXIS2')!r} for {path!r}"
        ) from exc


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for torchfits.table APIs. Install pyarrow to use Arrow-native tables."
        ) from exc
    return pa


def _arrow_column_to_python(pa: Any, column: Any, name: str) -> Any:
    import numpy as np

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    if column.null_count:
        raise ValueError(
            f"Arrow column '{name}' contains nulls (not supported for FITS updates)"
        )

    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column.to_pylist()
    if pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type):
        return column.to_pylist()
    if pa.types.is_fixed_size_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False)
        size = column.type.list_size
        return values.reshape((len(column), size))
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        pylist_values: list[Any] = column.to_pylist()
        out: list[Any] = []
        for item in pylist_values:
            if item is None:
                out.append([])
            else:
                out.append(np.asarray(item))
        return out

    return column.to_numpy(zero_copy_only=False)


def _normalize_row_slice(
    row_slice: Optional[slice | tuple[int, int]],
) -> tuple[int, int]:
    if row_slice is None:
        return 1, -1

    if isinstance(row_slice, tuple):
        if len(row_slice) != 2:
            raise ValueError("row_slice tuple must be (start, stop)")
        start, stop = row_slice
        step = 1
    elif isinstance(row_slice, slice):
        start = 0 if row_slice.start is None else row_slice.start
        stop = row_slice.stop
        step = 1 if row_slice.step is None else row_slice.step
    else:
        raise ValueError("row_slice must be a slice, (start, stop), or None")

    if step != 1:
        raise ValueError("row_slice step must be 1 for FITS row streaming")
    if start < 0:
        raise ValueError("row_slice start must be >= 0")
    if stop is not None and stop < 0:
        raise ValueError(
            "row_slice negative stop is not supported (total row count is "
            "unknown at parse time); use a non-negative stop or None"
        )

    start_row = start + 1
    if stop is None:
        return start_row, -1
    if stop < start:
        return start_row, 0
    return start_row, stop - start


def _fits_tform_is_bit(tform: Any) -> bool:
    return fits_schema.tform_is_bit(tform)
