"""Table mutation: row/column insert, replace, delete, update, rename."""

from __future__ import annotations

from typing import Any, Optional

from .._table.utils import (
    _column_tnull_map,
    _normalize_row_slice,
    _parse_tform,
)
from .._table.write import (
    _extract_table_schema_from_header,
    _ordered_dict_for_columns,
    _resolve_table_hdu_index_and_columns,
    _rewrite_table_hdu_with_schema,
    _sanitize_table_header_for_rewrite,
)

from ._mutation_coerce import (
    _COMPLEX_TFORM_CODES,
    _coerce_rows_from_arrow,
    _coerce_table_column_array,
    _coerce_table_complex_values,
    _coerce_table_string_values,
    _coerce_table_vla_values,
    _delete_column_rows,
    _ensure_dtype_maps,
    _infer_column_format_for_insert,
    _merge_insert_column,
    _mutation_cache_barrier,
    _normalize_column_values_for_format,
    _normalize_mutation_rows,
    _read_table_for_rewrite,
)

# Re-export private helpers for backward-compatible imports from this module.
from ._mutation_coerce import (  # noqa: F401
    _default_table_column_values,
    _infer_fits_scalar_code,
)


def insert_column(
    path: str,
    name: str,
    values: Any,
    *,
    hdu: int | str = 1,
    index: Optional[int] = None,
    format: Optional[str] = None,
    unit: Optional[str] = None,
    dim: Optional[str] = None,
    tnull: Optional[Any] = None,
    tscal: Optional[float] = None,
    tzero: Optional[float] = None,
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    target_hdu, header_map, columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    if name in columns:
        raise ValueError(f"Column '{name}' already exists")

    if index is None:
        index = len(columns)
    if not isinstance(index, int) or index < 0 or index > len(columns):
        raise ValueError(f"index must be in [0, {len(columns)}]")

    try:
        num_rows = int(header_map.get("NAXIS2", 0))
    except Exception:
        num_rows = 0

    fmt = (
        str(format).strip().upper()
        if format is not None
        else _infer_column_format_for_insert(name, values)
    )
    normalized_values = _normalize_column_values_for_format(name, values, fmt, num_rows)

    existing_data = _read_table_for_rewrite(path, target_hdu, columns)
    existing_schema = _extract_table_schema_from_header(header_map, columns)
    table_header = _sanitize_table_header_for_rewrite(header_map)
    table_type = (
        "ascii"
        if str(header_map.get("XTENSION", "")).strip().upper() == "TABLE"
        else "binary"
    )

    new_columns = list(columns)
    new_columns.insert(index, name)
    data_by_name = dict(existing_data)
    data_by_name[name] = normalized_values
    rewritten_data = _ordered_dict_for_columns(new_columns, data_by_name)

    new_meta: dict[str, Any] = {"format": fmt}
    if unit is not None:
        new_meta["unit"] = str(unit)
    if dim is not None:
        new_meta["dim"] = str(dim)
    if tnull is not None:
        new_meta["tnull"] = tnull
    if tscal is not None:
        new_meta["bscale"] = float(tscal)
    if tzero is not None:
        new_meta["bzero"] = float(tzero)

    schema_by_name = dict(existing_schema)
    schema_by_name[name] = new_meta
    rewritten_schema = _ordered_dict_for_columns(new_columns, schema_by_name)

    _mutation_cache_barrier(path)
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    _mutation_cache_barrier(path)


def replace_column(
    path: str,
    name: str,
    values: Any,
    *,
    hdu: int | str = 1,
    format: Optional[str] = None,
    unit: Optional[str] = None,
    dim: Optional[str] = None,
    tnull: Optional[Any] = None,
    tscal: Optional[float] = None,
    tzero: Optional[float] = None,
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    target_hdu, header_map, columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    if name not in columns:
        raise KeyError(f"Column '{name}' not found")

    try:
        num_rows = int(header_map.get("NAXIS2", 0))
    except Exception:
        num_rows = 0

    existing_schema = _extract_table_schema_from_header(header_map, columns)
    existing_meta = dict(existing_schema.get(name, {}))
    fmt = (
        str(format).strip().upper()
        if format is not None
        else str(existing_meta.get("format", "")).strip().upper()
    )
    if not fmt:
        fmt = _infer_column_format_for_insert(name, values)

    normalized_values = _normalize_column_values_for_format(name, values, fmt, num_rows)

    table_header = _sanitize_table_header_for_rewrite(header_map)
    table_type = (
        "ascii"
        if str(header_map.get("XTENSION", "")).strip().upper() == "TABLE"
        else "binary"
    )
    rewritten_data = _read_table_for_rewrite(path, target_hdu, columns)
    rewritten_data[name] = normalized_values

    merged_meta = dict(existing_meta)
    merged_meta["format"] = fmt
    if unit is not None:
        merged_meta["unit"] = str(unit)
    if dim is not None:
        merged_meta["dim"] = str(dim)
    if tnull is not None:
        merged_meta["tnull"] = tnull
    if tscal is not None:
        merged_meta["bscale"] = float(tscal)
    if tzero is not None:
        merged_meta["bzero"] = float(tzero)
    existing_schema[name] = merged_meta
    rewritten_schema = _ordered_dict_for_columns(columns, existing_schema)
    rewritten_data = _ordered_dict_for_columns(columns, rewritten_data)

    _mutation_cache_barrier(path)
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    _mutation_cache_barrier(path)


# -- public row mutation API -----------------------------------------------------


def append_rows(
    path: str,
    rows: dict[str, Any],
    hdu: int | str = 1,
) -> None:
    rows = _coerce_rows_from_arrow(rows)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("rows must be a non-empty dictionary")
    import torchfits._C as cpp

    target_hdu, header_map, columns, tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    tnull_map = _column_tnull_map(header_map)
    normalized, expected_rows = _normalize_mutation_rows(
        rows,
        columns,
        tform_map,
        tnull_map,
        allow_partial=True,
    )
    if expected_rows <= 0:
        return

    _mutation_cache_barrier(path)
    cpp.append_fits_table_rows(path, target_hdu, normalized)
    _mutation_cache_barrier(path)


def insert_rows(
    path: str,
    rows: dict[str, Any],
    *,
    row: int,
    hdu: int | str = 1,
) -> None:
    rows = _coerce_rows_from_arrow(rows)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("rows must be a non-empty dictionary")
    if not isinstance(row, int) or row < 0:
        raise ValueError("row must be a non-negative integer")

    import torchfits
    import torchfits._C as cpp

    target_hdu, header_map, columns, tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    try:
        total_rows = int(header_map.get("NAXIS2", 0))
    except Exception:
        total_rows = 0
    if row > total_rows:
        raise ValueError(
            f"row index {row} is out of range for insert (num_rows={total_rows})"
        )

    tnull_map = _column_tnull_map(header_map)
    normalized, expected_rows = _normalize_mutation_rows(
        rows,
        columns,
        tform_map,
        tnull_map,
        allow_partial=True,
    )
    if expected_rows <= 0:
        return

    start_row = row + 1
    _mutation_cache_barrier(path)
    if hasattr(cpp, "insert_fits_table_rows"):
        cpp.insert_fits_table_rows(path, target_hdu, normalized, start_row)
    else:
        existing = _read_table_for_rewrite(path, target_hdu, columns)
        rewritten: dict[str, Any] = {}
        for name in columns:
            rewritten[name] = _merge_insert_column(
                existing[name], normalized[name], row
            )
        torchfits.replace_hdu(path, target_hdu, rewritten)
    _mutation_cache_barrier(path)


def delete_rows(
    path: str,
    row_slice: int | slice | tuple[int, int],
    *,
    hdu: int | str = 1,
) -> None:
    if isinstance(row_slice, int):
        if row_slice < 0:
            raise ValueError("row index must be >= 0")
        norm_slice: slice | tuple[int, int] = slice(row_slice, row_slice + 1)
    else:
        norm_slice = row_slice

    start_row, num_rows = _normalize_row_slice(norm_slice)
    if num_rows == 0:
        return

    import torchfits
    import torchfits._C as cpp

    target_hdu, header_map, _columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    try:
        total_rows = int(header_map.get("NAXIS2", 0))
    except Exception:
        total_rows = 0
    if total_rows <= 0:
        return
    if start_row > total_rows:
        raise ValueError(
            f"row_slice start is out of range for delete (start={start_row - 1}, num_rows={total_rows})"
        )
    if num_rows < 0:
        num_rows = total_rows - start_row + 1
    if num_rows <= 0:
        return

    _mutation_cache_barrier(path)
    if hasattr(cpp, "delete_fits_table_rows"):
        cpp.delete_fits_table_rows(path, target_hdu, start_row, num_rows)
    else:
        columns = _columns
        existing = _read_table_for_rewrite(path, target_hdu, columns)
        start0 = start_row - 1
        rewritten: dict[str, Any] = {}
        for name in columns:
            rewritten[name] = _delete_column_rows(existing[name], start0, num_rows)
        torchfits.replace_hdu(path, target_hdu, rewritten)
    _mutation_cache_barrier(path)


def update_rows(
    path: str,
    rows: dict[str, Any],
    row_slice: slice | tuple[int, int],
    hdu: int | str = 1,
    *,
    mmap: bool | str = "auto",
) -> None:
    rows = _coerce_rows_from_arrow(rows)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("rows must be a non-empty dictionary")
    if row_slice is None:
        raise ValueError("row_slice is required for update_rows")

    target_hdu, _header, columns, tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    unknown = sorted({str(name) for name in rows} - set(columns))
    if unknown:
        raise ValueError(f"Unknown columns for table mutation: extra={unknown}")

    string_widths: dict[str, int] = {}
    vla_codes: dict[str, str] = {}
    complex_codes: dict[str, str] = {}
    _ensure_dtype_maps()
    for name, tform in tform_map.items():
        if not tform:
            continue
        is_vla, code, repeat = _parse_tform(tform)
        if is_vla:
            vla_codes[name] = code
        elif code in _COMPLEX_TFORM_CODES:
            complex_codes[name] = code
        elif code == "A":
            string_widths[name] = repeat

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        return

    normalized: dict[str, Any] = {}
    expected_rows: Optional[int] = None
    for name, value in rows.items():
        col_name = str(name)
        if col_name in vla_codes:
            values = _coerce_table_vla_values(
                col_name, value, vla_codes[col_name], expected_rows=expected_rows
            )
            if expected_rows is None:
                expected_rows = len(values)
            normalized[col_name] = values
        elif col_name in string_widths:
            values = _coerce_table_string_values(  # type: ignore[assignment]
                col_name, value, expected_rows=expected_rows
            )
            if expected_rows is None:
                expected_rows = len(values)
            import numpy as _np

            width = string_widths[col_name]
            arr = _np.full((expected_rows, width), 0x20, dtype=_np.uint8)
            for i, s in enumerate(values):
                if isinstance(s, (bytes, bytearray)):
                    encoded = bytes(s)
                elif isinstance(s, str):
                    encoded = s.encode("ascii", "ignore")
                else:
                    encoded = str(s).encode("ascii", "ignore")
                length = min(len(encoded), width)
                if length > 0:
                    arr[i, :length] = _np.frombuffer(encoded[:length], dtype=_np.uint8)
            normalized[col_name] = arr
        elif col_name in complex_codes:
            arr = _coerce_table_complex_values(
                col_name,
                value,
                complex_codes[col_name],
                expected_rows=expected_rows,
                allow_2d=True,
            )
            if expected_rows is None:
                expected_rows = int(arr.shape[0])
            normalized[col_name] = arr
        else:
            arr = _coerce_table_column_array(
                col_name, value, expected_rows=expected_rows, allow_2d=True
            )
            if expected_rows is None:
                expected_rows = int(arr.shape[0])
            normalized[col_name] = arr

    if expected_rows is None:
        return
    if num_rows < 0:
        num_rows = expected_rows
    if expected_rows != num_rows:
        raise ValueError(
            f"row_slice expects {num_rows} rows, but update payload has {expected_rows}"
        )

    import torchfits._C as cpp

    _mutation_cache_barrier(path)

    use_mmap = mmap in (True, "auto", "mmap")
    forced_mmap = mmap in (True, "mmap")
    unsupported_mmap = sorted(name for name in normalized if name in vla_codes)
    if forced_mmap and unsupported_mmap:
        raise ValueError(
            "mmap table updates do not support variable-length-array columns; "
            f"unsupported columns={unsupported_mmap}"
        )
    if use_mmap:
        has_string = any(isinstance(v, (list, tuple)) for v in normalized.values())
        if not has_string:
            try:
                cpp.update_fits_table_rows_mmap(
                    path, target_hdu, normalized, start_row, num_rows
                )
                _mutation_cache_barrier(path)
                return
            except Exception:
                if forced_mmap:
                    raise

    cpp.update_fits_table_rows(path, target_hdu, normalized, start_row, num_rows)
    _mutation_cache_barrier(path)


def rename_columns(
    path: str,
    mapping: dict[str, str],
    hdu: int | str = 1,
) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a non-empty dictionary")

    normalized: dict[str, str] = {}
    for old, new in mapping.items():
        old_name = str(old)
        new_name = str(new)
        if not old_name or not new_name:
            raise ValueError("column names must be non-empty strings")
        normalized[old_name] = new_name

    if len(set(normalized.values())) != len(normalized.values()):
        raise ValueError("rename_columns mapping has duplicate target names")

    target_hdu, _header, columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    existing = set(columns)
    missing = sorted(set(normalized) - existing)
    if missing:
        raise KeyError(f"Column(s) not found for rename_columns: {missing}")
    conflicts = sorted(set(normalized.values()) & (existing - set(normalized)))
    if conflicts:
        raise ValueError(
            "rename_columns target names collide with existing columns not being renamed: "
            f"{conflicts}"
        )

    import torchfits._C as cpp

    _mutation_cache_barrier(path)
    cpp.rename_fits_table_columns(path, target_hdu, normalized)
    _mutation_cache_barrier(path)


def drop_columns(
    path: str,
    columns: list[str] | tuple[str, ...],
    hdu: int | str = 1,
) -> None:
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("columns must be a non-empty list of column names")

    normalized = [str(name) for name in columns]
    if any(not name for name in normalized):
        raise ValueError("column names must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("drop_columns received duplicate column names")

    target_hdu, _header, existing_columns, _tform_map = (
        _resolve_table_hdu_index_and_columns(path, hdu)
    )
    missing = sorted(set(normalized) - set(existing_columns))
    if missing:
        raise KeyError(f"Column(s) not found for drop_columns: {missing}")

    import torchfits._C as cpp

    _mutation_cache_barrier(path)
    cpp.drop_fits_table_columns(path, target_hdu, normalized)
    _mutation_cache_barrier(path)
