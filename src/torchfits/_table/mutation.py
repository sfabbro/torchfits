"""Table mutation: row/column insert, replace, delete, update, rename, and coercion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

from .._io_engine.caches import invalidate_path_caches as _invalidate_path_caches
from .._table.utils import (
    _arrow_column_to_python,
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

# -- module-level dtype maps (populated once on first use) -----------------------

_VLA_DTYPE_MAP: dict[str, Any] = {}
_COMPLEX_DTYPE_MAP: dict[str, Any] = {}
_COMPLEX_TFORM_CODES: frozenset[str] = frozenset({"C", "M"})


def _ensure_dtype_maps() -> None:
    """Fill ``_VLA_DTYPE_MAP`` / ``_COMPLEX_DTYPE_MAP`` once (numpy-backed)."""
    global _VLA_DTYPE_MAP, _COMPLEX_DTYPE_MAP
    if _COMPLEX_DTYPE_MAP:
        return
    import numpy as np

    _VLA_DTYPE_MAP = {
        "L": np.bool_,
        "B": np.uint8,
        "I": np.int16,
        "J": np.int32,
        "K": np.int64,
        "E": np.float32,
        "D": np.float64,
        "C": np.complex64,
        "M": np.complex128,
    }
    _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}


def _mutation_cache_barrier(path: str) -> None:
    """Invalidate path-local caches around a table mutation.

    Called once before and once after the on-disk rewrite/append/delete op
    in each mutation function below, so callers never observe a stale
    handle or cached read for ``path``. Does not clear unrelated paths.
    """
    _invalidate_path_caches(path)


# -- helpers moved from write section (used only by mutation) --------------------


def _infer_fits_scalar_code(arr: "np.ndarray") -> str:
    kind = arr.dtype.kind
    itemsize = arr.dtype.itemsize
    if kind == "b":
        return "L"
    if kind == "u" and itemsize == 1:
        return "B"
    if kind == "u" and itemsize in (2, 4, 8):
        signed = {2: "int16", 4: "int32", 8: "int64"}[itemsize]
        raise TypeError(
            f"Cannot infer FITS TFORM for dtype={arr.dtype}: FITS has no native "
            f"unsigned {itemsize * 8}-bit integer format. Cast the column to "
            f"{signed} (values will be reinterpreted, not rescaled), or write it "
            "explicitly as a signed column with a BZERO offset (the FITS unsigned "
            "convention) instead of relying on format inference."
        )
    if kind == "i" and itemsize == 2:
        return "I"
    if kind == "i" and itemsize == 4:
        return "J"
    if kind == "i" and itemsize == 8:
        return "K"
    if kind == "f" and itemsize == 4:
        return "E"
    if kind == "f" and itemsize == 8:
        return "D"
    if kind == "c" and itemsize == 8:
        return "C"
    if kind == "c" and itemsize == 16:
        return "M"
    raise TypeError(f"Cannot infer FITS TFORM for dtype={arr.dtype}")


def _infer_fits_format(arr: "np.ndarray") -> str:
    import numpy as np

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim == 1 and arr.dtype.kind in {"U", "S"}:
        if arr.dtype.kind == "U":
            width = max(1, int(max((len(x) for x in arr.tolist()), default=1)))
        else:
            width = max(1, int(arr.dtype.itemsize))
        return f"{width}A"

    if arr.ndim == 2 and arr.dtype == np.uint8:
        return f"{int(arr.shape[1])}A"

    if arr.dtype == np.object_:
        raise TypeError("Object/VLA columns require explicit schema['format']")

    base = _infer_fits_scalar_code(arr)
    if arr.ndim == 1:
        return f"1{base}"
    repeat = int(np.prod(arr.shape[1:]))
    return f"{repeat}{base}"


def _prepare_array_for_column(arr: "np.ndarray", fmt: str) -> "np.ndarray":
    import numpy as np

    if arr.ndim == 0:
        return arr.reshape(1)

    tform = str(fmt).strip().upper()
    if tform.endswith("A") and arr.ndim == 2 and arr.dtype == np.uint8:
        width = int(arr.shape[1])
        return (
            np.ascontiguousarray(arr).view(np.dtype(f"S{width}")).reshape(arr.shape[0])
        )

    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)

    return arr


# -- row/value normalization ----------------------------------------------------


def _default_table_column_values(
    name: str,
    tform: str,
    num_rows: int,
    tnull: Any = None,
) -> Any:
    import numpy as np

    _ensure_dtype_maps()

    is_vla, code, repeat = _parse_tform(tform)
    if repeat <= 0:
        repeat = 1

    if is_vla:
        dtype = _VLA_DTYPE_MAP.get(code, np.float32)
        return [np.asarray([], dtype=dtype) for _ in range(num_rows)]

    if code == "A":
        return [""] * num_rows

    if code in _COMPLEX_TFORM_CODES:
        dtype = _COMPLEX_DTYPE_MAP[code]
        shape = (num_rows,) if repeat == 1 else (num_rows, repeat)
        return np.zeros(shape, dtype=dtype)

    dtype_map = {
        "L": np.bool_,
        "X": np.uint8,
        "B": np.uint8,
        "I": np.int16,
        "J": np.int32,
        "K": np.int64,
        "E": np.float32,
        "D": np.float64,
    }
    dtype = dtype_map.get(code, np.float32)
    shape = (num_rows,) if repeat == 1 else (num_rows, repeat)

    if tnull is not None and code not in {"A", "C", "M"}:
        try:
            fill: Any = np.asarray(tnull, dtype=dtype).item()
            return np.full(shape, fill, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"cannot coerce TNULL={tnull!r} for column {name!r} dtype {dtype}"
            ) from exc
    return np.zeros(shape, dtype=dtype)


def _normalize_mutation_rows(
    rows: dict[str, Any],
    columns: list[str],
    tform_map: dict[str, str],
    tnull_map: dict[str, Any],
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], int]:
    rows_by_name = {str(k): v for k, v in rows.items()}
    input_columns = set(rows_by_name)
    expected_columns = set(columns)
    extra = sorted(input_columns - expected_columns)
    if extra:
        raise ValueError(f"Unknown columns for table mutation: extra={extra}")
    if not input_columns:
        raise ValueError("rows must include at least one column")

    if not allow_partial and input_columns != expected_columns:
        missing = sorted(expected_columns - input_columns)
        raise ValueError(
            "Mutation payload must provide every table column; "
            f"missing={missing}, extra={extra}"
        )

    string_widths: dict[str, int] = {}
    vla_codes: dict[str, str] = {}
    complex_codes: dict[str, str] = {}
    for col_name in columns:
        tform = tform_map.get(col_name, "")
        if not tform:
            continue
        is_vla, code, repeat = _parse_tform(tform)
        if is_vla:
            vla_codes[col_name] = code
        elif code in _COMPLEX_TFORM_CODES:
            complex_codes[col_name] = code
        elif code == "A":
            string_widths[col_name] = repeat

    normalized: dict[str, Any] = {}
    expected_rows: Optional[int] = None
    deferred_defaults: list[str] = []

    for col_name in columns:
        if col_name not in rows_by_name:
            deferred_defaults.append(col_name)
            continue

        value = rows_by_name[col_name]
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
            normalized[col_name] = values
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
        raise ValueError("Could not infer row count from mutation payload")
    if expected_rows <= 0:
        return {}, 0

    for col_name in deferred_defaults:
        default_value = _default_table_column_values(
            col_name,
            tform_map.get(col_name, ""),
            expected_rows,
            tnull=tnull_map.get(col_name),
        )
        if col_name in vla_codes:
            normalized[col_name] = _coerce_table_vla_values(
                col_name,
                default_value,
                vla_codes[col_name],
                expected_rows=expected_rows,
            )
        elif col_name in string_widths:
            normalized[col_name] = _coerce_table_string_values(
                col_name, default_value, expected_rows=expected_rows
            )
        elif col_name in complex_codes:
            normalized[col_name] = _coerce_table_complex_values(
                col_name,
                default_value,
                complex_codes[col_name],
                expected_rows=expected_rows,
                allow_2d=True,
            )
        else:
            normalized[col_name] = _coerce_table_column_array(
                col_name, default_value, expected_rows=expected_rows, allow_2d=True
            )

    return normalized, expected_rows


def _read_table_for_rewrite(path: str, hdu: int, columns: list[str]) -> dict[str, Any]:
    import numpy as np
    import torchfits

    with torchfits.open(path) as hdul:
        table_hdu = hdul[hdu]
        schema = table_hdu.schema if hasattr(table_hdu, "schema") else {}
        string_cols = set(schema.get("string_columns", []))
        vla_cols = set(schema.get("vla_columns", []))

        out: dict[str, Any] = {}
        for name in columns:
            if name in vla_cols:
                values = table_hdu.get_vla_column(name)  # type: ignore[union-attr]
                converted = []
                for item in values:
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        converted.append(np.ascontiguousarray(t.numpy()))
                    else:
                        converted.append(np.ascontiguousarray(np.asarray(item)))
                out[name] = converted
            elif name in string_cols:
                out[name] = table_hdu.get_string_column(name)  # type: ignore[union-attr]
            else:
                value = table_hdu[name]  # type: ignore[index]
                if isinstance(value, torch.Tensor):
                    t = value.detach()
                    if t.device.type != "cpu":
                        t = t.cpu()
                    if not t.is_contiguous():
                        t = t.contiguous()
                    out[name] = np.ascontiguousarray(t.numpy())
                else:
                    out[name] = np.ascontiguousarray(np.asarray(value))
        return out


def _merge_insert_column(existing: Any, inserted: Any, row: int) -> Any:
    import numpy as np

    if isinstance(existing, list):
        if isinstance(inserted, list):
            values = inserted
        elif isinstance(inserted, np.ndarray):
            values = inserted.tolist()
        else:
            values = [inserted]
        return list(existing[:row]) + values + list(existing[row:])

    old_arr = np.asarray(existing)
    new_arr = np.asarray(inserted, dtype=old_arr.dtype)
    if old_arr.ndim == 2 and new_arr.ndim == 1 and old_arr.shape[1] == 1:
        new_arr = new_arr.reshape(-1, 1)
    if old_arr.ndim == 1 and new_arr.ndim == 2 and new_arr.shape[1] == 1:
        new_arr = new_arr.reshape(-1)
    out = np.concatenate([old_arr[:row], new_arr, old_arr[row:]], axis=0)
    return np.ascontiguousarray(out)


def _delete_column_rows(existing: Any, start0: int, num_rows: int) -> Any:
    import numpy as np

    if isinstance(existing, list):
        return list(existing[:start0]) + list(existing[start0 + num_rows :])

    arr = np.asarray(existing)
    out = np.concatenate([arr[:start0], arr[start0 + num_rows :]], axis=0)
    return np.ascontiguousarray(out)


def _coerce_rows_from_arrow(rows: Any) -> Any:
    try:
        import pyarrow as pa
    except ImportError:
        return rows

    if isinstance(rows, pa.RecordBatch):
        rows = pa.Table.from_batches([rows])
    if isinstance(rows, pa.Table):
        out: dict[str, Any] = {}
        for field in rows.schema:
            name = field.name
            out[name] = _arrow_column_to_python(pa, rows[name], name)
        return out
    return rows


# -- column value coercion helpers ----------------------------------------------


def _coerce_table_column_array(
    name: str,
    value: Any,
    *,
    expected_rows: Optional[int] = None,
    allow_2d: bool = True,
) -> "np.ndarray":
    import numpy as np

    _ensure_dtype_maps()
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        if tensor.dim() == 2 and not allow_2d:
            raise ValueError(f"Column '{name}' must be 1D for this operation")
        if tensor.dim() > 2:
            raise ValueError(f"Column '{name}' must be 1D or 2D, got {tensor.dim()}D")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        arr = tensor.numpy()
    else:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)

    if arr.dtype == np.object_:
        raise TypeError(f"Column '{name}' with object dtype is not supported")
    if arr.dtype.kind in {"U", "S"}:
        raise TypeError(f"Column '{name}' string dtype is not supported")
    if arr.dtype.kind == "c":
        raise TypeError(f"Column '{name}' complex dtype is not supported")

    if arr.ndim == 2 and not allow_2d:
        raise ValueError(f"Column '{name}' must be 1D for this operation")
    if arr.ndim > 2:
        raise ValueError(f"Column '{name}' must be 1D or 2D, got {arr.ndim}D")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ValueError(
            f"Column '{name}' has {arr.shape[0]} rows, expected {expected_rows}"
        )

    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"Column '{name}' dtype {arr.dtype} is not supported")
    return np.ascontiguousarray(arr)


def _coerce_table_string_values(
    name: str,
    value: Any,
    *,
    expected_rows: Optional[int] = None,
) -> list[str]:
    import numpy as np

    if isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in {"U", "S"}:
            raise TypeError(f"Column '{name}' string dtype is not supported")
        values = value.astype(str).tolist()
    else:
        values = [value]

    out: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("ascii", errors="ignore"))
        elif isinstance(item, np.bytes_):
            out.append(bytes(item).decode("ascii", errors="ignore"))
        else:
            out.append(str(item))

    if expected_rows is not None and len(out) != expected_rows:
        raise ValueError(
            f"Column '{name}' has {len(out)} rows, expected {expected_rows}"
        )
    return out


def _coerce_table_vla_values(
    name: str,
    value: Any,
    base_code: str,
    *,
    expected_rows: Optional[int] = None,
) -> "list[np.ndarray]":
    import numpy as np

    _ensure_dtype_maps()
    code = base_code.upper()
    if code not in _VLA_DTYPE_MAP:
        raise TypeError(f"Column '{name}' VLA code '{code}' is not supported")
    dtype = _VLA_DTYPE_MAP[code]

    if isinstance(value, np.ndarray) and value.dtype == np.object_:
        items = list(value)
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise TypeError(f"Column '{name}' VLA values must be a list/tuple of arrays")

    if expected_rows is not None and len(items) != expected_rows:
        raise ValueError(
            f"Column '{name}' has {len(items)} rows, expected {expected_rows}"
        )

    out: list[np.ndarray] = []
    for item in items:
        if item is None:
            arr = np.asarray([], dtype=dtype)
        elif isinstance(item, torch.Tensor):
            t = item.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            if t.dim() == 0:
                t = t.reshape(1)
            arr = t.numpy().astype(dtype, copy=False)
        else:
            arr = np.asarray(item, dtype=dtype)

        if arr.ndim > 1:
            arr = arr.reshape(-1)
        out.append(np.ascontiguousarray(arr))

    return out


def _coerce_table_complex_values(
    name: str,
    value: Any,
    code: str,
    *,
    expected_rows: Optional[int] = None,
    allow_2d: bool = True,
) -> "np.ndarray":
    import numpy as np

    _ensure_dtype_maps()
    base = code.upper()
    if base not in _COMPLEX_TFORM_CODES:
        raise TypeError(f"Column '{name}' complex code '{base}' is not supported")
    dtype = _COMPLEX_DTYPE_MAP[base]

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_complex():
            raise TypeError(f"Column '{name}' must be complex")
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        if tensor.dim() == 2 and not allow_2d:
            raise ValueError(f"Column '{name}' must be 1D for this operation")
        if tensor.dim() > 2:
            raise ValueError(f"Column '{name}' must be 1D or 2D, got {tensor.dim()}D")
        arr = tensor.numpy().astype(dtype, copy=False)
    else:
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)

    if arr.ndim == 2 and not allow_2d:
        raise ValueError(f"Column '{name}' must be 1D for this operation")
    if arr.ndim > 2:
        raise ValueError(f"Column '{name}' must be 1D or 2D, got {arr.ndim}D")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ValueError(
            f"Column '{name}' has {arr.shape[0]} rows, expected {expected_rows}"
        )
    return np.ascontiguousarray(arr)


# -- column format inference (from write section) --------------------------------


def _infer_column_format_for_insert(name: str, values: Any) -> str:
    import numpy as np

    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        arr = tensor.numpy()
        return _infer_fits_format(arr)

    if isinstance(values, np.ndarray):
        arr = values
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return _infer_fits_format(arr)

    if isinstance(values, (list, tuple)):
        items = list(values)
        if not items:
            raise ValueError(
                f"Cannot infer FITS format for empty column '{name}'; provide format=..."
            )
        if all(
            isinstance(item, (str, bytes, np.str_, np.bytes_)) or item is None
            for item in items
        ):
            max_len = 1
            for item in items:
                if item is None:
                    continue
                if isinstance(item, bytes):
                    max_len = max(max_len, len(item))
                else:
                    max_len = max(max_len, len(str(item)))
            return f"{max_len}A"

        if any(
            isinstance(item, (list, tuple, np.ndarray, torch.Tensor)) for item in items
        ):
            sample = None
            for item in items:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    t = item.detach()
                    if t.device.type != "cpu":
                        t = t.cpu()
                    if t.numel() == 0:
                        continue
                    sample = t.numpy()
                    break
                arr_item = np.asarray(item)
                if arr_item.size == 0:
                    continue
                sample = arr_item
                break
            if sample is None:
                raise ValueError(
                    f"Cannot infer VLA base dtype for column '{name}'; provide format=..."
                )
            code = _infer_fits_scalar_code(np.asarray(sample).reshape(-1))
            return f"1P{code}"

        arr = np.asarray(items)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return _infer_fits_format(arr)

    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return _infer_fits_format(arr)


def _normalize_column_values_for_format(
    name: str,
    values: Any,
    fmt: str,
    expected_rows: int,
) -> Any:
    import numpy as np

    is_vla, code, repeat = _parse_tform(fmt)
    if repeat <= 0:
        repeat = 1

    if is_vla:
        return _coerce_table_vla_values(name, values, code, expected_rows=expected_rows)

    if code == "A":
        return _coerce_table_string_values(name, values, expected_rows=expected_rows)

    if code in _COMPLEX_TFORM_CODES:
        arr = _coerce_table_complex_values(
            name, values, code, expected_rows=expected_rows, allow_2d=True
        )
    else:
        arr = _coerce_table_column_array(
            name, values, expected_rows=expected_rows, allow_2d=True
        )

    if repeat > 1 and arr.ndim == 1:
        if expected_rows == 1 and arr.size == repeat:
            arr = arr.reshape(1, repeat)
        elif expected_rows > 0 and arr.size == expected_rows * repeat:
            arr = arr.reshape(expected_rows, repeat)

    arr = _prepare_array_for_column(np.ascontiguousarray(arr), fmt)
    if (
        isinstance(arr, np.ndarray)
        and arr.ndim > 0
        and int(arr.shape[0]) != expected_rows
    ):
        raise ValueError(
            f"Column '{name}' has {arr.shape[0]} rows, expected {expected_rows}"
        )
    return arr


# -- public column mutation API --------------------------------------------------


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
                if mmap is True:
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
