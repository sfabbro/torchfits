"""Table mutation coercion helpers and internal normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

from .._io_engine.caches import invalidate_path_caches as _invalidate_path_caches
from .._table.utils import (
    _arrow_column_to_python,
    _parse_tform,
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
