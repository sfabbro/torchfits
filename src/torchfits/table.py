"""Arrow-native table I/O helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import itertools
import os
import logging
from typing import Any, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from . import fits_schema
from ._io_engine.caches import invalidate_path_caches as _invalidate_path_caches
from ._table.cache import acquire_cpp_handle as _acquire_cpp_handle
from ._table.cache import acquire_cpp_reader as _acquire_cpp_reader
from ._where import (
    parse_where_expression,
    where_columns_from_ast,
)
from ._table_engine import (
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_numpy_for_where,
    validate_table_backend,
)


logger = logging.getLogger(__name__)


def _normalize_cpp_table_data(data):
    from torchfits.io import _normalize_cpp_table_data as normalize

    return normalize(data)


def _write_header_cards_if_supported(path: str, hdu: int, hdr) -> None:
    from torchfits.io import _write_header_cards_if_supported as write_hdr

    write_hdr(path, hdu, hdr)


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

_VLA_DTYPE_MAP: dict[str, Any] = {}
_COMPLEX_DTYPE_MAP: dict[str, Any] = {}
# FITS binary-table TFORM codes for complex columns (membership checks use this so a
# corrupted ``_COMPLEX_DTYPE_MAP`` cannot turn ``x in map`` into a brittle set-only path).
_COMPLEX_TFORM_CODES: frozenset[str] = frozenset({"C", "M"})


def _parse_tform(tform: str) -> tuple[bool, str, int]:
    info = fits_schema.parse_tform(tform)
    return info.vla, info.code or "", info.repeat


def _column_tnull_map(header_map: dict[str, Any]) -> dict[str, Any]:
    return fits_schema.column_tnull_map(header_map)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc  # noqa: F401 # preload compute module to avoid slow first-call overhead
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for torchfits.table APIs. Install pyarrow to use Arrow-native tables."
        ) from exc
    return pa


def _arrow_column_to_python(pa, column, name: str) -> Any:
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
        out: list[np.ndarray] = []
        for item in pylist_values:
            if item is None:
                out.append([])
            else:
                out.append(np.asarray(item))
        return out

    return column.to_numpy(zero_copy_only=False)


def _default_table_column_values(
    name: str,
    tform: str,
    num_rows: int,
    tnull: Any = None,
):
    import numpy as np

    global _VLA_DTYPE_MAP, _COMPLEX_DTYPE_MAP
    if not _VLA_DTYPE_MAP:
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
    if not _COMPLEX_DTYPE_MAP:
        _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}

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
        except Exception:
            pass
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
            values = _coerce_table_string_values(
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
                values = table_hdu.get_vla_column(name)
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
                out[name] = table_hdu.get_string_column(name)
            else:
                value = table_hdu[name]
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


def _normalize_row_slice(
    row_slice: Optional[slice | tuple[int, int]],
) -> tuple[int, int]:
    """Convert python-style row slice to FITS 1-based (start_row, num_rows)."""
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

    start_row = start + 1
    if stop is None:
        return start_row, -1
    if stop < start:
        return start_row, 0
    return start_row, stop - start


def _compile_where_to_simple_predicates(
    where: str,
) -> Optional[list[tuple[str, str, Any]]]:
    """
    Compile a restricted where expression into C++ predicate tuples.

    Returns `None` when expression cannot be represented as a pure conjunction
    of simple binary comparisons.
    """
    try:
        ast = parse_where_expression(where)
    except Exception:
        return None

    predicates: list[tuple[str, str, Any]] = []

    def _visit(node) -> bool:
        kind = node[0]
        if kind == "cmp":
            _, col, op, literal = node
            if op not in {"==", "!=", ">", ">=", "<", "<="}:
                return False
            if literal is None:
                return False
            predicates.append((col, op, literal))
            return True
        if kind == "between":
            _, col, low, high, negate = node
            if bool(negate) or low is None or high is None:
                return False
            predicates.append((col, ">=", low))
            predicates.append((col, "<=", high))
            return True
        if kind == "and":
            return _visit(node[1]) and _visit(node[2])
        return False

    if not _visit(ast):
        return None
    return predicates


def _where_mask_for_table(table, where: str, parsed_ast=None) -> "np.ndarray":
    pa = _require_pyarrow()
    import pyarrow.compute as pc

    ast = parsed_ast if parsed_ast is not None else parse_where_expression(where)

    def _get_predicate_column(column_name: str):
        if column_name not in table.column_names:
            raise ValueError(f"where references unknown column '{column_name}'")

        column = table[column_name]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            raise ValueError(f"where does not support list/VLA column '{column_name}'")
        if pa.types.is_fixed_size_list(column.type):
            raise ValueError(
                f"where does not support fixed-size vector column '{column_name}'"
            )
        return column

    def _cmp_mask(column_name: str, op: str, literal: Any):
        column = _get_predicate_column(column_name)

        if literal is None:
            if op == "==":
                return pc.is_null(column)
            if op == "!=":
                return pc.invert(pc.is_null(column))
            raise ValueError("where comparisons with null only support == and !=")

        scalar = pa.scalar(literal)
        if op == "==":
            return pc.equal(column, scalar)
        if op == "!=":
            return pc.not_equal(column, scalar)
        if op == ">":
            return pc.greater(column, scalar)
        if op == ">=":
            return pc.greater_equal(column, scalar)
        if op == "<":
            return pc.less(column, scalar)
        if op == "<=":
            return pc.less_equal(column, scalar)
        raise ValueError(f"Unsupported where operator '{op}'")

    def _in_mask(column_name: str, literals: list[Any], negate: bool):
        column = _get_predicate_column(column_name)
        non_null = [v for v in literals if v is not None]
        has_null = any(v is None for v in literals)

        if non_null:
            value_set = _pa_array(pa, non_null)
            mask = pc.is_in(column, value_set=value_set)
        else:
            mask = _pa_array(pa, [False] * int(len(column)))

        if has_null:
            mask = pc.or_(pc.fill_null(mask, False), pc.is_null(column))
        mask = pc.fill_null(mask, False)

        if negate:
            return pc.invert(mask)
        return mask

    def _between_mask(column_name: str, low: Any, high: Any, negate: bool):
        column = _get_predicate_column(column_name)
        if low is None or high is None:
            raise ValueError("where BETWEEN does not support NULL bounds")
        low_s = pa.scalar(low)
        high_s = pa.scalar(high)
        ge = pc.greater_equal(column, low_s)
        le = pc.less_equal(column, high_s)
        mask = pc.and_(pc.fill_null(ge, False), pc.fill_null(le, False))
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _isnull_mask(column_name: str, negate: bool):
        column = _get_predicate_column(column_name)
        mask = pc.is_null(column)
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _eval(node):
        kind = node[0]
        if kind == "cmp":
            return pc.fill_null(_cmp_mask(node[1], node[2], node[3]), False)
        if kind == "in":
            return pc.fill_null(_in_mask(node[1], node[2], bool(node[3])), False)
        if kind == "between":
            return pc.fill_null(
                _between_mask(node[1], node[2], node[3], bool(node[4])), False
            )
        if kind == "isnull":
            return pc.fill_null(_isnull_mask(node[1], bool(node[2])), False)
        if kind == "and":
            left = pc.fill_null(_eval(node[1]), False)
            right = pc.fill_null(_eval(node[2]), False)
            return pc.and_(left, right)
        if kind == "or":
            left = pc.fill_null(_eval(node[1]), False)
            right = pc.fill_null(_eval(node[2]), False)
            return pc.or_(left, right)
        if kind == "not":
            child = pc.fill_null(_eval(node[1]), False)
            return pc.invert(child)
        raise ValueError("Invalid where AST")

    return pc.fill_null(_eval(ast), False)


def _row_slice_from_start_num(start_row: int, num_rows: int) -> Optional[slice]:
    if start_row == 1 and num_rows == -1:
        return None
    start0 = start_row - 1
    if num_rows == -1:
        return slice(start0, None)
    return slice(start0, start0 + num_rows)


def _resolve_rows_from_where_cpp(
    path: str,
    hdu: int,
    where: str,
    start_row: int,
    num_rows: int,
    mmap: bool,
    apply_fits_nulls: bool,
) -> Optional[list[int]]:
    where_ast = parse_where_expression(where)
    where_columns = where_columns_from_ast(where_ast)
    predicate_table = _read_cpp_numpy_table(
        path=path,
        hdu=hdu,
        columns=where_columns,
        row_slice=_row_slice_from_start_num(start_row, num_rows),
        rows=None,
        where=None,
        mmap=mmap,
        decode_bytes=True,
        encoding="utf-8",
        strip=True,
        include_fits_metadata=False,
        apply_fits_nulls=apply_fits_nulls,
    )
    if predicate_table is None:
        return None
    if predicate_table.num_rows == 0:
        return []

    import pyarrow.compute as pc

    mask = _where_mask_for_table(predicate_table, where, parsed_ast=where_ast)
    if len(mask) == 0 or pc.sum(mask).as_py() == 0:
        return []

    base_row0 = start_row - 1
    # Use Arrow to find indices of True values, then convert to numpy for offset addition
    selected = pc.indices_nonzero(mask).to_numpy()
    if selected.size == 0:
        return []
    # Vectorized addition and then conversion to list is faster than list comprehension
    # base_row0 is start_row - 1, which is 0-based.
    return (selected + base_row0).tolist()


def _build_fits_metadata(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]] = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    import torchfits

    header = torchfits.get_header(path, hdu)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {
        "fits_hdu": str(hdu),
    }

    try:
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        tf_count = 0

    for i in range(1, tf_count + 1):
        si = str(i)
        name = header.get("TTYPE" + si)
        if not isinstance(name, str) or not name:
            continue
        if selected_columns is not None and name not in selected_columns:
            continue

        # Optimize by unrolling the loop and avoiding inner string operations
        entry: dict[str, str] = {}

        v = header.get("TFORM" + si)
        if v is not None:
            entry["fits_tform"] = str(v)

        v = header.get("TUNIT" + si)
        if v is not None:
            entry["fits_tunit"] = str(v)

        v = header.get("TDIM" + si)
        if v is not None:
            entry["fits_tdim"] = str(v)

        v = header.get("TNULL" + si)
        if v is not None:
            entry["fits_tnull"] = str(v)

        v = header.get("TSCAL" + si)
        if v is not None:
            entry["fits_tscal"] = str(v)

        v = header.get("TZERO" + si)
        if v is not None:
            entry["fits_tzero"] = str(v)

        if entry:
            field_meta[name] = entry

    return field_meta, table_meta


def _column_tform_code_and_repeat(tform: Any) -> tuple[str, int] | None:
    return fits_schema.tform_code_and_repeat(tform)


def _fits_tform_is_bit(tform: Any) -> bool:
    return fits_schema.tform_is_bit(tform)


def _column_tforms_for_decode(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
) -> dict[str, str]:
    """Map column name -> FITS TFORM for uint8-matrix decode (BIT vs character)."""
    out: dict[str, str] = {}
    try:
        fm, _ = _build_fits_metadata(path, hdu, selected_columns)
        for col, meta in fm.items():
            tf = meta.get("fits_tform")
            if tf:
                out[col] = tf
    except Exception:
        pass
    return out


def _unsigned_column_dtypes(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
) -> dict[str, str]:
    """Map standard unsigned FITS table conventions to NumPy dtypes."""
    try:
        fm, _ = _build_fits_metadata(path, hdu, selected_columns)
    except Exception:
        return {}
    targets = {
        ("I", 32768.0): "uint16",
        ("J", 2147483648.0): "uint32",
    }
    out: dict[str, str] = {}
    for col, meta in fm.items():
        parsed = _column_tform_code_and_repeat(meta.get("fits_tform"))
        if parsed is None:
            continue
        code, _repeat = parsed
        try:
            tscal = float(meta.get("fits_tscal", "1"))
            tzero = float(meta.get("fits_tzero", "0"))
        except Exception:
            continue
        target = targets.get((code, tzero))
        if target is not None and tscal == 1.0:
            out[col] = target
    return out


def _can_use_mmap_row_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
) -> bool:
    """
    Heuristic for full-table Arrow reads:
    prefer mmap row path only for simple scalar numeric/logical columns.
    """
    import torchfits

    try:
        header = torchfits.get_header(path, hdu)
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        return False
    if tf_count <= 0:
        return False

    selected = set(selected_columns) if selected_columns else None
    supported_codes = {"L", "B", "I", "J", "K", "E", "D"}
    any_selected = False

    for i in range(1, tf_count + 1):
        name = header.get(f"TTYPE{i}")
        if not isinstance(name, str) or not name:
            continue
        if selected is not None and name not in selected:
            continue
        any_selected = True

        if header.get(f"TSCAL{i}") is not None or header.get(f"TZERO{i}") is not None:
            return False

        parsed = _column_tform_code_and_repeat(header.get(f"TFORM{i}"))
        if parsed is None:
            return False
        code, repeat = parsed
        if code not in supported_codes:
            return False
        # Keep mmap fast-path focused on scalar columns. Vector columns are often
        # faster on the non-mmap C++ table path.
        if repeat != 1:
            return False

    return any_selected


def _can_use_torch_table_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
) -> bool:
    """
    Heuristic for direct C++ torch-table full reads feeding Arrow conversion.

    This path is used when Arrow metadata/null semantics are not requested and
    we only need numeric/logical columns represented as dense tensors.
    """
    import torchfits

    try:
        header = torchfits.get_header(path, hdu)
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        return False
    if tf_count <= 0:
        return False

    selected = set(selected_columns) if selected_columns else None
    supported_codes = {"L", "B", "I", "J", "K", "E", "D"}
    any_selected = False

    for i in range(1, tf_count + 1):
        name = header.get(f"TTYPE{i}")
        if not isinstance(name, str) or not name:
            continue
        if selected is not None and name not in selected:
            continue
        any_selected = True

        parsed = _column_tform_code_and_repeat(header.get(f"TFORM{i}"))
        if parsed is None:
            return False
        code, repeat = parsed
        if code not in supported_codes:
            return False
        if repeat != 1:
            return False

    return any_selected


def _iter_chunks_cpp_numpy(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    batch_size: int,
    mmap: bool,
):
    import torchfits
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):
        return None

    header = torchfits.get_header(path, hdu)
    total_rows = header.get("NAXIS2", 0)
    try:
        total_rows = (
            int(float(total_rows)) if isinstance(total_rows, str) else int(total_rows)
        )
    except Exception:
        total_rows = 0
    if total_rows <= 0:
        return iter(())

    end_row = (
        total_rows if num_rows == -1 else min(total_rows, start_row + num_rows - 1)
    )
    col_list = columns if columns else []

    def _generator():
        can_mmap_rows = mmap and hasattr(cpp, "read_fits_table_rows")
        if can_mmap_rows:
            # Avoid exception-driven probing for BIT/VLA/scaled/vector tables.
            # This is a cheap header-based filter and helps make mmap=True safe
            # even when users pass `columns=[...]`.
            can_mmap_rows = _can_use_mmap_row_path_for_full_read(path, hdu, columns)
        file_handle = None
        try:
            row = start_row
            while row <= end_row:
                size = min(batch_size, end_row - row + 1)
                if can_mmap_rows:
                    try:
                        yield cpp.read_fits_table_rows(
                            path, hdu, col_list, row, size, True
                        )
                        row += size
                        continue
                    except Exception:
                        can_mmap_rows = False

                if file_handle is None:
                    file_handle = cpp.open_fits_file(path, "r")
                yield cpp.read_fits_table_rows_numpy_from_handle(
                    file_handle, hdu, col_list, row, size
                )
                row += size
        finally:
            if file_handle is not None:
                file_handle.close()

    return _generator()


def scan(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
) -> Iterator[Any]:
    """
    Stream a FITS table as Arrow record batches.

    This is out-of-core friendly: each yielded batch is independently materialized.
    FITS character columns are decoded to Python strings by default (`decode_bytes=True`).
    """
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    validate_table_backend(backend)

    if where is not None:
        table = read(
            path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            where=where,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )
        for batch in table.to_batches(max_chunksize=batch_size):
            yield batch
        return

    import torchfits

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        return
    selected = set(columns) if columns else None
    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected) if decode_bytes else None
    )
    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(path, hdu, selected)
        except Exception:
            field_meta, table_meta = {}, {}
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    chunk_iter = None
    if backend in {"auto", "cpp_numpy"}:
        chunk_iter = _iter_chunks_cpp_numpy(
            path, hdu, columns, start_row, num_rows, batch_size, mmap
        )
    if chunk_iter is None or backend == "torch":
        chunk_iter = torchfits.stream_table(
            path,
            hdu=hdu,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            chunk_rows=batch_size,
            mmap=mmap,
        )

    for chunk in chunk_iter:
        yield _chunk_to_record_batch(
            chunk,
            decode_bytes,
            encoding,
            strip,
            field_meta=field_meta if include_fits_metadata else None,
            table_meta=table_meta if include_fits_metadata else None,
            preferred_order=preferred_order,
            null_meta=field_meta,
            apply_fits_nulls=apply_fits_nulls,
            column_tforms=col_tforms,
            unsigned_dtypes=unsigned_dtypes,
        )


def _filter_table_with_where(pa, table: Any, where: str) -> Any:
    mask = _where_mask_for_table(table, where)
    if len(mask) == 0 or pa.compute.sum(mask).as_py() == 0:
        return table.slice(0, 0)
    return table.filter(mask)


def _read_table_from_scan_batches(
    *,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    pa = _require_pyarrow()
    batches = list(
        scan(
            path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            batch_size=batch_size,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )
    )
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def _read_table_unfiltered(
    *,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    if backend in {"auto", "cpp_numpy"}:
        single = _read_cpp_numpy_table(
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=None,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
        )
        if single is not None:
            return single
    return _read_table_from_scan_batches(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=backend,
    )


def _try_cpp_where_pushdown(
    *,
    pa,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
) -> Any | None:
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_filtered"):
        return None
    filters = _compile_where_to_simple_predicates(where)
    if filters is None:
        return None
    try:
        target_cols = columns
        if target_cols is None:
            target_cols = list(schema(path, hdu=hdu, backend="cpp_numpy").names)

        data_dict = cpp.read_fits_table_filtered(path, hdu, target_cols, filters)
        pushdown_tforms = (
            _column_tforms_for_decode(path, hdu, set(target_cols))
            if decode_bytes
            else None
        )
        arrays = []
        names_out = []
        for name in target_cols:
            if name not in data_dict:
                continue
            val = data_dict[name]
            if isinstance(val, torch.Tensor):
                if val.device.type != "cpu":
                    val = val.cpu()
                if not val.is_contiguous():
                    val = val.contiguous()
                arr = _numpy_to_arrow_array(
                    pa,
                    val.numpy(),
                    decode_bytes,
                    encoding,
                    strip,
                    fits_tform=pushdown_tforms.get(name) if pushdown_tforms else None,
                )
                arrays.append(arr)
                names_out.append(name)

        if not arrays:
            return pa.table({})
        return pa.Table.from_arrays(arrays, names=names_out)
    except Exception:
        return None


def _read_table_with_where(
    *,
    pa,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    where: str,
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    import torchfits

    header_ok = False
    hdr: Mapping[str, Any] = {}
    n_rows = 0
    try:
        hdr = torchfits.get_header(path, hdu)
        n_rows = int(hdr.get("NAXIS2", 0))
        header_ok = True
    except Exception:
        n_rows = 0

    plan = choose_where_read_plan(
        header=hdr,
        header_ok=header_ok,
        columns=columns,
        backend=backend,
        n_rows=n_rows,
    )

    if plan.strategy == WhereStrategy.CPP_PUSHDOWN:
        pushed = _try_cpp_where_pushdown(
            pa=pa,
            path=path,
            hdu=hdu,
            columns=columns,
            where=where,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
        )
        if pushed is not None:
            return pushed

    base = _read_table_unfiltered(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        rows=rows,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=plan.unfiltered_backend,
    )
    return _filter_table_with_where(pa, base, where)


def read(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    rows: Optional[list[int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    """Read a FITS table into an Arrow Table.

    FITS character columns are decoded to Python strings by default (`decode_bytes=True`).

    With ``where=``, the reader picks between loading the projected table and filtering
    in Arrow vs C++ predicate pushdown (see ``TORCHFITS_TABLE_SCANNER_THRESHOLD`` and
    ``backend=`` in :doc:`api`). Valid backends: see ``TABLE_BACKENDS``.
    """
    validate_table_backend(backend)
    pa = _require_pyarrow()
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if backend in {"auto", "cpp_numpy"} and not should_skip_cpp_numpy_for_where(
        backend, where
    ):
        single = _read_cpp_numpy_table(
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=where,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
        )
        if single is not None:
            return single

    if where is not None:
        return _read_table_with_where(
            pa=pa,
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=where,
            batch_size=batch_size,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )

    return _read_table_from_scan_batches(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=backend,
    )


def schema(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = False,
    backend: str = "auto",
):
    """Fetch Arrow schema for a FITS table with minimal read."""
    pa = _require_pyarrow()
    validate_table_backend(backend)
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]
    scan_backend = backend
    iterator = scan(
        path,
        hdu=hdu,
        columns=columns,
        where=where,
        batch_size=1,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=scan_backend,
    )
    first = next(iterator, None)
    if first is None:
        return pa.schema([])
    return first.schema


def _read_cpp_numpy_table(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    where: Optional[str],
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
):
    import numpy as np
    import torchfits._C as cpp

    has_numpy_row_api = hasattr(
        cpp, "read_fits_table_rows_numpy_from_handle"
    ) or hasattr(cpp, "read_fits_table_rows_numpy")
    has_torch_table_api = hasattr(cpp, "read_fits_table")
    if not has_numpy_row_api and not has_torch_table_api:
        return None

    if rows is not None and row_slice is not None:
        raise ValueError("Only one of rows or row_slice may be provided")

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        pa = _require_pyarrow()
        return pa.table({})

    if where is not None:
        where_rows = _resolve_rows_from_where_cpp(
            path=path,
            hdu=hdu,
            where=where,
            start_row=start_row,
            num_rows=num_rows,
            mmap=mmap,
            apply_fits_nulls=apply_fits_nulls,
        )
        if where_rows is None:
            return None
        if rows is not None:
            where_set = set(where_rows)
            rows = [int(r) for r in rows if int(r) in where_set]
        else:
            rows = where_rows
        start_row = 1
        num_rows = -1

    selected = set(columns) if columns else None
    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected) if decode_bytes else None
    )
    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(path, hdu, selected)
        except Exception:
            pass
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    col_list = columns if columns else []

    def _read_ranges_as_chunk(reader, ranges: list[tuple[int, int]]):
        # ranges are (start0, length) in 0-based row indices
        out_sorted: dict[str, Any] = {}
        n_total = sum(length for _, length in ranges)

        cursor = 0
        for start0, length in ranges:
            seg = reader.read_rows_numpy(col_list, start0 + 1, length)
            if not seg:
                cursor += length
                continue
            for name, value in seg.items():
                buf = out_sorted.get(name)
                if buf is None:
                    if isinstance(value, np.ndarray):
                        buf = np.empty((n_total,) + value.shape[1:], dtype=value.dtype)
                    elif isinstance(value, list):
                        buf = [None] * n_total
                    elif _is_vla_tuple(value):
                        buf = [None] * n_total
                    else:
                        buf = [None] * n_total
                    out_sorted[name] = buf

                if isinstance(value, np.ndarray):
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

    chunk = None
    prefer_torch_full_path = (
        start_row == 1
        and num_rows == -1
        and not decode_bytes
        and not include_fits_metadata
        and not apply_fits_nulls
        and _can_use_torch_table_path_for_full_read(path, hdu, columns)
    )
    if prefer_torch_full_path and has_torch_table_api:
        # For scalar numeric/logical columns, mmap tends to win. For vector
        # columns, the non-mmap torch path is often better.
        if mmap and _can_use_mmap_row_path_for_full_read(path, hdu, columns):
            try:
                chunk = cpp.read_fits_table(path, hdu, col_list, True)
            except Exception:
                chunk = None
        if chunk is None:
            try:
                if not col_list and hasattr(cpp, "read_fits_table_from_handle"):
                    file_handle = _acquire_cpp_handle(path, cpp)
                    chunk = cpp.read_fits_table_from_handle(file_handle, hdu)
                else:
                    chunk = cpp.read_fits_table(path, hdu, col_list, False)
            except Exception:
                chunk = None

    # rows=[...] selection: read minimal ranges and stitch.
    if chunk is None and rows is not None:
        if not hasattr(cpp, "TableReader") or not hasattr(
            cpp.TableReader, "read_rows_numpy"
        ):
            return None
        rows_arr = np.asarray(rows, dtype=np.int64)
        if rows_arr.size == 0:
            pa = _require_pyarrow()
            return pa.table({})
        if np.any(rows_arr < 0):
            raise ValueError("rows must be non-negative (0-based)")

        order = np.argsort(rows_arr, kind="stable")
        sorted_rows = rows_arr[order]

        if len(sorted_rows) == 0:
            ranges: list[tuple[int, int]] = []
        else:
            diffs = np.diff(sorted_rows)
            breaks = np.nonzero(diffs != 1)[0]
            start_indices = np.insert(breaks + 1, 0, 0)
            end_indices = np.append(breaks, len(sorted_rows) - 1)

            start0s = sorted_rows[start_indices]
            lengths = end_indices - start_indices + 1

            ranges = list(zip(start0s.tolist(), lengths.tolist()))

        try:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk_sorted = _read_ranges_as_chunk(reader, ranges)
        except Exception:
            chunk_sorted = None
        if chunk_sorted is None:
            return None

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        chunk = {}
        for name, value in chunk_sorted.items():
            if isinstance(value, np.ndarray):
                chunk[name] = value[inv]
            elif isinstance(value, list):
                chunk[name] = [value[i] for i in inv]
            else:
                chunk[name] = value

    # Keep cpp_numpy row reads on numpy-producing C++ APIs as a fallback.
    # Prefer reusing an open file handle to reduce repeated open/close overhead.
    if (
        chunk is None
        and hasattr(cpp, "TableReader")
        and hasattr(cpp.TableReader, "read_rows_numpy")
    ):
        try:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk = reader.read_rows_numpy(col_list, start_row, num_rows)
        except Exception:
            chunk = None
    if chunk is None and hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):
        try:
            file_handle = _acquire_cpp_handle(path, cpp)
            chunk = cpp.read_fits_table_rows_numpy_from_handle(
                file_handle, hdu, col_list, start_row, num_rows
            )
        except Exception:
            chunk = None
    if chunk is None and hasattr(cpp, "read_fits_table_rows_numpy"):
        try:
            chunk = cpp.read_fits_table_rows_numpy(
                path, hdu, col_list, start_row, num_rows, False
            )
        except Exception:
            chunk = None
    if chunk is None:
        return None

    pa = _require_pyarrow()
    if not chunk:
        return pa.table({})

    # Fast full-table conversion when metadata is not requested.
    if not field_meta and not table_meta:
        arrays: list[Any] = []
        names_out: list[str] = []
        names = preferred_order[:] if preferred_order else list(chunk.keys())
        for name in names:
            if name not in chunk:
                continue
            value = chunk[name]
            null_sentinel = (
                _column_tnull_from_meta(field_meta, name) if apply_fits_nulls else None
            )
            if isinstance(value, np.ndarray):
                arr = _numpy_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=col_tforms.get(name) if col_tforms else None,
                    unsigned_dtype=unsigned_dtypes.get(name),
                )
            elif isinstance(value, torch.Tensor):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                arr = _numpy_to_arrow_array(
                    pa,
                    t.numpy(),
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=col_tforms.get(name) if col_tforms else None,
                    unsigned_dtype=unsigned_dtypes.get(name),
                )
            elif isinstance(value, list):
                converted = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if not t.is_contiguous():
                            t = t.contiguous()
                        converted.append(t.numpy())
                    else:
                        converted.append(item)
                arr = _pa_array(pa, converted)
            elif _is_vla_tuple(value):
                arr = _vla_tuple_to_arrow_array(pa, value, null_sentinel=null_sentinel)
            else:
                arr = _pa_array(pa, value)
            names_out.append(name)
            arrays.append(arr)
        if not arrays:
            return pa.table({})
        return pa.Table.from_arrays(arrays, names=names_out)

    batch = _chunk_to_record_batch(
        chunk,
        decode_bytes,
        encoding,
        strip,
        field_meta=field_meta if include_fits_metadata else None,
        table_meta=table_meta if include_fits_metadata else None,
        preferred_order=preferred_order,
        null_meta=field_meta,
        apply_fits_nulls=apply_fits_nulls,
        column_tforms=col_tforms,
        unsigned_dtypes=unsigned_dtypes,
    )
    return pa.Table.from_batches([batch])


def scan_torch(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    device: str = "cpu",
    non_blocking: bool = True,
    pin_memory: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Stream table chunks as torch tensors, optionally moved to an accelerator.

    This keeps chunks bounded in memory and is useful for GPU training pipelines.
    """
    import torchfits

    start_row, num_rows = _normalize_row_slice(row_slice)
    use_mmap = mmap
    if use_mmap:
        # Avoid expensive exception-driven probing for BIT/VLA/scaled/vector tables.
        # This keeps `mmap=True` ergonomic: it silently uses the safe non-mmap path
        # when mmap row reads aren't supported (e.g., VLA columns).
        use_mmap = _can_use_mmap_row_path_for_full_read(path, hdu, columns)

    for chunk in torchfits.stream_table(
        path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        chunk_rows=batch_size,
        mmap=use_mmap,
    ):
        if device == "cpu":
            yield chunk
            continue

        moved: dict[str, Any] = {}
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                t = value
                if pin_memory and t.device.type == "cpu":
                    t = t.pin_memory()
                # MPS doesn't support float64; cast to float32 on transfer.
                if device == "mps" and t.dtype == torch.float64:
                    t = t.float()
                moved[key] = t.to(device, non_blocking=non_blocking)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        t = item
                        if pin_memory and t.device.type == "cpu":
                            t = t.pin_memory()
                        if device == "mps" and t.dtype == torch.float64:
                            t = t.float()
                        new_list.append(t.to(device, non_blocking=non_blocking))
                    else:
                        new_list.append(item)
                moved[key] = new_list
            else:
                moved[key] = value
        yield moved


def reader(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = True,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    """
    Return a PyArrow RecordBatchReader for streaming interoperability.

    This plugs directly into Arrow ecosystem tools without materializing the table.
    FITS character columns are decoded to Python strings by default (`decode_bytes=True`).
    """
    pa = _require_pyarrow()
    validate_table_backend(backend)
    scan_backend = backend
    batches = scan(
        path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        where=where,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=scan_backend,
    )
    it = iter(batches)
    first = next(it, None)
    if first is None:
        return pa.RecordBatchReader.from_batches(pa.schema([]), [])
    return pa.RecordBatchReader.from_batches(first.schema, itertools.chain([first], it))


def dataset(
    data: str | Any,
    **kwargs,
):
    """
    Return a pyarrow.dataset.Dataset from FITS table data.

    If `data` is a path, this uses `reader(...)` for streaming construction.
    """
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for dataset conversion") from exc

    if isinstance(data, str):
        return ds.dataset(reader(data, **kwargs))
    return ds.dataset(data)


def scanner(
    data: str | Any,
    *,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    filter: Any = None,
    batch_size: int = 65536,
    use_threads: bool = True,
    **kwargs,
):
    """
    Return a pyarrow.dataset.Scanner for projection/filter pushdown.

    For FITS paths, this builds a streaming dataset via `reader(...)`.
    """
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for scanner") from exc

    if where is not None:
        kwargs = dict(kwargs)
        kwargs["where"] = where

    if isinstance(data, str):
        dset = dataset(data, **kwargs)
    elif hasattr(data, "scanner"):
        dset = data
    else:
        dset = ds.dataset(data)
    return dset.scanner(
        columns=columns, filter=filter, batch_size=batch_size, use_threads=use_threads
    )


def _infer_fits_scalar_code(arr: "np.ndarray") -> str:
    kind = arr.dtype.kind
    itemsize = arr.dtype.itemsize
    if kind == "b":
        return "L"
    if kind == "u" and itemsize == 1:
        return "B"
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

    # If caller requests fixed-width strings and we have a uint8 matrix,
    # turn rows into fixed-width bytes.
    tform = str(fmt).strip().upper()
    if tform.endswith("A") and arr.ndim == 2 and arr.dtype == np.uint8:
        width = int(arr.shape[1])
        return (
            np.ascontiguousarray(arr).view(np.dtype(f"S{width}")).reshape(arr.shape[0])
        )

    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)

    return arr


def _apply_hdu_header_cards(hdu_header, header_map: dict[str, Any]) -> None:
    skip_keys = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTEND",
    }
    for key, value in (header_map or {}).items():
        key_upper = str(key).upper()
        if key_upper in skip_keys:
            continue
        if key_upper.startswith("TTYPE") or key_upper.startswith("TFORM"):
            continue
        if key_upper == "HISTORY":
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                hdu_header.add_history(str(item))
            continue
        if key_upper == "COMMENT":
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                hdu_header.add_comment(str(item))
            continue
        try:
            hdu_header[str(key)] = value
        except Exception:
            continue


def write(
    path: str,
    data: dict[str, Any],
    *,
    schema: Optional[dict[str, dict[str, Any]]] = None,
    header: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
    extname: Optional[str] = None,
    table_type: str = "binary",
) -> None:
    """
    Write a FITS table through the CFITSIO-native torchfits write path.
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("data must be a non-empty dictionary")
    table_kind = str(table_type).lower().strip()
    if table_kind not in {"binary", "ascii"}:
        raise ValueError("table_type must be 'binary' or 'ascii'")
    if schema is not None and not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary when provided")
    if extname is not None:
        hdr = dict(header or {})
        hdr["EXTNAME"] = str(extname)
    else:
        hdr = header
    import torchfits
    from ._io_engine.write_api import _prepare_unsigned_table_data_for_write

    data, schema, unsigned_converted = _prepare_unsigned_table_data_for_write(
        data, schema
    )

    if schema or unsigned_converted or table_kind == "ascii":
        import torchfits._C as cpp

        # Overwriting/creating a table can otherwise leave stale cached handles/metadata.
        _invalidate_path_caches(path)
        data = _normalize_cpp_table_data(data)
        cpp.write_fits_table(
            path,
            data,
            hdr if hdr else {},
            overwrite,
            schema if schema else None,
            table_kind,
        )
        if hdr:
            _write_header_cards_if_supported(path, 1, hdr)
        _invalidate_path_caches(path)
        return

    torchfits.write(path, data, header=hdr if hdr else None, overwrite=overwrite)


def _header_cards_to_mapping(header_cards: Any) -> dict[str, Any]:
    if isinstance(header_cards, dict):
        return {str(k): v for k, v in header_cards.items()}
    out: dict[str, Any] = {}
    if isinstance(header_cards, (list, tuple)):
        for card in header_cards:
            if hasattr(card, "key") and hasattr(card, "value"):
                out[str(card.key)] = card.value
                continue
            if not isinstance(card, (list, tuple)) or len(card) < 2:
                continue
            out[str(card[0])] = card[1]
    return out


def _column_tform_map(header_map: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    name_by_idx = _column_name_index_map(header_map)
    tform_by_idx: dict[int, str] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TFORM"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tform_by_idx[int(suffix)] = str(value)

    for idx, name in name_by_idx.items():
        out[name] = tform_by_idx.get(idx, "")
    return out


def _column_name_index_map(header_map: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}

    try:
        tfields = int(header_map.get("TFIELDS", 0))
    except (ValueError, TypeError):
        tfields = 0

    if tfields > 0:
        for i in range(1, tfields + 1):
            val = header_map.get(f"TTYPE{i}")
            if val is not None:
                out[i] = str(val)
        return out

    for key, value in header_map.items():
        key_u = str(key).upper()
        if not key_u.startswith("TTYPE"):
            continue
        suffix = key_u[5:]
        if suffix.isdigit():
            out[int(suffix)] = str(value)
    return out


def _extract_table_schema_from_header(
    header_map: dict[str, Any], columns: list[str]
) -> dict[str, dict[str, Any]]:
    name_by_idx = _column_name_index_map(header_map)
    index_by_name = {name: idx for idx, name in name_by_idx.items()}
    schema: dict[str, dict[str, Any]] = {}
    for name in columns:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        meta: dict[str, Any] = {}
        tform = header_map.get(f"TFORM{idx}")
        if tform is not None:
            meta["format"] = str(tform)
        tunit = header_map.get(f"TUNIT{idx}")
        if tunit is not None:
            meta["unit"] = str(tunit)
        tdim = header_map.get(f"TDIM{idx}")
        if tdim is not None:
            meta["dim"] = str(tdim)
        tnull = header_map.get(f"TNULL{idx}")
        if tnull is not None:
            meta["tnull"] = tnull
        tscal = header_map.get(f"TSCAL{idx}")
        if tscal is not None:
            meta["bscale"] = tscal
        tzero = header_map.get(f"TZERO{idx}")
        if tzero is not None:
            meta["bzero"] = tzero
        schema[name] = meta
    return schema


def _sanitize_table_header_for_rewrite(header_map: dict[str, Any]) -> dict[str, Any]:
    skip_exact = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "CHECKSUM",
        "DATASUM",
    }
    skip_prefixes = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO")
    out: dict[str, Any] = {}
    for key, value in header_map.items():
        key_s = str(key)
        key_u = key_s.upper()
        if key_u in skip_exact:
            continue
        if key_u.startswith(skip_prefixes):
            continue
        out[key_s] = value
    return out


def _infer_column_format_for_insert(name: str, values: Any) -> str:
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


def _ordered_dict_for_columns(
    columns: list[str], data_by_name: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in columns:
        out[name] = data_by_name[name]
    return out


def _rewrite_table_hdu_with_schema(
    path: str,
    target_hdu: int,
    data: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    header: dict[str, Any],
    table_type: str,
) -> None:
    import tempfile
    import torchfits
    import torchfits._C as cpp

    can_overwrite_table_only = False
    handle = cpp.open_fits_file(path, "r")
    try:
        num_hdus = int(cpp.get_num_hdus(handle))
        if num_hdus == 1 and target_hdu == 0:
            hdu_type = str(cpp.get_hdu_type(handle, 0))
            if hdu_type in {"BINARY_TABLE", "ASCII_TABLE"}:
                can_overwrite_table_only = True
        elif num_hdus == 2 and target_hdu == 1:
            hdu0_type = str(cpp.get_hdu_type(handle, 0))
            hdu1_type = str(cpp.get_hdu_type(handle, 1))
            if hdu0_type == "IMAGE" and hdu1_type in {"BINARY_TABLE", "ASCII_TABLE"}:
                h0_header = _header_cards_to_mapping(cpp.read_header(handle, 0))
                try:
                    naxis0 = int(h0_header.get("NAXIS", 0))
                except Exception:
                    naxis0 = 0
                if naxis0 == 0:
                    can_overwrite_table_only = True
                elif naxis0 == 1:
                    try:
                        naxis1 = int(h0_header.get("NAXIS1", 0))
                    except Exception:
                        naxis1 = -1
                    can_overwrite_table_only = naxis1 == 0
    finally:
        try:
            handle.close()
        except Exception:
            pass

    if can_overwrite_table_only:
        write(
            path,
            data=data,
            schema=schema,
            header=header,
            overwrite=True,
            table_type=table_type,
        )
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        write(
            tmp_path,
            data=data,
            schema=schema,
            header=header,
            overwrite=True,
            table_type=table_type,
        )
        with torchfits.open(tmp_path) as hdul:
            replacement = hdul[1].materialize(device="cpu")
        torchfits.replace_hdu(path, target_hdu, replacement)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _resolve_table_hdu_index_and_columns(
    path: str, hdu: int | str
) -> tuple[int, dict[str, Any], list[str], dict[str, str]]:
    import torchfits._C as cpp

    handle = cpp.open_fits_file(path, "r")
    try:
        num_hdus = int(cpp.get_num_hdus(handle))
        if num_hdus <= 0:
            raise RuntimeError(f"No HDUs found in '{path}'")

        target_idx: Optional[int] = None
        if isinstance(hdu, int):
            target_idx = hdu
        elif isinstance(hdu, str):
            wanted = hdu.strip().upper()
            if not wanted:
                raise ValueError("hdu name cannot be empty")
            for i in range(num_hdus):
                hdu_type = str(cpp.get_hdu_type(handle, i))
                if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
                    continue
                header_map = _header_cards_to_mapping(cpp.read_header(handle, i))
                extname = str(header_map.get("EXTNAME", "")).strip().upper()
                if extname == wanted:
                    target_idx = i
                    break
            if target_idx is None:
                raise KeyError(f"Table HDU named '{hdu}' not found in '{path}'")
        else:
            raise TypeError("hdu must be an int index or EXTNAME string")

        if target_idx is None or target_idx < 0 or target_idx >= num_hdus:
            raise IndexError(
                f"hdu index {hdu} out of range for '{path}' (num_hdus={num_hdus})"
            )

        hdu_type = str(cpp.get_hdu_type(handle, target_idx))
        if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
            raise ValueError(f"HDU {target_idx} is not a table (type={hdu_type})")

        header_map = _header_cards_to_mapping(cpp.read_header(handle, target_idx))
        col_map = _column_name_index_map(header_map)
        columns = [col_map[idx] for idx in sorted(col_map)]
        tform_map = _column_tform_map(header_map)
        return target_idx, header_map, columns, tform_map
    finally:
        try:
            handle.close()
        except Exception:
            pass


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
    """
    Insert a new table column, preserving existing schema metadata by default.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    import torchfits

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

    _invalidate_path_caches(path)
    torchfits.cache.clear()
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    torchfits.cache.clear()
    _invalidate_path_caches(path)


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
    """
    Replace an existing table column, preserving metadata unless overridden.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    import torchfits

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

    _invalidate_path_caches(path)
    torchfits.cache.clear()
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def _coerce_table_column_array(
    name: str,
    value: Any,
    *,
    expected_rows: Optional[int] = None,
    allow_2d: bool = True,
) -> "np.ndarray":
    import numpy as np

    global _COMPLEX_DTYPE_MAP
    if not _COMPLEX_DTYPE_MAP:
        _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
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

    global _VLA_DTYPE_MAP
    if not _VLA_DTYPE_MAP:
        _VLA_DTYPE_MAP = {
            "L": np.bool_,
            "B": np.uint8,
            "I": np.int16,
            "J": np.int32,
            "K": np.int64,
            "E": np.float32,
            "D": np.float64,
        }
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

    global _COMPLEX_DTYPE_MAP
    if not _COMPLEX_DTYPE_MAP:
        _COMPLEX_DTYPE_MAP = {
            "C": np.complex64,
            "M": np.complex128,
        }
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


def append_rows(
    path: str,
    rows: dict[str, Any],
    hdu: int | str = 1,
) -> None:
    """
    Append rows to an existing FITS table HDU (CFITSIO in-place).
    """
    rows = _coerce_rows_from_arrow(rows)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("rows must be a non-empty dictionary")
    import torchfits
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

    # Ensure no stale cached handles/metadata exist before mutating the file in-place.
    _invalidate_path_caches(path)
    torchfits.cache.clear()
    cpp.append_fits_table_rows(path, target_hdu, normalized)
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def insert_rows(
    path: str,
    rows: dict[str, Any],
    *,
    row: int,
    hdu: int | str = 1,
) -> None:
    """
    Insert rows at a 0-based row index in an existing FITS table HDU.

    Missing columns are filled with deterministic defaults (numeric/logical: TNULL
    when defined, else zero/False; strings: empty string; VLA: empty array).
    """
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

    start_row = row + 1  # FITS rows are 1-based.
    _invalidate_path_caches(path)
    torchfits.cache.clear()
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
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def delete_rows(
    path: str,
    row_slice: int | slice | tuple[int, int],
    *,
    hdu: int | str = 1,
) -> None:
    """
    Delete table rows by 0-based index/slice from an existing FITS table HDU.
    """
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

    _invalidate_path_caches(path)
    torchfits.cache.clear()
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
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def update_rows(
    path: str,
    rows: dict[str, Any],
    row_slice: slice | tuple[int, int],
    hdu: int | str = 1,
    *,
    mmap: bool | str = "auto",
) -> None:
    """
    Update an existing row slice in-place for selected columns.

    mmap="auto" attempts a direct mmap write for numeric columns (fastest), with
    a CFITSIO fallback for unsupported columns.
    """
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
    global _COMPLEX_DTYPE_MAP
    if not _COMPLEX_DTYPE_MAP:
        import numpy as np

        _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
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
            values = _coerce_table_string_values(
                col_name, value, expected_rows=expected_rows
            )
            if expected_rows is None:
                expected_rows = len(values)
            # Materialise fixed-width CHAR columns as a (num_rows, width)
            # uint8 ndarray so the mmap fast path
            # (cpp.update_fits_table_rows_mmap) routes through the new
            # STRING case in the C++ writer rather than the buffered
            # fallback (which accepts list[str]). The C++ writer copies
            # bytes left-to-right per row, so short user payloads are
            # right-padded with ASCII spaces (0x20) before they hit
            # disk. _coerce_table_string_values already truncates at
            # the column width when user payloads are wider, so we
            # only need to handle the short-payload case here.
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

    import torchfits
    import torchfits._C as cpp

    # Ensure no stale cached handles/metadata exist before mutating the file in-place.
    _invalidate_path_caches(path)

    use_mmap = mmap in (True, "auto", "mmap")
    forced_mmap = mmap in (True, "mmap")
    # VLA columns are still unsupported in-place (heap indirection); the
    # C++ layer additionally rejects scaled columns at write time.
    # Fixed-width STRING (A), BIT (X), and COMPLEX (C/M) columns are now
    # handled in the C++ mmap writer alongside numeric/logical columns.
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
                torchfits.cache.clear()
                cpp.update_fits_table_rows_mmap(
                    path, target_hdu, normalized, start_row, num_rows
                )
                torchfits.cache.clear()
                _invalidate_path_caches(path)
                return
            except Exception:
                if mmap is True:
                    raise

    torchfits.cache.clear()
    cpp.update_fits_table_rows(path, target_hdu, normalized, start_row, num_rows)
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def rename_columns(
    path: str,
    mapping: dict[str, str],
    hdu: int | str = 1,
) -> None:
    """
    Rename one or more table columns in-place.
    """
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

    import torchfits
    import torchfits._C as cpp

    _invalidate_path_caches(path)
    torchfits.cache.clear()
    cpp.rename_fits_table_columns(path, target_hdu, normalized)
    torchfits.cache.clear()
    _invalidate_path_caches(path)


def drop_columns(
    path: str,
    columns: list[str] | tuple[str, ...],
    hdu: int | str = 1,
) -> None:
    """
    Drop one or more table columns in-place.
    """
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

    import torchfits
    import torchfits._C as cpp

    _invalidate_path_caches(path)
    torchfits.cache.clear()
    cpp.drop_fits_table_columns(path, target_hdu, normalized)
    torchfits.cache.clear()
    _invalidate_path_caches(path)


# -- interop re-exports (implementations live in _table.interop) --------------

from ._table.interop import (  # noqa: E402,F401
    _materialize_arrow_table,
    _split_io_kwargs,
    duckdb_query,
    to_duckdb,
    to_pandas,
    to_polars,
    to_polars_lazy,
    write_parquet,
)


# -- arrow-convert re-exports (implementations live in _table.arrow_convert) ----

from ._table.arrow_convert import (  # noqa: E402,F401
    _chunk_to_record_batch,
    _coerce_null_sentinel,
    _column_tnull_from_meta,
    _decode_uint8_matrix_to_arrow,
    _is_vla_tuple,
    _numpy_to_arrow_array,
    _pa_array,
    _tensor_to_arrow_array,
    _uint8_matrix_to_fixed_binary,
    _uint8_matrix_to_fixed_bool_list,
    _vla_tuple_to_arrow_array,
)
