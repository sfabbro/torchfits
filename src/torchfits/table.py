"""Arrow-native table I/O helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from collections import OrderedDict
import atexit
import itertools
import os
import re
import threading
from typing import Any, Optional

import numpy as np
import torch

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

_TFORM_RE = re.compile(r"^\s*(\d+)?\s*([A-Za-z])")
_TFORM_VLA_RE = re.compile(r"^\s*(\d+)?\s*([PQ])\s*([A-Za-z])")
_WHERE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HANDLE_CACHE_MAX = max(1, int(os.getenv("TORCHFITS_TABLE_HANDLE_CACHE_SIZE", "8")))
_HANDLE_CACHE_ENABLED = os.getenv("TORCHFITS_TABLE_HANDLE_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_READER_CACHE_MAX = max(1, int(os.getenv("TORCHFITS_TABLE_READER_CACHE_SIZE", "8")))
_READER_CACHE_ENABLED = os.getenv("TORCHFITS_TABLE_READER_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_handle_cache_lock = threading.Lock()
_handle_cache: "OrderedDict[str, Any]" = OrderedDict()
_reader_cache_lock = threading.Lock()
_reader_cache: "OrderedDict[tuple[str, int], Any]" = OrderedDict()


def _close_cpp_handle(handle: Any) -> None:
    try:
        handle.close()
    except Exception:
        pass


def _acquire_cpp_handle(path: str, cpp) -> Any:
    if not _HANDLE_CACHE_ENABLED:
        return cpp.open_fits_file(path, "r")

    with _handle_cache_lock:
        handle = _handle_cache.get(path)
        if handle is not None:
            _handle_cache.move_to_end(path)
            return handle

    handle = cpp.open_fits_file(path, "r")
    with _handle_cache_lock:
        _handle_cache[path] = handle
        _handle_cache.move_to_end(path)
        while len(_handle_cache) > _HANDLE_CACHE_MAX:
            _, old = _handle_cache.popitem(last=False)
            _close_cpp_handle(old)
    return handle


def _acquire_cpp_reader(path: str, hdu: int, cpp) -> Any:
    """
    Return a cached C++ TableReader bound to a cached FITSFile handle.

    This avoids re-parsing the table schema on every small projected read.
    """
    file_handle = _acquire_cpp_handle(path, cpp)
    if not _READER_CACHE_ENABLED:
        return cpp.TableReader(file_handle, int(hdu))

    key = (path, int(hdu))
    with _reader_cache_lock:
        reader = _reader_cache.get(key)
        if reader is not None:
            _reader_cache.move_to_end(key)
            return reader

    reader = cpp.TableReader(file_handle, int(hdu))
    with _reader_cache_lock:
        _reader_cache[key] = reader
        _reader_cache.move_to_end(key)
        while len(_reader_cache) > _READER_CACHE_MAX:
            _reader_cache.popitem(last=False)
    return reader


def _close_all_cached_handles() -> None:
    # Readers borrow FITSFile pointers; clear them first.
    with _reader_cache_lock:
        _reader_cache.clear()
    with _handle_cache_lock:
        items = list(_handle_cache.items())
        _handle_cache.clear()
    for _, handle in items:
        _close_cpp_handle(handle)


def _invalidate_caches_for_path(path: str) -> None:
    """Drop cached readers/handles bound to a given file path."""
    with _reader_cache_lock:
        stale_reader_keys = [k for k in _reader_cache.keys() if k[0] == path]
        for key in stale_reader_keys:
            _reader_cache.pop(key, None)

    handle = None
    with _handle_cache_lock:
        handle = _handle_cache.pop(path, None)
    if handle is not None:
        _close_cpp_handle(handle)


atexit.register(_close_all_cached_handles)


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for torchfits.table APIs. Install pyarrow to use Arrow-native tables."
        ) from exc
    return pa


def _arrow_column_to_python(pa, column, name: str) -> Any:
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
        values = column.to_pylist()
        out = []
        for item in values:
            if item is None:
                out.append([])
            else:
                out.append(np.asarray(item))
        return out

    return column.to_numpy(zero_copy_only=False)


def _parse_tform(tform: str) -> tuple[bool, str, int]:
    tform = str(tform).strip().upper()
    if not tform:
        return False, "", 1
    m = _TFORM_VLA_RE.match(tform)
    if m:
        repeat = int(m.group(1)) if m.group(1) else 1
        code = m.group(3).upper()
        return True, code, repeat
    m = _TFORM_RE.match(tform)
    if m:
        repeat = int(m.group(1)) if m.group(1) else 1
        code = m.group(2).upper()
        return False, code, repeat
    return False, "", 1


def _column_tnull_map(header_map: dict[str, Any]) -> dict[str, Any]:
    name_by_idx: dict[int, str] = {}
    tnull_by_idx: dict[int, Any] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TTYPE"):
            suffix = key_u[5:]
            if suffix.isdigit():
                name_by_idx[int(suffix)] = str(value)
        elif key_u.startswith("TNULL"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tnull_by_idx[int(suffix)] = value

    out: dict[str, Any] = {}
    for idx, name in name_by_idx.items():
        if idx in tnull_by_idx:
            out[name] = tnull_by_idx[idx]
    return out


def _default_table_column_values(
    name: str,
    tform: str,
    num_rows: int,
    tnull: Any = None,
):
    is_vla, code, repeat = _parse_tform(tform)
    if repeat <= 0:
        repeat = 1

    if is_vla:
        dtype = _VLA_DTYPE_MAP.get(code, np.float32)
        return [np.asarray([], dtype=dtype) for _ in range(num_rows)]

    if code == "A":
        return [""] * num_rows

    if code in _COMPLEX_DTYPE_MAP:
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
            fill = np.asarray(tnull, dtype=dtype).item()
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
        elif code in _COMPLEX_DTYPE_MAP:
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


def _pa_array(pa, value, *, mask=None, type=None):
    kwargs: dict[str, Any] = {"from_pandas": False}
    if mask is not None:
        kwargs["mask"] = mask
    if type is not None:
        kwargs["type"] = type
    try:
        return pa.array(value, **kwargs)
    except TypeError:
        kwargs.pop("from_pandas", None)
        return pa.array(value, **kwargs)


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


def _parse_where_literal(raw: str) -> Any:
    token = raw.strip()
    if not token:
        raise ValueError("where literal cannot be empty")

    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        quote = token[0]
        inner = token[1:-1]
        return inner.replace(f"\\{quote}", quote)

    token_lower = token.lower()
    if token_lower == "true":
        return True
    if token_lower == "false":
        return False
    if token_lower in {"none", "null"}:
        return None

    if re.fullmatch(r"[+-]?\d+", token):
        try:
            return int(token)
        except Exception:
            pass

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        try:
            return float(token)
        except Exception:
            pass

    # Bare-word strings are accepted (e.g. where="NAME == STAR_A").
    return token


def _tokenize_where_expression(where: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(where)
    while i < n:
        ch = where[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ch))
            i += 1
            continue
        if ch == ",":
            tokens.append(("COMMA", ch))
            i += 1
            continue

        if i + 1 < n:
            op2 = where[i : i + 2]
            if op2 in {"==", "!=", ">=", "<="}:
                tokens.append(("OP", op2))
                i += 2
                continue
        if ch in {">", "<"}:
            tokens.append(("OP", ch))
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            i += 1
            buf: list[str] = []
            while i < n:
                cur = where[i]
                if cur == "\\" and i + 1 < n:
                    buf.append(where[i + 1])
                    i += 2
                    continue
                if cur == quote:
                    break
                buf.append(cur)
                i += 1
            if i >= n or where[i] != quote:
                raise ValueError("Unterminated quoted literal in where expression")
            i += 1
            tokens.append(("LITERAL", "".join(buf)))
            continue

        start = i
        while i < n:
            cur = where[i]
            if cur.isspace() or cur in {"(", ")", ",", ">", "<", "!", "="}:
                break
            i += 1
        token = where[start:i]
        if not token:
            raise ValueError(
                f"Unexpected token in where expression near position {start}"
            )
        tokens.append(("WORD", token))

    return tokens


def _parse_where_expression(where: str):
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty string expression")

    tokens = _tokenize_where_expression(where)
    if not tokens:
        raise ValueError("where must be a non-empty string expression")

    idx = 0

    def _peek() -> Optional[tuple[str, str]]:
        return tokens[idx] if idx < len(tokens) else None

    def _consume() -> tuple[str, str]:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("Unexpected end of where expression")
        out = tokens[idx]
        idx += 1
        return out

    def _consume_logic(expected: str) -> None:
        tok = _peek()
        if tok is None or tok[0] != "WORD" or tok[1].upper() != expected:
            raise ValueError(f"Expected '{expected}' in where expression")
        _consume()

    def _parse_literal_token(tok: tuple[str, str]) -> Any:
        if tok[0] == "LITERAL":
            return tok[1]
        if tok[0] == "WORD":
            return _parse_where_literal(tok[1])
        raise ValueError("where expects a literal value")

    def _parse_literal_list() -> list[Any]:
        head = _consume()
        if head[0] != "LPAREN":
            raise ValueError("where IN expects '(' after IN")
        literals: list[Any] = []
        while True:
            tok = _peek()
            if tok is None:
                raise ValueError("Unexpected end of where expression in IN list")
            if tok[0] == "RPAREN":
                _consume()
                break
            if literals:
                sep = _consume()
                if sep[0] != "COMMA":
                    raise ValueError("where IN expects ',' between list literals")
            tok = _consume()
            literals.append(_parse_literal_token(tok))
        return literals

    def _parse_comparison():
        lhs = _consume()
        if lhs[0] != "WORD" or _WHERE_IDENT_RE.fullmatch(lhs[1]) is None:
            raise ValueError(
                "where expects a column identifier before comparison operator"
            )
        op_tok = _peek()
        if op_tok is None:
            raise ValueError(
                "where expects a comparison operator after column identifier"
            )

        if op_tok[0] == "OP":
            _consume()
            rhs = _consume()
            literal = _parse_literal_token(rhs)
            return ("cmp", lhs[1], op_tok[1], literal)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "IN":
            _consume_logic("IN")
            return ("in", lhs[1], _parse_literal_list(), False)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "BETWEEN":
            _consume_logic("BETWEEN")
            low = _parse_literal_token(_consume())
            _consume_logic("AND")
            high = _parse_literal_token(_consume())
            return ("between", lhs[1], low, high, False)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "IS":
            _consume_logic("IS")
            next_tok = _peek()
            negate = False
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "NOT"
            ):
                _consume_logic("NOT")
                negate = True
            _consume_logic("NULL")
            return ("isnull", lhs[1], negate)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "NOT":
            _consume_logic("NOT")
            next_tok = _peek()
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "IN"
            ):
                _consume_logic("IN")
                return ("in", lhs[1], _parse_literal_list(), True)
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "BETWEEN"
            ):
                _consume_logic("BETWEEN")
                low = _parse_literal_token(_consume())
                _consume_logic("AND")
                high = _parse_literal_token(_consume())
                return ("between", lhs[1], low, high, True)
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "NULL"
            ):
                _consume_logic("NULL")
                return ("isnull", lhs[1], True)
            raise ValueError("where expects IN/BETWEEN/NULL after NOT")

        raise ValueError(
            "where expects a comparison operator or IN/BETWEEN/IS NULL variants after column identifier"
        )

    def _parse_primary():
        tok = _peek()
        if tok is None:
            raise ValueError("Unexpected end of where expression")
        if tok[0] == "LPAREN":
            _consume()
            node = _parse_or()
            tail = _consume()
            if tail[0] != "RPAREN":
                raise ValueError("Unbalanced parentheses in where expression")
            return node
        return _parse_comparison()

    def _parse_not():
        tok = _peek()
        if tok is not None and tok[0] == "WORD" and tok[1].upper() == "NOT":
            _consume_logic("NOT")
            return ("not", _parse_not())
        return _parse_primary()

    def _parse_and():
        node = _parse_not()
        while True:
            tok = _peek()
            if tok is not None and tok[0] == "WORD" and tok[1].upper() == "AND":
                _consume_logic("AND")
                rhs = _parse_not()
                node = ("and", node, rhs)
                continue
            return node

    def _parse_or():
        node = _parse_and()
        while True:
            tok = _peek()
            if tok is not None and tok[0] == "WORD" and tok[1].upper() == "OR":
                _consume_logic("OR")
                rhs = _parse_and()
                node = ("or", node, rhs)
                continue
            return node

    ast = _parse_or()
    if idx != len(tokens):
        raise ValueError("Unexpected trailing tokens in where expression")
    return ast


def _where_columns_from_ast(ast) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _visit(node) -> None:
        kind = node[0]
        if kind in {"cmp", "in", "between", "isnull"}:
            name = node[1]
            if name not in seen:
                seen.add(name)
                out.append(name)
        elif kind == "and" or kind == "or":
            _visit(node[1])
            _visit(node[2])
        elif kind == "not":
            _visit(node[1])
        else:
            raise ValueError("Invalid where AST")

    _visit(ast)
    return out


def _compile_where_to_simple_predicates(
    where: str,
) -> Optional[list[tuple[str, str, Any]]]:
    """
    Compile a restricted where expression into C++ predicate tuples.

    Returns `None` when expression cannot be represented as a pure conjunction
    of simple binary comparisons.
    """
    try:
        ast = _parse_where_expression(where)
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


def _where_mask_for_table(table, where: str, parsed_ast=None) -> np.ndarray:
    pa = _require_pyarrow()
    import pyarrow.compute as pc

    ast = parsed_ast if parsed_ast is not None else _parse_where_expression(where)

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

    mask = pc.fill_null(_eval(ast), False)
    return np.asarray(mask.to_pylist(), dtype=np.bool_)


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
    where_ast = _parse_where_expression(where)
    where_columns = _where_columns_from_ast(where_ast)
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

    mask = _where_mask_for_table(predicate_table, where, parsed_ast=where_ast)
    if mask.size == 0:
        return []

    base_row0 = start_row - 1
    selected = np.flatnonzero(mask)
    return [base_row0 + int(idx) for idx in selected]


def _split_io_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    io_kwargs = {k: v for k, v in kwargs.items() if k in _TABLE_IO_KEYS}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in _TABLE_IO_KEYS}
    return io_kwargs, other_kwargs


def _coerce_null_sentinel(value: np.ndarray, sentinel: Any) -> Any:
    if sentinel is None:
        return None
    arr = np.ascontiguousarray(value)
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    try:
        if arr.dtype.kind in {"i", "u", "b"}:
            return np.array(sentinel, dtype=arr.dtype).item()
        return float(sentinel)
    except Exception:
        return None


def _column_tnull_from_meta(
    null_meta: Optional[dict[str, dict[str, str]]], name: str
) -> Any:
    if not null_meta:
        return None
    field = null_meta.get(name)
    if not field:
        return None
    return field.get("fits_tnull")


def _tensor_to_arrow_array(
    pa,
    tensor: torch.Tensor,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
):
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()

    return _numpy_to_arrow_array(
        pa, t.numpy(), decode_bytes, encoding, strip, null_sentinel=null_sentinel
    )


def _uint8_matrix_to_fixed_binary(pa, value: np.ndarray):
    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr)
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [b""] * int(arr.shape[0]))
    byte_view = arr.view(np.dtype(f"S{width}")).reshape(arr.shape[0])
    return _pa_array(pa, byte_view, type=pa.binary(width))


def _decode_uint8_matrix_to_arrow(pa, value: np.ndarray, encoding: str, strip: bool):
    arr = np.ascontiguousarray(value)
    if arr.ndim != 2:
        return _pa_array(pa, arr)
    width = int(arr.shape[1])
    if width <= 0:
        return _pa_array(pa, [""] * int(arr.shape[0]))

    # Vectorized fixed-width bytes -> unicode decode.
    byte_view = arr.view(np.dtype(f"S{width}")).reshape(arr.shape[0])
    if (
        strip
        and encoding.lower() in {"ascii", "utf8", "utf-8"}
        and not np.any(arr[:, -1] == 32)
    ):
        # Fast path: if no row ends with a space, Arrow cast handles NUL trimming correctly.
        try:
            import pyarrow.compute as pc

            return pc.cast(_pa_array(pa, byte_view), pa.string())
        except Exception:
            pass
    if strip:
        # Stripping while still in bytes form is much faster than stripping unicode.
        byte_view = np.char.rstrip(byte_view, b" \x00")
    decoded = np.char.decode(byte_view, encoding=encoding, errors="ignore")
    return _pa_array(pa, decoded)


def _numpy_to_arrow_array(
    pa,
    value: np.ndarray,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    null_sentinel: Any = None,
):
    arr = np.ascontiguousarray(value)
    if arr.ndim <= 1:
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            return _pa_array(pa, arr)
        mask = arr == sentinel
        if mask.any():
            return _pa_array(pa, arr, mask=mask)
        return _pa_array(pa, arr)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            if decode_bytes:
                return _decode_uint8_matrix_to_arrow(pa, arr, encoding, strip)
            return _uint8_matrix_to_fixed_binary(pa, arr)
        flat = arr.reshape(-1)
        sentinel = _coerce_null_sentinel(arr, null_sentinel)
        if sentinel is None:
            values = _pa_array(pa, flat)
        else:
            flat_mask = flat == sentinel
            values = (
                _pa_array(pa, flat, mask=flat_mask)
                if flat_mask.any()
                else _pa_array(pa, flat)
            )
        return pa.FixedSizeListArray.from_arrays(values, int(arr.shape[1]))
    return _pa_array(pa, arr.tolist())


def _is_vla_tuple(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return isinstance(value[0], np.ndarray) and isinstance(value[1], np.ndarray)


def _vla_tuple_to_arrow_array(
    pa, value: tuple[np.ndarray, np.ndarray], null_sentinel: Any = None
):
    flat = np.ascontiguousarray(value[0]).reshape(-1)
    offsets64 = np.ascontiguousarray(value[1], dtype=np.int64).reshape(-1)
    if offsets64.size == 0:
        return _pa_array(pa, [])
    sentinel = _coerce_null_sentinel(flat, null_sentinel)
    if sentinel is None:
        values = _pa_array(pa, flat)
    else:
        mask = flat == sentinel
        values = _pa_array(pa, flat, mask=mask) if mask.any() else _pa_array(pa, flat)

    if int(offsets64[-1]) <= np.iinfo(np.int32).max:
        offsets = offsets64.astype(np.int32, copy=False)
        return pa.ListArray.from_arrays(_pa_array(pa, offsets), values)
    return pa.LargeListArray.from_arrays(_pa_array(pa, offsets64), values)


def _chunk_to_record_batch(
    chunk: dict[str, Any],
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    field_meta: Optional[dict[str, dict[str, str]]] = None,
    table_meta: Optional[dict[str, str]] = None,
    preferred_order: Optional[list[str]] = None,
    null_meta: Optional[dict[str, dict[str, str]]] = None,
    apply_fits_nulls: bool = False,
):
    pa = _require_pyarrow()

    # Fast path when schema metadata is not requested.
    if not field_meta and not table_meta:
        pydict: dict[str, Any] = {}
        ordered_names: list[str] = []
        if preferred_order:
            for name in preferred_order:
                if name in chunk:
                    ordered_names.append(name)
        for name in chunk.keys():
            if name not in ordered_names:
                ordered_names.append(name)
        if not ordered_names:
            ordered_names = sorted(chunk.keys())

        for name in ordered_names:
            value = chunk[name]
            null_sentinel = (
                _column_tnull_from_meta(null_meta, name) if apply_fits_nulls else None
            )
            if isinstance(value, torch.Tensor):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                pydict[name] = _numpy_to_arrow_array(
                    pa,
                    t.numpy(),
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                )
            elif isinstance(value, np.ndarray):
                pydict[name] = _numpy_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
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
                pydict[name] = converted
            elif _is_vla_tuple(value):
                pydict[name] = _vla_tuple_to_arrow_array(
                    pa, value, null_sentinel=null_sentinel
                )
            else:
                pydict[name] = value
        return pa.RecordBatch.from_pydict(pydict)

    arrays = []
    fields = []

    ordered_names: list[str] = []
    if preferred_order:
        for name in preferred_order:
            if name in chunk:
                ordered_names.append(name)
    for name in chunk.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    for name in ordered_names:
        value = chunk[name]
        null_sentinel = (
            _column_tnull_from_meta(null_meta, name) if apply_fits_nulls else None
        )
        if isinstance(value, torch.Tensor):
            arr = _tensor_to_arrow_array(
                pa, value, decode_bytes, encoding, strip, null_sentinel=null_sentinel
            )
        elif isinstance(value, np.ndarray):
            arr = _numpy_to_arrow_array(
                pa, value, decode_bytes, encoding, strip, null_sentinel=null_sentinel
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
        arrays.append(arr)
        meta = None
        if field_meta and name in field_meta:
            meta = {
                k.encode("utf-8"): v.encode("utf-8")
                for k, v in field_meta[name].items()
            }
        fields.append(pa.field(name, arr.type, metadata=meta))

    schema_meta = None
    if table_meta:
        schema_meta = {
            k.encode("utf-8"): v.encode("utf-8") for k, v in table_meta.items()
        }
    return pa.RecordBatch.from_arrays(
        arrays, schema=pa.schema(fields, metadata=schema_meta)
    )


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
        name = header.get(f"TTYPE{i}")
        if not isinstance(name, str) or not name:
            continue
        if selected_columns is not None and name not in selected_columns:
            continue

        entry: dict[str, str] = {}
        for key_name in ("TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO"):
            value = header.get(f"{key_name}{i}")
            if value is not None:
                entry[f"fits_{key_name.lower()}"] = str(value)
        if entry:
            field_meta[name] = entry

    return field_meta, table_meta


def _column_tform_code_and_repeat(tform: Any) -> tuple[str, int] | None:
    if not isinstance(tform, str):
        return None
    m = _TFORM_RE.match(tform)
    if not m:
        return None
    repeat_text, code = m.groups()
    repeat = int(repeat_text) if repeat_text else 1
    return code.upper(), repeat


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
    import torchfits.cpp as cpp

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
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
) -> Iterator[Any]:
    """
    Stream a FITS table as Arrow record batches.

    This is out-of-core friendly: each yielded batch is independently materialized.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if backend not in {"auto", "torch", "cpp_numpy"}:
        raise ValueError("backend must be one of: auto, torch, cpp_numpy")

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
        )


def read(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    rows: Optional[list[int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    """Read a FITS table into an Arrow Table."""
    if backend not in {"auto", "torch", "cpp_numpy"}:
        raise ValueError("backend must be one of: auto, torch, cpp_numpy")
    pa = _require_pyarrow()

    if backend in {"auto", "cpp_numpy"}:
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
        # Try fast path: Predicate Pushdown
        import torchfits.cpp as cpp

        if hasattr(cpp, "read_fits_table_filtered"):
            filters = _compile_where_to_simple_predicates(where)
            if filters is not None:
                # Use fast path
                try:
                    # columns=None in C++ means all columns? No, usually means return empty?
                    # read_columns_mmap checks empty.
                    # If columns is None here, we need to fetch all columns from header?
                    # Or pass empty list?
                    # If columns is None, we need schema to know ALL columns.
                    # Better to let fallback handle 'columns=None' or fetch schema first.
                    # But fetching schema defeats the purpose of speed if not careful.

                    target_cols = columns
                    if target_cols is None:
                        # Fetch all column names quickly
                        schema_ = schema(
                            path, hdu=hdu, backend="cpp_numpy"
                        )  # Minimal read?
                        target_cols = list(schema_.names)

                    data_dict = cpp.read_fits_table_filtered(
                        path, hdu, target_cols, filters
                    )

                    # Convert to Arrow Table
                    arrays = []
                    names_out = []
                    for name in target_cols:
                        if name in data_dict:
                            # Convert tensor to arrow
                            val = data_dict[name]
                            if isinstance(val, torch.Tensor):
                                # Ensure cpu/contiguous
                                if val.device.type != "cpu":
                                    val = val.cpu()
                                if not val.is_contiguous():
                                    val = val.contiguous()
                                arr = _numpy_to_arrow_array(
                                    pa, val.numpy(), decode_bytes, encoding, strip
                                )
                                arrays.append(arr)
                                names_out.append(name)

                    if not arrays:
                        return pa.table({})
                    return pa.Table.from_arrays(arrays, names=names_out)

                except Exception:
                    # If fast path fails (e.g. type mismatch), fall back to slow path
                    pass

        base = read(
            path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=None,
            batch_size=batch_size,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend="torch",
        )
        mask = _where_mask_for_table(base, where)
        if mask.size == 0:
            return base.slice(0, 0)
        return base.filter(_pa_array(pa, mask))

    scan_backend = backend
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
            backend=scan_backend,
        )
    )
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def schema(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = False,
    backend: str = "auto",
):
    """Fetch Arrow schema for a FITS table with minimal read."""
    pa = _require_pyarrow()
    if backend not in {"auto", "torch", "cpp_numpy"}:
        raise ValueError("backend must be one of: auto, torch, cpp_numpy")
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
    import torchfits.cpp as cpp

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
        ranges: list[tuple[int, int]] = []
        start0 = int(sorted_rows[0])
        length = 1
        for i in range(1, len(sorted_rows)):
            cur = int(sorted_rows[i])
            prev = int(sorted_rows[i - 1])
            if cur == prev + 1:
                length += 1
                continue
            ranges.append((start0, length))
            start0 = cur
            length = 1
        ranges.append((start0, length))

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
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = True,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    """
    Return a PyArrow RecordBatchReader for streaming interoperability.

    This plugs directly into Arrow ecosystem tools without materializing the table.
    """
    pa = _require_pyarrow()
    if backend not in {"auto", "torch", "cpp_numpy"}:
        raise ValueError("backend must be one of: auto, torch, cpp_numpy")
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


def write_parquet(
    where: str,
    data: str | Any | Iterable[Any],
    *,
    stream: bool = False,
    compression: str = "zstd",
    row_group_size: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Write Arrow-native table data to parquet.

    Args:
        where: Destination parquet file path.
        data: FITS file path, Arrow Table, RecordBatchReader, or iterable of RecordBatch.
        stream: Enable streaming parquet writes (bounded memory).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow.parquet is required for parquet export") from exc

    pa = _require_pyarrow()

    if isinstance(data, str):
        if stream:
            data = reader(data, **kwargs)
        else:
            data = read(data, **kwargs)

    if not stream:
        if hasattr(data, "read_next_batch"):
            table = pa.Table.from_batches(list(data))
        elif hasattr(data, "to_batches"):
            table = data
        else:
            table = pa.Table.from_batches(list(data))
        pq.write_table(
            table, where, compression=compression, row_group_size=row_group_size
        )
        return

    writer = None
    try:
        if hasattr(data, "read_next_batch"):
            while True:
                try:
                    batch = data.read_next_batch()
                except StopIteration:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
        else:
            for batch in data:
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()


def to_pandas(
    data: str | Any | Iterable[Any],
    stream: bool = False,
    **kwargs,
):
    """
    Convert Arrow table data to pandas.

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        stream: When True, return an iterator of DataFrames.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for to_pandas conversion") from exc

    pa = _require_pyarrow()

    if isinstance(data, str):
        io_kwargs, pandas_kwargs = _split_io_kwargs(kwargs)
        if stream:
            return (
                pa.Table.from_batches([batch]).to_pandas(**pandas_kwargs)
                for batch in scan(data, **io_kwargs)
            )
        return read(data, **io_kwargs).to_pandas(**pandas_kwargs)

    if hasattr(data, "to_pandas"):
        return data.to_pandas(**kwargs)

    if stream:
        return (pa.Table.from_batches([batch]).to_pandas(**kwargs) for batch in data)

    frames = [pa.Table.from_batches([batch]).to_pandas(**kwargs) for batch in data]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_polars(
    data: str | Any | Iterable[Any],
    stream: bool = False,
    **kwargs,
):
    """
    Convert Arrow table data to polars DataFrame(s).

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        stream: When True, return an iterator of polars DataFrames.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for to_polars conversion") from exc

    if isinstance(data, str):
        io_kwargs, _ = _split_io_kwargs(kwargs)
        if stream:
            return (pl.from_arrow(batch) for batch in scan(data, **io_kwargs))
        return pl.from_arrow(read(data, **io_kwargs))

    if stream:
        return (pl.from_arrow(batch) for batch in data)

    return pl.from_arrow(data)


def _materialize_arrow_table(data: str | Any | Iterable[Any], **kwargs):
    """Normalize path/reader/batches into a single pyarrow.Table."""
    pa = _require_pyarrow()

    if isinstance(data, str):
        io_kwargs, _ = _split_io_kwargs(kwargs)
        return read(data, **io_kwargs)

    if hasattr(data, "to_batches"):
        return data

    if hasattr(data, "read_next_batch"):
        return pa.Table.from_batches(list(data))

    if hasattr(pa, "RecordBatch") and isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

    return pa.Table.from_batches(list(data))


def to_polars_lazy(
    data: str | Any | Iterable[Any],
    **kwargs,
):
    """
    Convert table data into a Polars LazyFrame for complex expressions.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for to_polars_lazy conversion") from exc

    table = _materialize_arrow_table(data, **kwargs)
    return pl.from_arrow(table).lazy()


def to_duckdb(
    data: str | Any | Iterable[Any],
    relation_name: str = "fits_table",
    connection: Any = None,
    **kwargs,
):
    """
    Register table data in DuckDB and return a relation.

    This is intended for SQL-style joins/group-bys/windows while keeping torchfits
    focused on FITS-native I/O and conversion.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required for to_duckdb conversion") from exc

    if not isinstance(relation_name, str) or not relation_name:
        raise ValueError("relation_name must be a non-empty string")

    arrow_table = _materialize_arrow_table(data, **kwargs)
    con = connection if connection is not None else duckdb.connect()
    con.register(relation_name, arrow_table)
    return con.table(relation_name)


def duckdb_query(
    data: str | Any | Iterable[Any],
    query: str,
    relation_name: str = "fits_table",
    connection: Any = None,
    return_arrow: bool = True,
    **kwargs,
):
    """
    Execute a DuckDB SQL query over table data.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required for duckdb_query") from exc

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty SQL string")

    con = connection if connection is not None else duckdb.connect()
    _ = to_duckdb(
        data,
        relation_name=relation_name,
        connection=con,
        **kwargs,
    )
    result = con.sql(query)
    if return_arrow:
        return result.arrow()
    return result


def _infer_fits_scalar_code(arr: np.ndarray) -> str:
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


def _infer_fits_format(arr: np.ndarray) -> str:
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


def _prepare_array_for_column(arr: np.ndarray, fmt: str) -> np.ndarray:
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

    if schema or table_kind == "ascii":
        import torchfits.cpp as cpp

        # Overwriting/creating a table can otherwise leave stale cached handles/metadata.
        torchfits._invalidate_path_caches(path)
        data = torchfits._normalize_cpp_table_data(data)
        cpp.write_fits_table(
            path,
            data,
            hdr if hdr else {},
            overwrite,
            schema if schema else None,
            table_kind,
        )
        torchfits._invalidate_path_caches(path)
        return

    torchfits.write(path, data, header=hdr if hdr else None, overwrite=overwrite)


def _header_cards_to_mapping(header_cards: Any) -> dict[str, Any]:
    if isinstance(header_cards, dict):
        return {str(k): v for k, v in header_cards.items()}
    out: dict[str, Any] = {}
    if isinstance(header_cards, (list, tuple)):
        for card in header_cards:
            if not isinstance(card, (list, tuple)) or len(card) < 2:
                continue
            out[str(card[0])] = card[1]
    return out


def _column_tform_map(header_map: dict[str, Any]) -> dict[str, str]:
    name_by_idx: dict[int, str] = {}
    tform_by_idx: dict[int, str] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TTYPE"):
            suffix = key_u[5:]
            if suffix.isdigit():
                name_by_idx[int(suffix)] = str(value)
        elif key_u.startswith("TFORM"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tform_by_idx[int(suffix)] = str(value)

    out: dict[str, str] = {}
    for idx, name in name_by_idx.items():
        out[name] = tform_by_idx.get(idx, "")
    return out


def _column_name_index_map(header_map: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
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
        if any(key_u.startswith(prefix) for prefix in skip_prefixes):
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
    is_vla, code, repeat = _parse_tform(fmt)
    if repeat <= 0:
        repeat = 1

    if is_vla:
        return _coerce_table_vla_values(name, values, code, expected_rows=expected_rows)

    if code == "A":
        return _coerce_table_string_values(name, values, expected_rows=expected_rows)

    if code in _COMPLEX_DTYPE_MAP:
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
    import torchfits.cpp as cpp

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
    import torchfits.cpp as cpp

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
        col_map: dict[int, str] = {}
        for key, value in header_map.items():
            key_u = str(key).upper()
            if not key_u.startswith("TTYPE"):
                continue
            suffix = key_u[5:]
            if suffix.isdigit():
                col_map[int(suffix)] = str(value)
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

    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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

    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
    _rewrite_table_hdu_with_schema(
        path,
        target_hdu,
        rewritten_data,
        rewritten_schema,
        table_header,
        table_type,
    )
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


def _coerce_table_column_array(
    name: str,
    value: Any,
    *,
    expected_rows: Optional[int] = None,
    allow_2d: bool = True,
) -> np.ndarray:
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


_VLA_DTYPE_MAP = {
    "L": np.bool_,
    "B": np.uint8,
    "I": np.int16,
    "J": np.int32,
    "K": np.int64,
    "E": np.float32,
    "D": np.float64,
}

_COMPLEX_DTYPE_MAP = {
    "C": np.complex64,
    "M": np.complex128,
}


def _coerce_table_vla_values(
    name: str,
    value: Any,
    base_code: str,
    *,
    expected_rows: Optional[int] = None,
) -> list[np.ndarray]:
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
) -> np.ndarray:
    base = code.upper()
    if base not in _COMPLEX_DTYPE_MAP:
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
    import torchfits.cpp as cpp

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
    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
    cpp.append_fits_table_rows(path, target_hdu, normalized)
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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
    import torchfits.cpp as cpp

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
    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
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
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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
    import torchfits.cpp as cpp

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

    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
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
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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

    target_hdu, _header, _columns, tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    string_widths: dict[str, int] = {}
    vla_codes: dict[str, str] = {}
    complex_codes: dict[str, str] = {}
    for name, tform in tform_map.items():
        if not tform:
            continue
        is_vla, code, repeat = _parse_tform(tform)
        if is_vla:
            vla_codes[name] = code
        elif code in _COMPLEX_DTYPE_MAP:
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
        return
    if num_rows < 0:
        num_rows = expected_rows
    if expected_rows != num_rows:
        raise ValueError(
            f"row_slice expects {num_rows} rows, but update payload has {expected_rows}"
        )

    import torchfits
    import torchfits.cpp as cpp

    # Ensure no stale cached handles/metadata exist before mutating the file in-place.
    torchfits._invalidate_path_caches(path)

    use_mmap = mmap in (True, "auto", "mmap")
    if use_mmap:
        has_string = any(isinstance(v, (list, tuple)) for v in normalized.values())
        if not has_string:
            try:
                torchfits.clear_cache()
                cpp.update_fits_table_rows_mmap(
                    path, target_hdu, normalized, start_row, num_rows
                )
                torchfits.clear_cache()
                torchfits._invalidate_path_caches(path)
                return
            except Exception:
                if mmap is True:
                    raise

    torchfits.clear_cache()
    cpp.update_fits_table_rows(path, target_hdu, normalized, start_row, num_rows)
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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

    target_hdu, _header, _columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )

    import torchfits
    import torchfits.cpp as cpp

    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
    cpp.rename_fits_table_columns(path, target_hdu, normalized)
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)


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

    target_hdu, _header, _columns, _tform_map = _resolve_table_hdu_index_and_columns(
        path, hdu
    )

    import torchfits
    import torchfits.cpp as cpp

    torchfits._invalidate_path_caches(path)
    torchfits.clear_cache()
    cpp.drop_fits_table_columns(path, target_hdu, normalized)
    torchfits.clear_cache()
    torchfits._invalidate_path_caches(path)
