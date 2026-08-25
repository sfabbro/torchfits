"""WHERE filtering helpers for table reads."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

from .. import fits_schema
from .._table.cache import _acquire_cpp_reader
from .._where import parse_where_expression, where_columns_from_ast
from .._table_engine import (
    WhereReadPlan,
    WhereStrategy,
    choose_where_read_plan,
)
from .._table.utils import _require_pyarrow
from .._table.arrow_convert import (
    _pa_array,
    _tensor_to_arrow_array,
)
from ._read_schema import (
    _can_use_mmap_row_path_for_full_read,
    _column_tforms_for_decode,
    _empty_table_with_schema,
    schema,
)


logger = logging.getLogger(__name__)

# NOTE: ceiling — full-table torch WHERE materializes all selected columns before
# masking; above this row count fall through to chunked Arrow-filter instead (P0-1).
_TORCH_WHERE_MAX_ROWS = 1_000_000


@functools.lru_cache(maxsize=128)
def _compile_where_to_simple_predicates(
    where: str,
) -> Optional[tuple[tuple[str, str, Any], ...]]:
    """Parse a where string into simple predicates (cached).

    Returns a tuple of (col, op, literal) triples, or None if the where
    clause cannot be reduced to simple predicates.  The tuple is immutable
    so the cached value cannot be corrupted by callers.
    """
    try:
        ast = parse_where_expression(where)
    except (ValueError, TypeError, KeyError, RuntimeError):
        return None

    predicates: list[tuple[str, str, Any]] = []

    def _visit(node: Any) -> bool:
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
    return tuple(predicates)


def _torch_cmp_mask(tensor: torch.Tensor, op: str, literal: Any) -> torch.Tensor:
    # PyTorch wraps an out-of-range Python int scalar to the tensor's dtype
    # (e.g. comparing an int16 column against 40000 becomes `> -25536`), which
    # silently flips the predicate. Promote the tensor to int64 so the literal
    # is compared at full width instead (matching the C++ pushdown, which now
    # also compares integers in int64).
    if (
        isinstance(literal, int)
        and not isinstance(literal, bool)
        and tensor.dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.uint8,
            torch.uint16,
            torch.uint32,
        )
    ):
        info = torch.iinfo(tensor.dtype)
        if not (info.min <= literal <= info.max):
            tensor = tensor.to(torch.int64)

    if op == "==":
        return torch.eq(tensor, literal)
    if op == "!=":
        return torch.ne(tensor, literal)
    if op == ">":
        return torch.gt(tensor, literal)
    if op == ">=":
        return torch.ge(tensor, literal)
    if op == "<":
        return torch.lt(tensor, literal)
    if op == "<=":
        return torch.le(tensor, literal)
    raise ValueError(f"Unsupported where operator '{op}'")


def _try_torch_tensor_where_filter(
    *,
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    header: Any | None,
    apply_fits_nulls: bool = False,
) -> Any | None:
    """Buffered/mmap tensor read + torch mask + Arrow for simple numeric WHERE.

    Honors ``mmap`` via the existing C++ row readers (no secret mmap when
    ``mmap=False``). Skips Arrow conversion of the full column before filtering.
    """
    if row_slice is not None or rows is not None:
        return None
    predicates = _compile_where_to_simple_predicates(where)
    if predicates is None:
        return None

    import torchfits._C as cpp

    output_cols = columns
    if output_cols is None:
        if header is not None:
            output_cols = [col.name for col in fits_schema.iter_table_columns(header)]
        else:
            return None

    # decode_bytes only matters for string/bit columns; numeric WHERE paths stay eligible.
    if decode_bytes and header is not None:
        selected = set(output_cols)
        for col in fits_schema.iter_table_columns(header, selected=selected):
            if col.tform_info.is_string or col.tform_info.is_bit:
                return None
    elif decode_bytes and header is None:
        return None

    pred_cols = [col for col, _op, _lit in predicates]
    read_cols: list[str] = []
    seen: set[str] = set()
    for name in list(output_cols) + pred_cols:
        if name not in seen:
            read_cols.append(name)
            seen.add(name)

    n_rows = 0
    if header is not None:
        try:
            n_rows = int(header.get("NAXIS2", 0) or 0)
        except (TypeError, ValueError):
            n_rows = 0
    if n_rows <= 0:
        try:
            n_rows = int(cpp.read_nrows(path, hdu))
        except (AttributeError, TypeError, RuntimeError, OSError, ValueError):
            n_rows = 0
    if n_rows > _TORCH_WHERE_MAX_ROWS:
        return None

    try:
        if mmap and _can_use_mmap_row_path_for_full_read(
            path, hdu, read_cols, header=header
        ):
            chunk = cpp.read_fits_table(path, hdu, read_cols, True)
        else:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk = reader.read_rows(read_cols, 1, -1)
    except (
        RuntimeError,
        OSError,
        MemoryError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        # A truncated file must surface as corruption, not as a silent
        # fallback to a slower engine that fails again downstream.
        if "truncat" in str(exc).lower():
            raise
        return None
    if not isinstance(chunk, dict) or not chunk:
        return None

    # Sentinel→null exclusion must happen BEFORE predicate evaluation so the
    # torch engine agrees with the Arrow engine (comparisons against NULL
    # never match) regardless of which strategy runs.
    tnull_map = (
        fits_schema.column_tnull_map(header)
        if apply_fits_nulls and header is not None
        else {}
    )

    mask: torch.Tensor | None = None
    try:
        for pred_col, op, literal in predicates:
            tensor = chunk.get(pred_col)
            if not isinstance(tensor, torch.Tensor):
                return None
            part = _torch_cmp_mask(tensor, op, literal)
            sentinel = tnull_map.get(pred_col)
            if sentinel is not None:
                part = part & torch.ne(tensor, sentinel)
            mask = part if mask is None else (mask & part)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("torch WHERE mask build failed; falling back: %s", exc)
        return None
    if mask is None:
        return None

    arrays = []
    names_out = []
    for name in output_cols:
        value = chunk.get(name)
        if not isinstance(value, torch.Tensor):
            return None
        filtered = value[mask]
        arrays.append(
            _tensor_to_arrow_array(
                pa,
                filtered,
                decode_bytes,
                encoding,
                strip,
                null_sentinel=tnull_map.get(name),
                fits_tform=None,
            )
        )
        names_out.append(name)

    if not arrays:
        # Preserve projected schema for empty output.
        empty = []
        for name in output_cols:
            value = chunk.get(name)
            if isinstance(value, torch.Tensor):
                empty.append(
                    _tensor_to_arrow_array(
                        pa,
                        value[:0],
                        decode_bytes,
                        encoding,
                        strip,
                        null_sentinel=tnull_map.get(name),
                        fits_tform=None,
                    )
                )
            else:
                # Non-tensor column (string/bit) — fall back to header schema.
                return _empty_table_with_schema(
                    pa, path, hdu, output_cols, decode_bytes
                )
        return pa.Table.from_arrays(empty, names=list(output_cols))
    return pa.Table.from_arrays(arrays, names=names_out)


def _aligned_scalar(pa: Any, column: Any, value: Any) -> Any:
    """Build an Arrow scalar that matches the C++ pushdown comparison rule.

    Both engines must select identical rows for a given predicate. The C++
    mmap-filtered scan casts the literal to the column's storage precision
    before comparing, so the Arrow engine must do the same for float32
    columns instead of comparing against a float64 scalar (which would drop
    rows whose stored float32 equals the literal). Wider/integer columns keep
    Arrow's exact widening.
    """
    if pa.types.is_float32(column.type):
        return pa.scalar(float(value), type=column.type)
    return pa.scalar(value)


def _where_mask_for_table(
    table: Any, where: str, parsed_ast: Any = None
) -> "np.ndarray":
    pa = _require_pyarrow()
    import pyarrow.compute as _pc

    pc: Any = _pc

    ast = parsed_ast if parsed_ast is not None else parse_where_expression(where)

    def _get_predicate_column(column_name: str) -> Any:
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

    def _cmp_mask(column_name: str, op: str, literal: Any) -> Any:
        column = _get_predicate_column(column_name)

        if literal is None:
            if op == "==":
                return pc.is_null(column)
            if op == "!=":
                return pc.invert(pc.is_null(column))
            raise ValueError("where comparisons with null only support == and !=")

        scalar = _aligned_scalar(pa, column, literal)
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

    def _in_mask(column_name: str, literals: list[Any], negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        non_null = [v for v in literals if v is not None]
        has_null = any(v is None for v in literals)

        if non_null:
            if pa.types.is_float32(column.type):
                value_set = pa.array([float(v) for v in non_null], type=column.type)
            else:
                value_set = _pa_array(pa, non_null)
            mask = pc.is_in(column, value_set=value_set)
        else:
            mask = _pa_array(pa, [False] * int(len(column)))

        if has_null:
            mask = pc.or_(pc.fill_null(mask, False), pc.is_null(column))
        if negate:
            # Invert BEFORE null-fill so NULL rows stay excluded on negation
            # (SQL three-valued logic): NOT IN must not resurrect NULLs.
            return pc.invert(mask)
        return pc.fill_null(mask, False)

    def _between_mask(column_name: str, low: Any, high: Any, negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        if low is None or high is None:
            raise ValueError("where BETWEEN does not support NULL bounds")
        low_s = _aligned_scalar(pa, column, low)
        high_s = _aligned_scalar(pa, column, high)
        ge = pc.greater_equal(column, low_s)
        le = pc.less_equal(column, high_s)
        mask = pc.and_(ge, le)
        if negate:
            # Null-propagating inversion: NOT BETWEEN excludes NULLs.
            return pc.invert(mask)
        return pc.fill_null(mask, False)

    def _isnull_mask(column_name: str, negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        mask = pc.is_null(column)
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _eval(node: Any) -> Any:
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
            child = _eval(node[1])
            # Invert before the null-fill: NOT must keep NULL rows excluded
            # (NOT (X == 5) ≡ X != 5 under SQL three-valued logic).
            return pc.fill_null(pc.invert(child), False)
        raise ValueError("Invalid where AST")

    return pc.fill_null(_eval(ast), False)  # type: ignore[no-any-return]


def _filter_table_with_where(pa: Any, table: Any, where: str) -> Any:
    # table.filter preserves schema for empty (all-false) masks; skip a sum(mask) pass.
    mask = _where_mask_for_table(table, where)
    return table.filter(mask)


def _try_cpp_where_pushdown(
    *,
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    header: Any = None,
    apply_fits_nulls: bool = False,
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
            if header is not None:
                target_cols = [
                    col.name for col in fits_schema.iter_table_columns(header)
                ]
            else:
                target_cols = list(schema(path, hdu=hdu, backend="cpp").names)

        # filters is a tuple (immutable, cached) — C++ binding expects a list.
        data_dict = cpp.read_fits_table_filtered(path, hdu, target_cols, list(filters))

        # The C++ pushdown compares raw stored values, so TNULL sentinels can
        # satisfy numeric predicates (e.g. sentinel 25 matching "> 20") while
        # the Arrow engine treats them as NULL and excludes them. Drop rows
        # whose match came only from a sentinel so both engines agree.
        full_tnull_map = (
            fits_schema.column_tnull_map(header)
            if apply_fits_nulls and header is not None
            else {}
        )
        keep: Any = None
        for pred_col, _op, _lit in filters:
            sentinel = full_tnull_map.get(pred_col)
            if sentinel is None:
                continue
            col_tensor = data_dict.get(pred_col)
            if not isinstance(col_tensor, torch.Tensor):
                # Predicate column not in the projection: cannot verify
                # sentinel matches here — defer to the Arrow engine.
                return None
            valid = torch.ne(col_tensor, sentinel)
            keep = valid if keep is None else (keep & valid)
        if keep is not None:
            if bool(keep.any()):
                data_dict = {
                    key: (
                        value[keep]
                        if isinstance(value, torch.Tensor)
                        and value.shape[:1] == keep.shape[:1]
                        else value
                    )
                    for key, value in data_dict.items()
                }
            else:
                data_dict = {
                    key: (
                        value[:0]
                        if isinstance(value, torch.Tensor)
                        else ([] if isinstance(value, list) else value)
                    )
                    for key, value in data_dict.items()
                }

        # Only look up tforms when string/bit columns are present
        # (numeric 1D columns don't need tform for Arrow conversion).
        pushdown_tforms = None
        if decode_bytes:
            needs_tforms = header is None
            if not needs_tforms:
                for col in fits_schema.iter_table_columns(
                    header, selected=set(target_cols) if target_cols else None
                ):
                    if col.tform_info.is_string or col.tform_info.is_bit:
                        needs_tforms = True
                        break
            if needs_tforms:
                pushdown_tforms = _column_tforms_for_decode(path, hdu, set(target_cols))
        tnull_map = full_tnull_map
        arrays = []
        names_out = []
        for name in target_cols:
            if name not in data_dict:
                continue
            val = data_dict[name]
            if isinstance(val, torch.Tensor):
                arr = _tensor_to_arrow_array(
                    pa,
                    val,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=tnull_map.get(name),
                    fits_tform=pushdown_tforms.get(name) if pushdown_tforms else None,
                )
                arrays.append(arr)
                names_out.append(name)

        if not arrays:
            return _empty_table_with_schema(
                pa, path, hdu, columns, decode_bytes, include_fits_metadata=False
            )
        return pa.Table.from_arrays(arrays, names=names_out)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.debug("CPP WHERE pushdown read failed; falling back: %s", exc)
        return None


def _read_table_with_where(
    *,
    pa: Any,
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
    hdr: Any = {}
    n_rows = 0
    try:
        hdr = torchfits.read_header(path, hdu)
        n_rows = int(hdr.get("NAXIS2", 0))
        header_ok = True
    except (OSError, ValueError, TypeError):
        n_rows = 0

    plan = (
        choose_where_read_plan(
            header=hdr,
            header_ok=header_ok,
            columns=columns,
            backend=backend,
            n_rows=n_rows,
            mmap=mmap,
        )
        if header_ok
        else WhereReadPlan(
            strategy=WhereStrategy.ARROW_FILTER,
            cpp_pushdown_safe=False,
            unfiltered_backend=backend,
        )
    )

    if plan.strategy == WhereStrategy.CPP_PUSHDOWN:
        # The C++ pushdown filters the whole HDU and cannot express a row
        # window; using it here would silently ignore row_slice/rows. Only
        # take this shortcut for full-table predicates so the fallback below
        # can apply the window BEFORE filtering (filter-within-window).
        if row_slice is None and rows is None:
            pushed = _try_cpp_where_pushdown(
                pa=pa,
                path=path,
                hdu=hdu,
                columns=columns,
                where=where,
                decode_bytes=decode_bytes,
                encoding=encoding,
                strip=strip,
                header=hdr if header_ok else None,
                apply_fits_nulls=apply_fits_nulls,
            )
            if pushed is not None:
                return pushed

    torch_filtered = _try_torch_tensor_where_filter(
        pa=pa,
        path=path,
        hdu=hdu,
        columns=columns,
        where=where,
        row_slice=row_slice,
        rows=rows,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        header=hdr if header_ok else None,
        apply_fits_nulls=apply_fits_nulls,
    )
    if torch_filtered is not None:
        return torch_filtered

    # Read output ∪ predicate columns so WHERE can reference unprojected columns,
    # then drop hidden columns after filtering.
    read_columns = columns
    drop_after: list[str] = []
    if columns is not None:
        try:
            where_cols = where_columns_from_ast(parse_where_expression(where))
        except ValueError:
            where_cols = []
        if where_cols:
            seen = set(columns)
            merged = list(columns)
            for name in where_cols:
                if name not in seen:
                    merged.append(name)
                    drop_after.append(name)
                    seen.add(name)
            read_columns = merged

    # Lazy import: _read_scan depends on _read_where at call time only.
    from ._read_scan import _read_table_unfiltered  # noqa: F811

    base = _read_table_unfiltered(
        path=path,
        hdu=hdu,
        columns=read_columns,
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
    filtered = _filter_table_with_where(pa, base, where)
    if drop_after:
        keep = [name for name in filtered.column_names if name not in set(drop_after)]
        return filtered.select(keep)
    return filtered
