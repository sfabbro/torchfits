"""Table read path: scans, reads, WHERE filtering, schema inference, torch streaming."""

from __future__ import annotations

from collections.abc import Iterator
import functools
import itertools
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

from .. import fits_schema
from .._table.cache import _acquire_cpp_handle
from .._table.cache import _acquire_cpp_reader
from .._where import parse_where_expression, where_columns_from_ast
from .._table_engine import (
    WhereReadPlan,
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_for_where,
    validate_table_backend,
)
from .._table.utils import _normalize_row_slice, _require_pyarrow
from .._io_engine.paths import guard_fits_path
from .._table.arrow_convert import (
    _chunk_to_record_batch,
    _pa_array,
    _tensor_to_arrow_array,
)
from .._table.write import _resolve_table_hdu_index_and_columns

logger = logging.getLogger(__name__)

# NOTE: ceiling — full-table torch WHERE materializes all selected columns before
# masking; above this row count fall through to chunked Arrow-filter instead (P0-1).
_TORCH_WHERE_MAX_ROWS = 1_000_000


def _column_tform_code_and_repeat(tform: Any) -> tuple[str, int] | None:
    return fits_schema.tform_code_and_repeat(tform)


def _fits_tform_is_bit(tform: Any) -> bool:
    return fits_schema.tform_is_bit(tform)


def _row_slice_from_start_num(start_row: int, num_rows: int) -> Optional[slice]:
    if start_row == 1 and num_rows == -1:
        return None
    start0 = start_row - 1
    if num_rows == -1:
        return slice(start0, None)
    return slice(start0, start0 + num_rows)


def _empty_table_with_schema(
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    decode_bytes: bool,
    include_fits_metadata: bool = False,
) -> Any:
    """Build an empty Arrow table preserving column names/types from the FITS header.

    Falls back to null-typed columns when ``columns`` is set but the header
    cannot be fully typed (VLA / unknown TFORM), or to ``pa.table({})`` when
    neither schema nor column names are available.

    When ``columns`` is provided, the returned table preserves the *requested*
    column order, not the FITS file order.
    """
    header_schema = _schema_from_header(
        path, hdu, columns, decode_bytes, include_fits_metadata
    )
    if header_schema is not None:
        # Reorder fields to match the requested column order when specified.
        if columns is not None:
            ordered_fields = []
            for name in columns:
                idx = header_schema.get_field_index(name)
                if idx >= 0:
                    ordered_fields.append(header_schema.field(idx))
            if ordered_fields:
                ordered_schema = pa.schema(
                    ordered_fields,
                    metadata=header_schema.metadata,
                )
                return pa.Table.from_arrays(
                    [pa.array([], type=f.type) for f in ordered_schema],
                    schema=ordered_schema,
                )
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in header_schema],
            schema=header_schema,
        )
    # NOTE: VLA / unknown TFORM — keep requested names as null columns;
    # upgrade path is a typed scan-based empty schema when VLA decode is cheap.
    if columns:
        null_schema = pa.schema([pa.field(name, pa.null()) for name in columns])
        return pa.Table.from_arrays(
            [pa.array([], type=pa.null()) for _ in columns],
            schema=null_schema,
        )
    return pa.table({})


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
    except (RuntimeError, OSError, MemoryError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(chunk, dict) or not chunk:
        return None

    mask: torch.Tensor | None = None
    try:
        for pred_col, op, literal in predicates:
            tensor = chunk.get(pred_col)
            if not isinstance(tensor, torch.Tensor):
                return None
            part = _torch_cmp_mask(tensor, op, literal)
            mask = part if mask is None else (mask & part)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("torch WHERE mask build failed; falling back: %s", exc)
        return None
    if mask is None:
        return None

    tnull_map = (
        fits_schema.column_tnull_map(header)
        if apply_fits_nulls and header is not None
        else {}
    )

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

    def _in_mask(column_name: str, literals: list[Any], negate: bool) -> Any:
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

    def _between_mask(column_name: str, low: Any, high: Any, negate: bool) -> Any:
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
            child = pc.fill_null(_eval(node[1]), False)
            return pc.invert(child)
        raise ValueError("Invalid where AST")

    return pc.fill_null(_eval(ast), False)  # type: ignore[no-any-return]


def _build_fits_metadata(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]] = None,
    header: Any = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if header is None:
        import torchfits

        header = torchfits.read_header(path, hdu)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {
        "fits_hdu": str(hdu),
    }

    try:
        tf_count = int(header.get("TFIELDS", 0))
    except (TypeError, ValueError):
        tf_count = 0

    for i in range(1, tf_count + 1):
        si = str(i)
        name = header.get("TTYPE" + si)
        if not isinstance(name, str) or not name:
            continue
        if selected_columns is not None and name not in selected_columns:
            continue

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


def _column_tforms_for_decode(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
    header: Any = None,
) -> dict[str, str]:
    """Delegates to fits_schema for TFORM lookup."""
    if header is None:
        import torchfits

        try:
            header = torchfits.read_header(path, hdu)
        except (OSError, ValueError):
            return {}
    out: dict[str, str] = {}
    for col in fits_schema.iter_table_columns(header, selected=selected_columns):
        out[col.name] = col.tform
    return out


def _unsigned_column_dtypes(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
    header: Any = None,
) -> dict[str, str]:
    """Delegates to fits_schema.unsigned_column_dtypes_from_header."""
    if header is None:
        import torchfits

        try:
            header = torchfits.read_header(path, hdu)
        except (OSError, ValueError):
            return {}
    torch_dtype_map = fits_schema.unsigned_column_dtypes_from_header(header)
    return {
        col: str(dt).split(".")[-1]
        for col, dt in torch_dtype_map.items()
        if selected_columns is None or col in selected_columns
    }


_FULL_READ_SUPPORTED_CODES = frozenset({"L", "B", "I", "J", "K", "E", "D"})


def _can_use_full_read_path(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    *,
    reject_scaled: bool,
    header: Any = None,
) -> bool:
    """Whether all selected columns are scalar (repeat==1) rows of supported codes.

    ``reject_scaled`` additionally bails out on TSCAL/TZERO columns (needed by the
    raw mmap row path, which cannot apply scaling; the torch table path can).
    """
    if header is None:
        import torchfits

        try:
            header = torchfits.read_header(path, hdu)
        except (OSError, ValueError):
            return False
    try:
        tf_count = int(header.get("TFIELDS", 0))
    except (TypeError, ValueError):
        return False
    if tf_count <= 0:
        return False

    selected = set(columns) if columns else None
    any_selected = False

    for i in range(1, tf_count + 1):
        si = str(i)
        name = header.get("TTYPE" + si)
        if not isinstance(name, str) or not name:
            continue
        if selected is not None and name not in selected:
            continue
        any_selected = True

        if reject_scaled and (
            header.get("TSCAL" + si) is not None or header.get("TZERO" + si) is not None
        ):
            return False

        parsed = _column_tform_code_and_repeat(header.get("TFORM" + si))
        if parsed is None:
            return False
        code, repeat = parsed
        if code not in _FULL_READ_SUPPORTED_CODES:
            return False
        if repeat != 1:
            return False

    return any_selected


def _can_use_mmap_row_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
    header: Any = None,
) -> bool:
    return _can_use_full_read_path(
        path, hdu, selected_columns, reject_scaled=True, header=header
    )


def _can_use_torch_table_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
    header: Any = None,
) -> bool:
    return _can_use_full_read_path(
        path, hdu, selected_columns, reject_scaled=False, header=header
    )


def _iter_chunks_cpp_table(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    batch_size: int,
    mmap: bool,
) -> Any:
    import torchfits
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_rows_from_handle"):
        return None

    header = torchfits.read_header(path, hdu)
    total_rows = header.get("NAXIS2", 0)
    try:
        total_rows = (
            int(float(total_rows)) if isinstance(total_rows, str) else int(total_rows)
        )
    except (TypeError, ValueError):
        total_rows = 0
    if total_rows <= 0:
        return iter(())

    end_row = (
        total_rows if num_rows == -1 else min(total_rows, start_row + num_rows - 1)
    )
    col_list = columns if columns else []

    def _generator() -> Any:
        can_mmap_rows = mmap and hasattr(cpp, "read_fits_table_rows")
        if can_mmap_rows:
            can_mmap_rows = _can_use_mmap_row_path_for_full_read(
                path, hdu, columns, header=header
            )
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
                    except (RuntimeError, OSError) as exc:
                        logger.debug(
                            "mmap row read failed; falling back to handle read: %s", exc
                        )
                        can_mmap_rows = False

                if file_handle is None:
                    file_handle = cpp.open_fits_file(path, "r")
                yield cpp.read_fits_table_rows_from_handle(
                    file_handle, hdu, col_list, row, size
                )
                row += size
        finally:
            if file_handle is not None:
                file_handle.close()

    return _generator()


def _filter_table_with_where(pa: Any, table: Any, where: str) -> Any:
    # table.filter preserves schema for empty (all-false) masks; skip a sum(mask) pass.
    mask = _where_mask_for_table(table, where)
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
        return _empty_table_with_schema(
            pa, path, hdu, columns, decode_bytes, include_fits_metadata
        )
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
    if backend in {"auto", "cpp"}:
        single = _read_cpp_table_chunk(
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
        tnull_map = (
            fits_schema.column_tnull_map(header)
            if apply_fits_nulls and header is not None
            else {}
        )
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
    predicate_table = _read_cpp_table_chunk(
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
    import pyarrow.compute as _pc

    pc: Any = _pc

    mask = _where_mask_for_table(predicate_table, where, parsed_ast=where_ast)
    if len(mask) == 0 or pc.sum(mask).as_py() == 0:
        return []

    base_row0 = start_row - 1
    selected = pc.indices_nonzero(mask).to_numpy()
    if selected.size == 0:
        return []
    return (selected + base_row0).tolist()  # type: ignore[no-any-return]


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
    guard_fits_path(path)
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    backend = validate_table_backend(backend)

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

    # Read the header once and pass it to all helper functions to avoid
    # redundant read_header() calls.
    try:
        _hdr = torchfits.read_header(path, hdu)
    except (OSError, ValueError):
        _hdr = None

    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected, header=_hdr)
        if decode_bytes
        else None
    )
    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected, header=_hdr)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(
                path, hdu, selected, header=_hdr
            )
        except (OSError, ValueError):
            field_meta, table_meta = {}, {}
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    chunk_iter = None
    if backend in {"auto", "cpp"}:
        chunk_iter = _iter_chunks_cpp_table(
            path, hdu, columns, start_row, num_rows, batch_size, mmap
        )
    if chunk_iter is None or backend == "torch":
        # Lazy import: table_streaming → hdu at module import time is cyclic.
        from .._io_engine.table_streaming import stream_table as _engine_stream_table

        chunk_iter = _engine_stream_table(
            torchfits.read_header,
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
) -> Any:
    guard_fits_path(path)
    backend = validate_table_backend(backend)
    pa = _require_pyarrow()
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if backend in {"auto", "cpp"} and not should_skip_cpp_for_where(backend, where):
        single = _read_cpp_table_chunk(
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


def read_torch(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    device: str = "cpu",
    mmap: bool | str = "auto",
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
    where: str | None = None,
) -> Any:
    """Read a FITS table as dataframe columns mapped to ``torch.Tensor`` values.

    Prefer this ``table.read_torch`` entry point for new code. For Arrow
    dataframes use :func:`read` / :func:`read_arrow`.

    ``hdu`` is an index or EXTNAME (not ``None`` / ``\"auto\"``). Optional
    ``where`` uses C++ ``read_fits_table_filtered`` for simple numeric
    predicates (same dialect as ``table.read``).
    """
    guard_fits_path(path)
    import torchfits

    # Lazy import: table_api path must not pull hdu during _table.read import.
    from .._io_engine.table_api import read_table as _engine_read_table

    return _engine_read_table(
        torchfits.read,
        path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        device=device,
        mmap=mmap,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        where=where,
    )


def _arrow_type_from_tform(
    code: str, repeat: int, *, decode_bytes: bool, pa: Any
) -> Any | None:
    """Map a scalar FITS TFORM code + repeat to a pyarrow type, or None if unhandled.

    Bit columns (X) map to bool_(); for repeat > 1 the result is a
    FixedSizeList of bools, matching what the data path produces via
    ``_uint8_matrix_to_fixed_bool_list``.
    """
    _SCALAR: dict[str, Any] = {
        "L": pa.bool_(),
        "X": pa.bool_(),
        "B": pa.uint8(),
        "I": pa.int16(),
        "J": pa.int32(),
        "K": pa.int64(),
        "E": pa.float32(),
        "D": pa.float64(),
        "C": pa.float64(),
        "M": pa.float64(),
        "A": pa.utf8() if decode_bytes else pa.binary(),
    }
    base = _SCALAR.get(code)
    if base is None:
        return None
    if repeat == 1:
        return base
    return pa.list_(base, repeat)


def _schema_from_header(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    decode_bytes: bool,
    include_fits_metadata: bool,
) -> Any | None:
    """Build a pyarrow schema from FITS header cards only (no data rows read).

    Returns ``None`` when the header cannot be read, or when VLA columns,
    complex types, or unknown TFORM codes are present — callers must fall
    back to the scan-based schema path in those cases.
    """
    import torchfits

    try:
        header = torchfits.read_header(path, hdu)
    except (OSError, ValueError):
        return None

    pa = _require_pyarrow()
    selected = set(columns) if columns else None
    fields = []
    any_vla = False

    table_meta: dict[str, str] = {"fits_hdu": str(hdu)}

    for col in fits_schema.iter_table_columns(header, selected=selected):
        info = col.tform_info
        if info.vla or info.code is None:
            any_vla = True
            continue
        arrow_type = _arrow_type_from_tform(
            info.code, info.repeat, decode_bytes=decode_bytes, pa=pa
        )
        if arrow_type is None:
            any_vla = True
            continue

        metadata = None
        if include_fits_metadata:
            meta: dict[bytes, bytes] = {}
            if col.tform:
                meta[b"fits_tform"] = col.tform.encode("utf-8")
            if col.tdim is not None:
                meta[b"fits_tdim"] = col.tdim.encode("utf-8")
            if col.tnull is not None:
                meta[b"fits_tnull"] = str(col.tnull).encode("utf-8")
            if meta:
                metadata = meta

        fields.append(pa.field(col.name, arrow_type, metadata=metadata))

    if not fields and not any_vla:
        return pa.schema([], metadata=table_meta if include_fits_metadata else None)
    if any_vla:
        return None
    return pa.schema(fields, metadata=table_meta if include_fits_metadata else None)


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
) -> Any:
    pa = _require_pyarrow()
    backend = validate_table_backend(backend)
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    # Fast path: when no WHERE filter, infer schema from header cards only.
    if where is None:
        header_schema = _schema_from_header(
            path, hdu, columns, decode_bytes, include_fits_metadata
        )
        if header_schema is not None:
            return header_schema

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


def _read_cpp_table_chunk(
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
) -> Any:
    """Read a table chunk via C++ TableReader (torch tensors) and convert to Arrow."""
    import numpy as np
    import torchfits._C as cpp

    if rows is not None and row_slice is not None:
        raise ValueError("Only one of rows or row_slice may be provided")

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        pa = _require_pyarrow()
        return _empty_table_with_schema(
            pa, path, hdu, columns, decode_bytes, include_fits_metadata
        )

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

    # Read the header lazily and at most once, reusing it across every helper.
    # Deferring the read lets a read that fails or returns empty before any
    # header consumer runs skip the header I/O entirely — the common numeric
    # path (no decode/metadata/nulls) only touches the header via the
    # full-read capability check (P2-1).
    import torchfits

    _hdr_memo: list[Any] = []

    def _get_hdr() -> Any:
        if not _hdr_memo:
            try:
                _hdr_memo.append(torchfits.read_header(path, hdu))
            except (OSError, ValueError):
                _hdr_memo.append(None)
        return _hdr_memo[0]

    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected, header=_get_hdr())
        if decode_bytes
        else None
    )
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(
                path, hdu, selected, header=_get_hdr()
            )
        except (OSError, ValueError):
            pass
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    col_list = columns if columns else []

    from .engine import _read_ranges_as_chunk

    chunk = None
    prefer_torch_full_path = (
        start_row == 1
        and num_rows == -1
        and not decode_bytes
        and not include_fits_metadata
        and not apply_fits_nulls
        and _can_use_torch_table_path_for_full_read(
            path, hdu, columns, header=_get_hdr()
        )
    )
    if prefer_torch_full_path:
        if mmap and _can_use_mmap_row_path_for_full_read(
            path, hdu, columns, header=_get_hdr()
        ):
            try:
                chunk = cpp.read_fits_table(path, hdu, col_list, True)
            except (RuntimeError, OSError) as exc:
                logger.debug("mmap full table read failed; retrying: %s", exc)
                chunk = None
        if chunk is None:
            try:
                if not col_list:
                    file_handle = _acquire_cpp_handle(path, cpp)
                    try:
                        chunk = cpp.read_fits_table_from_handle(file_handle, hdu)
                    finally:
                        try:
                            file_handle.close()
                        except (RuntimeError, OSError):
                            pass
                else:
                    chunk = cpp.read_fits_table(path, hdu, col_list, False)
            except (RuntimeError, OSError) as exc:
                logger.debug("full table read failed; falling back: %s", exc)
                chunk = None

    if chunk is None and rows is not None:
        rows_arr = np.asarray(rows, dtype=np.int64)
        if rows_arr.size == 0:
            pa = _require_pyarrow()
            return _empty_table_with_schema(
                pa, path, hdu, columns, decode_bytes, include_fits_metadata
            )
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
            chunk_sorted = _read_ranges_as_chunk(reader, col_list, ranges)
        except (RuntimeError, OSError) as exc:
            logger.debug("ranged row read failed: %s", exc)
            chunk_sorted = None
        if chunk_sorted is None:
            return None

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        chunk = {}
        for name, value in chunk_sorted.items():
            if isinstance(value, torch.Tensor):
                chunk[name] = value[inv]
            elif isinstance(value, np.ndarray):
                chunk[name] = value[inv]
            elif isinstance(value, list):
                chunk[name] = [value[i] for i in inv]
            else:
                chunk[name] = value

    if chunk is None:
        try:
            if mmap:
                chunk = cpp.read_fits_table_rows(
                    path, hdu, col_list, start_row, num_rows, True
                )
            else:
                reader = _acquire_cpp_reader(path, hdu, cpp)
                chunk = reader.read_rows(col_list, start_row, num_rows)
        except (RuntimeError, OSError) as exc:
            logger.debug("row-slice table read failed: %s", exc)
            chunk = None
    if chunk is None:
        return None

    pa = _require_pyarrow()
    if not chunk:
        return _empty_table_with_schema(
            pa, path, hdu, columns, decode_bytes, include_fits_metadata
        )

    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected, header=_get_hdr())
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
    guard_fits_path(path)
    import torchfits

    start_row, num_rows = _normalize_row_slice(row_slice)
    use_mmap = mmap
    _hdr = None
    if use_mmap:
        # Read header once for the capability check.
        try:
            _hdr = torchfits.read_header(path, hdu)
        except (OSError, ValueError):
            _hdr = None
        use_mmap = _can_use_mmap_row_path_for_full_read(path, hdu, columns, header=_hdr)

    # Reuse the already-read header's NAXIS2 so stream_table does not re-read it.
    total_rows: Optional[int] = None
    if _hdr is not None:
        try:
            _nr = _hdr.get("NAXIS2", 0)
            total_rows = int(float(_nr)) if isinstance(_nr, str) else int(_nr)
        except (TypeError, ValueError):
            total_rows = None

    # Lazy import: table_streaming → hdu at module import time is cyclic.
    from .._io_engine.table_streaming import stream_table as _engine_stream_table

    for chunk in _engine_stream_table(
        torchfits.read_header,
        path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        chunk_rows=batch_size,
        mmap=use_mmap,
        total_rows=total_rows,
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
) -> Any:
    pa = _require_pyarrow()
    backend = validate_table_backend(backend)
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
    **kwargs: Any,
) -> Any:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for dataset conversion") from exc

    if isinstance(data, str):
        # In older pyarrow versions, ds.dataset() does not accept RecordBatchReader directly.
        # We read all batches into a Table first.
        return ds.dataset(reader(data, **kwargs).read_all())  # type: ignore[no-untyped-call]
    return ds.dataset(data)  # type: ignore[no-untyped-call]


def scanner(
    data: str | Any,
    *,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    filter: Any = None,
    batch_size: int = 65536,
    use_threads: bool = True,
    **kwargs: Any,
) -> Any:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for scanner") from exc

    if where is not None:
        kwargs = dict(kwargs)
        kwargs["where"] = where

    if isinstance(data, str):
        rdr = reader(data, **kwargs)
        return ds.Scanner.from_batches(  # type: ignore[attr-defined]
            rdr,
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            use_threads=use_threads,
        )
    elif hasattr(data, "scanner"):
        dset = data
    else:
        dset = ds.dataset(data)  # type: ignore[no-untyped-call]
    return dset.scanner(
        columns=columns, filter=filter, batch_size=batch_size, use_threads=use_threads
    )
