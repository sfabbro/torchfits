"""Schema inference and full-read capability checks for table reads."""

from __future__ import annotations

from typing import Any, Optional

from .. import fits_schema
from .._table.utils import _require_pyarrow
from .._table.write import _resolve_table_hdu_index_and_columns
from .._table_engine import validate_table_backend


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
    from . import read as _read_mod

    iterator = _read_mod.scan(
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
