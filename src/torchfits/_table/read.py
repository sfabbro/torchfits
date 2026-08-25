"""Table read path: scans, reads, WHERE filtering, schema inference, torch streaming."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any, Optional

from .._io_engine.paths import guard_fits_path
from .._table.utils import _require_pyarrow
from .._table.write import _resolve_table_hdu_index_and_columns
from .._table_engine import should_skip_cpp_for_where, validate_table_backend

from ._read_scan import (
    _read_cpp_table_chunk,
    _read_table_from_scan_batches,
    _scan_iter,
    _scan_torch_iter,
)
from ._read_schema import schema
from ._read_where import (
    _compile_where_to_simple_predicates,
    _read_table_with_where,
    _where_mask_for_table,
)

# Re-export private helpers for backward-compatible imports from this module.
from ._read_schema import (  # noqa: F401
    _arrow_type_from_tform,
    _build_fits_metadata,
    _can_use_full_read_path,
    _can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read,
    _column_tform_code_and_repeat,
    _column_tforms_for_decode,
    _empty_table_with_schema,
    _fits_tform_is_bit,
    _row_slice_from_start_num,
    _schema_from_header,
    _unsigned_column_dtypes,
)
from ._read_where import (  # noqa: F401
    _TORCH_WHERE_MAX_ROWS,
    _filter_table_with_where,
    _torch_cmp_mask,
    _try_cpp_where_pushdown,
    _try_torch_tensor_where_filter,
)
from ._read_scan import (  # noqa: F401
    _iter_chunks_cpp_table,
    _read_table_unfiltered,
    _resolve_rows_from_where_cpp,
)


_COMPLEX_ERR = (
    "complex table columns (TFORM 'C'/'M') are not supported by the Arrow "
    "table APIs (pyarrow has no complex type); use torchfits.read_torch() or "
    "torchfits.read(..., mode='table') for torch complex tensors"
)


def _reject_complex_columns(path: str, hdu: int | str) -> None:
    """Raise a clear error when the target HDU holds complex columns."""
    import torchfits as _tf
    from ..fits_schema import complex_column_names

    try:
        header = _tf.read_header(path, hdu)
    except Exception:
        return
    names = sorted(complex_column_names(header))
    if names:
        raise NotImplementedError(f"{_COMPLEX_ERR} (columns: {names})")


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
    # Eager guard: a generator body would defer this until first next().
    guard_fits_path(path)
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]
    _reject_complex_columns(path, hdu)
    return _scan_iter(
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
        backend=backend,
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
    _reject_complex_columns(path, hdu)

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
    # Eager guard: a generator body would defer this until first next().
    guard_fits_path(path)
    return _scan_torch_iter(
        path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        batch_size=batch_size,
        mmap=mmap,
        device=device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
    )


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
        return ds.dataset(reader(data, **kwargs).read_all())
    return ds.dataset(data)


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
        return ds.Scanner.from_batches(
            rdr,
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            use_threads=use_threads,
        )
    elif hasattr(data, "scanner"):
        dset = data
    else:
        dset = ds.dataset(data)
    return dset.scanner(
        columns=columns, filter=filter, batch_size=batch_size, use_threads=use_threads
    )


__all__ = [
    "dataset",
    "read",
    "read_torch",
    "reader",
    "scan",
    "scan_torch",
    "scanner",
    "schema",
    # Backward-compatible private re-exports
    "_TORCH_WHERE_MAX_ROWS",
    "_arrow_type_from_tform",
    "_build_fits_metadata",
    "_can_use_full_read_path",
    "_can_use_mmap_row_path_for_full_read",
    "_can_use_torch_table_path_for_full_read",
    "_column_tform_code_and_repeat",
    "_column_tforms_for_decode",
    "_compile_where_to_simple_predicates",
    "_empty_table_with_schema",
    "_filter_table_with_where",
    "_fits_tform_is_bit",
    "_iter_chunks_cpp_table",
    "_read_cpp_table_chunk",
    "_read_table_unfiltered",
    "_read_table_with_where",
    "_resolve_rows_from_where_cpp",
    "_row_slice_from_start_num",
    "_schema_from_header",
    "_torch_cmp_mask",
    "_try_cpp_where_pushdown",
    "_try_torch_tensor_where_filter",
    "_unsigned_column_dtypes",
    "_where_mask_for_table",
]
