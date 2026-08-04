"""Scan/chunk read bodies for table I/O."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import torch

from .._table.cache import _acquire_cpp_handle, _acquire_cpp_reader
from .._table.utils import _normalize_row_slice, _require_pyarrow
from .._table.arrow_convert import _chunk_to_record_batch
from .._table_engine import validate_table_backend
from .._table.write import _resolve_table_hdu_index_and_columns
from .._where import parse_where_expression, where_columns_from_ast
from ._read_schema import (
    _build_fits_metadata,
    _can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read,
    _column_tforms_for_decode,
    _empty_table_with_schema,
    _row_slice_from_start_num,
    _unsigned_column_dtypes,
)

logger = logging.getLogger(__name__)


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
        mmap_reader = None
        try:
            if can_mmap_rows and hasattr(cpp, "open_fits_mmap_reader"):
                try:
                    # Open once: per-batch reopens (open + header parse per
                    # batch) dominate the mmap row path for batched scans.
                    mmap_reader = cpp.open_fits_mmap_reader(path, hdu)
                except (RuntimeError, OSError) as exc:
                    logger.debug(
                        "mmap reader open failed; falling back to handle read: %s", exc
                    )
                    can_mmap_rows = False
            row = start_row
            while row <= end_row:
                size = min(batch_size, end_row - row + 1)
                if can_mmap_rows and mmap_reader is not None:
                    try:
                        yield cpp.read_fits_table_rows_mmap_from_reader(
                            mmap_reader, col_list, row, size
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
    from . import read as _read_mod

    batches = list(
        _read_mod.scan(
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


def _scan_iter(
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
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    backend = validate_table_backend(backend)

    if where is not None:
        from . import read as _read_mod

        table = _read_mod.read(
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


def _scan_torch_iter(
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


def _resolve_rows_from_where_cpp(
    path: str,
    hdu: int,
    where: str,
    start_row: int,
    num_rows: int,
    mmap: bool,
    apply_fits_nulls: bool,
) -> Optional[list[int]]:
    """Resolve WHERE predicate to a sorted list of 0-based row indices via C++ chunk read."""
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

    # Lazy import: _where_mask_for_table lives in _read_where, which imports
    # _read_table_unfiltered from this module lazily.
    from ._read_where import _where_mask_for_table

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
