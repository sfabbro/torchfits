"""Streaming FITS table readers."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterator, Optional, cast

from ..hdu import Header


def _total_rows_from_header(header: Header) -> int:
    total_rows = header.get("NAXIS2", 0)
    try:
        if isinstance(total_rows, str):
            return int(float(total_rows))
        return int(total_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid NAXIS2 value: {total_rows!r}") from exc


def stream_table(
    get_header_func: Callable[[str, int], Header],
    file_path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    chunk_rows: int = 65536,
    mmap: bool = False,
    max_chunks: Optional[int] = None,
    total_rows: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield FITS table data in row chunks.

    ``total_rows`` (``NAXIS2``) may be supplied by callers that already hold the
    header to skip the redundant ``get_header_func`` read at stream start.
    """
    import torchfits._C as cpp

    if chunk_rows <= 0:
        raise ValueError("batch_size must be > 0")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    col_list = columns if columns else []

    if not hasattr(cpp, "read_fits_table_rows"):
        result = cpp.read_fits_table(file_path, hdu, col_list, mmap)
        yield result
        return

    if total_rows is None:
        total_rows = _total_rows_from_header(get_header_func(file_path, hdu))
    if total_rows == 0:
        return

    if num_rows != -1:
        total_rows = min(total_rows, start_row + num_rows - 1)

    row = start_row
    emitted = 0
    if mmap and hasattr(cpp, "read_fits_table_rows"):
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cast(
                Dict[str, Any],
                cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap),
            )
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return
    elif hasattr(cpp, "read_fits_table_rows_from_handle"):
        file_handle = cpp.open_fits_file(file_path, "r")
        try:
            reader = None
            if hasattr(cpp, "TableReader"):
                reader = cpp.TableReader(file_handle, hdu)
            while row <= total_rows:
                remaining = total_rows - row + 1
                size = min(chunk_rows, remaining)
                if reader is not None:
                    yield cast(Dict[str, Any], reader.read_rows(col_list, row, size))
                else:
                    yield cast(
                        Dict[str, Any],
                        cpp.read_fits_table_rows_from_handle(
                            file_handle, hdu, col_list, row, size
                        ),
                    )
                row += size
                emitted += 1
                if max_chunks is not None and emitted >= max_chunks:
                    return
        finally:
            reader = None
            file_handle.close()
    else:
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cast(
                Dict[str, Any],
                cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap),
            )
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return
