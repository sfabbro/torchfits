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
    from .paths import cfitsio_base_path, guard_fits_path, is_cfitsio_network_url
    from .table_api import _squeeze_scalar_columns

    if chunk_rows <= 0:
        raise ValueError("batch_size must be > 0")
    guard_fits_path(file_path)
    # Network URLs are opened by CFITSIO; local paths may use ``file.fits[HDU]``.
    if not is_cfitsio_network_url(file_path):
        check_path = cfitsio_base_path(file_path)
        if not os.path.exists(check_path):
            raise FileNotFoundError(file_path)

    col_list = columns if columns else []

    if not hasattr(cpp, "read_fits_table_rows"):
        result = cpp.read_fits_table(file_path, hdu, col_list, mmap)
        yield _squeeze_scalar_columns(result)
        return

    header = None
    if total_rows is None:
        header = get_header_func(file_path, hdu)
        total_rows = _total_rows_from_header(header)
    if total_rows == 0:
        return

    if num_rows != -1:
        total_rows = min(total_rows, start_row + num_rows - 1)

    # ASCII tables (XTENSION=TABLE) have no binary row layout and cannot be
    # read through the mmap row path; route them through the CFITSIO reader.
    # The XTENSION probe must run even when the caller supplied total_rows —
    # otherwise ASCII tables would be routed into the binary mmap path.
    ascii_table = False
    if header is None:
        try:
            import torchfits as _tf

            ascii_table = _tf.read_hdu_type(file_path, hdu) == "ASCII_TABLE"
        except Exception:
            ascii_table = False
    else:
        ascii_table = (
            str(header.get("XTENSION", "")).strip().upper() == "TABLE"
        )

    # Scaled columns (TSCALn/TZEROn beyond the unsigned conventions) cannot be
    # decoded from raw mmap bytes; route to the buffered CFITSIO reader, which
    # applies scaling in-memory (same fallback as the non-streaming read).
    scaled_columns = False
    if mmap and not ascii_table:
        try:
            from ..fits_schema import iter_table_columns

            hdr = header if header is not None else None
            if hdr is None:
                import torchfits as _tf

                hdr = _tf.read_header(file_path, hdu)
            selected = set(col_list) if col_list else None
            for col in iter_table_columns(hdr, selected=selected):
                tscal = col.tscal if col.tscal is not None else 1.0
                tzero = col.tzero if col.tzero is not None else 0.0
                is_unsigned = (tscal == 1.0) and (
                    abs(tzero - 32768.0) < 1e-5
                    or abs(tzero + 32768.0) < 1e-5
                    or abs(tzero - 2147483648.0) < 1e-5
                )
                if (tscal != 1.0 or tzero != 0.0) and not is_unsigned:
                    scaled_columns = True
                    break
        except Exception:
            scaled_columns = False

    row = start_row
    emitted = 0
    if mmap and not ascii_table and not scaled_columns and hasattr(cpp, "read_fits_table_rows"):
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cast(
                Dict[str, Any],
                _squeeze_scalar_columns(
                    cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
                ),
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
                    yield cast(
                        Dict[str, Any],
                        _squeeze_scalar_columns(reader.read_rows(col_list, row, size)),
                    )
                else:
                    yield cast(
                        Dict[str, Any],
                        _squeeze_scalar_columns(
                            cpp.read_fits_table_rows_from_handle(
                                file_handle, hdu, col_list, row, size
                            )
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
                _squeeze_scalar_columns(
                    cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
                ),
            )
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return
