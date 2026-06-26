"""Large-table and streaming FITS table readers."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterator, Optional, Union

import torch

from ..hdu import Header


def _total_rows_from_header(header: Header) -> int:
    total_rows = header.get("NAXIS2", 0)
    try:
        if isinstance(total_rows, str):
            return int(float(total_rows))
        return int(total_rows)
    except Exception:
        return 0


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
):
    """Yield FITS table data in row chunks."""
    import torchfits._C as cpp

    if not os.path.exists(file_path):
        return

    col_list = columns if columns else []

    if not hasattr(cpp, "read_fits_table_rows"):
        result = cpp.read_fits_table(file_path, hdu, col_list, mmap)
        yield result
        return

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
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
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
                    yield reader.read_rows(col_list, row, size)
                else:
                    yield cpp.read_fits_table_rows_from_handle(
                        file_handle, hdu, col_list, row, size
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
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return


def read_large_table(
    get_header_func: Callable[[str, int], Header],
    file_path: str,
    hdu: int = 1,
    max_memory_mb: int = 100,
    streaming: bool = False,
    return_iterator: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Read a large FITS table with memory management."""
    try:
        import torchfits._C as cpp

        if not os.path.exists(file_path):
            return {}

        if not streaming:
            if return_iterator:
                return iter([cpp.read_fits_table(file_path, hdu)])
            return cpp.read_fits_table(file_path, hdu)

        if not hasattr(cpp, "read_fits_table_rows"):
            return cpp.read_fits_table(file_path, hdu)

        try:
            total_rows = _total_rows_from_header(get_header_func(file_path, hdu))
            if total_rows == 0:
                return cpp.read_fits_table(file_path, hdu)

            sample_rows = min(256, total_rows)
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    reader = None
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)
                        sample = reader.read_rows([], 1, sample_rows)
                    else:
                        sample = cpp.read_fits_table_rows_from_handle(
                            file_handle, hdu, [], 1, sample_rows
                        )
                finally:
                    reader = None
                    file_handle.close()
            else:
                sample = cpp.read_fits_table_rows(
                    file_path, hdu, [], 1, sample_rows, False
                )

            def _estimate_bytes(data: Dict[str, Any], rows: int) -> float:
                total = 0
                for value in data.values():
                    if isinstance(value, torch.Tensor):
                        total += value.numel() * value.element_size()
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, torch.Tensor):
                                total += item.numel() * item.element_size()
                return total / max(1, rows)

            bytes_per_row = _estimate_bytes(sample, sample_rows)
            if bytes_per_row <= 0:
                bytes_per_row = 1024.0

            max_bytes = max_memory_mb * 1024 * 1024
            rows_per_chunk = max(1, int(max_bytes / bytes_per_row))

            if return_iterator:
                return stream_table(
                    get_header_func,
                    file_path,
                    hdu=hdu,
                    columns=None,
                    start_row=1,
                    num_rows=-1,
                    chunk_rows=rows_per_chunk,
                    mmap=False,
                )

            accum = {}
            start = 1
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    reader = None
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)

                    while start <= total_rows:
                        remaining = total_rows - start + 1
                        num = min(rows_per_chunk, remaining)
                        if reader is not None:
                            chunk = reader.read_rows([], start, num)
                        else:
                            chunk = cpp.read_fits_table_rows_from_handle(
                                file_handle, hdu, [], start, num
                            )

                        for key, value in chunk.items():
                            if isinstance(value, torch.Tensor):
                                if key not in accum:
                                    out_shape = (total_rows,) + tuple(value.shape[1:])
                                    accum[key] = torch.empty(
                                        out_shape, dtype=value.dtype
                                    )
                                accum[key][start - 1 : start - 1 + value.shape[0]] = (
                                    value
                                )
                            elif isinstance(value, list):
                                accum.setdefault(key, []).extend(value)
                            else:
                                accum.setdefault(key, []).append(value)

                        start += num
                finally:
                    reader = None
                    file_handle.close()
            else:
                while start <= total_rows:
                    remaining = total_rows - start + 1
                    num = min(rows_per_chunk, remaining)
                    chunk = cpp.read_fits_table_rows(
                        file_path, hdu, [], start, num, False
                    )

                    for key, value in chunk.items():
                        if isinstance(value, torch.Tensor):
                            if key not in accum:
                                out_shape = (total_rows,) + tuple(value.shape[1:])
                                accum[key] = torch.empty(out_shape, dtype=value.dtype)
                            accum[key][start - 1 : start - 1 + value.shape[0]] = value
                        elif isinstance(value, list):
                            accum.setdefault(key, []).extend(value)
                        else:
                            accum.setdefault(key, []).append(value)

                    start += num

            return accum
        except Exception:
            return cpp.read_fits_table(file_path, hdu)

    except Exception:
        return {}
