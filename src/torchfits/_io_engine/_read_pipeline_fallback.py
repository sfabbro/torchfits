"""Fallback read paths for the unified FITS read dispatcher.

Extracted from ``_read_pipeline.py`` as part of the A2 strategy refactor (0.6.0).
These are the slow-but-reliable paths used when fast-path conditions are not met.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from ..hdu import Header
from .device import to_device, validate_device
from ._read_pipeline import (
    _coerce_bit_table_columns,
    _coerce_unsigned_table_columns,
)
from .caches import (
    get_cached_handle,
    path_signature,
    set_cached_hdu_type,
    store_cached_read,
)

_log = logging.getLogger(__name__)


def read_fallback(
    *,
    cpp_module: Any,
    path: str,
    hdu: Any,
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    cache_capacity: int,
    handle_cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    force_image: bool,
    force_table: bool,
    hdu_type_hint: Any,
    columns: Any,
    start_row: int,
    num_rows: int,
    read_check_cache: Callable[..., tuple[bool, Any, Any]],
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_header: Callable[[Any, int, bool], Any],
) -> Any:
    """Generic fallback read path for image/table HDUs."""
    hit, cached_res, cache_key = read_check_cache(
        path,
        hdu,
        device,
        fp16,
        bf16,
        columns,
        start_row,
        num_rows,
        return_header,
        cache_capacity,
    )
    if hit:
        return cached_res

    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be non-negative")
    if start_row < 1:
        raise ValueError("start_row must be >= 1 (FITS uses 1-based indexing)")
    if num_rows < -1 or num_rows == 0:
        raise ValueError("num_rows must be > 0 or -1 for all rows")
    validate_device(device)

    try:
        file_handle, cached_handle = get_cached_handle(path, handle_cache_capacity)
        try:
            if isinstance(hdu, str):
                hdu_num = None
                if hasattr(cpp_module, "resolve_hdu_name_cached"):
                    try:
                        hdu_num = int(cpp_module.resolve_hdu_name_cached(path, hdu))
                    except Exception as exc:
                        _log.debug(
                            "resolve_hdu_name_cached(%r, %r) failed: %s",
                            path,
                            hdu,
                            exc,
                        )
                        hdu_num = None

                if hdu_num is None:
                    num_hdus = cpp_module.get_num_hdus(file_handle)
                    for i in range(num_hdus):
                        try:
                            hdr = cpp_module.read_header(file_handle, i)
                            if hdr.get("EXTNAME") == hdu:
                                hdu_num = i
                                break
                        except Exception:
                            continue

                if hdu_num is None:
                    raise ValueError(f"HDU '{hdu}' not found in file")
            else:
                hdu_num = hdu

            header = None
            header_data = None
            hdu_type = hdu_type_hint if isinstance(hdu_num, int) else None
            if isinstance(hdu_num, int) and hdu_type is None:
                try:
                    hdu_type = cpp_module.get_hdu_type(file_handle, hdu_num)
                    set_cached_hdu_type(path, hdu_num, hdu_type)
                except Exception:
                    hdu_type = None

            is_table_hdu = force_table or (hdu_type in {"ASCII_TABLE", "BINARY_TABLE"})
            if force_image:
                is_table_hdu = False

            if not is_table_hdu:
                try:
                    return read_fallback_image(
                        cpp_module=cpp_module,
                        file_handle=file_handle,
                        path=path,
                        hdu_num=hdu_num,
                        device=device,
                        mmap=mmap,
                        fp16=fp16,
                        bf16=bf16,
                        cache_capacity=cache_capacity,
                        fast_header=fast_header,
                        return_header=return_header,
                        cache_key=cache_key,
                        use_cache=cache_capacity > 0,
                        resolve_image_mmap=resolve_image_mmap,
                        read_header=read_header,
                    )
                except (RuntimeError, TypeError):
                    if force_image:
                        raise
                    if isinstance(hdu_num, int):
                        try:
                            hdu_type = cpp_module.get_hdu_type(file_handle, hdu_num)
                            set_cached_hdu_type(path, hdu_num, hdu_type)
                        except Exception:
                            hdu_type = None
                    is_table_hdu = force_table or (
                        hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}
                    )
                    if not is_table_hdu:
                        raise

            try:
                return read_fallback_table(
                    cpp_module=cpp_module,
                    file_handle=file_handle,
                    path=path,
                    hdu_num=hdu_num,
                    device=device,
                    mmap=mmap,
                    cache_capacity=cache_capacity,
                    fast_header=fast_header,
                    return_header=return_header,
                    cache_key=cache_key,
                    use_cache=cache_capacity > 0,
                    columns=columns,
                    start_row=start_row,
                    num_rows=num_rows,
                    header_data=header_data,
                    header=header,
                    read_header=read_header,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to read table extension: {exc}")

        finally:
            if not cached_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass

    except Exception as exc:
        raise RuntimeError(f"Failed to read FITS file '{path}': {exc}") from exc


def read_fallback_image(
    *,
    cpp_module: Any,
    file_handle: Any,
    path: str,
    hdu_num: Any,
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    cache_key: Any,
    use_cache: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_header: Callable[[Any, int, bool], Any],
) -> Any:
    """Read an image HDU in the generic fallback path."""
    effective_mmap = resolve_image_mmap(path, hdu_num, mmap, cache_capacity)
    header = None
    header_data = None
    if isinstance(hdu_num, int) and not (fp16 or bf16):
        try:
            header_data = read_header(file_handle, hdu_num, fast_header)
            header = Header(header_data)
        except Exception:
            header = None
    data = cpp_module.read_full(file_handle, hdu_num, effective_mmap)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if str(device) != "cpu":
        data = to_device(data, device)

    if return_header:
        if header is None:
            header_data = read_header(file_handle, hdu_num, fast_header)
            header = Header(header_data)

    if use_cache and cache_key is not None:
        store_cached_read(
            cache_key,
            (
                data.cpu() if device != "cpu" else data,
                header,
                path_signature(path),
            ),
            cache_capacity,
        )

    if isinstance(hdu_num, int):
        set_cached_hdu_type(path, hdu_num, "IMAGE")
    return (data, header) if return_header else data


def read_fallback_table(
    *,
    cpp_module: Any,
    file_handle: Any,
    path: str,
    hdu_num: Any,
    device: str,
    mmap: bool | str,
    cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    cache_key: Any,
    use_cache: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    header_data: Any,
    header: Any,
    read_header: Callable[[Any, int, bool], Any],
) -> Any:
    """Read a table HDU in the generic fallback path."""
    if (return_header or isinstance(hdu_num, str)) and header_data is None:
        header_data = read_header(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    col_list = columns if columns else []
    table_result = None
    table_mmap = mmap if isinstance(mmap, bool) else True
    if table_mmap:
        try:
            if start_row > 1 or num_rows != -1:
                if hasattr(cpp_module, "read_fits_table_rows"):
                    table_result = cpp_module.read_fits_table_rows(
                        path, hdu_num, col_list, start_row, num_rows, True
                    )
                else:
                    table_result = cpp_module.read_fits_table(
                        path, hdu_num, col_list, True
                    )
            else:
                table_result = cpp_module.read_fits_table(path, hdu_num, col_list, True)
        except Exception as exc:
            # Corruption must not fall through to a slower reader that will
            # fail again with a more confusing error.
            if "truncated" in str(exc).lower():
                raise
            table_result = None

    if table_result is None:
        # Prefer the cold-path binding with the thread-local reader cache:
        # it reuses the CFITSIO handle, the row scratch buffer and the pread
        # fd across calls (steady-state reads stop re-faulting ~24 MiB of
        # anonymous memory per call). from_handle fallbacks keep working for
        # older extension builds.
        if hasattr(cpp_module, "read_fits_table_rows"):
            try:
                table_result = cpp_module.read_fits_table_rows(
                    path, hdu_num, col_list, start_row, num_rows, False
                )
            except Exception as exc:
                if "truncated" in str(exc).lower():
                    raise
                table_result = None
    if table_result is None:
        if columns is None and start_row == 1 and num_rows == -1:
            table_result = cpp_module.read_fits_table_from_handle(file_handle, hdu_num)
        elif hasattr(cpp_module, "read_fits_table_rows_from_handle"):
            table_result = cpp_module.read_fits_table_rows_from_handle(
                file_handle, hdu_num, col_list, start_row, num_rows
            )
        elif start_row > 1 or num_rows != -1:
            if hasattr(cpp_module, "read_fits_table_rows"):
                table_result = cpp_module.read_fits_table_rows(
                    path, hdu_num, col_list, start_row, num_rows, False
                )
            else:
                table_result = cpp_module.read_fits_table(
                    path, hdu_num, col_list, False
                )
        else:
            table_result = cpp_module.read_fits_table(path, hdu_num, col_list, False)

    table_data = table_result
    # C++ already applies BIT→bool and unsigned TZERO offsets. Skip the extra
    # header round-trip unless the caller asked for the header.
    if return_header:
        if header is None:
            try:
                header = Header(read_header(file_handle, hdu_num, fast_header))
            except Exception:
                header = None
        table_data = _coerce_bit_table_columns(table_data, header)
        table_data = _coerce_unsigned_table_columns(table_data, header)
    elif header is not None:
        table_data = _coerce_bit_table_columns(table_data, header)
        table_data = _coerce_unsigned_table_columns(table_data, header)

    # Reader-boundary normalization: FITS scalar columns surface as (N, 1);
    # hand every consumer (and the cache) rank-1 columns instead.
    from .table_api import _squeeze_scalar_columns

    table_data = _squeeze_scalar_columns(table_data)

    if (start_row > 1 or num_rows != -1) and not hasattr(
        cpp_module, "read_fits_table_rows"
    ):
        end_row = None
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                if end_row is None:
                    end_row = start_row + num_rows - 1 if num_rows != -1 else len(value)
                table_data[key] = value[start_row - 1 : end_row]

    if use_cache and cache_key is not None:
        store_cached_read(
            cache_key,
            (table_data, header, path_signature(path)),
            cache_capacity,
        )

    if str(device) != "cpu":
        new_data: dict[str, Any] = {}
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = to_device(value, device)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        new_list.append(to_device(item, device))
                    else:
                        new_list.append(item)
                new_data[key] = new_list
            else:
                new_data[key] = value
        table_data = new_data

    if isinstance(hdu_num, int):
        set_cached_hdu_type(path, hdu_num, "BINARY_TABLE")
    return (table_data, header) if return_header else table_data
