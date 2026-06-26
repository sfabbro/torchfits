"""CFITSIO-backed FITS I/O surface.

This module owns FITS file reads, writes, HDU operations, header extraction,
checksum helpers, subset reads, batch reads, table streaming, and FITS cache
controls. Broader convenience surfaces remain in :mod:`torchfits.io`.
"""

from __future__ import annotations

import logging as _stdlib_logging
import os
import sys
import atexit
from typing import Any

import torchfits._C as cpp

from . import table
from ._io_engine.batch import (
    get_batch_info as _get_batch_info_impl,
    read_batch as _read_batch_impl,
)
from ._io_engine.caches import (
    IO_CACHE_SUBSYSTEMS,
    cache_subsystem_policy as _cache_subsystem_policy_impl,
    clear_cache_subsystem as _clear_cache_subsystem_impl,
    check_read_cache as _check_read_cache_impl,
    clear_file_cache as _clear_file_cache_impl,
    get_cache_performance as _get_cache_performance_impl,
    get_cached_handle as _get_cached_handle_impl,
    invalidate_path_caches as _invalidate_path_caches_impl,
)
from ._io_engine.checksum_api import verify_checksums as _verify_checksums_impl
from ._io_engine.checksum_api import write_checksums as _write_checksums_impl
from ._io_engine.hdu_api import autodetect_hdu as _autodetect_hdu_impl
from ._io_engine.hdu_api import get_header as _get_header_impl

from ._io_engine.hdu_api import open_hdulist as _open_hdulist_impl
from ._io_engine.hdu_api import read_header_fast as _read_header_fast_impl
from ._io_engine.image import batch_to_device as _batch_to_device_impl
from ._io_engine.image import read_hdus as _read_hdus_impl
from ._io_engine.image import read_image as _read_image_impl
from ._io_engine.image_meta import (
    get_image_meta as _get_image_meta_impl,
    resolve_image_mmap as _resolve_image_mmap_impl,
    should_use_cold_nommap as _should_use_cold_nommap_impl,
)
from ._io_engine.read_dispatch import read_unified as _read_unified_impl
from ._io_engine.subset import open_subset_reader as _open_subset_reader_impl
from ._io_engine.subset import read_subset as _read_subset_impl
from ._io_engine.table_api import read_table as _read_table_impl
from ._io_engine.table_streaming import (
    read_large_table as _read_large_table_impl,
)
from ._io_engine.table_streaming import stream_table as _stream_table_impl
from ._io_engine.write_api import delete_hdu as _delete_hdu_impl
from ._io_engine.write_api import insert_hdu as _insert_hdu_impl
from ._io_engine.write_api import replace_hdu as _replace_hdu_impl
from ._io_engine.write_api import write as _write_impl
from ._io_engine.write_api import _normalize_cpp_table_data as _normalize_cpp_table_data_impl
from ._io_engine.write_api import _write_header_cards_if_supported as _write_header_cards_if_supported_impl

_log = _stdlib_logging.getLogger(__name__)
_DEBUG_SCALE = os.environ.get("TORCHFITS_DEBUG_SCALE") == "1"
_COLD_NOMMAP = os.environ.get("TORCHFITS_COLD_NOMMAP") == "1"
_COLD_NOCACHE = os.environ.get("TORCHFITS_COLD_NOCACHE") == "1"
_READ_EXC_TYPES = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    MemoryError,
)


def read_fast(*args: Any, **kwargs: Any):
    from ._fastio import read as _read_fast

    return _read_fast(*args, **kwargs)


def _invalidate_path_caches(path: str) -> None:
    _invalidate_path_caches_impl(path, table)


def _cpp_module():
    return cpp


def _read_check_cache(*args: Any, **kwargs: Any):
    return _check_read_cache_impl(
        path=args[0],
        hdu=args[1],
        device=args[2],
        fp16=args[3],
        bf16=args[4],
        columns=args[5],
        start_row=args[6],
        num_rows=args[7],
        return_header=args[8],
        cache_capacity=args[9],
        invalidate_path=_invalidate_path_caches,
    )


def _get_image_meta(path: str, hdu: int):
    return _get_image_meta_impl(path, hdu, cpp_module=_cpp_module())


def _should_use_cold_nommap(
    path: str, hdu: int, cache_capacity: int, mmap: bool
) -> bool:
    return _should_use_cold_nommap_impl(
        path,
        hdu,
        cache_capacity,
        mmap,
        force_cold_nommap=_COLD_NOMMAP,
        get_image_meta_func=_get_image_meta,
    )


def _resolve_image_mmap(path: str, hdu: int, mmap: bool | str, cache_capacity: int):
    return _resolve_image_mmap_impl(
        path,
        hdu,
        mmap,
        cache_capacity,
        get_image_meta_func=_get_image_meta,
        should_use_cold_nommap_func=_should_use_cold_nommap,
    )


def read(
    path: Any,
    hdu: Any = None,
    device: str = "cpu",
    mmap: bool | str = "auto",
    options: Any = None,
    return_header: bool = False,
    **kwargs: Any,
):
    return _read_unified_impl(
        cpp_module=_cpp_module(),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        options=options,
        return_header=return_header,
        kwargs=dict(kwargs),
        autodetect_hdu=_autodetect_hdu_impl,
        batch_to_device=_batch_to_device_impl,
        resolve_image_mmap=_resolve_image_mmap,
        read_check_cache=_read_check_cache,
        read_header=_read_header_fast_impl,
        debug_scale=_DEBUG_SCALE,
        cold_nocache=_COLD_NOCACHE,
        read_exc_types=_READ_EXC_TYPES,
        logger=_log,
    )


def read_image(*args: Any, **kwargs: Any):
    kwargs.setdefault("fallback_get_header", get_header)
    return _read_image_impl(*args, **kwargs)


def read_table(*args: Any, **kwargs: Any):
    return _read_table_impl(read, *args, **kwargs)


def read_hdus(*args: Any, **kwargs: Any):
    return _read_hdus_impl(*args, **kwargs)


def read_subset(
    path: str,
    hdu: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle_cache_capacity: int = 16,
):
    return _read_subset_impl(
        get_cached_handle=_get_cached_handle_impl,
        path=path,
        hdu=hdu,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        handle_cache_capacity=handle_cache_capacity,
    )


def open_subset_reader(*args: Any, **kwargs: Any):
    return _open_subset_reader_impl(*args, **kwargs)


def open(*args: Any, **kwargs: Any):
    return _open_hdulist_impl(*args, **kwargs)


def write(*args: Any, **kwargs: Any):
    return _write_impl(*args, **kwargs)


def insert_hdu(*args: Any, **kwargs: Any):
    return _insert_hdu_impl(*args, **kwargs)


def replace_hdu(*args: Any, **kwargs: Any):
    return _replace_hdu_impl(*args, **kwargs)


def delete_hdu(*args: Any, **kwargs: Any):
    return _delete_hdu_impl(*args, **kwargs)


def get_header(path: str, hdu: Any = None):
    return _get_header_impl(path, hdu, autodetect_hdu=_autodetect_hdu_impl)


def _write_header_cards_if_supported(*args: Any, **kwargs: Any):
    return _write_header_cards_if_supported_impl(*args, **kwargs)



def stream_table(*args: Any, **kwargs: Any):
    return _stream_table_impl(get_header, *args, **kwargs)


def read_large_table(*args: Any, **kwargs: Any):
    return _read_large_table_impl(get_header, *args, **kwargs)


def read_batch(
    file_paths: list[str],
    hdu: int = 0,
    device: str = "cpu",
    *,
    strict: bool = False,
):
    return _read_batch_impl(
        read_func=read,
        read_exc_types=_READ_EXC_TYPES,
        log=_log,
        file_paths=file_paths,
        hdu=hdu,
        device=device,
        strict=strict,
    )


def get_batch_info(file_paths: list[str]):
    return _get_batch_info_impl(file_paths)


def get_cache_performance():
    return _get_cache_performance_impl()


def clear_file_cache(*args: Any, **kwargs: Any):
    return _clear_file_cache_impl(*args, **kwargs)


def cache_subsystem_policy(name: str) -> dict[str, bool]:
    return _cache_subsystem_policy_impl(name)


def clear_cache_subsystem(name: str) -> None:
    _clear_cache_subsystem_impl(name, table_module=table)


def _shutdown_fits_io_caches() -> None:
    cpp_module = sys.modules.get("torchfits._C")
    _clear_cache_subsystem_impl(
        "all",
        table_module=table,
        cpp_module=cpp_module,
    )


atexit.register(_shutdown_fits_io_caches)


def write_checksums(*args: Any, **kwargs: Any):
    return _write_checksums_impl(*args, **kwargs)


def verify_checksums(*args: Any, **kwargs: Any):
    return _verify_checksums_impl(*args, **kwargs)


def read_table_rows(
    path: str,
    hdu: int = 1,
    start_row: int = 1,
    num_rows: int = 1000,
    columns: list[str] | None = None,
    device: str = "cpu",
    mmap: bool | str = True,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
):
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    if num_rows <= 0:
        raise ValueError("num_rows must be > 0 for read_table_rows")
    return read_table(
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
    )


def _normalize_cpp_table_data(table_dict: dict[str, Any]) -> dict[str, Any]:
    return _normalize_cpp_table_data_impl(table_dict)


__all__ = [
    "clear_file_cache",
    "cache_subsystem_policy",
    "clear_cache_subsystem",
    "delete_hdu",
    "get_batch_info",
    "get_cache_performance",
    "get_header",
    "insert_hdu",
    "open",
    "open_subset_reader",
    "read",
    "read_batch",
    "read_fast",
    "read_hdus",
    "read_image",
    "read_large_table",
    "read_subset",
    "read_table",
    "read_table_rows",
    "replace_hdu",
    "stream_table",
    "verify_checksums",
    "write",
    "write_checksums",
    "IO_CACHE_SUBSYSTEMS",
    "_normalize_cpp_table_data",
    "_write_header_cards_if_supported",
]

# Internal capability flags preserved for downstream tests and diagnostics.
_HAS_READ_HDUS_BATCH = hasattr(cpp, "read_hdus_batch")
_HAS_READ_FULL_RAW_WITH_SCALE = hasattr(cpp, "read_full_raw_with_scale")
_HAS_READ_FULL_RAW = hasattr(cpp, "read_full_raw")
_HAS_READ_FULL_UNMAPPED_RAW = hasattr(cpp, "read_full_unmapped_raw")
_HAS_READ_FULL_UNMAPPED = hasattr(cpp, "read_full_unmapped")
_HAS_READ_FULL_NOCACHE = hasattr(cpp, "read_full_nocache")
