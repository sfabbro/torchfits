"""CFITSIO-backed FITS I/O surface.

This module owns FITS file reads, writes, HDU operations, header extraction,
checksum helpers, subset reads, batch reads, and FITS cache controls.
Arrow-native table APIs live in :mod:`torchfits.table`.
"""

from __future__ import annotations

import atexit
import logging as _stdlib_logging
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hdu import HDUList, Header

import torchfits._C as cpp

from ._io_engine.batch import (
    get_batch_info as _get_batch_info_impl,
    read_batch as _read_batch_impl,
)
from ._io_engine.caches import (
    cache_subsystem_policy as _cache_subsystem_policy_impl,
    clear_cache_subsystem as _clear_cache_subsystem_impl,
    check_read_cache as _check_read_cache_impl,
    clear_file_cache as _clear_file_cache_impl,
    get_cache_performance as _get_cache_performance_impl,
    invalidate_path_caches as _invalidate_path_caches_impl,
)
from ._io_engine.checksum_api import verify_checksums as _verify_checksums_impl
from ._io_engine.checksum_api import write_checksums as _write_checksums_impl
from ._io_engine.hdu_api import autodetect_hdu as _autodetect_hdu_impl
from ._io_engine.hdu_api import get_header as _get_header_impl

from ._io_engine.hdu_api import open_hdulist as _open_hdulist_impl
from ._io_engine.hdu_api import read_header_fast as _read_header_fast_impl
from ._io_engine.hdu_api import read_colnames as _read_colnames_impl
from ._io_engine.hdu_api import read_extname as _read_extname_impl
from ._io_engine.hdu_api import read_hdu_type as _read_hdu_type_impl
from ._io_engine.hdu_api import read_keys as _read_keys_impl
from ._io_engine.hdu_api import read_nrows as _read_nrows_impl
from ._io_engine.hdu_api import read_num_hdus as _read_num_hdus_impl
from ._io_engine.hdu_api import read_shape as _read_shape_impl
from ._io_engine.hdu_api import read_table_info as _read_table_info_impl
from ._io_engine.image import batch_to_device as _batch_to_device_impl
from ._io_engine.image import read_hdus as _read_hdus_impl
from ._io_engine.image import read_image as _read_image_impl
from ._io_engine.image_meta import (
    get_image_meta as _get_image_meta_impl,
    resolve_image_mmap as _resolve_image_mmap_impl,
    should_use_cold_nommap as _should_use_cold_nommap_impl,
)
from ._io_engine._read_pipeline import read_unified as _read_unified_impl
from ._io_engine.subset import open_subset_reader as _open_subset_reader_impl
from ._io_engine.table_reader_api import open_table_reader as _open_table_reader_impl
from ._io_engine.write_api import delete_hdu as _delete_hdu_impl
from ._io_engine.write_api import insert_hdu as _insert_hdu_impl
from ._io_engine.write_api import replace_hdu as _replace_hdu_impl
from ._io_engine.write_api import write as _write_impl
from ._io_engine.write_api import (
    _normalize_cpp_table_data as _normalize_cpp_table_data_impl,
)
from ._io_engine.write_api import (
    _delete_header_key_if_supported as _delete_header_key_if_supported_impl,
)
from ._io_engine.write_api import (
    _write_header_cards_if_supported as _write_header_cards_if_supported_impl,
)

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


def _invalidate_path_caches(path: str) -> None:
    _invalidate_path_caches_impl(path)


def _cpp_module() -> Any:
    return cpp


def _read_check_cache(*args: Any, **kwargs: Any) -> Any:
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


def _get_image_meta(path: str, hdu: int) -> Any:
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


def _resolve_image_mmap(
    path: str, hdu: int, mmap: bool | str, cache_capacity: int
) -> Any:
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
    hdu: Any = 0,
    device: str = "cpu",
    mmap: bool | str = "auto",
    mode: str = "auto",
    options: Any = None,
    return_header: bool = False,
    **kwargs: Any,
) -> Any:
    """Read a FITS image or table from the given path and HDU.

    Returns a ``torch.Tensor`` (images) or ``dict[str, torch.Tensor]``
    (tables as column tensors), optionally with the FITS header.

    Default ``hdu=0`` (primary). Pass ``hdu=None`` or ``hdu="auto"`` to
    autodetect the first image/table payload HDU.

    For dataframe workflows (Arrow / ``where=`` / Polars), prefer
    :func:`torchfits.table.read` or :func:`torchfits.table.read_polars`.
    For explicit image tensors, prefer :func:`read_tensor`.
    """
    if "mode" in kwargs:
        raise TypeError("read() got multiple values for argument 'mode'")
    kwargs = dict(kwargs)
    kwargs["mode"] = mode
    return _read_unified_impl(
        cpp_module=_cpp_module(),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        options=options,
        return_header=return_header,
        kwargs=kwargs,
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


def read_tensor(
    path: str,
    hdu: int | str = 0,
    device: str = "cpu",
    mmap: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    return_header: bool = False,
    fallback_get_header: Any = None,
) -> Any:
    """Read an N-dimensional array directly to a PyTorch Tensor."""
    fallback = fallback_get_header if fallback_get_header is not None else read_header
    return _read_image_impl(
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        raw_scale=raw_scale,
        return_header=return_header,
        fallback_get_header=fallback,
    )


def read_hdus(
    path: str,
    hdus: list[int | str] | tuple[int | str, ...],
    *,
    device: str = "cpu",
    mmap: bool = True,
    return_header: bool = False,
) -> Any:
    """Read multiple HDUs from a single FITS file. Returns a list of tensors."""
    return _read_hdus_impl(
        path, hdus, device=device, mmap=mmap, return_header=return_header
    )


def read_subset(
    path: str,
    hdu: int | str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Any:
    """Read a rectangular subset from an image HDU: box corners (x1, y1)-(x2, y2), half-open."""
    with open_subset_reader(path, hdu=hdu) as reader:
        return reader.read_subset(x1, y1, x2, y2)


def open_subset_reader(path: str, hdu: int | str = 0, device: str = "cpu") -> Any:
    """Open a reusable subset reader for repeated cutout access on an image HDU."""
    return _open_subset_reader_impl(path, hdu=hdu, device=device)


def open_table_reader(path: str, hdu: int | str = 1) -> Any:
    """Open a reusable table reader for repeated column/row access on one HDU.

    Filtered reads (``where=``) are not supported on the persistent reader;
    use :func:`torchfits.table.read_torch` with ``where=`` instead.
    """
    return _open_table_reader_impl(path, hdu=hdu)


def open(path: str, mode: str = "r") -> HDUList:
    """Open a FITS file and return an HDUList for low-level HDU/header access."""
    return _open_hdulist_impl(path, mode=mode)


def write(
    path: str | os.PathLike[str],
    data: Any,
    header: Header | dict[str, Any] | None = None,
    overwrite: bool = False,
    compress: bool | str = False,
    quantize: Any = None,
    checksum: bool = False,
) -> None:
    """Write a tensor or numpy array to a FITS file (primary or image extension).

    ``quantize="robust"`` (or a ``lo_q``/``hi_q`` dict) packs float images to
    ``BITPIX=16`` with robust ``BSCALE``/``BZERO``. Default keeps native float.

    ``compress`` accepts an algorithm string (``"RICE_1"``, ``"GZIP_1"``,
    ``"HCOMPRESS_1"``). GZIP_1 and integer RICE_1 writes are lossless; float
    RICE_1 and HCOMPRESS_1 use CFITSIO's default quantization (lossy, the same
    behavior as astropy and fitsio defaults).

    ``checksum=True`` writes CFITSIO DATASUM/CHECKSUM keywords for every HDU
    after the payload lands; verify later with :func:`verify_checksums`.
    """
    return _write_impl(
        path,
        data,
        header=header,
        overwrite=overwrite,
        compress=compress,
        quantize=quantize,
        checksum=checksum,
    )


def write_tensor(
    path: str,
    tensor: Any,
    header: Any = None,
    overwrite: bool = False,
    compress: bool | str = False,
    quantize: Any = None,
    checksum: bool = False,
) -> None:
    """Write a single PyTorch Tensor directly to a FITS image extension."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    return write(
        path,
        tensor,
        header=header,
        overwrite=overwrite,
        compress=compress,
        quantize=quantize,
        checksum=checksum,
    )


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: dict[str, Any] | None = None,
    compress: bool | str = False,
) -> None:
    """Insert a new image HDU into an existing FITS file at the given index."""
    return _insert_hdu_impl(path, data, index=index, header=header, compress=compress)


def replace_hdu(
    path: str,
    hdu: int | str,
    data: Any,
    header: dict[str, Any] | None = None,
    compress: bool | str = False,
) -> None:
    """Replace an existing HDU in a FITS file with new data."""
    return _replace_hdu_impl(path, hdu, data, header=header, compress=compress)


def delete_hdu(
    path: str,
    hdu: int | str,
    compress: bool | str = False,
) -> None:
    """Delete an HDU from a FITS file by index or name."""
    return _delete_hdu_impl(path, hdu, compress=compress)


def read_header(path: str, hdu: Any = 0) -> Any:
    """Read the FITS header for the given HDU as a Header dict-like object.

    Default ``hdu=0``. Pass ``hdu=None`` or ``hdu="auto"`` to autodetect.
    """
    return _get_header_impl(path, hdu, autodetect_hdu=_autodetect_hdu_impl)


def read_nrows(path: str, hdu: Any = 1) -> int:
    """Return table ``NAXIS2`` / row count without materializing the full header.

    Uses CFITSIO ``fits_get_num_rows``. Default ``hdu=1``. For bloated headers
    prefer this over ``read_header(...).get("NAXIS2")``.
    """
    return _read_nrows_impl(path, hdu)


def read_keys(path: str, keys: Any, hdu: Any = 0) -> dict[str, Any]:
    """Read selected header keywords without materializing the full header.

    Uses CFITSIO ``fits_read_keyword`` per key. Missing keys raise.
    Default ``hdu=0`` matches :func:`read_header`.
    """
    return _read_keys_impl(path, keys, hdu=hdu)


def read_shape(path: str, hdu: Any = 0) -> tuple[int, tuple[int, ...]]:
    """Return ``(bitpix, shape)`` without materializing the full header.

    Shape is torch / row-major order. Default ``hdu=0``.
    """
    return _read_shape_impl(path, hdu)


def read_hdu_type(path: str, hdu: Any = 0) -> str:
    """Return HDU type (``IMAGE`` / ``BINARY_TABLE`` / …) without a full header dump."""
    return _read_hdu_type_impl(path, hdu)


def read_num_hdus(path: str) -> int:
    """Return the number of HDUs in ``path`` (one open; no header dump)."""
    return _read_num_hdus_impl(path)


def read_colnames(path: str, hdu: Any = 1) -> list[str]:
    """Return table column names without materializing the full header."""
    return _read_colnames_impl(path, hdu)


def read_extname(path: str, hdu: Any = 0) -> str | None:
    """Return ``EXTNAME`` for an HDU, or ``None`` if absent."""
    return _read_extname_impl(path, hdu)


def read_table_info(path: str, hdu: Any = 1) -> dict[str, Any]:
    """Return ``{nrows, colnames, tforms}`` in one open (no full header dump)."""
    return _read_table_info_impl(path, hdu)


def _write_header_cards_if_supported(*args: Any, **kwargs: Any) -> None:
    return _write_header_cards_if_supported_impl(*args, **kwargs)


def _delete_header_key_if_supported(*args: Any, **kwargs: Any) -> None:
    return _delete_header_key_if_supported_impl(*args, **kwargs)


def read_batch(
    file_paths: list[str],
    hdu: int = 0,
    device: str = "cpu",
    *,
    strict: bool = False,
) -> Any:
    """Read the same HDU from multiple FITS files.

    Returns a ``list`` of tensors containing only the successfully read
    files, in input order (with ``strict=False``, the default, failures are
    skipped with a warning naming each failed path — so result positions do
    not map 1:1 onto ``file_paths`` when any read fails). Pass
    ``strict=True`` to raise on the first failure instead.
    """
    return _read_batch_impl(
        read_func=read,
        read_exc_types=_READ_EXC_TYPES,
        log=_log,
        file_paths=file_paths,
        hdu=hdu,
        device=device,
        strict=strict,
    )


def read_batch_info(file_paths: list[str]) -> Any:
    """Inspect shape and dtype consistency across files for batched reading."""
    return _get_batch_info_impl(file_paths)


def get_cache_performance() -> Any:
    """Return cache hit/miss statistics for the handle and metadata caches."""
    return _get_cache_performance_impl()


def clear_file_cache(
    *,
    data: bool = True,
    handles: bool = True,
    meta: bool = True,
    hdu_types: bool = True,
    stats: bool = True,
    cpp: bool = True,
    cpp_module: Any = None,
) -> None:
    """Clear the FITS file handle and metadata caches selectively."""
    return _clear_file_cache_impl(
        data=data,
        handles=handles,
        meta=meta,
        hdu_types=hdu_types,
        stats=stats,
        cpp=cpp,
        cpp_module=cpp_module,
    )


def cache_subsystem_policy(name: str) -> dict[str, bool]:
    """Query which cache subsystems (data, handles, meta) are enabled for a policy."""
    return _cache_subsystem_policy_impl(name)


def clear_cache_subsystem(name: str) -> None:
    """Clear all caches for the given subsystem name."""
    _clear_cache_subsystem_impl(name)


def _shutdown_fits_io_caches() -> None:
    cpp_module = sys.modules.get("torchfits._C")
    _clear_cache_subsystem_impl("all", cpp_module=cpp_module)


atexit.register(_shutdown_fits_io_caches)


def write_checksums(path: str, hdu: int = 0) -> None:
    """Write DATASUM and CHECKSUM keywords for the given HDU."""
    return _write_checksums_impl(path, hdu=hdu)


def verify_checksums(path: str, hdu: int = 0) -> dict[str, Any]:
    """Verify DATASUM and CHECKSUM for the given HDU. Returns dict of status."""
    return _verify_checksums_impl(path, hdu=hdu)


def _normalize_cpp_table_data(table_dict: dict[str, Any]) -> dict[str, Any]:
    return _normalize_cpp_table_data_impl(table_dict)


__all__ = [
    "cache_subsystem_policy",
    "clear_cache_subsystem",
    "clear_file_cache",
    "delete_hdu",
    "get_cache_performance",
    "insert_hdu",
    "open",
    "open_subset_reader",
    "open_table_reader",
    "read",
    "read_batch",
    "read_batch_info",
    "read_colnames",
    "read_extname",
    "read_hdu_type",
    "read_header",
    "read_hdus",
    "read_keys",
    "read_nrows",
    "read_num_hdus",
    "read_shape",
    "read_subset",
    "read_table_info",
    "read_tensor",
    "replace_hdu",
    "verify_checksums",
    "write",
    "write_checksums",
    "write_tensor",
]
