"""HDU/header access helpers for root FITS I/O."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

from ..header_parser import fast_parse_header
from ..hdu import HDUList, Header

from .caches import (
    auto_hdu_cache,
    get_cached_handle,
    get_cached_hdu_type,
    path_signature,
    set_cached_hdu_type,
)


def read_header_fast(file_handle: Any, hdu_index: int, fast_header: bool = True):
    """Read header using fast bulk parsing or fallback to slow method."""
    import torchfits._C as cpp

    if fast_header:
        try:
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, RuntimeError, OSError):
            pass

    return cpp.read_header(file_handle, hdu_index)


def _header_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in {"T", "TRUE", "1", "YES", "Y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)




def find_first_hdu(
    path: str,
    handle_cache_capacity: int = 16,
) -> Optional[int]:
    """Find first payload HDU, preferring image/compressed-image over table."""
    import torchfits._C as cpp

    file_handle, cached = get_cached_handle(path, handle_cache_capacity)
    first_table_hdu: Optional[int] = None
    try:
        num_hdus = cpp.get_num_hdus(file_handle)
        for i in range(num_hdus):
            hdu_type = get_cached_hdu_type(path, i)
            if hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, i)
                    set_cached_hdu_type(path, i, hdu_type)
                except Exception:
                    hdu_type = None
            if hdu_type == "IMAGE":
                try:
                    shape = file_handle.get_shape(i)
                except Exception:
                    shape = []
                if shape and all(int(dim) > 0 for dim in shape):
                    return i
                continue

            if hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}:
                try:
                    hdr = read_header_fast(file_handle, i, fast_header=True)
                except Exception:
                    hdr = {}
                zimage = _header_truthy(hdr.get("ZIMAGE"))
                has_compression_keys = any(
                    k in hdr for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
                )
                if zimage or has_compression_keys:
                    return i
                if first_table_hdu is None:
                    first_table_hdu = i
    finally:
        if not cached:
            try:
                file_handle.close()
            except Exception:
                pass

    return first_table_hdu


def autodetect_hdu(path: str, handle_cache_capacity: int = 16) -> int:
    """Return the first HDU with payload, preferring image/compressed-image HDUs."""
    sig = path_signature(path)
    cache_key = (path, "payload")
    cached = auto_hdu_cache.get(cache_key)
    if cached is not None:
        cached_sig, cached_hdu = cached
        if sig is None or cached_sig is None or cached_sig == sig:
            auto_hdu_cache.move_to_end(cache_key)
            return int(cached_hdu)
        auto_hdu_cache.pop(cache_key, None)

    resolved = find_first_hdu(path, handle_cache_capacity=handle_cache_capacity)
    if resolved is None:
        return 0

    auto_hdu_cache[cache_key] = (sig, int(resolved))
    auto_hdu_cache.move_to_end(cache_key)
    while len(auto_hdu_cache) > 512:
        auto_hdu_cache.popitem(last=False)
    return int(resolved)


def open_hdulist(path: str, mode: str = "r") -> HDUList:
    """Open a FITS file for reading/writing."""
    if mode == "r" and not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open FITS file '{path}': {exc}") from exc


def get_header(
    path: str,
    hdu: Union[int, str, None] = None,
    *,
    autodetect_hdu: Callable[[str, int], int],
) -> Header:
    """Get the header of a FITS file."""
    import torchfits._C as cpp

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = autodetect_hdu(path, 16)

    if isinstance(hdu, str):
        if hasattr(cpp, "resolve_hdu_name_cached"):
            try:
                hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
                return Header(cpp.read_header_dict(path, hdu))
            except Exception:
                pass

        for i in range(100):  # Fallback scan
            try:
                header_data = cpp.read_header_dict(path, i)
                if not header_data:
                    break
                header = Header(header_data)
                if header.get("EXTNAME") == hdu:
                    return header
            except Exception:
                break
        raise ValueError(f"HDU '{hdu}' not found")

    return Header(cpp.read_header_dict(path, hdu))

