"""LRU caches for C++ FITS file handles and table readers."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_HANDLE_CACHE_MAX = max(1, int(os.getenv("TORCHFITS_TABLE_HANDLE_CACHE_SIZE", "8")))
_HANDLE_CACHE_ENABLED = os.getenv("TORCHFITS_TABLE_HANDLE_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_READER_CACHE_MAX = max(1, int(os.getenv("TORCHFITS_TABLE_READER_CACHE_SIZE", "8")))
_READER_CACHE_ENABLED = os.getenv("TORCHFITS_TABLE_READER_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

_handle_cache_lock = threading.Lock()
_handle_cache: OrderedDict[str, Any] = OrderedDict()
_reader_cache_lock = threading.Lock()
_reader_cache: OrderedDict[tuple[str, int], Any] = OrderedDict()


def _close_cpp_handle(handle: Any) -> None:
    try:
        handle.close()
    except Exception as e:
        logger.debug("Failed to close C++ handle: %s", e)


def acquire_cpp_handle(path: str, cpp: Any) -> Any:
    if not _HANDLE_CACHE_ENABLED:
        return cpp.open_fits_file(path, "r")

    with _handle_cache_lock:
        handle = _handle_cache.get(path)
        if handle is not None:
            _handle_cache.move_to_end(path)
            return handle

    handle = cpp.open_fits_file(path, "r")
    with _handle_cache_lock:
        _handle_cache[path] = handle
        _handle_cache.move_to_end(path)
        while len(_handle_cache) > _HANDLE_CACHE_MAX:
            _, old = _handle_cache.popitem(last=False)
            _close_cpp_handle(old)
    return handle


def acquire_cpp_reader(path: str, hdu: int, cpp: Any) -> Any:
    """Return a cached C++ TableReader bound to a cached FITSFile handle."""
    hdu_index = int(hdu)
    key = (path, hdu_index)
    if not _READER_CACHE_ENABLED:
        file_handle = acquire_cpp_handle(path, cpp)
        return cpp.TableReader(file_handle, hdu_index)

    with _reader_cache_lock:
        reader = _reader_cache.get(key)
        if reader is not None:
            _reader_cache.move_to_end(key)
            return reader

    file_handle = acquire_cpp_handle(path, cpp)
    reader = cpp.TableReader(file_handle, hdu_index)
    with _reader_cache_lock:
        _reader_cache[key] = reader
        _reader_cache.move_to_end(key)
        while len(_reader_cache) > _READER_CACHE_MAX:
            _reader_cache.popitem(last=False)
    return reader


def close_all_cached_handles() -> None:
    with _reader_cache_lock:
        _reader_cache.clear()
    with _handle_cache_lock:
        items = list(_handle_cache.items())
        _handle_cache.clear()
    for _, handle in items:
        _close_cpp_handle(handle)


def invalidate_caches_for_path(path: str) -> None:
    """Drop cached readers/handles bound to a given file path."""
    with _reader_cache_lock:
        stale_reader_keys = [k for k in _reader_cache.keys() if k[0] == path]
        for key in stale_reader_keys:
            _reader_cache.pop(key, None)

    handle = None
    with _handle_cache_lock:
        handle = _handle_cache.pop(path, None)
    if handle is not None:
        _close_cpp_handle(handle)


atexit.register(close_all_cached_handles)
