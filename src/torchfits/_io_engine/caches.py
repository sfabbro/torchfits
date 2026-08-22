"""Python-side cache state for root FITS I/O dispatch."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from types import MappingProxyType
from threading import RLock
from typing import Any

from .device import to_device

_CACHE_STATS_DEFAULT = {
    "total_requests": 0,
    "hits": 0,
    "misses": 0,
    "cache_size": 0,
}

cache_stats: dict[str, int] = dict(_CACHE_STATS_DEFAULT)
# OrderedDict operations are individually GIL-safe, but cache hit/move/evict
# sequences are not atomic across reader threads. Keep one re-entrant lock so
# callers can compose operations without exposing partially updated LRU state.
cache_lock = RLock()
file_cache: OrderedDict[Any, Any] = OrderedDict()
image_meta_cache: OrderedDict[tuple[str, int], Any] = OrderedDict()
# (path, hdu) -> (path_signature, cards tuple) for repeated read_header.
header_cards_cache: OrderedDict[
    tuple[str, int], tuple[Any, tuple[tuple[str, Any, str], ...]]
] = OrderedDict()
hdu_type_cache: OrderedDict[tuple[str, int], str | None] = OrderedDict()
cold_nommap_cache: OrderedDict[tuple[str, int], bool] = OrderedDict()
auto_mmap_cache: OrderedDict[tuple[str, int], bool] = OrderedDict()
auto_hdu_cache: OrderedDict[Any, Any] = OrderedDict()
_HEADER_CARDS_CACHE_MAX = 128

# Registry of open HDUList file handles, keyed by real path.
# NOTE: a list is sufficient while concurrent opens per path stay small;
# use weak references if long-lived high-fanout readers ever make this grow.
_open_hdulist_registry: dict[str, list[tuple[Any, Any]]] = {}

IO_CACHE_SUBSYSTEMS = MappingProxyType(
    {
        "fits_image_data": MappingProxyType(
            {
                "data": True,
                "handles": False,
                "meta": False,
                "hdu_types": False,
                "stats": False,
                "cpp": False,
            }
        ),
        "fits_table_data": MappingProxyType(
            {
                "data": True,
                "handles": False,
                "meta": False,
                "hdu_types": False,
                "stats": False,
                "cpp": False,
            }
        ),
        "fits_header_metadata": MappingProxyType(
            {
                "data": False,
                "handles": False,
                "meta": True,
                "hdu_types": True,
                "stats": False,
                "cpp": False,
            }
        ),
        "fits_header_hdu_metadata": MappingProxyType(
            {
                "data": False,
                "handles": False,
                "meta": True,
                "hdu_types": True,
                "stats": False,
                "cpp": False,
            }
        ),
        "all": MappingProxyType(
            {
                "data": True,
                "handles": True,
                "meta": True,
                "hdu_types": True,
                "stats": True,
                "cpp": True,
            }
        ),
    }
)


def cache_subsystem_policy(name: str) -> dict[str, bool]:
    """Return the concrete clear flags for a named FITS I/O cache subsystem."""
    try:
        return dict(IO_CACHE_SUBSYSTEMS[name])
    except KeyError as exc:
        valid = ", ".join(sorted(IO_CACHE_SUBSYSTEMS))
        raise KeyError(
            f"unknown FITS I/O cache subsystem {name!r}; valid: {valid}"
        ) from exc


def clear_cache_subsystem(
    name: str,
    *,
    cpp_module: Any = None,
) -> None:
    """Clear one named FITS I/O cache subsystem."""
    policy = cache_subsystem_policy(name)
    clear_file_cache(
        data=policy["data"],
        handles=policy["handles"],
        meta=policy["meta"],
        hdu_types=policy["hdu_types"],
        stats=policy["stats"],
        cpp=policy["cpp"],
        cpp_module=cpp_module,
    )


def path_signature(path: str) -> tuple[int, int, int] | None:
    """Return a compact file identity for stale-cache detection."""
    try:
        st = os.stat(path)
    except Exception:
        return None
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return (int(st.st_size), int(mtime_ns), int(st.st_ino))


def get_cached_handle(path: str, handle_cache_capacity: int) -> tuple[Any, bool]:
    """Open a fresh, privately-owned CFITSIO handle for this call.

    Per-path handle caching was removed: sharing one ``fitsfile*`` across threads
    corrupts CFITSIO's position state (§4 Option A). ``cached`` is always ``False``
    so every caller closes the handle it received. ``handle_cache_capacity`` is
    ignored (kept only so existing call sites need no signature churn).
    """
    import torchfits._C as cpp

    from .paths import guard_fits_path

    del handle_cache_capacity
    guard_fits_path(path)
    return cpp.open_fits_file(path, "r"), False


def get_cached_hdu_type(path: str, hdu: int) -> str | None:
    """Return a cached HDU payload type for path/HDU dispatch, if known."""
    value = signature_cached_get(hdu_type_cache, (path, hdu))
    return None if value is None else str(value)


def set_cached_hdu_type(path: str, hdu: int, hdu_type: str | None) -> None:
    """Record an HDU payload type for path/HDU dispatch."""
    if not hdu_type:
        return
    signature_cached_set(hdu_type_cache, (path, hdu), hdu_type, 512)


def signature_cached_get(cache: Any, key: tuple[str, int]) -> Any:
    """Get an entry validated against the file's current stat signature.

    Policy/meta caches must not outlive the file state they describe: a
    rewrite that swaps payloads in place would otherwise keep serving stale
    dispatch decisions. Entries store ``(signature, value)``; a mismatch
    drops the entry and reports a miss.
    """
    with cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        cache.move_to_end(key)
        stored_sig, value = entry
        current_sig = path_signature(key[0])
        if (
            stored_sig is not None
            and current_sig is not None
            and stored_sig != current_sig
        ):
            del cache[key]
            return None
        return value


def signature_cached_set(
    cache: Any, key: tuple[str, int], value: Any, max_size: int
) -> None:
    """Store ``(signature, value)`` for :func:`signature_cached_get`."""
    with cache_lock:
        cache[key] = (path_signature(key[0]), value)
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def check_read_cache(
    *,
    path: str,
    hdu: Any,
    device: str,
    fp16: bool,
    bf16: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    return_header: bool,
    cache_capacity: int,
    invalidate_path: Any,
) -> tuple[bool, Any, Any]:
    """Check the Python-side read cache under the cache lock."""
    with cache_lock:
        return _check_read_cache_locked(
            path=path,
            hdu=hdu,
            device=device,
            fp16=fp16,
            bf16=bf16,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            return_header=return_header,
            cache_capacity=cache_capacity,
            invalidate_path=invalidate_path,
        )


def _check_read_cache_locked(
    *,
    path: str,
    hdu: Any,
    device: str,
    fp16: bool,
    bf16: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    return_header: bool,
    cache_capacity: int,
    invalidate_path: Any,
) -> tuple[bool, Any, Any]:
    """Check the Python-side read cache and update cache counters."""
    import torch

    cache_stats["total_requests"] += 1
    use_cache = cache_capacity > 0

    cache_key = None
    if use_cache:
        try:
            cache_key = (
                path,
                hdu,
                device,
                fp16,
                bf16,
                tuple(columns) if columns else None,
                start_row,
                num_rows,
                return_header,
            )
        except TypeError:
            cache_key = None

        if cache_key is not None:
            if cache_key in file_cache:
                cached_entry = file_cache.pop(cache_key)
                if isinstance(cached_entry, tuple) and len(cached_entry) == 3:
                    cached_data, cached_header, cached_sig = cached_entry
                else:
                    cached_data, cached_header = cached_entry
                    cached_sig = None

                cur_sig = path_signature(path)
                stale_cache_entry = (
                    cached_sig is not None
                    and cur_sig is not None
                    and cached_sig != cur_sig
                )
                if stale_cache_entry:
                    invalidate_path(path)
                    cache_stats["misses"] += 1
                else:
                    cache_stats["hits"] += 1
                    file_cache[cache_key] = (cached_data, cached_header, cached_sig)

                    out_data: Any = cached_data
                    if str(device) != "cpu":
                        if isinstance(cached_data, torch.Tensor):
                            out_data = to_device(cached_data, device)
                        elif isinstance(cached_data, dict):
                            new_data: dict[str, Any] = {}
                            for key, value in cached_data.items():
                                if isinstance(value, torch.Tensor):
                                    new_data[key] = to_device(value, device)
                                elif isinstance(value, list):
                                    new_data[key] = [
                                        to_device(item, device)
                                        if isinstance(item, torch.Tensor)
                                        else item
                                        for item in value
                                    ]
                                else:
                                    new_data[key] = value
                            out_data = new_data
                    else:
                        # Hand out a private copy so callers mutating their
                        # result cannot alias (and corrupt) the cached entry.
                        out_data = _clone_read_value(cached_data)
                    return (
                        True,
                        ((out_data, cached_header) if return_header else out_data),
                        cache_key,
                    )
            else:
                cache_stats["misses"] += 1
        else:
            cache_stats["misses"] += 1
    else:
        cache_stats["misses"] += 1

    return False, None, cache_key


def _clone_read_value(value: Any) -> Any:
    """Deep-copy tensors inside a cached read result.

    The cache must never hand out (or hold) caller-owned buffers: a user who
    mutates ``result["flux"]`` in place would otherwise poison every later
    read served from the cache.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                out[key] = item.clone()
            elif isinstance(item, list):
                out[key] = [
                    entry.clone() if isinstance(entry, torch.Tensor) else entry
                    for entry in item
                ]
            else:
                out[key] = item
        return out
    if isinstance(value, list):
        return [_clone_read_value(entry) for entry in value]
    if isinstance(value, tuple):
        return tuple(_clone_read_value(entry) for entry in value)
    return value


def store_cached_read(cache_key: Any, value: Any, capacity: int) -> None:
    """Store one read result and evict the oldest entry under the cache lock.

    The stored copy is detached from the caller's tensors so later in-place
    mutation of the returned data cannot corrupt future cache hits.
    """
    if capacity <= 0 or cache_key is None:
        return
    with cache_lock:
        file_cache[cache_key] = _clone_read_value(value)
        while len(file_cache) > capacity:
            file_cache.popitem(last=False)
        cache_stats["cache_size"] = len(file_cache)


def _register_open_hdulist(path: str, handle: Any, hdulist: Any) -> None:
    """Register an HDUList's open file handle so mutations can close it."""
    try:
        real = os.path.realpath(path)
    except Exception:
        real = os.path.abspath(path)
    _open_hdulist_registry.setdefault(real, []).append((handle, hdulist))
    try:
        hdulist._registry_key = real
    except Exception:
        pass


def _close_hdulist_for_path(path: str) -> None:
    """Close and unregister any open HDUList file handle for *path*.

    Called by :func:`invalidate_path_caches` before file mutations so that
    the C++ CFITSIO backend can open the file for writing.
    """
    try:
        real = os.path.realpath(path)
    except Exception:
        real = os.path.abspath(path)
    entries = _open_hdulist_registry.pop(real, [])
    if not entries:
        return
    for handle, hdulist in entries:
        try:
            handle.close()
        except Exception:
            pass
        # Prevent double-close when HDUList.__exit__ calls close() later.
        try:
            hdulist._file_handle = None
            hdulist._registry_key = None
        except Exception:
            pass


def invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches and open handles for one path."""
    # Close any open HDUList file handle so mutations can open the file for writing.
    _close_hdulist_for_path(path)

    # Drop every thread's cached C++ TableReader for the path: a cached reader
    # holds the file open READONLY (CFITSIO refuses a READWRITE reopen while
    # registered) and its cached column metadata goes stale once the file is
    # mutated.
    try:
        import torchfits._C as cpp

        cpp.evict_cached_reader(path)
    except Exception:
        pass

    with cache_lock:
        _invalidate_path_caches_locked(path)


def _invalidate_path_caches_locked(path: str) -> None:
    """Remove all Python cache entries for *path*; caller holds cache_lock."""

    stale_data_keys = [
        key
        for key in list(file_cache.keys())
        if isinstance(key, tuple) and key and key[0] == path
    ]
    for key in stale_data_keys:
        file_cache.pop(key, None)

    for key in [key for key in image_meta_cache.keys() if key[0] == path]:
        image_meta_cache.pop(key, None)
    for key in [key for key in header_cards_cache.keys() if key[0] == path]:
        header_cards_cache.pop(key, None)
    for key in [key for key in hdu_type_cache.keys() if key[0] == path]:
        hdu_type_cache.pop(key, None)
    for key in [key for key in cold_nommap_cache.keys() if key[0] == path]:
        cold_nommap_cache.pop(key, None)
    for key in [key for key in auto_mmap_cache.keys() if key[0] == path]:
        auto_mmap_cache.pop(key, None)

    auto_hdu_cache.pop(path, None)
    for key in [
        key
        for key in auto_hdu_cache.keys()
        if isinstance(key, tuple) and key and key[0] == path
    ]:
        auto_hdu_cache.pop(key, None)


def get_cache_performance() -> dict[str, Any]:
    """Return root FITS I/O cache performance statistics."""
    with cache_lock:
        total = cache_stats["total_requests"]
        hits = cache_stats["hits"]
        misses = cache_stats["misses"]
        cache_size = cache_stats["cache_size"]

    return {
        "cache_size": cache_size,
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
    }


def reset_cache_stats() -> None:
    """Reset root FITS I/O cache counters in place."""
    with cache_lock:
        cache_stats.clear()
        cache_stats.update(_CACHE_STATS_DEFAULT)


def clear_python_caches(
    *,
    data: bool = True,
    handles: bool = True,
    meta: bool = True,
    hdu_types: bool = True,
    stats: bool = True,
) -> None:
    """Clear Python-side root FITS I/O caches."""
    with cache_lock:
        if data:
            file_cache.clear()

        if meta:
            image_meta_cache.clear()
            header_cards_cache.clear()
            cold_nommap_cache.clear()
            auto_mmap_cache.clear()

        if hdu_types:
            hdu_type_cache.clear()
            auto_hdu_cache.clear()

        if stats:
            cache_stats.clear()
            cache_stats.update(_CACHE_STATS_DEFAULT)


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
    """Clear Python/C++ FITS I/O caches."""
    clear_python_caches(
        data=data,
        handles=handles,
        meta=meta,
        hdu_types=hdu_types,
        stats=stats,
    )

    if not cpp:
        return

    try:
        if cpp_module is None:
            import torchfits._C as _cpp

            cpp_module = _cpp

        cpp_module.clear_file_cache()
        if hasattr(cpp_module, "clear_shared_read_meta_cache"):
            cpp_module.clear_shared_read_meta_cache()
    except (AttributeError, RuntimeError) as exc:
        warnings.warn(
            f"clear_file_cache: C++ cache clear skipped ({exc!s})",
            RuntimeWarning,
            stacklevel=2,
        )
