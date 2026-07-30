"""Python-side cache state for root FITS I/O dispatch."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from types import MappingProxyType
from typing import Any


_CACHE_STATS_DEFAULT = {
    "total_requests": 0,
    "hits": 0,
    "misses": 0,
    "cache_size": 0,
}

cache_stats: dict[str, int] = dict(_CACHE_STATS_DEFAULT)
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
    sig = (path, hdu)
    cached = hdu_type_cache.get(sig)
    if cached is not None:
        hdu_type_cache.move_to_end(sig)
    return cached


def set_cached_hdu_type(path: str, hdu: int, hdu_type: str | None) -> None:
    """Record an HDU payload type for path/HDU dispatch."""
    if not hdu_type:
        return
    sig = (path, hdu)
    hdu_type_cache[sig] = hdu_type
    hdu_type_cache.move_to_end(sig)
    while len(hdu_type_cache) > 512:
        hdu_type_cache.popitem(last=False)


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

                    if device != "cpu":
                        if isinstance(cached_data, torch.Tensor):
                            cached_data = cached_data.to(device)
                        elif isinstance(cached_data, dict):
                            new_data = {}
                            for key, value in cached_data.items():
                                if isinstance(value, torch.Tensor):
                                    new_data[key] = value.to(device)
                                else:
                                    new_data[key] = value
                            cached_data = new_data

                    return (
                        True,
                        (
                            (cached_data, cached_header)
                            if return_header
                            else cached_data
                        ),
                        cache_key,
                    )
            else:
                cache_stats["misses"] += 1
        else:
            cache_stats["misses"] += 1
    else:
        cache_stats["misses"] += 1

    return False, None, cache_key


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
    total = cache_stats["total_requests"]
    hits = cache_stats["hits"]
    misses = cache_stats["misses"]

    return {
        "cache_size": cache_stats["cache_size"],
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
    }


def reset_cache_stats() -> None:
    """Reset root FITS I/O cache counters in place."""
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
        reset_cache_stats()


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
