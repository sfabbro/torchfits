"""Image metadata and mmap policy helpers for FITS image reads."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, cast

from .caches import (
    auto_mmap_cache,
    cache_lock,
    cold_nommap_cache,
    image_meta_cache,
)
from ..hdu import Header


ImageMeta = tuple[int, int, tuple[int, ...], float, float, bool]


def _cache_get(cache: Any, key: Any) -> Any:
    with cache_lock:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _cache_set(cache: Any, key: Any, value: Any, max_size: int) -> None:
    with cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def _parse_image_meta(header_data: Mapping[str, Any]) -> ImageMeta:
    bitpix_raw = header_data.get("BITPIX", 0)
    naxis_raw = header_data.get("NAXIS", 0)
    bitpix = int(bitpix_raw) if bitpix_raw is not None else 0
    naxis = int(naxis_raw) if naxis_raw is not None else 0
    try:
        bscale = float(header_data.get("BSCALE", 1.0))
    except Exception:
        bscale = 1.0
    try:
        bzero = float(header_data.get("BZERO", 0.0))
    except Exception:
        bzero = 0.0
    dims = []
    for i in range(1, naxis + 1):
        key = f"NAXIS{i}"
        if key in header_data:
            try:
                dims.append(int(header_data.get(key, 0)))
            except Exception:
                break
    zimage = header_data.get("ZIMAGE", False)
    if isinstance(zimage, str):
        zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
    xtension = str(header_data.get("XTENSION", "")).strip().upper()
    has_compression_keys = any(
        k in header_data for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
    )
    is_compressed = bool(zimage) or (xtension == "BINTABLE" and has_compression_keys)
    return (bitpix, naxis, tuple(dims), bscale, bzero, is_compressed)


def _skinny_image_meta(path: str, hdu: int, cpp_module: Any) -> ImageMeta | None:
    """Build ImageMeta via skinny CFITSIO queries (no full header dump)."""
    try:
        bitpix, torch_shape = cpp_module.read_shape(path, hdu)
    except Exception:
        return None
    bitpix = int(bitpix)
    # read_shape returns torch order; ImageMeta dims are FITS NAXIS1..n order.
    dims = tuple(int(d) for d in reversed(tuple(torch_shape)))
    naxis = len(dims)

    def _key(name: str, default: Any) -> Any:
        try:
            return cpp_module.read_keys(path, hdu, [name])[name]
        except Exception:
            return default

    try:
        bscale = float(_key("BSCALE", 1.0))
    except Exception:
        bscale = 1.0
    try:
        bzero = float(_key("BZERO", 0.0))
    except Exception:
        bzero = 0.0

    zimage = _key("ZIMAGE", False)
    if isinstance(zimage, str):
        zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
    xtension = str(_key("XTENSION", "") or "").strip().upper()
    has_compression_keys = any(
        _key(k, None) is not None for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
    )
    is_compressed = bool(zimage) or (xtension == "BINTABLE" and has_compression_keys)
    return (bitpix, naxis, dims, bscale, bzero, is_compressed)


def get_image_meta(
    path: str, hdu: int, *, cpp_module: Any | None = None
) -> ImageMeta | None:
    """Fetch and cache compact FITS image metadata for policy decisions."""
    if cpp_module is None:
        import torchfits._C as _cpp

        cpp_module = _cpp

    sig = (path, hdu)
    cached = _cache_get(image_meta_cache, sig)
    if cached is not None:
        return cast(ImageMeta, cached)

    meta = _skinny_image_meta(path, hdu, cpp_module)
    if meta is None:
        try:
            meta = _parse_image_meta(Header(cpp_module.read_header_dict(path, hdu)))
        except Exception:
            meta = None

    _cache_set(image_meta_cache, sig, meta, 256)
    return meta


def get_image_meta_from_handle(
    file_handle: Any,
    path: str,
    hdu: int,
    *,
    read_header: Callable[[Any, int, bool], Mapping[str, Any]],
) -> ImageMeta | None:
    """Fetch image metadata using an already-open FITS handle."""
    sig = (path, hdu)
    cached = _cache_get(image_meta_cache, sig)
    if cached is not None:
        return cast(ImageMeta, cached)

    try:
        meta = _parse_image_meta(read_header(file_handle, hdu, True))
    except Exception:
        meta = None

    _cache_set(image_meta_cache, sig, meta, 256)
    return meta


def should_use_cold_nommap(
    path: str,
    hdu: int,
    cache_capacity: int,
    mmap: bool,
    *,
    force_cold_nommap: bool = False,
    get_image_meta_func: Callable[[str, int], ImageMeta | None] = get_image_meta,
) -> bool:
    """Return whether auto mmap should prefer direct reads for this image."""
    _ = cache_capacity
    if not mmap:
        return False
    if force_cold_nommap:
        return True

    cached = _cache_get(cold_nommap_cache, (path, hdu))
    if cached is not None:
        return bool(cached)

    try:
        file_size = os.path.getsize(path)
        if file_size < (1 << 20):
            _cache_set(cold_nommap_cache, (path, hdu), False, 512)
            return False
    except Exception:
        _cache_set(cold_nommap_cache, (path, hdu), False, 512)
        return False

    meta = _cache_get(image_meta_cache, (path, hdu))
    if meta is None:
        meta = get_image_meta_func(path, hdu)
    if not meta:
        _cache_set(cold_nommap_cache, (path, hdu), False, 512)
        return False

    try:
        bitpix = int(meta[0])
    except Exception:
        _cache_set(cold_nommap_cache, (path, hdu), False, 512)
        return False

    is_compressed = False
    if len(meta) >= 6:
        try:
            is_compressed = bool(meta[5])
        except Exception:
            is_compressed = False
    if is_compressed:
        _cache_set(cold_nommap_cache, (path, hdu), False, 512)
        return False

    if bitpix in (16, 32, -32):
        _cache_set(cold_nommap_cache, (path, hdu), True, 512)
        return True

    _cache_set(cold_nommap_cache, (path, hdu), False, 512)
    return False


def resolve_image_mmap(
    path: str,
    hdu: int,
    mmap: bool | str,
    cache_capacity: int,
    *,
    get_image_meta_func: Callable[[str, int], ImageMeta | None] = get_image_meta,
    should_use_cold_nommap_func: Callable[[str, int, int, bool], bool] | None = None,
) -> bool:
    """Resolve bool/'auto' mmap policy for image reads."""
    if isinstance(mmap, bool):
        return mmap

    if isinstance(mmap, str):
        mode = mmap.strip().lower()
        if mode != "auto":
            raise ValueError("mmap must be bool or 'auto'")

        sig = (path, hdu)
        cached = _cache_get(auto_mmap_cache, sig)
        if cached is not None:
            return bool(cached)

        meta = _cache_get(image_meta_cache, (path, hdu))
        if meta is None:
            meta = get_image_meta_func(path, hdu)

        if meta is not None and len(meta) >= 6:
            try:
                if bool(meta[5]):
                    _cache_set(auto_mmap_cache, sig, False, 512)
                    return False
            except Exception:
                pass

        if should_use_cold_nommap_func is None:
            resolved = not should_use_cold_nommap(
                path,
                hdu,
                cache_capacity,
                True,
                get_image_meta_func=get_image_meta_func,
            )
        else:
            resolved = not should_use_cold_nommap_func(path, hdu, cache_capacity, True)
        _cache_set(auto_mmap_cache, sig, bool(resolved), 512)
        return bool(resolved)

    raise ValueError("mmap must be bool or 'auto'")
