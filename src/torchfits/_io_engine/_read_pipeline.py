"""Private read dispatch helpers for root FITS I/O."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import fields
from typing import Any, cast

import torch
from torch import Tensor

from ..fits_schema import bit_column_names, unsigned_column_dtypes_from_header
from ..hdu import Header
from .options import ReadOptions
from .caches import (
    cache_stats,
    get_cached_hdu_type,
)

_CPP_ATTR_CACHE: dict[
    str, bool
] = {}  # Cached ReadOptions field names — computed once at module import.
_READ_OPTION_FIELD_NAMES: frozenset[str] = frozenset(
    f.name for f in fields(ReadOptions)
)


def _is_cpp_module_mocked(cpp_module: Any) -> bool:
    """Detect if *cpp_module* (or its ``read_full``) is a mock object.

    Avoid importing ``unittest.mock``. Mock objects set ``_mock_name`` when
    instantiated; real extension modules never carry this attribute. Covers
    both ``patch("torchfits.cpp", ...)`` (module mocked) and patching only
    ``read_full`` on a real module. The CPU fast path is skipped for mocks so
    tests that patch the C++ module fall through to the generic/fallback path.
    """
    if getattr(cpp_module, "_mock_name", None) is not None:
        return True
    read_full = getattr(cpp_module, "read_full", None)
    return getattr(read_full, "_mock_name", None) is not None


def _cpp_has(cpp_module: Any, attr: str) -> bool:
    try:
        return _CPP_ATTR_CACHE[attr]
    except KeyError:
        result = hasattr(cpp_module, attr)
        _CPP_ATTR_CACHE[attr] = result
        return result


def _clear_cpp_attr_cache() -> None:
    """Reset the cached ``hasattr(cpp_module, ...)`` results.

    The cache assumes the C++ extension's attribute surface is stable for the
    process lifetime. Tests that reload the extension (``importlib.reload``) or
    monkeypatch capability attributes must call this to avoid stale ``False``
    entries for attributes that are now present.
    """
    _CPP_ATTR_CACHE.clear()


def _bit_columns_from_header(header: Header | None) -> set[str]:
    if not header:
        return set()
    return bit_column_names(header)


def _unsigned_columns_from_header(header: Header | None) -> dict[str, torch.dtype]:
    if not header:
        return {}
    return unsigned_column_dtypes_from_header(header)


def _coerce_bit_table_columns(table_data: Any, header: Header | None) -> Any:
    if not isinstance(table_data, dict):
        return table_data
    bit_columns = _bit_columns_from_header(header)
    if not bit_columns:
        return table_data
    out = dict(table_data)
    for name in bit_columns:
        value = out.get(name)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            out[name] = value.to(dtype=torch.bool)
    return out


def _coerce_unsigned_table_columns(table_data: Any, header: Header | None) -> Any:
    if not isinstance(table_data, dict):
        return table_data
    unsigned_columns = _unsigned_columns_from_header(header)
    if not unsigned_columns:
        return table_data
    out = dict(table_data)
    for name, dtype in unsigned_columns.items():
        value = out.get(name)
        if isinstance(value, torch.Tensor) and (
            value.dtype.is_floating_point
            or value.dtype == torch.int32
            or value.dtype == torch.int64
        ):
            out[name] = value.to(dtype=dtype)
    return out


def _apply_unsigned_offset(
    data: torch.Tensor,
    dtype: torch.dtype,
    offset: int,
    *,
    device: str | None = None,
) -> torch.Tensor:
    """Convert signed FITS storage to unsigned dtype with minimal widening."""
    if device is not None and device != "cpu":
        data = data.to(device=device)
    if dtype == torch.uint16 and data.dtype == torch.int16:
        return (data.to(torch.int32) + offset).to(torch.uint16)
    if dtype == torch.uint32 and data.dtype == torch.int32:
        return (data.to(torch.int64) + offset).to(torch.uint32)
    if dtype == torch.int8 and data.dtype == torch.uint8 and offset == -128:
        # FITS signed-byte (BZERO=-128): XOR path matches _apply_scale_on_device.
        return (data ^ 0x80).view(torch.int8)
    return data.to(torch.int64).add_(offset).to(dtype=dtype)


def _apply_scale_on_device(
    data: torch.Tensor,
    *,
    scaled: bool,
    bscale: float,
    bzero: float,
    device: str,
) -> torch.Tensor:
    """Apply BSCALE/BZERO on ``device`` while preserving narrow integer H2D when possible."""
    if not scaled:
        return data.to(device=device)

    if data.dtype == torch.int16 and bscale == 1.0 and bzero == 32768.0:
        # Unsigned-16: convert on host, then one device copy (like signed-byte).
        logical = _apply_unsigned_offset(data, torch.uint16, 32768, device="cpu")
        return logical.to(device=device)
    if data.dtype == torch.int32 and bscale == 1.0 and bzero == 2147483648.0:
        logical = _apply_unsigned_offset(data, torch.uint32, 2147483648, device="cpu")
        return logical.to(device=device)
    if data.dtype == torch.uint8 and bscale == 1.0 and bzero == -128.0:
        # FITS signed-byte (BZERO=-128): convert on host, then one device copy.
        # Avoids CUDA/MPS int16/subtract/int8 cast kernels on tiny payloads.
        logical = (data ^ 0x80).view(torch.int8)
        return logical.to(device=device)

    # Generic BSCALE/BZERO: host float scale, then one H2D. No size gate —
    # device launch tax is never free for this one mul/add.
    out = data.to(dtype=torch.float32)
    if bscale != 1.0:
        out = out.mul(bscale)
    if bzero != 0.0:
        out = out.add(bzero)
    return out.to(device=device)


# ---------------------------------------------------------------------------
# Option parsing and validation (extracted from read_unified — A2 refactor)
# ---------------------------------------------------------------------------


def _parse_read_options(
    options: ReadOptions | None, kwargs: dict[str, Any]
) -> ReadOptions:
    """Merge an explicit ReadOptions with kwargs into a single options object."""
    unknown = sorted(set(kwargs) - _READ_OPTION_FIELD_NAMES)
    if unknown:
        raise TypeError(
            "read() got unexpected keyword argument(s): "
            + ", ".join(repr(k) for k in unknown)
        )
    if "handle_cache_capacity" in kwargs:
        warnings.warn(
            "handle_cache_capacity is deprecated and ignored: per-path handle "
            "caching was removed. Stop passing it; it will be dropped in a future "
            "release.",
            DeprecationWarning,
            stacklevel=3,
        )
    if options is not None:
        colliding = (set(kwargs) & _READ_OPTION_FIELD_NAMES) - {"mode"}
        if colliding:
            raise TypeError(
                "Pass either options= or individual read kwargs, not both; "
                f"collision on: {sorted(colliding)}"
            )
        opts = copy.copy(options)
    else:
        opts = ReadOptions()
    for field_name in _READ_OPTION_FIELD_NAMES:
        if field_name in kwargs:
            setattr(opts, field_name, kwargs[field_name])
    return opts


def _validate_single_path_params(
    path: str, hdu: Any, device: str, mmap: bool | str, mode: str
) -> tuple[bool, bool]:
    """Validate path/hdu/device/mmap/mode; return (force_image, force_table)."""
    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    if path.lower().endswith(".bz2"):
        raise ValueError(
            "CFITSIO does not support .bz2 compression natively. Please decompress the file first."
        )
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be a non-negative integer")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    if isinstance(mmap, str) and mmap.strip().lower() != "auto":
        raise ValueError("mmap must be bool or 'auto'")
    if not isinstance(mmap, (bool, str)):
        raise ValueError("mmap must be bool or 'auto'")
    mode = str(mode).strip().lower()
    if mode not in {"auto", "image", "table"}:
        raise ValueError("mode must be 'auto', 'image', or 'table'")
    return mode == "image", mode == "table"


# ---------------------------------------------------------------------------
# read_unified — the main unified read dispatcher
# ---------------------------------------------------------------------------


def read_unified(
    *,
    cpp_module: Any,
    path: Any,
    hdu: Any,
    device: str,
    mmap: bool | str,
    options: ReadOptions | None,
    return_header: bool,
    kwargs: dict[str, Any],
    autodetect_hdu: Callable[[str, int], int],
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_check_cache: Callable[..., tuple[bool, Any, Any]],
    read_header: Callable[[Any, int, bool], Any],
    debug_scale: bool,
    cold_nocache: bool,
    read_exc_types: tuple[type[BaseException], ...],
    logger: Any,
) -> Any:
    """Unified root FITS read dispatcher implementation."""
    # --- parse options ---
    opts = _parse_read_options(options, kwargs)

    fp16 = opts.fp16
    bf16 = opts.bf16
    raw_scale = opts.raw_scale
    scale_on_device = opts.scale_on_device
    use_cache = opts.use_cache
    columns = opts.columns
    start_row = opts.start_row
    num_rows = opts.num_rows
    cache_capacity = opts.cache_capacity
    handle_cache_capacity = opts.handle_cache_capacity
    fast_header = opts.fast_header
    mode = opts.mode

    # --- validate ---
    if not path:
        raise ValueError("Path must be a non-empty string")

    from .paths import guard_fits_path

    if isinstance(path, (list, tuple)):
        if any(not isinstance(item_path, str) or not item_path for item_path in path):
            raise ValueError("Path must be a string or list of strings")
        for item_path in path:
            guard_fits_path(item_path)
        return _read_batch_paths(
            cpp_module=cpp_module,
            path=path,
            hdu=hdu,
            device=device,
            mmap=mmap,
            fp16=fp16,
            bf16=bf16,
            raw_scale=raw_scale,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            cache_capacity=cache_capacity,
            handle_cache_capacity=handle_cache_capacity,
            fast_header=fast_header,
            return_header=return_header,
            mode=mode,
            autodetect_hdu=autodetect_hdu,
            batch_to_device=batch_to_device,
            resolve_image_mmap=resolve_image_mmap,
            read_check_cache=read_check_cache,
            read_header=read_header,
            debug_scale=debug_scale,
            cold_nocache=cold_nocache,
            read_exc_types=read_exc_types,
            logger=logger,
            strict=bool(kwargs.get("strict", False)),
        )

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    guard_fits_path(path)

    if use_cache is not None and not isinstance(use_cache, bool):
        raise ValueError("use_cache must be bool when provided")
    if use_cache is True:
        if cache_capacity <= 0:
            cache_capacity = 10
        handle_cache_capacity = 0
    elif use_cache is False:
        cache_capacity = 0
        handle_cache_capacity = 0

    # Fixed: pass hdu then device (matches function signature order)
    force_image, force_table = _validate_single_path_params(
        path, hdu, device, mmap, mode
    )
    if force_image and (columns is not None or start_row != 1 or num_rows != -1):
        raise ValueError("mode='image' does not support table row/column options")

    # --- resolve HDU ---
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = autodetect_hdu(path, handle_cache_capacity)

    hdu_type_hint = get_cached_hdu_type(path, hdu) if isinstance(hdu, int) else None
    is_cached_table_hdu = force_table or (
        hdu_type_hint in {"ASCII_TABLE", "BINARY_TABLE"}
    )
    skip_generic_image_fast_path = is_cached_table_hdu

    # --- batch HDUs ---
    if (
        isinstance(hdu, (list, tuple))
        and not return_header
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        return _read_batch_hdus(
            cpp_module=cpp_module,
            path=path,
            hdu=hdu,
            device=device,
            mmap=mmap,
            fp16=fp16,
            bf16=bf16,
            raw_scale=raw_scale,
            scale_on_device=scale_on_device,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            cache_capacity=cache_capacity,
            handle_cache_capacity=handle_cache_capacity,
            fast_header=fast_header,
            return_header=return_header,
            batch_to_device=batch_to_device,
            autodetect_hdu=autodetect_hdu,
            resolve_image_mmap=resolve_image_mmap,
            read_check_cache=read_check_cache,
            read_header=read_header,
            debug_scale=debug_scale,
            cold_nocache=cold_nocache,
            read_exc_types=read_exc_types,
            logger=logger,
        )

    # --- strategy 1: CPU image fast path ---
    cpp_is_mocked = _is_cpp_module_mocked(cpp_module)

    if (
        scale_on_device
        and not raw_scale
        and device == "cpu"
        and not return_header
        and isinstance(hdu, int)
        and not cpp_is_mocked
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not is_cached_table_hdu
    ):
        result, fallback = _read_cpu_fast_path(
            cpp_module=cpp_module,
            path=path,
            hdu=hdu,
            mmap=mmap,
            cache_capacity=cache_capacity,
            fp16=fp16,
            bf16=bf16,
            force_image=force_image,
            resolve_image_mmap=resolve_image_mmap,
            cache_stats=cache_stats,
            read_exc_types=read_exc_types,
            debug_scale=debug_scale,
            logger=logger,
        )
        if not fallback:
            return result
        skip_generic_image_fast_path = True

    # --- strategy 2: generic image fast path ---
    if (
        not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not skip_generic_image_fast_path
    ):
        result = _read_generic_fast_path(
            cpp_module=cpp_module,
            path=path,
            hdu=hdu,
            device=device,
            mmap=mmap,
            cache_capacity=cache_capacity,
            fp16=fp16,
            bf16=bf16,
            raw_scale=raw_scale,
            scale_on_device=scale_on_device,
            force_image=force_image,
            debug_scale=debug_scale,
            cold_nocache=cold_nocache,
            resolve_image_mmap=resolve_image_mmap,
            read_exc_types=read_exc_types,
            logger=logger,
        )
        if result is not None:
            return result

    # --- strategy 3: full fallback ---
    from ._read_pipeline_fallback import read_fallback

    return read_fallback(
        cpp_module=cpp_module,
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        force_image=force_image,
        force_table=force_table,
        hdu_type_hint=hdu_type_hint,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        read_check_cache=read_check_cache,
        resolve_image_mmap=resolve_image_mmap,
        read_header=read_header,
    )


# ---------------------------------------------------------------------------
# Batch dispatch helpers
# ---------------------------------------------------------------------------


def _read_batch_paths(
    *,
    cpp_module: Any,
    path: list[str] | tuple[str, ...],
    hdu: Any,
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    cache_capacity: int,
    handle_cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    mode: str,
    autodetect_hdu: Callable[[str, int], int],
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_check_cache: Callable[..., tuple[bool, Any, Any]],
    read_header: Callable[[Any, int, bool], Any],
    debug_scale: bool,
    cold_nocache: bool,
    read_exc_types: tuple[type[BaseException], ...],
    logger: Any,
    strict: bool = False,
) -> list[Any]:
    """Dispatch a list of FITS paths through batch C++ or recursive reads."""
    hdu_batch = hdu
    if hdu_batch is None or (
        isinstance(hdu_batch, str) and hdu_batch.strip().lower() == "auto"
    ):
        if not path:
            raise ValueError("Batch read requires a non-empty path list")
        hdu_batch = autodetect_hdu(path[0], handle_cache_capacity)
    if not isinstance(hdu_batch, int):
        raise ValueError("Batch read requires a single integer HDU")
    hdu = hdu_batch

    if mmap is not False:
        for item_path in path:
            if item_path.lower().endswith(".bz2"):
                raise ValueError(
                    "CFITSIO does not support .bz2 compression natively. Please decompress the file first."
                )
        try:
            data_list = cpp_module.read_images_batch(list(path), hdu)
            if device != "cpu":
                data_list = batch_to_device(data_list, device)
            return cast(list[Any], data_list)
        except read_exc_types as exc:
            if strict:
                raise
            logger.debug(
                "read_images_batch failed, falling back per file: %s",
                exc,
                exc_info=True,
            )

    data_list = []
    for item_path in path:
        data_list.append(
            read_unified(
                cpp_module=cpp_module,
                path=item_path,
                hdu=hdu,
                device=device,
                mmap=mmap,
                options=None,
                return_header=return_header,
                kwargs=dict(
                    mode=mode,
                    fp16=fp16,
                    bf16=bf16,
                    raw_scale=raw_scale,
                    columns=columns,
                    start_row=start_row,
                    num_rows=num_rows,
                    cache_capacity=cache_capacity,
                    handle_cache_capacity=handle_cache_capacity,
                    fast_header=fast_header,
                ),
                autodetect_hdu=autodetect_hdu,
                batch_to_device=batch_to_device,
                resolve_image_mmap=resolve_image_mmap,
                read_check_cache=read_check_cache,
                read_header=read_header,
                debug_scale=debug_scale,
                cold_nocache=cold_nocache,
                read_exc_types=read_exc_types,
                logger=logger,
            )
        )
    return data_list


def _read_batch_hdus(
    *,
    cpp_module: Any,
    path: str,
    hdu: list[int] | tuple[int, ...],
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    scale_on_device: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    cache_capacity: int,
    handle_cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    autodetect_hdu: Callable[[str, int], int],
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_check_cache: Callable[..., tuple[bool, Any, Any]],
    read_header: Callable[[Any, int, bool], Any],
    debug_scale: bool,
    cold_nocache: bool,
    read_exc_types: tuple[type[BaseException], ...],
    logger: Any,
) -> Any:
    """Dispatch multiple HDUs from one FITS path."""
    if hasattr(cpp_module, "read_hdus_batch"):
        try:
            data = cpp_module.read_hdus_batch(path, list(hdu))
        except TypeError:
            effective_mmap = True if isinstance(mmap, str) else mmap
            data = cpp_module.read_hdus_batch(path, list(hdu), effective_mmap)
        if device != "cpu":
            data = batch_to_device(data, device)
        return data
    return [
        read_unified(
            cpp_module=cpp_module,
            path=path,
            hdu=item_hdu,
            device=device,
            mmap=mmap,
            options=None,
            return_header=return_header,
            kwargs=dict(
                fp16=fp16,
                bf16=bf16,
                raw_scale=raw_scale,
                scale_on_device=scale_on_device,
                columns=columns,
                start_row=start_row,
                num_rows=num_rows,
                cache_capacity=cache_capacity,
                handle_cache_capacity=handle_cache_capacity,
                fast_header=fast_header,
            ),
            autodetect_hdu=autodetect_hdu,
            batch_to_device=batch_to_device,
            resolve_image_mmap=resolve_image_mmap,
            read_check_cache=read_check_cache,
            read_header=read_header,
            debug_scale=debug_scale,
            cold_nocache=cold_nocache,
            read_exc_types=read_exc_types,
            logger=logger,
        )
        for item_hdu in hdu
    ]


# ---------------------------------------------------------------------------
# Image fast path helpers
# ---------------------------------------------------------------------------


def read_scaled_cpu_fast(
    cpp_module: Any, path: str, hdu: int = 0, mmap: bool = True
) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    if not _cpp_has(cpp_module, "read_full_raw_with_scale"):
        raise RuntimeError("Scaled fast path unavailable in this build")

    data, scaled, bscale, bzero = cpp_module.read_full_raw_with_scale(path, hdu, mmap)
    if scaled:
        data = data.to(dtype=torch.float32)
        if bscale != 1.0:
            data.mul_(bscale)
        if bzero != 0.0:
            data.add_(bzero)
    return cast(Tensor, data)


def _read_cpu_fast_path(
    *,
    cpp_module: Any,
    path: str,
    hdu: int,
    mmap: bool | str,
    cache_capacity: int,
    fp16: bool,
    bf16: bool,
    force_image: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    cache_stats: dict[str, int],
    read_exc_types: tuple[type[BaseException], ...],
    debug_scale: bool,
    logger: Any,
) -> tuple[Tensor | None, bool]:
    """Try the CPU image fast path; return (data, fallback_required)."""
    # One-shot full-image read: thin CFITSIO → Tensor (matches fitsio+from_numpy).
    # Handle-cache scaffolding (read_full_cached) is for persistent subset readers,
    # not single full reads — it lost ~15% to fitsio on hcompress.
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if cache_capacity <= 0 and hasattr(cpp_module, "read_full_nocache"):
            data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
        else:
            data = cpp_module.read_full(path, hdu, effective_mmap)

        if fp16:
            data = data.to(torch.float16)
        elif bf16:
            data = data.to(torch.bfloat16)

        try:
            cache_stats["total_requests"] += 1
            cache_stats["misses"] += 1
        except Exception:
            pass
        return data, False
    except read_exc_types as exc:
        if debug_scale or logger.isEnabledFor(10):
            logger.debug(
                "read: CPU image fast-path fallback for %r hdu=%s: %s",
                path,
                hdu,
                exc,
                exc_info=True,
            )
        if force_image:
            raise
        return None, True


def _read_generic_fast_path(
    *,
    cpp_module: Any,
    path: str,
    hdu: int,
    device: str,
    mmap: bool | str,
    cache_capacity: int,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    scale_on_device: bool,
    force_image: bool,
    debug_scale: bool,
    cold_nocache: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_exc_types: tuple[type[BaseException], ...],
    logger: Any,
) -> Tensor | None:
    """Try the generic image fast path; return None when fallback is required."""
    _ = cold_nocache
    _ = read_exc_types  # callers still pass; fallback catch is intentionally narrow
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if scale_on_device and not raw_scale:
            # Thin device path: logical host tensor → one H2D. No size heuristics.
            if device != "cpu":
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: thin_device_logical")
                data = cpp_module.read_full(path, hdu, effective_mmap)
                data = data.to(device)
            else:
                # CPU logical scale is applied inside read_full; do not detour
                # through read_full_raw_with_scale (extra host ops vs fitsio).
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: thin_cpu_logical")
                if cache_capacity <= 0 and hasattr(cpp_module, "read_full_nocache"):
                    data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
                else:
                    data = cpp_module.read_full(path, hdu, effective_mmap)
        elif raw_scale:
            if debug_scale:
                print("TORCHFITS_DEBUG_SCALE: raw_scale")
            if not effective_mmap and _cpp_has(cpp_module, "read_full_unmapped_raw"):
                data = cpp_module.read_full_unmapped_raw(path, hdu)
            else:
                data = cpp_module.read_full_raw(path, hdu, effective_mmap)
        else:
            if debug_scale:
                print("TORCHFITS_DEBUG_SCALE: unscaled")
            if cache_capacity <= 0 and hasattr(cpp_module, "read_full_nocache"):
                data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
            else:
                data = cpp_module.read_full(path, hdu, effective_mmap)

        if fp16:
            data = data.to(torch.float16)
        elif bf16:
            data = data.to(torch.bfloat16)

        if device != "cpu" and data.device.type == "cpu":
            data = data.to(device)

        return cast(Tensor, data)
    except ValueError:
        raise
    except (RuntimeError, OSError, MemoryError) as exc:
        if force_image:
            raise
        if logger.isEnabledFor(10):
            logger.debug(
                "read: generic image fast-path fallback for %r hdu=%s: %s",
                path,
                hdu,
                exc,
                exc_info=True,
            )
        return None
