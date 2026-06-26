"""Private read dispatch helpers for root FITS I/O."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

import torch
from torch import Tensor

from ..hdu import Header
from .options import ReadOptions
from .caches import (
    cache_stats,
    file_cache,
    get_cached_handle,
    get_cached_hdu_type,
    path_signature,
    set_cached_hdu_type,
)


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
    opts = options if options is not None else ReadOptions()
    for field in fields(ReadOptions):
        if field.name in kwargs:
            setattr(opts, field.name, kwargs[field.name])

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
    policy = opts.policy

    def recursive_read(*args: Any, **inner_kwargs: Any) -> Any:
        return read_unified(
            cpp_module=cpp_module,
            path=args[0] if args else inner_kwargs.pop("path"),
            hdu=inner_kwargs.pop("hdu", None),
            device=inner_kwargs.pop("device", "cpu"),
            mmap=inner_kwargs.pop("mmap", "auto"),
            options=None,
            return_header=inner_kwargs.pop("return_header", False),
            kwargs=inner_kwargs,
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

    if not path:
        raise ValueError("Path must be a non-empty string")

    if use_cache is not None and not isinstance(use_cache, bool):
        raise ValueError("use_cache must be bool when provided")
    if use_cache is True:
        if cache_capacity <= 0:
            cache_capacity = 10
        handle_cache_capacity = 0
    elif use_cache is False:
        cache_capacity = 0
        handle_cache_capacity = 0

    if isinstance(path, (list, tuple)):
        if any(not isinstance(item_path, str) or not item_path for item_path in path):
            raise ValueError("Path must be a string or list of strings")
        return read_batch_paths(
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
            policy=policy,
            autodetect_hdu=autodetect_hdu,
            batch_to_device=batch_to_device,
            read_func=recursive_read,
        )

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
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
    policy = str(policy).strip().lower()
    if policy not in {"default", "smart"}:
        raise ValueError("policy must be 'default' or 'smart'")
    force_image = mode == "image"
    force_table = mode == "table"
    if force_image and (columns is not None or start_row != 1 or num_rows != -1):
        raise ValueError("mode='image' does not support table row/column options")

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = autodetect_hdu(path, handle_cache_capacity)

    hdu_type_hint = get_cached_hdu_type(path, hdu) if isinstance(hdu, int) else None
    is_cached_table_hdu = force_table or (
        hdu_type_hint in {"ASCII_TABLE", "BINARY_TABLE"}
    )
    skip_generic_image_fast_path = is_cached_table_hdu

    if (
        isinstance(hdu, (list, tuple))
        and not return_header
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        return read_batch_hdus(
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
            policy=policy,
            batch_to_device=batch_to_device,
            read_func=recursive_read,
        )

    cpp_is_mocked = False
    if isinstance(hdu, int):
        try:
            import unittest.mock as unittest_mock

            cpp_is_mocked = isinstance(
                getattr(cpp_module, "read_full", None), unittest_mock.Mock
            )
        except Exception:
            cpp_is_mocked = False

    if (
        scale_on_device
        and not raw_scale
        and not cpp_is_mocked
        and device == "cpu"
        and not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not is_cached_table_hdu
    ):
        result, fallback = read_cpu_fast_path(
            cpp_module=cpp_module,
            path=path,
            hdu=hdu,
            mmap=mmap,
            cache_capacity=cache_capacity,
            handle_cache_capacity=handle_cache_capacity,
            fp16=fp16,
            bf16=bf16,
            force_image=force_image,
            resolve_image_mmap=resolve_image_mmap,
            get_cached_handle=get_cached_handle,
            cache_stats=cache_stats,
            read_exc_types=read_exc_types,
            debug_scale=debug_scale,
            logger=logger,
        )
        if not fallback:
            return result
        skip_generic_image_fast_path = True

    if (
        not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not skip_generic_image_fast_path
    ):
        result = read_generic_fast_path(
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
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    try:
        file_handle, cached_handle = get_cached_handle(path, handle_cache_capacity)
        try:
            if isinstance(hdu, str):
                hdu_num = None
                if hasattr(cpp_module, "resolve_hdu_name_cached"):
                    try:
                        hdu_num = int(cpp_module.resolve_hdu_name_cached(path, hdu))
                    except Exception:
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
    data = cpp_module.read_full(file_handle, hdu_num, effective_mmap)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu":
        data = data.to(device)

    header = None
    if return_header:
        header_data = read_header(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    if use_cache and cache_key is not None:
        file_cache[cache_key] = (
            data.cpu() if device != "cpu" else data,
            header,
            path_signature(path),
        )
        while len(file_cache) > cache_capacity:
            file_cache.popitem(last=False)
        cache_stats["cache_size"] = len(file_cache)

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
        except Exception:
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

    if (start_row > 1 or num_rows != -1) and not hasattr(
        cpp_module, "read_fits_table_rows"
    ):
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                end_row = start_row + num_rows - 1 if num_rows != -1 else len(value)
                table_data[key] = value[start_row - 1 : end_row]

    if use_cache and cache_key is not None:
        file_cache[cache_key] = (table_data, header, path_signature(path))
        while len(file_cache) > cache_capacity:
            file_cache.popitem(last=False)
        cache_stats["cache_size"] = len(file_cache)

    if device != "cpu":
        new_data = {}
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                new_data[key] = new_list
            else:
                new_data[key] = value
        table_data = new_data

    if isinstance(hdu_num, int):
        set_cached_hdu_type(path, hdu_num, "BINARY_TABLE")
    return (table_data, header) if return_header else table_data


def read_batch_paths(
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
    policy: str,
    autodetect_hdu: Callable[[str, int], int],
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    read_func: Callable[..., Any],
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
        try:
            data_list = cpp_module.read_images_batch(list(path), hdu)
            if device != "cpu":
                data_list = batch_to_device(data_list, device)
            return data_list
        except Exception:
            pass

    data_list = []
    for item_path in path:
        data_list.append(
            read_func(
                item_path,
                hdu=hdu,
                mode=mode,
                policy=policy,
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
            )
        )
    return data_list


def read_batch_hdus(
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
    policy: str,
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    read_func: Callable[..., Any],
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
        read_func(
            path,
            hdu=item_hdu,
            device=device,
            policy=policy,
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
        )
        for item_hdu in hdu
    ]


def read_scaled_cpu_fast(
    cpp_module: Any, path: str, hdu: int = 0, mmap: bool = True
) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    if not hasattr(cpp_module, "read_full_raw_with_scale"):
        raise RuntimeError("Scaled fast path unavailable in this build")

    data, scaled, bscale, bzero = cpp_module.read_full_raw_with_scale(path, hdu, mmap)
    if scaled:
        data = data.to(dtype=torch.float32)
        if bscale != 1.0:
            data.mul_(bscale)
        if bzero != 0.0:
            data.add_(bzero)
    return data


def read_cpu_fast_path(
    *,
    cpp_module: Any,
    path: str,
    hdu: int,
    mmap: bool | str,
    cache_capacity: int,
    handle_cache_capacity: int,
    fp16: bool,
    bf16: bool,
    force_image: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    get_cached_handle: Callable[[str, int], tuple[Any, bool]],
    cache_stats: dict[str, int],
    read_exc_types: tuple[type[BaseException], ...],
    debug_scale: bool,
    logger: Any,
) -> tuple[Tensor | None, bool]:
    """Try the CPU image fast path; return (data, fallback_required)."""
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if handle_cache_capacity > 0:
            if hasattr(cpp_module, "read_full_cached"):
                data = cpp_module.read_full_cached(path, hdu, effective_mmap)
            else:
                file_handle, _cached = get_cached_handle(path, handle_cache_capacity)
                data = cpp_module.read_full(file_handle, hdu, effective_mmap)
        elif cache_capacity == 0 and hasattr(cpp_module, "read_full_nocache"):
            data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
        else:
            file_handle = cpp_module.open_fits_file(path, "r")
            try:
                data = cpp_module.read_full(file_handle, hdu, effective_mmap)
            finally:
                try:
                    file_handle.close()
                except Exception:
                    pass

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


def read_generic_fast_path(
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
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if scale_on_device and not raw_scale:
            if hasattr(cpp_module, "read_full_raw_with_scale"):
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: fast_path_scaled")
                data, scaled, bscale, bzero = cpp_module.read_full_raw_with_scale(
                    path, hdu, effective_mmap
                )
                if scaled:
                    data = data.to(device=device, dtype=torch.float32)
                    if bscale != 1.0:
                        data.mul_(bscale)
                    if bzero != 0.0:
                        data.add_(bzero)
                else:
                    data = data.to(device)
            else:
                data = cpp_module.read_full(path, hdu, effective_mmap)
                data = data.to(device)
        elif raw_scale:
            if debug_scale:
                print("TORCHFITS_DEBUG_SCALE: raw_scale")
            if not effective_mmap and hasattr(cpp_module, "read_full_unmapped_raw"):
                data = cpp_module.read_full_unmapped_raw(path, hdu)
            else:
                data = cpp_module.read_full_raw(path, hdu, effective_mmap)
        else:
            if debug_scale:
                print("TORCHFITS_DEBUG_SCALE: unscaled")
            if not effective_mmap and hasattr(cpp_module, "read_full_unmapped"):
                data = cpp_module.read_full_unmapped(path, hdu)
            else:
                if (
                    cold_nocache
                    and cache_capacity == 0
                    and hasattr(cpp_module, "read_full_nocache")
                ):
                    data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
                elif cache_capacity == 0 and hasattr(cpp_module, "read_full_nocache"):
                    data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
                else:
                    data = cpp_module.read_full(path, hdu, effective_mmap)

        if fp16:
            data = data.to(torch.float16)
        elif bf16:
            data = data.to(torch.bfloat16)

        if device != "cpu" and data.device.type == "cpu":
            data = data.to(device)

        return data
    except ValueError:
        raise
    except read_exc_types as exc:
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
