"""
torchfits: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from collections import OrderedDict

import numpy as np
import torch
import os
from torch import Tensor

from .buffer import clear_buffers, configure_buffers, get_buffer_stats
from .cache import clear_cache, configure_for_environment, get_cache_stats
from .core import CompressionType, FITSCore
from .dataloader import (
    create_dataloader,
    create_fits_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
)
from .datasets import FITSDataset, IterableFITSDataset, TableChunkDataset
from .frame import read_tensor_frame, to_tensor_frame, write_tensor_frame
from .hdu import HDUList, Header, TableHDU, TensorHDU
from .interop import to_arrow, to_pandas
from .header_parser import fast_parse_header
from . import io
from .transforms import (
    AsinhStretch,
    CenterCrop,
    Compose,
    GaussianNoise,
    LogStretch,
    MinMaxScale,
    Normalize,
    PerturbByError,
    PoissonNoise,
    PowerStretch,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    RedshiftShift,
    RobustScale,
    ToDevice,
    ZScale,
    create_inference_transform,
    create_training_transform,
    create_validation_transform,
)
from .wcs import WCS

# Simple cache tracking for tests
_cache_stats = {"total_requests": 0, "hits": 0, "misses": 0, "cache_size": 0}
_file_cache = OrderedDict()
_file_handle_cache = OrderedDict()
_image_meta_cache = OrderedDict()

# Auto-configure cache and buffers on import
configure_for_environment()
try:
    # Optional CFITSIO cache tuning via env (safe defaults if unset)
    import torchfits.cpp as cpp

    _cache_mb = os.environ.get("TORCHFITS_CFITSIO_CACHE_MB")
    _cache_files = os.environ.get("TORCHFITS_CFITSIO_CACHE_FILES")
    if _cache_mb is not None or _cache_files is not None:
        max_files = int(_cache_files) if _cache_files is not None else 32
        max_mb = int(_cache_mb) if _cache_mb is not None else 256
        cpp.configure_cache(max_files, max_mb)
except Exception:
    pass

__version__ = "0.1.1"
__all__ = [
    # Core I/O functions
    "read",
    "write",
    "open",
    "get_header",
    "read_subset",
    "io",
    # Batch operations
    "read_batch",
    "get_batch_info",
    "get_cache_performance",
    "clear_file_cache",
    "read_large_table",
    "stream_table",
    # HDU classes
    "HDUList",
    "TensorHDU",
    "TableHDU",
    "Header",
    # WCS functionality
    "WCS",
    # Dataset classes
    "FITSDataset",
    "IterableFITSDataset",
    "TableChunkDataset",
    # DataLoader factories
    "create_dataloader",
    "create_fits_dataloader",
    "create_streaming_dataloader",
    "create_table_dataloader",
    # Transforms
    "ZScale",
    "AsinhStretch",
    "LogStretch",
    "PowerStretch",
    "Normalize",
    "MinMaxScale",
    "RobustScale",
    "RandomCrop",
    "CenterCrop",
    "RandomFlip",
    "GaussianNoise",
    "PoissonNoise",
    "RandomRotation",
    "RedshiftShift",
    "PerturbByError",
    "ToDevice",
    "Compose",
    "create_training_transform",
    "create_validation_transform",
    "create_inference_transform",
    # Core types
    "FITSCore",
    "CompressionType",
    # Header parsing
    "fast_parse_header",
    # Utility functions
    "configure_for_environment",
    "get_cache_stats",
    "clear_cache",
    "configure_buffers",
    "get_buffer_stats",
    "clear_buffers",
    # Integration
    "to_tensor_frame",
    "read_tensor_frame",
    "write_tensor_frame",
    "to_pandas",
    "to_arrow",
]


def _read_header_fast(file_handle, hdu_index: int, fast_header: bool = True):
    """Read header using fast bulk parsing or fallback to slow method."""
    import torchfits.cpp as cpp

    if fast_header:
        try:
            # Try fast bulk header reading
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, RuntimeError, IOError):
            # Fall back to slow method if fast parsing fails
            pass

    # Fall back to keyword-by-keyword reading
    return cpp.read_header(file_handle, hdu_index)


def _get_vla_columns(header_data: Optional[Dict[str, Any]]) -> set:
    """Extract VLA column names from header metadata."""
    vla_columns = set()
    if not header_data:
        return vla_columns
    for key, value in header_data.items():
        if not isinstance(key, str) or not key.startswith("TFORM"):
            continue
        if not isinstance(value, str) or not value:
            continue
        code = value.strip().upper()
        if code.startswith("P") or code.startswith("Q"):
            idx = key[5:]
            name = header_data.get(f"TTYPE{idx}")
            if isinstance(name, str) and name:
                vla_columns.add(name)
            else:
                vla_columns.add(f"COL{idx}")
    return vla_columns


def _get_scaled_columns(header_data: Optional[Dict[str, Any]]) -> set:
    """Extract scaled column names (TSCALn/TZEROn) from header metadata."""
    scaled_columns = set()
    if not header_data:
        return scaled_columns
    for key, value in header_data.items():
        if not isinstance(key, str):
            continue
        key_upper = key.upper()
        if not (key_upper.startswith("TSCAL") or key_upper.startswith("TZERO")):
            continue
        idx = key_upper[5:]
        if not idx:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = None
        is_scaled = False
        if key_upper.startswith("TSCAL"):
            is_scaled = val is None or val != 1.0
        else:
            is_scaled = val is None or val != 0.0
        if is_scaled:
            name = header_data.get(f"TTYPE{idx}")
            if isinstance(name, str) and name:
                scaled_columns.add(name)
            else:
                scaled_columns.add(f"COL{idx}")
    return scaled_columns


def _get_cached_handle(path: str, cache_capacity: int):
    import torchfits.cpp as cpp

    if cache_capacity <= 0:
        return cpp.open_fits_file(path, "r"), False

    handle = _file_handle_cache.pop(path, None)
    if handle is None:
        handle = cpp.open_fits_file(path, "r")
    _file_handle_cache[path] = handle

    while len(_file_handle_cache) > cache_capacity:
        _, old_handle = _file_handle_cache.popitem(last=False)
        try:
            old_handle.close()
        except Exception:
            pass

    return handle, True


def _get_image_meta(path: str, hdu: int):
    import torchfits.cpp as cpp

    sig = (path, hdu)

    cached = _image_meta_cache.get(sig)
    if cached is not None:
        return cached

    try:
        header = Header(cpp.read_header_dict(path, hdu))
        bitpix = int(header.get("BITPIX", 0))
        naxis = int(header.get("NAXIS", 0))
        dims = []
        for i in range(1, naxis + 1):
            key = f"NAXIS{i}"
            if key in header:
                try:
                    dims.append(int(header.get(key)))
                except Exception:
                    break
        meta = (bitpix, naxis, tuple(dims))
    except Exception:
        meta = None

    _image_meta_cache[sig] = meta
    # Keep a bounded cache to avoid unbounded growth on many files.
    while len(_image_meta_cache) > 256:
        _image_meta_cache.popitem(last=False)
    return meta


def _should_use_cold_nommap(path: str, hdu: int, cache_capacity: int, mmap: bool) -> bool:
    if not mmap or cache_capacity != 0:
        return False
    if _COLD_NOMMAP:
        return True

    # Conservative heuristic: int8 2D images in cold mode often perform better
    # without mmap due to per-read setup overhead.
    meta = _get_image_meta(path, hdu)
    if not meta:
        return False
    bitpix, naxis, dims = meta
    if bitpix != 8 or naxis != 2 or len(dims) != 2:
        return False
    elems = dims[0] * dims[1]
    return 256 * 256 <= elems <= 4096 * 4096


def _read_scaled_cpu_fast(path: str, hdu: int = 0, mmap: bool = True) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    import torchfits.cpp as cpp

    if not hasattr(cpp, "read_full_raw_with_scale"):
        raise RuntimeError("Scaled fast path unavailable in this build")

    data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
    if scaled:
        data = data.to(dtype=torch.float32)
        if bscale != 1.0:
            data.mul_(bscale)
        if bzero != 0.0:
            data.add_(bzero)
    return data


def read(
    path: Union[str, List[str], Tuple[str, ...]],
    hdu: Union[int, str, List[int], Tuple[int, ...]] = 0,
    device: str = "cpu",
    mmap: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    fast_header: bool = True,
    return_header: bool = False,
):
    """Read FITS data with optimizations.

    Args:
        path: File path or cutout specification
        hdu: HDU index or name
        device: Target device ('cpu', 'cuda')
        mmap: Use memory mapping for large files
        fp16: Convert to half precision
        bf16: Convert to bfloat16
        columns: List of column names for table reading (None = all columns)
        start_row: Starting row for table reading (1-based)
        num_rows: Number of rows to read (-1 = all rows)
        cache_capacity: File cache capacity (max entries)
        fast_header: Use fast bulk header parsing (default: True)
        return_header: Whether to return header dict (default: False)

    Returns:
        torch.Tensor for images or dict for tables; header dict if return_header=True

    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: File not found
        RuntimeError: FITS reading errors
    """
    # Use C++ backend for high-performance I/O
    import torchfits.cpp as cpp
    global _cache_stats, _file_cache

    if not path:
        raise ValueError("Path must be a non-empty string")

    if isinstance(path, (list, tuple)):
        if not isinstance(hdu, int):
            raise ValueError("Batch read requires a single integer HDU")
        try:
            data_list = cpp.read_images_batch(list(path), hdu)
            if device != "cpu":
                data_list = [t.to(device) for t in data_list]
            return data_list
        except Exception:
            data_list = []
            for p in path:
                data_list.append(read(p, hdu=hdu, device=device, mmap=mmap, fp16=fp16, bf16=bf16, raw_scale=raw_scale, columns=columns, start_row=start_row, num_rows=num_rows, cache_capacity=cache_capacity, fast_header=fast_header, return_header=return_header))
            return data_list

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")

    # Fast-path: direct image read with no header and no table options.
    if isinstance(hdu, (list, tuple)) and not return_header and columns is None and start_row == 1 and num_rows == -1:
        if hasattr(cpp, "read_hdus_batch"):
            data = cpp.read_hdus_batch(path, list(hdu))
            if device != "cpu":
                data = [t.to(device) for t in data]
            return data

    # Deterministic CPU scaled fast-path (avoid fallback overheads)
    debug_scale = _DEBUG_SCALE
    if (
        scale_on_device
        and not raw_scale
        and device == "cpu"
        and not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        try:
            if debug_scale:
                print("TORCHFITS_DEBUG_SCALE: fast_cpu_scaled")
            effective_mmap = mmap and not _should_use_cold_nommap(
                path, hdu, cache_capacity, mmap
            )
            cache_hit = False
            if cache_capacity > 0:
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: fast_cpu_scaled_cached_handle")
                cache_hit = path in _file_handle_cache
                handle, cached = _get_cached_handle(path, cache_capacity)
                data = cpp.read_full(handle, hdu, effective_mmap)
                if not cached:
                    try:
                        handle.close()
                    except Exception:
                        pass
            else:
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: fast_cpu_scaled_direct")
                if _COLD_NOCACHE and hasattr(cpp, "read_full_nocache"):
                    data = cpp.read_full_nocache(path, hdu, effective_mmap)
                else:
                    data = cpp.read_full(path, hdu, effective_mmap)
            if fp16:
                data = data.to(torch.float16)
            elif bf16:
                data = data.to(torch.bfloat16)

            global _cache_stats
            _cache_stats["total_requests"] += 1
            if cache_capacity > 0 and cache_hit:
                _cache_stats["hits"] += 1
            else:
                _cache_stats["misses"] += 1

            return data
        except Exception:
            # Not an image HDU or fast path unavailable; fall back to generic path.
            pass

    if (
        not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        try:
            effective_mmap = mmap and not _should_use_cold_nommap(
                path, hdu, cache_capacity, mmap
            )
            if scale_on_device and not raw_scale:
                if hasattr(cpp, "read_full_raw_with_scale"):
                    if debug_scale:
                        print("TORCHFITS_DEBUG_SCALE: fast_path_scaled")
                    data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(
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
            elif raw_scale:
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: raw_scale")
                if not effective_mmap and hasattr(cpp, "read_full_unmapped_raw"):
                    data = cpp.read_full_unmapped_raw(path, hdu)
                else:
                    data = cpp.read_full_raw(path, hdu, effective_mmap)
            else:
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: unscaled")
                if not effective_mmap and hasattr(cpp, "read_full_unmapped"):
                    data = cpp.read_full_unmapped(path, hdu)
                else:
                    if _COLD_NOCACHE and cache_capacity == 0 and hasattr(
                        cpp, "read_full_nocache"
                    ):
                        data = cpp.read_full_nocache(path, hdu, effective_mmap)
                    else:
                        data = cpp.read_full(path, hdu, effective_mmap)

            if fp16:
                data = data.to(torch.float16)
            elif bf16:
                data = data.to(torch.bfloat16)

            if device != "cpu" and data.device.type == "cpu":
                data = data.to(device)

            return data
        except ValueError:
            raise
        except Exception:
            pass

    # Cache tracking
    _cache_stats["total_requests"] += 1
    use_cache = cache_capacity > 0

    # Use tuple for cache key (faster than f-string)
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
            if cache_key in _file_cache:
                _cache_stats["hits"] += 1
                cached_data, cached_header = _file_cache.pop(cache_key)
                _file_cache[cache_key] = (cached_data, cached_header)

                if device != "cpu":
                    if isinstance(cached_data, torch.Tensor):
                        cached_data = cached_data.to(device)
                    elif isinstance(cached_data, dict):
                        # Move tensors in dict to device
                        new_data = {}
                        for k, v in cached_data.items():
                            if isinstance(v, torch.Tensor):
                                new_data[k] = v.to(device)
                            else:
                                new_data[k] = v
                        cached_data = new_data

                return (cached_data, cached_header) if return_header else cached_data
            else:
                _cache_stats["misses"] += 1
        else:
            _cache_stats["misses"] += 1
    else:
        _cache_stats["misses"] += 1

    # Input validation
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be non-negative")

    if start_row < 1:
        raise ValueError("start_row must be >= 1 (FITS uses 1-based indexing)")

    if num_rows < -1 or num_rows == 0:
        raise ValueError("num_rows must be > 0 or -1 for all rows")

    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    try:

        # Use handle-based API (most reliable)
        file_handle, cached_handle = _get_cached_handle(path, cache_capacity)

        try:
            # Handle HDU selection (int or name)
            if isinstance(hdu, str):
                num_hdus = cpp.get_num_hdus(file_handle)
                hdu_num = None
                for i in range(num_hdus):
                    try:
                        hdr = cpp.read_header(file_handle, i)
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

            # Try reading as IMAGE first (for slow path) without header overhead
            try:
                data = cpp.read_full(file_handle, hdu_num, mmap)

                # Apply precision conversion
                if fp16:
                    data = data.to(torch.float16)
                elif bf16:
                    data = data.to(torch.bfloat16)

                # Move to device
                if device != "cpu":
                    data = data.to(device)

                if return_header:
                    header_data = _read_header_fast(file_handle, hdu_num, fast_header)
                    header = Header(header_data)

                # Cache result
                if use_cache and cache_key is not None:
                    _file_cache[cache_key] = (
                        data.cpu() if device != "cpu" else data,
                        header,
                    )
                    while len(_file_cache) > cache_capacity:
                        _file_cache.popitem(last=False)
                    _cache_stats["cache_size"] = len(_file_cache)

                return (data, header) if return_header else data

            except (RuntimeError, TypeError):
                # Not an image, read as table using C++ backend
                try:
                    if (return_header or isinstance(hdu, str)) and header_data is None:
                        header_data = _read_header_fast(
                            file_handle, hdu_num, fast_header
                        )
                        header = Header(header_data)

                    # Pass columns and mmap flag to C++ reader
                    col_list = columns if columns else []
                    table_result = None
                    if mmap:
                        try:
                            if start_row > 1 or num_rows != -1:
                                if hasattr(cpp, "read_fits_table_rows"):
                                    table_result = cpp.read_fits_table_rows(
                                        path,
                                        hdu_num,
                                        col_list,
                                        start_row,
                                        num_rows,
                                        True,
                                    )
                                else:
                                    table_result = cpp.read_fits_table(
                                        path, hdu_num, col_list, True
                                    )
                            else:
                                table_result = cpp.read_fits_table(
                                    path, hdu_num, col_list, True
                                )
                        except Exception:
                            table_result = None

                    if table_result is None:
                        if columns is None and start_row == 1 and num_rows == -1:
                            table_result = cpp.read_fits_table_from_handle(
                                file_handle, hdu_num
                            )
                        elif hasattr(cpp, "read_fits_table_rows_from_handle"):
                            table_result = cpp.read_fits_table_rows_from_handle(
                                file_handle, hdu_num, col_list, start_row, num_rows
                            )
                        elif start_row > 1 or num_rows != -1:
                            if hasattr(cpp, "read_fits_table_rows"):
                                table_result = cpp.read_fits_table_rows(
                                    path,
                                    hdu_num,
                                    col_list,
                                    start_row,
                                    num_rows,
                                    False,
                                )
                            else:
                                table_result = cpp.read_fits_table(
                                    path, hdu_num, col_list, False
                                )
                        else:
                            table_result = cpp.read_fits_table(
                                path, hdu_num, col_list, False
                            )

                    # table_result is the dictionary of tensors directly
                    table_data = table_result

                    # If columns were passed to C++, filtering is already done.
                    # But if columns was None, we got all columns.
                    # If columns was not None, we got only requested columns.

                    # Handle row slicing if needed when C++ row slicing is unavailable
                    if (
                        (start_row > 1 or num_rows != -1)
                        and not hasattr(cpp, "read_fits_table_rows")
                    ):
                        for k, v in table_data.items():
                            if isinstance(v, torch.Tensor):
                                end_row = (
                                    start_row + num_rows - 1
                                    if num_rows != -1
                                    else len(v)
                                )
                                table_data[k] = v[start_row - 1 : end_row]

                    # Cache result (move to CPU for storage)
                    if device != "cpu":
                        # Should already be on CPU from C++ reader, but just in case
                        # Actually C++ reader returns CPU tensors.
                        # We store them as is.
                        pass

                        if use_cache and cache_key is not None:
                            _file_cache[cache_key] = (table_data, header)
                            while len(_file_cache) > cache_capacity:
                                _file_cache.popitem(last=False)
                            _cache_stats["cache_size"] = len(_file_cache)

                    # Move to device if requested (for initial return)
                    if device != "cpu":
                        new_data = {}
                        for k, v in table_data.items():
                            if isinstance(v, torch.Tensor):
                                new_data[k] = v.to(device)
                            elif isinstance(v, list):
                                new_list = []
                                for item in v:
                                    if isinstance(item, torch.Tensor):
                                        new_list.append(item.to(device))
                                    else:
                                        new_list.append(item)
                                new_data[k] = new_list
                            else:
                                new_data[k] = v
                        table_data = new_data

                    return (table_data, header) if return_header else table_data
                except Exception as e:
                    raise RuntimeError(f"Failed to read table extension: {e}")

        finally:
            if not cached_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass

    except Exception as e:
        raise RuntimeError(f"Failed to read FITS file '{path}': {e}") from e


def read_subset(path: str, hdu: int, x1: int, y1: int, x2: int, y2: int) -> Tensor:
    """Read a rectangular subset of an image HDU.

    Args:
        path: Path to FITS file
        hdu: HDU index
        x1, y1: Start coordinates (0-based, inclusive)
        x2, y2: End coordinates (0-based, exclusive)

    Returns:
        Tensor containing the subset
    """
    import torchfits.cpp as cpp

    # Check cache first (optional, but consistent)
    # For now, no caching for subsets to avoid complexity

    try:
        file_handle = cpp.open_fits_file(path, "r")
        try:
            return file_handle.read_subset(hdu, x1, y1, x2, y2)
        finally:
            # Explicitly close to release resources
            # cpp.close_fits_file(file_handle) # FITSFile has close method, but let's rely on destructor or explicit close if exposed
            file_handle.close()
    except Exception as e:
        raise RuntimeError(f"Failed to read subset from '{path}': {e}") from e


def write(
    path: str,
    data,
    header: Header = None,
    overwrite: bool = False,
    compress: bool = False,
):
    """Write data to FITS file.

    Args:
        path: Output file path
        data: Data to write (Tensor, TensorFrame, or HDUList)
        header: Optional FITS header dictionary
        overwrite: Whether to overwrite existing files
        compress: Whether to use tile compression (Rice algorithm)
    """
    import os

    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f"File '{path}' already exists. Use overwrite=True to overwrite."
        )

    try:
        # Import C++ backend
        import torchfits.cpp as cpp

        hdus_to_write = []

        if isinstance(data, HDUList):
            cpp.write_fits_file(path, data._hdus, overwrite)
            return

        # Check for table dictionary (col_name -> tensor)
        if isinstance(data, dict) and "data" not in data:
            # Assume table if values are tensors
            is_table = True
            for k, v in data.items():
                if not isinstance(v, (torch.Tensor, np.ndarray)):
                    is_table = False
                    break

            if is_table:
                cpp.write_fits_table(path, data, header if header else {}, overwrite)
                return

        if isinstance(data, Tensor):
            # Single image HDU
            hdu_dict = {"data": data}
            if header:
                hdu_dict["header"] = header
            hdus_to_write.append(hdu_dict)

        elif hasattr(data, "__iter__") and not isinstance(data, (str, Tensor)):
            # List of HDUs
            for item in data:
                if isinstance(item, dict):
                    # Already a dict (e.g. from previous read)
                    # Ensure 'data' is present
                    if "data" in item:
                        hdus_to_write.append(item)
                elif isinstance(item, Tensor):
                    hdus_to_write.append({"data": item})
                # Handle objects with .data attribute (like TensorHDU)
                elif hasattr(item, "data") and isinstance(item.data, Tensor):
                    hdu_dict = {"data": item.data}
                    if hasattr(item, "header"):
                        hdu_dict["header"] = item.header
                    hdus_to_write.append(hdu_dict)
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")

        # Call C++ writer
        cpp.write_fits_file(path, hdus_to_write, overwrite)

    except Exception as e:
        # Clean up partial file on error
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e


def read_batch(file_paths: List[str], hdu: int = 0, device: str = "cpu") -> List[Tensor]:
    """Read multiple FITS files in batch.

    Args:
        file_paths: List of file paths to read
        hdu: HDU index to read from each file
        device: Target device for tensors

    Returns:
        List of tensors from each file
    """
    if not file_paths:
        return []

    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    try:
        if isinstance(hdu, int) and hdu >= 0:
            import torchfits.cpp as cpp

            tensors = cpp.read_images_batch(list(file_paths), hdu)
            if device != "cpu":
                tensors = [t.to(device) for t in tensors]
            return tensors
    except Exception:
        pass

    results = []
    for path in file_paths:
        try:
            tensor = read(path, hdu=hdu, device=device, return_header=False)
            results.append(tensor)
        except Exception:
            # Skip failed files
            continue
    return results


def get_batch_info(file_paths: List[str]) -> Dict[str, Any]:
    """Get information about a batch of FITS files.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary with batch statistics
    """
    valid_files = 0
    for path in file_paths:
        try:
            import os

            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue

    return {"num_files": len(file_paths), "valid_files": valid_files}


def get_cache_performance() -> Dict[str, Any]:
    """Get cache performance statistics.

    Returns:
        Dictionary with cache statistics
    """
    global _cache_stats
    total = _cache_stats["total_requests"]
    hits = _cache_stats["hits"]
    misses = _cache_stats["misses"]

    return {
        "cache_size": _cache_stats["cache_size"],
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
    }


def clear_file_cache():
    """Clear the file cache."""
    global _cache_stats, _file_cache, _file_handle_cache, _image_meta_cache
    _file_cache.clear()
    for _, handle in list(_file_handle_cache.items()):
        try:
            handle.close()
        except Exception:
            pass
    _file_handle_cache.clear()
    _image_meta_cache.clear()
    _cache_stats = {"total_requests": 0, "hits": 0, "misses": 0, "cache_size": 0}

    # Also try to clear C++ cache if available
    try:
        import torchfits.cpp as cpp

        cpp.clear_file_cache()
    except (AttributeError, RuntimeError):
        pass


def read_large_table(
    file_path: str,
    hdu: int = 1,
    max_memory_mb: int = 100,
    streaming: bool = False,
    return_iterator: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Read large FITS table with memory management.

    Args:
        file_path: Path to FITS file
        hdu: HDU index
        max_memory_mb: Maximum memory usage in MB
        streaming: Whether to use streaming mode
        return_iterator: If True, return an iterator over chunks instead of a dict

    Returns:
        Dictionary with table data, or an iterator if return_iterator=True
    """
    try:
        import os

        import torchfits.cpp as cpp

        if not os.path.exists(file_path):
            return {}

        if not streaming:
            if return_iterator:
                return iter([cpp.read_fits_table(file_path, hdu)])
            return cpp.read_fits_table(file_path, hdu)

        # Streaming-like chunked read (returns full dict, but limits peak memory)
        if not hasattr(cpp, "read_fits_table_rows"):
            return cpp.read_fits_table(file_path, hdu)

        try:
            header = get_header(file_path, hdu)
            total_rows = header.get("NAXIS2", 0)
            try:
                if isinstance(total_rows, str):
                    total_rows = int(float(total_rows))
                else:
                    total_rows = int(total_rows)
            except Exception:
                total_rows = 0
            if total_rows == 0:
                return cpp.read_fits_table(file_path, hdu)

            # Sample a small batch to estimate bytes per row
            sample_rows = min(256, total_rows)
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)
                        sample = reader.read_rows([], 1, sample_rows)
                    else:
                        sample = cpp.read_fits_table_rows_from_handle(
                            file_handle, hdu, [], 1, sample_rows
                        )
                finally:
                    file_handle.close()
            else:
                sample = cpp.read_fits_table_rows(
                    file_path, hdu, [], 1, sample_rows, False
                )

            def _estimate_bytes(data: Dict[str, Any], rows: int) -> float:
                total = 0
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, torch.Tensor):
                                total += item.numel() * item.element_size()
                return total / max(1, rows)

            bytes_per_row = _estimate_bytes(sample, sample_rows)
            if bytes_per_row <= 0:
                bytes_per_row = 1024.0  # Fallback guess

            max_bytes = max_memory_mb * 1024 * 1024
            rows_per_chunk = max(1, int(max_bytes / bytes_per_row))

            if return_iterator:
                return stream_table(
                    file_path,
                    hdu=hdu,
                    columns=None,
                    start_row=1,
                    num_rows=-1,
                    chunk_rows=rows_per_chunk,
                    mmap=False,
                )

            accum = {}
            start = 1
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    reader = None
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)

                    while start <= total_rows:
                        remaining = total_rows - start + 1
                        num = min(rows_per_chunk, remaining)
                        if reader is not None:
                            chunk = reader.read_rows([], start, num)
                        else:
                            chunk = cpp.read_fits_table_rows_from_handle(
                                file_handle, hdu, [], start, num
                            )

                        for k, v in chunk.items():
                            if isinstance(v, torch.Tensor):
                                if k not in accum:
                                    out_shape = (total_rows,) + tuple(v.shape[1:])
                                    accum[k] = torch.empty(out_shape, dtype=v.dtype)
                                accum[k][start - 1 : start - 1 + v.shape[0]] = v
                            elif isinstance(v, list):
                                accum.setdefault(k, []).extend(v)
                            else:
                                accum.setdefault(k, []).append(v)

                        start += num
                finally:
                    file_handle.close()
            else:
                while start <= total_rows:
                    remaining = total_rows - start + 1
                    num = min(rows_per_chunk, remaining)
                    chunk = cpp.read_fits_table_rows(
                        file_path, hdu, [], start, num, False
                    )

                    for k, v in chunk.items():
                        if isinstance(v, torch.Tensor):
                            if k not in accum:
                                out_shape = (total_rows,) + tuple(v.shape[1:])
                                accum[k] = torch.empty(out_shape, dtype=v.dtype)
                            accum[k][start - 1 : start - 1 + v.shape[0]] = v
                        elif isinstance(v, list):
                            accum.setdefault(k, []).extend(v)
                        else:
                            accum.setdefault(k, []).append(v)

                    start += num

            return accum
        except Exception:
            return cpp.read_fits_table(file_path, hdu)

    except Exception:
        return {}


def stream_table(
    file_path: str,
    hdu: int = 1,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    chunk_rows: int = 10000,
    mmap: bool = False,
    max_chunks: Optional[int] = None,
):
    """Yield table data in row chunks.

    Falls back to a single full-table read if row streaming is unavailable.
    """
    import os

    import torchfits.cpp as cpp

    if not os.path.exists(file_path):
        return

    col_list = columns if columns else []

    if not hasattr(cpp, "read_fits_table_rows"):
        result = cpp.read_fits_table(file_path, hdu, col_list, mmap)
        yield result
        return

    header = get_header(file_path, hdu)
    total_rows = header.get("NAXIS2", 0)
    try:
        if isinstance(total_rows, str):
            total_rows = int(float(total_rows))
        else:
            total_rows = int(total_rows)
    except Exception:
        total_rows = 0
    if total_rows == 0:
        return

    if num_rows != -1:
        total_rows = min(total_rows, start_row + num_rows - 1)

    row = start_row
    emitted = 0
    if mmap and hasattr(cpp, "read_fits_table_rows"):
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return
    elif hasattr(cpp, "read_fits_table_rows_from_handle"):
        file_handle = cpp.open_fits_file(file_path, "r")
        try:
            reader = None
            if hasattr(cpp, "TableReader"):
                reader = cpp.TableReader(file_handle, hdu)
            while row <= total_rows:
                remaining = total_rows - row + 1
                size = min(chunk_rows, remaining)
                if reader is not None:
                    yield reader.read_rows(col_list, row, size)
                else:
                    yield cpp.read_fits_table_rows_from_handle(
                        file_handle, hdu, col_list, row, size
                    )
                row += size
                emitted += 1
                if max_chunks is not None and emitted >= max_chunks:
                    return
        finally:
            file_handle.close()
    else:
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return


def open(path: str, mode: str = "r") -> HDUList:
    """Open FITS file for reading/writing.

    Args:
        path: File path to open
        mode: File mode ('r' for read, 'w' for write)

    Returns:
        HDUList object for accessing HDUs

    Raises:
        FileNotFoundError: If file doesn't exist in read mode
        PermissionError: If insufficient permissions
        RuntimeError: For other FITS-related errors
    """
    import os

    if mode == "r" and not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open FITS file '{path}': {e}") from e


def get_header(path: str, hdu: Union[int, str] = 0) -> Header:
    """Get the header of a FITS file.

    Args:
        path: Path to the FITS file.
        hdu: HDU index or name (default: 0).

    Returns:
        Header object.
    """
    import torchfits.cpp as cpp

    # Resolve HDU index if string
    if isinstance(hdu, str):
        for i in range(100):  # Arbitrary limit
            try:
                h_list = cpp.read_header_dict(path, i)
                if not h_list:
                    break
                h = Header(h_list)
                if h.get("EXTNAME") == hdu:
                    return h
            except Exception:
                break
        raise ValueError(f"HDU '{hdu}' not found")

    return Header(cpp.read_header_dict(path, hdu))
_DEBUG_SCALE = os.environ.get("TORCHFITS_DEBUG_SCALE") == "1"
_COLD_NOMMAP = os.environ.get("TORCHFITS_COLD_NOMMAP") == "1"
_COLD_NOCACHE = os.environ.get("TORCHFITS_COLD_NOCACHE") == "1"
