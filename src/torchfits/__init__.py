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
from .hdu import HDUList, Header, TableHDU, TableHDURef, TensorHDU
from .interop import to_arrow, to_pandas
from . import table
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
_file_handle_sig_cache = OrderedDict()
_image_meta_cache = OrderedDict()
_hdu_type_cache = OrderedDict()
_cold_nommap_cache = OrderedDict()
_auto_hdu_cache = OrderedDict()


def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches/handles for a path that is being modified."""
    global _file_cache, _file_handle_cache, _file_handle_sig_cache, _image_meta_cache, _hdu_type_cache, _cold_nommap_cache, _auto_hdu_cache

    _file_cache.pop(path, None)
    handle = _file_handle_cache.pop(path, None)
    _file_handle_sig_cache.pop(path, None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass

    for key in [k for k in _image_meta_cache.keys() if k[0] == path]:
        _image_meta_cache.pop(key, None)
    for key in [k for k in _hdu_type_cache.keys() if k[0] == path]:
        _hdu_type_cache.pop(key, None)
    for key in [k for k in _cold_nommap_cache.keys() if k[0] == path]:
        _cold_nommap_cache.pop(key, None)
    _auto_hdu_cache.pop(path, None)

    # Keep Arrow table reader/handle caches coherent across write/append/update paths.
    try:
        if hasattr(table, "_invalidate_caches_for_path"):
            table._invalidate_caches_for_path(path)
    except Exception:
        pass


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

__version__ = "0.2.0"
__all__ = [
    # Core I/O functions
    "read",
    "write",
    "insert_hdu",
    "replace_hdu",
    "delete_hdu",
    "open",
    "get_header",
    "get_wcs",
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
    "TableHDURef",
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
    "table",
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


def _path_signature(path: str):
    try:
        st = os.stat(path)
    except Exception:
        return None
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return (int(st.st_size), int(mtime_ns), int(st.st_ino))


def _get_cached_handle(path: str, handle_cache_capacity: int):
    import torchfits.cpp as cpp

    if handle_cache_capacity <= 0:
        return cpp.open_fits_file(path, "r"), False

    # Keep a small Python-level LRU of open CFITSIO handles. This avoids open/close
    # overhead on repeated reads and is intentionally very cheap per call.
    cur_sig = _path_signature(path)
    handle = _file_handle_cache.get(path)
    if handle is not None:
        prev_sig = _file_handle_sig_cache.get(path)
        if cur_sig is not None and prev_sig is not None and prev_sig != cur_sig:
            # File changed on disk (size/mtime/inode): drop stale handle.
            try:
                handle.close()
            except Exception:
                pass
            _file_handle_cache.pop(path, None)
            _file_handle_sig_cache.pop(path, None)
            handle = None
        else:
            _file_handle_cache.move_to_end(path)
            _file_handle_sig_cache.move_to_end(path)

    if handle is None:
        handle = cpp.open_fits_file(path, "r")
        _file_handle_cache[path] = handle
        _file_handle_sig_cache[path] = cur_sig

    while len(_file_handle_cache) > handle_cache_capacity:
        old_path, old_handle = _file_handle_cache.popitem(last=False)
        _file_handle_sig_cache.pop(old_path, None)
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
        try:
            bscale = float(header.get("BSCALE", 1.0))
        except Exception:
            bscale = 1.0
        try:
            bzero = float(header.get("BZERO", 0.0))
        except Exception:
            bzero = 0.0
        dims = []
        for i in range(1, naxis + 1):
            key = f"NAXIS{i}"
            if key in header:
                try:
                    dims.append(int(header.get(key)))
                except Exception:
                    break
        zimage = header.get("ZIMAGE", False)
        if isinstance(zimage, str):
            zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
        xtension = str(header.get("XTENSION", "")).strip().upper()
        has_compression_keys = any(
            k in header for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
        )
        is_compressed = bool(zimage) or (
            xtension == "BINTABLE" and has_compression_keys
        )
        meta = (bitpix, naxis, tuple(dims), bscale, bzero, is_compressed)
    except Exception:
        meta = None

    _image_meta_cache[sig] = meta
    # Keep a bounded cache to avoid unbounded growth on many files.
    while len(_image_meta_cache) > 256:
        _image_meta_cache.popitem(last=False)
    return meta


def _get_image_meta_from_handle(file_handle, path: str, hdu: int):
    """Fetch image metadata using an already-open FITS handle (avoids open/close)."""
    sig = (path, hdu)
    cached = _image_meta_cache.get(sig)
    if cached is not None:
        return cached

    try:
        header_data = _read_header_fast(file_handle, hdu, fast_header=True)
        bitpix = int(header_data.get("BITPIX", 0))
        naxis = int(header_data.get("NAXIS", 0))
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
                    dims.append(int(header_data.get(key)))
                except Exception:
                    break
        zimage = header_data.get("ZIMAGE", False)
        if isinstance(zimage, str):
            zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
        xtension = str(header_data.get("XTENSION", "")).strip().upper()
        has_compression_keys = any(
            k in header_data for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
        )
        is_compressed = bool(zimage) or (
            xtension == "BINTABLE" and has_compression_keys
        )
        meta = (bitpix, naxis, tuple(dims), bscale, bzero, is_compressed)
    except Exception:
        meta = None

    _image_meta_cache[sig] = meta
    while len(_image_meta_cache) > 256:
        _image_meta_cache.popitem(last=False)
    return meta


def _get_cached_hdu_type(path: str, hdu: int) -> Optional[str]:
    sig = (path, hdu)
    cached = _hdu_type_cache.get(sig)
    if cached is not None:
        _hdu_type_cache.move_to_end(sig)
    return cached


def _set_cached_hdu_type(path: str, hdu: int, hdu_type: Optional[str]) -> None:
    if not hdu_type:
        return
    sig = (path, hdu)
    _hdu_type_cache[sig] = hdu_type
    _hdu_type_cache.move_to_end(sig)
    while len(_hdu_type_cache) > 512:
        _hdu_type_cache.popitem(last=False)


def _to_int_header_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            return int(float(value))
        return int(value)
    except Exception:
        return default


def _header_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in {"T", "TRUE", "1", "YES", "Y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _autodetect_hdu(path: str, handle_cache_capacity: int = 16) -> int:
    """Return the first HDU with payload, preferring image/compressed-image HDUs."""
    import torchfits.cpp as cpp

    sig = _path_signature(path)
    cached = _auto_hdu_cache.get(path)
    if cached is not None:
        cached_sig, cached_hdu = cached
        if sig is None or cached_sig is None or cached_sig == sig:
            _auto_hdu_cache.move_to_end(path)
            return int(cached_hdu)
        _auto_hdu_cache.pop(path, None)

    first_image_hdu: Optional[int] = None
    first_table_hdu: Optional[int] = None

    file_handle, cached_handle = _get_cached_handle(path, handle_cache_capacity)
    try:
        num_hdus = cpp.get_num_hdus(file_handle)
        for i in range(num_hdus):
            hdu_type = _get_cached_hdu_type(path, i)
            if hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, i)
                    _set_cached_hdu_type(path, i, hdu_type)
                except Exception:
                    hdu_type = None

            try:
                hdr = _read_header_fast(file_handle, i, fast_header=True)
            except Exception:
                hdr = {}

            naxis = _to_int_header_value(hdr.get("NAXIS"), default=0)
            has_image_payload = False
            if naxis > 0:
                has_image_payload = any(
                    _to_int_header_value(hdr.get(f"NAXIS{axis}"), default=0) > 0
                    for axis in range(1, naxis + 1)
                )

            zimage = _header_truthy(hdr.get("ZIMAGE"))
            has_compression_keys = any(
                k in hdr for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
            )
            is_compressed_image = zimage or has_compression_keys

            if has_image_payload or is_compressed_image:
                first_image_hdu = i
                break

            if hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}:
                if _to_int_header_value(hdr.get("NAXIS2"), default=0) > 0:
                    if first_table_hdu is None:
                        first_table_hdu = i
    finally:
        if not cached_handle:
            try:
                file_handle.close()
            except Exception:
                pass

    resolved = first_image_hdu if first_image_hdu is not None else (
        first_table_hdu if first_table_hdu is not None else 0
    )
    _auto_hdu_cache[path] = (sig, int(resolved))
    _auto_hdu_cache.move_to_end(path)
    while len(_auto_hdu_cache) > 512:
        _auto_hdu_cache.popitem(last=False)
    return int(resolved)


def _should_use_cold_nommap(
    path: str, hdu: int, cache_capacity: int, mmap: bool
) -> bool:
    # Keep cache_capacity in the signature for backward compatibility with
    # existing call sites; nommap selection now depends on file/HDU traits.
    _ = cache_capacity
    if not mmap:
        return False
    if _COLD_NOMMAP:
        return True

    cached = _cold_nommap_cache.get((path, hdu))
    if cached is not None:
        return bool(cached)

    # Auto-select non-mmap for common uncompressed integer/float32 cold reads where
    # mmap has consistently higher latency than direct reads.
    # Keep this scoped to larger files so tiny-image latency is unaffected.
    try:
        file_size = os.path.getsize(path)
        if file_size < (1 << 20):  # 1 MiB
            _cold_nommap_cache[(path, hdu)] = False
            return False
    except Exception:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    meta = _image_meta_cache.get((path, hdu))
    if meta is None:
        meta = _get_image_meta(path, hdu)
    if not meta:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    try:
        bitpix = int(meta[0])
    except Exception:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    is_compressed = False
    if len(meta) >= 6:
        try:
            is_compressed = bool(meta[5])
        except Exception:
            is_compressed = False
    if is_compressed:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    # TARGET: common medium/large science images where direct reads are often
    # lower-latency than mmap+byteswap in real workloads.
    if bitpix in (16, 32, -32):
        _cold_nommap_cache[(path, hdu)] = True
        _cold_nommap_cache.move_to_end((path, hdu))
        while len(_cold_nommap_cache) > 512:
            _cold_nommap_cache.popitem(last=False)
        return True

    _cold_nommap_cache[(path, hdu)] = False
    _cold_nommap_cache.move_to_end((path, hdu))
    while len(_cold_nommap_cache) > 512:
        _cold_nommap_cache.popitem(last=False)
    return False


def _resolve_image_mmap(
    path: str,
    hdu: int,
    mmap: Union[bool, str],
    cache_capacity: int,
) -> bool:
    """Resolve mmap mode for image reads.

    - mmap=True/False are explicit user overrides.
    - mmap='auto' uses heuristics:
      * compressed HDUs -> mmap disabled by default
      * otherwise use cold-read nommap heuristic
    """
    if isinstance(mmap, bool):
        return mmap

    if isinstance(mmap, str):
        mode = mmap.strip().lower()
        if mode != "auto":
            raise ValueError("mmap must be bool or 'auto'")

        meta = _image_meta_cache.get((path, hdu))
        if meta is None:
            meta = _get_image_meta(path, hdu)

        if meta is not None and len(meta) >= 6:
            try:
                if bool(meta[5]):  # compressed image HDU
                    return False
            except Exception:
                pass

        return not _should_use_cold_nommap(path, hdu, cache_capacity, True)

    raise ValueError("mmap must be bool or 'auto'")


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


def _apply_cpu_scale(data: Tensor, scaled: bool, bscale: float, bzero: float) -> Tensor:
    """Apply FITS BSCALE/BZERO with a fast integer path for signed-byte images."""
    if not scaled:
        return data

    # Common case for signed BYTE_IMG encoded as uint8 with BZERO=-128.
    if data.dtype == torch.uint8 and bscale == 1.0 and bzero == -128.0:
        # Physical values are (raw + bzero) = (raw - 128). This is equivalent to
        # flipping the sign bit and then reinterpreting as int8.
        # Avoid the int16 intermediate; it is noticeably slower on tiny images.
        data.bitwise_xor_(0x80)
        return data.to(dtype=torch.int8)

    data = data.to(dtype=torch.float32)
    if bscale != 1.0:
        data.mul_(bscale)
    if bzero != 0.0:
        data.add_(bzero)
    return data


def read(
    path: Union[str, List[str], Tuple[str, ...]],
    hdu: Union[int, str, List[int], Tuple[int, ...], None] = 0,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
):
    """Read FITS data with optimizations.

    Args:
        path: File path or cutout specification
        hdu: HDU index, name, or `"auto"` (first HDU with payload)
        device: Target device ('cpu', 'cuda')
        mmap: Memory-mapping mode. `True`/`False` are explicit, `'auto'` chooses
            per-HDU defaults (compressed images default to non-mmap).
        fp16: Convert to half precision
        bf16: Convert to bfloat16
        columns: List of column names for table reading (None = all columns)
        start_row: Starting row for table reading (1-based)
        num_rows: Number of rows to read (-1 = all rows)
        cache_capacity: File cache capacity (max entries)
        handle_cache_capacity: Open-handle cache capacity (max open files). This caches
            CFITSIO handles only (no data caching) and is a big win for repeated reads,
            especially for compressed images.
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
        # C++ batch API does not expose mmap mode controls. Use it only when
        # mmap=True is explicitly requested; otherwise fall back to per-file reads.
        if mmap is True:
            try:
                data_list = cpp.read_images_batch(list(path), hdu)
                if device != "cpu":
                    data_list = [t.to(device) for t in data_list]
                return data_list
            except Exception:
                pass

        data_list = []
        for p in path:
            data_list.append(
                read(
                    p,
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
                )
            )
        return data_list

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    if isinstance(mmap, str) and mmap.strip().lower() != "auto":
        raise ValueError("mmap must be bool or 'auto'")
    if not isinstance(mmap, (bool, str)):
        raise ValueError("mmap must be bool or 'auto'")
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = _autodetect_hdu(path, handle_cache_capacity)

    hdu_type_hint = _get_cached_hdu_type(path, hdu) if isinstance(hdu, int) else None
    is_cached_table_hdu = hdu_type_hint in {"ASCII_TABLE", "BINARY_TABLE"}

    # Avoid an extra open() just to probe HDU type: for extension images (MEF/compressed)
    # this is pure overhead, and on table HDUs we can simply attempt the image fast-path
    # and fall back on failure.
    skip_generic_image_fast_path = is_cached_table_hdu

    # Fast-path: direct image read with no header and no table options.
    if (
        isinstance(hdu, (list, tuple))
        and not return_header
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        if isinstance(mmap, bool) and hasattr(cpp, "read_hdus_batch"):
            data = cpp.read_hdus_batch(path, list(hdu), mmap)
            if device != "cpu":
                data = [t.to(device) for t in data]
            return data
        return [
            read(
                path,
                hdu=h,
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
            )
            for h in hdu
        ]

    # Deterministic CPU scaled fast-path (avoid fallback overheads)
    debug_scale = _DEBUG_SCALE
    # CPU common-case fast-path:
    # If the user wants physical values on CPU (default), let the C++ reader handle
    # scaling internally. This avoids the expensive `read_full_raw_with_scale` path
    # (raw read + scale probe + Python-side scaling) which hurts unscaled images.
    if (
        scale_on_device
        and not raw_scale
        and device == "cpu"
        and not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not is_cached_table_hdu
    ):
        try:
            effective_mmap = _resolve_image_mmap(path, hdu, mmap, cache_capacity)
            # Prefer handle-cache fast path when enabled: cache_capacity only controls
            # data caching, while handle_cache_capacity controls open-handle reuse.
            if handle_cache_capacity > 0:
                if hasattr(cpp, "read_full_cached"):
                    # Keep the fast path fully inside C++ (shared handle/meta cache)
                    # to avoid Python-side handle-cache overhead per read.
                    data = cpp.read_full_cached(path, hdu, effective_mmap)
                else:
                    file_handle, _cached = _get_cached_handle(
                        path, handle_cache_capacity
                    )
                    data = cpp.read_full(file_handle, hdu, effective_mmap)
            elif cache_capacity == 0 and hasattr(cpp, "read_full_nocache"):
                # Explicit no-cache mode when handle-cache is disabled.
                data = cpp.read_full_nocache(path, hdu, effective_mmap)
            else:
                file_handle = cpp.open_fits_file(path, "r")
                try:
                    data = cpp.read_full(file_handle, hdu, effective_mmap)
                finally:
                    try:
                        file_handle.close()
                    except Exception:
                        pass

            if fp16:
                data = data.to(torch.float16)
            elif bf16:
                data = data.to(torch.bfloat16)

            # Keep cache performance stats consistent even on fast-path returns.
            try:
                _cache_stats["total_requests"] += 1
                _cache_stats["misses"] += 1
            except Exception:
                pass
            return data
        except Exception:
            # Not an image HDU; skip generic image fast path and fall through.
            skip_generic_image_fast_path = True
            pass

    if (
        not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not skip_generic_image_fast_path
    ):
        try:
            effective_mmap = _resolve_image_mmap(path, hdu, mmap, cache_capacity)
            if scale_on_device and not raw_scale:
                # Avoid the expensive open+read+scale-probe path when the HDU is
                # known to be unscaled (BSCALE=1, BZERO=0). This is especially
                # important for compressed images where `read_full_raw_with_scale`
                # otherwise pays extra open/close overhead on every call.
                handle = None
                if handle_cache_capacity > 0:
                    # Use a cached handle to fetch metadata without re-opening the file.
                    try:
                        handle, _ = _get_cached_handle(path, handle_cache_capacity)
                    except Exception:
                        handle = None

                if handle is not None:
                    meta = _get_image_meta_from_handle(handle, path, hdu)
                else:
                    meta = _image_meta_cache.get((path, hdu))
                    if meta is None:
                        meta = _get_image_meta(path, hdu)
                needs_scale = None
                if meta is not None and len(meta) >= 5:
                    try:
                        needs_scale = not (
                            float(meta[3]) == 1.0 and float(meta[4]) == 0.0
                        )
                    except Exception:
                        needs_scale = None

                if needs_scale is False:
                    if debug_scale:
                        print("TORCHFITS_DEBUG_SCALE: fast_path_unscaled_via_read_full")
                    if handle_cache_capacity > 0:
                        if handle is None:
                            handle, _ = _get_cached_handle(path, handle_cache_capacity)
                        data = cpp.read_full(handle, hdu, effective_mmap)
                    else:
                        if (
                            _COLD_NOCACHE
                            and cache_capacity == 0
                            and hasattr(cpp, "read_full_nocache")
                        ):
                            data = cpp.read_full_nocache(path, hdu, effective_mmap)
                        elif cache_capacity == 0 and hasattr(cpp, "read_full_nocache"):
                            data = cpp.read_full_nocache(path, hdu, effective_mmap)
                        else:
                            data = cpp.read_full(path, hdu, effective_mmap)
                    data = data.to(device)
                else:
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
                # Keep raw-scale path on the unified read_full_raw entry point.
                # This avoids a crash-prone unmapped-only variant on scaled HDUs.
                data = cpp.read_full_raw(path, hdu, effective_mmap)
            else:
                if debug_scale:
                    print("TORCHFITS_DEBUG_SCALE: unscaled")
                if not effective_mmap and hasattr(cpp, "read_full_unmapped"):
                    data = cpp.read_full_unmapped(path, hdu)
                else:
                    if (
                        _COLD_NOCACHE
                        and cache_capacity == 0
                        and hasattr(cpp, "read_full_nocache")
                    ):
                        data = cpp.read_full_nocache(path, hdu, effective_mmap)
                    else:
                        if cache_capacity == 0 and hasattr(cpp, "read_full_nocache"):
                            # Explicitly disable caching when cache_capacity==0.
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
        file_handle, cached_handle = _get_cached_handle(path, handle_cache_capacity)

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
            hdu_type = hdu_type_hint if isinstance(hdu_num, int) else None
            if isinstance(hdu_num, int) and hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, hdu_num)
                    _set_cached_hdu_type(path, hdu_num, hdu_type)
                except Exception:
                    hdu_type = None

            is_table_hdu = hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}

            # Read image path when we know this is not a table.
            if not is_table_hdu:
                try:
                    effective_mmap = _resolve_image_mmap(
                        path, hdu_num, mmap, cache_capacity
                    )
                    data = cpp.read_full(file_handle, hdu_num, effective_mmap)

                    # Apply precision conversion
                    if fp16:
                        data = data.to(torch.float16)
                    elif bf16:
                        data = data.to(torch.bfloat16)

                    # Move to device
                    if device != "cpu":
                        data = data.to(device)

                    if return_header:
                        header_data = _read_header_fast(
                            file_handle, hdu_num, fast_header
                        )
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

                    if isinstance(hdu_num, int):
                        _set_cached_hdu_type(path, hdu_num, "IMAGE")
                    return (data, header) if return_header else data
                except (RuntimeError, TypeError):
                    # Confirm table type before falling through to table reader.
                    if isinstance(hdu_num, int):
                        try:
                            hdu_type = cpp.get_hdu_type(file_handle, hdu_num)
                            _set_cached_hdu_type(path, hdu_num, hdu_type)
                        except Exception:
                            hdu_type = None
                    is_table_hdu = hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}
                    if not is_table_hdu:
                        raise

            # Read as table using C++ backend
            try:
                if (return_header or isinstance(hdu, str)) and header_data is None:
                    header_data = _read_header_fast(file_handle, hdu_num, fast_header)
                    header = Header(header_data)

                # Pass columns and mmap flag to C++ reader
                col_list = columns if columns else []
                table_result = None
                table_mmap = mmap if isinstance(mmap, bool) else True
                if table_mmap:
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

                # Handle row slicing if needed when C++ row slicing is unavailable
                if (start_row > 1 or num_rows != -1) and not hasattr(
                    cpp, "read_fits_table_rows"
                ):
                    for k, v in table_data.items():
                        if isinstance(v, torch.Tensor):
                            end_row = (
                                start_row + num_rows - 1 if num_rows != -1 else len(v)
                            )
                            table_data[k] = v[start_row - 1 : end_row]

                # Cache result (stored on CPU as returned by C++ table readers)
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

                if isinstance(hdu_num, int):
                    _set_cached_hdu_type(path, hdu_num, "BINARY_TABLE")
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


def read_subset(
    path: str,
    hdu: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle_cache_capacity: int = 16,
) -> Tensor:
    """Read a rectangular subset of an image HDU.

    Args:
        path: Path to FITS file
        hdu: HDU index
        x1, y1: Start coordinates (0-based, inclusive)
        x2, y2: End coordinates (0-based, exclusive)
        handle_cache_capacity: Open-handle cache capacity (max open files). Reusing
            handles can significantly reduce overhead for many small cutouts.

    Returns:
        Tensor containing the subset
    """
    try:
        file_handle, cached = _get_cached_handle(path, handle_cache_capacity)
        try:
            return file_handle.read_subset(hdu, x1, y1, x2, y2)
        finally:
            if not cached:
                try:
                    file_handle.close()
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to read subset from '{path}': {e}") from e


def write(
    path: str,
    data,
    header: Header = None,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
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

    # The unified C++ cache and the Python-side handle cache can otherwise return
    # stale views of an overwritten file (mtime/size can be unchanged).
    _invalidate_path_caches(path)

    try:
        # Import C++ backend
        import torchfits.cpp as cpp

        hdus_to_write = []

        if compress:
            compressed_hdus: List[Any] = []
            if isinstance(data, Tensor):
                compressed_hdus = [TensorHDU(data=data, header=Header(header or {}))]
            elif isinstance(data, HDUList):
                compressed_hdus = list(getattr(data, "_hdus", []))
            elif isinstance(data, dict):
                if "data" in data:
                    item_hdu = _coerce_compressed_hdu_item(data)
                    compressed_hdus.append(item_hdu)
                else:
                    compressed_hdus.append(TableHDU(data, header=Header(header or {})))
            elif isinstance(data, (list, tuple)):
                for item in data:
                    compressed_hdus.append(_coerce_compressed_hdu_item(item))
            else:
                raise NotImplementedError(
                    "Compressed FITS writing supports tensors, tables, or HDU lists."
                )

            if header and compressed_hdus:
                first = compressed_hdus[0]
                merged = Header(dict(getattr(first, "header", {})))
                merged.update(dict(header))
                if isinstance(first, TensorHDU):
                    first._header = merged
                else:
                    first.header = merged

            _write_hdus_with_optional_compression(
                path, compressed_hdus, compress=compress
            )
            return

        if isinstance(data, HDUList):
            data.write(path, overwrite=overwrite)
            return

        # Check for table dictionary (col_name -> tensor)
        if isinstance(data, dict) and "data" not in data:
            if _can_use_cpp_table_writer(data):
                data = _normalize_cpp_table_data(data)
                cpp.write_fits_table(path, data, header if header else {}, overwrite)
                return
            raise ValueError(
                "Dictionary table writes currently require CFITSIO-native column types "
                "(numeric/bool/complex, strings, or VLA lists). Unsupported object/structure "
                "columns should be converted before writing."
            )

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


def _can_use_cpp_table_writer(table_dict: Dict[str, Any]) -> bool:
    """Return True when all table columns can use the fast C++ writer."""
    if not table_dict:
        return False

    for value in table_dict.values():
        if isinstance(value, torch.Tensor):
            if value.dim() > 2:
                return False
            if value.is_complex():
                if value.dtype not in {torch.complex64, torch.complex128}:
                    return False
                continue
            if value.dtype not in {
                torch.bool,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
            }:
                return False
            continue

        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
            except ValueError:
                arr = np.asarray(value, dtype=object)
            if arr.dtype != np.object_:
                value = arr
            else:
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
                        continue
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.dim() > 2:
                            return False
                        if t.is_complex():
                            if t.dtype not in {torch.complex64, torch.complex128}:
                                return False
                            continue
                        if t.dtype not in {
                            torch.bool,
                            torch.uint8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.float32,
                            torch.float64,
                        }:
                            return False
                        continue
                    arr_item = np.asarray(item)
                    if arr_item.ndim > 2:
                        return False
                    if np.iscomplexobj(arr_item):
                        if arr_item.dtype not in (np.complex64, np.complex128):
                            return False
                        continue
                    kind = arr_item.dtype.kind
                    itemsize = arr_item.dtype.itemsize
                    if kind in {"U", "S"}:
                        continue
                    if kind == "b":
                        continue
                    if kind == "u" and itemsize == 1:
                        continue
                    if kind == "i" and itemsize in (2, 4, 8):
                        continue
                    if kind == "f" and itemsize in (4, 8):
                        continue
                    return False
                continue

        if not isinstance(value, np.ndarray):
            return False
        if value.ndim > 2:
            return False
        if np.iscomplexobj(value):
            if value.dtype not in (np.complex64, np.complex128):
                return False
            continue
        if value.dtype.kind in {"U", "S"}:
            continue
        kind = value.dtype.kind
        itemsize = value.dtype.itemsize
        if kind == "b":
            continue
        if kind == "u" and itemsize == 1:
            continue
        if kind == "i" and itemsize in (2, 4, 8):
            continue
        if kind == "f" and itemsize in (4, 8):
            continue
        return False

    return True


def _normalize_cpp_table_data(table_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize table data for the C++ writer (strings/VLA/object arrays)."""
    out: Dict[str, Any] = {}
    for name, value in table_dict.items():
        if isinstance(value, (list, tuple)):
            items = list(value)
            if items and all(
                isinstance(item, (str, bytes, np.str_, np.bytes_)) or item is None
                for item in items
            ):
                out[name] = items
                continue
            if any(
                isinstance(item, (list, tuple, np.ndarray, torch.Tensor))
                for item in items
            ):
                norm_items = []
                for item in items:
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if t.dim() == 0:
                            t = t.reshape(1)
                        norm_items.append(np.ascontiguousarray(t.numpy()))
                    elif isinstance(item, np.ndarray):
                        norm_items.append(np.ascontiguousarray(item))
                    elif item is None:
                        norm_items.append(np.asarray([], dtype=np.float32))
                    else:
                        norm_items.append(np.asarray(item))
                out[name] = norm_items
                continue
            out[name] = np.asarray(items)
            continue
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_:
                out[name] = list(value)
                continue
            if value.dtype.kind in {"U", "S"}:
                out[name] = value.astype(str).tolist()
                continue
        out[name] = value
    return out


def _resolve_compression_algorithm(compress: Union[bool, str]) -> Optional[str]:
    """Normalize compress flag to a backend algorithm string or None."""
    if compress is False:
        return None
    if compress is True:
        return "RICE_1"
    if isinstance(compress, str):
        algo = compress.strip()
        return algo if algo else "RICE_1"
    raise TypeError("compress must be bool or compression algorithm string")


def _coerce_compressed_hdu_item(item: Any) -> Any:
    """Normalize compressed-write inputs to TensorHDU/TableHDU objects."""
    if isinstance(item, (TensorHDU, TableHDU)):
        return item
    if isinstance(item, TableHDURef):
        return item.materialize(device="cpu")
    if isinstance(item, Tensor):
        return TensorHDU(data=item, header=Header())
    if isinstance(item, dict):
        if "data" in item:
            img = item["data"]
            if not isinstance(img, Tensor):
                raise NotImplementedError(
                    "Compressed FITS writing supports tensor image payloads for dict HDUs."
                )
            return TensorHDU(data=img, header=Header(item.get("header", {})))
        return TableHDU(item, header=Header())
    raise NotImplementedError(
        f"Unsupported HDU payload for compressed write: {type(item)}"
    )


def _detach_hdus_for_rewrite(path: str) -> List[Any]:
    """Materialize file-backed HDUs so rewrite paths never hold stale handles."""
    with open(path) as hdul:
        detached: List[Any] = []
        for hdu in list(hdul._hdus):
            if isinstance(hdu, TensorHDU):
                detached.append(TensorHDU(data=hdu.to_tensor("cpu"), header=hdu.header))
            elif isinstance(hdu, TableHDU):
                detached.append(
                    TableHDU(dict(getattr(hdu, "_raw_data", {})), header=hdu.header)
                )
            elif isinstance(hdu, TableHDURef):
                mat = hdu.materialize(device="cpu")
                detached.append(
                    TableHDU(dict(getattr(mat, "_raw_data", {})), header=hdu.header)
                )
            else:
                detached.append(hdu)
    return detached


def _sanitize_header_for_compressed_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop structural/compression keys so CFITSIO can emit canonical metadata."""
    if not header:
        return {}

    skip_exact = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
        "BSCALE",
        "BZERO",
        "DATASUM",
        "CHECKSUM",
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZPCOUNT",
        "ZGCOUNT",
        "ZHECKSUM",
        "ZDATASUM",
    }
    skip_prefix = (
        "NAXIS",
        "ZNAXIS",
        "ZTILE",
        "ZNAME",
        "ZVAL",
        "TTYPE",
        "TFORM",
        "TDIM",
        "TSCAL",
        "TZERO",
        "TNULL",
        "TUNIT",
        "TDISP",
    )

    out: Dict[str, Any] = {}
    for key, value in dict(header).items():
        key_str = str(key)
        key_upper = key_str.upper()
        if key_upper in skip_exact or any(
            key_upper.startswith(prefix) for prefix in skip_prefix
        ):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="ignore")
        out[key_str] = value
    return out


def _sanitize_table_header_for_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop FITS structural keywords before delegating table writes to CFITSIO."""
    skip_keys = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTEND",
        "THEAP",
        "DATASUM",
        "CHECKSUM",
    }
    out: Dict[str, Any] = {}
    for key, value in dict(header or {}).items():
        key_upper = str(key).upper()
        if key_upper in skip_keys or key_upper.startswith("NAXIS"):
            continue
        out[str(key)] = value
    return out


def _write_hdus_with_optional_compression(
    path: str, hdus: List[Any], compress: Union[bool, str] = False
) -> None:
    """Rewrite HDUs, optionally using CFITSIO compressed-image writer."""
    algorithm = _resolve_compression_algorithm(compress)
    if algorithm is None:
        write(path, HDUList(hdus), overwrite=True)
        return

    import torchfits.cpp as cpp

    class _TableWriteProxy:
        def __init__(self, raw_data, header):
            self._raw_data = raw_data
            self.header = header

    payload = []
    for idx, hdu in enumerate(hdus):
        if isinstance(hdu, TableHDURef):
            hdu = hdu.materialize(device="cpu")

        if isinstance(hdu, TableHDU):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(
                _TableWriteProxy(raw_data, _sanitize_table_header_for_write(hdu.header))
            )
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for rewrite at index {idx}: {type(hdu).__name__}"
            )

        # A compressed FITS file uses an empty primary HDU followed by compressed
        # image extensions; skip this placeholder to avoid duplicating it.
        naxis_value = getattr(hdu, "header", {}).get("NAXIS", -1)
        try:
            naxis = int(naxis_value)
        except Exception:
            naxis = -1
        if idx == 0 and naxis == 0:
            xtension = (
                str(getattr(hdu, "header", {}).get("XTENSION", "")).strip().upper()
            )
            if not xtension:
                continue

        hdu_dict: Dict[str, Any] = {"data": hdu.to_tensor("cpu")}
        header = getattr(hdu, "header", None)
        if header:
            hdu_dict["header"] = _sanitize_header_for_compressed_write(header)
        payload.append(hdu_dict)

    _invalidate_path_caches(path)
    cpp.write_fits_file_compressed_images(path, payload, True, algorithm)


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Insert a new HDU into an existing FITS file."""
    if not isinstance(index, int):
        raise TypeError("index must be an integer HDU position")

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if index < 0 or index > len(hdus):
        raise IndexError(f"index {index} out of range for {len(hdus)} HDUs")
    hdus.insert(index, new_hdu)
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


def replace_hdu(
    path: str,
    hdu: Union[int, str],
    data: Any,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Replace an HDU by index or EXTNAME."""

    preserve_header = header is None and not isinstance(data, (TableHDU, TensorHDU))

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target = hdu
    elif isinstance(hdu, str):
        target = None
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target is None:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    if preserve_header:
        # Keep the original header (e.g. EXTNAME/WCS) unless the caller overrides it.
        old_header = getattr(hdus[target], "header", None)
        if old_header is not None:
            if isinstance(new_hdu, TensorHDU):
                new_hdu._header = old_header
            else:
                try:
                    new_hdu.header = old_header
                except Exception:
                    pass

    hdus[target] = new_hdu
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


def delete_hdu(
    path: str,
    hdu: Union[int, str],
    compress: Union[bool, str] = False,
) -> None:
    """Delete an HDU by index or EXTNAME."""
    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target = hdu
    elif isinstance(hdu, str):
        target = None
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target is None:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    del hdus[target]
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


def read_batch(
    file_paths: List[str], hdu: int = 0, device: str = "cpu"
) -> List[Tensor]:
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


def clear_file_cache(
    data: bool = True,
    handles: bool = True,
    meta: bool = True,
    hdu_types: bool = True,
    stats: bool = True,
    cpp: bool = True,
):
    """Clear Python/C++ caches.

    - data: cached returned tensors/tables (Python LRU)
    - handles: cached open CFITSIO handles (Python LRU)
    - meta: cached image metadata (BITPIX/NAXIS/BSCALE/BZERO)
    - hdu_types: cached HDU type hints (IMAGE/TABLE)
    - stats: reset cache performance counters
    - cpp: clear C++-side caches (if available)
    """
    global _cache_stats, _file_cache, _file_handle_cache, _file_handle_sig_cache, _image_meta_cache, _hdu_type_cache, _cold_nommap_cache, _auto_hdu_cache

    if data:
        _file_cache.clear()

    if handles:
        for _, handle in list(_file_handle_cache.items()):
            try:
                handle.close()
            except Exception:
                pass
        _file_handle_cache.clear()
        _file_handle_sig_cache.clear()

    if meta:
        _image_meta_cache.clear()
    if hdu_types:
        _hdu_type_cache.clear()
        _auto_hdu_cache.clear()
    if meta:
        _cold_nommap_cache.clear()

    if stats:
        _cache_stats = {"total_requests": 0, "hits": 0, "misses": 0, "cache_size": 0}

    if cpp:
        try:
            import torchfits.cpp as _cpp

            _cpp.clear_file_cache()
            if hasattr(_cpp, "clear_shared_read_meta_cache"):
                _cpp.clear_shared_read_meta_cache()
        except (AttributeError, RuntimeError):
            pass


def write_checksums(path: str, hdu: int = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    import torchfits.cpp as cpp

    cpp.write_hdu_checksums(path, int(hdu))


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    Returns a dict with `datastatus`, `hdustatus`, and `ok`. Status values:
    - 1: verification correct
    - 0: keyword not present / undefined
    - -1: verification not correct
    """
    import torchfits.cpp as cpp

    datastatus, hdustatus = cpp.verify_hdu_checksums(path, int(hdu))
    return {
        "datastatus": int(datastatus),
        "hdustatus": int(hdustatus),
        "ok": int(datastatus) == 1 and int(hdustatus) == 1,
    }


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


def get_header(path: str, hdu: Union[int, str, None] = 0) -> Header:
    """Get the header of a FITS file.

    Args:
        path: Path to the FITS file.
        hdu: HDU index, name, or `"auto"`/`None` (default: 0).

    Returns:
        Header object.
    """
    import torchfits.cpp as cpp

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = _autodetect_hdu(path, handle_cache_capacity=16)

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


def get_wcs(
    path: str,
    hdu: Union[int, str, None] = "auto",
    device: Optional[Union[str, torch.device]] = None,
) -> WCS:
    """Build a WCS object directly from a FITS file/header HDU.

    Args:
        path: Path to FITS file.
        hdu: HDU index/name, or `"auto"`/`None` for first payload HDU.
        device: Optional torch device for WCS tensor buffers.

    Returns:
        WCS object initialized from the selected HDU header.
    """
    wcs = WCS(get_header(path, hdu=hdu))
    if device is not None:
        wcs = wcs.to(torch.device(device))
    return wcs


_DEBUG_SCALE = os.environ.get("TORCHFITS_DEBUG_SCALE") == "1"
_COLD_NOMMAP = os.environ.get("TORCHFITS_COLD_NOMMAP") == "1"
_COLD_NOCACHE = os.environ.get("TORCHFITS_COLD_NOCACHE") == "1"
