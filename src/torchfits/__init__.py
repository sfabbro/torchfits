"""Lean public API for torchfits.

The package root intentionally stays light: importing :mod:`torchfits` must not
load tensor runtimes, NumPy, compiled extensions, or optional integration packages.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.5.0b2"

_NAMESPACES: dict[str, str] = {
    "table": "torchfits.table",
    "cache": "torchfits.cache",
    "cpp": "torchfits.cpp",
}

_ROOT_FUNCTIONS: dict[str, tuple[str, str]] = {
    "read": ("torchfits.io", "read"),
    "write": ("torchfits.io", "write"),
    "open": ("torchfits.io", "open"),
    "get_header": ("torchfits.io", "get_header"),
    "read_image": ("torchfits.io", "read_image"),
    "read_tensor": ("torchfits.io", "read_tensor"),
    "read_table": ("torchfits.io", "read_table"),
    "read_hdus": ("torchfits.io", "read_hdus"),
    "read_subset": ("torchfits.io", "read_subset"),
    "open_subset_reader": ("torchfits.io", "open_subset_reader"),
    "read_batch": ("torchfits.io", "read_batch"),
    "get_batch_info": ("torchfits.io", "get_batch_info"),
    "get_cache_performance": ("torchfits.io", "get_cache_performance"),
    "read_large_table": ("torchfits.io", "read_large_table"),
    "read_table_rows": ("torchfits.io", "read_table_rows"),
    "stream_table": ("torchfits.io", "stream_table"),
    "clear_file_cache": ("torchfits.io", "clear_file_cache"),
    "verify_checksums": ("torchfits.io", "verify_checksums"),
    "insert_hdu": ("torchfits.io", "insert_hdu"),
    "replace_hdu": ("torchfits.io", "replace_hdu"),
    "delete_hdu": ("torchfits.io", "delete_hdu"),
    "write_checksums": ("torchfits.io", "write_checksums"),
    "write_tensor": ("torchfits.io", "write_tensor"),
    "read_fast": ("torchfits.io", "read_fast"),
    "to_pandas": ("torchfits.interop", "to_pandas"),
    "to_arrow": ("torchfits.interop", "to_arrow"),
    "to_polars": ("torchfits.interop", "to_polars"),
}

_ROOT_OBJECTS: dict[str, tuple[str, str]] = {
    "Header": ("torchfits.hdu", "Header"),
    "Card": ("torchfits.hdu", "Card"),
    "HDUList": ("torchfits.hdu", "HDUList"),
    "TensorHDU": ("torchfits.hdu", "TensorHDU"),
    "TableHDU": ("torchfits.hdu", "TableHDU"),
}

__all__ = tuple(
    [
        "read",
        "write",
        "open",
        "get_header",
        "read_image",
        "read_tensor",
        "read_table",
        "read_hdus",
        "read_subset",
        "open_subset_reader",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
        "read_batch",
        "get_cache_performance",
        "read_large_table",
        "read_table_rows",
        "stream_table",
        "clear_file_cache",
        "verify_checksums",
        "insert_hdu",
        "replace_hdu",
        "delete_hdu",
        "write_checksums",
        "write_tensor",
        "read_fast",
        "to_pandas",
        "to_arrow",
        "to_polars",
        *_NAMESPACES,
    ]
)

_RUNTIME_INITIALIZED = False


def _ensure_runtime_init() -> None:
    """Initialize optional runtime caches when an I/O entry point is used."""
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    cache = import_module("torchfits.cache")
    cache.configure_for_environment()
    try:
        # Pre-import torch so its dependency libraries (libcudart.so.12,
        # libtorch_cuda.so, libtorch_python.so) are loaded before torchfits._C
        # dlopens them. Otherwise `import torchfits._C` first fails at import
        # time with `libcudart.so.12: cannot open shared object file` even
        # though `import torch; torch.cuda.is_available()` succeeds.
        import torch  # noqa: F401

        cpp = import_module("torchfits._C")
        cache_mb = os.environ.get("TORCHFITS_CFITSIO_CACHE_MB")
        cache_files = os.environ.get("TORCHFITS_CFITSIO_CACHE_FILES")
        if cache_mb is not None or cache_files is not None:
            max_files = int(cache_files) if cache_files is not None else 32
            max_mb = int(cache_mb) if cache_mb is not None else 256
            cpp.configure_cache(max_files, max_mb)
    except Exception:
        pass

    _RUNTIME_INITIALIZED = True


def _runtime_function(name: str) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _ensure_runtime_init()
        module_name, attr_name = _ROOT_FUNCTIONS[name]
        return getattr(import_module(module_name), attr_name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = __name__
    return wrapper


def __getattr__(name: str) -> Any:
    if name in _NAMESPACES:
        module = import_module(_NAMESPACES[name])
        globals()[name] = module
        return module

    if name in _ROOT_FUNCTIONS:
        function = _runtime_function(name)
        globals()[name] = function
        return function

    if name in _ROOT_OBJECTS:
        module_name, attr_name = _ROOT_OBJECTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from . import (
        table as table,
        cache as cache,
    )
    from .hdu import Card as Card
    from .hdu import HDUList as HDUList
    from .hdu import Header as Header
    from .hdu import TableHDU as TableHDU
    from .hdu import TensorHDU as TensorHDU
    from .io import get_header as get_header
    from .io import open as open
    from .io import read as read
    from .io import write as write
