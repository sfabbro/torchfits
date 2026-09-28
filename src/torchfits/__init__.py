"""Lean public API for torchfits.

The package root intentionally stays light: importing :mod:`torchfits` must not
load tensor runtimes, NumPy, compiled extensions, or optional integration packages.

Transforms live under :mod:`torchfits.transforms`. Arrow tables under
:mod:`torchfits.table`. HDU types are available as root names and via
:mod:`torchfits.hdu`.
"""

import os
import sys

# Must run before libomp is loaded (import torch after torchfits, or pixi
# activation.env for torch-first). Required on macOS when both PyTorch and the
# extension link libomp; harmless elsewhere but process-wide so scope to Darwin.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "1.1.3"

_NAMESPACES: dict[str, str] = {
    "table": "torchfits.table",
    "cache": "torchfits.cache",
    "cpp": "torchfits.cpp",
    "transforms": "torchfits.transforms",
    "data": "torchfits.data",
    "where": "torchfits.where",
    "hdu": "torchfits.hdu",
}

_ROOT_FUNCTIONS: dict[str, tuple[str, str]] = {
    "read": ("torchfits.io", "read"),
    "write": ("torchfits.io", "write"),
    "open": ("torchfits.io", "open"),
    "read_header": ("torchfits.io", "read_header"),
    "read_colnames": ("torchfits.io", "read_colnames"),
    "read_extname": ("torchfits.io", "read_extname"),
    "read_hdu_type": ("torchfits.io", "read_hdu_type"),
    "read_keys": ("torchfits.io", "read_keys"),
    "read_nrows": ("torchfits.io", "read_nrows"),
    "read_num_hdus": ("torchfits.io", "read_num_hdus"),
    "read_shape": ("torchfits.io", "read_shape"),
    "read_table_info": ("torchfits.io", "read_table_info"),
    "read_tensor": ("torchfits.io", "read_tensor"),
    "read_hdus": ("torchfits.io", "read_hdus"),
    "read_subset": ("torchfits.io", "read_subset"),
    "open_subset_reader": ("torchfits.io", "open_subset_reader"),
    "open_table_reader": ("torchfits.io", "open_table_reader"),
    "read_batch": ("torchfits.io", "read_batch"),
    "read_batch_info": ("torchfits.io", "read_batch_info"),
    "get_cache_performance": ("torchfits.io", "get_cache_performance"),
    "clear_file_cache": ("torchfits.io", "clear_file_cache"),
    "clear_all_caches": ("torchfits.cache", "clear_all_caches"),
    "verify_checksums": ("torchfits.io", "verify_checksums"),
    "insert_hdu": ("torchfits.io", "insert_hdu"),
    "replace_hdu": ("torchfits.io", "replace_hdu"),
    "delete_hdu": ("torchfits.io", "delete_hdu"),
    "write_checksums": ("torchfits.io", "write_checksums"),
    "write_tensor": ("torchfits.io", "write_tensor"),
    "to_pandas": ("torchfits.interop", "to_pandas"),
    "to_arrow": ("torchfits.interop", "to_arrow"),
    "to_polars": ("torchfits.interop", "to_polars"),
    "to_astropy": ("torchfits.interop", "to_astropy"),
}

# These entry points return metadata or manage metadata caches only.  Keep
# their root lookup free of the tensor-runtime initializer; the native module
# itself defers the Python ``torch`` import until a tensor boundary is used.
_METADATA_ROOT_FUNCTIONS = frozenset(
    {
        "open",
        "read_header",
        "read_colnames",
        "read_extname",
        "read_hdu_type",
        "read_keys",
        "read_nrows",
        "read_num_hdus",
        "read_shape",
        "read_table_info",
        "read_batch_info",
        "get_cache_performance",
        "clear_file_cache",
        "clear_all_caches",
        "verify_checksums",
        "delete_hdu",
        "write_checksums",
    }
)

_ROOT_OBJECTS: dict[str, tuple[str, str]] = {
    "Header": ("torchfits.hdu", "Header"),
    "Card": ("torchfits.hdu", "Card"),
    "HDUList": ("torchfits.hdu", "HDUList"),
    "TensorHDU": ("torchfits.hdu", "TensorHDU"),
    "TableHDU": ("torchfits.hdu", "TableHDU"),
    "TableHDURef": ("torchfits.hdu", "TableHDURef"),
}

__all__ = tuple(
    [
        "read",
        "write",
        "open",
        "read_header",
        "read_colnames",
        "read_extname",
        "read_hdu_type",
        "read_keys",
        "read_nrows",
        "read_num_hdus",
        "read_shape",
        "read_table_info",
        "read_tensor",
        "read_hdus",
        "read_subset",
        "open_subset_reader",
        "open_table_reader",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
        "TableHDURef",
        "read_batch",
        "read_batch_info",
        "get_cache_performance",
        "clear_all_caches",
        "clear_file_cache",
        "verify_checksums",
        "insert_hdu",
        "replace_hdu",
        "delete_hdu",
        "write_checksums",
        "write_tensor",
        "to_pandas",
        "to_arrow",
        "to_polars",
        "to_astropy",
        *_NAMESPACES,
    ]
)

_RUNTIME_INITIALIZED = False
_ATTR_CACHE: dict[str, Any] = {}
# RLock: loading a namespace (e.g. table) may re-enter __getattr__ via
# ``from torchfits import fits_schema`` / similar relative imports.
_ATTR_LOCK = threading.RLock()


def _ensure_runtime_init() -> None:
    """Initialize optional runtime caches when an I/O entry point is used."""
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    cache = import_module("torchfits.cache")
    cache.configure_for_environment()
    # Pre-import torch so its dependency libraries (libcudart.so.12,
    # libtorch_cuda.so, libtorch_python.so) are loaded before torchfits._C.
    import torch  # noqa: F401

    native = import_module("torchfits._C")
    # Metadata may have imported the extension before torch was present.  The
    # module initializer then intentionally skipped this check, so enforce the
    # ABI again at the tensor boundary after loading torch.
    getattr(native, "_check_torch_abi")()
    _check_core_build_id(native)

    _RUNTIME_INITIALIZED = True


def _check_core_build_id(native: Any) -> None:
    """Fail if ``_C`` and the loaded ``libtorchfits_core`` are from different builds.

    ``_C`` resolves its CFITSIO symbols against ``libtorchfits_core`` at load
    time. A half-rebuilt checkout can leave a fresh ``_C`` next to a stale core,
    which every compile-time check accepts and which only misbehaves once a
    struct layout disagrees. This is the first point where both sides are
    loaded, so it is where the comparison can be made.
    """
    extension_id = getattr(native, "__core_build_id__", None)
    if extension_id is None:
        return
    from torchfits import _core as core

    library_id = core.core_library_build_id()
    if extension_id != library_id:
        raise ImportError(
            "torchfits._C and the libtorchfits_core it loaded are from different "
            f"builds:\n  _C:      {extension_id}\n  library: {library_id}\n"
            "Reinstall the package (pip install -e . --no-build-isolation) so every "
            "native artifact comes from the same build."
        )


def __getattr__(name: str) -> Any:
    cached = _ATTR_CACHE.get(name)
    if cached is not None:
        return cached

    with _ATTR_LOCK:
        cached = _ATTR_CACHE.get(name)
        if cached is not None:
            return cached

        if name in _NAMESPACES:
            if name == "cpp":
                _ensure_runtime_init()
            value: Any = import_module(_NAMESPACES[name])
        elif name in _ROOT_FUNCTIONS:
            if name not in _METADATA_ROOT_FUNCTIONS:
                _ensure_runtime_init()
            module_name, attr_name = _ROOT_FUNCTIONS[name]
            value = getattr(import_module(module_name), attr_name)
        elif name in _ROOT_OBJECTS:
            module_name, attr_name = _ROOT_OBJECTS[name]
            value = getattr(import_module(module_name), attr_name)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        _ATTR_CACHE[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_ATTR_CACHE))


if TYPE_CHECKING:
    from . import (
        cache as cache,
        cpp as cpp,
        data as data,
        hdu as hdu,
        table as table,
        transforms as transforms,
        where as where,
    )
    from .hdu import Card as Card
    from .hdu import HDUList as HDUList
    from .hdu import Header as Header
    from .hdu import TableHDU as TableHDU
    from .hdu import TableHDURef as TableHDURef
    from .hdu import TensorHDU as TensorHDU
    from .cache import clear_all_caches as clear_all_caches
    from .io import clear_file_cache as clear_file_cache
    from .io import delete_hdu as delete_hdu
    from .io import read_batch_info as read_batch_info
    from .io import get_cache_performance as get_cache_performance
    from .io import read_header as read_header
    from .io import read_colnames as read_colnames
    from .io import read_extname as read_extname
    from .io import read_hdu_type as read_hdu_type
    from .io import read_keys as read_keys
    from .io import read_nrows as read_nrows
    from .io import read_num_hdus as read_num_hdus
    from .io import read_shape as read_shape
    from .io import read_table_info as read_table_info
    from .io import insert_hdu as insert_hdu
    from .io import open as open
    from .io import open_subset_reader as open_subset_reader
    from .io import open_table_reader as open_table_reader
    from .io import read as read
    from .io import read_batch as read_batch
    from .io import read_hdus as read_hdus
    from .io import read_subset as read_subset
    from .io import read_tensor as read_tensor
    from .io import replace_hdu as replace_hdu
    from .io import verify_checksums as verify_checksums
    from .io import write as write
    from .io import write_checksums as write_checksums
    from .io import write_tensor as write_tensor
    from .interop import to_arrow as to_arrow
    from .interop import to_astropy as to_astropy
    from .interop import to_pandas as to_pandas
    from .interop import to_polars as to_polars
