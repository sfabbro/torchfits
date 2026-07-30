"""Explicit public surface for the native FITS extension.

New symbols added to :mod:`torchfits._C` stay private until deliberately added
to ``__all__``. Attribute access for names in ``__all__`` delegates to the
extension.

Path-taking entry points apply :func:`guard_fits_path` so private/loopback
``http``/``https``/``ftp`` URLs cannot reach CFITSIO through this module either
(matching the Python I/O façades). Handle-based APIs are unchanged.
"""

# ruff: noqa: F822  # nanobind symbols are installed into globals below.

from __future__ import annotations

from typing import Any, Callable

import torchfits._C as _C
from torchfits._io_engine.paths import guard_fits_path

__all__ = (
    "FITSFile",
    "HDUInfo",
    "SubsetReader",
    "TableReader",
    "append_fits_table_rows",
    "clear_file_cache",
    "clear_shared_read_meta_cache",
    "configure_cache",
    "delete_fits_table_rows",
    "drop_fits_table_columns",
    "get_cache_size",
    "get_hdu_type",
    "get_num_hdus",
    "insert_fits_table_rows",
    "open_and_read_headers",
    "open_fits_file",
    "read_fits_table",
    "read_fits_table_filtered",
    "read_fits_table_from_handle",
    "read_fits_table_rows",
    "read_fits_table_rows_from_handle",
    "read_fits_table_rows_numpy",
    "read_fits_table_rows_numpy_from_handle",
    "read_full",
    "read_full_cached",
    "read_full_nocache",
    "read_full_numpy",
    "read_full_numpy_cached",
    "read_full_raw",
    "read_full_raw_with_scale",
    "read_full_scaled_cpu",
    "read_full_unmapped",
    "read_full_unmapped_raw",
    "read_hdus_batch",
    "read_hdus_sequence_last",
    "read_header",
    "read_header_dict",
    "read_header_string",
    "read_colnames",
    "read_hdu_type",
    "read_keys",
    "read_nrows",
    "read_num_hdus",
    "read_shape",
    "read_table_info",
    "read_images_batch",
    "read_tensor_from_handle",
    "rename_fits_table_columns",
    "resolve_hdu_name_cached",
    "update_fits_table_rows",
    "update_fits_table_rows_mmap",
    "verify_hdu_checksums",
    "write_fits_file",
    "write_fits_file_compressed_images",
    "write_fits_table",
    "write_hdu_checksums",
    "write_hdu_header_cards",
    "delete_hdu_header_key",
)

# First positional arg is a FITS path (or CFITSIO URL).
_PATH_FIRST = frozenset(
    {
        "append_fits_table_rows",
        "delete_fits_table_rows",
        "delete_hdu_header_key",
        "drop_fits_table_columns",
        "insert_fits_table_rows",
        "open_and_read_headers",
        "open_fits_file",
        "read_colnames",
        "read_fits_table",
        "read_fits_table_filtered",
        "read_fits_table_rows",
        "read_fits_table_rows_numpy",
        "read_full",
        "read_full_cached",
        "read_full_nocache",
        "read_full_numpy",
        "read_full_numpy_cached",
        "read_full_raw",
        "read_full_raw_with_scale",
        "read_full_scaled_cpu",
        "read_full_unmapped",
        "read_full_unmapped_raw",
        "read_hdus_batch",
        "read_hdus_sequence_last",
        "read_header_dict",
        "read_hdu_type",
        "read_keys",
        "read_nrows",
        "read_num_hdus",
        "read_shape",
        "read_table_info",
        "rename_fits_table_columns",
        "resolve_hdu_name_cached",
        "update_fits_table_rows",
        "update_fits_table_rows_mmap",
        "verify_hdu_checksums",
        "write_fits_file",
        "write_fits_file_compressed_images",
        "write_fits_table",
        "write_hdu_checksums",
        "write_hdu_header_cards",
    }
)

# First positional arg is a sequence of paths.
_PATH_LIST_FIRST = frozenset({"read_images_batch"})

# Path-taking class constructors (nanobind types; wrap via factory).
_PATH_CTORS = frozenset({"FITSFile", "SubsetReader", "TableReader"})


def _guard_path_first(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(path: Any, *args: Any, **kwargs: Any) -> Any:
        guard_fits_path(path)
        return fn(path, *args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def _guard_path_list_first(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(paths: Any, *args: Any, **kwargs: Any) -> Any:
        for path in paths:
            guard_fits_path(path)
        return fn(paths, *args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def _guard_ctor(cls: Any) -> Callable[..., Any]:
    def factory(path: Any, *args: Any, **kwargs: Any) -> Any:
        guard_fits_path(path)
        return cls(path, *args, **kwargs)

    factory.__name__ = getattr(cls, "__name__", "factory")
    factory.__doc__ = getattr(cls, "__doc__", None)
    # Preserve nanobind type identity for isinstance when possible.
    factory.__wrapped__ = cls  # type: ignore[attr-defined]
    return factory


for _name in __all__:
    _obj = getattr(_C, _name)
    if _name in _PATH_FIRST:
        globals()[_name] = _guard_path_first(_obj)
    elif _name in _PATH_LIST_FIRST:
        globals()[_name] = _guard_path_list_first(_obj)
    elif _name in _PATH_CTORS:
        globals()[_name] = _guard_ctor(_obj)
    else:
        globals()[_name] = _obj


def __getattr__(name: str) -> Any:
    return getattr(_C, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_C)))
