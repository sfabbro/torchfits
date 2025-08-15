# ruff: noqa: I001  # import order in this file is intentionally constrained by import side-effects
import os
import sys

import torch  # noqa: F401  # Ensure torch initializes its ScalarType registry before extension loads

# Early OpenMP duplicate runtime mitigation (must run before importing C++ extension)
if sys.platform == "darwin" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    try:
        # Heuristic detection (cheap): presence of core torch + any clang/omp hints in sys.modules path strings
        # We set the var pre-emptively to avoid hard crash during extension import if duplication exists.
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    except Exception:
        pass

# Load extension after torch (import side-effect). Ignore type checkers for binary module symbol.
from . import fits_reader_cpp  # type: ignore[attr-defined]
from .fits_reader import (  # Torch-frame / DataFrame helpers
    FITS,
    HDU,
    _clear_cache,
    read_many_small_cutouts,
    read_many_cutouts_multi_hdu,
    dataframe_round_trip,
    fits_table_to_torch_frame,
    get_dims,
    get_hdu_type,
    get_header,
    get_header_value,
    get_num_hdus,
    read,
    read_table_with_null_masks,
    torch_frame_round_trip_file,
    torch_frame_to_fits,
)

# Public re-exports
from .dataset import FITSDataset, FITSItemSpec, FITSIterableDataset, read_multi_sky_cutouts
from .fits_writer import (
    append_hdu,
    update_data,
    update_header,
    write,
    write_cube,
    write_image,
    write_mef,
    write_table,
    write_variable_length_array,
)
from .remote import RemoteFetcher
from .smart_cache import SmartCache, configure_cache, get_cache, get_cache_manager
from .table import (
    ColumnInfo,
    FitsTable,
    GroupedFitsTable,
    apply_null_masks_to_dict,
    pad_ragged,
)
from .version import __version__
from .wcs_utils import (
    get_coordinate_names,
    get_wcs_info,
    is_celestial,
    is_spectral,
    pixel_to_world,
    transform_cutout_wcs,
    world_to_pixel,
)


# Diagnostics passthroughs from extension
def get_last_read_info():
    """Return diagnostic information from the C++ reader, if available."""
    try:
        return fits_reader_cpp.get_last_read_info()  # type: ignore[attr-defined]
    except Exception:
        return {}

# Temporary mitigation for OpenMP duplicate runtime crashes on macOS/conda
# If multiple libomp instances are detected (common with PyTorch + compiler toolchain),
# set KMP_DUPLICATE_LIB_OK early. This should be replaced by a proper build / linkage fix.
## (Old late-setting OpenMP mitigation removed; moved earlier)


def get_version():
    """Return the version of torchfits."""
    return __version__


def get_dependency_versions():
    """Return a dictionary of dependency versions."""
    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata

    deps = {}

    # Core dependencies
    for pkg_name in ["torch", "numpy"]:
        try:
            deps[pkg_name] = metadata.version(pkg_name)
        except metadata.PackageNotFoundError:
            deps[pkg_name] = "not installed"

    # Optional dependencies
    for pkg_name in ["astropy", "pytorch-frame", "fsspec", "fitsio"]:
        try:
            deps[pkg_name] = metadata.version(pkg_name)
        except metadata.PackageNotFoundError:
            deps[pkg_name] = "not installed"

    return deps


def cache_stats():
    """Return current SmartCache statistics as a dictionary.

    Convenience wrapper around get_cache_manager().get_cache_statistics().
    """
    try:
        mgr = get_cache_manager()
        return mgr.get_cache_statistics()
    except Exception:
        return {}


def memory_cache_stats():
    """Return in-memory RealSmartCache stats as a dictionary."""
    try:
        return fits_reader_cpp._memory_cache_stats()  # type: ignore[attr-defined]
    except Exception:
        return {}


def has_feature(feature_name):
    """Check if an optional feature is available.

    feature_name: one of {'remote', 'dataframe', 'examples'}.
    """
    import importlib.util as _util

    mapping = {
        "remote": "fsspec",
        "dataframe": "torch_frame",
        "examples": "matplotlib",
    }
    mod = mapping.get(feature_name)
    if mod is None:
        return False
    return _util.find_spec(mod) is not None


# Backwards compatibility functions
def read_image(filename_or_url, hdu=0, start=None, shape=None, **kwargs):
    """
    Backwards compatibility function for reading images.
    This is equivalent to read() but ensures tensor format return.
    """
    result = read(
        filename_or_url, hdu=hdu, start=start, shape=shape, format="tensor", **kwargs
    )
    return result


def read_table(
    filename_or_url, hdu=1, columns=None, start_row=0, num_rows=None, **kwargs
):
    """
    Backwards compatibility function for reading tables.
    This is equivalent to read() but ensures tensor format return.
    """
    result = read(
        filename_or_url,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        format="tensor",
        **kwargs,
    )
    return result


# Advanced CFITSIO features (to be implemented in future versions)
_has_advanced_features = False

# Check for optional dependencies
import importlib.util as _util  # noqa: E402  (kept near feature checks)

_TORCH_FRAME_AVAILABLE = _util.find_spec("torch_frame") is not None


def read_dataframe(*args, **kwargs):
    """Raise ImportError if PyTorch-Frame isn't installed (placeholder)."""
    if not _TORCH_FRAME_AVAILABLE:  # pragma: no cover - optional path
        raise ImportError(
            "PyTorch-Frame is required for read_dataframe(). Install with: pip install pytorch-frame"
        )
    raise NotImplementedError(
        "read_dataframe isn't provided in this package; use fits_table_to_torch_frame instead."
    )


__all__ = [
    # Core reading functions
    "read",
    "get_header",
    "get_dims",
    "get_header_value",
    "get_hdu_type",
    "get_num_hdus",
    "_clear_cache",
    "read_many_small_cutouts",
    "read_many_cutouts_multi_hdu",
    # Writing functions (v1.0)
    "write",
    "write_mef",
    "append_hdu",
    "update_header",
    "update_data",
    "write_image",
    "write_table",
    "write_cube",
    "write_variable_length_array",
    # Null mask helper
    "read_table_with_null_masks",
    # Backwards compatibility functions
    "read_image",
    "read_table",
    # Enhanced table classes
    "FitsTable",
    "GroupedFitsTable",
    "ColumnInfo",
    "pad_ragged",
    "apply_null_masks_to_dict",
    # WCS utilities
    "world_to_pixel",
    "pixel_to_world",
    "get_wcs_info",
    "is_celestial",
    "is_spectral",
    "get_coordinate_names",
    "transform_cutout_wcs",
    # Remote and caching
    "RemoteFetcher",
    "SmartCache",
    "get_cache",
    "configure_cache",
    "get_cache_manager",
    # In-memory cache stats
    "memory_cache_stats",
    # PyTorch-Frame integration (optional)
    "read_dataframe",
    # Round-trip helpers
    "dataframe_round_trip",
    "fits_table_to_torch_frame",
    "torch_frame_to_fits",
    "torch_frame_round_trip_file",
    # Low-level classes
    "FITS",
    "HDU",
    # Utility functions
    "has_torch_frame",
    "has_advanced_features",
    # Version
    "__version__",
]

__all__ += ["FITSItemSpec", "FITSDataset", "FITSIterableDataset", "read_multi_sky_cutouts"]

# Add advanced features to __all__ if available
# (Currently no advanced features implemented)
pass


def has_torch_frame():
    """Check if PyTorch-Frame is available."""
    return _TORCH_FRAME_AVAILABLE


def has_advanced_features():
    """Check if advanced CFITSIO features are available."""
    return _has_advanced_features
