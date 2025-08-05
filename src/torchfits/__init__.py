from .fits_reader import (
    FITS,
    HDU,
    _clear_cache,
    get_dims,
    get_hdu_type,
    get_header,
    get_header_value,
    get_num_hdus,
    read,
)
from .remote import RemoteFetcher
from .smart_cache import SmartCache, configure_cache, get_cache, get_cache_manager
from .table import ColumnInfo, FitsTable, GroupedFitsTable
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


def has_feature(feature_name):
    """Check if a feature is available.

    Args:
        feature_name (str): Name of feature to check.
                            One of: 'remote', 'dataframe', 'examples'

    Returns:
        bool: True if feature is available, False otherwise
    """
    if feature_name == "remote":
        try:
            import fsspec

            return True
        except ImportError:
            return False
    elif feature_name == "dataframe":
        try:
            import torch_frame

            return True
        except ImportError:
            return False
    elif feature_name == "examples":
        try:
            import matplotlib

            return True
        except ImportError:
            return False
    else:
        return False


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
try:
    import torch_frame

    _TORCH_FRAME_AVAILABLE = True
    # Import PyTorch-Frame integration
    from .table import _fits_table_to_torch_frame, read_dataframe
except ImportError:
    _TORCH_FRAME_AVAILABLE = False

    # Provide helpful error functions
    def read_dataframe(*args, **kwargs):
        raise ImportError(
            "PyTorch-Frame is required for read_dataframe(). "
            "Install with: pip install pytorch-frame"
        )

    def _fits_table_to_torch_frame(*args, **kwargs):
        raise ImportError(
            "PyTorch-Frame is required for DataFrame conversion. "
            "Install with: pip install pytorch-frame"
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
    # Backwards compatibility functions
    "read_image",
    "read_table",
    # Enhanced table classes
    "FitsTable",
    "GroupedFitsTable",
    "ColumnInfo",
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
    # PyTorch-Frame integration (optional)
    "read_dataframe",
    # Low-level classes
    "FITS",
    "HDU",
    # Utility functions
    "has_torch_frame",
    "has_advanced_features",
    # Version
    "__version__",
]

# Add advanced features to __all__ if available
# (Currently no advanced features implemented)
pass


def has_torch_frame():
    """Check if PyTorch-Frame is available."""
    return _TORCH_FRAME_AVAILABLE


def has_advanced_features():
    """Check if advanced CFITSIO features are available."""
    return _has_advanced_features
