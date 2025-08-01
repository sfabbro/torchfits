from .fits_reader import (
    read,
    get_header,
    get_dims,
    get_header_value,
    get_hdu_type,
    get_num_hdus,
    _clear_cache,
    FITS,
    HDU,
)
from .wcs_utils import (
    world_to_pixel,
    pixel_to_world,
    get_wcs_info,
    is_celestial,
    is_spectral,
    get_coordinate_names,
    transform_cutout_wcs,
)
from .remote import RemoteFetcher
from .smart_cache import SmartCache, get_cache, configure_cache, get_cache_manager
from .table import FitsTable, GroupedFitsTable, ColumnInfo
from .version import __version__

# Advanced CFITSIO features (to be implemented in future versions)
_has_advanced_features = False

# Check for optional dependencies
try:
    import torch_frame
    _TORCH_FRAME_AVAILABLE = True
    # Import PyTorch-Frame integration
    from .table import read_dataframe, _fits_table_to_torch_frame
except ImportError:
    _TORCH_FRAME_AVAILABLE = False
    
    # Provide helpful error functions
    def read_dataframe(*args, **kwargs):
        raise ImportError("PyTorch-Frame is required for read_dataframe(). "
                         "Install with: pip install pytorch-frame")
    
    def _fits_table_to_torch_frame(*args, **kwargs):
        raise ImportError("PyTorch-Frame is required for DataFrame conversion. "
                         "Install with: pip install pytorch-frame")

__all__ = [
    # Core reading functions
    "read", "get_header", "get_dims", "get_header_value", 
    "get_hdu_type", "get_num_hdus", "_clear_cache",
    
    # Enhanced table classes
    "FitsTable", "GroupedFitsTable", "ColumnInfo",
    
    # WCS utilities
    "world_to_pixel", "pixel_to_world", "get_wcs_info", 
    "is_celestial", "is_spectral", "get_coordinate_names", "transform_cutout_wcs",
    
    # Remote and caching
    "RemoteFetcher", "SmartCache", "get_cache", "configure_cache", "get_cache_manager",
    
    # PyTorch-Frame integration (optional)
    "read_dataframe",
    
    # Low-level classes
    "FITS", "HDU",
    
    # Utility functions
    "has_torch_frame", "has_advanced_features",
    
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
