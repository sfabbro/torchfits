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
from .table import FitsTable, GroupedFitsTable, ColumnInfo
from .version import __version__

# Check for optional dependencies
try:
    import torch_frame
    _TORCH_FRAME_AVAILABLE = True
except ImportError:
    _TORCH_FRAME_AVAILABLE = False

__all__ = [
    "read",
    "get_header",
    "get_dims",
    "get_header_value",
    "get_hdu_type",
    "get_num_hdus",
    "_clear_cache",
    "FITS",
    "HDU",
    "FitsTable",
    "GroupedFitsTable",
    "ColumnInfo",
    "world_to_pixel",
    "pixel_to_world",
    "get_wcs_info",
    "is_celestial", 
    "is_spectral",
    "get_coordinate_names",
    "transform_cutout_wcs",
    "__version__",
]
