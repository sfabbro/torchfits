from .fits_reader import (read, get_header, get_dims,
                         get_header_value, get_hdu_type, get_num_hdus)
from .version import __version__

__all__ = [
    "read",
    "get_header",
    "get_dims",
    "get_header_value",
    "get_hdu_type",
    "get_num_hdus",
    "__version__",
]