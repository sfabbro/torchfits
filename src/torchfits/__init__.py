"""
torchfits: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

from typing import Union
from torch import Tensor
import torch
import numpy as np

from torch_frame import TensorFrame

# Force torch symbols to load
_ = torch.empty(1)

from .hdu import HDUList, TensorHDU, TableHDU, Header
from .wcs import WCS
from .cache import configure_for_environment, get_cache_stats, clear_cache

# Auto-configure cache on import
configure_for_environment()

__version__ = "0.1.0"
__all__ = ["read", "write", "open", "HDUList", "TensorHDU", "TableHDU", "Header", "WCS"]


def read(path: str, hdu: Union[int, str] = 0, device: str = 'cpu', mmap: bool = False):
    from . import cpp
    # Support CFITSIO string format: "file.fits[0:200,400:600]"
    if '[' in path and ']' in path:
        # Let CFITSIO handle the parsing natively
        tensor = cpp.read_cfitsio_string(path)
    elif mmap:
        tensor = cpp.read_mmap(path, hdu)
    else:
        tensor = cpp.read_full(path, hdu)
    return tensor.to(device) if device != 'cpu' else tensor


def write(path: str, data, header: Header = None, overwrite: bool = False):
    try:
        if isinstance(data, Tensor):
            from . import cpp
            fits_file = cpp.FITSFile(path, 1)
            fits_file.write_image(data, 0)
            if header:
                # TODO: Implement header writing
                pass
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    except Exception as e:
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e


def open(path: str, mode: str = 'r') -> HDUList:
    return HDUList.fromfile(path, mode)