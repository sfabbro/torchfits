"""
torchfits: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

from typing import Union

import torch
import numpy as np
from torch import Tensor

from torch_frame import TensorFrame

# Force torch symbols to load
_ = torch.empty(1)

from .hdu import HDUList, TensorHDU, TableHDU, Header
from .wcs import WCS
from .cache import configure_for_environment, get_cache_stats, clear_cache
from .dataloader import create_dataloader, create_fits_dataloader, create_streaming_dataloader
from .datasets import FITSDataset, IterableFITSDataset
from .transforms import (
    ZScale, AsinhStretch, LogStretch, PowerStretch, Normalize,
    RandomCrop, CenterCrop, RandomFlip, GaussianNoise, ToDevice, Compose,
    create_training_transform, create_validation_transform, create_inference_transform
)
from .buffer import configure_buffers, get_buffer_stats, clear_buffers
from .core import FITSCore, FITSDataType, CompressionType

# Auto-configure cache and buffers on import
configure_for_environment()

__version__ = "0.1.0"
__all__ = [
    # Core I/O functions
    "read", "write", "open",
    # HDU classes
    "HDUList", "TensorHDU", "TableHDU", "Header",
    # WCS functionality
    "WCS",
    # Dataset classes
    "FITSDataset", "IterableFITSDataset",
    # DataLoader factories
    "create_dataloader", "create_fits_dataloader", "create_streaming_dataloader",
    # Transforms
    "ZScale", "AsinhStretch", "LogStretch", "PowerStretch", "Normalize",
    "RandomCrop", "CenterCrop", "RandomFlip", "GaussianNoise", "ToDevice", "Compose",
    "create_training_transform", "create_validation_transform", "create_inference_transform",
    # Core types
    "FITSCore", "FITSDataType", "CompressionType",
    # Utility functions
    "configure_for_environment", "get_cache_stats", "clear_cache",
    "configure_buffers", "get_buffer_stats", "clear_buffers"
]


def read(path: str, hdu: Union[int, str] = 0, device: str = 'cpu', 
         mmap: bool = False, fp16: bool = False, bf16: bool = False):
    """Read FITS data with optimizations.
    
    Args:
        path: File path or cutout specification
        hdu: HDU index or name
        device: Target device ('cpu', 'cuda')
        mmap: Use memory mapping for large files
        fp16: Convert to half precision
        bf16: Convert to bfloat16
    """
    from . import cpp
    from .core import FITSCore
    
    # Support CFITSIO string format: "file.fits[0][100:200,300:400]"
    if '[' in path and ']' in path:
        try:
            # Parse cutout specification
            file_path, hdu_index, slices = FITSCore.parse_cutout_spec(path)
            
            # Open file and read subset
            fits_file = cpp.FITSFile(file_path, 0)
            if len(slices) == 2:
                y_slice, x_slice = slices
                tensor = fits_file.read_subset(hdu_index, 
                                             x_slice.start or 0, y_slice.start or 0,
                                             x_slice.stop or -1, y_slice.stop or -1)
            else:
                # Fall back to full read for non-2D cutouts
                tensor = fits_file.read_image(hdu_index)
        except Exception:
            # Fall back to CFITSIO native parsing
            tensor = cpp.read_cfitsio_string(path)
    elif mmap:
        # Use memory mapping when requested
        tensor = cpp.read_mmap(path, hdu)
    else:
        tensor = cpp.read_full(path, hdu)
    
    # Apply mixed precision conversion
    if fp16:
        tensor = tensor.to(torch.float16)
    elif bf16:
        tensor = tensor.to(torch.bfloat16)
    
    return tensor.to(device) if device != 'cpu' else tensor


def write(path: str, data, header: Header = None, overwrite: bool = False):
    """Write data to FITS file.
    
    Args:
        path: Output file path
        data: Data to write (Tensor or TensorFrame)
        header: Optional FITS header
        overwrite: Whether to overwrite existing files
    """
    import os
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File '{path}' already exists. Use overwrite=True to overwrite.")
    
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
    
    if mode == 'r' and not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")
    
    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open FITS file '{path}': {e}") from e