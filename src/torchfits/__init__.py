"""
torchfits: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

from typing import Union, Optional, List

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
from .header_parser import fast_parse_header

# Main API functions
__all__ = [
    'read', 'write', 'writeto', 'open', 'info',
    'HDUList', 'TensorHDU', 'TableHDU', 'Header',
    'WCS', 'FITSDataset', 'IterableFITSDataset',
    'create_dataloader', 'create_fits_dataloader', 'create_streaming_dataloader',
    'configure_for_environment', 'get_cache_stats', 'clear_cache',
    'configure_buffers', 'get_buffer_stats', 'clear_buffers',
    'ZScale', 'AsinhStretch', 'LogStretch', 'PowerStretch', 'Normalize',
    'RandomCrop', 'CenterCrop', 'RandomFlip', 'GaussianNoise', 'ToDevice', 'Compose',
    'create_training_transform', 'create_validation_transform', 'create_inference_transform',
    'FITSCore', 'FITSDataType', 'CompressionType', 'fast_parse_header'
]

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
    # Header parsing
    "fast_parse_header",
    # Utility functions
    "configure_for_environment", "get_cache_stats", "clear_cache",
    "configure_buffers", "get_buffer_stats", "clear_buffers"
]

def _read_header_fast(file_handle, hdu_index: int, fast_header: bool = True):
    """Read header using fast bulk parsing or fallback to slow method."""
    from . import cpp
    
    if fast_header:
        try:
            # Try fast bulk header reading
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, Exception):
            # Fall back to slow method if fast parsing fails
            pass
    
    # Fall back to keyword-by-keyword reading
    return cpp.read_header(file_handle, hdu_index)


def read(path: str, hdu: Union[int, str] = 0, device: str = 'cpu', 
         mmap: bool = False, fp16: bool = False, bf16: bool = False,
         columns: Optional[List[str]] = None, start_row: int = 1, num_rows: int = -1,
         cache_capacity: int = 10, fast_header: bool = True):
    """Read FITS data with optimizations.
    
    Args:
        path: File path or cutout specification
        hdu: HDU index or name
        device: Target device ('cpu', 'cuda')
        mmap: Use memory mapping for large files
        fp16: Convert to half precision
        bf16: Convert to bfloat16
        columns: List of column names for table reading (None = all columns)
        start_row: Starting row for table reading (1-based)
        num_rows: Number of rows to read (-1 = all rows)
        cache_capacity: File cache capacity
        fast_header: Use fast bulk header parsing (default: True)
        
    Returns:
        torch.Tensor for images, dict for tables, and header dict
    """
    from . import cpp
    from .core import FITSCore
    from .hdu import TableHDU, Header
    
    # Support CFITSIO string format: "file.fits[0][100:200,300:400]"
    if '[' in path and ']' in path:
        try:
            # Parse cutout specification
            file_path, hdu_index, slices = FITSCore.parse_cutout_spec(path)
            
            # Open file and read subset
            file_handle = cpp.open_fits_file(file_path, "r")
            try:
                if len(slices) == 2:
                    y_slice, x_slice = slices
                    tensor = cpp.read_subset(file_handle, hdu_index, 
                                           x_slice.start or 0, y_slice.start or 0,
                                           x_slice.stop or -1, y_slice.stop or -1)
                else:
                    # Fall back to full read for non-2D cutouts
                    tensor = cpp.read_full(file_handle, hdu_index)
                return tensor, {}
            finally:
                cpp.close_fits_file(file_handle)
        except Exception:
            # Fall back to CFITSIO native parsing
            tensor = cpp.read_cfitsio_string(path)
            return tensor, {}
    else:
        # Resolve HDU name to index if needed
        hdu_index = hdu
        if isinstance(hdu, str):
            # Open file to resolve HDU name
            try:
                file_handle = cpp.open_fits_file(path, "r")
                try:
                    num_hdus = cpp.get_num_hdus(file_handle)
                    hdu_index = None
                    for i in range(num_hdus):
                        header = cpp.read_header(file_handle, i)
                        if header.get('EXTNAME') == hdu:
                            hdu_index = i
                            break
                    if hdu_index is None:
                        raise ValueError(f"HDU '{hdu}' not found in file")
                finally:
                    cpp.close_fits_file(file_handle)
            except Exception as e:
                raise RuntimeError(f"Failed to resolve HDU name '{hdu}': {e}")
        
        # Determine HDU type and read accordingly
        try:
            file_handle = cpp.open_fits_file(path, "r")
            try:
                hdu_type = cpp.get_hdu_type(file_handle, hdu_index)
                
                if hdu_type in ['BINARY_TBL', 'ASCII_TBL', 'TABLE']:
                    # Table reading - try different approaches
                    try:
                        # Use table reader with column/row selection support
                        result = cpp.read_fits_table_from_handle(file_handle, hdu_index)
                        tensor_dict = result.get('tensor_dict', {})
                        header = _read_header_fast(file_handle, hdu_index, fast_header)
                        
                        # Filter columns if requested
                        if columns is not None:
                            tensor_dict = {k: v for k, v in tensor_dict.items() if k in columns}
                        
                        return tensor_dict, header
                    except Exception as e:
                        # Fall back to simple table reader
                        result = cpp.read_fits_table(path, hdu_index)
                        header = cpp.read_header_dict(path, hdu_index)
                        return result, header
                else:
                    # Image reading
                    if mmap:
                        tensor = cpp.read_mmap(path, hdu_index)
                    else:
                        tensor = cpp.read_full(file_handle, hdu_index)
                        
                    # Apply mixed precision conversion
                    if fp16:
                        tensor = tensor.to(torch.float16)
                    elif bf16:
                        tensor = tensor.to(torch.bfloat16)
                    
                    final_tensor = tensor.to(device) if device != 'cpu' else tensor
                    header = _read_header_fast(file_handle, hdu_index, fast_header)
                    return final_tensor, header
            finally:
                cpp.close_fits_file(file_handle)
                
        except Exception as e:
            # Fall back to original behavior for backward compatibility
            if mmap:
                tensor = cpp.read_mmap(path, hdu_index if isinstance(hdu_index, int) else hdu)
            else:
                tensor = cpp.read_full(path, hdu_index if isinstance(hdu_index, int) else hdu)
            
            # Apply mixed precision conversion
            if fp16:
                tensor = tensor.to(torch.float16)
            elif bf16:
                tensor = tensor.to(torch.bfloat16)
            
            return tensor.to(device) if device != 'cpu' else tensor, {}


def write(path: str, data, header: Header = None, overwrite: bool = False, compress: bool = False):
    """Write data to FITS file.
    
    Args:
        path: Output file path
        data: Data to write (Tensor, TensorFrame, or HDUList)
        header: Optional FITS header dictionary
        overwrite: Whether to overwrite existing files
        compress: Whether to use tile compression (Rice algorithm)
    """
    import os
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File '{path}' already exists. Use overwrite=True to overwrite.")
    
    try:
        # Remove existing file if overwriting
        if overwrite and os.path.exists(path):
            os.remove(path)
            
        from . import cpp
        
        if isinstance(data, Tensor):
            # Write single tensor as primary HDU
            fits_file = cpp.FITSFile(path, 1)  # Create mode
            fits_file.write_image(data, 0)
            
            # TODO: Implement header writing and compression in C++ layer
            if header:
                pass  # Header writing will be implemented in C++
                            
        elif hasattr(data, '__iter__') and not isinstance(data, (str, Tensor)):
            # Write multiple HDUs
            hdus = []
            for item in data:
                if isinstance(item, dict):
                    hdus.append(item)
                elif hasattr(item, '__dict__'):
                    hdus.append(item.__dict__)
                else:
                    hdus.append({'data': item})
            cpp.write_fits_file(path, hdus, overwrite)
            
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")
            
    except Exception as e:
        # Clean up partial file on error
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e


def writeto(path: str, data, header: Header = None, overwrite: bool = False, compress: bool = False):
    """Alias for write() function for astropy compatibility."""
    return write(path, data, header, overwrite, compress)


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