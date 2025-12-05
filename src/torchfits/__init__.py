"""
torchfits: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

from typing import Union, Optional, List, Dict, Any

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
from .core import FITSCore, CompressionType
from .header_parser import fast_parse_header
from .frame import to_tensor_frame, read_tensor_frame, write_tensor_frame



# Simple cache tracking for tests
_cache_stats = {'total_requests': 0, 'hits': 0, 'misses': 0, 'cache_size': 0}
_file_cache = {}

# Auto-configure cache and buffers on import
configure_for_environment()

__version__ = "0.1.0"
__all__ = [
    # Core I/O functions
    "read", "write", "open", "writeto", "get_header", "read_subset",
    # Batch operations
    "read_batch", "get_batch_info", "get_cache_performance", "clear_file_cache", "read_large_table",
    # HDU classes
    "HDUList", "TensorHDU", "TableHDU", "Header",
    "HDU", "PrimaryHDU", "ImageHDU", "BinTableHDU",
    # WCS functionality
    "WCS",
    # Dataset classes
    "FITSDataset", "IterableFITSDataset",
    # DataLoader factories
    "create_dataloader", "create_fits_dataloader", "create_streaming_dataloader",
    # Transforms
    "ZScale", "AsinhStretch", "LogStretch", "PowerStretch", "Normalize", "MinMaxScale", "RobustScale",
    "RandomCrop", "CenterCrop", "RandomFlip", "GaussianNoise", "PoissonNoise", "RandomRotation", "RedshiftShift", "PerturbByError",
    "ToDevice", "Compose",
    "create_training_transform", "create_validation_transform", "create_inference_transform",
    # Core types
    "FITSCore", "CompressionType",
    # Header parsing
    "fast_parse_header",
    # Utility functions
    "configure_for_environment", "get_cache_stats", "clear_cache",
    "configure_buffers", "get_buffer_stats", "clear_buffers",
    # Vertical Slice
    "read_image_fast_int16",
    # Phase 2
    "read_image_fast_new",
    # Integration
    "to_tensor_frame", "read_tensor_frame", "write_tensor_frame",
    "to_pandas",
    "to_arrow"
]

# Import C++ extension functions
try:
    from .cpp import read_image_fast_int16, read_image_fast_new
except ImportError:
    def read_image_fast_int16(*args, **kwargs):
        raise ImportError("C++ extension not loaded correctly")
    def read_image_fast_new(*args, **kwargs):
        raise ImportError("C++ extension not loaded correctly")

def _read_header_fast(file_handle, hdu_index: int, fast_header: bool = True):
    """Read header using fast bulk parsing or fallback to slow method."""
    import torchfits.cpp as cpp
    
    if fast_header:
        try:
            # Try fast bulk header reading
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, RuntimeError, IOError):
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
        
    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: File not found
        RuntimeError: FITS reading errors
    """
    # Input validation
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")
    
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be non-negative")
    
    if start_row < 1:
        raise ValueError("start_row must be >= 1 (FITS uses 1-based indexing)")
    
    if num_rows < -1 or num_rows == 0:
        raise ValueError("num_rows must be > 0 or -1 for all rows")
    
    if device not in ['cpu', 'cuda', 'mps'] and not device.startswith('cuda:'):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    # Use C++ backend for high-performance I/O
    import torchfits.cpp as cpp

    # Cache tracking
    global _cache_stats, _file_cache
    _cache_stats['total_requests'] += 1

    # Use tuple for cache key (faster than f-string)
    cache_key = (path, hdu, device, fp16, bf16, tuple(columns) if columns else None, start_row, num_rows)
    if cache_key in _file_cache:
        _cache_stats['hits'] += 1
        cached_data, cached_header = _file_cache[cache_key]
        
        if device != 'cpu':
            if isinstance(cached_data, torch.Tensor):
                cached_data = cached_data.to(device)
            elif isinstance(cached_data, dict):
                # Move tensors in dict to device
                new_data = {}
                for k, v in cached_data.items():
                    if isinstance(v, torch.Tensor):
                        new_data[k] = v.to(device)
                    else:
                        new_data[k] = v
                cached_data = new_data
                
        return cached_data, cached_header
    else:
        _cache_stats['misses'] += 1

    try:
        # Use handle-based API (most reliable)
        file_handle = cpp.open_fits_file(path, "r")

        try:
            # Handle HDU selection (int or name)
            if isinstance(hdu, str):
                num_hdus = cpp.get_num_hdus(file_handle)
                hdu_num = None
                for i in range(num_hdus):
                    try:
                        hdr = cpp.read_header(file_handle, i)
                        if hdr.get('EXTNAME') == hdu:
                            hdu_num = i
                            break
                    except:
                        continue
                if hdu_num is None:
                    raise ValueError(f"HDU '{hdu}' not found in file")
            else:
                hdu_num = hdu

            # Get header
            header_data = cpp.read_header(file_handle, hdu_num)
            header = Header(header_data)

            # Try reading as IMAGE first (for slow path)
            try:
                data = cpp.read_full(file_handle, hdu_num, mmap)

                # Apply precision conversion
                if fp16:
                    data = data.to(torch.float16)
                elif bf16:
                    data = data.to(torch.bfloat16)

                # Move to device
                if device != 'cpu':
                    data = data.to(device)

                # Cache result
                _file_cache[cache_key] = (data.cpu() if device != 'cpu' else data, header)
                _cache_stats['cache_size'] = len(_file_cache)

                return data, header

            except (RuntimeError, TypeError):
                # Not an image, read as table using C++ backend
                try:
                    # table_result = cpp.read_fits_table_from_handle(file_handle, hdu_num)
                    # Pass columns and mmap flag to C++ reader
                    col_list = columns if columns else []
                    table_result = cpp.read_fits_table(path, hdu_num, col_list, mmap)
                    
                    # table_result is the dictionary of tensors directly
                    table_data = table_result
                    
                    # If columns were passed to C++, filtering is already done.
                    # But if columns was None, we got all columns.
                    # If columns was not None, we got only requested columns.
                    
                    # Handle row slicing if needed (though C++ reader reads all by default for now)
                    # Ideally C++ reader should support slicing, but for now we slice after reading
                    if start_row > 1 or num_rows != -1:
                        for k, v in table_data.items():
                            end_row = start_row + num_rows - 1 if num_rows != -1 else len(v)
                            # Adjust for 1-based indexing
                            table_data[k] = v[start_row-1:end_row]

                    # Cache result (move to CPU for storage)
                    if device != 'cpu':
                        # Should already be on CPU from C++ reader, but just in case
                        # Actually C++ reader returns CPU tensors.
                        # We store them as is.
                        pass
                        
                    _file_cache[cache_key] = (table_data, header)
                    _cache_stats['cache_size'] = len(_file_cache)
                    
                    # Move to device if requested (for initial return)
                    if device != 'cpu':
                        new_data = {}
                        for k, v in table_data.items():
                            if isinstance(v, torch.Tensor):
                                new_data[k] = v.to(device)
                            else:
                                new_data[k] = v
                        table_data = new_data
                        
                    return table_data, header
                except Exception as e:
                    raise RuntimeError(f"Failed to read table extension: {e}")

        finally:
            try:
                cpp.close_fits_file(file_handle)
            except:
                pass


    except Exception as e:
        raise RuntimeError(f"Failed to read FITS file '{path}': {e}") from e


def read_subset(path: str, hdu: int, x1: int, y1: int, x2: int, y2: int) -> Tensor:
    """Read a rectangular subset of an image HDU.
    
    Args:
        path: Path to FITS file
        hdu: HDU index
        x1, y1: Start coordinates (0-based, inclusive)
        x2, y2: End coordinates (0-based, exclusive)
        
    Returns:
        Tensor containing the subset
    """
    import torchfits.cpp as cpp
    
    # Check cache first (optional, but consistent)
    # For now, no caching for subsets to avoid complexity
    
    try:
        file_handle = cpp.open_fits_file(path, "r")
        try:
            return file_handle.read_subset(hdu, x1, y1, x2, y2)
        finally:
            # Explicitly close to release resources
            # cpp.close_fits_file(file_handle) # FITSFileV2 has close method, but let's rely on destructor or explicit close if exposed
            file_handle.close()
    except Exception as e:
        raise RuntimeError(f"Failed to read subset from '{path}': {e}") from e

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
        # Import C++ backend
        import torchfits.cpp as cpp
        
        hdus_to_write = []
        
        # Check for table dictionary (col_name -> tensor)
        if isinstance(data, dict) and 'data' not in data:
             # Assume table if values are tensors
             is_table = True
             for k, v in data.items():
                 if not isinstance(v, (torch.Tensor, np.ndarray)):
                     is_table = False
                     break
             
             if is_table:
                 cpp.write_fits_table(path, data, header if header else {}, overwrite)
                 return
        
        if isinstance(data, Tensor):
            # Single image HDU
            hdu_dict = {'data': data}
            if header:
                hdu_dict['header'] = header
            hdus_to_write.append(hdu_dict)
            
        elif hasattr(data, '__iter__') and not isinstance(data, (str, Tensor)):
            # List of HDUs
            for item in data:
                if isinstance(item, dict):
                    # Already a dict (e.g. from previous read)
                    # Ensure 'data' is present
                    if 'data' in item:
                        hdus_to_write.append(item)
                elif isinstance(item, Tensor):
                    hdus_to_write.append({'data': item})
                # Handle objects with .data attribute (like TensorHDU)
                elif hasattr(item, 'data') and isinstance(item.data, Tensor):
                     hdu_dict = {'data': item.data}
                     if hasattr(item, 'header'):
                         hdu_dict['header'] = item.header
                     hdus_to_write.append(hdu_dict)
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")
            
        # Call C++ writer
        cpp.write_fits_file(path, hdus_to_write, overwrite)
            
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


def read_batch(file_paths: List[str], device: str = 'cpu') -> List[Tensor]:
    """Read multiple FITS files in batch.
    
    Args:
        file_paths: List of file paths to read
        device: Target device for tensors
        
    Returns:
        List of tensors from each file
    """
    results = []
    for path in file_paths:
        try:
            tensor, _ = read(path, device=device)
            results.append(tensor)
        except Exception:
            # Skip failed files
            continue
    return results


def get_batch_info(file_paths: List[str]) -> Dict[str, Any]:
    """Get information about a batch of FITS files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary with batch statistics
    """
    valid_files = 0
    for path in file_paths:
        try:
            import os
            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue
    
    return {
        'num_files': len(file_paths),
        'valid_files': valid_files
    }


def get_cache_performance() -> Dict[str, Any]:
    """Get cache performance statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    global _cache_stats
    total = _cache_stats['total_requests']
    hits = _cache_stats['hits']
    misses = _cache_stats['misses']
    
    return {
        'cache_size': _cache_stats['cache_size'],
        'hit_rate': hits / total if total > 0 else 0.0,
        'miss_rate': misses / total if total > 0 else 0.0,
        'total_requests': total,
        'hits': hits,
        'misses': misses
    }


def clear_file_cache():
    """Clear the file cache."""
    global _cache_stats, _file_cache
    _file_cache.clear()
    _cache_stats = {'total_requests': 0, 'hits': 0, 'misses': 0, 'cache_size': 0}
    
    # Also try to clear C++ cache if available
    try:
        import torchfits.cpp as cpp
        cpp.clear_file_cache()
    except (AttributeError, RuntimeError):
        pass


def read_large_table(file_path: str, hdu: int = 1, max_memory_mb: int = 100, 
                    streaming: bool = False) -> Dict[str, Any]:
    """Read large FITS table with memory management.
    
    Args:
        file_path: Path to FITS file
        hdu: HDU index
        max_memory_mb: Maximum memory usage in MB
        streaming: Whether to use streaming mode
        
    Returns:
        Dictionary with table data
    """
    try:
        import torchfits.cpp as cpp
        import os
        
        if not os.path.exists(file_path):
            return {}
        
        # Use C++ table reader
        return cpp.read_fits_table(file_path, hdu)
            
    except Exception:
        return {}


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


def get_header(path: str, hdu: Union[int, str] = 0) -> Header:
    """Get the header of a FITS file.
    
    Args:
        path: Path to the FITS file.
        hdu: HDU index or name (default: 0).
        
    Returns:
        Header object.
    """
    import torchfits.cpp as cpp
    
    # Resolve HDU index if string
    if isinstance(hdu, str):
        for i in range(100): # Arbitrary limit
            try:
                h_list = cpp.read_header_dict(path, i)
                if not h_list: break
                h = Header(h_list)
                if h.get('EXTNAME') == hdu:
                    return h
            except:
                break
        raise ValueError(f"HDU '{hdu}' not found")
    
    return Header(cpp.read_header_dict(path, hdu))