from .fits_reader_cpp import read as _read_cpp
from .fits_reader_cpp import (
    get_header as _get_header,
    get_header_by_name as _get_header_by_name, 
    get_header_by_number as _get_header_by_number,
    get_header_value as _get_header_value,
    get_dims as _get_dims,
    get_hdu_type as _get_hdu_type,
    get_num_hdus as _get_num_hdus,
    clear_cache as _cpp_clear_cache,  # Renamed to avoid recursion
    world_to_pixel as _world_to_pixel,
    pixel_to_world as _pixel_to_world
)
import torch

def read(
    filename_or_url,
    hdu=None,
    start=None,
    shape=None,
    columns=None,
    start_row=0,
    num_rows=None,
    cache_capacity=0,
    device="cpu"
):
    """
    Read data from a FITS file into a PyTorch tensor.
    
    Args:
        filename_or_url: Path or URL to FITS file
        hdu: HDU to read (number or name)
        start: Starting indices for cutout (list/tuple)
        shape: Shape for cutout (list/tuple)
        columns: Columns to read for tables (list of strings)
        start_row: Starting row for tables
        num_rows: Number of rows to read for tables
        cache_capacity: LRU cache capacity in MB
        device: PyTorch device to store tensor on
        
    Returns:
        For images: CacheEntry with data tensor and header
        For tables: Dict of column name to tensor
    """
    # Convert string device to torch.Device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Call the C++ implementation
    return _read_cpp(
        filename_or_url,
        hdu,
        start,
        shape,
        columns,
        start_row,
        num_rows,
        cache_capacity,
        device
    )

def get_header(filename, hdu=1):
    """
    Reads the FITS header from the specified HDU.
    """
    return _get_header(filename, hdu)

def get_header_by_name(filename, hdu_name):
    """
    Reads the FITS header from the specified HDU name.
    """
    return _get_header_by_name(filename, hdu_name)

def get_header_by_number(filename, hdu_num):
    """
    Reads the FITS header from the specified HDU number.
    """
    return _get_header_by_number(filename, hdu_num)

def get_header_value(filename, hdu=1, key=""):
    """
    Gets the value of a specific FITS header keyword.
    """
    return _get_header_value(filename, hdu, key)

def get_dims(filename, hdu=1):
    """
    Gets the dimensions of a FITS HDU.
    """
    return _get_dims(filename, hdu)

def get_hdu_type(filename, hdu=1):
    """
    Gets the type of a FITS HDU.
    """
    return _get_hdu_type(filename, hdu)

def get_num_hdus(filename):
    """
    Gets the number of HDUs in a FITS file.
    """
    return _get_num_hdus(filename)

def _clear_cache():
    """
    Clears the internal C++ cache.
    """
    # Don't call _clear_cache() (self) but call the imported C++ function
    return _cpp_clear_cache()

def world_to_pixel(world_coords, header):
    """
    Convert world coordinates to pixel coordinates.
    """
    return _world_to_pixel(world_coords, header)

def pixel_to_world(pixel_coords, header):
    """
    Convert pixel coordinates to world coordinates.
    """
    return _pixel_to_world(pixel_coords, header)
