"""
FITS writing functionality for TorchFits v1.0

This module provides comprehensive writing capabilities for FITS files,
including images, tables, and multi-extension files.
"""

import warnings
from typing import Dict, List, Optional, Union, Any

import torch

try:
    from . import fits_reader_cpp  # The C++ module includes both reading and writing
    _WRITER_AVAILABLE = True
except ImportError:
    _WRITER_AVAILABLE = False
    fits_reader_cpp = None


def write(
    filename: str,
    data: Union[torch.Tensor, dict, "FitsTable"],
    header: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
    hdu_type: str = "auto",
    **kwargs
) -> None:
    """
    Write PyTorch tensors or table data to a FITS file.

    This is the main writing function that can handle images, cubes, and tables.
    It automatically detects the appropriate format based on the input data type.

    Parameters:
    -----------
    filename : str
        Output FITS filename
    data : torch.Tensor, dict, or FitsTable
        Data to write. Can be:
        - torch.Tensor: Written as image/cube HDU
        - dict: Written as table HDU (keys are column names, values are tensors)
        - FitsTable: Written as table HDU with metadata
    header : dict, optional
        Header keywords to write as key-value pairs
    overwrite : bool, optional
        Whether to overwrite existing file. Default: False
    hdu_type : str, optional
        Force specific HDU type ('image', 'table', 'auto'). Default: 'auto'
    **kwargs
        Additional keyword arguments:
        - column_units: List of units for table columns
        - column_descriptions: List of descriptions for table columns
        - extname: Extension name for the HDU

    Example:
    --------
    >>> import torch
    >>> import torchfits
    >>> 
    >>> # Write an image
    >>> image = torch.randn(100, 100)
    >>> torchfits.write("output.fits", image, {"OBJECT": "Test Image"})
    >>> 
    >>> # Write a table
    >>> table_data = {
    ...     "RA": torch.tensor([120.1, 120.2, 120.3]),
    ...     "DEC": torch.tensor([-30.1, -30.2, -30.3])
    ... }
    >>> torchfits.write("catalog.fits", table_data, 
    ...                 column_units=["deg", "deg"])
    """
    if header is None:
        header = {}

    # Auto-detect HDU type if not specified
    if hdu_type == "auto":
        if isinstance(data, torch.Tensor):
            hdu_type = "image"
        elif isinstance(data, dict):
            hdu_type = "table"
        elif hasattr(data, "__class__") and data.__class__.__name__ == "FitsTable":
            hdu_type = "table"
        else:
            raise ValueError(f"Cannot auto-detect HDU type for data of type {type(data)}")

    if hdu_type == "image":
        if not isinstance(data, torch.Tensor):
            raise ValueError("Image HDU requires torch.Tensor data")
        
        if not _WRITER_AVAILABLE:
            raise RuntimeError("C++ writing functionality not available")
        
        fits_reader_cpp.write_tensor_to_fits(filename, data, header, overwrite)
        
    elif hdu_type == "table":
        if not _WRITER_AVAILABLE:
            raise RuntimeError("C++ writing functionality not available")
            
        if isinstance(data, dict):
            column_units = kwargs.get("column_units", [])
            column_descriptions = kwargs.get("column_descriptions", [])
            fits_reader_cpp.write_table_to_fits(
                filename, data, header, column_units, column_descriptions, overwrite
            )
        elif hasattr(data, "__class__") and data.__class__.__name__ == "FitsTable":
            fits_reader_cpp.write_fits_table(filename, data, overwrite)
        else:
            raise ValueError("Table HDU requires dict or FitsTable data")
    else:
        raise ValueError(f"Unknown HDU type: {hdu_type}")


def write_mef(
    filename: str,
    data_list: List[torch.Tensor],
    headers: Optional[List[Dict[str, str]]] = None,
    extnames: Optional[List[str]] = None,
    overwrite: bool = False
) -> None:
    """
    Write multiple tensors to a Multi-Extension FITS (MEF) file.

    Parameters:
    -----------
    filename : str
        Output FITS filename
    data_list : List[torch.Tensor]
        List of tensors to write as separate HDUs
    headers : List[dict], optional
        List of header dictionaries, one per HDU
    extnames : List[str], optional
        List of extension names for each HDU
    overwrite : bool, optional
        Whether to overwrite existing file. Default: False

    Example:
    --------
    >>> primary = torch.randn(50, 50)
    >>> science = torch.randn(100, 100) 
    >>> error = torch.randn(100, 100)
    >>> 
    >>> torchfits.write_mef("multi.fits", [primary, science, error],
    ...                     headers=[{"OBSTYPE": "BIAS"}, {"OBSTYPE": "SCIENCE"}, {"OBSTYPE": "ERROR"}],
    ...                     extnames=["PRIMARY", "SCI", "ERR"])
    """
    if headers is None:
        headers = []
    if extnames is None:
        extnames = []

    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")

    fits_reader_cpp.write_tensors_to_mef(filename, data_list, headers, extnames, overwrite)


def append_hdu(
    filename: str,
    data: torch.Tensor,
    header: Optional[Dict[str, str]] = None,
    extname: str = ""
) -> None:
    """
    Append a new HDU to an existing FITS file.

    Parameters:
    -----------
    filename : str
        Existing FITS filename to append to
    data : torch.Tensor
        Tensor data to append as new HDU
    header : dict, optional
        Header keywords for the new HDU
    extname : str, optional
        Extension name for the new HDU

    Example:
    --------
    >>> # Create initial file
    >>> torchfits.write("base.fits", torch.randn(50, 50))
    >>> 
    >>> # Append additional HDU
    >>> torchfits.append_hdu("base.fits", torch.randn(100, 100), 
    ...                      {"EXTNAME": "SCIENCE"})
    """
    if header is None:
        header = {}

    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")

    fits_reader_cpp.append_hdu_to_fits(filename, data, header, extname)


def update_header(
    filename: str,
    updates: Dict[str, str],
    hdu: Union[int, str] = 1
) -> None:
    """
    Update header keywords in an existing FITS file.

    Parameters:
    -----------
    filename : str
        FITS filename to update
    updates : dict
        Dictionary of keyword-value pairs to update
    hdu : int or str, optional
        HDU number (1-based) or name to update. Default: 1

    Example:
    --------
    >>> torchfits.update_header("data.fits", {
    ...     "OBJECT": "Updated Object Name",
    ...     "EXPTIME": "300.0"
    ... })
    """
    if isinstance(hdu, str):
        # Convert HDU name to number (would need implementation)
        raise NotImplementedError("HDU name lookup not yet implemented")
    
    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")
    
    fits_reader_cpp.update_fits_header(filename, hdu, updates)


def update_data(
    filename: str,
    new_data: torch.Tensor,
    hdu: Union[int, str] = 1,
    start: Optional[List[int]] = None,
    shape: Optional[List[int]] = None
) -> None:
    """
    Update data in an existing FITS file (in-place modification).

    Parameters:
    -----------
    filename : str
        FITS filename to update
    new_data : torch.Tensor
        New data to write
    hdu : int or str, optional
        HDU number (1-based) or name to update. Default: 1
    start : List[int], optional
        Starting coordinates for partial update (0-based)
    shape : List[int], optional
        Shape of region to update

    Example:
    --------
    >>> # Update entire HDU
    >>> torchfits.update_data("data.fits", torch.randn(100, 100))
    >>> 
    >>> # Update subset
    >>> torchfits.update_data("data.fits", torch.randn(50, 50), 
    ...                       start=[25, 25], shape=[50, 50])
    """
    if isinstance(hdu, str):
        # Convert HDU name to number (would need implementation)
        raise NotImplementedError("HDU name lookup not yet implemented")
    
    if start is None:
        start = []
    if shape is None:
        shape = []

    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")

    fits_reader_cpp.update_fits_data(filename, hdu, new_data, start, shape)


# Convenience functions for specific use cases

def write_image(
    filename: str,
    image: torch.Tensor,
    header: Optional[Dict[str, str]] = None,
    overwrite: bool = False
) -> None:
    """
    Write a tensor as a FITS image file.
    
    This is a convenience function equivalent to:
    write(filename, image, header, overwrite, hdu_type="image")
    """
    write(filename, image, header, overwrite, hdu_type="image")


def write_table(
    filename: str,
    table_data: Union[dict, Any],  # Any to avoid FitsTable type error
    header: Optional[Dict[str, str]] = None,
    column_units: Optional[List[str]] = None,
    column_descriptions: Optional[List[str]] = None,
    overwrite: bool = False
) -> None:
    """
    Write table data as a FITS binary table.
    
    This is a convenience function for writing tables with optional metadata.
    """
    write(filename, table_data, header, overwrite, hdu_type="table",
          column_units=column_units or [], 
          column_descriptions=column_descriptions or [])


def write_cube(
    filename: str,
    cube: torch.Tensor,
    header: Optional[Dict[str, str]] = None,
    overwrite: bool = False
) -> None:
    """
    Write a 3D tensor as a FITS data cube.
    
    This is a convenience function for 3D data with appropriate header defaults.
    """
    if cube.ndim != 3:
        raise ValueError("Data cube must be 3-dimensional")
    
    cube_header = header.copy() if header else {}
    if "CTYPE3" not in cube_header:
        cube_header["CTYPE3"] = "WAVE"
    
    write(filename, cube, cube_header, overwrite, hdu_type="image")


# Integration with FitsTable class
def _register_fits_table_writer():
    """Register write method with FitsTable class if available."""
    try:
        from .table import FitsTable
        
        def write_to_fits(self, filename: str, overwrite: bool = False):
            """Write this FitsTable to a FITS file."""
            if not _WRITER_AVAILABLE:
                raise RuntimeError("C++ writing functionality not available")
            fits_reader_cpp.write_fits_table(filename, self, overwrite)
        
        # Try to add method to FitsTable class - may fail due to type checker
        try:
            FitsTable.write = write_to_fits
        except AttributeError:
            pass  # Type checker doesn't like dynamic attribute assignment
        
    except ImportError:
        pass  # FitsTable not available


# Register the writer method when module is imported
_register_fits_table_writer()
