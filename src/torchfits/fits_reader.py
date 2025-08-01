import torch
import warnings
from . import fits_reader_cpp

class FITS:
    """
    A FITS file handler that acts as a context manager for convenient access to FITS files.

    This class provides an object-oriented interface for interacting with FITS files.
    It allows you to open a FITS file and access its HDUs by index or name.

    Example:
        >>> with torchfits.FITS('my_image.fits') as f:
        ...     hdu = f[0]
        ...     data = hdu.read()
        ...     header = hdu.header
    """
    def __init__(self, filename):
        self.filename = filename
        self._hdu_cache = {}
        self._num_hdus = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __len__(self):
        if self._num_hdus is None:
            self._num_hdus = get_num_hdus(self.filename)
        return self._num_hdus

    def __getitem__(self, hdu_spec):
        if hdu_spec not in self._hdu_cache:
            self._hdu_cache[hdu_spec] = HDU(self.filename, hdu_spec)
        return self._hdu_cache[hdu_spec]


class HDU:
    """
    Represents a single Header Data Unit (HDU) in a FITS file.

    This class provides methods to read data and access the header of a specific HDU.
    You typically don't create `HDU` objects directly, but rather access them through
    the `FITS` object.

    Example:
        >>> with torchfits.FITS('my_image.fits') as f:
        ...     primary_hdu = f[0]
        ...     image_data = primary_hdu.read()
    """
    def __init__(self, filename, hdu_spec):
        self.filename = filename
        self.hdu_spec = hdu_spec
        self._header = None

    @property
    def header(self):
        """
        The header of the HDU as a dictionary.

        The header is cached after the first access.
        """
        if self._header is None:
            self._header = get_header(self.filename, self.hdu_spec)
        return self._header

    def read(self, start=None, shape=None, device="cpu"):
        """
        Reads data from this HDU.

        Args:
            start (list[int], optional): The starting pixel coordinates (0-based) for a cutout.
            shape (list[int], optional): The shape of the cutout to read.
            device (str, optional): The device for the output tensor ('cpu' or 'cuda').

        Returns:
            torch.Tensor or dict: A PyTorch tensor for image/cube data, or a dictionary
            of tensors for table data.
        """
        data, _ = read(self.filename, hdu=self.hdu_spec, start=start, shape=shape, device=device)
        return data

def read(
    filename_or_url,
    hdu=1,
    start=None,
    shape=None,
    columns=None,
    start_row=0,
    num_rows=None,
    cache_capacity=0,
    device="cpu",
    format="auto",
    return_metadata=False,
):
    """
    Read data from a FITS file into a PyTorch tensor or FitsTable.

    This function provides a high-level interface for reading data from FITS files,
    supporting images, data cubes, and tables. It automatically handles different
    data types and can read data from local or remote sources.

    Args:
        filename_or_url (str or dict): Path to the FITS file, a CFITSIO-compatible
            URL, or a dictionary with fsspec parameters for remote files.
        hdu (int or str, optional): HDU number (1-based) or name. Defaults to 1 (the primary HDU).
        start (list[int], optional): The starting pixel coordinates (0-based) for a cutout.
            For a 2D image, this would be `[row, column]`.
        shape (list[int], optional): The shape of the cutout to read. If `start` is
            provided, `shape` must also be provided. Use `-1` to read to the end of a dimension.
        columns (list[str], optional): A list of column names to read from a table HDU.
            If `None`, all columns are read.
        start_row (int, optional): The starting row (0-based) to read from a table. Defaults to 0.
        num_rows (int or None, optional): The number of rows to read from a table. If `None`,
            all remaining rows are read.
        cache_capacity (int, optional): The capacity of the in-memory cache in megabytes.
            Defaults to an automatically determined size. Set to 0 to disable.
        device (str, optional): The device to place the output tensor on ('cpu' or 'cuda').
            Defaults to 'cpu'.
        format (str, optional): Output format for table data. Options:
            - 'auto': Auto-detect based on HDU type (tensor for images, table for tables)
            - 'tensor': Return dict of tensors (backward compatible)
            - 'table': Return FitsTable object
            - 'dataframe': Return PyTorch-Frame DataFrame (requires torch_frame)
        return_metadata (bool, optional): Include column metadata for tables. Defaults to False.

    Returns:
        tuple, dict, or FitsTable: 
        - For image/cube HDUs: tuple `(data, header)` where `data` is PyTorch tensor
        - For table HDUs with format='tensor': dict of column_name -> tensor
        - For table HDUs with format='table': FitsTable object
        - For table HDUs with format='dataframe': torch_frame.DataFrame (if available)
    """
    from .table import FitsTable
    
    # Auto-detect format based on HDU type
    if format == "auto":
        try:
            hdu_type = get_hdu_type(filename_or_url, hdu)
            if hdu_type in ["BINTABLE", "TABLE", "BINARY_TBL"]:
                format = "table"  # Use enhanced table format for tables
            else:
                format = "tensor"  # Use tensor format for images/cubes
        except Exception:
            format = "tensor"  # Fallback to tensor format
    
    # Call the C++ backend
    result = fits_reader_cpp.read(
        filename_or_url,
        hdu=hdu,
        start=start,
        shape=shape,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        device=device,
    )
    
    # Handle different formats for table data
    if isinstance(result, dict) and format != "tensor":
        # This is table data, convert to requested format
        if format == "table":
            # Create FitsTable from tensor dict
            metadata = {}
            if return_metadata:
                # Extract metadata from FITS headers
                metadata = _extract_table_metadata(filename_or_url, hdu, columns)
                metadata = _update_metadata_dtypes(metadata, result)
            return FitsTable(result, metadata)
            
        elif format == "dataframe":
            # Convert to PyTorch-Frame DataFrame
            from . import _TORCH_FRAME_AVAILABLE
            if not _TORCH_FRAME_AVAILABLE:
                raise ImportError("PyTorch-Frame is required for dataframe format. "
                                "Install with: pip install pytorch-frame")
            
            # Extract metadata for enhanced DataFrame
            metadata = {}
            if return_metadata:
                metadata = _extract_table_metadata(filename_or_url, hdu, columns)
                metadata = _update_metadata_dtypes(metadata, result)
            
            fits_table = FitsTable(result, metadata)
            return _fits_table_to_torch_frame(fits_table)
    
    # Return original result for images/cubes or when format='tensor'
    return result

def get_header(filename, hdu=1):
    """
    Returns the FITS header as a dictionary.

    Args:
        filename (str): Path to the FITS file.
        hdu (int or str, optional): HDU number (1-based) or name. Defaults to 1.

    Returns:
        dict: The FITS header as a dictionary.
    """
    return fits_reader_cpp.get_header(filename, hdu)


def get_num_hdus(filename):
    """
    Returns the total number of HDUs in the FITS file.

    Args:
        filename (str): Path to the FITS file.

    Returns:
        int: The number of HDUs.
    """
    return fits_reader_cpp.get_num_hdus(filename)

def get_dims(filename, hdu_spec=1):
    """
    Returns the dimensions of a FITS image/cube HDU.
    """
    return fits_reader_cpp.get_dims(filename, hdu_spec)

def get_header_value(filename, hdu_spec, key):
    """
    Returns the value of a single header keyword.
    """
    return fits_reader_cpp.get_header_value(filename, hdu_spec, key)

def get_hdu_type(filename, hdu_spec=1):
    """
    Returns the type of a specific HDU.
    """
    return fits_reader_cpp.get_hdu_type(filename, hdu_spec)

def _clear_cache():
    """
    Clears the internal FITS file cache.
    """
    fits_reader_cpp._clear_cache()


def _extract_table_metadata(filename_or_url, hdu, columns=None):
    """
    Extract table metadata from FITS headers.
    
    Parameters:
    -----------
    filename_or_url : str
        FITS file path or URL
    hdu : int or str
        HDU specification
    columns : list, optional
        Column names to extract metadata for
        
    Returns:
    --------
    Dict[str, ColumnInfo]
        Column metadata
    """
    try:
        # Get full header for the table HDU
        header = get_header(filename_or_url, hdu)
        
        # Extract table structure information
        column_metadata = {}
        
        # Find number of columns
        tfields = header.get('TFIELDS', 0)
        
        for i in range(1, tfields + 1):
            # Get column name
            col_name = header.get(f'TTYPE{i}', f'COL{i}')
            
            # Skip if specific columns requested and this isn't one
            if columns and col_name not in columns:
                continue
                
            # Collect all header info for this column
            col_header = {}
            for key, value in header.items():
                if key.endswith(str(i)) and len(key) > 1:
                    base_key = key[:-len(str(i))]
                    if base_key in ['TTYPE', 'TFORM', 'TUNIT', 'TNULL', 'TSCAL', 
                                   'TZERO', 'TDISP', 'TCOMM', 'TDIM']:
                        col_header[base_key] = value
            
            # Create ColumnInfo (dtype will be set later when we have the tensor)
            from .table import ColumnInfo
            column_metadata[col_name] = ColumnInfo.from_fits_header(
                col_name, col_header, torch.float32  # Placeholder dtype
            )
            
        return column_metadata
        
    except Exception as e:
        # If metadata extraction fails, return empty dict
        warnings.warn(f"Could not extract table metadata: {e}")
        return {}


def _update_metadata_dtypes(metadata, tensor_dict):
    """Update ColumnInfo dtypes with actual tensor dtypes."""
    for col_name, col_info in metadata.items():
        if col_name in tensor_dict:
            # Create new ColumnInfo with correct dtype
            from .table import ColumnInfo
            metadata[col_name] = ColumnInfo(
                name=col_info.name,
                dtype=tensor_dict[col_name].dtype,
                unit=col_info.unit,
                description=col_info.description,
                null_value=col_info.null_value,
                display_format=col_info.display_format,
                coordinate_type=col_info.coordinate_type,
                **col_info.fits_metadata
            )
    return metadata


def _fits_table_to_torch_frame(fits_table):
    """Convert FitsTable to PyTorch-Frame DataFrame."""
    try:
        import torch_frame
        from torch_frame import DataFrame
        
        # Convert tensors to appropriate format for torch_frame
        col_dict = {}
        for name, tensor in fits_table.data.items():
            # Handle different tensor types
            if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                col_dict[name] = tensor.long()
            elif tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                col_dict[name] = tensor.float()
            else:
                # Convert other types to float
                col_dict[name] = tensor.float()
        
        return DataFrame(col_dict)
    except ImportError:
        raise ImportError("PyTorch-Frame is required for dataframe format. "
                         "Install with: pip install pytorch-frame")