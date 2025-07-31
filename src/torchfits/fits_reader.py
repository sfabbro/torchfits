import torch
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
):
    """
    Read data from a FITS file into a PyTorch tensor.

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

    Returns:
        tuple or dict: For image or cube HDUs, returns a tuple `(data, header)` where
        `data` is a PyTorch tensor and `header` is a dictionary of FITS header keywords.
        For table HDUs, returns a dictionary where keys are column names and values
        are PyTorch tensors.
    """
    return fits_reader_cpp.read(
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