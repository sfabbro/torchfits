"""Python bindings and helpers for reading FITS files with torchfits."""
import warnings

import torch

try:  # pragma: no cover - extension import side effects
    from . import fits_reader_cpp  # type: ignore
except Exception:  # pragma: no cover
    fits_reader_cpp = None  # type: ignore


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

    def read(
        self,
        start=None,
        shape=None,
        device="cpu",
        enable_buffered: bool | None = None,
        enable_mmap: bool | None = None,
    ):
        """
        Read data from this HDU.

        Args:
            start (list[int], optional): The starting pixel coordinates (0-based) for a cutout.
            shape (list[int], optional): The shape of the cutout to read.
            device (str, optional): The device for the output tensor ('cpu' or 'cuda').

        Returns:
            torch.Tensor or dict: A PyTorch tensor for image/cube data, or a dictionary
            of tensors for table data.
        """
        data, _ = read(
            self.filename,
            hdu=self.hdu_spec,
            start=start,
            shape=shape,
            device=device,
            enable_buffered=enable_buffered,
            enable_mmap=enable_mmap,
        )
        return data

    def update_header(self, updates: dict):
        """Update header keywords for this HDU in-place.

        Parameters
        ----------
        updates : dict
            Mapping of FITS keyword -> value to write/update.

        Notes
        -----
            This invalidates the cached header on this HDU instance so a subsequent
            access will fetch the updated values.
        """
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dict of keyword->value")
        from .fits_writer import (
            update_header as _update_header,  # local import to avoid cycle
        )

        # hdu_spec here follows user 0-based style; update_header handles int/str
        _update_header(
            self.filename,
            updates,
            hdu=(
                self.hdu_spec
                if not isinstance(self.hdu_spec, int)
                else self.hdu_spec + 1
            ),
        )
        # Invalidate cached header
        self._header = None

    def refresh_header(self):
        """Force re-read of header from disk (cache bypass)."""
        self._header = None
        return self.header

    def __repr__(self):
        try:
            hdr = self.header
            keys = list(hdr.keys())[:5]
            preview = ", ".join(keys)
            return f"HDU(spec={self.hdu_spec}, keys=[{preview}...], file='{self.filename}')"
        except Exception:
            return f"HDU(spec={self.hdu_spec}, file='{self.filename}')"


def read(
    filename_or_url,
    hdu=0,  # Changed to 0-based indexing for astropy/fitsio compatibility
    start=None,
    shape=None,
    columns=None,
    start_row=0,
    num_rows=None,
    cache_capacity=0,
    device="cpu",
    format="auto",
    return_metadata=False,
    use_cache: bool = True,
    enable_buffered: bool | None = None,
    enable_mmap: bool | None = None,
):
    """
    Read data from a FITS file into a PyTorch tensor or FitsTable.

    This function provides a high-level interface for reading data from FITS files,
    supporting images, data cubes, and tables. It automatically handles different
    data types and can read data from local or remote sources.

    Args:
        filename_or_url (str or dict): Path to the FITS file, a CFITSIO-compatible
            URL, or a dictionary with fsspec parameters for remote files.
        hdu (int or str, optional): HDU number (0-based like astropy/fitsio) or name.
                                   Defaults to 0 (the primary HDU).
                                   Note: Internally converted to 1-based for CFITSIO compatibility.
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

    # Convert 0-based HDU indexing (astropy/fitsio style) to 1-based (CFITSIO style)
    cfitsio_hdu = hdu + 1 if isinstance(hdu, int) else hdu

    # Auto-detect format based on HDU type
    if format == "auto":
        try:
            hdu_type = get_hdu_type(filename_or_url, hdu)  # Use original 0-based hdu
            if hdu_type in ["BINTABLE", "TABLE", "BINARY_TBL"]:
                format = "table"  # Use enhanced table format for tables
            else:
                format = "tensor"  # Use tensor format for images/cubes
        except Exception:
            format = "tensor"  # Fallback to tensor format

    # Smart cache integration for remote URLs (http/https)
    src_obj = filename_or_url
    try:
        if (
            use_cache
            and isinstance(filename_or_url, str)
            and filename_or_url.startswith(("http://", "https://"))
        ):
            # Deferred import to avoid any potential import cycles
            from .smart_cache import get_cache

            local_path = get_cache().get_or_fetch_file(
                filename_or_url, hdu=None, format_type="file"
            )
            src_obj = local_path
    except Exception:
        # Non-fatal: fall back to original source on any cache error
        src_obj = filename_or_url

    # Heuristic: if caller didn’t force flags and we’re doing a full-image read (no start/shape)
    # on a local path, decide mmap/buffered automatically. The backend is still authoritative
    # and will fall back safely.
    try:
        auto_flags = (enable_mmap is None and enable_buffered is None and isinstance(src_obj, str)
                      and start is None and shape is None and not str(src_obj).startswith(("http://", "https://")))
        if auto_flags:
            from .heuristics import choose_read_mode_for_image
            choice = choose_read_mode_for_image(src_obj, hdu)
            # Only set if not already specified
            enable_mmap = choice.get("enable_mmap", enable_mmap)
            enable_buffered = choice.get("enable_buffered", enable_buffered)
    except Exception:
        pass

    # Call the C++ backend with converted HDU number
    # Debug markers for segfault localization
    # (debug removed for stability)
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    result = fits_reader_cpp.read(
        src_obj,
        hdu=cfitsio_hdu,  # Use 1-based HDU for CFITSIO
        start=start,
        shape=shape,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        enable_mmap=enable_mmap,
        enable_buffered=enable_buffered,
        device=device,
    )

    # Handle different formats for table data
    # For table HDUs, result is tuple (table_dict, header)
    # For image HDUs, result is tuple (tensor, header)
    if isinstance(result, tuple) and len(result) == 2:
        data, header = result

        # Check if this is table data (data is a dict) and we want enhanced format
        if isinstance(data, dict) and format != "tensor":
            if format == "table":
                # Create FitsTable from tensor dict
                metadata = {}
                if return_metadata:
                    # Extract metadata from FITS headers
                    metadata = _extract_table_metadata(filename_or_url, hdu, columns)
                    metadata = _update_metadata_dtypes(metadata, data)
                return FitsTable(data, metadata)

            elif format == "dataframe":
                # Convert to PyTorch-Frame DataFrame
                from . import _TORCH_FRAME_AVAILABLE

                if not _TORCH_FRAME_AVAILABLE:
                    raise ImportError(
                        "PyTorch-Frame is required for dataframe format. "
                        "Install with: pip install pytorch-frame"
                    )

                # Fast path: when metadata isn't requested, build DataFrame directly
                # from the tensor dict, avoiding constructing a FitsTable wrapper.
                if not return_metadata:
                    return _tensor_dict_to_torch_frame(data)

                # Metadata-aware path: build FitsTable to preserve schema info
                metadata = _extract_table_metadata(filename_or_url, hdu, columns)
                metadata = _update_metadata_dtypes(metadata, data)
                fits_table = FitsTable(data, metadata)
                return _fits_table_to_torch_frame(fits_table)

        # For tensor format table data: return (dict, header) to maintain stable API
        if format == "tensor" and isinstance(data, dict):
            return data, header

    # Return original result for other cases
    return result


def read_many_small_cutouts(filename_or_url, hdu=0, starts=None, shape=None, device="cpu"):
    """Fast path to read many small cutouts from the same image HDU.

    Parameters
    ----------
    filename_or_url : str or dict
        FITS path or URL.
    hdu : int or str, default 0
        0-based HDU index or name.
    starts : list[list[int]]
        List of 0-based start coordinates for each cutout.
    shape : list[int]
        Cutout shape common to all starts.
    device : str, default 'cpu'
        Target device for tensors.

    Returns
    -------
    list[torch.Tensor]
        Tensors in the same order as starts.
    """
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    if not starts or not shape:
        return []
    cfitsio_hdu = hdu + 1 if isinstance(hdu, int) else hdu
    return fits_reader_cpp.read_many_cutouts(
        filename_or_url, cfitsio_hdu, starts, shape, device
    )


def _build_null_masks(table_dict, header):
    """Build boolean masks for columns with TNULLn sentinel in header.

    Parameters
    ----------
    table_dict : Dict[str, torch.Tensor]
        Column data as returned by backend
    header : Dict[str, Any]
        FITS header for the table HDU

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping colname -> bool tensor where True indicates null (only for columns with sentinel)
    """
    try:
        import torch
    except Exception:
        return {}

    # Map column index -> (name, sentinel)
    raw_tfields = header.get("TFIELDS", 0)
    try:
        tfields = int(raw_tfields)
    except Exception:
        tfields = 0
    idx_to_name = {}
    for i in range(1, tfields + 1):
        name = header.get(f"TTYPE{i}")
        if not name:
            continue
        tnull_key = f"TNULL{i}"
        if tnull_key in header:
            try:
                sentinel = int(header[tnull_key])
            except Exception:
                continue
            idx_to_name[name] = sentinel

    masks = {}
    for col, sentinel in idx_to_name.items():
        if col in table_dict and isinstance(table_dict[col], torch.Tensor):
            tensor = table_dict[col]
            # Only apply to integer-like columns (leave floats which may already encode NaN)
            if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                masks[col] = tensor.eq(sentinel)
    return masks


def read_table_with_null_masks(filename_or_url, hdu=1, **kwargs):
    """Stable helper: read table tensor data plus null masks.

    Performs a standard read() call (format='tensor') then derives null masks from header via separate header fetch.
    Avoids modifying the core read() return shape (which caused instability on some builds).
    Returns (table_dict, header, null_masks).
    """
    data, header = read(
        filename_or_url,
        hdu=hdu,
        format="tensor",
        **{k: v for k, v in kwargs.items() if k not in ("return_null_masks",)},
    )
    # Fetch header explicitly to ensure we have full table keywords
    hdr = get_header(filename_or_url, hdu)
    masks = _build_null_masks(data, hdr)
    return data, header, masks


def get_header(filename, hdu: int | str = 0):
    """Return the FITS header as a dictionary.

    Args:
        filename (str): Path to the FITS file.
        hdu (int or str, optional): HDU number (0-based like astropy/fitsio) or name. Defaults to 0.

    Returns:
        dict: The FITS header as a dictionary.
    """
    cfitsio_hdu = hdu + 1 if isinstance(hdu, int) else hdu
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    return fits_reader_cpp.get_header(filename, cfitsio_hdu)


def get_num_hdus(filename):
    """Return the total number of HDUs in the FITS file.

    Args:
        filename (str): Path to the FITS file.

    Returns:
        int: The number of HDUs.
    """
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    return fits_reader_cpp.get_num_hdus(filename)


def get_dims(filename, hdu_spec=0):
    """Return the dimensions of a FITS image/cube HDU.
    """
    cfitsio_hdu = hdu_spec + 1 if isinstance(hdu_spec, int) else hdu_spec
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    return fits_reader_cpp.get_dims(filename, cfitsio_hdu)


def get_header_value(filename, hdu_spec: int | str, key):
    """Return the value of a single header keyword.
    """
    cfitsio_hdu = hdu_spec + 1 if isinstance(hdu_spec, int) else hdu_spec
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    return fits_reader_cpp.get_header_value(filename, cfitsio_hdu, key)


def get_hdu_type(filename, hdu_spec: int | str = 0):
    """Return the type of a specific HDU.
    """
    cfitsio_hdu = hdu_spec + 1 if isinstance(hdu_spec, int) else hdu_spec
    if fits_reader_cpp is None:
        raise RuntimeError("fits_reader_cpp extension not loaded")
    return fits_reader_cpp.get_hdu_type(filename, cfitsio_hdu)


def _clear_cache():
    """Clear the internal FITS file cache.
    """
    if fits_reader_cpp is None:
        return
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
        # Header values are returned as strings; coerce TFIELDS to int
        raw_tfields = header.get("TFIELDS", 0)
        try:
            tfields = int(raw_tfields)
        except Exception:
            tfields = 0

        for i in range(1, tfields + 1):
            # Get column name
            col_name = header.get(f"TTYPE{i}", f"COL{i}")

            # Skip if specific columns requested and this isn't one
            if columns and col_name not in columns:
                continue

            # Collect relevant header info for this column, mapping suffixed keys to base attributes
            col_header = {}
            suffix = str(i)
            wanted = {
                "TFORM",
                "TUNIT",
                "TNULL",
                "TSCAL",
                "TZERO",
                "TDISP",
                "TCOMM",
                "TDIM",
            }
            for key, value in header.items():
                if key.endswith(suffix) and len(key) > len(suffix):
                    base_key = key[: -len(suffix)]
                    if base_key in wanted:
                        # Normalize value to plain string to avoid later concat issues
                        try:
                            if isinstance(value, int | float):
                                v_norm = str(value)
                            else:
                                v_norm = str(value).strip()
                        except Exception:
                            v_norm = str(value)
                        col_header[base_key] = v_norm

            # Inject TTYPE (name) explicitly for completeness
            col_header.setdefault("TTYPE", col_name)

            # Create ColumnInfo (dtype will be set later when we have the tensor)
            from .table import ColumnInfo

            column_metadata[col_name] = ColumnInfo.from_fits_header(
                col_name, col_header, torch.float32  # Placeholder dtype
            )

        return column_metadata

    except Exception as e:
        # If metadata extraction fails, return empty dict
        warnings.warn(f"Could not extract table metadata: {e}", stacklevel=2)
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
                coordinate_type=getattr(col_info, "coordinate_type", None),
                **col_info.fits_metadata,
            )
    return metadata


def _fits_table_to_torch_frame(fits_table):
    """Convert FitsTable to PyTorch-Frame DataFrame."""
    try:
        from torch_frame import DataFrame

        # Convert tensors to appropriate format for torch_frame
        col_dict = {}
        for name, tensor in fits_table.data.items():
            # Ensure we hand torch tensors to torch_frame; normalize dtypes.
            if not isinstance(tensor, torch.Tensor):
                # Skip non-tensor columns for now (e.g., string lists)
                continue
            # Avoid unnecessary clones/casts; only cast when required.
            dt = tensor.dtype
            if dt in (torch.int8, torch.int16, torch.int32):
                col_dict[name] = tensor.to(torch.int64)
            elif dt == torch.int64:
                col_dict[name] = tensor
            elif dt in (torch.float16, torch.float64):
                col_dict[name] = tensor.to(torch.float32)
            elif dt == torch.float32:
                col_dict[name] = tensor
            elif dt == torch.bool:
                col_dict[name] = tensor
            else:
                # Unsupported/rare dtype: attempt a safe float32 cast
                try:
                    col_dict[name] = tensor.to(torch.float32)
                except Exception:
                    continue

        return DataFrame(col_dict)
    except ImportError:
        raise ImportError(
            "PyTorch-Frame is required for dataframe format. "
            "Install with: pip install pytorch-frame"
        ) from None


def _tensor_dict_to_torch_frame(tensor_dict: dict[str, torch.Tensor]):
    """Convert a plain dict of tensors to torch_frame.DataFrame efficiently.

    This mirrors _fits_table_to_torch_frame without the FitsTable construction.
    Only numeric/bool tensors are included; string/VLA columns are skipped.
    """
    try:
        from torch_frame import DataFrame
    except ImportError:  # pragma: no cover
        raise ImportError(
            "PyTorch-Frame is required for dataframe format. "
            "Install with: pip install pytorch-frame"
        ) from None

    col_dict: dict[str, torch.Tensor] = {}
    for name, tensor in tensor_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        dt = tensor.dtype
        if dt in (torch.int8, torch.int16, torch.int32):
            col_dict[name] = tensor.to(torch.int64)
        elif dt == torch.int64:
            col_dict[name] = tensor
        elif dt in (torch.float16, torch.float64):
            col_dict[name] = tensor.to(torch.float32)
        elif dt == torch.float32:
            col_dict[name] = tensor
        elif dt == torch.bool:
            col_dict[name] = tensor
        else:
            # Skip unsupported dtypes to avoid surprising casts
            continue
    return DataFrame(col_dict)


def _torch_frame_to_fits_table(df):
    """Convert a torch_frame.DataFrame to FitsTable (lossless for numeric columns).

    Placeholder semantic-type (stype) inference: maps integer->numerical, float->numerical.
    String/categorical columns are ignored until torch_frame integration matured.
    """
    try:
        import torch_frame  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch-Frame is required for conversion. Install with: pip install pytorch-frame"
        ) from None

    from .table import ColumnInfo, FitsTable

    data = {}
    metadata = {}
    for name in df:
        col = df[name]
        # torch_frame DataFrame columns are pandas Series wrapping numeric data
    # pandas optional; use duck-typing on column below
        # Accept torch Tensor directly or pandas Series of numeric values
        if isinstance(col, torch.Tensor):
            data[name] = col
            metadata[name] = ColumnInfo(name=name, dtype=col.dtype)
        elif "pandas" in type(col).__module__:
            values = col.to_numpy()
            # Convert numpy array to torch tensor with appropriate dtype
            if values.dtype.kind in ("i", "u"):
                t = torch.from_numpy(values.astype("int64"))
            elif values.dtype.kind == "f":
                t = torch.from_numpy(values.astype("float32"))
            else:
                # Skip unsupported dtypes for now
                continue
            data[name] = t
            metadata[name] = ColumnInfo(name=name, dtype=t.dtype)
    return FitsTable(data, metadata)


def dataframe_round_trip(filename: str, hdu: int = 1):
    """Read table HDU into torch_frame DataFrame then back to FitsTable for parity testing.

    Returns (df, fits_table_roundtrip)
    """
    tbl = read(filename, hdu=hdu, format="table")
    df = _fits_table_to_torch_frame(tbl)
    back = _torch_frame_to_fits_table(df)
    return df, back


# --- Enhanced torch-frame integration (schema-preserving) ---
def fits_table_to_torch_frame(fits_table):
    """Convert FitsTable -> (torch_frame.DataFrame, metadata dict).

    Metadata dict maps column name -> {unit, description, dtype(str)} for schema round-trip.
    """
    try:
        from torch_frame import DataFrame
    except ImportError:  # pragma: no cover
        raise ImportError(
            "PyTorch-Frame required. Install with: pip install pytorch-frame"
        ) from None
        raise ImportError(
            "PyTorch-Frame required. Install with: pip install pytorch-frame"
        ) from None

    col_dict = {}
    meta = {}
    for name, tensor in fits_table.data.items():
        if hasattr(tensor, "dtype"):
            if tensor.dtype.is_floating_point:
                col_dict[name] = tensor.float()
            elif tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                # Keep track of original dtype; cast to int64 for torch_frame but store orig
                col_dict[name] = tensor.to(torch.int64)
            else:
                col_dict[name] = tensor
        else:
            # Skip non-tensor columns (e.g., raw string lists) for now
            continue
        ci = fits_table.column_info.get(name)
        meta[name] = {
            "unit": getattr(ci, "unit", None),
            "description": getattr(ci, "description", None),
            "dtype": str(tensor.dtype),  # original dtype
        }
    return DataFrame(col_dict), meta


def torch_frame_to_fits(
    filename: str,
    df,
    metadata: dict,
    header: dict | None = None,
    overwrite: bool = False,
):
    """Write a torch_frame.DataFrame back to FITS preserving units from metadata.

    Parameters
    ----------
    filename : str
        Output FITS path
    df : torch_frame.DataFrame
    metadata : dict
        Column metadata mapping produced by fits_table_to_torch_frame
    header : dict, optional
        Extra header keywords
    overwrite : bool
        Overwrite existing file
    """
    try:
        import torch_frame  # noqa: F401
    except ImportError:  # pragma: no cover
        raise ImportError(
            "PyTorch-Frame required. Install with: pip install pytorch-frame"
        ) from None
    from . import write_table

    data = {}
    units = []
    descriptions = []
    ordered_cols = list(df.keys())
    for name in ordered_cols:
        col = df[name]
        # Normalize column to tensor (torch_frame uses pandas Series currently)
        if isinstance(col, torch.Tensor):
            t = col
        elif "pandas" in type(col).__module__:
            arr = col.to_numpy()
            if arr.dtype.kind in ("i", "u"):
                t = torch.from_numpy(arr.astype("int64"))
            elif arr.dtype.kind == "f":
                t = torch.from_numpy(arr.astype("float32"))
            else:
                # Skip unsupported dtype
                continue
        else:
            continue
        # Restore original dtype if present in metadata
        orig_dtype_str = metadata.get(name, {}).get("dtype")
        if orig_dtype_str:
            try:
                if "int32" in orig_dtype_str:
                    t = t.to(torch.int32)
                elif "int16" in orig_dtype_str:
                    t = t.to(torch.int16)
                elif "int8" in orig_dtype_str:
                    t = t.to(torch.int8)
            except Exception:
                pass
        data[name] = t
        col_meta = metadata.get(name, {})
        units.append(col_meta.get("unit") or "")
        descriptions.append(col_meta.get("description") or "")
    write_table(
        filename,
        data,
        header=header or {},
        column_units=units,
        column_descriptions=descriptions,
        overwrite=overwrite,
    )


def torch_frame_round_trip_file(
    src_filename: str, dst_filename: str, hdu: int = 1, overwrite: bool = True
):
    """Full file round-trip: FITS -> FitsTable -> torch_frame.DataFrame -> FITS -> FitsTable.

    Returns tuple (orig_table, df, new_table).
    """
    tbl = read(src_filename, hdu=hdu, format="table", return_metadata=True)
    df, meta = fits_table_to_torch_frame(tbl)
    torch_frame_to_fits(
        dst_filename, df, meta, header={"EXTNAME": "RT"}, overwrite=overwrite
    )
    new_tbl = read(dst_filename, hdu=1, format="table", return_metadata=True)
    return tbl, df, new_tbl
