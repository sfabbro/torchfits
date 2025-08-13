"""
FITS writing functionality for TorchFits v1.0

This module provides comprehensive writing capabilities for FITS files,
including images, tables, and multi-extension files.
"""

from typing import Any

import torch

try:
    from . import (
        fits_reader_cpp,  # type: ignore  # The C++ module includes both reading and writing
    )

    fits_reader_cpp = fits_reader_cpp  # type: ignore[assignment]
    _WRITER_AVAILABLE = True
except ImportError:
    _WRITER_AVAILABLE = False
    fits_reader_cpp = None  # type: ignore[assignment]

# Help static analyzers only under TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import fits_reader_cpp as fits_reader_cpp  # type: ignore


def _to_compression_config(cfg_like):
    """Convert a user compression spec (dict or CompressionConfig) into a CompressionConfig instance.

    Accepted forms:
    - None or {"type": "none"}: no compression
    - {"type": "GZIP" | "RICE" | "HCOMPRESS" | "PLIO", "quantize_level": int, ...}
    - Existing CompressionConfig instance
    Returns a tuple (compression_cfg, use_advanced) where use_advanced indicates whether to call advanced writer.
    """
    if not _WRITER_AVAILABLE:
        return None, False
    if cfg_like is None:
        return None, False
    # Alias and assert C++ module for static analyzers
    fr = fits_reader_cpp
    assert fr is not None
    # If already a CompressionConfig, just use it (unless type None)
    if hasattr(fr, "CompressionConfig") and isinstance(cfg_like, fr.CompressionConfig):
        if getattr(cfg_like, "type", None) == getattr(fr.CompressionType, "None", None):
            return cfg_like, False
        return cfg_like, True
    if isinstance(cfg_like, dict):
        if "type" not in cfg_like:
            raise ValueError("compression dict must include 'type'")
        c = fr.CompressionConfig()
        comp_type_name = str(cfg_like.get("type")).upper()
        # Map string to enum
        if not hasattr(fr, "CompressionType"):
            raise RuntimeError("CompressionType enum not exposed in C++ module")
        if not hasattr(fr.CompressionType, comp_type_name.capitalize()) and not hasattr(
            fr.CompressionType, comp_type_name
        ):
            raise ValueError(f"Unknown compression type: {cfg_like.get('type')}")
        # Support either exact case in enum or capitalized
        enum_attr = (
            comp_type_name
            if hasattr(fr.CompressionType, comp_type_name)
            else comp_type_name.capitalize()
        )
        c.type = getattr(fr.CompressionType, enum_attr)
        if "quantize_level" in cfg_like:
            c.quantize_level = int(cfg_like["quantize_level"])
        if "quantize_dither" in cfg_like and hasattr(c, "quantize_dither"):
            c.quantize_dither = int(cfg_like["quantize_dither"])
        if "preserve_zeros" in cfg_like and hasattr(c, "preserve_zeros"):
            c.preserve_zeros = bool(cfg_like["preserve_zeros"])
        if (
            hasattr(c, "tile_dimensions") and "tile_dimensions" in cfg_like
        ):  # future extension
            # Placeholder if tile_dimensions added to binding later
            pass
        if c.type == getattr(fr.CompressionType, "None", None):
            return c, False
        return c, True
    # Fallback: treat as disabled
    return None, False


def write(
    filename: str,
    data: torch.Tensor | dict | Any,
    header: dict[str, str] | None = None,
    overwrite: bool = False,
    append: bool = False,
    hdu_type: str = "auto",
    compression: dict | Any | None = None,
    checksum: bool = False,
    **kwargs,
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
        Whether to overwrite existing file (mutually exclusive with append). Default: False
    append : bool, optional
        If True and file exists, append data as a new HDU (images only for now). Ignored if overwriting.
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
            raise ValueError(
                f"Cannot auto-detect HDU type for data of type {type(data)}"
            )

    # Basic mutual exclusion
    if overwrite and append:
        raise ValueError("'overwrite' and 'append' cannot both be True")

    if append and not overwrite:
        import os

        if os.path.exists(filename):
            if hdu_type == "image":
                if not isinstance(data, torch.Tensor):
                    raise ValueError("Append image requires torch.Tensor data")
                if not _WRITER_AVAILABLE:
                    raise RuntimeError("C++ writing functionality not available")
                assert fits_reader_cpp is not None
                fits_reader_cpp.append_hdu_to_fits(
                    filename, data, header or {}, kwargs.get("extname", "")
                )
                return
            elif hdu_type == "table":
                if not _WRITER_AVAILABLE:
                    raise RuntimeError("C++ writing functionality not available")
                # Normalize to a plain dict plus metadata lists for the C++ appender
                tbl_dict = None
                column_units = kwargs.get("column_units")
                column_descriptions = kwargs.get("column_descriptions")
                null_sentinels = kwargs.get("null_sentinels") or {}

                if isinstance(data, dict):
                    tbl_dict = data
                    # Ensure lists exist even if not provided
                    if column_units is None:
                        column_units = [""] * len(tbl_dict)
                    if column_descriptions is None:
                        column_descriptions = [""] * len(tbl_dict)
                elif (
                    hasattr(data, "__class__")
                    and data.__class__.__name__ == "FitsTable"
                ):
                    # Extract data and metadata from FitsTable
                    tbl_dict = data.data  # type: ignore[attr-defined]
                    if not isinstance(tbl_dict, dict):
                        raise ValueError("FitsTable.data must be a dict of columns")
                    # Build units/descriptions aligned with dict iteration order
                    if column_units is None:
                        column_units = []
                    else:
                        column_units = list(column_units)
                    if column_descriptions is None:
                        column_descriptions = []
                    else:
                        column_descriptions = list(column_descriptions)
                    if not column_units or not column_descriptions:
                        column_units = []
                        column_descriptions = []
                        colinfo = getattr(data, "column_info", {})  # type: ignore[attr-defined]
                        if not isinstance(colinfo, dict):
                            colinfo = {}
                        for name in tbl_dict:
                            meta = colinfo.get(name)
                            unit = (
                                getattr(meta, "unit", None)
                                if meta is not None
                                else None
                            )
                            desc = (
                                getattr(meta, "description", None)
                                if meta is not None
                                else None
                            )
                            column_units.append(unit or "")
                            column_descriptions.append(desc or "")
                    # Populate null_sentinels from ColumnInfo.null_value if not explicitly given
                    if not null_sentinels:
                        colinfo = getattr(data, "column_info", {})  # type: ignore[attr-defined]
                        if isinstance(colinfo, dict):
                            for name, meta in colinfo.items():
                                nv = getattr(meta, "null_value", None)
                                if nv is not None:
                                    try:
                                        null_sentinels[name] = int(nv)
                                    except Exception:
                                        pass
                else:
                    raise ValueError("Table append requires dict or FitsTable data")

                assert fits_reader_cpp is not None
                fits_reader_cpp.append_table_to_fits(
                    filename,
                    tbl_dict,
                    header or {},
                    column_units,
                    column_descriptions,
                    null_sentinels,
                )
                return
            else:
                raise NotImplementedError(
                    f"Appending HDU type '{hdu_type}' is not supported."
                )

    if hdu_type == "image":
        if not isinstance(data, torch.Tensor):
            raise ValueError("Image HDU requires torch.Tensor data")
        if not _WRITER_AVAILABLE:
            raise RuntimeError("C++ writing functionality not available")
        comp_cfg, use_adv = _to_compression_config(compression)
        if use_adv and hasattr(fits_reader_cpp, "write_tensor_to_fits_advanced"):
            assert fits_reader_cpp is not None
            fits_reader_cpp.write_tensor_to_fits_advanced(
                filename, data, header, comp_cfg, overwrite, checksum
            )
        else:
            assert fits_reader_cpp is not None
            fits_reader_cpp.write_tensor_to_fits(filename, data, header, overwrite)

    elif hdu_type == "table":
        if not _WRITER_AVAILABLE:
            raise RuntimeError("C++ writing functionality not available")

        if isinstance(data, dict):
            column_units = kwargs.get("column_units", [])
            column_descriptions = kwargs.get("column_descriptions", [])
            null_sentinels = kwargs.get("null_sentinels", {})
            assert fits_reader_cpp is not None
            fits_reader_cpp.write_table_to_fits(
                filename,
                data,
                header,
                column_units,
                column_descriptions,
                null_sentinels,
                overwrite,
            )
        elif hasattr(data, "__class__") and data.__class__.__name__ == "FitsTable":
            assert fits_reader_cpp is not None
            fits_reader_cpp.write_fits_table(filename, data, overwrite)
        else:
            raise ValueError("Table HDU requires dict or FitsTable data")
    else:
        raise ValueError(f"Unknown HDU type: {hdu_type}")


def write_mef(
    filename: str,
    data_list: list[torch.Tensor],
    headers: list[dict[str, str]] | None = None,
    extnames: list[str] | None = None,
    overwrite: bool = False,
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

    if not data_list:
        raise ValueError("data_list must be non-empty")
    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")
    # Heuristic: if no extnames provided, create a dummy primary so user hdu=1 maps to first tensor
    # (dataset tests rely on this), otherwise use standard primary=first tensor ordering.
    if not extnames:
        dummy = torch.zeros(
            1,
            dtype=(
                data_list[0].dtype if hasattr(data_list[0], "dtype") else torch.float32
            ),
        )
        # Prepend dummy and align headers/extnames lengths
        adj_headers: list[dict[str, str]] = (
            ([{}] + list(headers)) if headers else ([{}])
        )
        adj_extnames: list[str] = ["PRIMARY"] + [""] * len(data_list)
        assert fits_reader_cpp is not None
        fits_reader_cpp.write_tensors_to_mef(
            filename, [dummy] + data_list, adj_headers, adj_extnames, overwrite
        )
    else:
        assert fits_reader_cpp is not None
        fits_reader_cpp.write_tensors_to_mef(
            filename, data_list, headers, extnames, overwrite
        )


def append_hdu(
    filename: str,
    data: torch.Tensor,
    header: dict[str, str] | None = None,
    extname: str = "",
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

    assert fits_reader_cpp is not None
    fits_reader_cpp.append_hdu_to_fits(filename, data, header, extname)


def _resolve_hdu_spec(filename: str, hdu: int | str) -> int:
    """Resolve an HDU spec (int or EXTNAME) to an integer HDU number (1-based)."""
    if isinstance(hdu, int):
        return hdu
    if not isinstance(hdu, str):
        raise TypeError("hdu must be int or str")
    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ functionality not available for HDU name resolution")
    # Use get_hdu_type binding to probe names by iterating; we lack direct name->num helper exposed.
    # We will brute-force up to a reasonable number of HDUs (e.g., 256) until failure.
    # TODO: expose get_hdu_num_by_name in bindings for efficiency.
    try:
        total = fits_reader_cpp.get_num_hdus(filename)  # type: ignore[union-attr]
    except Exception:
        total = 32  # fallback bound
    name_upper = hdu.upper()
    # We can read headers to find EXTNAME
    for i in range(1, total + 1):
        try:
            assert fits_reader_cpp is not None
            header = fits_reader_cpp.get_header(filename, i)
            extname = header.get("EXTNAME") or header.get("EXTNAME".lower())
            if isinstance(extname, str) and extname.upper() == name_upper:
                return i
        except Exception:
            break
    raise ValueError(f"HDU name '{hdu}' not found in {filename}")


def update_header(
    filename: str, updates: dict[str, str], hdu: int | str = 1
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
    """
    if isinstance(hdu, str):
        hdu = _resolve_hdu_spec(filename, hdu)

    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")

    assert fits_reader_cpp is not None
    fits_reader_cpp.update_fits_header(filename, hdu, updates)
    try:
        fits_reader_cpp._clear_cache()
    except Exception:
        pass


def update_data(
    filename: str,
    new_data: torch.Tensor,
    hdu: int | str = 1,
    start: list[int] | None = None,
    shape: list[int] | None = None,
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
        hdu = _resolve_hdu_spec(filename, hdu)

    if start is None:
        start = []
    if shape is None:
        shape = []

    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")

    assert fits_reader_cpp is not None
    fits_reader_cpp.update_fits_data(filename, hdu, new_data, start, shape)
    # Invalidate cache so reads return modified data
    try:
        fits_reader_cpp._clear_cache()
    except Exception:
        pass


# Convenience functions for specific use cases


def write_image(
    filename: str,
    image: torch.Tensor,
    header: dict[str, str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write a tensor as a FITS image file.

    This is a convenience function equivalent to:
    write(filename, image, header, overwrite, hdu_type="image")
    """
    write(filename, image, header, overwrite, hdu_type="image")


def write_table(
    filename: str,
    table_data: dict | Any,  # Any to avoid FitsTable type error
    header: dict[str, str] | None = None,
    column_units: list[str] | None = None,
    column_descriptions: list[str] | None = None,
    null_sentinels: dict[str, int] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write table data as a FITS binary table.

    This is a convenience function for writing tables with optional metadata.
    """
    write(
        filename,
        table_data,
        header,
        overwrite,
        hdu_type="table",
        column_units=column_units or [],
        column_descriptions=column_descriptions or [],
        null_sentinels=null_sentinels or {},
    )


def write_cube(
    filename: str,
    cube: torch.Tensor,
    header: dict[str, str] | None = None,
    overwrite: bool = False,
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


def write_variable_length_array(
    filename: str,
    arrays: list[torch.Tensor],
    header: dict[str, str] | None = None,
    overwrite: bool = False,
) -> None:
    """Write a list of 1D tensors as a variable-length array table.

    Each tensor becomes a row in a single-column binary table (column name: ARRAY_DATA).
    Currently supports floating point tensors (will be converted to float64 by backend if needed).
    """
    if header is None:
        header = {}
    if not _WRITER_AVAILABLE:
        raise RuntimeError("C++ writing functionality not available")
    if not isinstance(arrays, list | tuple) or not arrays:
        raise ValueError("arrays must be a non-empty list of tensors")
    for t in arrays:
        if not isinstance(t, torch.Tensor):
            raise ValueError("All elements must be tensors")
        if t.dim() != 1:
            raise ValueError("Only 1D tensors supported for variable-length arrays")
    # Delegate to C++ backend
    assert fits_reader_cpp is not None
    fits_reader_cpp.write_variable_length_array(
        filename, list(arrays), header, overwrite
    )


# Integration with FitsTable class
def _register_fits_table_writer():
    """Register write method with FitsTable class if available."""
    try:
        from .table import FitsTable

        def write_to_fits(self, filename: str, overwrite: bool = False):
            """Write this FitsTable to a FITS file."""
            if not _WRITER_AVAILABLE:
                raise RuntimeError("C++ writing functionality not available")
            assert fits_reader_cpp is not None
            fits_reader_cpp.write_fits_table(filename, self, overwrite)

        # Try to add method to FitsTable class - may fail due to type checker
        # Assign method dynamically; safe in runtime, silence linter warning about setattr on class
        try:
            setattr(FitsTable, "write", write_to_fits)  # noqa: B010
        except Exception:
            pass

    except ImportError:
        pass  # FitsTable not available


# Register the writer method when module is imported
_register_fits_table_writer()
