"""
Enhanced table operations for torchfits.

This module provides pandas-like operations on FITS tables using pure PyTorch,
without requiring external dependencies. Includes rich metadata support for
scientific data analysis.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch


class ColumnInfo:
    """
    Rich metadata information for a table column.

    Stores scientific metadata commonly found in FITS tables including
    units, descriptions, data types, and constraints.
    """

    def __init__(
        self,
        name: str,
        dtype: torch.dtype | None = None,
        *,
        unit: str | None = None,
        description: str | None = None,
        null_value: Any | None = None,
        display_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Column name
        dtype : torch.dtype, optional
            PyTorch dtype of the column
        unit : str, optional
            Physical unit
        description : str, optional
            Human-readable description
        null_value : Any, optional
            Value representing missing/null data
        display_format : str, optional
            Preferred display format
        **kwargs
            Additional FITS header metadata
        """
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.description = description
        self.null_value = null_value
        self.display_format = display_format
        self.fits_metadata = kwargs

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'", f"dtype={self.dtype}"]
        if self.unit:
            parts.append(f"unit='{self.unit}'")
        if self.description:
            parts.append(f"description='{self.description[:30]}...'")
        return f"ColumnInfo({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "dtype": str(self.dtype),
            "unit": self.unit,
            "description": self.description,
            "null_value": self.null_value,
            "display_format": self.display_format,
        }
        result.update(self.fits_metadata)
        return result

    @classmethod
    def from_fits_header(
    cls, colname: str, header_dict: dict[str, Any], dtype: torch.dtype
    ) -> ColumnInfo:
        """
        Create ColumnInfo from FITS header information.

        Parameters:
        -----------
        colname : str
            Column name
        header_dict : Dict[str, Any]
            FITS header keywords for this column
        dtype : torch.dtype
            PyTorch data type

        Returns:
        --------
        ColumnInfo
            Column metadata object
        """
        # Extract common FITS keywords
        unit = header_dict.get("TUNIT", None)
        description = header_dict.get("TCOMM", header_dict.get("TTYPE", None))
        null_value = header_dict.get("TNULL", None)
        display_format = header_dict.get("TDISP", None)

        # Remove processed keys from additional metadata
        fits_metadata = {
            k: v
            for k, v in header_dict.items()
            if k not in ["TUNIT", "TCOMM", "TTYPE", "TNULL", "TDISP"]
        }

        return cls(
            name=colname,
            dtype=dtype,
            unit=unit,
            description=description,
            null_value=null_value,
            display_format=display_format,
            **fits_metadata,
        )


class FitsTable:
    """
    A pandas-like interface for FITS table data using pure PyTorch tensors.

    Provides familiar operations like filtering, sorting, grouping, and selection
    while maintaining PyTorch tensor compatibility for GPU acceleration.
    Enhanced with rich metadata support for scientific data analysis.
    """

    def __init__(
    self,
    data_dict: dict[str, Any],
    metadata: dict | dict[str, ColumnInfo] | None = None,
    ):
        """
        Initialize FitsTable with tensor data and optional metadata.

        Parameters:
        -----------
        data_dict : Dict[str, Any]
            Dictionary mapping column names to column data. Values can be torch.Tensor for
            fixed-size columns or list/tuple (e.g., list[Tensor] for VLA, list[str] for strings).
        metadata : Dict or Dict[str, ColumnInfo], optional
            Column metadata - can be simple dict or ColumnInfo objects
        """
        if not data_dict:
            raise ValueError("FitsTable requires non-empty data dictionary")

        # Validate all tensors have same length
        lengths = {}
        for name, value in data_dict.items():
            if hasattr(value, "shape"):
                # This is a tensor
                lengths[name] = value.shape[0]
            elif isinstance(value, list | tuple):
                # This is a list/tuple (e.g., string columns)
                lengths[name] = len(value)
            else:
                # Single value or other type - assume length 1
                lengths[name] = 1

        if len(set(lengths.values())) > 1:
            raise ValueError(f"All columns must have same length. Got: {lengths}")

        self.data = data_dict
        self.columns = list(data_dict.keys())

        # Get the length from any tensor or list
        first_value = next(iter(data_dict.values()))
        if hasattr(first_value, "shape"):
            self._length = first_value.shape[0]
        elif isinstance(first_value, list | tuple):
            self._length = len(first_value)
        else:
            self._length = 1

        # Process metadata
        self.column_info = {}
        if metadata:
            for col_name, col_meta in metadata.items():
                if isinstance(col_meta, ColumnInfo):
                    self.column_info[col_name] = col_meta
                elif isinstance(col_meta, dict):
                    # Convert dict to ColumnInfo
                    if col_name in data_dict:
                        value = data_dict[col_name]
                        if hasattr(value, "dtype"):
                            dtype = value.dtype
                        else:
                            dtype = None  # For non-tensor data like string lists
                    else:
                        dtype = torch.float32
                    self.column_info[col_name] = ColumnInfo(
                        name=col_name,
                        dtype=dtype,
                        unit=col_meta.get("unit"),
                        description=col_meta.get("description"),
                        **{
                            k: v
                            for k, v in col_meta.items()
                            if k not in ["unit", "description"]
                        },
                    )

        # Create ColumnInfo for columns without metadata
        for col_name in self.columns:
            if col_name not in self.column_info:
                value = data_dict[col_name]
                if hasattr(value, "dtype"):
                    dtype = value.dtype
                else:
                    dtype = None  # For non-tensor data like string lists
                self.column_info[col_name] = ColumnInfo(name=col_name, dtype=dtype)
        # Lazy cache for derived null masks
        self._cached_null_masks = None

    @property
    def metadata(self) -> dict[str, ColumnInfo]:
        """Access column metadata (backward compatibility)."""
        return self.column_info

    def __repr__(self) -> str:
        """Return a compact string showing shape and columns."""
        return f"FitsTable(shape={self.shape}, columns={self.columns[:5]}{'...' if len(self.columns) > 5 else ''})"

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        return int(self._length)

    def __getitem__(self, key) -> Any | torch.Tensor | FitsTable:
        """
        Access data by column name, row slice, or boolean mask.

        Parameters:
        -----------
        key : str, slice, torch.Tensor, or tuple
            - str: Column name, returns tensor
            - slice: Row slice, returns FitsTable
            - torch.Tensor: Boolean mask, returns filtered FitsTable
            - tuple: (rows, columns) selection
        """
        if isinstance(key, str):
            # Column access
            if key not in self.data:
                raise KeyError(f"Column '{key}' not found. Available: {self.columns}")
            # Column may be a tensor or list; type: ignore for list return case.
            return self.data[key]  # type: ignore[return-value]

        elif isinstance(key, int):
            # Single row access - return a single-row FitsTable for consistency
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(
                    f"Row index {key} out of range for table with {len(self)} rows"
                )
            # Slice row to preserve column types (tensor or list element)
            return self._slice_rows(slice(key, key + 1))

        elif isinstance(key, slice):
            # Row slicing
            return self._slice_rows(key)

        elif isinstance(key, torch.Tensor):
            # Boolean indexing
            if key.dtype != torch.bool:
                raise TypeError("Tensor indexing requires boolean tensor")
            if key.shape[0] != self._length:
                raise ValueError(
                    f"Boolean mask length {key.shape[0]} doesn't match table length {self._length}"
                )
            return self.filter(key)

        elif isinstance(key, tuple) and len(key) == 2:
            # (rows, columns) selection
            rows, cols = key
            if isinstance(cols, str):
                cols = [cols]
            elif isinstance(cols, slice):
                cols = self.columns[cols]

            # First select columns, then rows
            selected_data = {col: self.data[col] for col in cols if col in self.data}
            temp_table = FitsTable(selected_data, self.column_info)

            if isinstance(rows, slice):
                return temp_table._slice_rows(rows)
            elif isinstance(rows, torch.Tensor) and rows.dtype == torch.bool:
                return temp_table.filter(rows)
            else:
                raise TypeError("Row selection must be slice or boolean tensor")

        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def _slice_rows(self, row_slice: slice) -> FitsTable:
        """Create new FitsTable with sliced rows."""
        sliced_data = {name: tensor[row_slice] for name, tensor in self.data.items()}
        return FitsTable(sliced_data, self.column_info)

    def to(self, device: str | torch.device) -> FitsTable:
        """
        Move all columns to specified device.

        Parameters:
        -----------
        device : str or torch.device
            Target device ('cpu', 'cuda', etc.)

        Returns:
        --------
        FitsTable
            New FitsTable with tensors on target device
        """
        device_data = {name: tensor.to(device) for name, tensor in self.data.items()}
        return FitsTable(device_data, self.column_info)

    def select(self, columns: str | list[str]) -> FitsTable:
        """
        Select specific columns.

        Parameters:
        -----------
        columns : str or List[str]
            Column name(s) to select

        Returns:
        --------
        FitsTable
            New FitsTable with selected columns
        """
        if isinstance(columns, str):
            columns = [columns]

        missing = [col for col in columns if col not in self.data]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        selected_data = {col: self.data[col] for col in columns}
        selected_metadata = {
            col: self.column_info.get(col) for col in columns if col in self.column_info
        }
        return FitsTable(selected_data, selected_metadata)

    def filter(self, mask: torch.Tensor) -> FitsTable:
        """
        Filter rows using boolean mask.

        Parameters:
        -----------
        mask : torch.Tensor
            Boolean tensor for row selection

        Returns:
        --------
        FitsTable
            New FitsTable with filtered rows
        """
        if mask.dtype != torch.bool:
            raise TypeError("Filter mask must be boolean tensor")
        if mask.shape[0] != self._length:
            raise ValueError(
                f"Mask length {mask.shape[0]} doesn't match table length {self._length}"
            )

        # Apply mask to tensors and lists (VLA columns as list[Tensor])
        filtered_data = {}
        # Create python list of indices where mask is True
        true_idx = [i for i, v in enumerate(mask.tolist()) if bool(v)]
        for name, value in self.data.items():
            if hasattr(value, "__getitem__") and hasattr(value, "dtype"):
                # Torch tensor path
                filtered_data[name] = value[mask]
            elif isinstance(value, list | tuple):
                filtered_data[name] = [value[i] for i in true_idx]
            else:
                # Scalar or unknown type, replicate selection semantics
                filtered_data[name] = value
        return FitsTable(filtered_data, self.column_info)

    def sort(self, column: str, descending: bool = False) -> FitsTable:
        """
        Sort table by column values.

        Parameters:
        -----------
        column : str
            Column name to sort by
        descending : bool, default False
            Sort in descending order

        Returns:
        --------
        FitsTable
            New FitsTable with sorted rows
        """
        if column not in self.data:
            raise KeyError(f"Column '{column}' not found")

        sort_indices = torch.argsort(self.data[column], descending=descending)
        idx_list = sort_indices.tolist()
        sorted_data = {}
        for name, value in self.data.items():
            if hasattr(value, "__getitem__") and hasattr(value, "dtype"):
                sorted_data[name] = value[sort_indices]
            elif isinstance(value, list | tuple):
                sorted_data[name] = [value[i] for i in idx_list]
            else:
                sorted_data[name] = value
        return FitsTable(sorted_data, self.column_info)

    def groupby(self, column: str) -> GroupedFitsTable:
        """
        Group table by column values.

        Parameters:
        -----------
        column : str
            Column name to group by

        Returns:
        --------
        GroupedFitsTable
            Grouped table for aggregation operations
        """
        if column not in self.data:
            raise KeyError(f"Column '{column}' not found")

        return GroupedFitsTable(self, column)

    def query(self, condition: str) -> FitsTable:
        """
        Filter table using query string (simple implementation).

        Parameters:
        -----------
        condition : str
            Query condition like 'MAG_G < 20.0'

        Returns:
        --------
        FitsTable
            Filtered table

        Note:
        -----
        This is a simple implementation. For complex queries,
        use direct tensor operations.
        """
        # Simple parser for basic conditions
        # Format: "COLUMN OPERATOR VALUE"
        parts = condition.strip().split()
        if len(parts) != 3:
            raise ValueError("Query must be in format 'COLUMN OPERATOR VALUE'")

        column, operator, value_str = parts

        if column not in self.data:
            raise KeyError(f"Column '{column}' not found")

        # Try to convert value to appropriate type
        rhs: Any
        try:
            if "." in value_str:
                rhs = float(value_str)
            else:
                rhs = int(value_str)
        except ValueError:
            rhs = value_str  # Keep as string

        col_data = self.data[column]

        # Convert value to tensor for comparison
        if isinstance(rhs, int | float):
            value_tensor = torch.tensor(rhs, dtype=col_data.dtype)
        else:
            # For string comparisons, this is more complex - simplified here
            raise ValueError("String comparisons not yet implemented in query()")

        # Apply operator
        if operator == "<":
            mask = col_data < value_tensor
        elif operator == "<=":
            mask = col_data <= value_tensor
        elif operator == ">":
            mask = col_data > value_tensor
        elif operator == ">=":
            mask = col_data >= value_tensor
        elif operator == "==":
            mask = col_data == value_tensor
        elif operator == "!=":
            mask = col_data != value_tensor
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return self.filter(mask)

    @property
    def shape(self) -> tuple[int, int]:
        """Table shape as (num_rows, num_columns)."""
        return (self._length, len(self.columns))

    @property
    def dtypes(self) -> dict[str, torch.dtype | None]:
        """Dictionary of column data types."""
        out: dict[str, torch.dtype | None] = {}
        for name, value in self.data.items():
            dtype = getattr(value, "dtype", None)
            if dtype is None and isinstance(value, list | tuple) and value:
                # Infer from first element (for VLA columns as list[Tensor])
                first = value[0]
                dtype = getattr(first, "dtype", None)
            # dtype can be None for non-tensor types (e.g., list of strings)
            out[name] = dtype
        return out

    # --- Null mask helpers ---
    def get_null_masks(
        self, header: dict[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        """Build boolean null masks per column.

        Prefers ColumnInfo.null_value when available; if not and a FITS header is provided,
        derives masks from TNULLn header keywords.

        Returns a dict mapping column name -> bool tensor (True where value is null).
        """
        masks: dict[str, torch.Tensor] = {}
        # ColumnInfo-based masks
        for name, info in self.column_info.items():
            if info is None or getattr(info, "null_value", None) is None:
                continue
            val = info.null_value
            col = self.data.get(name)
            if isinstance(col, torch.Tensor) and col.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                try:
                    masks[name] = col.eq(int(val))
                except Exception:
                    continue
        # Header-derived masks (fills gaps where ColumnInfo not present)
        if header:
            try:
                # Local import to avoid circular import at module import time
                from .fits_reader import _build_null_masks as _build_masks

                hdr_masks = _build_masks(self.data, header)
                for k, m in hdr_masks.items():
                    masks.setdefault(k, m)
            except Exception:
                pass
        return masks

    @property
    def null_masks(self) -> dict[str, torch.Tensor]:
        """Lazily compute and cache null masks using ColumnInfo null_value if present.

        Note: Does not consider FITS header. For header-aware masks, call get_null_masks(header).
        """
        if self._cached_null_masks is None:
            self._cached_null_masks = self.get_null_masks()
        return self._cached_null_masks

    def with_applied_null_masks(
        self,
        masks: dict[str, torch.Tensor] | None = None,
        *,
        fill_value: float | dict[str, float] | str = "nan",
        float_dtype: torch.dtype = torch.float32,
    ) -> FitsTable:
        """Return a new FitsTable with nulls applied according to masks.

        Parameters
        ----------
        masks : dict[str, torch.Tensor], optional
            Precomputed boolean masks per column. If None, will derive from ColumnInfo/null_value.
        fill_value : float | dict | 'nan'
            Replacement value for nulls. If 'nan', integer columns will be cast to float_dtype and filled with NaN.
            If a dict, map column name -> fill value.
        float_dtype : torch.dtype
            Float dtype to cast integer columns to when using 'nan' fill.
        """
        # Build masks if not provided
        if masks is None:
            masks = self.null_masks
        if not masks:
            return FitsTable(dict(self.data), dict(self.column_info))

        def _col_fill(name: str) -> float | None:
            if isinstance(fill_value, dict):
                if name in fill_value:
                    return float(fill_value[name])
                return None
            if isinstance(fill_value, int | float):
                return float(fill_value)
            return None

        new_data: dict[str, Any] = {}
        new_meta = dict(self.column_info)
        for name, value in self.data.items():
            mask = masks.get(name)
            if mask is None or not isinstance(value, torch.Tensor):
                new_data[name] = value
                continue
            if mask.dtype != torch.bool:
                new_data[name] = value
                continue
            # Apply replacement
            if isinstance(fill_value, str) and fill_value.lower() == "nan":
                if value.dtype.is_floating_point:
                    col = value.clone()
                    col[mask] = torch.nan
                    new_data[name] = col
                elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    col = value.to(float_dtype)
                    col[mask] = torch.nan
                    new_data[name] = col
                    # Update metadata dtype
                    if (
                        name in new_meta
                        and getattr(new_meta[name], "dtype", None) is not None
                    ):
                        new_meta[name] = ColumnInfo(
                            name=name,
                            dtype=float_dtype,
                            unit=new_meta[name].unit,
                            description=new_meta[name].description,
                            null_value=new_meta[name].null_value,
                            display_format=new_meta[name].display_format,
                            **new_meta[name].fits_metadata,
                        )
                else:
                    # Unsupported dtype for NaN; leave unchanged
                    new_data[name] = value
            else:
                fv = _col_fill(name)
                if fv is None:
                    new_data[name] = value
                else:
                    # Ensure dtype compatibility
                    col = value.clone()
                    if not col.dtype.is_floating_point and isinstance(fv, float):
                        col = col.to(float_dtype)
                        if (
                            name in new_meta
                            and getattr(new_meta[name], "dtype", None) is not None
                        ):
                            new_meta[name] = ColumnInfo(
                                name=name,
                                dtype=float_dtype,
                                unit=new_meta[name].unit,
                                description=new_meta[name].description,
                                null_value=new_meta[name].null_value,
                                display_format=new_meta[name].display_format,
                                **new_meta[name].fits_metadata,
                            )
                    # fv guaranteed float here
                    if col.dtype.is_floating_point:
                        col[mask] = float(fv)
                    else:
                        col[mask] = int(fv)
                    new_data[name] = col
        return FitsTable(new_data, new_meta)

    def head(self, n: int = 5) -> FitsTable:
        """Return first n rows."""
        result = self[:n]
        if isinstance(result, FitsTable):
            return result
        else:
            raise TypeError("head() should return FitsTable")

    def tail(self, n: int = 5) -> FitsTable:
        """Return last n rows."""
        result = self[-n:]
        if isinstance(result, FitsTable):
            return result
        else:
            raise TypeError("tail() should return FitsTable")

    def info(self) -> None:
        """Print table information with rich metadata."""
        print(f"FitsTable with {self.shape[0]} rows and {self.shape[1]} columns")
        print("\nColumns:")
        print(f"{'#':>2} {'Name':<15} {'Type':<15} {'Unit':<10} {'Description':<30}")
        print("-" * 75)

        for i, name in enumerate(self.columns):
            tensor = self.data[name]
            col_info = self.column_info.get(name)

            if col_info:
                unit = col_info.unit or ""
                desc = col_info.description or ""
                # Truncate long descriptions
                if len(desc) > 30:
                    desc = desc[:27] + "..."
            else:
                unit = ""
                desc = ""

            print(f"{i:>2} {name:<15} {str(tensor.dtype):<15} {unit:<10} {desc:<30}")

    def get_column_info(self, column: str) -> ColumnInfo:
        """
        Get detailed information about a column.

        Parameters:
        -----------
        column : str
            Column name

        Returns:
        --------
        ColumnInfo
            Column metadata object
        """
        if column not in self.column_info:
            raise KeyError(f"Column '{column}' not found")
        return self.column_info[column]

    def get_units(self) -> dict[str, str | None]:
        """Get units for all columns."""
        return {name: info.unit for name, info in self.column_info.items()}

    def get_descriptions(self) -> dict[str, str | None]:
        """Get descriptions for all columns."""
        return {name: info.description for name, info in self.column_info.items()}

    def describe(self) -> dict[str, dict[str, float]]:
        """Compute basic statistics for numeric columns."""
        stats = {}
        for name, tensor in self.data.items():
            if tensor.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                stats[name] = {
                    "count": tensor.numel(),
                    "mean": tensor.float().mean().item(),
                    "std": tensor.float().std().item(),
                    "min": tensor.min().item(),
                    "max": tensor.max().item(),
                }
        return stats

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary of tensors (for backward compatibility)."""
        return self.data.copy()

    def copy(self) -> FitsTable:
        """Create a deep copy of the table."""
        copied_data = {name: tensor.clone() for name, tensor in self.data.items()}
        copied_metadata = {name: info for name, info in self.column_info.items()}
        return FitsTable(copied_data, copied_metadata)

    def convert_units(self, column: str, target_unit: str) -> FitsTable:
        """
        Convert column units (basic implementation).

        Parameters:
        -----------
        column : str
            Column name to convert
        target_unit : str
            Target unit

        Returns:
        --------
        FitsTable
            New table with converted units

        Note:
        -----
        Currently supports basic angular conversions: deg/rad/arcsec/arcmin
        """
        if column not in self.data:
            raise KeyError(f"Column '{column}' not found")

        col_info = self.column_info.get(column)
        if not col_info or not col_info.unit:
            warnings.warn(
                f"No unit information for column '{column}'", stacklevel=2
            )
            return self.copy()

        current_unit = col_info.unit.lower()
        target_unit = target_unit.lower()

        if current_unit == target_unit:
            return self.copy()

        # Basic angular unit conversions
        conversion_factors = {
            ("deg", "rad"): torch.pi / 180.0,
            ("rad", "deg"): 180.0 / torch.pi,
            ("deg", "arcsec"): 3600.0,
            ("arcsec", "deg"): 1.0 / 3600.0,
            ("deg", "arcmin"): 60.0,
            ("arcmin", "deg"): 1.0 / 60.0,
            ("arcsec", "arcmin"): 1.0 / 60.0,
            ("arcmin", "arcsec"): 60.0,
        }

        factor = conversion_factors.get((current_unit, target_unit))
        if factor is None:
            raise ValueError(
                f"Conversion from '{current_unit}' to '{target_unit}' not supported"
            )

        # Create new table with converted values
        new_data = self.data.copy()
        converted_tensor = self.data[column] * factor
        new_data[column] = converted_tensor

        new_metadata = self.column_info.copy()
        if col_info:
            new_col_info = ColumnInfo(
                name=col_info.name,
                dtype=converted_tensor.dtype,  # Use actual converted dtype
                unit=target_unit,
                description=col_info.description,
                null_value=col_info.null_value,
                display_format=col_info.display_format,
                **col_info.fits_metadata,
            )
            new_metadata[column] = new_col_info

        return FitsTable(new_data, new_metadata)

    def correlate(self, col1: str, col2: str) -> float:
        """
        Compute correlation coefficient between two columns.

        Parameters:
        -----------
        col1, col2 : str
            Column names

        Returns:
        --------
        float
            Pearson correlation coefficient
        """
        if col1 not in self.data or col2 not in self.data:
            raise KeyError("One or both columns not found")

        x = self.data[col1].float()
        y = self.data[col2].float()

        # Remove any NaN values
        valid_mask = ~(torch.isnan(x) | torch.isnan(y))
        if not valid_mask.any():
            return float("nan")

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # Compute correlation
        x_centered = x_valid - x_valid.mean()
        y_centered = y_valid - y_valid.mean()

        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())

        if denominator == 0:
            return float("nan")

        return float((numerator / denominator).item())

    def percentile(self, column: str, q: float) -> float:
        """
        Compute percentile of a column.

        Parameters:
        -----------
        column : str
            Column name
        q : float
            Percentile (0-100)

        Returns:
        --------
        float
            Percentile value
        """
        if column not in self.data:
            raise KeyError(f"Column '{column}' not found")

        data = self.data[column].float()
        valid_data = data[~torch.isnan(data)]

        if len(valid_data) == 0:
            return float("nan")

        return torch.quantile(valid_data, q / 100.0).item()

    def join(self, other: FitsTable, on: str, how: str = "inner") -> FitsTable:
        """
        Join with another FitsTable on a key column.

        Supports 'inner' and 'left' joins for numeric key columns.
        """
        if on not in self.data or on not in other.data:
            raise KeyError(f"Join column '{on}' not found in both tables")

        left_values = self.data[on]
        right_values = other.data[on]

        left_indices: list[int] = []
        right_indices: list[int] = []
        for i, left_val in enumerate(left_values):
            matches = torch.where(right_values == left_val)[0]
            if matches.numel() > 0:
                for m in matches.tolist():
                    left_indices.append(i)
                    right_indices.append(int(m))
            elif how == "left":
                left_indices.append(i)
                right_indices.append(-1)

        if not left_indices and how == "inner":
            return FitsTable({on: torch.tensor([], dtype=left_values.dtype)})

        left_idx_t = torch.tensor(left_indices, dtype=torch.long)
        right_idx_filtered = [idx for idx in right_indices if idx >= 0]
        if len(right_idx_filtered) == 0:
            right_idx_t = torch.tensor([], dtype=torch.long)
        else:
            right_idx_t = torch.tensor(right_idx_filtered, dtype=torch.long)

        result_data: dict[str, torch.Tensor] = {}
        result_metadata: dict[str, ColumnInfo] = {}

        for col_name, col_data in self.data.items():
            result_data[col_name] = col_data[left_idx_t]
            if col_name in self.column_info:
                result_metadata[col_name] = self.column_info[col_name]

        for col_name, col_data in other.data.items():
            if col_name == on:
                continue
            out_name = col_name if col_name not in result_data else f"{col_name}_right"
            result_data[out_name] = col_data[right_idx_t]
            if col_name in other.column_info:
                result_metadata[out_name] = other.column_info[col_name]

        return FitsTable(result_data, result_metadata)


def apply_null_masks_to_dict(
    data: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    *,
    fill_value: float | int | dict[str, float] = float("nan"),
    float_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Apply null masks to a plain dict of tensors and return a new dict.

    If fill_value is NaN (default), integer tensors are cast to float_dtype and masked positions set to NaN.
    If fill_value is numeric or a per-column dict, masked positions are set to that value.
    """
    out: dict[str, torch.Tensor] = {}
    for name, t in data.items():
        m = masks.get(name)
        if m is None or not isinstance(t, torch.Tensor) or m.dtype != torch.bool:
            out[name] = t
            continue
        # Check for NaN fill
        if isinstance(fill_value, (int, float)) and (  # noqa: UP038 - runtime isinstance requires a tuple
            float(fill_value) != float(fill_value)
        ):
            if t.dtype.is_floating_point:
                col = t.clone()
                col[m] = torch.nan
                out[name] = col
            elif t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                col = t.to(float_dtype)
                col[m] = torch.nan
                out[name] = col
            else:
                out[name] = t
        else:
            if isinstance(fill_value, dict):
                if name not in fill_value:
                    out[name] = t
                    continue
                raw_fv = fill_value[name]
            else:
                raw_fv = fill_value
            # Normalize and cast dtype if needed
            fv_num = float(raw_fv)
            col = t.clone()
            # If a float fill is provided, cast to float dtype even if integer-valued like 0.0
            if not col.dtype.is_floating_point and isinstance(raw_fv, float):
                col = col.to(float_dtype)
            if col.dtype.is_floating_point:
                col[m] = float(fv_num)
            else:
                col[m] = int(fv_num)
            out[name] = col
    return out


class GroupedFitsTable:
    """
    Grouped FitsTable for aggregation operations.
    """

    def __init__(self, table: FitsTable, group_column: str):
        self.table = table
        self.group_column = group_column
        self._group_indices = None
        self._compute_groups()

    def _compute_groups(self):
        """Compute group indices for aggregation."""
        group_values = self.table.data[self.group_column]
        unique_values, inverse_indices = torch.unique(group_values, return_inverse=True)

        self.unique_values = unique_values
        self.inverse_indices = inverse_indices

        # Create mapping from group value to row indices
        self.group_indices = {}
        for i, value in enumerate(unique_values):
            mask = inverse_indices == i
            self.group_indices[value.item()] = torch.where(mask)[0]

    def agg(self, operations: dict[str, str | list[str]]) -> FitsTable:
        """
        Aggregate grouped data.

        Parameters:
        -----------
        operations : Dict[str, str or List[str]]
            Dictionary mapping column names to aggregation operations
            Operations: 'mean', 'sum', 'count', 'min', 'max', 'std'

        Returns:
        --------
        FitsTable
            Aggregated results
        """
        result_data = {self.group_column: self.unique_values}

        for column, ops in operations.items():
            if isinstance(ops, str):
                ops = [ops]

            if column not in self.table.data:
                raise KeyError(f"Column '{column}' not found")

            col_data = self.table.data[column]

            for op in ops:
                result_name = f"{column}_{op}" if len(ops) > 1 else column
                result_values = []

                for group_value in self.unique_values:
                    indices = self.group_indices[group_value.item()]
                    group_data = col_data[indices]

                    if op == "mean":
                        result_values.append(group_data.float().mean())
                    elif op == "sum":
                        result_values.append(group_data.sum())
                    elif op == "count":
                        result_values.append(torch.tensor(len(group_data)))
                    elif op == "min":
                        result_values.append(group_data.min())
                    elif op == "max":
                        result_values.append(group_data.max())
                    elif op == "std":
                        result_values.append(group_data.float().std())
                    else:
                        raise ValueError(f"Unsupported operation: {op}")

                result_data[result_name] = torch.stack(result_values)

        return FitsTable(result_data)

    def mean(self) -> FitsTable:
        """Compute mean for all numeric columns."""
        numeric_columns: dict[str, str | list[str]] = {}
        for name, tensor in self.table.data.items():
            if name != self.group_column and tensor.dtype in [
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
            ]:
                numeric_columns[name] = "mean"
        return self.agg(numeric_columns)

    def sum(self) -> FitsTable:
        """Compute sum for all numeric columns."""
        numeric_columns: dict[str, str | list[str]] = {}
        for name, tensor in self.table.data.items():
            if name != self.group_column and tensor.dtype in [
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
            ]:
                numeric_columns[name] = "sum"
        return self.agg(numeric_columns)

    def count(self) -> FitsTable:
        """Count rows in each group."""
        # Use first non-group column for counting
        first_col = next(
            name for name in self.table.columns if name != self.group_column
        )
        return self.agg({first_col: "count"})


def pad_ragged(
    sequences: list[torch.Tensor],
    pad_value: float = 0.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1D tensors (ragged) into a 2D tensor [rows, max_len].

    Returns (padded, lengths) where lengths is a 1D int64 tensor of original lengths.

    Useful for variable-length array (VLA) FITS columns returned as list[Tensor].
    """
    if not sequences:
        dt = dtype or torch.float32
        return torch.empty((0, 0), dtype=dt), torch.empty((0,), dtype=torch.int64)
    max_len = max(int(s.numel()) for s in sequences)
    dt = dtype or sequences[0].dtype
    rows = len(sequences)
    padded = torch.full((rows, max_len), pad_value, dtype=dt)
    lengths = torch.empty((rows,), dtype=torch.int64)
    for i, s in enumerate(sequences):
        n = int(s.numel())
        lengths[i] = n
        if n > 0:
            padded[i, :n] = s.to(dt)
    return padded, lengths
