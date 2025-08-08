"""
Enhanced table operations for torchfits.

This module provides pandas-like operations on FITS tables using pure PyTorch,
without requiring external dependencies. Includes rich metadata support for
scientific data analysis.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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
        dtype: Optional[torch.dtype] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
        null_value: Optional[Any] = None,
        display_format: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize column metadata.

        Parameters:
        -----------
        name : str
            Column name
        dtype : torch.dtype
            PyTorch data type
        unit : str, optional
            Physical unit (e.g., 'mag', 'deg', 'arcsec')
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

    def to_dict(self) -> Dict[str, Any]:
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
        cls, colname: str, header_dict: Dict[str, Any], dtype: torch.dtype
    ) -> "ColumnInfo":
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
        data_dict: Dict[str, torch.Tensor],
        metadata: Optional[Union[Dict, Dict[str, ColumnInfo]]] = None,
    ):
        """
        Initialize FitsTable with tensor data and optional metadata.

        Parameters:
        -----------
        data_dict : Dict[str, torch.Tensor]
            Dictionary mapping column names to PyTorch tensors
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
            elif isinstance(value, (list, tuple)):
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
        elif isinstance(first_value, (list, tuple)):
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

    @property
    def metadata(self) -> Dict[str, ColumnInfo]:
        """Access to column metadata (backward compatibility)."""
        return self.column_info

    def __repr__(self) -> str:
        """String representation showing shape and columns."""
        return f"FitsTable(shape={self.shape}, columns={self.columns[:5]}{'...' if len(self.columns) > 5 else ''})"

    def __len__(self) -> int:
        """Number of rows in the table."""
        return self._length

    def __getitem__(self, key) -> Union[torch.Tensor, "FitsTable"]:
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
            return self.data[key]

        elif isinstance(key, int):
            # Single row access - return dict of column values at that row
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Row index {key} out of range for table with {len(self)} rows")
            return {col: self.data[col][key] for col in self.data}

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

    def _slice_rows(self, row_slice: slice) -> "FitsTable":
        """Create new FitsTable with sliced rows."""
        sliced_data = {name: tensor[row_slice] for name, tensor in self.data.items()}
        return FitsTable(sliced_data, self.column_info)

    def to(self, device: Union[str, torch.device]) -> "FitsTable":
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

    def select(self, columns: Union[str, List[str]]) -> "FitsTable":
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

    def filter(self, mask: torch.Tensor) -> "FitsTable":
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

        filtered_data = {name: tensor[mask] for name, tensor in self.data.items()}
        return FitsTable(filtered_data, self.column_info)

    def sort(self, column: str, descending: bool = False) -> "FitsTable":
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
        sorted_data = {name: tensor[sort_indices] for name, tensor in self.data.items()}
        return FitsTable(sorted_data, self.column_info)

    def groupby(self, column: str) -> "GroupedFitsTable":
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

    def query(self, condition: str) -> "FitsTable":
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
        try:
            if "." in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str  # Keep as string

        col_data = self.data[column]

        # Convert value to tensor for comparison
        if isinstance(value, (int, float)):
            value_tensor = torch.tensor(value, dtype=col_data.dtype)
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
    def shape(self) -> Tuple[int, int]:
        """Table shape as (num_rows, num_columns)."""
        return (self._length, len(self.columns))

    @property
    def dtypes(self) -> Dict[str, torch.dtype]:
        """Dictionary of column data types."""
        return {name: tensor.dtype for name, tensor in self.data.items()}

    def head(self, n: int = 5) -> "FitsTable":
        """Return first n rows."""
        result = self[:n]
        if isinstance(result, FitsTable):
            return result
        else:
            raise TypeError("head() should return FitsTable")

    def tail(self, n: int = 5) -> "FitsTable":
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

    def get_units(self) -> Dict[str, Optional[str]]:
        """Get units for all columns."""
        return {name: info.unit for name, info in self.column_info.items()}

    def get_descriptions(self) -> Dict[str, Optional[str]]:
        """Get descriptions for all columns."""
        return {name: info.description for name, info in self.column_info.items()}

    def describe(self) -> Dict[str, Dict[str, float]]:
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

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary of tensors (for backward compatibility)."""
        return self.data.copy()

    def copy(self) -> "FitsTable":
        """Create a deep copy of the table."""
        copied_data = {name: tensor.clone() for name, tensor in self.data.items()}
        copied_metadata = {name: info for name, info in self.column_info.items()}
        return FitsTable(copied_data, copied_metadata)

    def convert_units(self, column: str, target_unit: str) -> "FitsTable":
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
            warnings.warn(f"No unit information for column '{column}'")
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

        return (numerator / denominator).item()

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

    def join(self, other: "FitsTable", on: str, how: str = "inner") -> "FitsTable":
        """
        Join with another FitsTable.

        Parameters:
        -----------
        other : FitsTable
            Other table to join with
        on : str
            Column name to join on
        how : str, default 'inner'
            Join type: 'inner', 'left'

        Returns:
        --------
        FitsTable
            Joined table

        Note:
        -----
        Simple implementation for common join patterns
        """
        if on not in self.data or on not in other.data:
            raise KeyError(f"Join column '{on}' not found in both tables")

        # Get unique values and create index mappings
        left_values = self.data[on]
        right_values = other.data[on]

        # Find matching indices
        left_indices = []
        right_indices = []

        for i, left_val in enumerate(left_values):
            matches = torch.where(right_values == left_val)[0]
            if len(matches) > 0:
                for match_idx in matches:
                    left_indices.append(i)
                    right_indices.append(match_idx.item())
            elif how == "left":
                left_indices.append(i)
                right_indices.append(-1)  # Will handle this below

        if not left_indices:
            # No matches found
            if how == "inner":
                # Return empty table
                return FitsTable({on: torch.tensor([])})

        left_indices = torch.tensor(left_indices)
        right_indices = torch.tensor([idx for idx in right_indices if idx >= 0])

        # Build result data
        result_data = {}
        result_metadata = {}

        # Add left table columns
        for col_name, col_data in self.data.items():
            result_data[col_name] = col_data[left_indices]
            if col_name in self.column_info:
                result_metadata[col_name] = self.column_info[col_name]

        # Add right table columns (avoiding duplicates)
        for col_name, col_data in other.data.items():
            if col_name != on:  # Don't duplicate join column
                # Handle name conflicts
                result_col_name = col_name
                if col_name in result_data:
                    result_col_name = f"{col_name}_right"

                if how == "left" and -1 in [idx for idx in right_indices if idx >= 0]:
                    # Handle left join with missing values
                    # This is a simplified implementation
                    result_data[result_col_name] = col_data[
                        right_indices[right_indices >= 0]
                    ]
                else:
                    result_data[result_col_name] = col_data[right_indices]

                if col_name in other.column_info:
                    result_metadata[result_col_name] = other.column_info[col_name]

        return FitsTable(result_data, result_metadata)


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

    def agg(self, operations: Dict[str, Union[str, List[str]]]) -> FitsTable:
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
        numeric_columns = {}
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
        numeric_columns = {}
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
