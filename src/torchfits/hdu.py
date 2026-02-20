"""
Core HDU classes for torchfits.

This module implements the main data structures for FITS HDUs:
- HDUList: Container for multiple HDUs
- TensorHDU: Image/cube data with lazy loading
- TableHDU: Tabular data with torch-frame integration
- Header: FITS header management
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    from torch_frame import TensorFrame

    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False

    class TensorFrame:
        """Minimal fallback when torch_frame is unavailable."""

        def __init__(self, feat_dict=None, col_names_dict=None):
            self.feat_dict = feat_dict or {}
            self.col_names_dict = col_names_dict or {}


# Import torch first
_ = torch.empty(1)  # Force torch C++ symbols to load
# import torchfits.cpp as cpp  <-- Removed to avoid circular import


class Header(dict):
    """FITS header as dict."""

    def __init__(self, cards=None):
        super().__init__()
        self._version = 0
        self._cards = []  # List of (key, value, comment)
        if cards:
            if isinstance(cards, dict):
                # Legacy support or if passed a dict
                for k, v in cards.items():
                    self[k] = v
                    self._cards.append((k, v, ""))
            elif isinstance(cards, list):
                # List of tuples (key, value, comment)
                for card in cards:
                    if len(card) == 3:
                        k, v, c = card
                    elif len(card) == 2:
                        k, v = card
                        c = ""
                    else:
                        continue

                    self._cards.append((k, v, c))

                    # Handle special keys
                    if k == "HISTORY" or k == "COMMENT":
                        # For dict access, we might want to append?
                        # Standard dict behavior overwrites.
                        # We keep dict behavior for compatibility, but _cards has everything.
                        # Maybe store as list in dict? No, that breaks expectation of string value.
                        # Just store the last one in dict, or join them?
                        # Astropy stores them in a special way.
                        # For now, we just let dict overwrite, so last one wins.
                        # But we provide methods to access all.
                        pass
                    else:
                        self[k] = v

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._version += 1

    def __delitem__(self, key):
        super().__delitem__(key)
        self._version += 1

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._version += 1

    def clear(self):
        super().clear()
        self._version += 1

    def pop(self, *args):
        res = super().pop(*args)
        self._version += 1
        return res

    def popitem(self):
        res = super().popitem()
        self._version += 1
        return res

    def setdefault(self, key, default=None):
        res = super().setdefault(key, default)
        self._version += 1
        return res

    def add_history(self, value):
        self._cards.append(("HISTORY", value, ""))
        # Update dict?

    def add_comment(self, value):
        self._cards.append(("COMMENT", value, ""))

    def get_history(self):
        return [c[1] for c in self._cards if c[0] == "HISTORY"]

    def get_comment(self):
        return [c[1] for c in self._cards if c[0] == "COMMENT"]


class DataView:
    """Lazy data accessor."""

    def __init__(self, file_handle, hdu_index: int):
        self._handle = file_handle
        self._index = hdu_index

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._handle.get_shape(self._index))

    @property
    def dtype(self) -> torch.dtype:
        # Map BITPIX to torch dtype
        bitpix = self._handle.get_dtype(self._index)
        if bitpix == 8:
            return torch.uint8
        elif bitpix == 16:
            return torch.int16
        elif bitpix == 32:
            return torch.int32
        elif bitpix == -32:
            return torch.float32
        elif bitpix == -64:
            return torch.float64
        return torch.float32  # Default

    def __getitem__(self, slice_spec) -> Tensor:
        # Parse slice_spec to x1, y1, x2, y2 (exclusive bounds)
        shape = self.shape
        if len(shape) < 2:
            raise ValueError("Subset reading requires at least 2D data")

        if slice_spec is Ellipsis:
            slice_spec = (slice(None), slice(None))
        elif not isinstance(slice_spec, tuple):
            slice_spec = (slice_spec, slice(None))

        if len(slice_spec) != 2:
            raise ValueError("Subset slicing supports exactly 2 dimensions (y, x)")

        def _normalize_index(s, dim):
            if isinstance(s, int):
                idx = s + dim if s < 0 else s
                return max(0, min(dim, idx)), max(0, min(dim, idx + 1))
            if isinstance(s, slice):
                if s.step not in (None, 1):
                    raise ValueError("Only step=1 slices are supported")
                start = 0 if s.start is None else s.start
                stop = dim if s.stop is None else s.stop
                if start < 0:
                    start += dim
                if stop < 0:
                    stop += dim
                start = max(0, min(dim, start))
                stop = max(0, min(dim, stop))
                return start, stop
            raise TypeError("Slice spec must be int or slice")

        y1, y2 = _normalize_index(slice_spec[0], shape[0])
        x1, x2 = _normalize_index(slice_spec[1], shape[1])

        return self._handle.read_subset(self._index, x1, y1, x2, y2)


class TensorHDU:
    """Represents image, cube, or array data with lazy loading."""

    def __init__(
        self,
        data: Optional[Tensor] = None,
        header: Optional[Header] = None,
        file_handle=None,
        hdu_index: int = 0,
    ):
        self._data = data
        self._header = header or Header()
        self._file_handle = file_handle
        self._hdu_index = hdu_index
        self._data_view = DataView(file_handle, hdu_index) if file_handle else None
        self._wcs_cache = None
        self._wcs_version = -1

    @property
    def data(self) -> DataView:
        if self._data_view is None:
            raise ValueError("No file handle available")
        return self._data_view

    @property
    def header(self) -> Header:
        return self._header

    @property
    def wcs(self):
        """WCS object for coordinate transformations."""
        current_version = getattr(self._header, "_version", None)
        if current_version is None:
            # Header does not support versioning (e.g. plain dict), disable caching
            from .wcs import WCS

            return WCS(self._header)

        if self._wcs_cache is None or self._wcs_version != current_version:
            from .wcs import WCS

            self._wcs_cache = WCS(self._header)
            self._wcs_version = current_version
        return self._wcs_cache

    def to_tensor(self, device: str = "cpu") -> Tensor:
        if self._data is not None:
            return self._data.to(device)

        elif self._file_handle is not None:
            import torchfits.cpp as cpp

            return cpp.read_full(self._file_handle, self._hdu_index).to(device)
        else:
            # Return dummy data if no data available
            return torch.zeros(10, 10).to(device)

    def chunks(self, chunk_size: Tuple[int, ...]) -> Iterator[Tensor]:
        import torchfits.cpp as cpp

        return cpp.iter_chunks(self._file_handle, self._hdu_index, chunk_size)

    def stats(self) -> Dict[str, float]:
        import torchfits.cpp as cpp

        return cpp.compute_stats(self._file_handle, self._hdu_index)

    def _get_shape_str(self) -> str:
        """Get shape string representation."""
        if self._data is not None:
            return str(tuple(self._data.shape))
        elif self._file_handle:
            # Try to get from header first to avoid C++ call if possible
            naxis = self.header.get("NAXIS", 0)
            if naxis == 0:
                return "()"
            # FITS stores dimensions in reverse order of C/Python usually?
            # But NAXIS1 is closest to contiguous in FITS (Fortran order).
            # PyTorch is C order. torchfits likely handles this.
            # Let's just list NAXISn values.
            dims = [str(self.header.get(f"NAXIS{i + 1}", 0)) for i in range(naxis)]
            # Reverse to match typical python shape (C-order) if strictly following numpy?
            # But FITS convention is usually (NAXIS1, NAXIS2) -> (x, y)
            # Python image is (y, x).
            # Let's trust what DataView would return if we could call it,
            # but for purely header based, (NAXIS2, NAXIS1) is a good guess for 2D.
            return f"({', '.join(reversed(dims))})"
        return "()"

    def _get_dtype_str(self) -> str:
        """Get dtype string representation."""
        if self._data is not None:
            return str(self._data.dtype).replace("torch.", "")
        elif self._file_handle:
            bitpix = self.header.get("BITPIX", 0)
            mapping = {
                8: "uint8",
                16: "int16",
                32: "int32",
                -32: "float32",
                -64: "float64",
            }
            return mapping.get(bitpix, str(bitpix))
        return "unknown"

    def __repr__(self):
        name = self.header.get("EXTNAME", "PRIMARY")
        return f"TensorHDU(name='{name}', shape={self._get_shape_str()}, dtype={self._get_dtype_str()})"


class TableDataAccessor:
    """Dictionary-like accessor for table data."""

    def __init__(self, table_hdu):
        self._table = table_hdu

    def __getitem__(self, key):
        """Get column data."""
        if hasattr(self._table, "_raw_data") and key in self._table._raw_data:
            value = self._table._raw_data[key]
            if isinstance(value, torch.Tensor):
                if value.dim() > 1:
                    return value.squeeze()
                return value
            return value
        if hasattr(self._table, "feat_dict") and key in self._table.feat_dict:
            tensor = self._table.feat_dict[key]
            if tensor.dim() > 1:
                return tensor.squeeze()
            return tensor
        raise KeyError(f"Column '{key}' not found")

    def __contains__(self, key):
        """Check if column exists."""
        return (hasattr(self._table, "_raw_data") and key in self._table._raw_data) or (
            hasattr(self._table, "feat_dict") and key in self._table.feat_dict
        )

    def keys(self):
        """Get column names."""
        if hasattr(self._table, "_raw_data"):
            return self._table._raw_data.keys()
        if hasattr(self._table, "feat_dict"):
            return self._table.feat_dict.keys()
        return []

    @property
    def columns(self):
        """Get column names as list."""
        return list(self.keys())

    def __len__(self):
        """Get number of rows."""
        return self._table.num_rows


class TableHDU(TensorFrame):
    """FITS table as TensorFrame."""

    def __init__(
        self,
        tensor_dict: dict,
        col_stats: dict = None,
        header: Optional[Header] = None,
        source_path: Optional[str] = None,
        source_hdu: Optional[int] = None,
    ):
        self._raw_data = tensor_dict or {}
        self._source_path = source_path
        self._source_hdu = source_hdu
        # Convert raw data to proper TensorFrame format
        feat_dict = {}
        col_names_dict = {}
        string_cols = self._get_string_columns(header)

        for col_name, data in tensor_dict.items():
            try:
                if col_name in string_cols:
                    # Keep strings in raw data only
                    continue
                if isinstance(data, torch.Tensor):
                    # Ensure tensor is 2D
                    if data.dim() == 1:
                        data = data.unsqueeze(1)  # Convert 1D to 2D
                    elif data.dim() == 0:
                        data = data.unsqueeze(0).unsqueeze(1)  # Convert scalar to 2D
                    feat_dict[col_name] = data
                    # Generate column names based on second dimension
                    col_names_dict[col_name] = [str(i) for i in range(data.shape[1])]
                elif isinstance(data, (list, tuple)):
                    # Convert list/tuple to tensor
                    try:
                        # Try to infer the dtype from the data
                        if data and isinstance(data[0], str):
                            # Keep strings in raw data only
                            continue
                        if data and isinstance(data[0], torch.Tensor):
                            # VLA-like list of tensors: keep in raw data only
                            continue
                        if data and isinstance(data[0], np.ndarray):
                            # VLA-like list of arrays: keep in raw data only
                            continue
                        if data and isinstance(data[0], (int, np.integer)):
                            tensor_data = torch.tensor(data, dtype=torch.long)
                        else:
                            tensor_data = torch.tensor(data, dtype=torch.float32)

                        if tensor_data.dim() == 1:
                            tensor_data = tensor_data.unsqueeze(1)
                        feat_dict[col_name] = tensor_data
                        col_names_dict[col_name] = [
                            str(i) for i in range(tensor_data.shape[1])
                        ]
                    except Exception:
                        # Skip problematic columns
                        continue
                elif isinstance(data, dict):
                    # Skip dict data for now - this might be complex FITS data
                    continue
                elif isinstance(data, np.ndarray):
                    # Skip complex/object/string arrays for TensorFrame
                    if np.iscomplexobj(data) or data.dtype.kind in {"U", "S", "O"}:
                        continue
                    tensor_data = torch.tensor(data)
                    if tensor_data.dim() == 1:
                        tensor_data = tensor_data.unsqueeze(1)
                    feat_dict[col_name] = tensor_data
                    col_names_dict[col_name] = [
                        str(i) for i in range(tensor_data.shape[1])
                    ]
                else:
                    # Convert other data types to tensor
                    try:
                        if isinstance(data, str):
                            # Skip string data
                            continue
                        tensor_data = torch.tensor(
                            [data] if not hasattr(data, "__len__") else data,
                            dtype=torch.float32,
                        )
                        if tensor_data.dim() == 1:
                            tensor_data = tensor_data.unsqueeze(1)
                        feat_dict[col_name] = tensor_data
                        col_names_dict[col_name] = [
                            str(i) for i in range(tensor_data.shape[1])
                        ]
                    except Exception:
                        continue
            except Exception:
                # Skip any problematic columns
                continue

        # If no valid columns, create empty structure with correct row count
        if not feat_dict:
            nrows = 1
            if header and "NAXIS2" in header:
                try:
                    nrows = int(header["NAXIS2"])
                except Exception:
                    nrows = 1
            feat_dict = {"dummy": torch.zeros(max(1, nrows), 1)}
            col_names_dict = {"dummy": ["0"]}

        self.header = header or Header()
        super().__init__(feat_dict, col_names_dict)

    def _get_string_columns(self, header: Optional[Header]) -> set:
        """Infer string columns from TTYPE/TFORM header cards."""
        if not header:
            return set()
        name_by_idx = {}
        string_cols = set()
        for key, value in header.items():
            if isinstance(key, str) and key.startswith("TTYPE"):
                idx = key[5:]
                name_by_idx[str(idx)] = value
        for key, value in header.items():
            if isinstance(key, str) and key.startswith("TFORM"):
                idx = key[5:]
                tform = str(value)
                if "A" in tform:
                    col_name = name_by_idx.get(str(idx))
                    if col_name:
                        string_cols.add(col_name)
        return string_cols

    @property
    def string_columns(self) -> List[str]:
        """Return column names inferred as strings."""
        return sorted(self._get_string_columns(self.header))

    @property
    def schema(self) -> Dict[str, Any]:
        """Return basic schema info parsed from FITS headers."""
        return self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        if not self.header:
            return {"columns": [], "string_columns": [], "vla_columns": []}

        name_by_idx: Dict[int, str] = {}
        tform_by_idx: Dict[int, str] = {}
        tdim_by_idx: Dict[int, str] = {}
        for key, value in self.header.items():
            if isinstance(key, str) and key.startswith("TTYPE"):
                idx = int(key[5:]) if key[5:].isdigit() else None
                if idx is not None:
                    name_by_idx[idx] = str(value)
            if isinstance(key, str) and key.startswith("TFORM"):
                idx = int(key[5:]) if key[5:].isdigit() else None
                if idx is not None:
                    tform_by_idx[idx] = str(value)
            if isinstance(key, str) and key.startswith("TDIM"):
                idx = int(key[4:]) if key[4:].isdigit() else None
                if idx is not None:
                    tdim_by_idx[idx] = str(value)

        def _parse_tform(tform: str) -> Dict[str, Any]:
            # Examples: "E", "20A", "1PB", "1PJ"
            import re

            tform = tform.strip().upper()
            m = re.match(r"(\d*)([PQ]?)([A-Z])", tform)
            if not m:
                return {"tform": tform, "repeat": None, "vla": False, "code": None}
            repeat = int(m.group(1)) if m.group(1) else 1
            vla = m.group(2) in ("P", "Q")
            code = m.group(3)
            base = {"tform": tform, "repeat": repeat, "vla": vla, "code": code}
            if vla:
                base["vla_descriptor"] = m.group(2)
            return base

        columns = []
        string_cols = []
        vla_cols = []
        for idx in sorted(name_by_idx.keys()):
            name = name_by_idx[idx]
            tform = tform_by_idx.get(idx, "")
            info = (
                _parse_tform(tform)
                if tform
                else {"tform": "", "repeat": None, "vla": False, "code": None}
            )
            tdim = tdim_by_idx.get(idx)
            is_string = info["code"] == "A"
            if is_string:
                string_cols.append(name)
            if info["vla"]:
                vla_cols.append(name)
            columns.append(
                {
                    "name": name,
                    "tform": info["tform"],
                    "repeat": info["repeat"],
                    "code": info["code"],
                    "string": is_string,
                    "vla": info["vla"],
                    "tdim": tdim,
                }
            )

        return {
            "columns": columns,
            "string_columns": string_cols,
            "vla_columns": vla_cols,
        }

    def get_vla_column(self, name: str) -> List[Tensor]:
        """Return a VLA column as a list of tensors."""
        value = self._raw_data.get(name)
        if isinstance(value, list):
            return value

        raise KeyError(f"Column '{name}' is not a VLA list")

    def get_vla_lengths(self, name: str) -> List[int]:
        """Return per-row lengths for a VLA column."""
        values = self.get_vla_column(name)
        lengths: List[int] = []
        for item in values:
            if isinstance(item, torch.Tensor):
                lengths.append(int(item.numel()))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
            else:
                lengths.append(1)
        return lengths

    @property
    def vla_lengths(self) -> Dict[str, List[int]]:
        """Return per-row lengths for all VLA columns."""
        out: Dict[str, List[int]] = {}
        for col in self.schema.get("vla_columns", []):
            try:
                out[col] = self.get_vla_lengths(col)
            except Exception:
                continue
        return out

    def get_string_column(
        self, name: str, encoding: str = "ascii", strip: bool = True
    ) -> List[str]:
        """Decode a string column stored as uint8 bytes."""
        value = self._raw_data.get(name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Column '{name}' is not a tensor string column")
        if value.dtype != torch.uint8:
            raise TypeError(f"Column '{name}' is not uint8 encoded")
        if value.dim() != 2:
            raise ValueError(f"Column '{name}' must be 2D (rows, width)")

        strings: List[str] = []
        for row in value.cpu().numpy():
            s = bytes(row.tolist()).decode(encoding, errors="ignore")
            if strip:
                s = s.rstrip(" \x00")
            strings.append(s)
        return strings

    @property
    def num_rows(self) -> int:
        """Get number of rows in the table."""
        if hasattr(self, "_raw_data") and self._raw_data:
            for value in self._raw_data.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0] if value.dim() > 0 else 1
                if isinstance(value, (list, tuple)):
                    return len(value)
            return 0
        if self.header and "NAXIS2" in self.header:
            try:
                return int(self.header["NAXIS2"])
            except Exception:
                pass
        if hasattr(self, "feat_dict") and self.feat_dict:
            first_tensor = next(iter(self.feat_dict.values()))
            return first_tensor.shape[0] if hasattr(first_tensor, "shape") else 0
        return 0

    @property
    def data(self):
        """Access table data like a dictionary."""
        return TableDataAccessor(self)

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        if hasattr(self, "_raw_data"):
            return [str(k) for k in self._raw_data.keys()]
        if hasattr(self, "feat_dict"):
            return [str(k) for k in self.feat_dict.keys()]
        return []

    @property
    def col_names(self) -> List[str]:
        """Get column names."""
        return self.columns

    @property
    def feat_types(self) -> Dict[str, str]:
        """Get feature types."""
        types = {}
        if hasattr(self, "feat_dict"):
            for name, tensor_data in self.feat_dict.items():
                # For TensorFrame, tensor_data might be nested
                if hasattr(tensor_data, "dtype"):
                    if tensor_data.dtype.is_floating_point:
                        types[str(name)] = "numerical"
                    else:
                        types[str(name)] = "categorical"
                else:
                    types[str(name)] = "categorical"
        return types

    def select(self, cols: List[str]) -> "TableHDU":
        """Select specific columns."""
        if hasattr(self, "_raw_data"):
            selected_dict = {k: v for k, v in self._raw_data.items() if str(k) in cols}
            return TableHDU(selected_dict, {}, self.header)
        if hasattr(self, "feat_dict"):
            selected_dict = {k: v for k, v in self.feat_dict.items() if str(k) in cols}
            return TableHDU(selected_dict, {}, self.header)
        return self

    def filter(self, condition: str) -> "TableHDU":
        """Filter rows by condition."""
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("condition must be a non-empty string")

        # Prefer raw table data to preserve strings/VLA columns.
        data_map = self._raw_data if hasattr(self, "_raw_data") else {}
        if not data_map and hasattr(self, "feat_dict"):
            data_map = self.feat_dict
        if not data_map:
            return self

        num_rows = self.num_rows
        if num_rows <= 0:
            return self

        eval_locals: Dict[str, Any] = {}
        for name, value in data_map.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.shape[0] == num_rows
            ):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                arr = t.numpy()
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                eval_locals[str(name)] = arr
            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_rows
            ):
                arr = value
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                eval_locals[str(name)] = arr
            elif isinstance(value, list) and len(value) == num_rows:
                eval_locals[str(name)] = np.asarray(value, dtype=object)

        if not eval_locals:
            raise ValueError("No row-aligned columns available for filtering")

        mask_result = None
        try:
            import numexpr as ne

            mask_result = ne.evaluate(condition, local_dict=eval_locals)
        except Exception:
            mask_result = eval(
                condition,
                {"__builtins__": {}, "np": np},
                eval_locals,
            )

        mask_arr = np.asarray(mask_result)
        if mask_arr.ndim == 0:
            mask = np.full(num_rows, bool(mask_arr.item()), dtype=bool)
        else:
            mask = mask_arr.astype(bool, copy=False).reshape(-1)
            if mask.shape[0] != num_rows:
                raise ValueError(
                    f"Filter produced mask of length {mask.shape[0]}, expected {num_rows}"
                )

        filtered: Dict[str, Any] = {}
        for name, value in data_map.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.shape[0] == num_rows
            ):
                mask_t = torch.from_numpy(mask)
                if value.device.type != "cpu":
                    mask_t = mask_t.to(value.device)
                filtered[name] = value[mask_t]
            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_rows
            ):
                filtered[name] = value[mask]
            elif isinstance(value, list) and len(value) == num_rows:
                filtered[name] = [item for item, keep in zip(value, mask) if keep]
            else:
                filtered[name] = value

        return TableHDU(
            filtered,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def head(self, n: int) -> "TableHDU":
        """Limit to first n rows."""
        if hasattr(self, "_raw_data") and self._raw_data:
            new_dict = {}
            for k, v in self._raw_data.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    new_dict[k] = v[:n]
                elif isinstance(v, list):
                    new_dict[k] = v[:n]
                else:
                    new_dict[k] = v
            return TableHDU(new_dict, {}, self.header)
        if hasattr(self, "feat_dict") and self.feat_dict:
            new_dict = {}
            for k, v in self.feat_dict.items():
                if hasattr(v, "shape") and len(v.shape) > 0:
                    new_dict[k] = v[:n]
                else:
                    new_dict[k] = v
            return TableHDU(new_dict, {}, self.header)
        return self

    @staticmethod
    def _value_num_rows(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.dim() > 0 else 1
        if isinstance(value, np.ndarray):
            return int(value.shape[0]) if value.ndim > 0 else 1
        if isinstance(value, (list, tuple)):
            return len(value)
        raise TypeError(f"Unsupported column type: {type(value)}")

    @staticmethod
    def _append_column_values(name: str, old_value: Any, new_value: Any) -> Any:
        if isinstance(old_value, torch.Tensor):
            if not isinstance(new_value, torch.Tensor):
                new_value = torch.as_tensor(
                    np.asarray(new_value), dtype=old_value.dtype
                )
            if old_value.device.type != new_value.device.type:
                new_value = new_value.to(old_value.device)

            if old_value.dim() == 0:
                old_value = old_value.reshape(1)
            if new_value.dim() == 0:
                new_value = new_value.reshape(1)
            if old_value.dim() == 2 and new_value.dim() == 1:
                new_value = new_value.unsqueeze(1)
            if (
                old_value.dim() == 1
                and new_value.dim() == 2
                and new_value.shape[1] == 1
            ):
                new_value = new_value.squeeze(1)
            if old_value.dim() != new_value.dim():
                raise ValueError(
                    f"Column '{name}' rank mismatch: {old_value.dim()} vs {new_value.dim()}"
                )
            if old_value.dim() > 1 and tuple(old_value.shape[1:]) != tuple(
                new_value.shape[1:]
            ):
                raise ValueError(
                    f"Column '{name}' shape mismatch: {tuple(old_value.shape[1:])} vs {tuple(new_value.shape[1:])}"
                )
            return torch.cat([old_value, new_value.to(dtype=old_value.dtype)], dim=0)

        if isinstance(old_value, np.ndarray):
            new_arr = np.asarray(new_value, dtype=old_value.dtype)
            old_arr = old_value
            if old_arr.ndim == 0:
                old_arr = old_arr.reshape(1)
            if new_arr.ndim == 0:
                new_arr = new_arr.reshape(1)
            if old_arr.ndim == 2 and new_arr.ndim == 1:
                new_arr = np.expand_dims(new_arr, axis=1)
            if old_arr.ndim == 1 and new_arr.ndim == 2 and new_arr.shape[1] == 1:
                new_arr = np.squeeze(new_arr, axis=1)
            if old_arr.ndim != new_arr.ndim:
                raise ValueError(
                    f"Column '{name}' rank mismatch: {old_arr.ndim} vs {new_arr.ndim}"
                )
            if old_arr.ndim > 1 and tuple(old_arr.shape[1:]) != tuple(
                new_arr.shape[1:]
            ):
                raise ValueError(
                    f"Column '{name}' shape mismatch: {tuple(old_arr.shape[1:])} vs {tuple(new_arr.shape[1:])}"
                )
            return np.concatenate([old_arr, new_arr], axis=0)

        if isinstance(old_value, list):
            if not isinstance(new_value, (list, tuple)):
                raise ValueError(
                    f"Column '{name}' expects a list/tuple for append; got {type(new_value)}"
                )
            return list(old_value) + list(new_value)

        raise TypeError(
            f"Unsupported column type for append in '{name}': {type(old_value)}"
        )

    def add_column(self, name: str, values: Any, overwrite: bool = False) -> "TableHDU":
        """Return a new table with one additional column."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if name in raw and not overwrite:
            raise KeyError(f"Column '{name}' already exists")

        nrows = self.num_rows
        new_rows = self._value_num_rows(values)
        if nrows > 0 and new_rows != nrows:
            raise ValueError(f"Column '{name}' has {new_rows} rows, expected {nrows}")
        raw[name] = values
        return TableHDU(
            raw,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def drop_columns(self, columns: List[str]) -> "TableHDU":
        """Return a new table with selected columns removed."""
        if not columns:
            return self
        to_drop = {str(c) for c in columns}
        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        missing = [name for name in to_drop if name not in raw]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        kept = {k: v for k, v in raw.items() if k not in to_drop}
        return TableHDU(
            kept,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def rename_column(self, old_name: str, new_name: str) -> "TableHDU":
        """Return a new table with one renamed column."""
        if not isinstance(old_name, str) or not old_name:
            raise ValueError("old_name must be a non-empty string")
        if not isinstance(new_name, str) or not new_name:
            raise ValueError("new_name must be a non-empty string")
        if old_name == new_name:
            return self

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if old_name not in raw:
            raise KeyError(f"Column '{old_name}' not found")
        if new_name in raw:
            raise KeyError(f"Column '{new_name}' already exists")

        renamed: Dict[str, Any] = {}
        for key, value in raw.items():
            renamed[new_name if key == old_name else key] = value
        return TableHDU(
            renamed,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def append_rows(self, rows: Dict[str, Any]) -> "TableHDU":
        """Return a new table with additional rows appended."""
        if not isinstance(rows, dict) or not rows:
            raise ValueError("rows must be a non-empty dictionary")

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if not raw:
            return TableHDU(
                dict(rows),
                {},
                self.header,
                source_path=self._source_path,
                source_hdu=self._source_hdu,
            )

        current_cols = set(raw.keys())
        incoming_cols = set(rows.keys())
        if incoming_cols != current_cols:
            missing = sorted(current_cols - incoming_cols)
            extra = sorted(incoming_cols - current_cols)
            raise ValueError(
                f"append_rows requires exactly matching columns; missing={missing}, extra={extra}"
            )

        appended: Dict[str, Any] = {}
        append_rows_count: Optional[int] = None
        for name in raw.keys():
            new_count = self._value_num_rows(rows[name])
            if append_rows_count is None:
                append_rows_count = new_count
            elif new_count != append_rows_count:
                raise ValueError(
                    f"All appended columns must have same row count; column '{name}' has {new_count}, expected {append_rows_count}"
                )
            appended[name] = self._append_column_values(name, raw[name], rows[name])

        return TableHDU(
            appended,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def __getitem__(self, col_name: str) -> Any:
        """Get column data by name."""
        if hasattr(self, "_raw_data") and col_name in self._raw_data:
            return self._raw_data[col_name]
        if hasattr(self, "feat_dict") and col_name in self.feat_dict:
            return self.feat_dict[col_name]
        raise KeyError(f"Column '{col_name}' not found")

    def materialize(self) -> "TensorFrame":
        """Return self as a materialized TensorFrame."""
        return self

    def to_tensor_dict(self) -> Dict[str, Any]:
        """Return the tensor dictionary."""
        if hasattr(self, "_raw_data"):
            return {
                str(k): v
                for k, v in self._raw_data.items()
                if isinstance(v, torch.Tensor)
            }
        if hasattr(self, "feat_dict"):
            return {str(k): v for k, v in self.feat_dict.items()}
        return {}

    def iter_rows(self, batch_size: int = 1000):
        """Iterate over table rows in batches."""
        # Simple implementation - yield the data in chunks
        if hasattr(self, "_raw_data") and self._raw_data:
            total_rows = self.num_rows
            for start in range(0, total_rows, batch_size):
                batch = {}
                for k, v in self._raw_data.items():
                    if isinstance(v, torch.Tensor):
                        batch[str(k)] = v[start : start + batch_size]
                    elif isinstance(v, list):
                        batch[str(k)] = v[start : start + batch_size]
                    else:
                        batch[str(k)] = v
                yield batch
        elif hasattr(self, "feat_dict") and self.feat_dict:
            total_rows = self.num_rows
            for start in range(0, total_rows, batch_size):
                batch = {}
                for k, v in self.feat_dict.items():
                    if hasattr(v, "shape") and len(v.shape) > 0:
                        batch[str(k)] = v[start : start + batch_size]
                    else:
                        batch[str(k)] = v
                yield batch

    @classmethod
    def from_fits(cls, file_path: str, hdu_index: int = 1) -> "TableHDU":
        """Create TableHDU from FITS file.

        Args:
            file_path: Path to FITS file
            hdu_index: HDU index (1-based)
        """
        # Input validation
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        if not isinstance(hdu_index, int) or hdu_index < 0:
            raise ValueError("hdu_index must be a non-negative integer")

        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FITS file not found: {file_path}")

        try:
            import torchfits.cpp as cpp

            tensor_dict = cpp.read_fits_table(file_path, hdu_index)
            header = Header(cpp.read_header_dict(file_path, hdu_index))

            return cls(
                tensor_dict, {}, header, source_path=file_path, source_hdu=hdu_index
            )
        except (IOError, RuntimeError) as e:
            from .logging import logger

            logger.error(
                f"Failed to read table from {file_path}[{hdu_index}]: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to read table from {file_path}[{hdu_index}]: {e}"
            ) from e
        except Exception as e:
            from .logging import logger

            logger.critical(
                f"Unexpected error reading {file_path}[{hdu_index}]: {str(e)}"
            )
            raise

    def to_fits(self, file_path: str, overwrite: bool = False):
        """Write this table to a FITS file (materialized table payload)."""
        # Use top-level writer so non-tensor columns (e.g., strings/VLA/complex)
        # are preserved via the table fallback path.
        import torchfits

        payload = (
            dict(self._raw_data)
            if hasattr(self, "_raw_data")
            else self.to_tensor_dict()
        )
        for name in self.string_columns:
            value = payload.get(name)
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.dim() == 2
            ):
                decoded: List[str] = []
                arr = value.detach().cpu().numpy()
                for row in arr:
                    decoded.append(
                        bytes(row.tolist())
                        .decode("ascii", errors="ignore")
                        .rstrip(" \x00")
                    )
                payload[name] = decoded

        torchfits.write(
            file_path,
            payload,
            header=self.header if self.header is not None else {},
            overwrite=overwrite,
        )

    def __repr__(self):
        name = self.header.get("EXTNAME", "TABLE")
        return (
            f"TableHDU(name='{name}', rows={self.num_rows}, cols={len(self.columns)})"
        )


class TableHDURef:
    """
    Lazy, file-backed table handle.

    Unlike TableHDU (TensorFrame), this does not materialize the full table at open().
    It exposes:
    - metadata (columns/num_rows/schema) from headers
    - explicit materialization via .materialize()/read()
    - out-of-core streaming via .iter_rows() and Arrow via torchfits.table.*
    """

    def __init__(
        self,
        *,
        header: Optional[Header] = None,
        source_path: Optional[str] = None,
        source_hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
    ):
        self.header = header or Header()
        self._source_path = source_path
        self._source_hdu = source_hdu
        self._columns = columns[:] if columns else None
        self._row_slice = row_slice

    def _require_source(self) -> tuple[str, int]:
        if not self._source_path or self._source_hdu is None:
            raise RuntimeError("This TableHDURef is not associated with a FITS file")
        return self._source_path, int(self._source_hdu)

    @property
    def num_rows(self) -> int:
        try:
            total = int(self.header.get("NAXIS2", 0))
        except Exception:
            total = 0
        if total <= 0:
            return 0
        if self._row_slice is None:
            return total
        # Respect view row slice.
        if isinstance(self._row_slice, tuple):
            start, stop = self._row_slice
        else:
            start = 0 if self._row_slice.start is None else int(self._row_slice.start)
            stop = self._row_slice.stop
        start = int(start)
        if start < 0:
            start = 0
        if stop is None:
            return max(0, total - start)
        stop = int(stop)
        stop = min(stop, total)
        return max(0, stop - start)

    def __len__(self) -> int:
        return self.num_rows

    @property
    def columns(self) -> List[str]:
        # Prefer explicit projection if this is a view.
        if self._columns is not None:
            return list(self._columns)
        try:
            n = int(self.header.get("TFIELDS", 0))
        except Exception:
            n = 0
        out: List[str] = []
        for i in range(1, n + 1):
            name = self.header.get(f"TTYPE{i}")
            if isinstance(name, str) and name:
                out.append(name)
            else:
                out.append(f"COL{i}")
        return out

    @property
    def string_columns(self) -> List[str]:
        # Mirror TableHDU behavior: infer from header TFORMn with 'A' code.
        cols = []
        name_by_idx: Dict[str, str] = {}
        for k, v in self.header.items():
            if isinstance(k, str) and k.upper().startswith("TTYPE"):
                name_by_idx[k[5:]] = str(v)
        for k, v in self.header.items():
            if not isinstance(k, str) or not k.upper().startswith("TFORM"):
                continue
            idx = k[5:]
            tform = str(v).strip().upper()
            if "A" in tform:
                name = name_by_idx.get(idx)
                if name:
                    cols.append(name)
        if self._columns is not None:
            cols = [c for c in cols if c in set(self._columns)]
        return sorted(set(cols))

    @property
    def schema(self) -> Dict[str, Any]:
        # Reuse the same basic schema format as TableHDU._build_schema().
        if not self.header:
            return {"columns": [], "string_columns": [], "vla_columns": []}

        name_by_idx: Dict[int, str] = {}
        tform_by_idx: Dict[int, str] = {}
        tdim_by_idx: Dict[int, str] = {}
        for key, value in self.header.items():
            if isinstance(key, str) and key.startswith("TTYPE"):
                idx = int(key[5:]) if key[5:].isdigit() else None
                if idx is not None:
                    name_by_idx[idx] = str(value)
            if isinstance(key, str) and key.startswith("TFORM"):
                idx = int(key[5:]) if key[5:].isdigit() else None
                if idx is not None:
                    tform_by_idx[idx] = str(value)
            if isinstance(key, str) and key.startswith("TDIM"):
                idx = int(key[4:]) if key[4:].isdigit() else None
                if idx is not None:
                    tdim_by_idx[idx] = str(value)

        def _parse_tform(tform: str) -> Dict[str, Any]:
            # Examples: "E", "20A", "1PB", "1PJ"
            import re

            tform = tform.strip().upper()
            m = re.match(r"(\d*)([PQ]?)([A-Z])", tform)
            if not m:
                return {"tform": tform, "repeat": None, "vla": False, "code": None}
            repeat = int(m.group(1)) if m.group(1) else 1
            vla = m.group(2) in ("P", "Q")
            code = m.group(3)
            base = {"tform": tform, "repeat": repeat, "vla": vla, "code": code}
            if vla:
                base["vla_descriptor"] = m.group(2)
            return base

        selected = set(self._columns) if self._columns is not None else None
        columns = []
        string_cols = []
        vla_cols = []
        for idx in sorted(name_by_idx.keys()):
            name = name_by_idx[idx]
            if selected is not None and name not in selected:
                continue
            tform = tform_by_idx.get(idx, "")
            info = (
                _parse_tform(tform)
                if tform
                else {"tform": "", "repeat": None, "vla": False, "code": None}
            )
            tdim = tdim_by_idx.get(idx)
            is_string = info.get("code") == "A"
            if is_string:
                string_cols.append(name)
            if info.get("vla"):
                vla_cols.append(name)
            columns.append(
                {
                    "name": name,
                    "tform": info.get("tform", ""),
                    "repeat": info.get("repeat"),
                    "code": info.get("code"),
                    "string": is_string,
                    "vla": bool(info.get("vla")),
                    "tdim": tdim,
                }
            )

        return {
            "columns": columns,
            "string_columns": string_cols,
            "vla_columns": vla_cols,
        }

    def select(self, cols: List[str]) -> "TableHDURef":
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise TypeError("cols must be a list[str]")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=cols,
            row_slice=self._row_slice,
        )

    def head(self, n: int) -> "TableHDURef":
        if n < 0:
            raise ValueError("n must be >= 0")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=self._columns,
            row_slice=slice(0, n),
        )

    def filter(self, condition: str) -> "TableHDU":
        """
        In-memory filter convenience (materializes the table, then filters).

        For large/out-of-core workflows, prefer Arrow:
        - torchfits.table.scan(...)/reader(...)/scanner(...)
        """
        return self.materialize().filter(condition)

    def _normalize_row_slice(self, row_slice: Optional[slice | tuple[int, int]]):
        # Convert python-style slice to (start_row, num_rows) where start_row is 1-based.
        if row_slice is None:
            return 1, -1
        if isinstance(row_slice, tuple):
            if len(row_slice) != 2:
                raise ValueError("row_slice tuple must be (start, stop)")
            start, stop = row_slice
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = row_slice.stop
            if row_slice.step not in (None, 1):
                raise ValueError("row_slice step is not supported")
        start = int(start)
        if start < 0:
            raise ValueError("row_slice start must be >= 0")
        if stop is None:
            # Until end
            return start + 1, -1
        stop = int(stop)
        if stop < start:
            return start + 1, 0
        return start + 1, stop - start

    def _is_ascii_table(self) -> bool:
        try:
            return str(self.header.get("XTENSION", "")).strip().upper() == "TABLE"
        except Exception:
            return False

    def read(
        self,
        *,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
        mmap: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        import torchfits

        path, hdu = self._require_source()
        if columns is None:
            columns = self._columns
        if row_slice is None:
            row_slice = self._row_slice
        start_row, num_rows = self._normalize_row_slice(row_slice)
        # ASCII tables are not supported by the mmap column path; force CFITSIO reads.
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False
        return torchfits.read(
            path,
            hdu=hdu,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            mmap=effective_mmap,
            device=device,
            # Don't use the python-side result cache for an open() handle: it can
            # return stale data if the file is updated in-place.
            cache_capacity=0,
        )

    def materialize(self, *, mmap: bool = True, device: str = "cpu") -> "TableHDU":
        data = self.read(mmap=mmap, device=device)
        return TableHDU(
            data,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def iter_rows(self, batch_size: int = 65536, *, mmap: bool = True):
        import torchfits

        path, hdu = self._require_source()
        start_row, num_rows = self._normalize_row_slice(self._row_slice)
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False

        for chunk in torchfits.stream_table(
            path,
            hdu=hdu,
            columns=self._columns,
            start_row=start_row,
            num_rows=num_rows,
            chunk_rows=batch_size,
            mmap=effective_mmap,
        ):
            yield chunk

    def __getitem__(self, col_name: str) -> Any:
        # Materialize only the requested column (still potentially large).
        data = self.read(columns=[col_name])
        if col_name not in data:
            raise KeyError(f"Column '{col_name}' not found")
        return data[col_name]

    def get_string_column(
        self, name: str, encoding: str = "ascii", strip: bool = True
    ) -> List[str]:
        value = self[name]
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Column '{name}' is not a tensor string column")
        if value.dtype != torch.uint8 or value.dim() != 2:
            raise TypeError(
                f"Column '{name}' is not a uint8 (rows,width) encoded string column"
            )
        out: List[str] = []
        arr = value.detach().cpu().numpy()
        for row in arr:
            s = bytes(row.tolist()).decode(encoding, errors="ignore")
            if strip:
                s = s.rstrip(" \x00")
            out.append(s)
        return out

    def get_vla_column(self, name: str) -> List[Tensor]:
        value = self[name]
        if isinstance(value, list):
            # torchfits.read returns VLA as list[Tensor] in the torch path.
            return value  # type: ignore[return-value]
        raise KeyError(f"Column '{name}' is not a VLA list")

    def get_vla_lengths(self, name: str) -> List[int]:
        values = self.get_vla_column(name)
        lengths: List[int] = []
        for item in values:
            if isinstance(item, torch.Tensor):
                lengths.append(int(item.numel()))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
            else:
                lengths.append(1)
        return lengths

    @property
    def vla_lengths(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for col in self.schema.get("vla_columns", []):
            try:
                out[col] = self.get_vla_lengths(col)
            except Exception:
                continue
        return out

    @property
    def data(self):
        """Dictionary-like column access (lazy per-column reads)."""

        class _Wrapper:
            def __init__(self, parent: "TableHDURef"):
                self._parent = parent

            def __getitem__(self, key: str) -> Any:
                return self._parent[key]

            def __contains__(self, key: str) -> bool:
                return key in set(self._parent.columns)

            def keys(self):
                return self._parent.columns

        return _Wrapper(self)

    def to_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        # Prefer Arrow-native table APIs.
        return torchfits.table.read(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def scan_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.scan(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def reader_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.reader(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def _refresh_file_view(self) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        header = Header(torchfits.get_header(path, hdu))
        # Return a full refreshed view after in-place mutations.
        return TableHDURef(header=header, source_path=path, source_hdu=hdu)

    def append_rows_file(self, rows: Dict[str, Any]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.append_rows(path, rows, hdu=hdu)
        return self._refresh_file_view()

    def insert_column_file(
        self,
        name: str,
        values: Any,
        *,
        index: Optional[int] = None,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.insert_column(
            path,
            name,
            values,
            hdu=hdu,
            index=index,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        return self._refresh_file_view()

    def replace_column_file(
        self,
        name: str,
        values: Any,
        *,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.replace_column(
            path,
            name,
            values,
            hdu=hdu,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        return self._refresh_file_view()

    def insert_rows_file(self, rows: Dict[str, Any], *, row: int) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.insert_rows(path, rows, row=row, hdu=hdu)
        return self._refresh_file_view()

    def delete_rows_file(
        self, row_slice: Union[int, slice, tuple[int, int]]
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.delete_rows(path, row_slice, hdu=hdu)
        return self._refresh_file_view()

    def update_rows_file(
        self,
        rows: Dict[str, Any],
        row_slice: Union[slice, tuple[int, int]],
        *,
        mmap: Union[bool, str] = "auto",
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.update_rows(path, rows, row_slice=row_slice, hdu=hdu, mmap=mmap)
        return self._refresh_file_view()

    def rename_columns_file(self, mapping: Dict[str, str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.rename_columns(path, mapping, hdu=hdu)
        return self._refresh_file_view()

    def drop_columns_file(self, columns: List[str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.drop_columns(path, columns, hdu=hdu)
        return self._refresh_file_view()

    def __repr__(self):
        name = self.header.get("EXTNAME", "TABLE")
        proj = f", cols={len(self.columns)}" if self._columns is not None else ""
        return f"TableHDURef(name='{name}', rows={self.num_rows}{proj})"


class HDUList:
    """HDU container."""

    def __init__(
        self, hdus: Optional[List[Union[TensorHDU, TableHDU, TableHDURef]]] = None
    ):
        self._hdus = hdus or []
        self._file_handle = None

    @classmethod
    def fromfile(cls, path: str, mode: str = "r") -> "HDUList":
        # Input validation
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")

        if mode not in ["r", "w", "rw"]:
            raise ValueError("Mode must be 'r', 'w', or 'rw'")

        import os

        if mode == "r" and not os.path.exists(path):
            raise FileNotFoundError(f"FITS file not found: {path}")

        hdul = cls()

        try:
            import torchfits.cpp as cpp

            # Open file using C++ backend with optimized batch header reading
            # Returns (FITSFile object, list of HDUInfo)
            try:
                handle, hdu_infos = cpp.open_and_read_headers(
                    path, 0 if mode == "r" else 1
                )
            except AttributeError:
                # Fallback if binding missing (should not happen)
                handle = cpp.open_fits_file(path, mode)
                hdu_infos = []
                num_hdus = cpp.get_num_hdus(handle)
                for i in range(num_hdus):
                    header_dict = cpp.read_header(handle, i)
                    hdu_type = cpp.get_hdu_type(handle, i)

                    # Create dummy object with attributes
                    class Info:
                        pass

                    info = Info()
                    info.index = i
                    info.type = hdu_type
                    info.header = header_dict
                    hdu_infos.append(info)

            hdul._file_handle = handle

            for info in hdu_infos:
                # Read header
                header = Header(info.header)
                if not hasattr(header, "_cards"):
                    header._cards = []
                for k, v in info.header.items():
                    if k in ("HISTORY", "COMMENT"):
                        header._cards.append((k, v, ""))

                # Determine HDU type
                hdu_type = info.type
                i = info.index

                if hdu_type == "IMAGE":
                    hdu = TensorHDU(header=header, file_handle=handle, hdu_index=i)
                elif hdu_type in ["ASCII_TABLE", "BINARY_TABLE"]:
                    # Safe-by-default: do not materialize tables at open().
                    hdu = TableHDURef(header=header, source_path=path, source_hdu=i)
                else:
                    # Unknown type, treat as empty image
                    hdu = TensorHDU(header=header)

                hdul._hdus.append(hdu)

            return hdul

        except Exception as e:
            # Clean up handle if open failed partly
            if hdul._file_handle:
                try:
                    hdul._file_handle.close()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to open FITS file '{path}': {str(e)}") from e

    def __len__(self) -> int:
        return len(self._hdus)

    def __getitem__(self, key: Union[int, str]) -> Union[TensorHDU, TableHDU]:
        if isinstance(key, int):
            return self._hdus[key]
        for hdu in self._hdus:
            if hdu.header.get("EXTNAME") == key:
                return hdu
        raise KeyError(f"HDU '{key}' not found")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def write(self, path: str, overwrite: bool = False):
        import torchfits
        import torchfits.cpp as cpp

        class _TableWriteProxy:
            def __init__(self, raw_data, header):
                self._raw_data = raw_data
                self.header = header

        def _sanitize_table_header_for_write(header):
            skip_keys = {
                "SIMPLE",
                "XTENSION",
                "BITPIX",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
                "EXTEND",
                "THEAP",
            }
            out = {}
            for key, value in dict(header or {}).items():
                key_upper = str(key).upper()
                if key_upper in skip_keys or key_upper.startswith("NAXIS"):
                    continue
                out[str(key)] = value
            return out

        payload_hdus = []
        for hdu in list(self._hdus):
            if isinstance(hdu, TableHDU):
                raw_data = dict(getattr(hdu, "_raw_data", {}))
                raw_data = torchfits._normalize_cpp_table_data(raw_data)
                payload_hdus.append(
                    _TableWriteProxy(
                        raw_data, _sanitize_table_header_for_write(hdu.header)
                    )
                )
            elif isinstance(hdu, TableHDURef):
                # Materialize lazy tables so rewrite/replace paths keep extension contents.
                materialized = hdu.materialize()
                raw_data = dict(getattr(materialized, "_raw_data", {}))
                raw_data = torchfits._normalize_cpp_table_data(raw_data)
                payload_hdus.append(
                    _TableWriteProxy(
                        raw_data, _sanitize_table_header_for_write(hdu.header)
                    )
                )
            else:
                payload_hdus.append(hdu)

        cpp.write_fits_file(path, payload_hdus, overwrite)

    def append(self, hdu: Union[TensorHDU, TableHDU]):
        self._hdus.append(hdu)

    def validate(self) -> bool:
        """Validate the FITS file structure and contents.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Iterate over HDUs
            for i, hdu in enumerate(self._hdus):
                # Check header
                if not hdu.header:
                    return False

                # Check data access
                if isinstance(hdu, TensorHDU):
                    if hdu._file_handle:
                        # Try reading shape/dtype
                        _ = hdu.data.shape
                        _ = hdu.data.dtype
                elif isinstance(hdu, TableHDU):
                    # Check columns
                    _ = hdu.columns
                    _ = hdu.num_rows

            return True
        except Exception:
            return False

    def info(self, output=None):
        """Print summary info about the HDUList."""
        summary = self._get_summary()
        if output is None:
            print(summary)
        else:
            output.write(summary + "\n")

    def _get_summary(self) -> str:
        """Generate summary table."""
        # Header
        lines = []
        # Try to get filename
        filename = "(No file associated)"
        if self._file_handle and hasattr(self._file_handle, "name"):
            filename = self._file_handle.name

        lines.append(f"Filename: {filename}")
        lines.append("No.    Name         Type       Cards   Dimensions   Format")

        for idx, hdu in enumerate(self._hdus):
            # Name
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))

            # Type
            if isinstance(hdu, (TableHDU, TableHDURef)):
                hdu_type = "TableHDU"
            else:
                if idx == 0 and name == "PRIMARY":
                    hdu_type = "PrimaryHDU"
                else:
                    hdu_type = "ImageHDU"

            # Cards
            cards = (
                len(hdu.header._cards)
                if hasattr(hdu.header, "_cards")
                else len(hdu.header)
            )

            # Dimensions & Format
            if isinstance(hdu, (TableHDU, TableHDURef)):
                dims = f"({hdu.num_rows}R x {len(hdu.columns)}C)"
                fmt = "Table"
            elif isinstance(hdu, TensorHDU):
                dims = hdu._get_shape_str()
                fmt = hdu._get_dtype_str()
            else:
                dims = ""
                fmt = ""

            lines.append(
                f"{idx:<6d} {name:<12s} {hdu_type:<10s} {cards:<7d} {dims:<12s} {fmt}"
            )

        return "\n".join(lines)

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""
        html = ["<table style='border-collapse: collapse; width: 100%;'><thead><tr>"]
        headers = ["No.", "Name", "Type", "Cards", "Dimensions", "Format"]
        styles = (
            ["text-align: left;"] * 3
            + ["text-align: right;"]
            + ["text-align: left;"] * 2
        )
        for h, s in zip(headers, styles):
            html.append(
                f"<th style='{s} padding: 4px; border-bottom: 2px solid #ddd;'>{h}</th>"
            )
        html.append("</tr></thead><tbody>")

        for idx, hdu in enumerate(self._hdus):
            name = str(hdu.header.get("EXTNAME", "PRIMARY"))
            if isinstance(hdu, (TableHDU, TableHDURef)):
                hdu_type = "TableHDU"
                dims = f"({hdu.num_rows}R x {len(hdu.columns)}C)"
                fmt = "Table"
            elif isinstance(hdu, TensorHDU):
                hdu_type = (
                    "PrimaryHDU" if idx == 0 and name == "PRIMARY" else "ImageHDU"
                )
                dims, fmt = hdu._get_shape_str(), hdu._get_dtype_str()
            else:
                hdu_type, dims, fmt = "Unknown", "", ""

            cards = (
                len(hdu.header._cards)
                if hasattr(hdu.header, "_cards")
                else len(hdu.header)
            )

            row = [idx, name, hdu_type, cards, dims, fmt]
            html.append("<tr>")
            for val, s in zip(row, styles):
                html.append(
                    f"<td style='{s} padding: 4px; border-bottom: 1px solid #eee;'>{val}</td>"
                )
            html.append("</tr>")

        html.append("</tbody></table>")
        return "".join(html)

    def __repr__(self):
        return self._get_summary()
