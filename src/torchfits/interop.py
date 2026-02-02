from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def to_pandas(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "object",
) -> pd.DataFrame:
    """
    Convert a dictionary of PyTorch tensors to a Pandas DataFrame.
    Attempts to use zero-copy conversion where possible (via numpy).

    Args:
        data: Dictionary mapping column names to PyTorch tensors or lists of tensors (VLA).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas is required for to_pandas conversion.")

    processed_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if decode_bytes and value.dtype == torch.uint8 and value.dim() == 2:
                rows = value.cpu().numpy()
                strings = []
                for row in rows:
                    s = bytes(row.tolist()).decode(encoding, errors="ignore")
                    if strip:
                        s = s.rstrip(" \x00")
                    strings.append(s)
                processed_data[key] = strings
            else:
                processed_data[key] = value.numpy()
        elif isinstance(value, list):
            # Handle VLA (list of tensors)
            # We convert each tensor to numpy. This creates a list of numpy arrays.
            # Pandas handles this as object column.
            if vla_policy == "object":
                processed_data[key] = [
                    t.numpy() if isinstance(t, torch.Tensor) else t for t in value
                ]
            elif vla_policy == "drop":
                continue
            else:
                raise ValueError("vla_policy must be 'object' or 'drop'")
        else:
            # Pass through other types (e.g. strings if any)
            processed_data[key] = value

    return pd.DataFrame(processed_data)


def to_arrow(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "list",
) -> pa.Table:
    """
    Convert a dictionary of PyTorch tensors to a PyArrow Table.
    Attempts to use zero-copy conversion where possible.

    Args:
        data: Dictionary mapping column names to PyTorch tensors or lists of tensors (VLA).

    Returns:
        pa.Table: A PyArrow Table containing the data.
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError("PyArrow is required for to_arrow conversion.")

    arrays = []
    names = []

    for key, value in data.items():
        names.append(key)
        if isinstance(value, torch.Tensor):
            if decode_bytes and value.dtype == torch.uint8 and value.dim() == 2:
                rows = value.cpu().numpy()
                strings = []
                for row in rows:
                    s = bytes(row.tolist()).decode(encoding, errors="ignore")
                    if strip:
                        s = s.rstrip(" \x00")
                    strings.append(s)
                arrays.append(pa.array(strings))
            else:
                arrays.append(pa.array(value.numpy()))
        elif isinstance(value, list):
            # Handle VLA
            # For VLA, we might want a ListArray.
            # But constructing it from list of numpy arrays might involve copy.
            # Let's just let pyarrow infer.
            # If elements are tensors, convert to numpy first.
            if vla_policy == "list":
                converted_list = [
                    t.numpy() if isinstance(t, torch.Tensor) else t for t in value
                ]
                arrays.append(pa.array(converted_list))
            elif vla_policy == "drop":
                names.pop()
            else:
                raise ValueError("vla_policy must be 'list' or 'drop'")
        else:
            arrays.append(pa.array(value))

    return pa.Table.from_arrays(arrays, names=names)
