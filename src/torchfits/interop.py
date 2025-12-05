from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def to_pandas(data: Dict[str, Any]) -> pd.DataFrame:
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
            # Convert tensor to numpy
            # Note: tensor.numpy() is zero-copy if tensor is on CPU and contiguous.
            # torchfits returns contiguous CPU tensors, so this should be zero-copy.
            processed_data[key] = value.numpy()
        elif isinstance(value, list):
            # Handle VLA (list of tensors)
            # We convert each tensor to numpy. This creates a list of numpy arrays.
            # Pandas handles this as object column.
            processed_data[key] = [
                t.numpy() if isinstance(t, torch.Tensor) else t for t in value
            ]
        else:
            # Pass through other types (e.g. strings if any)
            processed_data[key] = value

    return pd.DataFrame(processed_data)


def to_arrow(data: Dict[str, Any]) -> pa.Table:
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
            # Convert tensor to numpy, then to arrow
            # pa.array(numpy_array) is zero-copy if possible
            arrays.append(pa.array(value.numpy()))
        elif isinstance(value, list):
            # Handle VLA
            # For VLA, we might want a ListArray.
            # But constructing it from list of numpy arrays might involve copy.
            # Let's just let pyarrow infer.
            # If elements are tensors, convert to numpy first.
            converted_list = [
                t.numpy() if isinstance(t, torch.Tensor) else t for t in value
            ]
            arrays.append(pa.array(converted_list))
        else:
            arrays.append(pa.array(value))

    return pa.Table.from_arrays(arrays, names=names)
