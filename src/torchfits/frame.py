import warnings
from typing import Any, Dict, List, Optional

import torch

try:
    from torch_frame import TensorFrame, stype

    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False
    TensorFrame = Any  # type placeholder


def to_tensor_frame(data: Dict[str, torch.Tensor]) -> TensorFrame:
    """
    Convert a dictionary of PyTorch tensors (from torchfits.read_columns) to a TorchFrame TensorFrame.

    Args:
        data: Dictionary mapping column names to tensors.

    Returns:
        A torch_frame.TensorFrame object.

    Raises:
        ImportError: If torch_frame is not installed.
        ValueError: If data is incompatible.
    """
    if not HAS_TORCH_FRAME:
        raise ImportError(
            "torch-frame is not installed. Please install it with 'pip install pytorch-frame'."
        )

    if not data:
        raise ValueError("Input data is empty")

    # Group columns by stype
    num_cols = []
    num_col_names = []

    cat_cols = []
    cat_col_names = []

    num_rows = -1

    for name, tensor in data.items():
        if num_rows == -1:
            num_rows = tensor.shape[0]
        elif tensor.shape[0] != num_rows:
            raise ValueError(
                f"Column {name} has {tensor.shape[0]} rows, expected {num_rows}"
            )

        # Determine stype based on dtype
        if tensor.dtype in [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int16,
            torch.uint8,
        ]:
            # Treat integers as numerical for now, unless user specifies otherwise (TODO: allow schema override)
            # Ensure 2D shape (rows, 1)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() > 2:
                warnings.warn(
                    f"Skipping column {name} with dimension {tensor.dim()} (only 1D/2D supported)"
                )
                continue

            num_cols.append(
                tensor.to(torch.float32)
            )  # torch-frame expects float for numerical
            num_col_names.append(name)

        elif tensor.dtype in [torch.int64, torch.bool]:
            # Treat int64 and bool as categorical
            # Note: torchfits currently returns int64 for strings (hashed)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)

            if tensor.dtype == torch.bool:
                tensor = tensor.to(torch.int64)

            cat_cols.append(tensor)
            cat_col_names.append(name)
        else:
            warnings.warn(
                f"Skipping column {name} with unsupported dtype {tensor.dtype}"
            )

    # Construct feat_dict
    feat_dict = {}
    col_names_dict = {}

    if num_cols:
        feat_dict[stype.numerical] = torch.cat(num_cols, dim=1)
        col_names_dict[stype.numerical] = num_col_names

    if cat_cols:
        feat_dict[stype.categorical] = torch.cat(cat_cols, dim=1)
        col_names_dict[stype.categorical] = cat_col_names

    if not feat_dict:
        raise ValueError("No valid columns found to convert to TensorFrame")

    return TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)


def read_tensor_frame(
    path: str, hdu: int = 1, columns: Optional[List[str]] = None
) -> TensorFrame:
    """
    Read a FITS table directly into a TensorFrame.

    Args:
        path: Path to FITS file.
        hdu: HDU index (default: 1 for first table extension).
        columns: Optional list of column names to read.

    Returns:
        A torch_frame.TensorFrame object.
    """
    import torchfits

    data, _ = torchfits.read(path, hdu=hdu, columns=columns)
    return to_tensor_frame(data)


def write_tensor_frame(path: str, tf: TensorFrame, overwrite: bool = False):
    """
    Write a TensorFrame to a FITS file.

    Args:
        path: Output file path.
        tf: TensorFrame object to write.
        overwrite: Whether to overwrite existing file.
    """
    import torchfits

    if not HAS_TORCH_FRAME:
        raise ImportError("torch-frame is not installed.")

    data = {}

    # Extract numerical columns
    if stype.numerical in tf.feat_dict:
        tensor = tf.feat_dict[stype.numerical]
        names = tf.col_names_dict[stype.numerical]
        for i, name in enumerate(names):
            data[name] = tensor[:, i]

    # Extract categorical columns
    if stype.categorical in tf.feat_dict:
        tensor = tf.feat_dict[stype.categorical]
        names = tf.col_names_dict[stype.categorical]
        for i, name in enumerate(names):
            data[name] = tensor[:, i]

    # TODO: Handle other stypes if needed

    torchfits.write(path, data, overwrite=overwrite)
