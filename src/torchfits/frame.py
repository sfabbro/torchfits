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

    data = torchfits.read(path, hdu=hdu, columns=columns, return_header=False)
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

    # Process all available stypes dynamically
    for stype_enum, tensor in tf.feat_dict.items():
        names = tf.col_names_dict[stype_enum]

        # 1D/2D continuous types
        if stype_enum in [stype.numerical, stype.categorical, getattr(stype, "timestamp", None)]:
            for i, name in enumerate(names):
                data[name] = tensor[:, i]

        # 3D continuous types (e.g. embeddings)
        elif stype_enum in [getattr(stype, "embedding", None), getattr(stype, "text_embedded", None), getattr(stype, "image_embedded", None)]:
            for i, name in enumerate(names):
                data[name] = tensor[:, i, :]

        # Variable-length nested types
        elif stype_enum in [getattr(stype, "sequence_numerical", None), getattr(stype, "multicategorical", None)]:
            from torch_frame.data import MultiNestedTensor
            if isinstance(tensor, MultiNestedTensor):
                num_rows = tensor.num_rows
                num_cols = tensor.num_cols
                values = tensor.values
                offset = tensor.offset

                for j, name in enumerate(names):
                    col_data = []
                    for i in range(num_rows):
                        start = offset[i * num_cols + j].item()
                        end = offset[i * num_cols + j + 1].item()
                        col_data.append(values[start:end])
                    data[name] = col_data
            else:
                warnings.warn(f"Expected MultiNestedTensor for stype {stype_enum.name}, got {type(tensor)}")
        else:
            warnings.warn(f"Skipping unsupported stype: {stype_enum.name}")

    torchfits.write(path, data, overwrite=overwrite)
