import warnings
from typing import Any, Dict, List, Optional

import torch

try:
    from torch_frame import TensorFrame, stype

    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False
    TensorFrame = Any  # type placeholder


def to_tensor_frame(
    data: Dict[str, torch.Tensor], schema: Optional[Dict[str, Any]] = None
) -> TensorFrame:
    """
    Convert a dictionary of PyTorch tensors (from torchfits.read_columns) to a TorchFrame TensorFrame.

    Args:
        data: Dictionary mapping column names to tensors.
        schema: Optional dictionary mapping column names to stype.

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

    if schema is None:
        schema = {}

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

        target_stype = schema.get(name)

        # Determine stype based on schema or dtype
        if target_stype == stype.numerical or (
            target_stype is None
            and tensor.dtype
            in [
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int16,
                torch.uint8,
            ]
        ):
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

        elif target_stype == stype.categorical or (
            target_stype is None and tensor.dtype in [torch.int64, torch.bool]
        ):
            # Treat int64 and bool as categorical
            # Note: torchfits currently returns int64 for strings (hashed)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)

            if tensor.dtype != torch.int64:
                tensor = tensor.to(torch.int64)

            cat_cols.append(tensor)
            cat_col_names.append(name)
        else:
            warnings.warn(
                f"Skipping column {name} with unsupported dtype {tensor.dtype} or stype {target_stype}"
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
    path: str,
    hdu: int = 1,
    columns: Optional[List[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> TensorFrame:
    """
    Read a FITS table directly into a TensorFrame.

    Args:
        path: Path to FITS file.
        hdu: HDU index (default: 1 for first table extension).
        columns: Optional list of column names to read.
        schema: Optional dictionary mapping column names to stype.

    Returns:
        A torch_frame.TensorFrame object.
    """
    from .io import read

    data = read(path, hdu=hdu, columns=columns, return_header=False)
    return to_tensor_frame(data, schema=schema)


def write_tensor_frame(path: str, tf: TensorFrame, overwrite: bool = False):
    """
    Write a TensorFrame to a FITS file.

    Args:
        path: Output file path.
        tf: TensorFrame object to write.
        overwrite: Whether to overwrite existing file.
    """
    from .io import write

    if not HAS_TORCH_FRAME:
        raise ImportError("torch-frame is not installed.")

    data = {}

    # Process all available stypes dynamically
    for stype_enum, tensor in tf.feat_dict.items():
        names = tf.col_names_dict[stype_enum]

        # 1D/2D continuous types
        if stype_enum in [
            stype.numerical,
            stype.categorical,
            getattr(stype, "timestamp", None),
        ]:
            for i, name in enumerate(names):
                data[name] = tensor[:, i]

    # Handle other standard 2D tensor stypes safely
    timestamp_stype = getattr(stype, "timestamp", None)
    if timestamp_stype and timestamp_stype in tf.feat_dict:
        tensor = tf.feat_dict[timestamp_stype]
        names = tf.col_names_dict[timestamp_stype]
        for i, name in enumerate(names):
            data[name] = tensor[:, i]

    # Handle MultiEmbeddingTensor stypes
    emb_stypes = []
    if hasattr(stype, "embedding"):
        emb_stypes.append(stype.embedding)
    if hasattr(stype, "text_embedded"):
        emb_stypes.append(stype.text_embedded)
    if hasattr(stype, "image_embedded"):
        emb_stypes.append(stype.image_embedded)

    for st in emb_stypes:
        if st in tf.feat_dict:
            tensor = tf.feat_dict[st]
            names = tf.col_names_dict[st]
            for i, name in enumerate(names):
                col = tensor[:, i]
                # Avoid builtin Tensor.values() method; we want MultiEmbeddingTensor.values property.
                if hasattr(col, "values") and not callable(col.values):
                    data[name] = col.values
                else:
                    data[name] = col

    # Handle MultiNestedTensor stypes (VLA)
    nested_stypes = []
    if hasattr(stype, "multicategorical"):
        nested_stypes.append(stype.multicategorical)
    if hasattr(stype, "sequence_numerical"):
        nested_stypes.append(stype.sequence_numerical)
    if hasattr(stype, "text_tokenized"):
        nested_stypes.append(stype.text_tokenized)

    for st in nested_stypes:
        if st in tf.feat_dict:
            tensor = tf.feat_dict[st]
            names = tf.col_names_dict[st]
            for i, name in enumerate(names):
                col = tensor[:, i]
                values = col.values
                offset = col.offset
                # Convert to list of tensors
                list_of_tensors = [
                    values[offset[j] : offset[j + 1]] for j in range(len(offset) - 1)
                ]
                data[name] = list_of_tensors

    write(path, data, overwrite=overwrite)
