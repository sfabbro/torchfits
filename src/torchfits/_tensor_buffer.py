"""Tensor-to-PyArrow conversion via the buffer protocol.

Provides :func:`tensor_to_arrow_array`, which converts a 1-D PyTorch tensor
to a ``pyarrow.Array`` using the tensor's NumPy buffer view and
``pa.Array.from_buffers``.

Supported dtypes: float32, float64, float16, int8, int16, int32, int64,
uint8.  Unsupported dtypes (bool, complex, bfloat16, …) fall back to
``pa.array(tensor.tolist())``.
"""

from __future__ import annotations

from typing import Any

import torch

# torch dtype → Arrow type factory name (resolved via ``getattr(pa, name)()``).
_TORCH_DTYPE_ARROW: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
}


def tensor_to_arrow_array(tensor: torch.Tensor, pa: Any) -> Any:
    """Convert a 1-D PyTorch tensor to a PyArrow Array without copying.

    Uses the NumPy view's buffer for supported numeric dtypes, falling back to
    ``pa.array(tensor.tolist())`` for types that don't have a direct Arrow
    buffer representation (bool, complex, bfloat16).

    Args:
        tensor: 1-D PyTorch tensor (any device, any contiguity).
        pa: The ``pyarrow`` module (passed by caller to avoid a hard import).

    Returns:
        A ``pyarrow.Array`` of the appropriate type.
    """
    # detach so we never keep an autograd graph alive for raw byte access
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tensor.ndim == 2:
        width = int(tensor.shape[1])
        flat = tensor_to_arrow_array(tensor.reshape(-1), pa)
        return pa.FixedSizeListArray.from_arrays(flat, width)
    if tensor.ndim != 1:
        raise ValueError(
            f"tensor_to_arrow_array expects 1-D or 2-D tensors, got shape {tuple(tensor.shape)}"
        )

    arrow_name = _TORCH_DTYPE_ARROW.get(tensor.dtype)
    if arrow_name is None:
        # Unsupported buffer dtype (bool, complex, bfloat16, …) — fall back.
        return pa.array(tensor.tolist())

    arrow_type = getattr(pa, arrow_name)()
    buf = pa.py_buffer(tensor.numpy())
    return pa.Array.from_buffers(arrow_type, tensor.numel(), [None, buf])
