"""Device normalization, validation, and MPS-safe tensor transfer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

_MPS_F64_WARNING = (
    "MPS does not support float64; downcasting to float32 (precision loss)"
)
_MPS_C128_WARNING = (
    "MPS does not support complex128; downcasting to complex64 (precision loss)"
)


def validate_device(device: str | torch.device) -> str:
    """Validate and normalize a device identifier.

    Accepts 'cpu', 'cuda', 'cuda:N', 'mps', 'mps:N', or torch.device instances.
    """
    dev_str = str(device)
    if dev_str in ("cpu", "cuda", "mps"):
        return dev_str
    if dev_str.startswith(("cuda:", "mps:")):
        index = dev_str.split(":", 1)[1]
        if index.isascii() and index.isdigit():
            return dev_str
    raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', 'mps' or 'mps:N'")


def to_device(
    tensor: Tensor,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> Tensor:
    """Move a tensor to a device, adapting MPS-unsupported dtypes (float64/complex128)."""
    # Fast path: the overwhelmingly common str-device cases without building
    # torch.device objects or touching dtype tables.
    import torch

    dev_str = device if type(device) is str else str(device)
    if dev_str == "cpu":
        return (
            tensor
            if tensor.device.type == "cpu"
            else tensor.to("cpu", non_blocking=non_blocking)
        )
    if dev_str == "mps" or dev_str.startswith("mps:"):
        if tensor.dtype == torch.float64:
            warnings.warn(_MPS_F64_WARNING, UserWarning, stacklevel=2)
            tensor = tensor.float()
        elif tensor.dtype == torch.complex128:
            warnings.warn(_MPS_C128_WARNING, UserWarning, stacklevel=2)
            tensor = tensor.to(torch.complex64)
    return tensor.to(dev_str, non_blocking=non_blocking)


def batch_to_device(tensors: list[Tensor], device: str | torch.device) -> list[Tensor]:
    """Move a list of tensors to a device, stacking when shapes match."""
    import torch

    if not tensors:
        return []
    dev_str = str(device)
    if dev_str == "mps" or dev_str.startswith("mps:"):
        # Warn once per call (and thus once per site under the default warning
        # filters) instead of once per tensor.
        warn_f64 = warn_c128 = False
        downcast: list[Tensor] = []
        for t in tensors:
            if t.dtype == torch.float64:
                if not warn_f64:
                    warn_f64 = True
                    warnings.warn(_MPS_F64_WARNING, UserWarning, stacklevel=2)
                downcast.append(t.float())
            elif t.dtype == torch.complex128:
                if not warn_c128:
                    warn_c128 = True
                    warnings.warn(_MPS_C128_WARNING, UserWarning, stacklevel=2)
                downcast.append(t.to(torch.complex64))
            else:
                downcast.append(t)
        tensors = downcast
    if len(tensors) == 1:
        return [tensors[0].to(device, non_blocking=True)]

    first = tensors[0]
    shape = first.shape
    dtype = first.dtype

    if all(t.shape == shape and t.dtype == dtype for t in tensors):
        return list(torch.stack(tensors).to(device, non_blocking=True).unbind(0))
    return [t.to(device, non_blocking=True) for t in tensors]
