"""Device normalization, validation, and MPS-safe tensor transfer."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_device(device: str | torch.device) -> str:
    """Validate and normalize a device identifier.

    Accepts 'cpu', 'cuda', 'cuda:N', 'mps', 'mps:N', or torch.device instances.
    """
    dev_str = str(device)
    if (
        dev_str not in ["cpu", "cuda", "mps"]
        and not dev_str.startswith("cuda:")
        and not dev_str.startswith("mps:")
    ):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    return dev_str


def to_device(
    tensor: Tensor,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> Tensor:
    """Move a tensor to a device, adapting MPS-unsupported dtypes (float64/complex128)."""
    # Fast path: the overwhelmingly common str-device cases without building
    # torch.device objects or touching dtype tables.
    dev_str = device if type(device) is str else str(device)
    if dev_str == "cpu":
        return (
            tensor
            if tensor.device.type == "cpu"
            else tensor.to("cpu", non_blocking=non_blocking)
        )
    if dev_str == "mps" or dev_str.startswith("mps:"):
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype == torch.complex128:
            tensor = tensor.to(torch.complex64)
    return tensor.to(dev_str, non_blocking=non_blocking)


def batch_to_device(tensors: list[Tensor], device: str | torch.device) -> list[Tensor]:
    """Move a list of tensors to a device, stacking when shapes match."""
    if not tensors:
        return []
    dev_str = str(device)
    if dev_str == "mps" or dev_str.startswith("mps:"):
        tensors = [
            t.float()
            if t.dtype == torch.float64
            else (t.to(torch.complex64) if t.dtype == torch.complex128 else t)
            for t in tensors
        ]
    if len(tensors) == 1:
        return [tensors[0].to(device, non_blocking=True)]

    first = tensors[0]
    shape = first.shape
    dtype = first.dtype

    if all(t.shape == shape and t.dtype == dtype for t in tensors):
        return list(torch.stack(tensors).to(device, non_blocking=True).unbind(0))
    return [t.to(device, non_blocking=True) for t in tensors]
