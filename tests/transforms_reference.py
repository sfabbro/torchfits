"""Reference implementation of SigmaClip for parity testing.

Mirrors the zero-alloc buffer implementation using freshly allocated
tensors at each iteration, making it suitable for verifying correctness
of the performant production code.

This is a **test utility only** — it is deliberately slow and allocates
freely to maximise clarity and auditability.
"""

from __future__ import annotations

import math

import torch

# Deliberately self-contained: a parity reference that imports production
# helpers cannot detect bugs in those helpers, so everything it needs is
# reimplemented below with straightforward torch/numpy primitives.
import numpy as np


def _normalize_dims(ndim: int, dim: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({d if d >= 0 else ndim + d for d in dim}))


def _flatten_dims(x, dims):
    keep = [d for d in range(x.ndim) if d not in dims]
    moved = x.permute(*keep, *dims)
    return moved.reshape(*moved.shape[: len(keep)], -1)


def _unflatten_result(reduced, shape, dims):
    shape_out = list(shape)
    for d in dims:
        shape_out[d] = 1
    return reduced.reshape(shape_out)


def _naive_median_lastdim(values: np.ndarray) -> np.ndarray:
    """Interpolated median along the last axis, NaN-excluding."""
    return np.nanmedian(values, axis=-1, keepdims=True)


def _median(x, dim, mask=None):
    """Independent median: upcast, mask, then numpy nanmedian per group."""
    arr = x.double().numpy()
    m = ~np.isnan(arr)
    if mask is not None:
        m &= mask.numpy()
    arr = np.where(m, arr, np.nan)

    def np_med(t, d, keepdim):
        moved = np.moveaxis(t, d, -1)
        med = _naive_median_lastdim(moved)
        back = np.moveaxis(med, -1, d)
        shape = list(t.shape)
        shape[d] = 1
        return torch.from_numpy(back.reshape(shape))

    if len(dim) == 1:
        out_t = np_med(arr, dim[0], True)
    else:
        flat = arr.reshape(*[arr.shape[d] for d in range(arr.ndim) if d not in dim], -1)
        med = _naive_median_lastdim(flat)
        shape_out = list(x.shape)
        for d in dim:
            shape_out[d] = 1
        out_t = torch.from_numpy(np.ascontiguousarray(med.reshape(shape_out)))
    return out_t.to(x.dtype)


def sigma_clip_naive(
    x: torch.Tensor,
    n_sigma: float,
    max_iter: int,
    dims: tuple[int, ...],
    fill: str,
) -> torch.Tensor:
    """Reference naive SigmaClip: allocates fresh tensors each iteration.

    Mirrors the zero-alloc buffer implementation using the simpler
    (less memory-efficient) approach of freshly allocating zeros
    and intermediate tensors in every iteration.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    n_sigma : float
        Number of standard deviations for clipping threshold.
    max_iter : int
        Maximum number of clipping iterations.
    dims : tuple[int, ...]
        Dimensions along which statistics are computed; empty for global.
    fill : str
        Fill value for clipped pixels: ``"mean"`` or ``"median"``.

    Returns
    -------
    torch.Tensor
        Clipped tensor with same shape and dtype as *x*.
    """
    ndim = x.ndim
    dims_norm: tuple[int, ...] = ()
    if len(dims) > 0:
        dims_norm = _normalize_dims(ndim, dims)

    mask = torch.ones_like(x, dtype=torch.bool)

    for _ in range(max_iter):
        # Compute mean of unmasked values (naive: fresh allocations)
        zeros = torch.zeros_like(x)
        masked_values = torch.where(mask, x, zeros)
        mask_f = mask.to(x.dtype)

        if len(dims_norm) > 0:
            x_flat = _flatten_dims(masked_values, dims_norm)
            c_flat = _flatten_dims(mask_f, dims_norm)
            total_sum = x_flat.sum(dim=-1, keepdim=True)
            total_cnt = c_flat.sum(dim=-1, keepdim=True)
            mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
            mean_v_full = _unflatten_result(mean_v, x.shape, dims_norm)

            # Compute std: diff^2 only on unmasked pixels
            diff_sq = (x - mean_v_full) ** 2
            var_sum = torch.where(mask, diff_sq, zeros)
            d_flat = _flatten_dims(var_sum, dims_norm)
            var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(total_cnt, 1.0)
            std_v_full = _unflatten_result(
                torch.sqrt(torch.clamp_min(var, 0.0)), x.shape, dims_norm
            )
        else:
            cnt = mask_f.sum()
            mean_scalar_val = (masked_values.sum() / max(cnt.item(), 1.0)).item()
            mean_v_full = torch.full_like(x, mean_scalar_val)
            diff_sq = (x - mean_v_full) ** 2
            var = torch.where(mask, diff_sq, zeros).sum() / max(cnt.item(), 1.0)
            std_scalar = math.sqrt(max(var.item(), 0.0))
            std_v_full = torch.full_like(x, std_scalar)

        new_mask = (x >= mean_v_full - n_sigma * std_v_full) & (
            x <= mean_v_full + n_sigma * std_v_full
        )
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    # Fill clipped values
    if fill == "mean":
        zeros = torch.zeros_like(x)
        masked_values = torch.where(mask, x, zeros)
        mask_f = mask.to(x.dtype)
        if len(dims_norm) > 0:
            xf = _flatten_dims(masked_values, dims_norm)
            cf = _flatten_dims(mask_f, dims_norm)
            fill_val = _unflatten_result(
                xf.sum(dim=-1, keepdim=True)
                / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                x.shape,
                dims_norm,
            )
        else:
            cnt = mask_f.sum()
            fill_val = masked_values.sum() / max(cnt.item(), 1.0)
    else:
        fill_val = _median(
            torch.where(
                mask,
                x,
                torch.tensor(float("inf"), device=x.device, dtype=x.dtype),
            ),
            dims_norm if dims_norm else (-1,),
        )
        fill_val = torch.where(
            torch.isinf(fill_val), torch.zeros_like(fill_val), fill_val
        )

    return torch.where(mask, x, fill_val)
