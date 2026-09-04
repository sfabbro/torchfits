from __future__ import annotations

from typing import Callable, Tuple

import torch


def _normalize_dims(ndim: int, dim: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert negative dims to positive and return sorted unique dims."""
    return tuple(sorted({d if d >= 0 else ndim + d for d in dim}))


def _stats_upcast(x: torch.Tensor) -> torch.Tensor:
    """Promote integer inputs for stats reductions, like astropy does.

    FITS subsets of BZERO-scaled data come back as UInt16, and torch ships
    no reduction kernels for the uint16/32/64 line; other integer dtypes
    reduce but produce integer stats that break downstream arithmetic
    (min/max sentinels, division). Float32 is the stats dtype; int64 keeps
    precision as float64.
    """
    if x.dtype.is_floating_point:
        return x
    return x.float() if x.dtype != torch.int64 else x.double()


def _mask_fill(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Dtype-safe fill for masked positions in stats reductions."""
    if x.dtype.is_floating_point:
        value = {"amin": float("inf"), "amax": float("-inf"), "nan": float("nan")}[mode]
        return torch.tensor(value, dtype=x.dtype, device=x.device)
    if mode == "nan":
        raise ValueError("NaN sentinel requires a floating dtype; upcast first")
    info = torch.iinfo(x.dtype)
    return torch.tensor(
        info.max if mode == "amin" else info.min, dtype=x.dtype, device=x.device
    )


def _get_valid_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Combine an optional explicit mask with an implicit NaN mask.

    Returns a boolean tensor where ``True`` indicates a valid (non-NaN,
    non-masked) element.  When *mask* is ``None``, the result is simply
    ``~torch.isnan(x)``.
    """
    valid = ~torch.isnan(x)
    if mask is not None:
        valid = valid & mask.to(torch.bool)
    return valid


def _flatten_dims(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """Collapse *dims* (sorted, positive) into a single trailing dim."""
    ndim = x.ndim
    keep = [d for d in range(ndim) if d not in dims]
    x_moved = x.permute(*keep, *dims)
    return x_moved.reshape(*x_moved.shape[: len(keep)], -1)


def _unflatten_result(
    reduced: torch.Tensor, shape: tuple[int, ...], dims: tuple[int, ...]
) -> torch.Tensor:
    """Reshape a reduced tensor back to *shape* with *dims* set to 1."""
    shape_out = list(shape)
    for d in dims:
        shape_out[d] = 1
    return reduced.reshape(shape_out)


def _reduce_keepdim(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    func: Callable[[torch.Tensor, int, bool], torch.Tensor],
) -> torch.Tensor:
    """Reduce *x* over *dim* using *func* (single-dim reducer), keepdim."""
    ndim = x.ndim
    dims = _normalize_dims(ndim, dim)
    if len(dims) == 1:
        return func(x, dims[0], True)
    x_flat = _flatten_dims(x, dims)
    result = func(x_flat, -1, True)
    # Reshape back to original ndim with reduced dims set to 1
    shape_out = list(x.shape)
    for d in dims:
        shape_out[d] = 1
    return result.reshape(shape_out)


def _median(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware *interpolated* median over tuple dim.

    Uses ``torch.nanquantile(0.5)`` so even-sized groups interpolate between
    the two central samples (matching numpy/astropy) instead of
    ``torch.median``'s lower-middle element. Masked-out pixels and NaNs are
    excluded; an all-masked group yields NaN.
    """
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    if not x.dtype.is_floating_point:
        # NaN sentinel needs a float dtype; masked integer medians upcast.
        x = x.float() if x.dtype != torch.int64 else x.double()
        valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "nan"))

    def nan_median(t: torch.Tensor, d: int, keepdim: bool) -> torch.Tensor:
        return torch.nanquantile(t, 0.5, dim=d, keepdim=keepdim)

    return _reduce_keepdim(x_clean, dim, nan_median)


def _amin(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amin over tuple dim."""
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "amin"))
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.amin(t, dim=d, keepdim=k)
    )


def _amax(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amax over tuple dim."""
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "amax"))
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.amax(t, dim=d, keepdim=k)
    )


def _quantile(
    x: torch.Tensor,
    q: float,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.quantile over tuple dim."""
    x = _stats_upcast(x)
    if not x.dtype.is_floating_point:
        # torch.quantile only supports float dtypes.
        x = x.float() if x.dtype != torch.int64 else x.double()
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "nan"))
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.nanquantile(t, q, dim=d, keepdim=k)
    )


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------


def _upcast_for_precision(x: torch.Tensor, *, precision: str = "auto") -> torch.Tensor:
    """Upcast for numerical stability.

    ``precision="auto"`` (default): float32 stays float32 (sufficient for
    visualization stretches); float16/bfloat16 → float32; float64 unchanged.
    ``precision="float64"`` always upcasts non-float64 inputs to float64.
    """
    if precision not in ("auto", "float64"):
        raise ValueError("precision must be 'auto' or 'float64'")
    if x.dtype == torch.float64:
        return x
    if precision == "float64":
        return x.float() if x.device.type == "mps" else x.double()
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    if x.dtype == torch.float32:
        return x
    return x.float()


def safe_arcsinh(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``arcsinh(scale * x)`` with precision-aware upcasting.

    float16/bfloat16 inputs are computed in float32; float32 stays float32
    (``precision="float64"`` forces float64). Matches the LSST/SDSS asinh
    convention across large dynamic ranges without hidden dtype changes.
    """
    orig_dtype = x.dtype
    out = torch.arcsinh(_upcast_for_precision(x) * scale)
    return out.to(orig_dtype)


def safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute ``log(x)`` with a floor at *eps* to avoid -inf.

    Upcasts per :func:`_upcast_for_precision` (see ``precision=`` there).
    """
    orig_dtype = x.dtype
    out = torch.log(torch.clamp_min(_upcast_for_precision(x), eps))
    return out.to(orig_dtype)


def estimate_background(
    x: torch.Tensor,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust background estimator: median and MAD-based dispersion.

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any NaN values) are excluded from the
        median and MAD computation.

    Returns
    -------
    med : Tensor
        Per-pixel-group median (keepdim=True).
    std_approx : Tensor
        MAD × 1.4826 ≈ standard deviation of the background.
    """
    with torch.no_grad():
        x = _stats_upcast(x)
        med = _median(x, dim, mask=mask)
        mad = _median(torch.abs(x - med), dim, mask=mask)
        std_approx = mad.mul_(1.4826)
    return med, std_approx


def zscale_limits(
    x: torch.Tensor,
    contrast: float = 0.25,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRAF-style zscale auto-contrast limits (fast proxy).

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any NaN values) are excluded from the
        median, MAD, min, and max computations.

    Returns (z1, z2) clipped to [vmin, vmax] with a fallback when the image
    is constant (z1 == z2).
    """
    with torch.no_grad():
        med, std = estimate_background(x, dim=dim, mask=mask)
        z1 = med - (std / max(contrast, 1e-5))
        z2 = med + (std / max(contrast, 1e-5))

        vmin = _amin(x, dim, mask=mask)
        vmax = _amax(x, dim, mask=mask)
        z1 = torch.where(std == 0, vmin, torch.maximum(z1, vmin))
        z2 = torch.where(
            std == 0, vmax, torch.minimum(z2, vmax)
        )  # Use a data-relative epsilon to avoid float32 underflow for large values.
        # 1e-6 relative guarantees >1 ULP margin even at float32 extremes.
        _eps = torch.maximum(
            torch.tensor(1e-6, device=x.device, dtype=z1.dtype), z1.abs() * 1e-6
        )
        z2 = torch.where(z1 == z2, z1 + _eps, z2)
    return z1, z2


# ---------------------------------------------------------------------------
# Base transform
# ---------------------------------------------------------------------------

__all__ = [
    "_normalize_dims",
    "_get_valid_mask",
    "_flatten_dims",
    "_unflatten_result",
    "_reduce_keepdim",
    "_median",
    "_amin",
    "_amax",
    "_quantile",
    "_upcast_for_precision",
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
