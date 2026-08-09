from __future__ import annotations

import math
from typing import Tuple

import torch

from .base import FITSTransform
from .helpers import (
    _normalize_dims,
    _get_valid_mask,
    _flatten_dims,
    _unflatten_result,
    _median,
    estimate_background,
)


class SigmaClip(FITSTransform):
    """Iterative sigma-clipping outlier rejection.

    Iteratively computes the mean and standard deviation over *dim*,
    masks values outside ``[mean - n_sigma*std, mean + n_sigma*std]``,
    and replaces them with the final mean.  Stops when no new pixels
    are clipped or *max_iter* is reached.

    ``inverse`` is not available — clipped values are irrecoverable.

    Parameters
    ----------
    n_sigma : float
        Number of standard deviations for the clipping threshold.
    max_iter : int
        Maximum number of clipping iterations.
    dim :
        Dimensions along which stats are computed independently.
    fill : str
        Replacement strategy: ``"mean"`` (default) or ``"median"``.
    """

    def __init__(
        self,
        n_sigma: float = 3.0,
        max_iter: int = 5,
        dim: Tuple[int, ...] = (-2, -1),
        fill: str = "mean",
    ) -> None:
        self.n_sigma = float(n_sigma)
        self.max_iter = int(max_iter)
        self.dim = tuple(dim)
        if fill not in ("mean", "median"):
            raise ValueError("fill must be 'mean' or 'median'")
        self.fill = fill
        self._last_mask: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Iteratively sigma-clip outliers and fill with mean or median.

        Optimised to minimise per-iteration allocations: uses a single
        pre-allocated masked-copy buffer and in-place arithmetic instead
        of allocating fresh ``torch.zeros_like`` / ``torch.where`` tensors
        each iteration.
        """
        ndim = x.ndim
        dims: tuple[int, ...] = ()
        if len(self.dim) > 0:
            dims = _normalize_dims(ndim, self.dim)

        with torch.no_grad():
            # Integer images promote to float (like astropy.sigma_clip);
            # UInt16/32/64 additionally lack reduction kernels in torch.
            if not x.dtype.is_floating_point:
                x = x.float() if x.dtype != torch.int64 else x.double()
            # Combine user mask with NaN mask if provided.
            if mask is not None:
                internal_mask = _get_valid_mask(x, mask)
            else:
                internal_mask = torch.ones_like(x, dtype=torch.bool)
            # Pre-allocate working buffers for masked values and zeros
            # to avoid per-iteration torch.zeros_like allocations.
            masked_buf = x.clone()
            zero = x.new_zeros(())
            for _ in range(self.max_iter):
                # Zero out masked-out positions, sum, and count.
                torch.where(internal_mask, x, zero, out=masked_buf)
                mask_f = internal_mask.to(x.dtype)

                if len(dims) > 0:
                    x_flat = _flatten_dims(masked_buf, dims)
                    c_flat = _flatten_dims(mask_f, dims)
                    total_sum = x_flat.sum(dim=-1, keepdim=True)
                    total_cnt = c_flat.sum(dim=-1, keepdim=True)
                    mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
                    mean_v_full = _unflatten_result(mean_v, x.shape, dims)
                    # Compute variance using the same buffer
                    masked_buf.sub_(mean_v_full).pow_(2)
                    torch.where(internal_mask, masked_buf, zero, out=masked_buf)
                    d_flat = _flatten_dims(masked_buf, dims)
                    var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(
                        total_cnt, 1.0
                    )
                    std_v_full = _unflatten_result(
                        torch.sqrt(torch.clamp_min(var, 0.0)), x.shape, dims
                    )
                else:
                    cnt = mask_f.sum()
                    mean_scalar = (masked_buf.sum() / max(cnt.item(), 1.0)).item()
                    mean_v_full = x.new_full(x.shape, mean_scalar)
                    masked_buf.sub_(mean_scalar).pow_(2)
                    torch.where(internal_mask, masked_buf, zero, out=masked_buf)
                    var = masked_buf.sum() / max(cnt.item(), 1.0)
                    std_scalar = math.sqrt(max(var.item(), 0.0))
                    std_v_full = x.new_full(x.shape, std_scalar)

                new_mask = (x >= mean_v_full - self.n_sigma * std_v_full) & (
                    x <= mean_v_full + self.n_sigma * std_v_full
                )
                new_mask = new_mask & internal_mask
                if torch.equal(new_mask, internal_mask):
                    break
                internal_mask = new_mask

            self._last_mask = internal_mask

            # Fill clipped values with per-group mean or median
            if self.fill == "mean":
                torch.where(internal_mask, x, zero, out=masked_buf)
                mask_f = internal_mask.to(x.dtype)
                if len(dims) > 0:
                    xf = _flatten_dims(masked_buf, dims)
                    cf = _flatten_dims(mask_f, dims)
                    fill_val = _unflatten_result(
                        xf.sum(dim=-1, keepdim=True)
                        / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                        x.shape,
                        dims,
                    )
                else:
                    cnt = mask_f.sum()
                    fill_val = masked_buf.sum() / max(cnt.item(), 1.0)
            else:
                # Median fill: use the existing _median helper; masked
                # positions are excluded by the mask (no inf sentinel).
                fill_val = _median(
                    x,
                    dims if dims else (-1,),
                    mask=internal_mask,
                )
                # Replace inf (all-masked groups) with 0
                fill_val = torch.where(
                    torch.isinf(fill_val), torch.zeros_like(fill_val), fill_val
                )

            return torch.where(internal_mask, x, fill_val)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise RuntimeError(
            "SigmaClip.inverse() is not available — clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"SigmaClip(n_sigma={self.n_sigma}, max_iter={self.max_iter}, "
            f"dim={self.dim}, fill={self.fill!r})"
        )


class AsymmetricSigmaClip(FITSTransform):
    """Simple one-pass asymmetric sigma-clipping outlier rejection.

    Computes per-group median and MAD-derived std (:func:`estimate_background`),
    then replaces values outside ``[median - n_low*std, median + n_high*std]``
    with the per-group median.  Non-iterative — faster and simpler than the
    full :class:`SigmaClip`, and supports different thresholds for the lower
    and upper tails.

    ``inverse`` is not available — clipped values are irrecoverable.

    Parameters
    ----------
    n_low : float
        Number of std deviations below median to clip (default 3.0).
        Set higher to preserve more faint pixels.
    n_high : float
        Number of std deviations above median to clip (default 3.0).
        Set higher to preserve more bright pixels.
    dim :
        Dimensions along which stats are computed independently.
        Default ``(-2, -1)`` for per-image clipping.

    Examples
    --------
    >>> # Clip negative outliers aggressively, preserve bright sources
    >>> clip = AsymmetricSigmaClip(n_low=5.0, n_high=3.0)
    >>>
    >>> # Per-spectrum clipping along the spectral axis
    >>> clip = AsymmetricSigmaClip(n_low=2.5, n_high=2.5, dim=(-1,))
    """

    def __init__(
        self,
        n_low: float = 3.0,
        n_high: float = 3.0,
        dim: Tuple[int, ...] = (-2, -1),
    ) -> None:
        if n_low <= 0 or n_high <= 0:
            raise ValueError("n_low and n_high must be > 0")
        self.n_low = float(n_low)
        self.n_high = float(n_high)
        self.dim = tuple(dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            med, std = estimate_background(x, dim=self.dim, mask=mask)
            lower = med - self.n_low * std
            upper = med + self.n_high * std
            clip_mask = (x >= lower) & (x <= upper)
            return torch.where(clip_mask, x, med)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise RuntimeError(
            "AsymmetricSigmaClip.inverse() is not available — "
            "clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"AsymmetricSigmaClip(n_low={self.n_low}, "
            f"n_high={self.n_high}, dim={self.dim})"
        )


# ---------------------------------------------------------------------------
# Header-aware normalization
# ---------------------------------------------------------------------------
