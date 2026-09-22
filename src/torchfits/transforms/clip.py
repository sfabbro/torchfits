from __future__ import annotations

import math
from typing import Any, Tuple

import torch

from .base import FITSTransform
from .helpers import (
    _ThreadedAttr,
    _flatten_dims,
    _get_valid_mask,
    _median,
    _normalize_dims,
    _stats_upcast,
    _unflatten_result,
    estimate_background,
)


class SigmaClip(FITSTransform):
    """Iterative sigma-clipping outlier rejection.

    Iteratively computes the mean and standard deviation over *dim*,
    masks values outside ``[mean - n_sigma*std, mean + n_sigma*std]``,
    and replaces them with the final mean.  Stops when no new pixels
    are clipped or *max_iter* is reached.

    ``inverse`` is not available — clipped values are irrecoverable.
    Replacement is nonlinear, so a companion ``ivar`` is passed through
    unchanged (and flagged with a warning).

    Parameters
    ----------
    n_sigma : float
        Number of standard deviations for the clipping threshold.
    max_iter : int
        Maximum number of clipping iterations.
    dim :
        Dimensions along which stats are computed independently.
    fill : str
        Replacement strategy for clipped/masked pixels: ``"mean"``,
        ``"median"``, or ``"nan"`` (keep the rejection visible as NaN
        instead of silently filling with a plausible background value).
    """

    propagates_ivar = False
    _last_mask = _ThreadedAttr()

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
        if fill not in ("mean", "median", "nan"):
            raise ValueError("fill must be 'mean', 'median', or 'nan'")
        self.fill = fill

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        """Iteratively sigma-clip outliers and fill with mean or median.

        Optimised to minimise per-iteration allocations: uses a single
        pre-allocated masked-copy buffer and in-place arithmetic instead
        of allocating fresh ``torch.zeros_like`` / ``torch.where`` tensors
        each iteration.
        """
        view = self.view(x)
        if view.ivar is not None:
            self._warn_ivar_not_propagated()
        data = view.flux
        effective = view.effective_mask(mask)
        ndim = data.ndim
        dims: tuple[int, ...] = ()
        if len(self.dim) > 0:
            dims = _normalize_dims(ndim, self.dim)
        # Integer images promote to float (like astropy.sigma_clip);
        # UInt16/32/64 additionally lack reduction kernels in torch;
        # complex inputs are rejected before any cast. Done outside
        # ``no_grad`` so the promoted tensor is still the one autograd sees
        # (integers never require grad anyway).
        data = _stats_upcast(data)

        # Statistics and replacements are constants: compute them without
        # tracking gradients. The final selection below stays OUTSIDE the
        # ``no_grad`` block so the kept pixels remain differentiable.
        with torch.no_grad():
            # Seed with the valid-element mask: excludes user-masked
            # positions AND non-finite values even without an explicit mask,
            # so a single NaN cannot poison mean/std into wiping the frame.
            internal_mask = _get_valid_mask(data, effective)
            # Pre-allocate working buffers for masked values and zeros
            # to avoid per-iteration torch.zeros_like allocations.
            masked_buf = data.clone()
            zero = data.new_zeros(())
            for _ in range(self.max_iter):
                # Zero out masked-out positions, sum, and count.
                torch.where(internal_mask, data, zero, out=masked_buf)
                mask_f = internal_mask.to(data.dtype)

                if len(dims) > 0:
                    x_flat = _flatten_dims(masked_buf, dims)
                    c_flat = _flatten_dims(mask_f, dims)
                    total_sum = x_flat.sum(dim=-1, keepdim=True)
                    total_cnt = c_flat.sum(dim=-1, keepdim=True)
                    mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
                    mean_v_full = _unflatten_result(mean_v, data.shape, dims)
                    # Compute variance using the same buffer
                    masked_buf.sub_(mean_v_full).pow_(2)
                    torch.where(internal_mask, masked_buf, zero, out=masked_buf)
                    d_flat = _flatten_dims(masked_buf, dims)
                    var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(
                        total_cnt, 1.0
                    )
                    std_v_full = _unflatten_result(
                        torch.sqrt(torch.clamp_min(var, 0.0)), data.shape, dims
                    )
                else:
                    cnt = mask_f.sum()
                    mean_scalar = (masked_buf.sum() / max(cnt.item(), 1.0)).item()
                    mean_v_full = data.new_full(data.shape, mean_scalar)
                    masked_buf.sub_(mean_scalar).pow_(2)
                    torch.where(internal_mask, masked_buf, zero, out=masked_buf)
                    var = masked_buf.sum() / max(cnt.item(), 1.0)
                    std_scalar = math.sqrt(max(var.item(), 0.0))
                    std_v_full = data.new_full(data.shape, std_scalar)

                new_mask = (data >= mean_v_full - self.n_sigma * std_v_full) & (
                    data <= mean_v_full + self.n_sigma * std_v_full
                )
                new_mask = new_mask & internal_mask
                if torch.equal(new_mask, internal_mask):
                    break
                internal_mask = new_mask

            self._last_mask = internal_mask

            # Keep the rejection visible: clipped/masked positions become NaN
            # instead of a plausible background value (fill="nan").
            if self.fill == "nan":
                fill_val = data.new_full((), float("nan"))
            elif self.fill == "mean":
                # Fill clipped values with per-group mean
                torch.where(internal_mask, data, zero, out=masked_buf)
                mask_f = internal_mask.to(data.dtype)
                if len(dims) > 0:
                    xf = _flatten_dims(masked_buf, dims)
                    cf = _flatten_dims(mask_f, dims)
                    fill_val = _unflatten_result(
                        xf.sum(dim=-1, keepdim=True)
                        / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                        data.shape,
                        dims,
                    )
                else:
                    cnt = mask_f.sum()
                    fill_val = masked_buf.sum() / max(cnt.item(), 1.0)
            else:
                # Median fill: use the existing _median helper; masked
                # positions are excluded by the mask (no inf sentinel).
                # dim=() clips globally, so the fill must reduce globally
                # too — (-1,) would fill per-row while thresholds are global.
                fill_val = _median(
                    data,
                    dims if dims else tuple(range(data.ndim)),
                    mask=internal_mask,
                )
                # Replace non-finite fills (all-masked groups yield NaN from
                # nanmedian, never inf) with 0
                fill_val = torch.where(
                    torch.isfinite(fill_val), fill_val, torch.zeros_like(fill_val)
                )

        return view.replace(torch.where(internal_mask, data, fill_val))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
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
    fill : str
        Replacement for clipped pixels: ``"median"`` (default) or ``"nan"``
        to keep the rejection visible instead of filling with background.
    weighted : bool
        Use inverse-variance weighted background statistics when ``ivar``
        is present.

    Examples
    --------
    >>> # Clip negative outliers aggressively, preserve bright sources
    >>> clip = AsymmetricSigmaClip(n_low=5.0, n_high=3.0)
    >>>
    >>> # Per-spectrum clipping along the spectral axis
    >>> clip = AsymmetricSigmaClip(n_low=2.5, n_high=2.5, dim=(-1,))
    """

    propagates_ivar = False
    _last_mask = _ThreadedAttr()

    def __init__(
        self,
        n_low: float = 3.0,
        n_high: float = 3.0,
        dim: Tuple[int, ...] = (-2, -1),
        fill: str = "median",
        *,
        weighted: bool = False,
    ) -> None:
        if n_low <= 0 or n_high <= 0:
            raise ValueError("n_low and n_high must be > 0")
        if fill not in ("median", "nan"):
            raise ValueError("fill must be 'median' or 'nan'")
        self.n_low = float(n_low)
        self.n_high = float(n_high)
        self.dim = tuple(dim)
        self.fill = fill
        self.weighted = bool(weighted)

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        view = self.view(x)
        if view.ivar is not None:
            self._warn_ivar_not_propagated()
        # Same promotion as SigmaClip: integer images promote to float and
        # complex inputs are rejected before any fill sentinel is built.
        data = _stats_upcast(view.flux)
        # Thresholds are constants; the selection stays outside ``no_grad`` so
        # kept pixels keep their gradient (consistent with the normalizers).
        with torch.no_grad():
            med, std = estimate_background(
                data,
                dim=self.dim,
                mask=view.effective_mask(mask),
                ivar=view.ivar,
                weighted=self.weighted,
            )
            lower = med - self.n_low * std
            upper = med + self.n_high * std
            clip_mask = (data >= lower) & (data <= upper)
            # True = kept pixel (same convention as SigmaClip._last_mask).
            self._last_mask = clip_mask
            fill_val = data.new_full((), float("nan")) if self.fill == "nan" else med
        return view.replace(torch.where(clip_mask, data, fill_val))

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise RuntimeError(
            "AsymmetricSigmaClip.inverse() is not available — "
            "clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"AsymmetricSigmaClip(n_low={self.n_low}, "
            f"n_high={self.n_high}, dim={self.dim}, fill={self.fill!r}, "
            f"weighted={self.weighted})"
        )
