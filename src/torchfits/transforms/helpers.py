from __future__ import annotations

import threading
import weakref
from typing import Any, Callable, Tuple

import torch


class _ThreadedAttr:
    """Data descriptor routing ``self._x = v`` to per-(instance, thread) storage.

    Transforms cache their ``inverse()`` statistics in ``_last_*`` attributes
    during ``forward()``. Plain instance attributes make one instance unsafe
    under concurrent callers: the statistics a thread's ``inverse()`` needs get
    clobbered by another thread's ``forward()``, silently restoring the wrong
    values. This descriptor keeps the attribute contract (reads and writes look
    identical; the calling thread sees its own latest ``forward()``) while the
    instance ``__dict__`` is never mutated and no state is shared across
    threads.

    Only for call-time caches: a value written in ``__init__`` would be visible
    only on the constructing thread, so constructor-derived constants must stay
    plain instance attributes. The cache is intentionally not pickled — an
    unpickled instance behaves like a fresh one (``inverse()`` raises until the
    next ``forward()``). Assigning ``None`` clears the slot.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _table(self) -> "weakref.WeakKeyDictionary[Any, Any]":
        table = getattr(self._local, "table", None)
        if table is None:
            table = weakref.WeakKeyDictionary()
            self._local.table = table
        return table

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        return self._table().get(obj)

    def __set__(self, obj: Any, value: Any) -> None:
        if value is None:
            self._table().pop(obj, None)
        else:
            self._table()[obj] = value


def _normalize_dims(ndim: int, dim: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert negative dims to positive and return sorted unique dims.

    Raises ValueError for dims outside ``[-ndim, ndim)``: an unchecked
    too-negative dim wraps forward (``-ndim - 1`` reduces the *last* axis
    instead of failing) and an unchecked too-large dim dies deep inside a
    torch kernel with an unrelated error.
    """
    out: set[int] = set()
    for d in dim:
        norm = d if d >= 0 else ndim + d
        if not 0 <= norm < ndim:
            raise ValueError(
                f"dim {d} out of range for {ndim}-D input (valid: {-ndim}..{ndim - 1})"
            )
        out.add(norm)
    return tuple(sorted(out))


def _stats_upcast(x: torch.Tensor) -> torch.Tensor:
    """Promote the input to the dtype stats reductions compute in.

    Complex dtypes are rejected: ordering-based statistics are undefined on
    them and silently dropping the imaginary part is data loss. float16 and
    bfloat16 promote to float32 (``torch.quantile`` rejects them and f16
    thresholds underflow the eps floors used across the normalizers).
    Integers promote like astropy: FITS subsets of BZERO-scaled data come
    back as UInt16, and torch ships no reduction kernels for the
    uint16/32/64 line; other integer dtypes reduce but produce integer stats
    that break downstream arithmetic (min/max sentinels, division). Float32
    is the stats dtype; int64 keeps precision as float64.
    """
    if x.dtype.is_complex:
        raise TypeError(
            f"stats transforms do not support complex dtypes (got {x.dtype}); "
            "reduce to a real component first"
        )
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
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
    """Combine an optional explicit mask with an implicit non-finite mask.

    Returns a boolean tensor where ``True`` indicates a valid (finite,
    non-masked) element. NaN *and* ±inf are invalid: an infinite pixel is
    not a measurement of the background, and letting one count as valid
    poisons mean/std into wiping the frame (and matches the ``isfinite``
    convention already used by the weighted stats and IRAF zscale paths).
    """
    valid = torch.isfinite(x)
    if mask is not None:
        valid = valid & mask.to(torch.bool)
    return valid


def _flatten_dims(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """Collapse *dims* (sorted, positive) into a single trailing dim."""
    ndim = x.ndim
    keep = [d for d in range(ndim) if d not in dims]
    x_moved = x.permute(*keep, *dims)
    folded = 1
    for d in dims:
        folded *= x.shape[d]
    return x_moved.reshape(*x_moved.shape[: len(keep)], folded)


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
    *,
    empty_fill: float = float("nan"),
) -> torch.Tensor:
    """Reduce *x* over *dim* using *func* (single-dim reducer), keepdim.

    An empty reduction (zero-size tensor or empty reduced dims) yields
    *empty_fill* with the reduced shape instead of a torch kernel error —
    NaN matches the all-masked group result for medians/quantiles; the
    min/max helpers pass their ±inf sentinels instead.
    """
    ndim = x.ndim
    dims = _normalize_dims(ndim, dim)
    shape_out = list(x.shape)
    for d in dims:
        shape_out[d] = 1
    if x.numel() == 0:
        return torch.full(shape_out, empty_fill, dtype=x.dtype, device=x.device)
    if len(dims) == 1:
        return func(x, dims[0], True)
    x_flat = _flatten_dims(x, dims)
    result = func(x_flat, -1, True)
    # Reshape back to original ndim with reduced dims set to 1
    return result.reshape(shape_out)


def _median(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware *interpolated* median over tuple dim.

    Uses ``torch.nanquantile(0.5)`` so even-sized groups interpolate between
    the two central samples (matching numpy/astropy) instead of
    ``torch.median``'s lower-middle element. Masked-out pixels and non-finite
    values (NaN and ±inf) are excluded; an all-masked or empty group yields
    NaN.
    """
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x = torch.where(valid, x, _mask_fill(x, "nan"))

    def nan_median(t: torch.Tensor, d: int, keepdim: bool) -> torch.Tensor:
        return torch.nanquantile(t, 0.5, dim=d, keepdim=keepdim)

    return _reduce_keepdim(x, dim, nan_median)


def _amin(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amin over tuple dim.

    All-masked and empty groups yield +inf (the masked-fill sentinel), which
    callers treat as "no valid pixels".
    """
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "amin"))
    return _reduce_keepdim(
        x_clean,
        dim,
        lambda t, d, k: torch.amin(t, dim=d, keepdim=k),
        empty_fill=float("inf"),
    )


def _amax(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amax over tuple dim.

    All-masked and empty groups yield -inf (the masked-fill sentinel), which
    callers treat as "no valid pixels".
    """
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(valid, x, _mask_fill(x, "amax"))
    return _reduce_keepdim(
        x_clean,
        dim,
        lambda t, d, k: torch.amax(t, dim=d, keepdim=k),
        empty_fill=float("-inf"),
    )


def _quantile(
    x: torch.Tensor,
    q: float,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.quantile over tuple dim.

    Masked-out pixels and non-finite values are excluded; an all-masked or
    empty group yields NaN.
    """
    x = _stats_upcast(x)
    valid = _get_valid_mask(x, mask)
    x = torch.where(valid, x, _mask_fill(x, "nan"))
    return _reduce_keepdim(
        x, dim, lambda t, d, k: torch.nanquantile(t, q, dim=d, keepdim=k)
    )


# ---------------------------------------------------------------------------
# Weighted (inverse-variance) statistics — opt-in
# ---------------------------------------------------------------------------


def _weight_flat(
    x_flat: torch.Tensor,
    *,
    dims: Tuple[int, ...],
    ivar: torch.Tensor | None,
) -> torch.Tensor:
    """Per-element weights on the flattened reduction plane."""
    if ivar is None:
        return torch.ones_like(x_flat)
    w = _flatten_dims(_stats_upcast(ivar).to(x_flat.dtype), dims).to(x_flat.dtype)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    return w.clamp_min(0.0)


def _weighted_quantile(
    x: torch.Tensor,
    q: float,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
    ivar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse-variance weighted quantile over tuple *dim*.

    Uses the weighted inverted-CDF definition: the value at the first index
    where the cumulative weight reaches ``q * total_weight``. With uniform
    weights this returns the lower-middle element for even-sized groups —
    deliberately *not* the interpolated median used by :func:`_median`, so
    ``weighted=True`` is opt-in and documented as such.

    Non-finite (NaN/±inf), masked and non-positive-weight samples get zero
    weight; a group with zero total weight — all invalid, all zero-weight, or
    empty — yields NaN.
    """
    xf = _stats_upcast(x)
    dims = _normalize_dims(xf.ndim, dim)
    flat = _flatten_dims(xf, dims)
    if flat.numel() == 0:
        return _unflatten_result(
            torch.full(
                (*flat.shape[:-1], 1), float("nan"), dtype=xf.dtype, device=xf.device
            ),
            xf.shape,
            dims,
        )
    valid = torch.isfinite(flat)
    if mask is not None:
        valid = valid & _flatten_dims(mask.to(torch.bool), dims)
    w = _weight_flat(flat, dims=dims, ivar=ivar)
    w = torch.where(valid, w, torch.zeros_like(w))
    values = torch.where(valid, flat, torch.full_like(flat, float("nan")))

    sorted_values, order = torch.sort(values, dim=-1)  # NaNs sort last
    sorted_w = torch.gather(w, -1, order)
    cumulative = torch.cumsum(sorted_w, dim=-1)
    total = cumulative[..., -1:]
    target = float(q) * total
    reached = cumulative >= target
    # argmax returns the first True; all-False (empty group) falls back to 0
    # and is overwritten with NaN below.
    index = torch.argmax(reached.to(torch.int8), dim=-1, keepdim=True)
    picked = torch.gather(sorted_values, -1, index)
    empty = total <= 0
    picked = torch.where(empty, torch.full_like(picked, float("nan")), picked)
    return _unflatten_result(picked, xf.shape, dims)


def _weighted_dispersion(
    x: torch.Tensor,
    center: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
    ivar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted RMS about *center*: ``sqrt(sum w d^2 / sum w)``.

    Zero total weight (all invalid, all zero-weight, or empty groups) yields
    NaN; a constant group yields 0.
    """
    xf = _stats_upcast(x)
    dims = _normalize_dims(xf.ndim, dim)
    flat = _flatten_dims(xf, dims)
    valid = torch.isfinite(flat)
    if mask is not None:
        valid = valid & _flatten_dims(mask.to(torch.bool), dims)
    w = _weight_flat(flat, dims=dims, ivar=ivar)
    w = torch.where(valid, w, torch.zeros_like(w))
    c = _flatten_dims(center.to(xf.dtype), dims)
    d2 = torch.where(valid, (flat - c) ** 2, torch.zeros_like(flat))
    num = _unflatten_result((w * d2).sum(dim=-1, keepdim=True), xf.shape, dims)
    den = _unflatten_result(w.sum(dim=-1, keepdim=True), xf.shape, dims)
    var = torch.where(
        den > 0,
        num / den.clamp_min(torch.finfo(xf.dtype).tiny),
        torch.full_like(num, float("nan")),
    )
    return torch.sqrt(torch.clamp_min(var, 0.0))


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------


def _upcast_for_precision(x: torch.Tensor, *, precision: str = "auto") -> torch.Tensor:
    """Upcast for numerical stability.

    ``precision="auto"`` (default): float32 stays float32 (sufficient for
    visualization stretches); float16/bfloat16 → float32; float64 unchanged.
    ``precision="float64"`` always upcasts non-float64 inputs to float64.
    Complex dtypes are rejected (silently dropping the imaginary part is
    data loss).
    """
    if x.dtype.is_complex:
        raise TypeError(
            f"stretch primitives do not support complex dtypes (got {x.dtype}); "
            "reduce to a real component first"
        )
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


def _stretch_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype a stretch should *return* for an input of dtype *x*.

    Float inputs keep their dtype; integer inputs promote to float (returning
    an integer would silently truncate the stretch to 0/1/2).
    """
    return x.dtype if x.dtype.is_floating_point else torch.float32


def safe_arcsinh(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``arcsinh(scale * x)`` with precision-aware upcasting.

    float16/bfloat16 inputs are computed in float32; float32 stays float32
    (``precision="float64"`` forces float64). Integer inputs return float32.
    Matches the LSST/SDSS asinh convention across large dynamic ranges
    without hidden dtype changes.
    """
    orig_dtype = _stretch_dtype(x)
    out = torch.arcsinh(_upcast_for_precision(x) * scale)
    return out.to(orig_dtype)


def safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute ``log(x)`` with a floor at *eps* to avoid -inf.

    Upcasts per :func:`_upcast_for_precision` (see ``precision=`` there).
    Integer inputs return float32.
    """
    orig_dtype = _stretch_dtype(x)
    out = torch.log(torch.clamp_min(_upcast_for_precision(x), eps))
    return out.to(orig_dtype)


def estimate_background(
    x: torch.Tensor,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
    *,
    ivar: torch.Tensor | None = None,
    weighted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust background estimator: median and MAD-based dispersion.

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any non-finite values) are excluded from the
        median and MAD computation.
    ivar :
        Optional inverse-variance companion. Only used when ``weighted=True``.
    weighted :
        Use inverse-variance weighted statistics instead of the default
        interpolated median / MAD. Off by default so outputs stay identical
        to the unweighted path unless explicitly requested, and a **no-op when
        no ``ivar`` is supplied** (there is nothing to weight by). Note that
        the weighted median uses the inverted-CDF definition, so with uniform
        weights it returns the lower-middle element rather than the
        interpolated median — the two paths agree to within one order-statistic
        spacing, not bit-for-bit. Zero-information groups are defined: an
        empty group, a fully masked group, or a group with only non-positive /
        non-finite weights (e.g. ``ivar = inf``) yields ``(NaN, NaN)``; a
        constant group yields ``(c, 0)``.

    Returns
    -------
    med : Tensor
        Per-pixel-group median (keepdim=True).
    std_approx : Tensor
        MAD × 1.4826 ≈ standard deviation of the background (unweighted), or
        the inverse-variance weighted RMS about the median (weighted).
    """
    with torch.no_grad():
        x = _stats_upcast(x)
        if weighted and ivar is not None:
            med = _weighted_quantile(x, 0.5, dim, mask=mask, ivar=ivar)
            std_approx = _weighted_dispersion(x, med, dim, mask=mask, ivar=ivar)
            return med, std_approx
        med = _median(x, dim, mask=mask)
        mad = _median(torch.abs(x - med), dim, mask=mask)
        std_approx = mad.mul_(1.4826)
    return med, std_approx


# ---------------------------------------------------------------------------
# IRAF zscale (faithful port of astropy.visualization.ZScaleInterval)
# ---------------------------------------------------------------------------


def _dilate_or(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Boolean dilation matching ``np.convolve(mask, ones(width), "same")``.

    numpy's ``same`` mode takes the central window, which is left-biased for
    even widths; replicated here so the port matches astropy exactly.
    """
    if width <= 1:
        return mask
    n = mask.shape[-1]
    out = torch.zeros_like(mask)
    off0 = -width + 1 + (width - 1) // 2
    for t in range(width):
        shift = off0 + t
        if shift == 0:
            out |= mask
        elif shift < 0:
            # out[i] |= mask[i + shift] for every in-range i; shift < 0 means
            # sampling from the right, so the *left* edge of the output is
            # undefined and the tail of the input is unused.
            out[..., -shift:] |= mask[..., : n + shift]
        else:
            out[..., : n - shift] |= mask[..., shift:]
    return out


def _iraf_zscale_groups(
    samples: torch.Tensor,
    counts: torch.Tensor,
    *,
    contrast: float,
    max_reject: float,
    min_npixels: int,
    krej: float,
    max_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IRAF zscale limits for a batch of pre-sorted sample rows.

    ``samples`` is ``(G, npix)`` ascending with invalid entries (NaN) last;
    ``counts`` is the number of valid entries per row.

    The fit runs in float64 (astropy's dtype) and the limits are cast back to
    the input dtype, so results match ``ZScaleInterval`` to float32 rounding.
    """
    out_dtype = samples.dtype
    samples = samples.to(torch.float64)
    groups, npix = samples.shape
    dtype = samples.dtype
    x = torch.arange(npix, dtype=dtype, device=samples.device).expand(groups, npix)

    valid = torch.isfinite(samples)
    badpix = ~valid
    ngood = valid.sum(dim=-1)
    last_ngood = ngood + 1
    minpix = torch.maximum(
        torch.full_like(ngood, int(min_npixels)),
        (counts.to(dtype) * float(max_reject)).floor().to(ngood.dtype),
    )
    ngrow = max(1, int(npix * 0.01))

    slope_final = torch.zeros(groups, dtype=dtype, device=samples.device)
    for _ in range(int(max_iterations)):
        cont = (ngood < last_ngood) & (ngood >= minpix)
        if not bool(cont.any()):
            break
        cf = cont.to(dtype).unsqueeze(-1)
        w = (~badpix).to(dtype) * cf
        sw = w.sum(dim=-1)
        swx = (w * x).sum(dim=-1)
        swy = (w * samples).sum(dim=-1)
        swxx = (w * x * x).sum(dim=-1)
        swxy = (w * x * samples).sum(dim=-1)
        denom = sw * swxx - swx * swx
        slope = torch.where(
            denom != 0,
            (sw * swxy - swx * swy)
            / torch.where(denom == 0, torch.ones_like(denom), denom),
            torch.zeros_like(denom),
        )
        intercept = torch.where(
            sw > 0,
            (swy - slope * swx) / torch.where(sw == 0, torch.ones_like(sw), sw),
            torch.zeros_like(sw),
        )
        fitted = slope.unsqueeze(-1) * x + intercept.unsqueeze(-1)
        flat = samples - fitted
        # k-sigma threshold from the surviving residuals (population std).
        good = (~badpix).to(dtype)
        sg = good.sum(dim=-1)
        mean_g = (flat * good).sum(dim=-1) / torch.where(
            sg == 0, torch.ones_like(sg), sg
        )
        var_g = (((flat - mean_g.unsqueeze(-1)) ** 2) * good).sum(dim=-1) / torch.where(
            sg == 0, torch.ones_like(sg), sg
        )
        threshold = float(krej) * torch.sqrt(torch.clamp_min(var_g, 0.0))
        newly_bad = (flat < -threshold.unsqueeze(-1)) | (flat > threshold.unsqueeze(-1))
        badpix = torch.where(cont.unsqueeze(-1), badpix | newly_bad, badpix)
        badpix = torch.where(cont.unsqueeze(-1), _dilate_or(badpix, ngrow), badpix)
        slope_final = torch.where(cont, slope, slope_final)
        last_ngood = torch.where(cont, ngood, last_ngood)
        ngood = torch.where(cont, (~badpix).sum(dim=-1), ngood)

    # Anchor limits: min/max of valid samples, tightened by the fitted slope.
    first = samples.gather(
        1, torch.zeros(groups, 1, dtype=torch.long, device=samples.device)
    ).squeeze(1)
    last_idx = (counts - 1).clamp_min(0).unsqueeze(1)
    last = samples.gather(1, last_idx).squeeze(1)
    vmin = first
    vmax = last

    can_adjust = ngood >= minpix
    adjusted = slope_final / contrast if contrast > 0 else slope_final
    center_pixel = (npix - 1) // 2
    masked = torch.where(valid, samples, torch.full_like(samples, float("nan")))
    median = torch.nanmedian(masked, dim=-1).values
    lo = median - (center_pixel - 1) * adjusted
    hi = median + (npix - center_pixel) * adjusted
    vmin = torch.where(can_adjust, torch.maximum(vmin, lo), vmin)
    vmax = torch.where(can_adjust, torch.minimum(vmax, hi), vmax)
    return vmin.to(out_dtype), vmax.to(out_dtype)


def zscale_limits(
    x: torch.Tensor,
    contrast: float = 0.25,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
    *,
    ivar: torch.Tensor | None = None,
    weighted: bool = False,
    algorithm: str = "proxy",
    n_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRAF-style zscale auto-contrast limits.

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any NaN values) are excluded from the
        median, MAD, min, and max computations.
    ivar, weighted :
        Opt-in inverse-variance weighting. Non-positive / non-finite weights
        are treated as invalid samples.
    algorithm :
        ``"proxy"`` (default) — median ± MAD/contrast, fast and differentiable.
        ``"iraf"`` — the iterative line-fit IRAF zscale algorithm, matching
        ``astropy.visualization.ZScaleInterval``.

    Returns (z1, z2) clipped to [vmin, vmax] with a fallback when the image
    is constant (z1 == z2).
    """
    if algorithm not in ("proxy", "iraf"):
        raise ValueError(f"algorithm must be 'proxy' or 'iraf', got {algorithm!r}")
    if algorithm == "iraf":
        return _zscale_iraf(
            x,
            contrast=contrast,
            dim=dim,
            mask=mask,
            ivar=ivar,
            weighted=weighted,
            n_samples=n_samples,
        )
    with torch.no_grad():
        med, std = estimate_background(
            x, dim=dim, mask=mask, ivar=ivar, weighted=weighted
        )
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


def _zscale_iraf(
    x: torch.Tensor,
    *,
    contrast: float,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None,
    ivar: torch.Tensor | None,
    weighted: bool,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized IRAF zscale over tuple *dim* (one fit per group)."""
    with torch.no_grad():
        xf = _stats_upcast(x)
        dims = _normalize_dims(xf.ndim, dim)
        flat = _flatten_dims(xf, dims)
        n = flat.shape[-1]
        groups = flat.numel() // max(n, 1)
        sample = flat.reshape(groups, n)
        valid = torch.isfinite(sample)
        if mask is not None:
            valid = valid & _flatten_dims(mask.to(torch.bool), dims).reshape(groups, n)
        if weighted and ivar is not None:
            w = _weight_flat(sample, dims=dims, ivar=ivar).reshape(groups, n)
            valid = valid & torch.isfinite(w) & (w > 0)
        sample = torch.where(valid, sample, torch.full_like(sample, float("nan")))
        # Stride the *original* order (as IRAF/astropy do) and only then sort;
        # striding the sorted array would sample quantiles instead.
        stride = int(max(1.0, n / float(n_samples)))
        sample = sample[:, ::stride][:, : int(n_samples)]
        sample, _ = torch.sort(sample, dim=-1)  # NaNs to the end
        counts = torch.isfinite(sample).sum(dim=-1)

        z1, z2 = _iraf_zscale_groups(
            sample,
            counts,
            contrast=float(contrast),
            max_reject=0.5,
            min_npixels=5,
            krej=2.5,
            max_iterations=5,
        )
        vmin = torch.nan_to_num(z1, nan=0.0)
        vmax = torch.nan_to_num(z2, nan=0.0)
        eps = torch.maximum(
            torch.tensor(1e-6, device=xf.device, dtype=xf.dtype),
            vmin.abs() * 1e-6,
        )
        vmax = torch.where(vmax <= vmin, vmin + eps, vmax)
        return (
            _unflatten_result(vmin.unsqueeze(-1), xf.shape, dims),
            _unflatten_result(vmax.unsqueeze(-1), xf.shape, dims),
        )


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
    "_weighted_quantile",
    "_weighted_dispersion",
    "_upcast_for_precision",
    "_stretch_dtype",
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
