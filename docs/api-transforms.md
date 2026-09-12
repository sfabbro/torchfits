# Transforms

Header-aware preprocessing for FITS images and tables.

## When to use

- High-dynamic-range **visualization** (arcsinh, zscale, log/sqrt stretches)
- **Model input** scaling that you want reusable across a Dataset
- FITS **BSCALE / null** hygiene (`FITSHeaderScale`, `TNullToNan`, column scale)

## When not to

- You need raw ADU / physical values as stored — use `read_tensor` / `table.read`
- One-off arithmetic on a single tensor — plain PyTorch is enough
- Catalog filtering — use `table.read(..., where=)` (C++ pushdown), not transforms

Wire a pipeline into training with `FitsTensorDataset(..., transform=pipeline)`
or call it on a tensor from `read_tensor`. See [Data module](api-data.md) for
when to introduce a Dataset / `make_loader`.

!!! note "RGB"
    Pretty auto RGB is `torchfits.transforms.rgb` (shortest wavelength first;
    also `torchfits convert … --to png`). Astropy-parity Lupton remains
    `lupton_rgb` (`--recipe lupton`).

All transforms implement the `FITSTransform` callable protocol
(`forward` / `inverse` / `__call__`). They are **not**
`torch.nn.Module` subclasses — wrap with :func:`as_module` (or
:class:`AsModule`) for `nn.Sequential`. Inverse state is
**instance-local** (`_last_*` fields); construct one pipeline per worker
when `num_workers > 0`.

```python
from torchfits.transforms import (
    ArcsinhStretch,
    BackgroundSubtract,
    Compose,
    ZScaleNormalize,
)

pipeline = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()])
normalized = pipeline(image)
restored = pipeline.inverse(normalized)
```

### Companions: `ivar`, `mask`, and data state

Every transform accepts a bare `torch.Tensor`, the
`{"flux", "ivar"?, "mask"?}` dict that `torchfits.data` emits when you pass
`ivar_hdu=` / `mask_hdu=`, or the typed `Payload` container. Built-ins read and
rebuild whichever container you handed them — a dict in gives a dict out, and
unknown dict keys (`wavelength`, `meta`, …) are carried through untouched.

**Inverse variance propagates exactly through affine transforms.** If flux
becomes `a·x + b`, then `ivar` becomes `ivar / a²` (an additive offset leaves
the variance alone). This holds for every normalizer, `BackgroundSubtract`,
`MeshBackgroundSubtract`, `AffineTransform`, the header-scaling transforms and
`SigmaClip`-free pipelines.

For the **nonlinear stretches** — `ArcsinhStretch`, `LogStretch`, `SqrtStretch` —
pass `propagate_ivar=True` to propagate `ivar` by the delta method
(`ivar / (df/dx)²`). For `SqrtStretch` on Poisson counts (`var(x) = x`) this
recovers the classic variance-stabilised constant `ivar = 4`. Where a stretch
clamps — `LogStretch`/`SqrtStretch` below zero — the map is locally flat and the
output no longer constrains the input, so `ivar` is set to `0` there instead of
a spuriously finite value.

Left at the default (`False`), a companion `ivar` is passed through unchanged
and a warning fires **once per instance**, because it no longer strictly
describes the transformed flux. The clipping transforms (`SigmaClip`,
`AsymmetricSigmaClip`, `PercentileClipNormalize`) are projections rather than
invertible maps, so they always pass `ivar` through and warn; the pixels they
altered are reported in their `_last_mask`.

**Data states.** `DataState` names the processing stage of a payload so a
transform can refuse data it would silently corrupt:

| State | Meaning | Produced by |
|---|---|---|
| `STORED` | Raw FITS storage codes, `BSCALE`/`BZERO` not applied | `read_tensor(..., raw_scale=True)` |
| `PHYSICAL` | Calibrated physical values | Default `read_tensor` / `table.read_torch` |
| `CONTINUUM_NORMALIZED` | Spectra divided by their continuum | Your pipeline |
| `NORMALIZED` | Dimensionless model input | The normalizers here |

Data read with the default reader is already `PHYSICAL`, so
`FITSHeaderScale` would scale it a **second** time. Declare the state in the
payload and the transform raises `DataStateError` instead:

```python
from torchfits.transforms import FITSHeaderScale

FITSHeaderScale(bscale=0.5)({"flux": counts, "state": "physical"})
# DataStateError: FITSHeaderScale expects state [stored], but the payload
# declares 'physical'.  The default reader returns *physical* values already
# (BSCALE/BZERO applied). Read with raw_scale=True for stored codes, or drop
# the header-scaling transform.
```

Bare tensors carry no declared state and are never rejected, so existing
pipelines keep working. `calibration_state(header, raw_scale=True)` returns the
state a reader actually produced, and `Payload(flux=..., ivar=..., mask=...,
state=...)` is the typed equivalent of the dict payload.

The state is **maintained**: a transform stamps what it produced onto any
payload that already declares a state, so `FITSHeaderScale` on a `STORED`
payload yields a `PHYSICAL` one and a second `FITSHeaderScale` in the same
pipeline raises instead of scaling twice. `inverse()` accepts what its own
`forward()` produced and restores the state it consumed, so
`t.inverse(t(payload))` round-trips. Payloads that never declared a state are
left exactly as they were (no `state` key is added).

Flux-scaling transforms (the normalizers, `BackgroundSubtract`,
`MeshBackgroundSubtract`, `FITSHeaderNormalize`) accept `STORED`, `PHYSICAL`
and `NORMALIZED` but **refuse `CONTINUUM_NORMALIZED`**: dividing an
already-continuum-normalized spectrum by another per-spectrum statistic
discards the common flux scale the normalization established. Clear the state
if you really mean to rescale such data.

### Masks

Most transforms accept an optional boolean `mask` (`True` = valid).
`Compose` forwards the same mask to every child:

```python
pipeline(image, mask=finite_mask)
pipeline.inverse(normalized, mask=finite_mask)
```
### Invertibility

| Kind | `inverse()` |
|---|---|
| Stretches, `AffineTransform`, `FITSHeaderScale`, `FITSScaleColumns` | Yes |
| Normalizers (`ZScale`, `Robust`, `MinMax`, `GlobalScalar`, `InterquantileScale`, `SigmaNormalize`, `BackgroundSubtract`, `MeshBackgroundSubtract`) | Yes (the affine coefficients are cached on the instance during `forward`) |
| `PercentileClipNormalize` | Approximate only — forward clips to `[lower, upper]`, so clipped pixels cannot be recovered exactly |
| `SigmaClip`, `AsymmetricSigmaClip`, `TNullToNan` | No — lossy / many-to-one |
| Any transform you instantiate but never call `forward()` on | Raises `RuntimeError` (there are no cached limits yet) |

!!! note "Integer inputs are promoted, never truncated"
    All stretches promote integer tensors to float32 and return float —
    `ArcsinhStretch`, `LogStretch` and `SqrtStretch` share one contract, so a
    `uint16` frame can be stretched directly without collapsing to `0/1/2`.
    Float64 input stays float64. Non-identity `FITSHeaderScale` /
    `FITSScaleColumns` also **return float** (physical values) rather than
    casting back to the storage dtype.

Stateless stretches are the most likely to work under `torch.compile`;
data-dependent normalizers cache Python-side state and may graph-break.
There is no certified compile matrix yet.

The implementation lives under `torchfits.transforms` as a small package
(`stretch`, `normalize`, `fits_meta`, `clip`, `rgb`) re-exported from
`torchfits.transforms`.

---

## Stretches

Stateless, with analytic inverses. `LogStretch` / `SqrtStretch` clamp
negative input to zero, so their inverse cannot recover negatives.

### `ArcsinhStretch(a=1.0)`

Lupton+ (2004) arcsinh stretch — LSST/SDSS standard.

$$\text{output} = \frac{\operatorname{arcsinh}(a \cdot x)}{\operatorname{arcsinh}(a)}$$

**Inverse:** $x = \frac{\sinh(\text{output} \cdot \operatorname{arcsinh}(a))}{a}$

| Param | Default | Description |
|---|---|---|
| `a` | `1.0` | Softening parameter — smaller values are more linear near zero |
| `propagate_ivar` | `False` | Delta-method propagation of a companion `ivar` |

!!! info "When to use"
    The default choice for astronomical image display. Handles the huge dynamic
    range of sky images gracefully — stars and galaxies both remain visible.

### `LogStretch(a=1000.0, eps=1e-9)`

Logarithmic stretch for heavy-tailed flux distributions. Negative input is
clamped to zero (inverse cannot recover negatives).

$$\text{output} = \frac{\log_{10}(1 + a \cdot \max(x, 0))}{\log_{10}(1 + a)}$$

**Inverse:** $x = \frac{10^{\text{output} \cdot \log_{10}(1 + a)} - 1}{a}$

| Param | Default | Description |
|---|---|---|
| `a` | `1000.0` | Scale factor before log — larger compresses low flux more gently |
| `eps` | `1e-9` | Floor to prevent log(0) |
| `propagate_ivar` | `False` | Delta-method propagation of a companion `ivar` |

!!! info "When to use"
    When you need stronger compression than arcsinh, e.g., for very wide-field
    images with bright stars and faint diffuse emission.

### `SqrtStretch()`

Square-root stretch — stabilizes Poisson variance. Negative input is clamped
to zero (inverse cannot recover negatives).

$$\text{output} = \sqrt{\max(x, 0)}$$

**Inverse:** $x = \text{output}^2$

| Param | Default | Description |
|---|---|---|
| `propagate_ivar` | `False` | Delta-method propagation of a companion `ivar`; for Poisson counts (`ivar = 1/x`) the stretched data then carry a near-constant `ivar ≈ 4` |

!!! info "When to use"
    Quick and simple. Good default for photon-counting data where Poisson
    statistics apply — especially with `propagate_ivar=True`, which makes the
    noise level uniform across the frame.

---

## Normalizers

Data-dependent — compute statistics from the image and cache them for
inverse transforms. Require float input; see [Invertibility](#invertibility).

### `ZScaleNormalize(contrast=0.25, dim=(-2, -1))`

IRAF zscale auto-contrast algorithm. Maps the display range `[z1, z2]` to
`[0, 1]` — pixels outside `[z1, z2]` land outside `[0, 1]` (no clamp on the
output).

$$z_1 = \text{median} - \frac{\text{MAD} \times 1.4826}{\max(\text{contrast}, 10^{-5})}$$

$$z_2 = \text{median} + \frac{\text{MAD} \times 1.4826}{\max(\text{contrast}, 10^{-5})}$$

Both clamped to $[x_{\min}, x_{\max}]$.

$$\text{output} = \frac{x - z_1}{z_2 - z_1}$$

**Inverse:** $x = \text{output} \cdot (z_2 - z_1) + z_1$

| Param | Default | Description |
|---|---|---|
| `contrast` | `0.25` | Controls how aggressively to trim outliers — smaller = tighter range |
| `dim` | `(-2, -1)` | Dimensions for statistics |
| `algorithm` | `"proxy"` | `"proxy"` = `median ± MAD/contrast` (fast). `"iraf"` = the iterative line-fit IRAF algorithm, matching `astropy.visualization.ZScaleInterval` |
| `weighted` | `False` | Use inverse-variance weighted limits when the payload carries `ivar` |

!!! info "When to use"
    The standard choice for astronomical image display. Automatically adapts to
    the dynamic range of the data. Use for quick visualization or when you want
    a [0, 1] normalized image that preserves relative contrast.

!!! note "Proxy vs IRAF limits"
    `algorithm="proxy"` is fast but is *not* IRAF's algorithm. Pass
    `algorithm="iraf"` for astropy-parity zscale limits (thin sampling,
    iterative `krej=2.5` rejection); it is still fully differentiable-free and
    runs on GPU, just slower. `zscale_limits(x, algorithm="iraf")` exposes the
    raw limits.

### `RobustNormalize(dim=(-2, -1))`

Subtract median, divide by MAD-derived standard deviation.

$$\text{output} = \frac{x - \text{median}(x)}{\max(\text{MAD} \times 1.4826,\ 10^{-9})}$$

where $\text{MAD} = \text{median}(|x - \text{median}(x)|)$.

**Inverse:** $x = \text{output} \cdot \text{std\_approx} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for statistics |
| `weighted` | `False` | Inverse-variance weighted median/MAD when `ivar` is present |

!!! info "When to use"
    Universal ML preprocessing. Produces zero-mean, unit-variance-like data
    robust to outliers. Use as the last step before feeding to a neural
    network.

### `SigmaNormalize(dim=(-2, -1), stat="mad", zero_preserving=True, eps=1e-12, weighted=False)`

Rescale so the sky noise is unit variance. This is the normalization the
image foundation-model papers actually use (AstroCLIP / AstroPT style
per-frame robust scaling), and unlike `RobustNormalize` it does **not** shift
the data by default:

$$\sigma = \text{MAD} \times 1.4826 \quad (\text{or the population RMS about the median for } \texttt{stat="std"})$$

Zero-preserving mode (default):

$$\text{output} = \frac{x}{\max(\sigma,\ \text{eps})}$$

Centered mode (`zero_preserving=False`):

$$\text{output} = \frac{x - \text{median}(x)}{\max(\sigma,\ \text{eps})}$$

**Inverse:** $x = \text{output} \times \sigma + \text{offset}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions over which sigma is estimated |
| `stat` | `"mad"` | `"mad"` (robust) or `"std"` (population RMS about the median) |
| `zero_preserving` | `True` | Divide without subtracting an offset, so zero flux and band ratios survive exactly |
| `eps` | `1e-12` | Divisor floor for constant groups |
| `weighted` | `False` | Inverse-variance weighted dispersion |

!!! info "When to use"
    The default image normalization for survey ML. Keep `zero_preserving=True`
    when colour ratios matter (photometric redshift, multiband classification);
    switch to centered mode when the downstream model expects zero-mean input.

### `BackgroundSubtract(dim=(-2, -1))`

Subtract median background level.

$$\text{output} = x - \text{median}(x)$$

**Inverse:** $x = \text{output} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for background estimation |
| `weighted` | `False` | Use inverse-variance weighted statistics when the payload carries `ivar` |

!!! info "When to use"
    First step in most image pipelines. Removes the constant sky background
    before stretching or normalization. If the sky is not flat across the
    frame, use `MeshBackgroundSubtract` instead.

### `MeshBackgroundSubtract(mesh=(8, 8), n_sigma=3.0, filter_mesh=True, min_tile_pixels=4, weighted=False)`

SExtractor-style tile-mesh background: estimate the sky independently in each
of `mesh` tiles (sigma-clipped median + MAD), optionally median-filter the tile
grid, then interpolate it back to full resolution and subtract.

$$\text{output} = x - \text{interp}\big(\text{med}_{\text{tile}}(x)\big)$$

The two trailing dimensions are spatial; leading dimensions are batched, so a
`[C, H, W]` stack or cube works directly. Tiles with fewer than
`min_tile_pixels` valid pixels fall back continuously to the frame-level
background.

**Inverse:** $x = \text{output} + \text{background}$ (the interpolated map is
cached from `forward`).

| Param | Default | Description |
|---|---|---|
| `mesh` | `(8, 8)` | Number of tiles along `(H, W)` |
| `n_sigma` | `3.0` | Clipping threshold for the per-tile statistics |
| `filter_mesh` | `True` | 3×3 median filter the tile grid (stops bright sources imprinting on the sky map) |
| `min_tile_pixels` | `4` | Below this, a tile falls back to the frame background |
| `weighted` | `False` | Inverse-variance weighted tile statistics when `ivar` is present |

!!! info "When to use"
    Wide-field images, mosaics and any frame with real sky gradients. It is
    the difference between a science-ready frame and one with a bright
    gradient baked in. It only *subtracts*, so `ivar` is unchanged.

### `PercentileClipNormalize(lower_pct=1, upper_pct=99, dim=(-2, -1))`

Clip to percentile range, scale to [0, 1]. `inverse()` is **approximate** —
values clipped by the clamp cannot be recovered exactly.

$$\text{lower} = Q_{\text{lower\_pct}/100}(x), \quad \text{upper} = Q_{\text{upper\_pct}/100}(x)$$

$$\text{output} = \frac{\text{clamp}(x,\ \text{lower},\ \text{upper}) - \text{lower}}{\text{upper} - \text{lower}}$$

**Inverse:** $x = \text{output} \cdot (\text{upper} - \text{lower}) + \text{lower}$

| Param | Default | Description |
|---|---|---|
| `lower_pct` | `1.0` | Lower percentile |
| `upper_pct` | `99.0` | Upper percentile |
| `dim` | `(-2, -1)` | Dimensions for quantile computation |

!!! info "When to use"
    More aggressive than zscale — hard-clips outliers. Good for display when
    you know the percentile range of "interesting" data.

### `MinMaxNormalize(dim=(-2, -1))`

Min-max normalization to [0, 1] with ULP-safe epsilon.

$$\text{output} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

**Inverse:** $x = \text{output} \cdot (v_{\max} - v_{\min}) + v_{\min}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for min/max |

!!! info "When to use"
    Simple normalization when you know the data has no outliers. Avoid for
    astronomical images — a single bright star dominates the range.

### `GlobalScalarNorm(stat="median", dim=None)`

Divide by a single scalar statistic. Minimal linear prep.

$$\text{output} = \frac{x}{\max(\text{scalar},\ 10^{-30})}$$

where scalar is one of: `median(x)`, `max(x)`, `mean(x)`, or
$\sqrt{\text{mean}(x^2)}$ (RMS).

**Inverse:** $x = \text{output} \times \text{scalar}$

| Param | Default | Description |
|---|---|---|
| `stat` | `"median"` | `"median"`, `"max"`, `"mean"`, or `"rms"` |
| `dim` | `None` | Dimensions (None = all) |

!!! info "When to use"
    When you want the simplest possible normalization — just scale by the
    typical value. Good for quick per-image scaling before comparison.

### `InterquantileScale(q_low=0.05, q_high=0.95, dim=None, zero_preserving=True, eps=1e-9)`

Zero-preserving or centered interquantile scale normalization.

$$s = \max(Q_{q_{\text{high}}}(x) - Q_{q_{\text{low}}}(x),\ \text{eps})$$

In zero-preserving mode (default):

$$\text{output} = \frac{x}{s}$$

In centered mode (`zero_preserving=False`):

$$\text{output} = \frac{x - \text{median}(x)}{s}$$

**Inverse:** $x = \text{output} \times s + \text{offset}$

| Param | Default | Description |
|---|---|---|
| `q_low` | `0.05` | Lower quantile (0.0 to 1.0) |
| `q_high` | `0.95` | Upper quantile (0.0 to 1.0) |
| `dim` | `None` | Dimensions for joint quantiles (None = all dims) |
| `zero_preserving` | `True` | Scale without subtracting offset (preserves colours and zero-point) |
| `eps` | `1e-9` | Divisor floor |

!!! info "When to use"
    Standard choice for multi-band astronomical images where relative colour
    ratios ($f_g / f_r$) must remain strictly invariant (e.g. photometric
    redshift models). Supports both plain tensors and `{"flux", "ivar"?, "mask"?}`
    companion dicts (where `ivar` scales by $s^2$).

### `AffineTransform(scale=1.0, offset=0.0)`

Explicit `flux * scale + offset` with exact companion propagation. It is the
linear building block every normalizer reduces to, and the right tool when you
already know the coefficients — e.g. converting counts to flux from a zeropoint:

```python
from torchfits.transforms import AffineTransform

zp = 25.0
counts_to_flux = AffineTransform(scale=10 ** (-0.4 * zp))
flux = counts_to_flux(payload)  # ivar scaled by scale**-2
```

$$\text{output} = \text{scale} \cdot x + \text{offset} \qquad
\text{ivar}_{\text{out}} = \frac{\text{ivar}}{\text{scale}^2}$$

**Inverse:** $x = (\text{output} - \text{offset}) / \text{scale}$

| Param | Default | Description |
|---|---|---|
| `scale` | `1.0` | Multiplicative factor; must be non-zero |
| `offset` | `0.0` | Additive factor (does not change `ivar`) |

---

## Outlier Rejection

### `SigmaClip(n_sigma=3.0, max_iter=5, dim=(-2,-1), fill="mean")`

Iterative sigma-clipping with mean or median fill.

1. Compute mean $\mu$ and std $\sigma$ over dims.
2. Mask pixels where $|x - \mu| > n_\sigma \cdot \sigma$.
3. Repeat until convergence or `max_iter`.
4. Replace clipped values with surviving mean or median.

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `n_sigma` | `3.0` | Clipping threshold |
| `max_iter` | `5` | Max iterations |
| `dim` | `(-2, -1)` | Dimensions for statistics |
| `fill` | `"mean"` | `"mean"` or `"median"` replacement |

!!! info "When to use"
    Standard for cleaning cosmic rays and hot pixels from images. Use
    `fill="median"` for more robust replacement in the presence of many
    outliers.

### `AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-2,-1))`

One-pass asymmetric sigma-clip via `estimate_background` (median + MAD).

$$\text{lower} = \text{med} - n_{\text{low}} \cdot \text{std}, \qquad \text{upper} = \text{med} + n_{\text{high}} \cdot \text{std}$$

Outliers replaced with median.

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `n_low` | `3.0` | Std below median to clip |
| `n_high` | `3.0` | Std above median to clip |
| `dim` | `(-2, -1)` | Dimensions for statistics |

!!! info "When to use"
    Faster than iterative sigma-clip. Use different `n_low`/`n_high` when
    the outlier distribution is asymmetric (e.g., bright stars are more
    common than dark holes).

---

## Masks & robust statistics

!!! warning "One mask convention everywhere"
    A torchfits mask is a boolean tensor where **`True` means valid**. FITS
    stores the opposite in two common shapes — a `DQ` integer extension where
    set bits flag defects, and an `IVAR` extension where `0` means "no data" —
    so convert before you combine.

| Function | Role |
|---|---|
| `mask_from_dq(dq, bad_bits=None, good_bits=None, require_good=False)` | `DQ` bitfield → validity mask. Bits are 0-based positions, so `bad_bits=[0, 11]` rejects DQ values `1` and `2048`; the default `bad_bits=None` treats any non-zero value as invalid |
| `mask_from_ivar(ivar, min_ivar=0.0)` | `ivar > min_ivar` and finite |
| `mask_from_nan(x)` | Finite values are valid |
| `combine_masks(*masks)` | Logical AND, skipping `None`; returns `None` when all are `None` |
| `apply_mask(x, mask, fill=nan)` | Replace invalid pixels with `fill` (promotes integers for a NaN fill) |

```python
from torchfits.transforms import apply_mask, combine_masks, mask_from_dq, mask_from_ivar

valid = combine_masks(
    mask_from_dq(payload["dq"], bad_bits=[0, 11]), mask_from_ivar(payload["ivar"])
)
flux = apply_mask(payload["flux"], valid)
```

Transform statistics are mask-aware and inverse-variance aware. Pass
`mask=` explicitly, or let a dict payload's `mask` field be picked up
automatically (an explicit `mask=` argument wins). `weighted=True` on the
normalizers, the clippers and `estimate_background` switches the underlying
median / mean / RMS to inverse-variance weights, so low-weight pixels stop
dragging the estimate, while the default (`weighted=False`) stays byte-identical
to previous releases.

`weighted=True` is an **exact no-op when no `ivar` is supplied** (there is
nothing to weight by), so it can be switched on unconditionally. Note the
weighted median uses the inverted-CDF definition, so with *uniform* weights it
returns the lower-middle element rather than the interpolated median; the two
agree to within one order-statistic spacing, not bit-for-bit.

```python
from torchfits.transforms import SigmaNormalize

clean = SigmaNormalize(weighted=True)({"flux": flux, "ivar": ivar, "mask": valid})
```

### Helpers

| Function | Role |
|---|---|
| `safe_arcsinh`, `safe_log` | Numerically stable stretch primitives |
| `estimate_background(x, dim, mask=None, ivar=None, weighted=False)` | Shared robust median/MAD estimator for normalizers and clippers |
| `zscale_limits(x, contrast=0.25, algorithm="proxy")` | IRAF-style zscale limit finder used by `ZScaleNormalize` |
| `_weighted_quantile` | Inverse-variance weighted quantiles behind `weighted=True` |

---

## FITS Metadata

### `FITSHeaderScale(bscale=1.0, bzero=0.0)`

Apply/remove FITS BSCALE/BZERO linear scaling.

$$\text{output} = \text{BSCALE} \cdot x + \text{BZERO}$$

**Inverse:** $x = \frac{\text{output} - \text{BZERO}}{\text{BSCALE}}$

| Param | Default | Description |
|---|---|---|
| `bscale` | `1.0` | FITS BSCALE keyword |
| `bzero` | `0.0` | FITS BZERO keyword |

Factory: `FITSHeaderScale.from_header(header)` — extracts BSCALE/BZERO from
a FITS header dict.

### `FITSScaleColumns(scales)`

Per-column BSCALE/BZERO for table tensors.

$$\text{output}[c] = \text{TSCAL}_c \cdot x[c] + \text{TZERO}_c$$

**Inverse:** $x[c] = \frac{\text{output}[c] - \text{TZERO}_c}{\text{TSCAL}_c}$

| Param | Default | Description |
|---|---|---|
| `scales` | *(required)* | `dict[str, (TSCAL, TZERO)]` |

Factory: `FITSScaleColumns.from_header(header)`.

### `TNullToNan(nulls)`

Map FITS TNULL sentinels to NaN in table columns.

$$\text{output}[i] = \begin{cases} \text{NaN} & \text{if } x[i] = \text{TNULL} \\ x[i] & \text{otherwise} \end{cases}$$

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `nulls` | *(required)* | `dict[str, TNULL_value]` |

Factory: `TNullToNan.from_header(header)`.

### `FITSHeaderNormalize(header, scale_floats=False)`

Auto-normalize from BITPIX/BSCALE/BZERO. Integer types mapped to [0, 1].

- **Integer types** (BITPIX 8/16/32/64): min-max normalize using the
  native range scaled by BSCALE/BZERO.
- **Float types** (BITPIX -32/-64): identity unless `scale_floats=True`.

**Inverse:** Yes (for normalized types).

| Param | Default | Description |
|---|---|---|
| `header` | *(required)* | FITS header dict |
| `scale_floats` | `False` | Also normalize float data |

---

## Utility

### `Compose(transforms)`

Chain transforms; `inverse()` unwinds in reverse order.

```python
pipeline = Compose(
    [
        BackgroundSubtract(),
        ArcsinhStretch(a=0.1),
        ZScaleNormalize(),
    ]
)
normalized = pipeline(image)
original = pipeline.inverse(normalized)
```
### `FITSTransform`

Base class for custom transforms. Override `forward()` and optionally
`inverse()`. `__call__` delegates to `forward()`. Not an `nn.Module`.

### `as_module(transform)` / `AsModule`

Wrap a `FITSTransform` as a thin `nn.Module` for `nn.Sequential`:

```python
import torch.nn as nn
from torchfits.transforms import ArcsinhStretch, as_module

model = nn.Sequential(as_module(ArcsinhStretch(a=0.1)), nn.Linear(64, 10))
```

Only the forward pass is exposed; call `transform.inverse` on the wrapped
instance for undo.

### `rgb(*bands, *, brightness=0.15, saturation=2.0, scene="auto", weights=None, calibrated=False, zeropoints=None)`

Auto RGB from 1–7 aligned images (or one `(C, H, W)` cube). Band order is
**shortest wavelength first**: `rgb(g, r, i)` or `rgb(u, g, r, i, z)`.
Returns a `[H, W, 3]` float tensor in `[0, 1]` after sRGB encoding.

Uncalibrated (default) subtracts each band's sky median and divides by MAD
so ADU scale and sky pedestals do not paint the colour. `calibrated=True`
skips that (bands already on one flux scale). `zeropoints=` is the AB
magnitude of 1 count per band, converted to nanomaggies as
`counts * 10**(-0.4*(zp - 22.5))`.

See `examples/example_rgb_sky.py`.

### `lupton_rgb(r, g, b, *, Q=8.0, stretch=0.5, minimum=0.0)`

Lupton asinh RGB from three single-band tensors (same shape). Returns a
`[H, W, 3]` **float64** tensor in `[0, 1]` (channel last; accepts any
tensor-convertible inputs). Matches Astropy's
`make_lupton_rgb` / `LuptonAsinhStretch` mapping. See
`examples/example_lupton_rgb_sdss.py`.

#### Writing a custom transform

Subclass `FITSTransform`, implement `forward()` (and `inverse()` if the
operation is invertible):

```python
import torch
from torchfits.transforms import FITSTransform


class ScaleOffset(FITSTransform):
    """Affine transform: forward(x) = x * scale + offset."""

    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x * self.scale + self.offset

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (x - self.offset) / self.scale
```

Compose it with built-in transforms — `Compose` calls each child with the
same tensor and `mask=` kwarg, and unwinds `inverse()` in reverse order:

```python
from torchfits.transforms import BackgroundSubtract, Compose

xf = ScaleOffset(scale=2.0, offset=-10.0)
pipeline = Compose([BackgroundSubtract(), xf])
out = pipeline(image)
restored = pipeline.inverse(out)
```

Wire it into a Dataset so every sample gets the same preprocessing:

```python
from torchfits.data import FitsImageDataset

dataset = FitsImageDataset("images/*.fits", hdu=0, transform=xf)
```

!!! note "Custom transforms and dict payloads"
    The transforms documented here unwrap a `Tensor`, a `{"flux", …}` dict or a
    `Payload` themselves, so a Dataset that emits companions works unchanged.
    A *custom* transform that starts from `x` as a bare tensor will not — and
    note the Dataset calls `transform(payload)` with one positional argument,
    it does **not** unpack `mask` into the `mask=` kwarg for you (see
    [Dataset `transform=` signature](api-data.md#choosing-a-dataset)).
    Subclasses can use the same plumbing the built-ins do:

    ```python
    def forward(self, x, mask=None):
        view = self.view(x)  # validates the declared data state
        flux = view.flux * self.scale + self.offset
        return view.replace(flux, ivar=self.scale_ivar(view.ivar, self.scale))
    ```

    `view.effective_mask(mask)` returns the explicit mask when given and the
    payload's own mask otherwise, and `view.replace(...)` rebuilds whatever
    container the caller passed in.

Full runnable version, including the `FitsImageDataset` wiring above:
`examples/example_custom_transform.py`.

---

## Importing

Import transform classes from `torchfits.transforms` (namespace-only since
0.9.2 — they are not re-exported at the package root):

```python
from torchfits.transforms import (
    AffineTransform,
    ArcsinhStretch,
    AsModule,
    AsymmetricSigmaClip,
    BackgroundSubtract,
    Compose,
    DataState,
    DataStateError,
    FITSScaleColumns,
    FITSHeaderNormalize,
    FITSHeaderScale,
    GlobalScalarNorm,
    InterquantileScale,
    InterquantileNormalize,
    LogStretch,
    MeshBackgroundSubtract,
    MinMaxNormalize,
    Payload,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SigmaNormalize,
    SqrtStretch,
    TNullToNan,
    ZScaleNormalize,
    apply_mask,
    as_module,
    calibration_state,
    combine_masks,
    estimate_background,
    lupton_rgb,
    mask_from_dq,
    mask_from_ivar,
    mask_from_nan,
    rgb,
    zscale_limits,
)
```
See `examples/example_transforms.py` (image pipeline),
`examples/example_rgb_sky.py` (auto RGB), and
`examples/example_lupton_rgb_sdss.py` (Lupton RGB) for runnable demos.
