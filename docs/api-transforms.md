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
| Stretches, `FITSHeaderScale`, `FITSScaleColumns` | Yes |
| Normalizers (`ZScale`, `Robust`, `BackgroundSubtract`, `MinMax`, `GlobalScalar`) | Yes (state cached on the instance) |
| `PercentileClipNormalize` | Approximate only — forward clips to `[lower, upper]`, so clipped pixels cannot be recovered exactly |
| `SigmaClip`, `AsymmetricSigmaClip`, `TNullToNan` | No — lossy / many-to-one |

!!! note "Float inputs required"
    The normalizers and `ArcsinhStretch` require float tensors — integer
    inputs raise `RuntimeError`. Convert first (`image.float()`); `LogStretch`,
    `SqrtStretch`, and `FITSHeaderScale` accept integer tensors.

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

!!! info "When to use"
    When you need stronger compression than arcsinh, e.g., for very wide-field
    images with bright stars and faint diffuse emission.

### `SqrtStretch()`

Square-root stretch — stabilizes Poisson variance. Negative input is clamped
to zero (inverse cannot recover negatives).

$$\text{output} = \sqrt{\max(x, 0)}$$

**Inverse:** $x = \text{output}^2$

!!! info "When to use"
    Quick and simple. Good default for photon-counting data where Poisson
    statistics apply.

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

!!! info "When to use"
    The standard choice for astronomical image display. Automatically adapts to
    the dynamic range of the data. Use for quick visualization or when you want
    a [0, 1] normalized image that preserves relative contrast.

### `RobustNormalize(dim=(-2, -1))`

Subtract median, divide by MAD-derived standard deviation.

$$\text{output} = \frac{x - \text{median}(x)}{\max(\text{MAD} \times 1.4826,\ 10^{-9})}$$

where $\text{MAD} = \text{median}(|x - \text{median}(x)|)$.

**Inverse:** $x = \text{output} \cdot \text{std\_approx} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for statistics |

!!! info "When to use"
    Universal ML preprocessing. Produces zero-mean, unit-variance-like data
    robust to outliers. Use as the last step before feeding to a neural
    network.

### `BackgroundSubtract(dim=(-2, -1))`

Subtract median background level.

$$\text{output} = x - \text{median}(x)$$

**Inverse:** $x = \text{output} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for background estimation |

!!! info "When to use"
    First step in most image pipelines. Removes the constant sky background
    before stretching or normalization.

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

!!! warning "Dict payloads vs tensor-only Compose"
    `Compose` and the built-in transforms operate on a single tensor plus an
    optional `mask=` kwarg. A Dataset payload is a plain `Tensor` only when
    no `ivar_hdu=` / `mask_hdu=` was requested; otherwise it is a
    `{"flux", "ivar"?, "mask"?}` dict, and the Dataset calls
    `transform(payload)` with that dict as the single positional argument —
    it does **not** unpack `mask` into a `mask=` kwarg for you (see
    [Dataset `transform=` signature](api-data.md#choosing-a-dataset)). A
    custom transform that must handle both cases branches on `isinstance(x,
    dict)`:

    ```python
    def forward(self, x, mask=None):
        if isinstance(x, dict):
            flux = x["flux"] * self.scale + self.offset
            return {**x, "flux": flux}
        return x * self.scale + self.offset
    ```

Full runnable version, including the `FitsImageDataset` wiring above:
`examples/example_custom_transform.py`.

### Helpers

| Function | Role |
|---|---|
| `safe_arcsinh`, `safe_log` | Numerically stable stretch primitives |
| `estimate_background` | Shared background estimator for normalizers / clip |
| `zscale_limits` | IRAF-style zscale limit finder used by `ZScaleNormalize` |

---

## Importing

Import transform classes from `torchfits.transforms` (namespace-only since
0.9.2 — they are not re-exported at the package root):

```python
from torchfits.transforms import (
    ArcsinhStretch,
    AsModule,
    BackgroundSubtract,
    Compose,
    ZScaleNormalize,
    RobustNormalize,
    MinMaxNormalize,
    PercentileClipNormalize,
    LogStretch,
    SqrtStretch,
    GlobalScalarNorm,
    InterquantileScale,
    InterquantileNormalize,
    AsymmetricSigmaClip,
    SigmaClip,
    FITSScaleColumns,
    TNullToNan,
    FITSHeaderNormalize,
    as_module,
    lupton_rgb,
    rgb,
)
```
See `examples/example_transforms.py` (image pipeline),
`examples/example_rgb_sky.py` (auto RGB), and
`examples/example_lupton_rgb_sdss.py` (Lupton RGB) for runnable demos.
