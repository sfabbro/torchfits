"""Robust float → int16 linear quantization (images + table columns).

Linear min→max packing onto BITPIX=16 / TFORM=I wastes codes on rare extremes
when the value distribution is skewed. Prefer native float storage for science;
when int16 size is mandatory, use :func:`quantize_int16_robust` (percentile bulk
range + clip) and write explicit BSCALE/BZERO or TSCAL/TZERO.

ponytail: global min→max (poloka FitsImage::Write) is intentionally not offered
as a default — it is the failure mode this helper avoids. Torch-only (no numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping

import torch
from torch import Tensor

# Match poloka's ±2 margin from hard int16 limits (overflow safety).
_SHRT_MIN_EFF = -32766
_SHRT_MAX_EFF = 32765
_SPAN = _SHRT_MAX_EFF - _SHRT_MIN_EFF  # 65531

# Reserved sentinel code for non-finite samples (outside the used range):
# written as FITS BLANK/TNULL so NaN can never masquerade as real data (B4).
BLANK_CODE: Final = -32767


@dataclass(frozen=True)
class QuantizeInt16Result:
    """Packed int16 codes plus FITS linear scale keywords.

    ``blank_code`` is ``BLANK_CODE`` when any non-finite sample was encoded
    as the reserved sentinel (writers must emit BLANK/TNULL accordingly);
    ``None`` when every sample was finite.
    """

    codes: Tensor
    scale: float
    zero: float
    lo: float
    hi: float
    n_clipped: int
    blank_code: int | None = None


@dataclass(frozen=True)
class QuantizeOptions:
    lo_q: float = 0.1
    hi_q: float = 99.9
    keep_zero: bool = False


def parse_quantize_options(spec: Any) -> QuantizeOptions | None:
    """Normalize ``quantize=`` for a single array/column.

    Accepts ``None``, ``\"robust\"``, ``True``, or a mapping with ``lo_q`` /
    ``hi_q`` / ``keep_zero``.
    """
    if spec is None or spec is False:
        return None
    if spec is True or spec == "robust":
        return QuantizeOptions()
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"lo_q", "hi_q", "keep_zero"}
        if unknown:
            raise TypeError(
                f"quantize option dict has unknown keys {sorted(unknown)!r}; "
                "expected lo_q, hi_q, keep_zero"
            )
        lo_q = float(spec.get("lo_q", 0.1))
        hi_q = float(spec.get("hi_q", 99.9))
        keep_zero = bool(spec.get("keep_zero", False))
        if not (0.0 <= lo_q < hi_q <= 100.0):
            raise ValueError(
                f"quantize lo_q/hi_q must satisfy 0 <= lo_q < hi_q <= 100, "
                f"got {lo_q!r}, {hi_q!r}"
            )
        return QuantizeOptions(lo_q=lo_q, hi_q=hi_q, keep_zero=keep_zero)
    raise TypeError(
        "quantize must be None, True, 'robust', or a dict with lo_q/hi_q/keep_zero"
    )


def parse_image_quantize_spec(spec: Any) -> QuantizeOptions | None:
    """Parse image ``write(..., quantize=)`` (options dict, not column map)."""
    return parse_quantize_options(spec)


def _is_floating_column(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(value.is_floating_point())
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, RuntimeError, ValueError):
        return False
    return bool(tensor.is_floating_point())


def parse_table_quantize_spec(
    spec: Any,
    columns: list[str],
    data: Mapping[str, Any] | None = None,
) -> dict[str, QuantizeOptions]:
    """Parse table ``write(..., quantize=)`` into per-column options.

    - ``\"robust\"`` / ``True`` → all *floating* columns (integer columns skipped)
    - ``{name: spec, ...}`` → only named columns (must be floating)
    """
    if spec is None or spec is False:
        return {}
    if spec is True or spec == "robust":
        opts = QuantizeOptions()
        if data is None:
            return {name: opts for name in columns}
        return {
            name: opts
            for name in columns
            if name in data and _is_floating_column(data[name])
        }
    if not isinstance(spec, Mapping):
        raise TypeError(
            "table quantize must be None, True, 'robust', or a dict of column specs"
        )
    # Disambiguate image-style option dict vs per-column map.
    keys = set(spec)
    if keys and keys <= {"lo_q", "hi_q", "keep_zero"}:
        parsed_opts = parse_quantize_options(spec)
        if parsed_opts is None:
            return {}
        if data is None:
            return {name: parsed_opts for name in columns}
        return {
            name: parsed_opts
            for name in columns
            if name in data and _is_floating_column(data[name])
        }

    out: dict[str, QuantizeOptions] = {}
    for name, col_spec in spec.items():
        key = str(name)
        if key not in columns:
            raise KeyError(f"quantize column {key!r} not in data columns {columns!r}")
        parsed = parse_quantize_options(col_spec)
        if parsed is None:
            continue
        if data is not None and not _is_floating_column(data[key]):
            raise TypeError(
                f"quantize column {key!r} must be floating-point, "
                f"got dtype={getattr(data[key], 'dtype', type(data[key]))}"
            )
        out[key] = parsed
    return out


def _as_work_flat(values: Tensor | Any) -> tuple[Tensor, tuple[int, ...], torch.device]:
    """Return contiguous float flat view (float32 preferred), shape, device.

    float32 inputs stay float32 (less bandwidth than float64 upcast). float16 /
    bfloat16 promote to float32; float64 stays float64. Array-likes are accepted
    via ``torch.as_tensor`` (no numpy import).
    """
    if isinstance(values, Tensor):
        tensor = values
    else:
        try:
            tensor = torch.as_tensor(values)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise TypeError(
                "quantize_int16_robust requires a floating Tensor or array-like"
            ) from exc

    if tensor.numel() == 0:
        raise ValueError("quantize_int16_robust: empty array")
    if not tensor.is_floating_point():
        raise TypeError(
            f"quantize_int16_robust requires floating values, got dtype={tensor.dtype}"
        )
    work_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    host = tensor.detach().to(device="cpu", dtype=work_dtype).reshape(-1).contiguous()
    return host, tuple(tensor.shape), tensor.device


def _percentile_sample(finite: Tensor, lo_q: float, hi_q: float) -> Tensor:
    """Finite samples used for percentile bounds.

    Large arrays with interior percentiles use a deterministic strided sample
    (~128k points) — rare extremes are usually excluded, which matches the
    robust goal and avoids an O(n log n) full partition.
    """
    n = int(finite.numel())
    # Exact endpoints need the full population min/max.
    if lo_q <= 0.0 and hi_q >= 100.0:
        return finite
    if n <= 262_144:
        return finite
    step = max(1, n // 131_072)
    return finite[::step]


def _percentile(sample: Tensor, q: float) -> float:
    if q <= 0.0:
        return float(sample.min())
    if q >= 100.0:
        return float(sample.max())
    return float(torch.quantile(sample, q / 100.0))


def _pack_codes(physical: Tensor, scale: float, zero: float) -> Tensor:
    """Round physical → int16 codes; clip to effective short range."""
    codes = torch.round((physical.to(dtype=torch.float64) - zero) / scale)
    return codes.clamp(_SHRT_MIN_EFF, _SHRT_MAX_EFF).to(torch.int16)


def quantize_int16_robust(
    values: Tensor | Any,
    *,
    lo_q: float = 0.1,
    hi_q: float = 99.9,
    keep_zero: bool = False,
) -> QuantizeInt16Result:
    """Pack float values to int16 with robust linear BSCALE/BZERO (or TSCAL/TZERO).

    ``lo_q`` / ``hi_q`` are percentiles over finite flattened samples. Values
    outside ``[lo, hi]`` (and non-finite samples) are clipped before rounding.
    Shape is preserved. Endpoint identity: ``lo`` → code ``-32766``, ``hi`` →
    ``32765`` (poloka ±2 margin).
    """
    if not (0.0 <= lo_q < hi_q <= 100.0):
        raise ValueError(
            f"lo_q/hi_q must satisfy 0 <= lo_q < hi_q <= 100, got {lo_q!r}, {hi_q!r}"
        )

    flat, shape, device = _as_work_flat(values)
    finite_mask = torch.isfinite(flat)
    # Avoid a copy when every sample is finite (common image/table path).
    finite = flat if bool(finite_mask.all()) else flat[finite_mask]
    if finite.numel() == 0:
        raise ValueError("quantize_int16_robust: no finite values to quantize")

    if keep_zero:
        # Weight/mask path: force BZERO=0; negatives clip to 0 (poloka KEEPZERO).
        positive = finite[finite > 0.0]
        if positive.numel() == 0:
            codes = torch.zeros(flat.shape, dtype=torch.int16).reshape(shape)
            if device.type != "cpu":
                codes = codes.to(device)
            return QuantizeInt16Result(
                codes=codes, scale=1.0, zero=0.0, lo=0.0, hi=0.0, n_clipped=0
            )
        sample = _percentile_sample(positive, 0.0, hi_q)
        hi = _percentile(sample, hi_q)
        if hi <= 0.0:
            hi = float(positive.max())
        scale = hi / float(_SHRT_MAX_EFF)
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        zero = 0.0
        lo = 0.0
        clipped = flat.to(dtype=torch.float64).clamp(0.0, hi)
        if not bool(finite_mask.all()):
            clipped = clipped.clone()
            clipped[~finite_mask] = 0.0
    else:
        sample = _percentile_sample(finite, lo_q, hi_q)
        lo = _percentile(sample, lo_q)
        hi = _percentile(sample, hi_q)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("quantize_int16_robust: non-finite percentile bounds")
        if hi <= lo:
            scale = 1.0
            zero = lo
            codes = torch.zeros(flat.shape, dtype=torch.int16).reshape(shape)
            n_clipped = int((~finite_mask).sum().item())
            blank_code = None
            if n_clipped:
                codes = codes.reshape(-1)
                codes[~finite_mask] = BLANK_CODE
                codes = codes.reshape(shape)
                blank_code = BLANK_CODE
            if device.type != "cpu":
                codes = codes.to(device)
            return QuantizeInt16Result(
                codes=codes,
                scale=scale,
                zero=zero,
                lo=lo,
                hi=hi,
                n_clipped=n_clipped,
                blank_code=blank_code,
            )
        scale = (hi - lo) / float(_SPAN)
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        zero = lo - scale * float(_SHRT_MIN_EFF)
        clipped = flat.to(dtype=torch.float64).clamp(lo, hi)
        if not bool(finite_mask.all()):
            clipped = clipped.clone()
            # Non-finite positions get placeholder values for packing; they
            # are overwritten with the BLANK sentinel below (B4).
            clipped[~finite_mask] = lo

    codes_flat = _pack_codes(clipped, scale, zero).reshape(-1)
    blank_code = None
    if not keep_zero and not bool(finite_mask.all()):
        codes_flat[~finite_mask] = BLANK_CODE
        blank_code = BLANK_CODE
    codes = codes_flat.reshape(shape)
    in_range = finite_mask & (flat >= lo) & (flat <= hi)
    n_clipped = int((~in_range).sum().item())

    if device.type != "cpu":
        codes = codes.to(device)
    return QuantizeInt16Result(
        codes=codes,
        scale=float(scale),
        zero=float(zero),
        lo=float(lo),
        hi=float(hi),
        n_clipped=n_clipped,
        blank_code=blank_code,
    )


def quantize_int16_minmax(values: Tensor | Any) -> QuantizeInt16Result:
    """Poloka-style global min→max pack (for tests / comparison only)."""
    flat, shape, device = _as_work_flat(values)
    finite_mask = torch.isfinite(flat)
    finite = flat if bool(finite_mask.all()) else flat[finite_mask]
    if finite.numel() == 0:
        raise ValueError("quantize_int16_minmax: no finite values")
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        scale = 1.0
        zero = lo
        codes = torch.zeros(flat.shape, dtype=torch.int16)
    else:
        scale = (hi - lo) / float(_SPAN)
        zero = lo - scale * float(_SHRT_MIN_EFF)
        clipped = flat.to(dtype=torch.float64).clamp(lo, hi)
        if not bool(finite_mask.all()):
            clipped = clipped.clone()
            clipped[~finite_mask] = lo
        codes = _pack_codes(clipped, scale, zero)
    codes = codes.reshape(shape)
    if device.type != "cpu":
        codes = codes.to(device)
    return QuantizeInt16Result(
        codes=codes,
        scale=float(scale),
        zero=float(zero),
        lo=lo,
        hi=hi,
        n_clipped=0,
    )


def dequantize_int16(
    codes: Tensor,
    scale: float,
    zero: float,
    *,
    dtype: torch.dtype = torch.float32,
    blank: int | None = None,
) -> Tensor:
    """Apply physical = scale * code + zero; ``blank`` codes decode to NaN."""
    out = codes.to(dtype=dtype) * float(scale) + float(zero)
    if blank is not None:
        out = torch.where(codes == blank, torch.nan, out)
    return out
