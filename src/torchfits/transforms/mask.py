"""Canonical mask construction for FITS companions (DQ, IVAR, NaN).

torchfits uses one convention everywhere: a boolean mask where ``True`` means
**valid**. FITS files store the opposite in two common shapes:

- a ``DQ`` integer extension where ``0`` is good and *set bits* flag defects
  (SDSS, DESI, MaNGA, HST all ship one), and
- an ``IVAR`` extension where ``0`` means "no data".

These helpers convert both to the torchfits convention so transforms,
statistics and datasets all agree on what a mask means.
"""

from __future__ import annotations

from typing import Iterable

import torch

__all__ = [
    "mask_from_dq",
    "mask_from_ivar",
    "mask_from_nan",
    "combine_masks",
    "apply_mask",
]


def _bit_mask(bits: int | Iterable[int]) -> int:
    """Turn a bit *position* (or iterable of positions) into a mask value.

    Bits are 0-based positions, matching how FITS DQ tables are documented
    ("bit 11: flux low"): ``_bit_mask(0) == 1``, ``_bit_mask([0, 11]) == 2049``.
    Positions ride on int64 bit ops, so 0..63 only.
    """
    if isinstance(bits, int):
        if bits < 0:
            raise ValueError(f"DQ bit positions must be non-negative, got {bits}")
        if bits > 63:
            raise ValueError(
                f"DQ bit positions must be in 0..63 (bit ops ride on int64), got {bits}"
            )
        return 1 << bits
    if not isinstance(bits, Iterable):
        raise TypeError(
            f"DQ bit positions must be ints or an iterable of ints, got {bits!r}"
        )
    total = 0
    for bit in bits:
        total |= _bit_mask(bit)
    return total


def mask_from_dq(
    dq: torch.Tensor,
    *,
    bad_bits: int | Iterable[int] | None = None,
    good_bits: int | Iterable[int] | None = None,
    require_good: bool = False,
) -> torch.Tensor:
    """Convert a FITS ``DQ`` integer extension to a boolean validity mask.

    Parameters
    ----------
    bad_bits :
        Bit *position*(s) that mark a pixel invalid, 0-based just like the
        FITS DQ tables are documented. ``None`` (default) treats *any*
        non-zero value as invalid, which is the right default when you have
        not looked up the instrument's bit definitions.
    good_bits :
        Bit position(s) that must be set for a pixel to count as valid.
        Only consulted when ``require_good=True``.
    require_good :
        Require ``good_bits``. With ``bad_bits=None`` the validity rule is
        then *only* "good bits set", so pixels carrying unrelated flags are
        still accepted (a "science" flag rather than a defect list).

    Returns
    -------
    Tensor
        Boolean mask where ``True`` means valid, same shape as *dq*.

    Examples
    --------
    >>> import torch
    >>> dq = torch.tensor([[0, 1, 4, 5, 2048]], dtype=torch.int32)
    >>> # MaNGA-style: any non-zero DQ value is suspect.
    >>> valid = mask_from_dq(dq)
    >>> # Only bit 0 (dead, value 1) and bit 11 (flux low, value 2048) are fatal.
    >>> valid = mask_from_dq(dq, bad_bits=[0, 11])
    >>> # Keep pixels explicitly tagged science-good (bit 2 set).
    >>> valid = mask_from_dq(dq, bad_bits=None, good_bits=[2], require_good=True)
    """
    if require_good and good_bits is None:
        raise ValueError(
            "require_good=True needs good_bits=... to name the science-good "
            "flag bit position(s)"
        )
    if dq.dtype.is_floating_point or dq.dtype.is_complex:
        raise TypeError(
            f"DQ extensions must be integer-typed, got {dq.dtype}. "
            "Read the extension with its native dtype."
        )
    wide = dq.to(torch.int64)
    if bad_bits is None:
        # No defect list: the default is "zero is good", except when the
        # caller opts into an explicit science-good flag instead.
        valid = torch.ones_like(wide, dtype=torch.bool) if require_good else wide == 0
    else:
        valid = (wide & _bit_mask(bad_bits)) == 0
    if require_good and good_bits is not None:
        good = _bit_mask(good_bits)
        valid = valid & ((wide & good) == good)
    return valid


def mask_from_ivar(
    ivar: torch.Tensor,
    *,
    min_ivar: float = 0.0,
    require_finite: bool = True,
) -> torch.Tensor:
    """Validity mask from an inverse-variance array (``0`` means "no data").

    Pixels with ``ivar <= min_ivar`` are invalid; NaN input is always invalid
    because comparisons against NaN are False.

    Parameters
    ----------
    ivar :
        Inverse-variance companion; ``0`` (and negative) means "no data".
    min_ivar :
        Inclusive-invalid floor: pixels need ``ivar > min_ivar`` to be valid.
    require_finite :
        Also reject non-finite entries (default). ``False`` keeps ``+inf``
        (zero-variance) valid; NaN stays invalid either way.
    """
    if require_finite:
        return torch.isfinite(ivar) & (ivar > float(min_ivar))
    return ivar > float(min_ivar)


def mask_from_nan(x: torch.Tensor) -> torch.Tensor:
    """Validity mask for a flux tensor: finite values are valid."""
    return torch.isfinite(x)


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Logical AND of any number of masks, ignoring ``None`` entries.

    Returns ``None`` when every argument is ``None``.
    """
    combined: torch.Tensor | None = None
    for mask in masks:
        if mask is None:
            continue
        current = mask.to(torch.bool)
        combined = current if combined is None else (combined & current)
    return combined


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    fill: float = float("nan"),
) -> torch.Tensor:
    """Replace invalid (``mask == False``) pixels with *fill*.

    Integer tensors are promoted to float — the fill may be NaN, which
    integers cannot represent. The promotion keeps every *valid* pixel exact:
    32/64-bit integers go to float64 (float32 would round counts above
    ``2**24``), smaller ones to float32; on MPS, which has no float64, the
    promotion is float32 throughout.
    """
    if mask is None:
        return x
    if not x.dtype.is_floating_point:
        x = x.to(
            torch.float32
            if x.dtype.itemsize <= 2 or x.device.type == "mps"
            else torch.float64
        )
    value = torch.tensor(fill, dtype=x.dtype, device=x.device)
    return torch.where(mask.to(torch.bool), x, value)
