"""CPU-first spherical harmonic primitives for HEALPix maps."""

from __future__ import annotations

import os
import math
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

try:
    from .. import cpp as _cpp
except Exception:  # pragma: no cover - optional extension
    _cpp = None

from ..wcs import healpix as _healpix

_ANGLE_CACHE: dict[tuple[int, bool], tuple[Tensor, Tensor, Tensor, float]] = {}
_YLM_CACHE: dict[tuple[int, bool, int, int], tuple[tuple[Tensor, ...], tuple[Tensor, ...]]] = {}
_ALM_ELL_CACHE: dict[tuple[int, int], Tensor] = {}
_ALM_M_CACHE: dict[tuple[int, int], Tensor] = {}
_SPIN_YLM_CACHE: dict[tuple[int, bool, int, int, int], tuple[tuple[Tensor, ...], tuple[Tensor, ...]]] = {}
_SPIN_MAP2ALM_MAT_CACHE: "OrderedDict[tuple[int, bool, int, int, int], tuple[Tensor, Tensor]]" = OrderedDict()
_SPIN_ALM2MAP_MAT_CACHE: "OrderedDict[tuple[int, bool, int, int, int], tuple[Tensor, Tensor, Tensor, Tensor]]" = OrderedDict()
_SCALAR_MAP2ALM_MAT_CACHE: "OrderedDict[tuple[int, bool, int, int], Tensor]" = OrderedDict()
_SCALAR_ALM2MAP_MAT_CACHE: "OrderedDict[tuple[int, bool, int, int], Tensor]" = OrderedDict()
_RING_LAYOUT_CACHE: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
_RING_SCALAR_BASIS_CACHE: dict[tuple[int, int, int], tuple[Tensor, ...]] = {}
_RING_PHI0_PHASE_CACHE: dict[tuple[int, int, int], Tensor] = {}
_RING_AZIMUTH_PHASE_CACHE: dict[tuple[int, int], Tensor] = {}
_RING_ALIAS_INDEX_CACHE: dict[tuple[int, int], Tensor] = {}
_RING_SPIN_CONJ_INDEX_CACHE: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]] = {}
_RING_PIX2RING_CACHE: dict[int, Tensor] = {}
_RING2NEST_PERM_CACHE: dict[int, Tensor] = {}
_NEST2RING_PERM_CACHE: dict[int, Tensor] = {}
_RING_SPIN_BASIS_CACHE: dict[tuple[int, int, int, int], tuple[Tensor, ...]] = {}
_RING_GROUP_CACHE: dict[int, tuple[tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]]] = {}
_SPIN_MAP2ALM_MAT_CACHE_ENTRY_BYTES: dict[tuple[int, bool, int, int, int], int] = {}
_SPIN_ALM2MAP_MAT_CACHE_ENTRY_BYTES: dict[tuple[int, bool, int, int, int], int] = {}
_SCALAR_MAP2ALM_MAT_CACHE_ENTRY_BYTES: dict[tuple[int, bool, int, int], int] = {}
_SCALAR_ALM2MAP_MAT_CACHE_ENTRY_BYTES: dict[tuple[int, bool, int, int], int] = {}
_SPIN_MAP2ALM_MAT_CACHE_TOTAL_BYTES = 0
_SPIN_ALM2MAP_MAT_CACHE_TOTAL_BYTES = 0
_SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES = 0
_SCALAR_ALM2MAP_MAT_CACHE_TOTAL_BYTES = 0
_SPIN_MAX_CACHE_BYTES = int(os.environ.get("TORCHFITS_SPIN_MAX_CACHE_BYTES", str(512 * 1024 * 1024)))
_SCALAR_MAX_CACHE_BYTES = int(os.environ.get("TORCHFITS_SCALAR_MAX_CACHE_BYTES", str(512 * 1024 * 1024)))
_SCALAR_RING_AUTO_MIN_BYTES = int(os.environ.get("TORCHFITS_SCALAR_RING_AUTO_MIN_BYTES", str(64 * 1024 * 1024)))
_SPIN_RING_AUTO_MIN_BYTES = int(os.environ.get("TORCHFITS_SPIN_RING_AUTO_MIN_BYTES", str(32 * 1024 * 1024)))
_SPIN_MAP2ALM_RING_AUTO_MIN_BYTES = int(
    os.environ.get("TORCHFITS_SPIN_MAP2ALM_RING_AUTO_MIN_BYTES", str(_SPIN_RING_AUTO_MIN_BYTES))
)
_SPIN_ALM2MAP_RING_AUTO_MIN_BYTES = int(
    os.environ.get("TORCHFITS_SPIN_ALM2MAP_RING_AUTO_MIN_BYTES", str(_SPIN_RING_AUTO_MIN_BYTES))
)
_SPIN_RING_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get("TORCHFITS_SPIN_RING_CONCAT_FAST_MAX_BYTES", str(256 * 1024 * 1024))
)
_SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get(
        "TORCHFITS_SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES",
        str(_SPIN_RING_CONCAT_FAST_MAX_BYTES),
    )
)
_SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get(
        "TORCHFITS_SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES",
        str(_SPIN_RING_CONCAT_FAST_MAX_BYTES),
    )
)
_SPIN_RING_INTEGRATE_CONCAT_MAX_NALM = int(
    os.environ.get("TORCHFITS_SPIN_RING_INTEGRATE_CONCAT_MAX_NALM", "4096")
)
_SPIN_RING_INTERP_CONCAT_CPP_ENABLE = os.environ.get("TORCHFITS_SPIN_RING_INTERP_CONCAT_CPP", "1") != "0"
_SPIN_RING_INTEGRATE_CONCAT_CPP_ENABLE = os.environ.get("TORCHFITS_SPIN_RING_INTEGRATE_CONCAT_CPP", "1") != "0"
_SPIN_RING_FINALIZE_CPP_ENABLE = os.environ.get("TORCHFITS_SPIN_RING_FINALIZE_CPP", "1") != "0"
_SPIN_MAP2ALM_RING_CONCAT_CPP_ENABLE = os.environ.get("TORCHFITS_SPIN_MAP2ALM_RING_CONCAT_CPP", "0") != "0"
_SPIN_RING_NEST_ENABLE = os.environ.get("TORCHFITS_SPIN_RING_NEST_ENABLE", "1") != "0"
_RING_SMALL_DFT_MAX_NPH = int(os.environ.get("TORCHFITS_RING_SMALL_DFT_MAX_NPH", "0"))
_RING_FOURIER_CPP_ENABLE = os.environ.get("TORCHFITS_RING_FOURIER_CPP", "0") != "0"
_RING_FOURIER_MODES_CPP_ENABLE = os.environ.get("TORCHFITS_RING_FOURIER_MODES_CPP", "0") != "0"
_RING_FOURIER_SYNTH_CPP_ENABLE = os.environ.get("TORCHFITS_RING_FOURIER_SYNTH_CPP", "0") != "0"
_RING_FOURIER_MODES_CPP_MAX_M = int(os.environ.get("TORCHFITS_RING_FOURIER_MODES_CPP_MAX_M", "64"))
_RING_FOURIER_SYNTH_CPP_MAX_M = int(os.environ.get("TORCHFITS_RING_FOURIER_SYNTH_CPP_MAX_M", "64"))
_PIXWIN_TABLE_CACHE: dict[int, tuple[Tensor, Tensor]] = {}
_PIXWIN_DATA_DIR = Path(__file__).resolve().parent / "data" / "pixel_window_functions"


def _preferred_real_dtype(device: torch.device) -> torch.dtype:
    return torch.float64 if device.type == "cpu" else torch.float32


def _preferred_complex_dtype(device: torch.device) -> torch.dtype:
    return torch.complex128 if device.type == "cpu" else torch.complex64


def alm_size(lmax: int, mmax: int | None = None) -> int:
    """Return number of alm coefficients in healpy ordering."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    mm = lmax if mmax is None else int(mmax)
    if mm < 0 or mm > lmax:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    return int(sum(lmax - m + 1 for m in range(mm + 1)))


def alm_index(ell: int, m: int, lmax: int, mmax: int | None = None) -> int:
    """Return flat alm index for mode (ell, m) in healpy ordering."""
    if ell < 0 or m < 0:
        raise ValueError("l and m must be non-negative")
    if m > ell:
        raise ValueError("m must be <= l")
    mm = lmax if mmax is None else int(mmax)
    if ell > lmax or m > mm:
        raise ValueError("(l,m) outside lmax/mmax bounds")
    start = m * (lmax + 1) - (m * (m - 1)) // 2
    return int(start + (ell - m))


def _infer_lmax_mmax_from_nalm(nalm: int, lmax: int | None, mmax: int | None) -> tuple[int, int]:
    if nalm <= 0:
        raise ValueError("nalm must be positive")
    if lmax is None and mmax is None:
        disc = 1 + 8 * nalm
        root = int(math.isqrt(disc))
        if root * root != disc:
            raise ValueError("cannot infer lmax from nalm")
        ell = (root - 3) // 2
        if (ell + 1) * (ell + 2) // 2 != nalm:
            raise ValueError("cannot infer lmax from nalm")
        return int(ell), int(ell)
    if lmax is None:
        raise ValueError("lmax must be provided when mmax is provided")
    mm = int(lmax if mmax is None else mmax)
    if alm_size(int(lmax), mm) != nalm:
        raise ValueError("nalm does not match provided lmax/mmax")
    return int(lmax), mm


def _infer_lmax_for_fixed_mmax(nalm: int, mmax: int) -> int:
    if nalm <= 0:
        raise ValueError("nalm must be positive")
    mm = int(mmax)
    if mm < 0:
        raise ValueError("mmax must be non-negative")
    num = nalm + (mm * (mm - 1)) // 2
    den = mm + 1
    if num % den != 0:
        raise ValueError("cannot infer lmax from nalm/mmax")
    ll = (num // den) - 1
    if ll < mm or alm_size(ll, mm) != nalm:
        raise ValueError("cannot infer lmax from nalm/mmax")
    return int(ll)


def _alm_ell_array(lmax: int, mmax: int) -> Tensor:
    key = (int(lmax), int(mmax))
    cached = _ALM_ELL_CACHE.get(key)
    if cached is not None:
        return cached
    vals = []
    for m in range(mmax + 1):
        vals.extend(range(m, lmax + 1))
    out = torch.tensor(vals, dtype=torch.int64)
    _ALM_ELL_CACHE[key] = out
    return out


def _alm_m_array(lmax: int, mmax: int) -> Tensor:
    key = (int(lmax), int(mmax))
    cached = _ALM_M_CACHE.get(key)
    if cached is not None:
        return cached
    vals = []
    for m in range(mmax + 1):
        vals.extend([m] * (lmax - m + 1))
    out = torch.tensor(vals, dtype=torch.int64)
    _ALM_M_CACHE[key] = out
    return out


def _map_to_cpu_rows(map_values: Tensor | list[float] | list[list[float]]) -> tuple[Tensor, bool]:
    t = torch.as_tensor(map_values)
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("map_values must be shape (npix,) or (nmaps, npix)")
    return rows.to(dtype=torch.float64, device="cpu"), single


def _map_to_rows(map_values: Tensor | list[float] | list[list[float]]) -> tuple[Tensor, bool]:
    t = torch.as_tensor(map_values)
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("map_values must be shape (npix,) or (nmaps, npix)")
    return rows.to(dtype=_preferred_real_dtype(rows.device)), single


def _alm_to_cpu_rows(alm_values: Tensor | list[complex] | list[list[complex]]) -> tuple[Tensor, bool]:
    t = torch.as_tensor(alm_values)
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("alm_values must be shape (nalm,) or (nmaps, nalm)")
    return rows.to(dtype=torch.complex128, device="cpu"), single


def _alm_to_rows(alm_values: Tensor | list[complex] | list[list[complex]]) -> tuple[Tensor, bool]:
    t = torch.as_tensor(alm_values)
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("alm_values must be shape (nalm,) or (nmaps, nalm)")
    return rows.to(dtype=_preferred_complex_dtype(rows.device)), single


def _angles_for_nside(nside: int, nest: bool) -> tuple[Tensor, Tensor, Tensor, float]:
    key = (int(nside), bool(nest))
    cached = _ANGLE_CACHE.get(key)
    if cached is not None:
        return cached
    npix = _healpix.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    theta, phi = _healpix.pix2ang(nside, pix, nest=nest, lonlat=False)
    theta = theta.to(dtype=torch.float64, device="cpu")
    phi = phi.to(dtype=torch.float64, device="cpu")
    x = torch.cos(theta)
    w = 4.0 * math.pi / float(npix)
    cached = (theta, phi, x, w)
    _ANGLE_CACHE[key] = cached
    return cached


def _ring2nest_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)
    cached = _RING2NEST_PERM_CACHE.get(ns)
    if cached is not None:
        return cached
    npix = _healpix.nside2npix(ns)
    idx = torch.arange(npix, dtype=torch.int64)
    perm = _healpix.ring2nest(ns, idx).to(dtype=torch.int64, device="cpu").contiguous()
    _RING2NEST_PERM_CACHE[ns] = perm
    return perm


def _nest2ring_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)
    cached = _NEST2RING_PERM_CACHE.get(ns)
    if cached is not None:
        return cached
    npix = _healpix.nside2npix(ns)
    idx = torch.arange(npix, dtype=torch.int64)
    perm = _healpix.nest2ring(ns, idx).to(dtype=torch.int64, device="cpu").contiguous()
    _NEST2RING_PERM_CACHE[ns] = perm
    return perm


def _reorder_nest_to_ring_rows(rows: Tensor, nside: int) -> Tensor:
    perm = _ring2nest_perm_for_nside(int(nside))
    if rows.device.type != "cpu":
        perm = perm.to(device=rows.device)
    return torch.index_select(rows, rows.ndim - 1, perm)


def _reorder_ring_to_nest_rows(rows: Tensor, nside: int) -> Tensor:
    perm = _nest2ring_perm_for_nside(int(nside))
    if rows.device.type != "cpu":
        perm = perm.to(device=rows.device)
    return torch.index_select(rows, rows.ndim - 1, perm)


def _pmm_base(m: int, x: Tensor) -> Tensor:
    if m == 0:
        return torch.ones_like(x)
    # (2m-1)!! = (2m)! / (2^m m!)
    log_coeff = math.lgamma(2 * m + 1) - (m * math.log(2.0)) - math.lgamma(m + 1)
    
    # base = (1-x^2)^(m/2)
    # log_base = (m/2) * log(1-x^2)
    sin2theta = torch.clamp(1.0 - x * x, min=1e-30)
    log_base = 0.5 * m * torch.log(sin2theta)
    
    val = torch.exp(log_coeff + log_base)
    sign = -1.0 if (m % 2) else 1.0
    return sign * val


def _legendre_l_sequence(m: int, x: Tensor, lmax: int) -> Tensor:
    """Associated Legendre P_l^m(x) for l=m..lmax. Output shape [lmax-m+1, npix]."""
    pmm = _pmm_base(m, x)
    if lmax == m:
        return pmm.unsqueeze(0)
    pm1 = (2 * m + 1) * x * pmm
    seq = [pmm, pm1]
    prev2 = pmm
    prev1 = pm1
    for ell in range(m + 2, lmax + 1):
        cur = ((2 * ell - 1) * x * prev1 - (ell + m - 1) * prev2) / float(ell - m)
        seq.append(cur)
        prev2 = prev1
        prev1 = cur
    return torch.stack(seq, dim=0)


def _ylm_norm(ell: int, m: int) -> float:
    return math.sqrt(
        ((2 * ell + 1) / (4.0 * math.pi)) * math.exp(math.lgamma(ell - m + 1) - math.lgamma(ell + m + 1))
    )


def _ylm_basis(nside: int, nest: bool, lmax: int, mmax: int) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Cached Y_lm basis blocks by m: each block has shape [l_count, npix]."""
    key = (int(nside), bool(nest), int(lmax), int(mmax))
    cached = _YLM_CACHE.get(key)
    if cached is not None:
        return cached

    _, phi, x, _ = _angles_for_nside(nside, nest)
    y_blocks: list[Tensor] = []
    y_conj_blocks: list[Tensor] = []
    for m in range(mmax + 1):
        p_seq = _legendre_l_sequence(m, x, lmax)
        phase = torch.exp(torch.complex(torch.zeros_like(phi), m * phi))  # exp(+i m phi)
        norms = torch.tensor([_ylm_norm(ell, m) for ell in range(m, lmax + 1)], dtype=torch.float64)
        y = (norms.unsqueeze(1) * p_seq).to(dtype=torch.complex128) * phase.unsqueeze(0)
        y_blocks.append(y)
        y_conj_blocks.append(torch.conj(y))
    out = (tuple(y_blocks), tuple(y_conj_blocks))
    _YLM_CACHE[key] = out
    return out


@torch.jit.script
def _wigner_d_element(ell: int, m: int, mp: int, theta: Tensor) -> Tensor:
    """
    Compute Wigner small-d element d^ell_{m,mp}(theta).
    Direct implementation using Wigner's formula.
    WARNING: Numerically unstable for large ell. Use for small ell only (e.g. bootstrapping).
    """
    if abs(m) > ell or abs(mp) > ell:
        return torch.zeros_like(theta, dtype=torch.float64)

    ct = torch.cos(theta * 0.5)
    st = torch.sin(theta * 0.5)
    out = torch.zeros_like(theta, dtype=torch.float64)

    log_pref = 0.5 * (
        math.lgamma(float(ell + m + 1))
        + math.lgamma(float(ell - m + 1))
        + math.lgamma(float(ell + mp + 1))
        + math.lgamma(float(ell - mp + 1))
    )
    kmin = max(0, m - mp)
    kmax = min(ell + m, ell - mp)
    if kmin > kmax:
        return out

    for k in range(kmin, kmax + 1):
        a = ell + m - k
        b = k
        c = mp - m + k
        d = ell - mp - k
        if min(a, b, c, d) < 0:
            continue
        sign = -1.0 if ((k + mp - m) & 1) else 1.0
        log_den = math.lgamma(float(a + 1)) + math.lgamma(float(b + 1)) + math.lgamma(float(c + 1)) + math.lgamma(float(d + 1))
        coeff = sign * math.exp(log_pref - log_den)
        p_ct = 2 * ell + m - mp - 2 * k
        p_st = mp - m + 2 * k
        out = out + coeff * (ct**p_ct) * (st**p_st)
    return out


@torch.jit.script
def _wigner_d_l_sequence(ell_min: int, ell_max: int, m: int, mp: int, theta: Tensor) -> Tensor:
    """
    Compute Wigner small-d elements d^ell_{m,mp}(theta) for ell in [ell_min, ell_max].
    Returns tensor of shape [ell_max - ell_min + 1, npix].
    Uses a stable three-term recurrence relation.
    """
    if ell_max < ell_min:
        return torch.empty((0, theta.shape[0]), dtype=torch.float64, device=theta.device)

    # Bootstrap with direct formula for the first two terms
    # The direct formula is stable and exact for terms near the minimal l (max(|m|,|mp|))
    # where the summation has few terms.
    d0 = _wigner_d_element(ell_min, m, mp, theta)
    if ell_max == ell_min:
        return d0.unsqueeze(0)

    d1 = _wigner_d_element(ell_min + 1, m, mp, theta)
    seq = [d0, d1]
    
    prev2 = d0
    prev1 = d1
    
    x = torch.cos(theta)
    
    # Recurrence relation:
    # l*sqrt((l^2-m^2)(l^2-mp^2)) * d^l_{m,mp} =
    #    (2l-1)(l(l-1)x - m*mp) * d^{l-1}_{m,mp}
    #    - (l-1)*sqrt(((l-1)^2-m^2)((l-1)^2-mp^2)) * d^{l-2}_{m,mp}
    
    for ell in range(ell_min + 2, ell_max + 1):
        # Recurrence:
        # (K_l / l) * d_l = (2l-1)(x - m*mp/(l(l-1))) * d_{l-1} - (K_{l-1} / (l-1)) * d_{l-2}
        # where K_l = sqrt((l^2 - m^2)(l^2 - mp^2))
        
        # Current K_l
        sq_l = math.sqrt((ell * ell - m * m) * (ell * ell - mp * mp))
        # Previous K_{l-1} needs to be recomputed or cached. 
        # (ell-1)^2 - m^2 ...
        # Actually it's cleaner to compute just what's needed.
        
        sq_lm1 = math.sqrt(((ell - 1) ** 2 - m * m) * ((ell - 1) ** 2 - mp * mp))
        
        # Coefficients
        # LHS factor: pre_factor = l / K_l
        if sq_l == 0:
             # Should not happen for ell > ell_min >= |m|, |mp|
             # But strictly, if l=|m| or l=|mp|, sq_l=0.
             # Loop starts ell_min+2, so l > |m| and l > |mp| strictly.
             pre_factor = 0.0 # Should be safe
        else:
             pre_factor = float(ell) / sq_l
             
        term1_scale = (2 * ell - 1)
        term1_shift = float(m * mp) / float(ell * (ell - 1))
        
        term2_val = (sq_lm1 / float(ell - 1))
        
        cur = pre_factor * ((term1_scale * (x - term1_shift)) * prev1 - term2_val * prev2)
        
        seq.append(cur)
        prev2 = prev1
        prev1 = cur

    return torch.stack(seq, dim=0)


def _spin_ylm_norm(ell: int) -> float:
    return math.sqrt((2 * ell + 1) / (4.0 * math.pi))


def _spin_ylm_basis(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Cached spin-weighted basis blocks by m: each block has shape [l_count, npix]."""
    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    cached = _SPIN_YLM_CACHE.get(key)
    if cached is not None:
        return cached

    theta, phi, _, _ = _angles_for_nside(nside, nest)
    y_blocks: list[Tensor] = []
    y_conj_blocks: list[Tensor] = []
    phase_multipliers = [torch.exp(torch.complex(torch.zeros_like(phi), m * phi)) for m in range(mmax + 1)]
    spin_phase = -1.0 if (spin & 1) else 1.0  # (-1)^spin

    for m in range(mmax + 1):
        phase = phase_multipliers[m]
        m_phase = -1.0 if (m & 1) else 1.0  # Condon-Shortley convention
        
        ell_start = max(m, abs(spin))
        if ell_start > lmax:
            y_blocks.append(torch.zeros((0, theta.shape[0]), dtype=torch.complex128, device=theta.device))
            y_conj_blocks.append(torch.zeros((0, theta.shape[0]), dtype=torch.complex128, device=theta.device))
            continue

        # Get d-matrix sequence for all required l
        d_seq = _wigner_d_l_sequence(ell_start, lmax, m, -spin, theta)
        
        # Calculate normalization factors
        ell_vec = torch.arange(ell_start, lmax + 1, dtype=torch.float64, device=theta.device)
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * math.pi))
        
        # Broadcast and multiply: (spin_phase * m_phase) * norm * d * phase
        combined_scale = (spin_phase * m_phase) * norms.unsqueeze(1)
        y = (combined_scale * d_seq).to(torch.complex128) * phase.unsqueeze(0)
        
        if ell_start > m:
            n_pre = ell_start - m
            # We need to prepend n_pre rows of zeros
            # y currently has shape [ell_max - ell_start + 1, npix]
            # We want [ell_max - m + 1, npix]
            pre_zeros = torch.zeros((n_pre, theta.shape[0]), dtype=torch.complex128, device=theta.device)
            y = torch.cat([pre_zeros, y], dim=0)

        y_blocks.append(y)
        y_conj_blocks.append(torch.conj(y))

    out = (tuple(y_blocks), tuple(y_conj_blocks))
    _SPIN_YLM_CACHE[key] = out
    return out


_SPIN_YLM_CONCAT_CACHE = {}
_SPIN_YLM_CONCAT_CACHE_ENTRY_BYTES = {}
_SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES = 0


def _spin_ylm_basis_concat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (nalm, npix) matrices for the spin-weighted basis.
    (Y, Y_conj)
    """
    key = (nside, nest, lmax, mmax, spin)
    if key in _SPIN_YLM_CONCAT_CACHE:
        # LRU: move to end
        val = _SPIN_YLM_CONCAT_CACHE.pop(key)
        _SPIN_YLM_CONCAT_CACHE[key] = val
        return val

    # Estimate size before computing to fail fast if too large? 
    # Or just compute and not cache if too large.
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    # Two complex128 matrices: 2 * (nalm * npix * 16 bytes)
    bytes_needed = 2 * nalm * npix * 16
    
    # If single entry is too big for cache limit, we can't cache it.
    # But we still return it (computed on fly).
    if bytes_needed > _SPIN_MAX_CACHE_BYTES:
        y_blocks, y_conj_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
        if not y_blocks:
             # handle empty...
             device = torch.device('cpu') 
             dtype = torch.complex128
             Y = torch.empty((0, npix), dtype=dtype, device=device)
             Y_conj = torch.empty((0, npix), dtype=dtype, device=device)
             return (Y, Y_conj)
        Y = torch.cat(y_blocks, dim=0)
        Y_conj = torch.cat(y_conj_blocks, dim=0)
        return (Y, Y_conj)

    global _SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES
    
    # Evict to make room
    while (_SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES + bytes_needed) > _SPIN_MAX_CACHE_BYTES and _SPIN_YLM_CONCAT_CACHE:
        old_key = next(iter(_SPIN_YLM_CONCAT_CACHE))
        _SPIN_YLM_CONCAT_CACHE.pop(old_key)
        old_bytes = _SPIN_YLM_CONCAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
        _SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES -= old_bytes

    y_blocks, y_conj_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
    
    if not y_blocks:
        device = torch.device('cpu') # distinct from default
        dtype = torch.complex128
        Y = torch.empty((0, 12 * nside**2), dtype=dtype, device=device)
        Y_conj = torch.empty((0, 12 * nside**2), dtype=dtype, device=device)
    else:
        Y = torch.cat(y_blocks, dim=0)
        Y_conj = torch.cat(y_conj_blocks, dim=0)

    out = (Y, Y_conj)
    _SPIN_YLM_CONCAT_CACHE[key] = out
    _SPIN_YLM_CONCAT_CACHE_ENTRY_BYTES[key] = bytes_needed
    _SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES += bytes_needed
    return out


def _spin_map2alm_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor] | None:
    """
    Optionally return cached concatenated map2alm matrices for spin transforms.

    Returns `(Yc_plus_t, Yc_minus_t)` with shape `[npix, nalm]` each, where
    `Yc` is the conjugated spin-weighted basis.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)  # two complex128 matrices
    if bytes_needed > _SPIN_MAX_CACHE_BYTES:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    cached = _SPIN_MAP2ALM_MAT_CACHE.get(key)
    if cached is not None:
        _SPIN_MAP2ALM_MAT_CACHE.move_to_end(key)
        return cached

    _, y_conj_plus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
    _, y_conj_minus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
    yc_plus_t = torch.cat([blk.transpose(0, 1) for blk in y_conj_plus_blocks], dim=1).contiguous()
    yc_minus_t = torch.cat([blk.transpose(0, 1) for blk in y_conj_minus_blocks], dim=1).contiguous()
    mats = (yc_plus_t, yc_minus_t)
    global _SPIN_MAP2ALM_MAT_CACHE_TOTAL_BYTES
    while (_SPIN_MAP2ALM_MAT_CACHE_TOTAL_BYTES + bytes_needed) > _SPIN_MAX_CACHE_BYTES and _SPIN_MAP2ALM_MAT_CACHE:
        old_key, _ = _SPIN_MAP2ALM_MAT_CACHE.popitem(last=False)
        old_bytes = _SPIN_MAP2ALM_MAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
        _SPIN_MAP2ALM_MAT_CACHE_TOTAL_BYTES -= old_bytes
    _SPIN_MAP2ALM_MAT_CACHE[key] = mats
    _SPIN_MAP2ALM_MAT_CACHE_ENTRY_BYTES[key] = bytes_needed
    _SPIN_MAP2ALM_MAT_CACHE_TOTAL_BYTES += bytes_needed
    return mats


def _spin_alm2map_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """
    Optionally return cached concatenated alm2map matrices for spin transforms.

    Returns `(Y_plus, Y_minus, Y_plus_m0, Y_minus_m0)` where:
    - `Y_plus`, `Y_minus` have shape `[nalm, npix]`.
    - `Y_plus_m0`, `Y_minus_m0` are the `m=0` blocks with shape `[lmax+1, npix]`.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)  # two complex128 matrices
    if bytes_needed > _SPIN_MAX_CACHE_BYTES:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    cached = _SPIN_ALM2MAP_MAT_CACHE.get(key)
    if cached is not None:
        _SPIN_ALM2MAP_MAT_CACHE.move_to_end(key)
        return cached

    y_plus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
    y_minus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
    y_plus = torch.cat([blk for blk in y_plus_blocks], dim=0).contiguous()
    y_minus = torch.cat([blk for blk in y_minus_blocks], dim=0).contiguous()
    mats = (y_plus, y_minus, y_plus_blocks[0], y_minus_blocks[0])
    global _SPIN_ALM2MAP_MAT_CACHE_TOTAL_BYTES
    while (_SPIN_ALM2MAP_MAT_CACHE_TOTAL_BYTES + bytes_needed) > _SPIN_MAX_CACHE_BYTES and _SPIN_ALM2MAP_MAT_CACHE:
        old_key, _ = _SPIN_ALM2MAP_MAT_CACHE.popitem(last=False)
        old_bytes = _SPIN_ALM2MAP_MAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
        _SPIN_ALM2MAP_MAT_CACHE_TOTAL_BYTES -= old_bytes
    _SPIN_ALM2MAP_MAT_CACHE[key] = mats
    _SPIN_ALM2MAP_MAT_CACHE_ENTRY_BYTES[key] = bytes_needed
    _SPIN_ALM2MAP_MAT_CACHE_TOTAL_BYTES += bytes_needed
    return mats


def _scalar_map2alm_mat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
) -> Tensor | None:
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16)
    if bytes_needed > _SCALAR_MAX_CACHE_BYTES:
        return None
    key = (int(nside), bool(nest), int(lmax), int(mmax))
    cached = _SCALAR_MAP2ALM_MAT_CACHE.get(key)
    if cached is not None:
        _SCALAR_MAP2ALM_MAT_CACHE.move_to_end(key)
        return cached
    _, y_conj_blocks = _ylm_basis(nside, nest, lmax, mmax)
    mat = torch.cat([blk.transpose(0, 1) for blk in y_conj_blocks], dim=1).contiguous()
    global _SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES
    while (_SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES + bytes_needed) > _SCALAR_MAX_CACHE_BYTES and _SCALAR_MAP2ALM_MAT_CACHE:
        old_key, _ = _SCALAR_MAP2ALM_MAT_CACHE.popitem(last=False)
        old_bytes = _SCALAR_MAP2ALM_MAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
        _SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES -= old_bytes
    _SCALAR_MAP2ALM_MAT_CACHE[key] = mat
    _SCALAR_MAP2ALM_MAT_CACHE_ENTRY_BYTES[key] = bytes_needed
    _SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES += bytes_needed
    return mat


def _scalar_alm2map_mat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
) -> Tensor | None:
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16)
    if bytes_needed > _SCALAR_MAX_CACHE_BYTES:
        return None
    key = (int(nside), bool(nest), int(lmax), int(mmax))
    cached = _SCALAR_ALM2MAP_MAT_CACHE.get(key)
    if cached is not None:
        _SCALAR_ALM2MAP_MAT_CACHE.move_to_end(key)
        return cached
    y_blocks, _ = _ylm_basis(nside, nest, lmax, mmax)
    mat = torch.cat([blk for blk in y_blocks], dim=0).contiguous()
    global _SCALAR_ALM2MAP_MAT_CACHE_TOTAL_BYTES
    while (_SCALAR_ALM2MAP_MAT_CACHE_TOTAL_BYTES + bytes_needed) > _SCALAR_MAX_CACHE_BYTES and _SCALAR_ALM2MAP_MAT_CACHE:
        old_key, _ = _SCALAR_ALM2MAP_MAT_CACHE.popitem(last=False)
        old_bytes = _SCALAR_ALM2MAP_MAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
        _SCALAR_ALM2MAP_MAT_CACHE_TOTAL_BYTES -= old_bytes
    _SCALAR_ALM2MAP_MAT_CACHE[key] = mat
    _SCALAR_ALM2MAP_MAT_CACHE_ENTRY_BYTES[key] = bytes_needed
    _SCALAR_ALM2MAP_MAT_CACHE_TOTAL_BYTES += bytes_needed
    return mat


def _ring_layout_for_nside(nside: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ns = int(nside)
    cached = _RING_LAYOUT_CACHE.get(ns)
    if cached is not None:
        return cached

    npix = _healpix.nside2npix(ns)
    nrings = (4 * ns) - 1
    theta, phi, _, _ = _angles_for_nside(ns, False)

    starts: list[int] = []
    lengths: list[int] = []
    theta_ring: list[float] = []
    phi0_ring: list[float] = []
    cursor = 0
    for ring in range(1, nrings + 1):
        if ring < ns:
            nph = 4 * ring
        elif ring <= (3 * ns):
            nph = 4 * ns
        else:
            nph = 4 * ((4 * ns) - ring)
        starts.append(cursor)
        lengths.append(nph)
        theta_ring.append(float(theta[cursor]))
        phi0_ring.append(float(phi[cursor]))
        cursor += nph
    if cursor != npix:
        raise RuntimeError(f"ring layout mismatch for nside={ns}: {cursor} != {npix}")

    out = (
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(theta_ring, dtype=torch.float64),
        torch.tensor(phi0_ring, dtype=torch.float64),
    )
    _RING_LAYOUT_CACHE[ns] = out
    return out


def _build_ring_grouping(
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
) -> tuple[tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]]:
    lengths_list = [int(v) for v in lengths_cpu.tolist()]
    if not lengths_list:
        return None, ()

    counts: dict[int, int] = {}
    for nph in lengths_list:
        counts[nph] = counts.get(nph, 0) + 1
    common_len = max(counts, key=counts.get)
    common_rings = [i for i, nph in enumerate(lengths_list) if nph == common_len]

    common_block: tuple[int, int, int, int, int] | None = None
    processed = [False] * len(lengths_list)
    if common_rings and all((b - a) == 1 for a, b in zip(common_rings, common_rings[1:])):
        start_ring = int(common_rings[0])
        end_ring = int(common_rings[-1])
        pix_start = int(starts_cpu[start_ring].item())
        pix_end = int(starts_cpu[end_ring].item()) + int(common_len)
        common_block = (int(common_len), start_ring, end_ring, pix_start, pix_end)
        for rid in range(start_ring, end_ring + 1):
            processed[rid] = True

    by_len: dict[int, list[int]] = {}
    for rid, nph in enumerate(lengths_list):
        if processed[rid]:
            continue
        by_len.setdefault(int(nph), []).append(int(rid))

    groups: list[tuple[int, Tensor, Tensor]] = []
    for nph, ring_ids in sorted(by_len.items()):
        ring_idx = torch.tensor(ring_ids, dtype=torch.int64)
        starts_sel = torch.index_select(starts_cpu, 0, ring_idx).unsqueeze(1)
        pix_offsets = torch.arange(int(nph), dtype=torch.int64).unsqueeze(0)
        pix_idx = starts_sel + pix_offsets
        groups.append((int(nph), ring_idx, pix_idx))
    return common_block, tuple(groups)


def _ring_grouping_for_nside(
    nside: int,
) -> tuple[tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]]:
    key = int(nside)
    cached = _RING_GROUP_CACHE.get(key)
    if cached is not None:
        return cached
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(key)
    out = _build_ring_grouping(starts_cpu, lengths_cpu)
    _RING_GROUP_CACHE[key] = out
    return out


def _ring_scalar_bases_for_nside(nside: int, lmax: int, mmax: int) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax))
    cached = _RING_SCALAR_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    x_ring = torch.cos(theta_ring_cpu)
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        p_seq = _legendre_l_sequence(m, x_ring, int(lmax))
        norms = torch.tensor([_ylm_norm(ell, m) for ell in range(m, int(lmax) + 1)], dtype=torch.float64)
        blocks.append((norms.unsqueeze(1) * p_seq).to(torch.complex128))
    out = tuple(blocks)
    _RING_SCALAR_BASIS_CACHE[key] = out
    return out


def _ring_phi0_phase_for_nside(nside: int, mmax: int, *, sign: int) -> Tensor:
    s = 1 if int(sign) >= 0 else -1
    key = (int(nside), int(mmax), int(s))
    cached = _RING_PHI0_PHASE_CACHE.get(key)
    if cached is not None:
        return cached
    _, _, _, phi0_ring_cpu = _ring_layout_for_nside(int(nside))
    m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64)
    ang = (float(s) * m_vals.unsqueeze(1)) * phi0_ring_cpu.unsqueeze(0)
    out = torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(torch.complex128)
    _RING_PHI0_PHASE_CACHE[key] = out
    return out


def _ring_azimuth_phase(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))
    cached = _RING_AZIMUTH_PHASE_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(nph)
    m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    ang = m_vals * (2.0 * math.pi / float(n)) * j
    out = torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(torch.complex128)
    _RING_AZIMUTH_PHASE_CACHE[key] = out
    return out


def _ring_alias_indices(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))
    cached = _RING_ALIAS_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    out = torch.remainder(torch.arange(int(mmax) + 1, dtype=torch.int64), int(nph))
    _RING_ALIAS_INDEX_CACHE[key] = out
    return out


def _ring_spin_conj_indices(nph: int, mmax: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Cached index vectors for spin conjugate mode extraction:
      +m indices, -m indices, and concatenated (+m|-m) indices.
    """
    key = (int(nph), int(mmax))
    cached = _RING_SPIN_CONJ_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(nph)
    mm = int(mmax)
    if n > mm:
        idx_plus = torch.arange(mm + 1, dtype=torch.int64)
    else:
        idx_plus = _ring_alias_indices(n, mm)
    idx_minus = torch.remainder(-idx_plus, n)
    idx_both = torch.cat([idx_plus, idx_minus], dim=0)
    out = (idx_plus, idx_minus, idx_both)
    _RING_SPIN_CONJ_INDEX_CACHE[key] = out
    return out


def _ring_spin_bases_for_nside(nside: int, lmax: int, mmax: int, spin: int) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax), int(spin))
    cached = _RING_SPIN_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    sp = int(spin)
    spin_phase = -1.0 if (sp & 1) else 1.0
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        m_phase = -1.0 if (m & 1) else 1.0
        
        ell_start = max(m, abs(sp))
        if ell_start > int(lmax):
            blocks.append(torch.zeros((0, theta_ring_cpu.shape[0]), dtype=torch.complex128))
            continue
            
        d_seq = _wigner_d_l_sequence(ell_start, int(lmax), m, -sp, theta_ring_cpu)
        
        ell_vec = torch.arange(ell_start, int(lmax) + 1, dtype=torch.float64, device=theta_ring_cpu.device)
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * math.pi))
        
        # norms: [L], d_seq: [L, N]
        block = (spin_phase * m_phase * norms.unsqueeze(1) * d_seq).to(torch.complex128)
        
        if ell_start > m:
            n_pre = ell_start - m
            pre_zeros = torch.zeros((n_pre, theta_ring_cpu.shape[0]), dtype=torch.complex128)
            block = torch.cat([pre_zeros, block], dim=0)

        blocks.append(block)
    out = tuple(blocks)
    _RING_SPIN_BASIS_CACHE[key] = out
    return out


_RING_SPIN_CONCAT_CACHE = {}
_RING_SPIN_CONCAT_CACHE_ENTRY_BYTES = {}
_RING_SPIN_CONCAT_CACHE_TOTAL_BYTES = 0

def _ring_spin_basis_concat(nside: int, lmax: int, mmax: int, spin: int) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (N_alm, N_rings) basis matrices for +spin and -spin.
    """
    key = (nside, lmax, mmax, spin)
    if key in _RING_SPIN_CONCAT_CACHE:
        val = _RING_SPIN_CONCAT_CACHE.pop(key)
        _RING_SPIN_CONCAT_CACHE[key] = val
        return val

    # Fetch blocks (this uses its own cache)
    y_plus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, spin)
    y_minus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, -spin)
    
    if not y_plus_blocks:
        device = torch.device('cpu') 
        dtype = torch.complex128
        Yp = torch.empty((0, 0), dtype=dtype, device=device)
        Ym = torch.empty((0, 0), dtype=dtype, device=device)
        return Yp, Ym

    # Concatenate
    Y_plus = torch.cat(y_plus_blocks, dim=0)
    Y_minus = torch.cat(y_minus_blocks, dim=0)
    
    # Cache Logic
    bytes_needed = Y_plus.numel() * 16 * 2
    if bytes_needed <= _SPIN_MAX_CACHE_BYTES:
        global _RING_SPIN_CONCAT_CACHE_TOTAL_BYTES
        while (_RING_SPIN_CONCAT_CACHE_TOTAL_BYTES + bytes_needed) > _SPIN_MAX_CACHE_BYTES and _RING_SPIN_CONCAT_CACHE:
            old_key = next(iter(_RING_SPIN_CONCAT_CACHE))
            _RING_SPIN_CONCAT_CACHE.pop(old_key)
            old_bytes = _RING_SPIN_CONCAT_CACHE_ENTRY_BYTES.pop(old_key, 0)
            _RING_SPIN_CONCAT_CACHE_TOTAL_BYTES -= old_bytes
            
        _RING_SPIN_CONCAT_CACHE[key] = (Y_plus, Y_minus)
        _RING_SPIN_CONCAT_CACHE_ENTRY_BYTES[key] = bytes_needed
        _RING_SPIN_CONCAT_CACHE_TOTAL_BYTES += bytes_needed
        
    return Y_plus, Y_minus



def _ring_fourier_modes(
    rows: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> Tensor:
    # rows: [nrows, npix] complex tensor.
    device = rows.device
    dtype = rows.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_RING_FOURIER_MODES_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _cpp is not None
        and hasattr(_cpp, "_healpix_ring_fourier_modes_cpu")
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not rows.requires_grad)
    )
    if use_cpp:
        return _cpp._healpix_ring_fourier_modes_cpu(
            rows.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out = torch.zeros((int(rows.shape[0]), int(mmax) + 1, nrings), dtype=dtype, device=device)
    m_int = torch.arange(int(mmax) + 1, dtype=torch.int64, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = rows[:, pix_start:pix_end]
        belt_stack = belt_pixels.view(rows.shape[0], n_common, common_len)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            if int(common_len) > int(mmax):
                phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(device=device, dtype=dtype)
                dft_sel = torch.matmul(belt_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(int(common_len), int(common_len) - 1).to(device=device, dtype=dtype)
                dft_base = torch.matmul(belt_stack, torch.conj(phase_base).transpose(0, 1))
                col_idx = torch.remainder(m_int, int(common_len))
                dft_sel = torch.index_select(dft_base, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = dft_sel.permute(0, 2, 1)
        else:
            fft_block = torch.fft.fft(belt_stack, dim=-1)
            if int(common_len) > int(mmax):
                # No aliasing when nph > mmax: use direct contiguous frequency slice.
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                col_idx = torch.remainder(m_int, int(common_len))
                fft_sel = torch.index_select(fft_block, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = fft_sel.permute(0, 2, 1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = rows[:, pix_idx]
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            if int(nph) > int(mmax):
                phase = _ring_azimuth_phase(int(nph), int(mmax)).to(device=device, dtype=dtype)
                dft_sel = torch.matmul(ring_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(int(nph), int(nph) - 1).to(device=device, dtype=dtype)
                dft_base = torch.matmul(ring_stack, torch.conj(phase_base).transpose(0, 1))
                dft_sel = torch.index_select(dft_base, 2, torch.remainder(m_int, int(nph)))
            out.index_copy_(2, ring_idx, dft_sel.permute(0, 2, 1))
        else:
            fft_block = torch.fft.fft(ring_stack, dim=-1)
            if int(nph) > int(mmax):
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                fft_sel = torch.index_select(fft_block, 2, torch.remainder(m_int, int(nph)))
            out.index_copy_(2, ring_idx, fft_sel.permute(0, 2, 1))

    return out


def _ring_fourier_modes_spin_conj(
    p_plus: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Specialized spin helper for p_minus = conj(p_plus).

    Returns (s_plus, s_minus) with shape (mmax+1, nrings), where
    s_plus[m, r]  = FFT(p_plus_ring_r)[m mod nph_r]
    s_minus[m, r] = FFT(conj(p_plus_ring_r))[m mod nph_r]
                  = conj(FFT(p_plus_ring_r)[(-m) mod nph_r]).
    """
    if p_plus.ndim != 1:
        raise ValueError("p_plus must be a 1D complex tensor")
    device = p_plus.device
    dtype = p_plus.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_RING_FOURIER_MODES_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _cpp is not None
        and hasattr(_cpp, "_healpix_ring_fourier_modes_spin_conj_cpu")
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not p_plus.requires_grad)
    )
    if use_cpp:
        out_cpp = _cpp._healpix_ring_fourier_modes_spin_conj_cpu(
            p_plus.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )
        return out_cpp[0], out_cpp[1]

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    k = int(mmax) + 1

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = p_plus[pix_start:pix_end]
        belt_stack = belt_pixels.view(n_common, common_len)
        fft_block = torch.fft.fft(belt_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(int(common_len), int(mmax))
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(common_len) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        s_plus[:, start_ring : end_ring + 1] = plus_sel.transpose(0, 1)
        s_minus[:, start_ring : end_ring + 1] = minus_sel.transpose(0, 1)

    n_group_rings = 0
    for _, ring_idx_cpu, _ in groups:
        n_group_rings += int(ring_idx_cpu.numel())
    if n_group_rings > 0:
        plus_all = torch.empty((k, n_group_rings), dtype=dtype, device=device)
        minus_all = torch.empty_like(plus_all)
        if device.type == "cpu":
            ring_idx_all = torch.empty((n_group_rings,), dtype=torch.int64)
        else:
            ring_idx_all = torch.empty((n_group_rings,), dtype=torch.int64, device=device)
    else:
        plus_all = None
        minus_all = None
        ring_idx_all = None
    write_off = 0

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = p_plus[pix_idx]
        fft_block = torch.fft.fft(ring_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(int(nph), int(mmax))
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(nph) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        ng = int(ring_idx.numel())
        if ng > 0 and plus_all is not None and minus_all is not None and ring_idx_all is not None:
            plus_all[:, write_off : write_off + ng] = plus_sel.transpose(0, 1)
            minus_all[:, write_off : write_off + ng] = minus_sel.transpose(0, 1)
            ring_idx_all[write_off : write_off + ng] = ring_idx
            write_off += ng

    if n_group_rings > 0 and plus_all is not None and minus_all is not None and ring_idx_all is not None:
        s_plus.index_copy_(1, ring_idx_all, plus_all)
        s_minus.index_copy_(1, ring_idx_all, minus_all)

    return s_plus, s_minus


def _scalar_alm2map_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_real = _preferred_real_dtype(device)
    dtype_cplx = _preferred_complex_dtype(device)

    starts = starts_cpu.to(device=device)
    y_base_blocks = _ring_scalar_bases_for_nside(int(nside), int(lmax), int(mmax))
    phi0_phase = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1).to(device=device, dtype=dtype_cplx)

    nmaps = int(rows.shape[0])
    nrings = int(theta_ring_cpu.numel())
    npix = _healpix.nside2npix(int(nside))
    out = torch.empty((nmaps, npix), dtype=dtype_real, device=device)
    lengths_list = [int(v) for v in lengths_cpu.tolist()]
    unique_lengths = sorted(set(lengths_list))

    f_m_ring = torch.zeros((nmaps, mmax + 1, nrings), dtype=dtype_cplx, device=device)
    idx = 0
    for m in range(mmax + 1):
        l_count = lmax - m + 1
        coeff = rows[:, idx : idx + l_count]
        y_base = y_base_blocks[m].to(device=device, dtype=dtype_cplx)
        fm = coeff @ y_base
        if m > 0:
            fm = fm * phi0_phase[m].unsqueeze(0)
        f_m_ring[:, m, :] = fm
        idx += l_count

    c_all = f_m_ring
    if mmax > 0:
        c_all = c_all.clone()
        c_all[:, 1:, :] = 2.0 * c_all[:, 1:, :]

    for nph in unique_lengths:
        ring_ids = [i for i, nph_i in enumerate(lengths_list) if nph_i == int(nph)]
        ring_idx = torch.tensor(ring_ids, dtype=torch.int64, device=device)
        c_group = torch.index_select(c_all, 2, ring_idx)
        phase = _ring_azimuth_phase(int(nph), int(mmax)).to(device=device, dtype=dtype_cplx)
        vals_group = torch.real(torch.einsum("amr,mn->arn", c_group, phase)).to(dtype_real)
        for j, rid in enumerate(ring_ids):
            start = int(starts[rid].item())
            out[:, start : start + int(nph)] = vals_group[:, j, :]

    return out


def _scalar_map2alm_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_cplx = _preferred_complex_dtype(device)

    nmaps = int(rows.shape[0])
    npix = int(rows.shape[1])
    nalm = alm_size(int(lmax), int(mmax))

    pix_w = (4.0 * math.pi) / float(npix)
    out = torch.zeros((nmaps, nalm), dtype=dtype_cplx, device=device)
    s_all = _ring_fourier_modes(
        rows.to(dtype=dtype_cplx),
        starts_cpu=starts_cpu,
        lengths_cpu=lengths_cpu,
        mmax=mmax,
        nside=int(nside),
    )

    phase0_neg = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1).to(device=device, dtype=dtype_cplx)
    s_all = s_all * phase0_neg.unsqueeze(0)

    y_base_blocks = _ring_scalar_bases_for_nside(int(nside), int(lmax), int(mmax))
    idx = 0
    for m in range(mmax + 1):
        y_base = y_base_blocks[m].to(device=device, dtype=dtype_cplx)
        s_m = s_all[:, m, :]
        l_count = lmax - m + 1
        out[:, idx : idx + l_count] = (s_m @ y_base.transpose(0, 1)) * pix_w
        idx += l_count

    return out


def _ring_fourier_synthesis(
    s_m: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    nside: int | None = None,
) -> Tensor:
    """
    Evaluate ring Fourier series from m>=0 coefficients.
    s_m: (mmax+1, nrings), complex128
    Returns: (npix,), complex128
    """
    if s_m.ndim == 2:
        s_m_batched = s_m.unsqueeze(0)
        squeeze_out = True
    elif s_m.ndim == 3:
        s_m_batched = s_m
        squeeze_out = False
    else:
        raise ValueError("s_m must have shape (mmax+1, nrings) or (nrows, mmax+1, nrings)")

    mmax = int(s_m_batched.shape[1]) - 1
    npix = int(starts_cpu[-1].item()) + int(lengths_cpu[-1].item())
    device = s_m_batched.device
    dtype = s_m_batched.dtype
    nrows = int(s_m_batched.shape[0])

    use_cpp = (
        (_RING_FOURIER_SYNTH_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _cpp is not None
        and hasattr(_cpp, "_healpix_ring_fourier_synthesis_cpu")
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_SYNTH_CPP_MAX_M
        and (not s_m_batched.requires_grad)
    )
    if use_cpp:
        out_cpp = _cpp._healpix_ring_fourier_synthesis_cpu(
            s_m_batched.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
        )
        if squeeze_out:
            return out_cpp[0]
        return out_cpp

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out_map = torch.zeros((nrows, npix), dtype=dtype, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        s_group = s_m_batched[:, :, start_ring : end_ring + 1]
        s_group_perm = s_group.permute(0, 2, 1)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(device=device, dtype=dtype)
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(common_len) > int(mmax):
                map_vals = torch.fft.ifft(s_group_perm * float(common_len), n=int(common_len), dim=-1)
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(common_len)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(common_len), int(mmax))
                alias_idx = alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                s_fold.index_add_(2, alias_idx, s_group_perm)
                map_vals = torch.fft.ifft(s_fold * float(common_len), dim=-1)
        out_map[:, pix_start:pix_end] = map_vals.reshape(nrows, -1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        s_group = torch.index_select(s_m_batched, 2, ring_idx)
        s_group_perm = s_group.permute(0, 2, 1)
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(nph), int(mmax)).to(device=device, dtype=dtype)
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(nph) > int(mmax):
                map_vals = torch.fft.ifft(s_group_perm * float(nph), n=int(nph), dim=-1)
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(nph)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(nph), int(mmax))
                alias_idx = alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                s_fold.index_add_(2, alias_idx, s_group_perm)
                map_vals = torch.fft.ifft(s_fold * float(nph), dim=-1)
        out_map[:, pix_idx] = map_vals

    if squeeze_out:
        return out_map[0]
    return out_map


def _ring_pix2ring_index(nside: int) -> Tensor:
    key = int(nside)
    cached = _RING_PIX2RING_CACHE.get(key)
    if cached is not None:
        return cached
    _, lengths_cpu, _, _ = _ring_layout_for_nside(key)
    nrings = int(lengths_cpu.numel())
    ring_ids = torch.arange(nrings, dtype=torch.int64)
    out = torch.repeat_interleave(ring_ids, lengths_cpu)
    _RING_PIX2RING_CACHE[key] = out
    return out


def _ring_expand_m0(s0: Tensor, *, nside: int) -> Tensor:
    """
    Expand per-ring m=0 values to per-pixel map.

    s0: (nrings,) complex
    returns: (npix,) complex
    """
    ring_idx_cpu = _ring_pix2ring_index(int(nside))
    ring_idx = ring_idx_cpu if s0.device.type == "cpu" else ring_idx_cpu.to(device=s0.device)
    if s0.ndim == 1:
        return torch.index_select(s0, 0, ring_idx)
    if s0.ndim == 2:
        return torch.index_select(s0, 1, ring_idx)
    raise ValueError("s0 must have shape (nrings,) or (nrows, nrings)")


def _ring_spin_interpolate_blocks(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = bytes_concat <= _SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(int(nside), int(lmax), int(mmax), int(spin))
        device = coeff_plus.device
        dtype = coeff_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((int(mmax) + 1, 0), dtype=dtype, device=device)
            return empty, empty
        y_plus = y_plus.to(device=device, dtype=dtype)
        y_minus = y_minus.to(device=device, dtype=dtype)
        use_cpp = (
            _SPIN_RING_INTERP_CONCAT_CPP_ENABLE
            and _cpp is not None
            and hasattr(_cpp, "_healpix_spin_interpolate_concat_cpu")
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not coeff_plus.requires_grad)
            and (not coeff_minus.requires_grad)
        )
        if use_cpp:
            s_plus_cpp, s_minus_cpp = _cpp._healpix_spin_interpolate_concat_cpu(
                coeff_plus.contiguous(),
                coeff_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
            )
            return s_plus_cpp, s_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        term_plus = coeff_plus.unsqueeze(1) * y_plus
        term_minus = coeff_minus.unsqueeze(1) * y_minus
        s_plus = torch.zeros((int(mmax) + 1, y_plus.shape[1]), dtype=dtype, device=device)
        s_minus = torch.zeros_like(s_plus)
        s_plus.index_add_(0, m_idx, term_plus)
        s_minus.index_add_(0, m_idx, term_minus)
        return s_plus, s_minus

    y_plus_blocks = _ring_spin_bases_for_nside(int(nside), int(lmax), int(mmax), int(spin))
    y_minus_blocks = _ring_spin_bases_for_nside(int(nside), int(lmax), int(mmax), -int(spin))
    nrings = int(y_plus_blocks[0].shape[1]) if y_plus_blocks else 0
    device = coeff_plus.device
    dtype = coeff_plus.dtype
    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = y_plus_blocks[m].to(device=device, dtype=dtype)
        y_minus = y_minus_blocks[m].to(device=device, dtype=dtype)
        c_plus_m = coeff_plus[idx : idx + l_count]
        c_minus_m = coeff_minus[idx : idx + l_count]
        s_plus[m] = c_plus_m @ y_plus
        s_minus[m] = c_minus_m @ y_minus
        idx += l_count
    return s_plus, s_minus


def _ring_spin_integrate_blocks(
    s_plus: Tensor,
    s_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
    pix_w: float,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = (
        bytes_concat <= _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES
        and nalm <= _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM
    )
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(int(nside), int(lmax), int(mmax), int(spin))
        device = s_plus.device
        dtype = s_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((nalm,), dtype=dtype, device=device)
            return empty, empty
        y_plus = y_plus.to(device=device, dtype=dtype)
        y_minus = y_minus.to(device=device, dtype=dtype)
        use_cpp = (
            _SPIN_RING_INTEGRATE_CONCAT_CPP_ENABLE
            and _cpp is not None
            and hasattr(_cpp, "_healpix_spin_integrate_concat_cpu")
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not s_plus.requires_grad)
            and (not s_minus.requires_grad)
        )
        if use_cpp:
            c_plus_cpp, c_minus_cpp = _cpp._healpix_spin_integrate_concat_cpu(
                s_plus.contiguous(),
                s_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
                float(pix_w),
            )
            return c_plus_cpp, c_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        s_plus_exp = torch.index_select(s_plus, 0, m_idx)
        s_minus_exp = torch.index_select(s_minus, 0, m_idx)
        c_plus = torch.sum(s_plus_exp * y_plus, dim=1) * pix_w
        c_minus = torch.sum(s_minus_exp * y_minus, dim=1) * pix_w
        return c_plus, c_minus

    y_plus_blocks = _ring_spin_bases_for_nside(int(nside), int(lmax), int(mmax), int(spin))
    y_minus_blocks = _ring_spin_bases_for_nside(int(nside), int(lmax), int(mmax), -int(spin))
    device = s_plus.device
    dtype = s_plus.dtype
    nalm = alm_size(int(lmax), int(mmax))
    c_plus = torch.empty((nalm,), dtype=dtype, device=device)
    c_minus = torch.empty_like(c_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = y_plus_blocks[m].to(device=device, dtype=dtype)
        y_minus = y_minus_blocks[m].to(device=device, dtype=dtype)
        c_plus[idx : idx + l_count] = (y_plus @ s_plus[m]) * pix_w
        c_minus[idx : idx + l_count] = (y_minus @ s_minus[m]) * pix_w
        idx += l_count
    return c_plus, c_minus


def _spin_alm2map_ring_torch(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> Tensor:
    """
    Helper for ring-based alm2map_spin.
    """
    use_cpp = (
        coeff_plus.device.type == "cpu"
        and getattr(_cpp, "_healpix_spin_interpolate_recurrence_cpu", None) is not None
        and (not coeff_plus.requires_grad)
        and os.environ.get("TORCHFITS_SPIN_ALM2MAP_RECURRENCE_CPP", "1") != "0"
    )

    if use_cpp:
        # print("DEBUG: Using C++ Recurrence")
        _, _, theta_ring, _ = _ring_layout_for_nside(int(nside))
        
        # Stack coeffs: (2, Nalm)
        alms_stacked = torch.stack([coeff_plus, coeff_minus], dim=0)
        
        # Returns stacked S: (2, mmax+1, nrings)
        # s_plus = S[0], s_minus = S[1]
        s_stacked = _cpp._healpix_spin_interpolate_recurrence_cpu(
            alms_stacked, theta_ring, int(lmax), int(mmax), int(spin)
        )
        s_plus = s_stacked[0]
        s_minus = s_stacked[1]
        
    else:
        s_plus, s_minus = _ring_spin_interpolate_blocks(
            coeff_plus,
            coeff_minus,
            nside=int(nside),
            lmax=int(lmax),
            mmax=int(mmax),
            spin=int(spin),
        )

    # 2. Apply phase factors and inverse FFT to get ring values
    phase0_pos = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1).to(dtype=torch.complex128)
    s_plus = s_plus * phase0_pos
    s_minus = s_minus * phase0_pos

    # 3. Inverse FFT (synthesis) + final spin combination
    start_pix, lengths, _, _ = _ring_layout_for_nside(int(nside))
    use_cpp_finalize = (
        _SPIN_RING_FINALIZE_CPP_ENABLE
        and _cpp is not None
        and hasattr(_cpp, "_healpix_spin_ring_finalize_cpu")
        and s_plus.device.type == "cpu"
        and (not s_plus.requires_grad)
        and (not s_minus.requires_grad)
    )
    if use_cpp_finalize:
        return _cpp._healpix_spin_ring_finalize_cpu(
            s_plus.contiguous(),
            s_minus.contiguous(),
            start_pix.contiguous(),
            lengths.contiguous(),
        )

    p_pm_pos = _ring_fourier_synthesis(
        torch.stack([s_plus, s_minus], dim=0),
        starts_cpu=start_pix,
        lengths_cpu=lengths,
        nside=int(nside),
    )
    p_plus_pos = p_pm_pos[0]
    p_minus_pos = p_pm_pos[1]
    p_pm_m0 = _ring_expand_m0(torch.stack([s_plus[0], s_minus[0]], dim=0), nside=int(nside))
    p_plus_m0 = p_pm_m0[0]
    p_minus_m0 = p_pm_m0[1]

    if mmax == 0:
        p_plus = p_plus_pos
        p_minus = p_minus_pos
    else:
        p_plus = p_plus_pos + torch.conj(p_minus_pos - p_minus_m0)
        p_minus = p_minus_pos + torch.conj(p_plus_pos - p_plus_m0)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    return torch.stack([(-q).real.to(torch.float64), (-u).real.to(torch.float64)], dim=0)


def _cross_cl_from_alm_pair(a1: Tensor, a2: Tensor, lmax: int, mmax: int) -> Tensor:
    ell = _alm_ell_array(lmax, mmax).to(a1.device)
    m = _alm_m_array(lmax, mmax).to(a1.device)
    weights = torch.where(m == 0, torch.ones_like(m, dtype=torch.float64), torch.full_like(m, 2.0, dtype=torch.float64))
    prod = torch.real(a1 * torch.conj(a2)).to(torch.float64) * weights
    cl = torch.zeros(lmax + 1, dtype=torch.float64, device=a1.device)
    cl.scatter_add_(0, ell, prod)
    norm = (2.0 * torch.arange(lmax + 1, dtype=torch.float64, device=a1.device)) + 1.0
    return cl / norm


def map2alm(
    map_values: Tensor | list[float] | list[list[float]],
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    iter: int = 0,
    pol: bool = False,
    use_pixel_weights: bool = False,
) -> Tensor:
    """
    Compute alm coefficients in healpy ordering (m>=0 only).

    With `pol=True`, input map must be shape `(3, npix)` and return is shape
    `(3, nalm)` in `(T,E,B)` ordering.
    """
    rows, single = _map_to_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    if _healpix.nside2npix(ns) != npix:
        raise ValueError("map length does not match nside")

    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if ll < 0 or mm < 0 or mm > ll:
        raise ValueError("invalid lmax/mmax")

    if pol:
        rows_pol = rows.to(dtype=torch.float64, device="cpu")
        if rows_pol.shape[0] != 3:
            raise ValueError("pol=True requires map_values shape (3, npix)")
        if backend == "healpy":
            try:
                import healpy as hp
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("healpy backend requested but healpy is not available") from exc
            arr = rows_pol.detach().cpu().numpy()
            if nest:
                arr = np.stack([hp.reorder(arr[i], n2r=True) for i in range(3)], axis=0)
            alm_np = hp.map2alm(
                arr,
                lmax=ll,
                mmax=mm,
                iter=int(iter),
                pol=True,
                use_pixel_weights=bool(use_pixel_weights),
            )
            return torch.from_numpy(np.asarray(alm_np)).to(dtype=torch.complex128)
        a_t = map2alm(
            rows_pol[0],
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            backend="torch",
            iter=iter,
            pol=False,
            use_pixel_weights=use_pixel_weights,
        )
        a_e, a_b = map2alm_spin(rows_pol[1:3], spin=2, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch")
        return torch.stack([a_t, a_e, a_b], dim=0)

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        out = []
        for i in range(rows.shape[0]):
            m_np = rows[i].detach().cpu().numpy()
            if nest:
                m_np = hp.reorder(m_np, n2r=True)
            alm_np = hp.map2alm(
                m_np,
                lmax=ll,
                mmax=mm,
                iter=int(iter),
                pol=False,
                use_pixel_weights=bool(use_pixel_weights),
            )
            out.append(torch.from_numpy(alm_np))
        alm = torch.stack(out, dim=0).to(dtype=torch.complex128)
        return alm[0] if single else alm

    npix = _healpix.nside2npix(ns)
    nalm = alm_size(ll, mm)
    bytes_needed = int(npix * nalm * 16)
    ring_mode = os.environ.get("TORCHFITS_SCALAR_MAP2ALM_RING_TORCH", "auto")
    auto_ring = ring_mode == "auto" and bytes_needed >= _SCALAR_RING_AUTO_MIN_BYTES
    use_ring_torch = (not nest) and (ring_mode == "force" or ring_mode == "1" or auto_ring)
    if use_ring_torch:
        alm_ring = _scalar_map2alm_ring_torch(rows, nside=ns, lmax=ll, mmax=mm)
        if torch.isfinite(alm_ring).all():
            return alm_ring[0] if single else alm_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHFITS_SCALAR_MAP2ALM_RING_TORCH=force produced non-finite values; "
                "disable force or lower lmax/nside."
            )

    _, _, _, pix_w = _angles_for_nside(ns, nest)
    complex_dtype = _preferred_complex_dtype(rows.device)
    map_c = rows.to(dtype=complex_dtype)
    mat = _scalar_map2alm_mat(ns, nest, ll, mm)
    if mat is not None:
        if mat.device != map_c.device or mat.dtype != complex_dtype:
            mat = mat.to(device=map_c.device, dtype=complex_dtype)
        alm = (map_c @ mat) * pix_w
    else:
        _, y_conj_blocks = _ylm_basis(ns, nest, ll, mm)
        alm = torch.zeros((rows.shape[0], nalm), dtype=complex_dtype, device=map_c.device)
        for m in range(mm + 1):
            y_conj = y_conj_blocks[m].to(device=map_c.device, dtype=complex_dtype)
            l_count = ll - m + 1
            coeffs = (map_c @ y_conj.transpose(0, 1)) * pix_w
            start = alm_index(m, m, ll, mm)
            alm[:, start : start + l_count] = coeffs

    return alm[0] if single else alm


def map2alm_lsq(
    map_values: Tensor | list[float] | list[list[float]],
    lmax: int,
    mmax: int | None = None,
    *,
    nside: int | None = None,
    pol: bool = True,
    tol: float = 1e-10,
    maxiter: int = 20,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    use_pixel_weights: bool = False,
) -> tuple[Tensor, float, int]:
    """
    Iterative least-squares map analysis up to (lmax, mmax).

    Returns `(alm, rel_res, n_iter)` where `rel_res` is the residual L2 norm
    divided by the input map L2 norm.
    """
    rows, single = _map_to_cpu_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    if _healpix.nside2npix(ns) != npix:
        raise ValueError("map length does not match nside")

    ll = int(lmax)
    mm = ll if mmax is None else int(mmax)
    if ll < 0 or mm < 0 or mm > ll:
        raise ValueError("invalid lmax/mmax")

    pol_req = bool(pol)
    if pol_req and rows.shape[0] not in (1, 3):
        raise ValueError("Wrong input map (must be a valid healpix map or a sequence of 1 or 3 maps)")
    pol_eff = pol_req and rows.shape[0] == 3

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        arr = rows.detach().cpu().numpy()
        if nest:
            if arr.ndim == 1:
                arr = hp.reorder(arr, n2r=True)
            else:
                arr = np.stack([hp.reorder(arr[i], n2r=True) for i in range(arr.shape[0])], axis=0)
        hp_in = arr[0] if single else arr
        alm_np, rel_res_np, n_iter_np = hp.map2alm_lsq(
            hp_in,
            lmax=ll,
            mmax=mm,
            pol=bool(pol_eff),
            tol=float(tol),
            maxiter=int(maxiter),
        )
        alm_t = torch.from_numpy(np.asarray(alm_np)).to(dtype=torch.complex128)
        return alm_t, float(rel_res_np), int(n_iter_np)

    alm0 = map2alm(
        rows,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        backend="torch",
        iter=0,
        pol=pol_eff,
        use_pixel_weights=use_pixel_weights,
    )
    alm_rows, _ = _alm_to_cpu_rows(alm0)
    map_norm = float(torch.linalg.norm(rows).item())
    n_iter = 0
    rel_res = float("inf")

    while True:
        rec = alm2map(alm_rows, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", pol=pol_eff, pixwin=False)
        rec_rows, _ = _map_to_cpu_rows(rec)
        residual = rows - rec_rows
        res_norm = float(torch.linalg.norm(residual).item())
        if map_norm > 0.0:
            rel_res = res_norm / map_norm
        else:
            rel_res = 0.0

        if n_iter >= int(maxiter) or rel_res <= float(tol):
            break

        delta = map2alm(
            residual,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            backend="torch",
            iter=0,
            pol=pol_eff,
            use_pixel_weights=use_pixel_weights,
        )
        delta_rows, _ = _alm_to_cpu_rows(delta)
        alm_rows = alm_rows + delta_rows
        n_iter += 1

    if single and not pol_eff:
        return alm_rows[0], rel_res, n_iter
    return alm_rows, rel_res, n_iter


def almxfl(
    alm_values: Tensor | list[complex] | list[list[complex]],
    fl: Tensor | list[float],
    *,
    mmax: int | None = None,
    inplace: bool = False,
) -> Tensor:
    """
    Multiply alm coefficient(s) by an ell-dependent transfer function.

    Behaves like healpy's `almxfl`: if `fl` is shorter than `lmax+1`, missing
    multipliers are treated as zero; extra entries are ignored.
    """
    alms_in = torch.as_tensor(alm_values)
    single = alms_in.ndim == 1
    if alms_in.ndim not in (1, 2):
        raise ValueError("alm_values must be shape (nalm,) or (nmaps, nalm)")
    can_inplace = (
        inplace
        and isinstance(alm_values, Tensor)
        and torch.is_complex(alm_values)
        and alm_values.dtype == torch.complex128
        and alms_in.dtype == torch.complex128
    )
    if can_inplace:
        rows = alms_in.unsqueeze(0) if single else alms_in
    else:
        rows = (alms_in.unsqueeze(0) if single else alms_in).to(dtype=torch.complex128)
    nalm = int(rows.shape[-1])
    if mmax is None:
        ll, mm = _infer_lmax_mmax_from_nalm(nalm, None, None)
    else:
        mm = int(mmax)
        ll = _infer_lmax_for_fixed_mmax(nalm, mm)

    fl_t = torch.as_tensor(fl, dtype=torch.float64)
    if fl_t.ndim != 1:
        raise ValueError("fl must be 1D")
    ell = _alm_ell_array(ll, mm).to(rows.device)
    scale = torch.zeros((ll + 1,), dtype=torch.float64, device=rows.device)
    keep = min(int(fl_t.numel()), ll + 1)
    if keep > 0:
        scale[:keep] = fl_t[:keep].to(device=rows.device, dtype=torch.float64)
    gain = scale.index_select(0, ell).to(torch.complex128)

    out = rows if can_inplace else rows.clone()
    out.mul_(gain.unsqueeze(0))
    return out[0] if single else out


def alm2cl(
    alms1: Tensor | list[complex] | list[list[complex]],
    alms2: Tensor | list[complex] | list[list[complex]] | None = None,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
) -> Tensor:
    """
    Compute auto/cross power spectra from alm coefficients.

    For a single alm input vector, returns shape `(lmax_out+1,)`.
    For multiple inputs, returns stacked spectra in healpy-compatible ordering.
    """
    a1_rows, a1_single = _alm_to_cpu_rows(alms1)
    ll, mm = _infer_lmax_mmax_from_nalm(int(a1_rows.shape[1]), lmax, mmax)
    if alms2 is None:
        a2_rows = a1_rows
    else:
        a2_rows, _ = _alm_to_cpu_rows(alms2)
        ll2, mm2 = _infer_lmax_mmax_from_nalm(int(a2_rows.shape[1]), lmax, mmax)
        if ll2 != ll or mm2 != mm:
            raise ValueError("alms2 shape incompatible with lmax/mmax")
    if a1_rows.shape[0] != a2_rows.shape[0]:
        raise ValueError("alms1 and alms2 must have same number of components")

    ncomp = int(a1_rows.shape[0])
    spectra: list[Tensor] = []

    for i in range(ncomp):
        spectra.append(_cross_cl_from_alm_pair(a1_rows[i], a2_rows[i], ll, mm))

    for delta in range(1, ncomp):
        for i in range(0, ncomp - delta):
            j = i + delta
            spectra.append(_cross_cl_from_alm_pair(a1_rows[i], a2_rows[j], ll, mm))

    cls = torch.stack(spectra, dim=0)
    if nspec is not None:
        cls = cls[: int(nspec)]
    ll_out = ll if lmax_out is None else int(lmax_out)
    if ll_out < 0:
        raise ValueError("lmax_out must be non-negative")
    ll_out = min(ll_out, ll)
    cls = cls[..., : ll_out + 1]
    if a1_single and (alms2 is None or torch.as_tensor(alms2).ndim == 1):
        return cls[0]
    return cls


def alm2map(
    alm_values: Tensor | list[complex] | list[list[complex]],
    nside: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    pol: bool = False,
    pixwin: bool = False,
) -> Tensor:
    """
    Synthesize HEALPix map(s) from alm coefficients.

    For scalar maps, m>0 terms use the real-map relation `2*Re(a_lm Y_lm)`.
    With `pol=True`, `alm_values` must be shape `(3, nalm)` in `(T,E,B)` order.
    """
    rows, single = _alm_to_rows(alm_values)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if pol:
        alms_pol = torch.as_tensor(alm_values, dtype=torch.complex128, device="cpu")
        if alms_pol.ndim != 2 or alms_pol.shape[0] != 3:
            raise ValueError("pol=True requires alm_values shape (3, nalm)")
        ll, mm = _infer_lmax_mmax_from_nalm(int(alms_pol.shape[1]), lmax, mmax)
        if backend == "healpy":
            try:
                import healpy as hp
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("healpy backend requested but healpy is not available") from exc
            maps_np = hp.alm2map(
                alms_pol.detach().cpu().numpy(),
                nside=nside,
                lmax=ll,
                mmax=mm,
                pol=True,
                pixwin=bool(pixwin),
            )
            out = torch.from_numpy(np.asarray(maps_np)).to(dtype=torch.float64)
            if nest:
                out = torch.stack(
                    [torch.from_numpy(hp.reorder(out[i].cpu().numpy(), r2n=True)).to(torch.float64) for i in range(3)],
                    dim=0,
                )
            return out

        a_t = alms_pol[0]
        a_e = alms_pol[1]
        a_b = alms_pol[2]
        if pixwin:
            pixwin_loader = globals()["pixwin"]
            pw_t, pw_p = pixwin_loader(int(nside), lmax=ll, pol=True)
            ell = _alm_ell_array(ll, mm)
            st = pw_t.index_select(0, ell).to(torch.complex128)
            sp = pw_p.index_select(0, ell).to(torch.complex128)
            a_t = a_t * st
            a_e = a_e * sp
            a_b = a_b * sp
        t = alm2map(a_t, nside=nside, lmax=ll, mmax=mm, nest=nest, backend="torch", pol=False, pixwin=False)
        qu = alm2map_spin((a_e, a_b), nside=nside, spin=2, lmax=ll, mmax=mm, nest=nest, backend="torch")
        return torch.stack([t, qu[0], qu[1]], dim=0)

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        out = []
        for i in range(rows.shape[0]):
            m_np = hp.alm2map(
                rows[i].detach().cpu().numpy(),
                nside=nside,
                lmax=ll,
                mmax=mm,
                pol=False,
                pixwin=bool(pixwin),
            )
            if nest:
                m_np = hp.reorder(m_np, r2n=True)
            out.append(torch.from_numpy(m_np))
        m = torch.stack(out, dim=0).to(dtype=torch.float64)
        return m[0] if single else m

    rows_eff = rows
    if pixwin:
        pixwin_loader = globals()["pixwin"]
        pw = pixwin_loader(int(nside), lmax=ll, pol=False)
        rows_eff = smoothalm(rows, lmax=ll, mmax=mm, beam=pw, pol=False)
        rows_eff, _ = _alm_to_rows(rows_eff)

    ring_mode = os.environ.get("TORCHFITS_SCALAR_ALM2MAP_RING_TORCH", "auto")
    npix = _healpix.nside2npix(nside)
    nalm = alm_size(ll, mm)
    bytes_needed = int(npix * nalm * 16)
    auto_ring = ring_mode == "auto" and bytes_needed >= _SCALAR_RING_AUTO_MIN_BYTES
    use_ring_torch = (not nest) and (ring_mode == "force" or ring_mode == "1" or auto_ring)
    if use_ring_torch:
        out_ring = _scalar_alm2map_ring_torch(rows_eff, nside=int(nside), lmax=ll, mmax=mm)
        if torch.isfinite(out_ring).all():
            return out_ring[0] if single else out_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHFITS_SCALAR_ALM2MAP_RING_TORCH=force produced non-finite values; "
                "disable force or lower lmax/nside."
            )

    use_cpp_direct = (
        _cpp is not None
        and hasattr(_cpp, "_healpix_scalar_alm2map_direct_cpu")
        and rows_eff.device.type == "cpu"
        and (not rows_eff.requires_grad)
        and os.environ.get("TORCHFITS_SCALAR_ALM2MAP_CPP", "0") != "0"
    )
    if use_cpp_direct:
        _, phi, x, _ = _angles_for_nside(int(nside), bool(nest))
        out_cpp = _cpp._healpix_scalar_alm2map_direct_cpu(
            rows_eff.contiguous(),
            x.contiguous(),
            phi.contiguous(),
            int(ll),
            int(mm),
        ).to(dtype=torch.float64)
        return out_cpp[0] if single else out_cpp

    mat = _scalar_alm2map_mat(nside, nest, ll, mm)
    real_dtype = _preferred_real_dtype(rows_eff.device)
    complex_dtype = _preferred_complex_dtype(rows_eff.device)
    if mat is not None:
        if mat.device != rows_eff.device or mat.dtype != complex_dtype:
            mat = mat.to(device=rows_eff.device, dtype=complex_dtype)
        m_idx = _alm_m_array(ll, mm)
        gain = torch.where(
            m_idx == 0,
            torch.ones_like(m_idx, dtype=real_dtype),
            torch.full_like(m_idx, 2.0, dtype=real_dtype),
        ).to(dtype=complex_dtype, device=rows_eff.device)
        out = torch.real((rows_eff * gain.unsqueeze(0)) @ mat).to(real_dtype)
    else:
        y_blocks, _ = _ylm_basis(nside, nest, ll, mm)
        out = torch.zeros((rows.shape[0], npix), dtype=real_dtype, device=rows_eff.device)
        for m in range(mm + 1):
            y = y_blocks[m].to(device=rows_eff.device, dtype=complex_dtype)
            l_count = ll - m + 1
            start = alm_index(m, m, ll, mm)
            coeffs = rows_eff[:, start : start + l_count]
            contrib = coeffs @ y
            if m == 0:
                out = out + contrib.real
            else:
                out = out + 2.0 * contrib.real
    return out[0] if single else out


def anafast(
    map1: Tensor | list[float],
    map2: Tensor | list[float] | None = None,
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    nspec: int | None = None,
    iter: int = 0,
    alm: bool = False,
    pol: bool = True,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Compute auto/cross angular power spectrum C_l from scalar or polarized maps."""
    m1 = torch.as_tensor(map1)
    if m1.ndim not in (1, 2):
        raise ValueError("map1 must be 1D or 2D")
    if map2 is None:
        m2 = m1
    else:
        m2 = torch.as_tensor(map2)
        if m2.ndim != m1.ndim or m2.shape != m1.shape:
            raise ValueError("map2 must match map1 shape")

    npix = int(m1.shape[-1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if mm < 0 or mm > ll:
        raise ValueError("invalid mmax")

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        m1_np = m1.detach().cpu().numpy()
        m2_np = m2.detach().cpu().numpy()
        if nest:
            if m1.ndim == 1:
                m1_np = hp.reorder(m1_np, n2r=True)
                m2_np = hp.reorder(m2_np, n2r=True)
            else:
                m1_np = np.stack([hp.reorder(m1_np[i], n2r=True) for i in range(m1_np.shape[0])], axis=0)
                m2_np = np.stack([hp.reorder(m2_np[i], n2r=True) for i in range(m2_np.shape[0])], axis=0)
        out = hp.anafast(
            m1_np,
            map2=None if map2 is None else m2_np,
            nspec=nspec,
            lmax=ll,
            mmax=mm,
            iter=int(iter),
            alm=bool(alm),
            pol=bool(pol),
            use_weights=bool(use_weights),
            use_pixel_weights=bool(use_pixel_weights),
            gal_cut=float(gal_cut),
        )
        if alm:
            if map2 is None:
                cls_np, alm1_np = out
                return torch.from_numpy(np.asarray(cls_np)).to(torch.float64), torch.from_numpy(np.asarray(alm1_np)).to(
                    torch.complex128
                )
            cls_np, alm1_np, alm2_np = out
            return (
                torch.from_numpy(np.asarray(cls_np)).to(torch.float64),
                torch.from_numpy(np.asarray(alm1_np)).to(torch.complex128),
                torch.from_numpy(np.asarray(alm2_np)).to(torch.complex128),
            )
        return torch.from_numpy(np.asarray(out)).to(torch.float64)

    if use_weights or use_pixel_weights or gal_cut != 0.0:
        raise ValueError("torch backend does not support use_weights/use_pixel_weights/gal_cut yet")

    if m1.ndim == 2 and bool(pol):
        if m1.shape[0] != 3:
            raise ValueError("pol=True expects map shape (3, npix)")
        a1 = map2alm(m1, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=True)
        a2 = (
            a1
            if map2 is None
            else map2alm(m2, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=True)
        )
        cls = alm2cl(a1, a2, lmax=ll, mmax=mm, nspec=nspec)
        if alm:
            return (cls, a1) if map2 is None else (cls, a1, a2)
        return cls

    if m1.ndim != 1:
        raise ValueError("pol=False torch backend currently supports only 1D maps")

    a1 = map2alm(m1, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=False)
    a2 = map2alm(m2, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=False)
    cl = alm2cl(a1, a2, lmax=ll, mmax=mm, nspec=nspec)
    if alm:
        return (cl, a1) if map2 is None else (cl, a1, a2)
    return cl


def gaussian_beam(fwhm_rad: float, lmax: int, *, pol: bool = False) -> Tensor:
    """Return Gaussian beam transfer function(s) B_l."""
    if fwhm_rad < 0.0:
        raise ValueError("fwhm_rad must be >= 0")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    sigma = float(fwhm_rad) / math.sqrt(8.0 * math.log(2.0))
    ell = torch.arange(lmax + 1, dtype=torch.float64)
    base = torch.exp(-0.5 * ell * (ell + 1.0) * (sigma * sigma))
    if not pol:
        return base
    fac_p = math.exp(2.0 * sigma * sigma)
    fac_tp = math.exp(sigma * sigma)
    return torch.stack([base, base * fac_p, base * fac_p, base * fac_tp], dim=1)


def _load_pixwin_table(nside: int) -> tuple[Tensor, Tensor]:
    ns = int(nside)
    cached = _PIXWIN_TABLE_CACHE.get(ns)
    if cached is not None:
        return cached
    path = _PIXWIN_DATA_DIR / f"pixel_window_n{ns:04d}.fits"
    if not path.is_file():
        raise ValueError(f"pixel window table not available for nside={ns}: {path}")

    data = None
    if _cpp is not None and hasattr(_cpp, "read_fits_table"):
        try:
            data = _cpp.read_fits_table(str(path), 1)
        except Exception:
            data = None
    if data is None:
        try:
            from .. import read_table  # Local import to avoid import cycles at module load.
        except Exception as exc:
            raise RuntimeError("pixwin requires torchfits table reader when C++ table API is unavailable") from exc
        data = read_table(str(path), hdu=1)

    def _get_col(name: str) -> Tensor:
        candidates = (name, name.upper(), name.lower())
        for c in candidates:
            if c in data:
                return torch.as_tensor(data[c], dtype=torch.float64, device="cpu").reshape(-1)
        raise RuntimeError(f"missing '{name}' column in pixel window table: {path}")

    pw_t = _get_col("TEMPERATURE")
    pw_p = _get_col("POLARIZATION")
    out = (pw_t, pw_p)
    _PIXWIN_TABLE_CACHE[ns] = out
    return out


def pixwin(
    nside: int,
    *,
    pol: bool = False,
    lmax: int | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Return HEALPix pixel window function(s).

    Data are loaded from packaged HEALPix pixel-window FITS tables.
    """
    ns = int(nside)
    if not bool(_healpix.isnsideok(ns)):
        raise ValueError("Wrong nside value (must be a power of two).")
    ll = int((3 * ns - 1) if lmax is None else lmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    pw_t_all, pw_p_all = _load_pixwin_table(ns)
    if ll >= int(pw_t_all.numel()):
        ll = int(pw_t_all.numel()) - 1
    if bool(pol):
        return pw_t_all[: ll + 1].clone(), pw_p_all[: ll + 1].clone()
    return pw_t_all[: ll + 1].clone()


def bl2beam(bl: Tensor | list[float], theta: Tensor | list[float]) -> Tensor:
    """Compute circular beam profile b(theta) from harmonic window b_l."""
    bl_t = torch.as_tensor(bl, dtype=torch.float64)
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    if bl_t.ndim != 1:
        raise ValueError("bl must be 1D")
    if bl_t.numel() == 0:
        raise ValueError("bl must be non-empty")
    x = torch.cos(theta_t.reshape(-1))
    lmax = int(bl_t.numel() - 1)
    p = _legendre_l_sequence(0, x, lmax)
    coeff = ((2.0 * torch.arange(lmax + 1, dtype=torch.float64)) + 1.0) * bl_t / (4.0 * math.pi)
    out = coeff @ p
    return out.reshape(theta_t.shape)


def beam2bl(beam: Tensor | list[float], theta: Tensor | list[float], lmax: int) -> Tensor:
    """Compute harmonic window b_l from circular beam profile b(theta)."""
    b_t = torch.as_tensor(beam, dtype=torch.float64).reshape(-1)
    th_t = torch.as_tensor(theta, dtype=torch.float64).reshape(-1)
    if b_t.numel() != th_t.numel():
        raise ValueError("beam and theta must have matching size")
    if b_t.numel() < 2:
        raise ValueError("beam/theta must contain at least 2 samples")
    ll = int(lmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")

    order = torch.argsort(th_t)
    th_s = th_t.index_select(0, order)
    b_s = b_t.index_select(0, order)
    x = torch.cos(th_s)
    p = _legendre_l_sequence(0, x, ll)
    integrand = p * (b_s * torch.sin(th_s)).unsqueeze(0)
    dth = th_s[1:] - th_s[:-1]
    trap = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dth.unsqueeze(0)
    return 2.0 * math.pi * torch.sum(trap, dim=1)


def _cls_order_pairs(n_fields: int, *, new: bool) -> list[tuple[int, int]]:
    if int(n_fields) <= 0:
        raise ValueError("n_fields must be positive")
    n = int(n_fields)
    pairs: list[tuple[int, int]] = []
    if bool(new):
        for d in range(n):
            for i in range(0, n - d):
                pairs.append((i, i + d))
    else:
        for i in range(n):
            for j in range(i, n):
                pairs.append((i, j))
    return pairs


def _infer_field_count_from_ncls(n_cls: int) -> int:
    if n_cls <= 0:
        raise ValueError("number of spectra must be positive")
    disc = 1 + (8 * n_cls)
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError("cannot infer number of fields from number of spectra")
    n = (root - 1) // 2
    if n * (n + 1) // 2 != n_cls:
        raise ValueError("cannot infer number of fields from number of spectra")
    return int(n)


def _normalize_cls_input(
    cls: Tensor | list[Tensor | list[float] | None] | tuple[Tensor | list[float] | None, ...],
    *,
    new: bool,
) -> tuple[list[Tensor | None], list[tuple[int, int]], int]:
    if isinstance(cls, Tensor):
        if cls.ndim == 1:
            return [cls.to(torch.float64)], [(0, 0)], 1
        if cls.ndim == 2:
            seq: list[Tensor | None] = [cls[i].to(torch.float64) for i in range(int(cls.shape[0]))]
        else:
            raise ValueError("cls tensor must be 1D or 2D")
    elif isinstance(cls, (list, tuple)):
        seq = [None if c is None else torch.as_tensor(c, dtype=torch.float64) for c in cls]
    else:
        arr = torch.as_tensor(cls, dtype=torch.float64)
        if arr.ndim != 1:
            raise ValueError("cls must be a 1D spectrum or a sequence of spectra")
        return [arr], [(0, 0)], 1

    if len(seq) == 1:
        if seq[0] is None:
            raise ValueError("single cls spectrum cannot be None")
        return [seq[0]], [(0, 0)], 1

    if len(seq) == 4:
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1)] if bool(new) else [(0, 0), (0, 1), (1, 1), (2, 2)]
        return seq, pairs, 3

    n_fields = _infer_field_count_from_ncls(len(seq))
    return seq, _cls_order_pairs(n_fields, new=bool(new)), n_fields


def synalm(
    cls: Tensor | list[Tensor | list[float] | None] | tuple[Tensor | list[float] | None, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    *,
    new: bool = False,
    verbose: bool = True,  # noqa: ARG001 - compatibility signature
) -> Tensor:
    """Generate alm coefficient vector(s) from power spectra."""
    seq, pairs, n_fields = _normalize_cls_input(cls, new=bool(new))
    ll_data = -1
    for c in seq:
        if c is not None:
            if c.ndim != 1:
                raise ValueError("each cls spectrum must be 1D")
            ll_data = max(ll_data, int(c.numel()) - 1)
    ll = int(ll_data if (lmax is None or int(lmax) < 0) else lmax)
    if ll < 0:
        raise ValueError("cannot infer lmax from empty cls input")
    mm_in = ll if mmax is None else int(mmax)
    mm = ll if mm_in < 0 else min(mm_in, ll)

    cov = torch.zeros((ll + 1, n_fields, n_fields), dtype=torch.float64)
    for spec, (i, j) in zip(seq, pairs, strict=False):
        if spec is None:
            continue
        take = min(ll + 1, int(spec.numel()))
        if take <= 0:
            continue
        cov[:take, i, j] = spec[:take]
        cov[:take, j, i] = spec[:take]

    nalm = alm_size(ll, mm)
    out = torch.zeros((n_fields, nalm), dtype=torch.complex128)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for ell in range(ll + 1):
        c = 0.5 * (cov[ell] + cov[ell].transpose(0, 1))
        try:
            fac = torch.linalg.cholesky(c)
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(c)
            eigvals = torch.clamp(eigvals, min=0.0)
            fac = eigvecs @ torch.diag(torch.sqrt(eigvals))

        idx0 = alm_index(ell, 0, ll, mm)
        z0 = torch.randn((n_fields,), dtype=torch.float64)
        out[:, idx0] = torch.complex(fac @ z0, torch.zeros((n_fields,), dtype=torch.float64))

        mlim = min(mm, ell)
        for m in range(1, mlim + 1):
            zr = torch.randn((n_fields,), dtype=torch.float64)
            zi = torch.randn((n_fields,), dtype=torch.float64)
            vr = (fac @ zr) * inv_sqrt2
            vi = (fac @ zi) * inv_sqrt2
            out[:, alm_index(ell, m, ll, mm)] = torch.complex(vr, vi)

    if n_fields == 1:
        return out[0]
    return out


def synfast(
    cls: Tensor | list[Tensor | list[float] | None] | tuple[Tensor | list[float] | None, ...],
    nside: int,
    lmax: int | None = None,
    mmax: int | None = None,
    *,
    alm: bool = False,
    pol: bool = True,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float | None = None,
    new: bool = False,
    verbose: bool = True,  # noqa: ARG001 - compatibility signature
) -> Tensor | tuple[Tensor, Tensor]:
    """Synthesize map(s) from input C_l spectrum/spectra."""
    alms = synalm(cls, lmax=lmax, mmax=mmax, new=new, verbose=verbose)
    rows, single = _alm_to_cpu_rows(alms)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if sigma is not None and float(fwhm) != 0.0:
        raise ValueError("provide only one of fwhm or sigma")
    fwhm_use = float(fwhm)
    if sigma is not None:
        fwhm_use = float(sigma) * math.sqrt(8.0 * math.log(2.0))

    pol_eff = bool(pol) and rows.shape[0] == 3
    rows_eff: Tensor = rows
    if pixwin:
        pixwin_loader = globals()["pixwin"]
        if pol_eff:
            pw_t, pw_p = pixwin_loader(int(nside), pol=True, lmax=ll)
            pw_beam = torch.stack([pw_t, pw_p, pw_p], dim=1)
            rows_eff = smoothalm(rows_eff, lmax=ll, mmax=mm, beam=pw_beam, pol=True)
        else:
            pw = pixwin_loader(int(nside), pol=False, lmax=ll)
            rows_eff = smoothalm(rows_eff, lmax=ll, mmax=mm, beam=torch.as_tensor(pw, dtype=torch.float64), pol=False)
        rows_eff, _ = _alm_to_cpu_rows(rows_eff)

    if fwhm_use != 0.0:
        rows_eff = smoothalm(rows_eff, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, pol=pol_eff)
        rows_eff, _ = _alm_to_cpu_rows(rows_eff)

    map_out = alm2map(
        rows_eff[0] if single else rows_eff,
        nside=int(nside),
        lmax=ll,
        mmax=mm,
        nest=False,
        backend="torch",
        pol=pol_eff,
        pixwin=False,
    )
    if alm:
        alm_out = rows_eff[0] if single else rows_eff
        return map_out, alm_out
    return map_out


def smoothalm(
    alm_values: Tensor | list[complex] | list[list[complex]],
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    fwhm_rad: float | None = None,
    beam: Tensor | None = None,
    pol: bool = False,
) -> Tensor:
    """Apply scalar/polarized beam smoothing to alm coefficients."""
    rows, single = _alm_to_cpu_rows(alm_values)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if pol:
        if rows.shape[0] != 3:
            raise ValueError("pol=True requires alm_values shape (3, nalm)")
        if beam is None:
            if fwhm_rad is None:
                raise ValueError("provide either beam or fwhm_rad")
            beam_t = gaussian_beam(float(fwhm_rad), ll, pol=True)
        else:
            beam_t = torch.as_tensor(beam, dtype=torch.float64)
        if beam_t.ndim == 1:
            bt = be = bb = beam_t
        elif beam_t.ndim == 2:
            if beam_t.shape[0] >= ll + 1:
                bt = beam_t[:, 0]
                be = beam_t[:, 1] if beam_t.shape[1] > 1 else beam_t[:, 0]
                bb = beam_t[:, 2] if beam_t.shape[1] > 2 else be
            elif beam_t.shape[1] >= ll + 1:
                bt = beam_t[0]
                be = beam_t[1] if beam_t.shape[0] > 1 else beam_t[0]
                bb = beam_t[2] if beam_t.shape[0] > 2 else be
            else:
                raise ValueError("beam must contain length >= lmax+1")
        else:
            raise ValueError("beam must be 1D or 2D")
        if bt.shape[0] < ll + 1 or be.shape[0] < ll + 1 or bb.shape[0] < ll + 1:
            raise ValueError("beam must contain length >= lmax+1")
        ell = _alm_ell_array(ll, mm)
        st = bt.index_select(0, ell).to(torch.complex128)
        se = be.index_select(0, ell).to(torch.complex128)
        sb = bb.index_select(0, ell).to(torch.complex128)
        return torch.stack([rows[0] * st, rows[1] * se, rows[2] * sb], dim=0)

    if beam is None:
        if fwhm_rad is None:
            raise ValueError("provide either beam or fwhm_rad")
        beam_t = gaussian_beam(float(fwhm_rad), ll)
    else:
        beam_t = torch.as_tensor(beam, dtype=torch.float64)
        if beam_t.ndim != 1 or beam_t.shape[0] < ll + 1:
            raise ValueError("beam must be 1D with length >= lmax+1")
    ell = _alm_ell_array(ll, mm)
    scale = beam_t.index_select(0, ell).to(dtype=torch.complex128)
    out = rows * scale.unsqueeze(0)
    return out[0] if single else out


def smoothmap(
    map_values: Tensor | list[float] | list[list[float]],
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    fwhm_rad: float | None = None,
    sigma: float | None = None,
    beam: Tensor | None = None,
    beam_window: Tensor | None = None,
    pol: bool = False,
    iter: int = 0,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    verbose: bool = True,
    backend: Literal["torch", "healpy"] = "torch",
) -> Tensor:
    """Smooth HEALPix map(s) with scalar or polarized beam."""
    rows, single = _map_to_cpu_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if mm < 0 or mm > ll:
        raise ValueError("invalid mmax")
    if sigma is not None and fwhm_rad is not None:
        raise ValueError("provide only one of fwhm_rad/sigma")
    if beam is not None and beam_window is not None:
        raise ValueError("provide only one of beam/beam_window")
    beam_use = beam if beam_window is None else beam_window
    fwhm_use = fwhm_rad
    if sigma is not None:
        fwhm_use = float(sigma) * math.sqrt(8.0 * math.log(2.0))

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        m_np = rows.detach().cpu().numpy()
        if single:
            m_np = m_np[0]
        out = hp.smoothing(
            m_np,
            fwhm=0.0 if fwhm_use is None else float(fwhm_use),
            sigma=None if sigma is None else float(sigma),
            beam_window=None if beam_use is None else torch.as_tensor(beam_use, dtype=torch.float64).detach().cpu().numpy(),
            pol=bool(pol),
            iter=int(iter),
            lmax=ll,
            mmax=mm,
            use_weights=bool(use_weights),
            use_pixel_weights=bool(use_pixel_weights),
            verbose=bool(verbose),
            nest=bool(nest),
        )
        return torch.from_numpy(np.asarray(out)).to(dtype=torch.float64)

    if use_weights:
        raise ValueError("torch backend does not support use_weights yet")
    if pol:
        if rows.shape[0] != 3:
            raise ValueError("pol=True requires map_values shape (3, npix)")
        alm = map2alm(rows, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=True)
        alm_sm = smoothalm(alm, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, beam=beam_use, pol=True)
        return alm2map(alm_sm, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", pol=True, pixwin=False)

    alm = map2alm(rows, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", iter=iter, pol=False)
    alm_sm = smoothalm(alm, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, beam=beam_use, pol=False)
    sm = alm2map(alm_sm, nside=ns, lmax=ll, mmax=mm, nest=nest, backend="torch", pol=False, pixwin=False)
    return sm[0] if single else sm


def map2alm_spin(
    maps: Tensor,
    spin: int,
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
) -> tuple[Tensor, Tensor]:
    """
    Spin-weighted map->alm transform (Q/U-like pair).

    Input `maps` shape must be `(2, npix)`.
    """
    maps_t = torch.as_tensor(maps, dtype=torch.float64)
    if maps_t.ndim != 2 or maps_t.shape[0] != 2:
        raise ValueError("maps must have shape (2, npix)")
    raw_spin = float(torch.as_tensor(spin).item())
    npix = int(maps_t.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if mm < 0 or mm > ll:
        raise ValueError("invalid mmax")
    try:
        spin_i = int(spin)
    except Exception as exc:
        raise ValueError("spin must be an integer") from exc
    if float(spin_i) != raw_spin:
        raise ValueError("spin must be an integer")
    if spin_i < 0:
        raise ValueError("spin must be non-negative")

    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        arr = maps_t.detach().cpu().numpy()
        if nest:
            arr = np.stack([hp.reorder(arr[0], n2r=True), hp.reorder(arr[1], n2r=True)], axis=0)
        alm1, alm2 = hp.map2alm_spin(arr, spin=spin_i, lmax=ll, mmax=mm)
        return torch.from_numpy(alm1).to(torch.complex128), torch.from_numpy(alm2).to(torch.complex128)

    rows = maps_t.to(device="cpu")
    pix_w = 4.0 * math.pi / float(npix)

    nalm = alm_size(ll, mm)
    ring_mode = os.environ.get("TORCHFITS_SPIN_MAP2ALM_RING_TORCH", "auto")
    bytes_needed = int(npix * nalm * 32)  # two complex matrices at complex128-equivalent footprint
    auto_ring = ring_mode == "auto" and bytes_needed >= _SPIN_MAP2ALM_RING_AUTO_MIN_BYTES
    use_ring_torch = (ring_mode == "force" or ring_mode == "1" or auto_ring) and (_SPIN_RING_NEST_ENABLE or (not nest))
    if use_ring_torch:
        starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(ns)
        rows_ring = _reorder_nest_to_ring_rows(rows, int(ns)) if nest else rows
        p_plus = torch.complex(rows_ring[0], rows_ring[1])
        nrings = (4 * int(ns)) - 1
        bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
        use_concat_fast = (
            bytes_concat <= _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES
            and nalm <= _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM
        )
        use_cpp_fused = (
            _SPIN_MAP2ALM_RING_CONCAT_CPP_ENABLE
            and use_concat_fast
            and _cpp is not None
            and hasattr(_cpp, "_healpix_spin_map2alm_ring_concat_cpu")
            and p_plus.device.type == "cpu"
            and (not p_plus.requires_grad)
        )
        if use_cpp_fused:
            y_plus, y_minus = _ring_spin_basis_concat(int(ns), int(ll), int(mm), int(spin_i))
            phase0_neg = _ring_phi0_phase_for_nside(int(ns), int(mm), sign=-1).to(dtype=torch.complex128)
            c_plus, c_minus = _cpp._healpix_spin_map2alm_ring_concat_cpu(
                p_plus.contiguous(),
                starts_cpu.contiguous(),
                lengths_cpu.contiguous(),
                phase0_neg.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(ll),
                int(mm),
                float(pix_w),
            )
        else:
            s_plus, s_minus = _ring_fourier_modes_spin_conj(
                p_plus,
                starts_cpu=starts_cpu,
                lengths_cpu=lengths_cpu,
                mmax=mm,
                nside=int(ns),
            )

            phase0_neg = _ring_phi0_phase_for_nside(int(ns), int(mm), sign=-1).to(dtype=torch.complex128)
            s_plus = s_plus * phase0_neg
            s_minus = s_minus * phase0_neg

            # Check for C++ Recurrence Backend (Hybrid Approach)
            if (
                s_plus.device.type == "cpu"
                and getattr(_cpp, "_healpix_spin_integrate_recurrence_cpu", None) is not None
                and (not s_plus.requires_grad)
                and os.environ.get("TORCHFITS_SPIN_MAP2ALM_RECURRENCE_CPP", "1") != "0"
            ):
                # Re-fetch theta from cached ring layout and use uniform ring weights.
                _, _, theta_ring, _ = _ring_layout_for_nside(int(ns))
                w_vec = torch.full_like(theta_ring, pix_w)

                alms_stacked = _cpp._healpix_spin_integrate_recurrence_cpu(
                    s_plus, s_minus, theta_ring, w_vec, int(ll), int(mm), int(spin_i)
                )
                c_plus = alms_stacked[0]
                c_minus = alms_stacked[1]

            else:
                c_plus, c_minus = _ring_spin_integrate_blocks(
                    s_plus,
                    s_minus,
                    nside=int(ns),
                    lmax=int(ll),
                    mmax=int(mm),
                    spin=int(spin_i),
                    pix_w=float(pix_w),
                )
    else:
        mats = _spin_map2alm_mats(ns, nest, ll, mm, spin_i)
        if mats is None:
            # Dense cached matrices unavailable (memory cap); use ring path fallback.
            starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(ns)
            p_plus = torch.complex(rows[0], rows[1])
            s_plus, s_minus = _ring_fourier_modes_spin_conj(
                p_plus,
                starts_cpu=starts_cpu,
                lengths_cpu=lengths_cpu,
                mmax=mm,
                nside=int(ns),
            )
            phase0_neg = _ring_phi0_phase_for_nside(int(ns), int(mm), sign=-1).to(dtype=torch.complex128)
            s_plus = s_plus * phase0_neg
            s_minus = s_minus * phase0_neg
            c_plus, c_minus = _ring_spin_integrate_blocks(
                s_plus,
                s_minus,
                nside=int(ns),
                lmax=int(ll),
                mmax=int(mm),
                spin=int(spin_i),
                pix_w=float(pix_w),
            )
        else:
            ycp_t, ycm_t = mats
            use_cpp = (
                _cpp is not None
                and hasattr(_cpp, "_healpix_spin_map2alm_from_basis_cpu")
                and rows.device.type == "cpu"
                and (not maps_t.requires_grad)
                # Torch matmul path is currently faster on Apple CPU; keep C++ path opt-in.
                and os.environ.get("TORCHFITS_SPIN_MAP2ALM_CPP", "0") != "0"
            )
            if use_cpp:
                c_plus, c_minus = _cpp._healpix_spin_map2alm_from_basis_cpu(
                    rows[0].contiguous(),
                    rows[1].contiguous(),
                    ycp_t,
                    ycm_t,
                    float(pix_w),
                )
            else:
                p_plus = torch.complex(rows[0], rows[1])
                p_minus = torch.conj(p_plus)
                c_plus = (p_plus @ ycp_t) * pix_w
                c_minus = (p_minus @ ycm_t) * pix_w

    neg_sign = -1.0 if (spin_i & 1) else 1.0  # (-1)^spin
    a_e = -0.5 * ((neg_sign * c_plus) + c_minus)
    a_b = 0.5j * ((neg_sign * c_plus) - c_minus)
    ell = _alm_ell_array(ll, mm)
    low_ell = ell < abs(spin_i)
    if torch.any(low_ell):
        a_e = a_e.clone()
        a_b = a_b.clone()
        a_e[low_ell] = 0.0
        a_b[low_ell] = 0.0
    return a_e, a_b


def alm2map_spin(
    alms: tuple[Tensor, Tensor],
    nside: int,
    spin: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
) -> Tensor:
    """
    Spin-weighted alm->map transform (Q/U-like pair).

    Returns tensor of shape `(2, npix)`.
    """
    alm1 = torch.as_tensor(alms[0], dtype=torch.complex128)
    alm2 = torch.as_tensor(alms[1], dtype=torch.complex128)
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin:
        raise ValueError("spin must be an integer")
    if spin_i < 0:
        raise ValueError("spin must be non-negative")
    ll, mm = _infer_lmax_mmax_from_nalm(int(alm1.numel()), lmax, mmax)
    if int(alm2.numel()) != alm_size(ll, mm):
        raise ValueError("alm arrays must have matching size")
    if backend == "healpy":
        try:
            import healpy as hp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("healpy backend requested but healpy is not available") from exc
        maps = hp.alm2map_spin(
            [alm1.detach().cpu().numpy(), alm2.detach().cpu().numpy()],
            nside=int(nside),
            spin=spin_i,
            lmax=ll,
            mmax=mm,
        )
        out = torch.from_numpy(np.asarray(maps)).to(torch.float64)
        if nest:
            out = torch.stack(
                [
                    torch.from_numpy(hp.reorder(out[0].cpu().numpy(), r2n=True)).to(dtype=torch.float64),
                    torch.from_numpy(hp.reorder(out[1].cpu().numpy(), r2n=True)).to(dtype=torch.float64),
                ],
                dim=0,
            )
        return out

    npix = _healpix.nside2npix(int(nside))
    neg_sign = -1.0 if (spin_i & 1) else 1.0  # (-1)^spin
    coeff_plus = (alm1 + 1j * alm2) * neg_sign
    coeff_minus = alm1 - 1j * alm2

    nalm = alm_size(ll, mm)
    ring_mode = os.environ.get("TORCHFITS_SPIN_ALM2MAP_RING_TORCH", "auto")
    bytes_needed = int(npix * nalm * 32)  # two complex basis-like work arrays
    auto_ring = ring_mode == "auto" and bytes_needed >= _SPIN_ALM2MAP_RING_AUTO_MIN_BYTES
    use_ring_torch = (ring_mode == "force" or ring_mode == "1" or auto_ring) and (_SPIN_RING_NEST_ENABLE or (not nest))
    if use_ring_torch:
        out_ring = _spin_alm2map_ring_torch(
            coeff_plus,
            coeff_minus,
            nside=int(nside),
            lmax=ll,
            mmax=mm,
            spin=spin_i,
        )
        if torch.isfinite(out_ring).all():
            if nest:
                return _reorder_ring_to_nest_rows(out_ring, int(nside))
            return out_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHFITS_SPIN_ALM2MAP_RING_TORCH=force produced non-finite values; "
                "disable force or lower lmax/nside."
            )

    mats = _spin_alm2map_mats(int(nside), nest, ll, mm, spin_i)
    if mats is not None:
        y_plus, y_minus, y_plus_m0, y_minus_m0 = mats
        use_cpp = (
            _cpp is not None
            and hasattr(_cpp, "_healpix_spin_alm2map_from_basis_cpu")
            and (not alm1.requires_grad)
            and (not alm2.requires_grad)
            # Torch matmul path is currently faster on Apple CPU; keep C++ path opt-in.
            and os.environ.get("TORCHFITS_SPIN_ALM2MAP_CPP", "0") != "0"
        )
        if use_cpp:
            return _cpp._healpix_spin_alm2map_from_basis_cpu(
                coeff_plus.contiguous(),
                coeff_minus.contiguous(),
                y_plus,
                y_minus,
                y_plus_m0,
                y_minus_m0,
            )
        p_plus_pos = coeff_plus @ y_plus
        p_minus_pos = coeff_minus @ y_minus
        if mm == 0:
            p_plus = p_plus_pos
            p_minus = p_minus_pos
        else:
            p_plus_m0 = coeff_plus[: ll + 1] @ y_plus_m0
            p_minus_m0 = coeff_minus[: ll + 1] @ y_minus_m0
            p_plus = p_plus_pos + torch.conj(p_minus_pos - p_minus_m0)
            p_minus = p_minus_pos + torch.conj(p_plus_pos - p_plus_m0)
    else:
        Y_plus, _ = _spin_ylm_basis_concat(int(nside), nest, ll, mm, spin_i)
        Y_minus, _ = _spin_ylm_basis_concat(int(nside), nest, ll, mm, -spin_i)
        
        # P = C @ Y
        # C: (Nalm,), Y: (Nalm, Npix)
        # P: (Npix,)
        # P = C[None, :] @ Y = (1, Nalm) @ (Nalm, Npix) -> (1, Npix) -> squeeze
        # OR torch.mv(Y.T, C)? Y is (Nalm, Npix). Y.T is (Npix, Nalm).
        # P = Y.T @ C
        
        # But we also have the complex conjugation symmetry logic from original code?
        # Original:
        # p_plus = p_plus + p_plus_m
        # p_minus = p_minus + p_minus_m
        # if m > 0:
        #     p_plus = p_plus + torch.conj(p_minus_m)
        #     p_minus = p_minus + torch.conj(p_plus_m)
        
        # This implies cross-contribution for m > 0.
        # This relationship P_- = conj(P_+) holds for real tensor fields.
        # But here we are computing P from alms.
        # The loop handles general case.
        # But wait, p_plus and p_minus summation is:
        # P_+ = sum_{lm} (E_lm + i B_lm) Y_{s,lm}
        # P_- = sum_{lm} (E_lm - i B_lm) Y_{-s,lm}
        # The input `alms` are (E_lm, B_lm). 
        # `coeff_plus` = E + iB = a_{s,lm}
        # `coeff_minus` = E - iB = a_{-s,lm}
        # So P_+ = coeff_plus @ Y_plus
        # P_- = coeff_minus @ Y_minus
        
        # But what about the `if m > 0` part?
        # That logic in original code:
        # `p_plus = p_plus + torch.conj(p_minus_m)`
        # `p_minus_m` comes from `cm @ ym`.
        # `cm` is `coeff_minus` segment. `ym` is `Y_minus` segment.
        # This suggests that we are enforcing real-field constraints?
        # If the map is real, P_- = conj(P_+).
        # But here `alm2map_spin` returns a map.
        # If the stored ALMs are for real map (only m >= 0 stored), then we must add negative m terms.
        # The stored ALMs typically only have m >= 0.
        # The loop logic accounts for negative m by using symmetry:
        # Y_{s, l, -m} = (-1)^{s+m} Y^*_{-s, l, m}
        # a_{s, l, -m} = (-1)^{s+m} a^*_{-s, l, m} (for real fields)
        # So contribution from -m is related to +m.
        # The code: `p_plus += conj(p_minus_m)`
        # `p_minus_m` = a_{-s, l, m} * Y_{-s, l, m}
        # conj(p_minus_m) = a^*_{-s, l, m} * Y^*_{-s, l, m}
        # This matches the symmetry for negative m terms contributing to P_+.
        
        # So we can compute positive m contribution (m >= 0) using one matmul.
        # Then we need to add negative m contribution (m < 0).
        # The negative m contribution comes from `m > 0` terms in the loop.
        
        # Let P_plus_pos = coeff_plus @ Y_plus (sum over m>=0)
        # Let P_minus_pos = coeff_minus @ Y_minus (sum over m>=0)
        
        P_plus_pos = torch.mv(Y_plus.T, coeff_plus)
        P_minus_pos = torch.mv(Y_minus.T, coeff_minus)
        
        # Now add contributions from m < 0.
        # Contribution to P_+ from m<0 is sum_{m<0} a_{s,l,m} Y_{s,l,m}
        # By symmetry: a_{s,l,-m} Y_{s,l,-m} = conj(a_{-s, l, m} Y_{-s, l, m})
        # = conj(contribution to P_- from mode m)
        # = conj(P_minus_m)
        
        # BUT P_minus_pos = sum_{m>=0} P_minus_m.
        # We need sum_{m>0} conj(P_minus_m).
        # We need to subtract m=0 component from P_minus_pos before conjugating?
        # Yes.
        
        # We need P_minus_m=0 separately?
        # m=0 corresponds to the first (lmax+1) elements of the vectors.
        # We can compute m=0 part separately or extract it?
        # Computing separately is safer/easier if we have the block.
        # Or we can just use the indices.
        
        # Let's extract m=0 part of P_minus_pos efficiently?
        # No, P_minus_pos is summed over all m. We can't extract after sum.
        # We must compute m=0 and m>0 separately.
        
        # Or:
        # P_plus_total = P_plus_pos + conj(P_minus_pos - P_minus_{m=0})
        # P_minus_total = P_minus_pos + conj(P_plus_pos - P_plus_{m=0})
        
        # To do this, we need P_minus_{m=0}.
        # This is coeff_minus[0:lmax+1] @ Y_minus[0:lmax+1]
        
        l_count_m0 = ll + 1
        
        P_plus_m0 = torch.mv(Y_plus[:l_count_m0].T, coeff_plus[:l_count_m0])
        P_minus_m0 = torch.mv(Y_minus[:l_count_m0].T, coeff_minus[:l_count_m0])
        
        p_plus = P_plus_pos + torch.conj(P_minus_pos - P_minus_m0)
        p_minus = P_minus_pos + torch.conj(P_plus_pos - P_plus_m0)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    # Align sign convention with healpy Q/U synthesis.
    return torch.stack([(-q).real.to(torch.float64), (-u).real.to(torch.float64)], dim=0)


__all__ = [
    "alm2map",
    "alm2cl",
    "alm2map_spin",
    "alm_index",
    "alm_size",
    "almxfl",
    "anafast",
    "beam2bl",
    "bl2beam",
    "gaussian_beam",
    "map2alm",
    "map2alm_lsq",
    "map2alm_spin",
    "pixwin",
    "smoothalm",
    "smoothmap",
    "synalm",
    "synfast",
]

# End of file
