"""Compatibility-focused spherical API surface."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Literal

import torch
from torch import Tensor

from ..wcs import healpix as _healpix
from .core import fit_monopole_dipole as _fit_monopole_dipole
from .geom import query_ellipse as _query_ellipse
from .spectral import alm2map as _alm2map
from .spectral import alm2cl as _alm2cl
from .spectral import alm2map_spin as _alm2map_spin
from .spectral import almxfl as _almxfl
from .spectral import anafast as _anafast
from .spectral import beam2bl as _beam2bl
from .spectral import bl2beam as _bl2beam
from .spectral import gaussian_beam as _gaussian_beam
from .spectral import map2alm as _map2alm
from .spectral import map2alm_lsq as _map2alm_lsq
from .spectral import map2alm_spin as _map2alm_spin
from .spectral import pixwin as _pixwin
from .spectral import smoothalm as _smoothalm
from .spectral import smoothmap as _smoothmap
from .spectral import synalm as _synalm
from .spectral import synfast as _synfast


_STRICT_MODE = False


def set_strict_mode(enabled: bool) -> None:
    """Set global strict compatibility mode."""
    global _STRICT_MODE
    _STRICT_MODE = bool(enabled)


def get_strict_mode() -> bool:
    """Return global strict compatibility mode."""
    return _STRICT_MODE


@contextmanager
def strict_mode(enabled: bool = True):
    """Context manager for temporary strict compatibility mode."""
    prev = get_strict_mode()
    set_strict_mode(enabled)
    try:
        yield
    finally:
        set_strict_mode(prev)


def _resolve_strict(strict: bool | None) -> bool:
    if strict is None:
        return get_strict_mode()
    return bool(strict)


def ang2pix(
    nside: int, theta: Tensor, phi: Tensor, nest: bool = False, lonlat: bool = False
) -> Tensor:
    return _healpix.ang2pix(nside, theta, phi, nest=nest, lonlat=lonlat)


def pix2ang(
    nside: int, ipix: Tensor, nest: bool = False, lonlat: bool = False
) -> tuple[Tensor, Tensor]:
    return _healpix.pix2ang(nside, ipix, nest=nest, lonlat=lonlat)


def get_all_neighbors(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    return _healpix.get_all_neighbours(nside, theta, phi=phi, nest=nest, lonlat=lonlat)


def get_all_neighbours(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    return _healpix.get_all_neighbours(nside, theta, phi=phi, nest=nest, lonlat=lonlat)


def _require_hpgeom_for_strict(func_name: str):
    try:
        import hpgeom as hpg
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"strict compat.{func_name} requires optional dependency 'hpgeom'"
        ) from exc
    return hpg


def query_circle(
    nside: int,
    lon: Tensor | float,
    lat: Tensor | float,
    radius: float,
    *,
    degrees: bool = True,
    inclusive: bool = False,
    nest: bool = False,
    chunk_size: int = 1_000_000,
    fact: int = 4,
    strict: bool | None = None,
) -> Tensor:
    if _resolve_strict(strict):
        hpg = _require_hpgeom_for_strict("query_circle")
        out = hpg.query_circle(
            nside,
            float(torch.as_tensor(lon).item()),
            float(torch.as_tensor(lat).item()),
            float(radius),
            inclusive=bool(inclusive),
            fact=int(fact),
            nest=bool(nest),
            lonlat=True,
            degrees=bool(degrees),
        )
        return torch.as_tensor(out, dtype=torch.int64)
    return _healpix.query_circle(
        nside,
        lon,
        lat,
        radius,
        degrees=degrees,
        inclusive=inclusive,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_circle_vec(
    nside: int,
    vec: Tensor,
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,
    nest: bool = False,
    chunk_size: int = 1_000_000,
    strict: bool | None = None,
) -> Tensor:
    if _resolve_strict(strict):
        hpg = _require_hpgeom_for_strict("query_circle_vec")
        out = hpg.query_circle_vec(
            nside,
            torch.as_tensor(vec, dtype=torch.float64).detach().cpu().numpy(),
            float(radius),
            inclusive=bool(inclusive),
            fact=int(fact),
            nest=bool(nest),
        )
        return torch.as_tensor(out, dtype=torch.int64)
    return _healpix.query_circle_vec(
        nside,
        vec,
        radius,
        inclusive=inclusive,
        fact=fact,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_ellipse(
    nside: int,
    a: float | Tensor,
    b: float | Tensor,
    semi_major: float,
    semi_minor: float,
    alpha: float | None = None,
    *,
    pa_deg: float | None = None,
    nest: bool = False,
    inclusive: bool = False,
    fact: int = 4,
    strict: bool | None = None,
    backend: Literal["auto", "torch", "hpgeom"] = "auto",
    lonlat: bool = True,
    degrees: bool = True,
    return_pixel_ranges: bool = False,
) -> Tensor:
    if backend not in {"auto", "torch", "hpgeom"}:
        raise ValueError("backend must be one of {'auto', 'torch', 'hpgeom'}")
    fact_i = int(fact)
    if fact_i < 1:
        raise ValueError("fact must be >= 1")
    if return_pixel_ranges and not nest:
        raise ValueError("return_pixel_ranges is only supported with nest=True")

    strict_eff = _resolve_strict(strict)
    use_hpgeom = False
    hpg = None
    if strict_eff or backend == "hpgeom":
        hpg = _require_hpgeom_for_strict("query_ellipse")
        use_hpgeom = True
    elif backend == "auto":
        try:
            import hpgeom as hpg

            use_hpgeom = True
        except Exception:
            use_hpgeom = False

    if pa_deg is not None and alpha is not None:
        raise ValueError("set only one of alpha/pa_deg")
    if alpha is None:
        alpha_in = 0.0 if pa_deg is None else float(pa_deg)
        alpha_units = (
            math.radians(alpha_in) if (not lonlat or not degrees) else alpha_in
        )
        pa_deg_eff = alpha_in
    else:
        alpha_units = float(alpha)
        pa_deg_eff = alpha_units if (lonlat and degrees) else math.degrees(alpha_units)

    if use_hpgeom:
        assert hpg is not None
        out = hpg.query_ellipse(
            nside,
            float(torch.as_tensor(a).item()),
            float(torch.as_tensor(b).item()),
            float(semi_major),
            float(semi_minor),
            float(alpha_units),
            inclusive=bool(inclusive),
            fact=fact_i,
            nest=bool(nest),
            lonlat=bool(lonlat),
            degrees=bool(degrees),
            return_pixel_ranges=bool(return_pixel_ranges),
        )
        if return_pixel_ranges:
            return torch.as_tensor(out, dtype=torch.int64).reshape(-1, 2)
        return torch.as_tensor(out, dtype=torch.int64).reshape(-1)

    if lonlat:
        if degrees:
            lon_deg = float(torch.as_tensor(a).item())
            lat_deg = float(torch.as_tensor(b).item())
            semi_major_deg = float(semi_major)
            semi_minor_deg = float(semi_minor)
        else:
            lon_deg = math.degrees(float(torch.as_tensor(a).item()))
            lat_deg = math.degrees(float(torch.as_tensor(b).item()))
            semi_major_deg = math.degrees(float(semi_major))
            semi_minor_deg = math.degrees(float(semi_minor))
    else:
        theta = float(torch.as_tensor(a).item())
        phi = float(torch.as_tensor(b).item())
        lon_deg = math.degrees(phi)
        lat_deg = 90.0 - math.degrees(theta)
        semi_major_deg = math.degrees(float(semi_major))
        semi_minor_deg = math.degrees(float(semi_minor))

    if inclusive and fact_i > 1:
        # Approximate overlap semantics by supersampling at fact*nside then
        # projecting matched child pixels back to parent pixels.
        if nest and (fact_i & (fact_i - 1)) != 0:
            raise ValueError("for nest ordering, fact must be a power of two")
        nside_hi = int(nside) * fact_i
        if _healpix.isnsideok(nside_hi):
            pix_hi = _query_ellipse(
                nside_hi,
                lon_deg,
                lat_deg,
                semi_major_deg,
                semi_minor_deg,
                pa_deg=pa_deg_eff,
                nest=True,
                inclusive=True,
            )
            if pix_hi.numel() == 0:
                return (
                    _healpix.pixels_to_pixel_ranges(pix_hi)
                    if return_pixel_ranges
                    else pix_hi
                )
            parent_nest = torch.unique(
                torch.div(pix_hi, fact_i * fact_i, rounding_mode="floor")
            )
            parent_nest = torch.sort(parent_nest).values
            pix = (
                parent_nest
                if nest
                else torch.sort(_healpix.nest2ring(nside, parent_nest)).values
            )
            return _healpix.pixels_to_pixel_ranges(pix) if return_pixel_ranges else pix
    pix = _query_ellipse(
        nside,
        lon_deg,
        lat_deg,
        semi_major_deg,
        semi_minor_deg,
        pa_deg=pa_deg_eff,
        nest=nest,
        inclusive=inclusive,
    )
    return _healpix.pixels_to_pixel_ranges(pix) if return_pixel_ranges else pix


def fit_monopole(
    map_values: Tensor,
    *,
    nest: bool = False,
    bad: float = _healpix.UNSEEN,
    gal_cut: float = 0.0,
) -> Tensor:
    """Fit monopole term with healpy-like masking semantics."""
    vals = torch.as_tensor(map_values)
    if vals.ndim != 1:
        raise ValueError("map_values must be 1D with shape (npix,)")
    nside = _healpix.npix2nside(int(vals.numel()))
    valid = torch.isfinite(vals) & (vals != bad)
    if gal_cut > 0.0:
        pix = torch.arange(vals.numel(), dtype=torch.int64, device=vals.device)
        _, lat = _healpix.pix2ang(nside, pix, nest=nest, lonlat=True)
        valid = valid & (torch.abs(lat) >= float(gal_cut))
    if int(valid.sum().item()) == 0:
        raise ValueError("no valid pixels left after masking")
    return vals[valid].to(torch.float64).mean()


def remove_monopole(
    map_values: Tensor,
    *,
    nest: bool = False,
    bad: float = _healpix.UNSEEN,
    gal_cut: float = 0.0,
    fitval: bool = False,
    copy: bool = True,
    verbose: bool = True,  # noqa: ARG001 - compatibility signature
) -> Tensor | tuple[Tensor, Tensor]:
    """Remove fitted monopole with healpy-like masking semantics."""
    vals = torch.as_tensor(map_values)
    if vals.ndim != 1:
        raise ValueError("map_values must be 1D with shape (npix,)")
    mono = fit_monopole(vals, nest=nest, bad=bad, gal_cut=gal_cut)
    out = vals.clone() if copy else vals

    valid = torch.isfinite(out) & (out != bad)
    out[valid] = out[valid] - mono.to(dtype=out.dtype)

    if fitval:
        return out, mono
    return out


def fit_dipole(
    map_values: Tensor,
    *,
    nest: bool = False,
    bad: float = _healpix.UNSEEN,
    gal_cut: float = 0.0,
    nside: int | None = None,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Fit monopole and dipole to a HEALPix map.

    Returns `(monopole, dipole_xyz)`.
    """
    vals = torch.as_tensor(map_values)
    if nside is None:
        nside = _healpix.npix2nside(int(vals.numel()))
    valid = torch.isfinite(vals) & (vals != bad)
    if gal_cut > 0.0:
        pix = torch.arange(vals.numel(), dtype=torch.int64, device=vals.device)
        _, lat = _healpix.pix2ang(nside, pix, nest=nest, lonlat=True)
        valid = valid & (torch.abs(lat) >= float(gal_cut))
    if valid_mask is not None:
        mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=vals.device)
        if mask_t.shape != vals.shape:
            raise ValueError("valid_mask must have shape (npix,)")
        valid = valid & mask_t
    return _fit_monopole_dipole(vals, nside, nest=nest, valid_mask=valid)


def remove_dipole(
    map_values: Tensor,
    *,
    nest: bool = False,
    bad: float = _healpix.UNSEEN,
    gal_cut: float = 0.0,
    fitval: bool = False,
    copy: bool = True,
    verbose: bool = True,  # noqa: ARG001 - compatibility signature
    nside: int | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Remove fitted monopole+dipole from a HEALPix map."""
    vals = torch.as_tensor(map_values)
    if nside is None:
        nside = _healpix.npix2nside(int(vals.numel()))
    mono, dip = fit_dipole(
        vals,
        nside=nside,
        nest=nest,
        bad=bad,
        gal_cut=gal_cut,
        valid_mask=valid_mask,
    )

    out = vals.clone() if copy else vals
    pix = torch.arange(out.numel(), dtype=torch.int64, device=out.device)
    x, y, z = _healpix.pix2vec(nside, pix, nest=nest)
    model = mono + (dip[0] * x) + (dip[1] * y) + (dip[2] * z)
    writable = torch.isfinite(out) & (out != bad)
    out[writable] = out[writable] - model[writable].to(dtype=out.dtype)

    if fitval:
        return out, mono, dip
    return out


def map2alm(
    map_values: Tensor,
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
    return _map2alm(
        map_values,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        backend=backend,
        iter=iter,
        pol=pol,
        use_pixel_weights=use_pixel_weights,
    )


def map2alm_lsq(
    maps: Tensor,
    lmax: int,
    mmax: int | None = None,
    pol: bool = True,
    tol: float = 1e-10,
    maxiter: int = 20,
    *,
    nside: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    use_pixel_weights: bool = False,
) -> tuple[Tensor, float, int]:
    return _map2alm_lsq(
        maps,
        lmax=lmax,
        mmax=mmax,
        nside=nside,
        pol=pol,
        tol=tol,
        maxiter=maxiter,
        nest=nest,
        backend=backend,
        use_pixel_weights=use_pixel_weights,
    )


def almxfl(
    alm: Tensor,
    fl: Tensor,
    mmax: int | None = None,
    inplace: bool = False,
) -> Tensor:
    return _almxfl(alm, fl, mmax=mmax, inplace=inplace)


def alm2cl(
    alms1: Tensor,
    alms2: Tensor | None = None,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
) -> Tensor:
    return _alm2cl(
        alms1, alms2=alms2, lmax=lmax, mmax=mmax, lmax_out=lmax_out, nspec=nspec
    )


def alm2map(
    alm_values: Tensor,
    nside: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
    pol: bool = False,
    pixwin: bool = False,
) -> Tensor:
    return _alm2map(
        alm_values,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        backend=backend,
        pol=pol,
        pixwin=pixwin,
    )


def anafast(
    map1: Tensor,
    map2: Tensor | None = None,
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
    return _anafast(
        map1,
        map2=map2,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        backend=backend,
        nspec=nspec,
        iter=iter,
        alm=alm,
        pol=pol,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
    )


def gaussian_beam(fwhm_rad: float, lmax: int, *, pol: bool = False) -> Tensor:
    return _gaussian_beam(fwhm_rad, lmax, pol=pol)


def gauss_beam(fwhm: float, lmax: int = 512, pol: bool = False) -> Tensor:
    return _gaussian_beam(fwhm, lmax, pol=pol)


def pixwin(
    nside: int, pol: bool = False, lmax: int | None = None
) -> Tensor | tuple[Tensor, Tensor]:
    return _pixwin(nside, pol=pol, lmax=lmax)


def bl2beam(bl: Tensor, theta: Tensor) -> Tensor:
    return _bl2beam(bl, theta)


def beam2bl(beam: Tensor, theta: Tensor, lmax: int) -> Tensor:
    return _beam2bl(beam, theta, lmax)


def synalm(
    cls: Tensor
    | list[Tensor | list[float] | None]
    | tuple[Tensor | list[float] | None, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    *,
    new: bool = False,
    verbose: bool = True,
) -> Tensor:
    return _synalm(cls, lmax=lmax, mmax=mmax, new=new, verbose=verbose)


def synfast(
    cls: Tensor
    | list[Tensor | list[float] | None]
    | tuple[Tensor | list[float] | None, ...],
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
    verbose: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    return _synfast(
        cls,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        alm=alm,
        pol=pol,
        pixwin=pixwin,
        fwhm=fwhm,
        sigma=sigma,
        new=new,
        verbose=verbose,
    )


def smoothalm(
    alm_values: Tensor,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    fwhm_rad: float | None = None,
    beam: Tensor | None = None,
    pol: bool = False,
) -> Tensor:
    return _smoothalm(
        alm_values, lmax=lmax, mmax=mmax, fwhm_rad=fwhm_rad, beam=beam, pol=pol
    )


def smoothmap(
    map_values: Tensor,
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
    return _smoothmap(
        map_values,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        fwhm_rad=fwhm_rad,
        sigma=sigma,
        beam=beam,
        beam_window=beam_window,
        pol=pol,
        iter=iter,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        verbose=verbose,
        backend=backend,
    )


def smoothing(
    map_in: Tensor,
    fwhm: float = 0.0,
    sigma: float | None = None,
    beam_window: Tensor | None = None,
    pol: bool = True,
    iter: int = 3,
    lmax: int | None = None,
    mmax: int | None = None,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    verbose: bool = True,
    nest: bool = False,
    backend: Literal["torch", "healpy"] = "torch",
) -> Tensor:
    map_t = torch.as_tensor(map_in)
    pol_eff = bool(pol) and (map_t.ndim == 2 and map_t.shape[0] == 3)
    return _smoothmap(
        map_in,
        nside=None,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        fwhm_rad=fwhm,
        sigma=sigma,
        beam_window=beam_window,
        pol=pol_eff,
        iter=iter,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        verbose=verbose,
        backend=backend,
    )


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
    return _map2alm_spin(
        maps, spin, nside=nside, lmax=lmax, mmax=mmax, nest=nest, backend=backend
    )


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
    return _alm2map_spin(
        alms, nside, spin, lmax=lmax, mmax=mmax, nest=nest, backend=backend
    )


# Re-export common APIs with familiar naming.
ang2vec = _healpix.ang2vec
boundaries = _healpix.boundaries
get_interp_val = _healpix.get_interp_val
get_interp_weights = _healpix.get_interp_weights
get_map_size = _healpix.get_map_size
get_nside = _healpix.get_nside
isnpixok = _healpix.isnpixok
isnsideok = _healpix.isnsideok
max_pixrad = _healpix.max_pixrad
nest2ring = _healpix.nest2ring
npix2nside = _healpix.npix2nside
nside2npix = _healpix.nside2npix
nside2order = _healpix.nside2order
order2nside = _healpix.order2nside
pix2vec = _healpix.pix2vec
pixel_ranges_to_pixels = _healpix.pixel_ranges_to_pixels
pixels_to_pixel_ranges = _healpix.pixels_to_pixel_ranges
query_box = _healpix.query_box
query_disc = _healpix.query_disc
query_polygon = _healpix.query_polygon
query_polygon_vec = _healpix.query_polygon_vec
query_strip = _healpix.query_strip
reorder = _healpix.reorder
ring2nest = _healpix.ring2nest
ud_grade = _healpix.ud_grade
upgrade_pixel_ranges = _healpix.upgrade_pixel_ranges
vec2ang = _healpix.vec2ang
vec2pix = _healpix.vec2pix


__all__ = [
    "ang2pix",
    "ang2vec",
    "boundaries",
    "map2alm",
    "map2alm_lsq",
    "almxfl",
    "alm2cl",
    "alm2map",
    "anafast",
    "gaussian_beam",
    "gauss_beam",
    "pixwin",
    "bl2beam",
    "beam2bl",
    "synalm",
    "synfast",
    "fit_dipole",
    "fit_monopole",
    "get_all_neighbors",
    "get_all_neighbours",
    "get_strict_mode",
    "get_interp_val",
    "get_interp_weights",
    "get_map_size",
    "get_nside",
    "isnpixok",
    "isnsideok",
    "max_pixrad",
    "nest2ring",
    "npix2nside",
    "nside2npix",
    "nside2order",
    "order2nside",
    "pix2ang",
    "pix2vec",
    "pixel_ranges_to_pixels",
    "pixels_to_pixel_ranges",
    "query_box",
    "query_circle",
    "query_circle_vec",
    "query_disc",
    "query_ellipse",
    "query_polygon",
    "query_polygon_vec",
    "query_strip",
    "reorder",
    "smoothalm",
    "smoothmap",
    "smoothing",
    "map2alm_spin",
    "alm2map_spin",
    "remove_dipole",
    "remove_monopole",
    "ring2nest",
    "set_strict_mode",
    "strict_mode",
    "ud_grade",
    "upgrade_pixel_ranges",
    "vec2ang",
    "vec2pix",
]
