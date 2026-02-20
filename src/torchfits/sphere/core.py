"""Mathematically strict spherical primitives and sampling utilities."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ..wcs import healpix as _healpix


def _float_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _as_float_tensor(x: Tensor | float) -> Tensor:
    t = torch.as_tensor(x)
    if t.is_floating_point():
        return t
    return t.to(dtype=_float_dtype(t.device))


def wrap_longitude(lon_deg: Tensor | float, *, center_deg: float = 180.0) -> Tensor:
    """
    Wrap longitudes into [center_deg - 180, center_deg + 180).

    Examples:
    - center_deg=180 -> [0, 360)
    - center_deg=0   -> [-180, 180)
    """
    lon_t = _as_float_tensor(lon_deg)
    offset = center_deg - 180.0
    return torch.remainder(lon_t - offset, 360.0) + offset


def lonlat_to_unit_xyz(lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
    """Convert longitude/latitude in degrees to unit Cartesian vectors [..., 3]."""
    lon_t = _as_float_tensor(lon_deg)
    lat_t = _as_float_tensor(lat_deg)
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    x, y, z = _healpix.lonlat_to_xyz(lon_t, lat_t)
    return torch.stack((x, y, z), dim=-1)


def unit_xyz_to_lonlat(vectors: Tensor, *, wrap_center_deg: float = 180.0) -> tuple[Tensor, Tensor]:
    """Convert unit Cartesian vectors [..., 3] to longitude/latitude in degrees."""
    if vectors.shape[-1] != 3:
        raise ValueError("vectors must have last dimension size 3")
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    lon, lat = _healpix.xyz_to_lonlat(x, y, z)
    return wrap_longitude(lon, center_deg=wrap_center_deg), lat


def great_circle_distance(
    lon1_deg: Tensor | float,
    lat1_deg: Tensor | float,
    lon2_deg: Tensor | float,
    lat2_deg: Tensor | float,
    *,
    degrees: bool = True,
) -> Tensor:
    """Return great-circle angular distance using a numerically stable atan2 formulation."""
    v1 = lonlat_to_unit_xyz(lon1_deg, lat1_deg)
    v2 = lonlat_to_unit_xyz(lon2_deg, lat2_deg)
    v1, v2 = torch.broadcast_tensors(v1, v2)
    cross = torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1)
    dot = torch.clamp((v1 * v2).sum(dim=-1), -1.0, 1.0)
    dist_rad = torch.atan2(cross, dot)
    if degrees:
        return torch.rad2deg(dist_rad)
    return dist_rad


def pairwise_angular_distance(
    lon_deg: Tensor,
    lat_deg: Tensor,
    *,
    degrees: bool = True,
) -> Tensor:
    """
    Pairwise angular distance matrix for N sky positions.

    Inputs are 1D tensors of shape [N]. Output has shape [N, N].
    """
    lon_t = _as_float_tensor(lon_deg)
    lat_t = _as_float_tensor(lat_deg)
    if lon_t.ndim != 1 or lat_t.ndim != 1:
        raise ValueError("lon_deg and lat_deg must be 1D tensors")
    if lon_t.shape[0] != lat_t.shape[0]:
        raise ValueError("lon_deg and lat_deg must have same length")

    v = lonlat_to_unit_xyz(lon_t, lat_t)
    dot = torch.clamp(v @ v.transpose(0, 1), -1.0, 1.0)
    dist_rad = torch.acos(dot)
    dist_rad.fill_diagonal_(0.0)
    if degrees:
        return torch.rad2deg(dist_rad)
    return dist_rad


def slerp_lonlat(
    lon1_deg: Tensor | float,
    lat1_deg: Tensor | float,
    lon2_deg: Tensor | float,
    lat2_deg: Tensor | float,
    t: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """
    Spherical linear interpolation between two positions.

    `t=0` returns the start point, `t=1` returns the end point.
    """
    v1 = lonlat_to_unit_xyz(lon1_deg, lat1_deg)
    v2 = lonlat_to_unit_xyz(lon2_deg, lat2_deg)
    t_t = _as_float_tensor(t).to(device=v1.device, dtype=v1.dtype)

    dot = torch.clamp((v1 * v2).sum(dim=-1), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    omega, sin_omega, t_t = torch.broadcast_tensors(omega, sin_omega, t_t)
    v1, v2 = torch.broadcast_tensors(v1, v2)

    near = sin_omega.abs() < 1e-12
    coeff1 = torch.sin((1.0 - t_t) * omega) / torch.where(near, torch.ones_like(sin_omega), sin_omega)
    coeff2 = torch.sin(t_t * omega) / torch.where(near, torch.ones_like(sin_omega), sin_omega)
    interp = coeff1.unsqueeze(-1) * v1 + coeff2.unsqueeze(-1) * v2

    if near.any():
        lerp = (1.0 - t_t).unsqueeze(-1) * v1 + t_t.unsqueeze(-1) * v2
        lerp = lerp / torch.linalg.norm(lerp, dim=-1, keepdim=True).clamp_min(1e-15)
        interp = torch.where(near.unsqueeze(-1), lerp, interp)

    interp = interp / torch.linalg.norm(interp, dim=-1, keepdim=True).clamp_min(1e-15)
    return unit_xyz_to_lonlat(interp, wrap_center_deg=180.0)


def sample_healpix_map(
    values: Tensor,
    nside: int,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    *,
    nest: bool = False,
    interpolation: Literal["nearest", "bilinear"] = "bilinear",
) -> Tensor:
    """
    Sample HEALPix maps on arbitrary sky coordinates.

    `values` shape is `[..., npix]`, output shape is `[..., *coord_shape]`.
    """
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError("interpolation must be one of {'nearest', 'bilinear'}")

    expected_npix = _healpix.nside2npix(nside)
    if values.shape[-1] != expected_npix:
        raise ValueError(f"values last dimension must be npix={expected_npix} for nside={nside}")

    lon_t = _as_float_tensor(lon_deg).to(device=values.device, dtype=_float_dtype(values.device))
    lat_t = _as_float_tensor(lat_deg).to(device=values.device, dtype=_float_dtype(values.device))
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    coord_shape = lon_t.shape
    lon_flat = lon_t.reshape(-1)
    lat_flat = lat_t.reshape(-1)

    vals = values.reshape(-1, values.shape[-1])
    if interpolation == "nearest":
        pix = _healpix.ang2pix(nside, lon_flat, lat_flat, nest=nest, lonlat=True)
        out = vals[:, pix]
    else:
        pix4, w4 = _healpix.get_interp_weights(nside, lon_flat, lat_flat, nest=nest, lonlat=True)
        gathered = vals[:, pix4.reshape(-1)].reshape(vals.shape[0], 4, lon_flat.shape[0])
        weights = w4.to(device=gathered.device, dtype=gathered.dtype)
        out = (gathered * weights.unsqueeze(0)).sum(dim=1)

    return out.reshape(*values.shape[:-1], *coord_shape)


def sample_multiband_healpix(
    cube: Tensor,
    nside: int,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    *,
    nest: bool = False,
    interpolation: Literal["nearest", "bilinear"] = "bilinear",
) -> Tensor:
    """
    Sample multi-band HEALPix cube with shape `[..., n_band, npix]`.

    Output has shape `[..., n_band, *coord_shape]`.
    """
    if cube.ndim < 2:
        raise ValueError("cube must have at least 2 dimensions: [..., n_band, npix]")
    return sample_healpix_map(
        cube,
        nside,
        lon_deg,
        lat_deg,
        nest=nest,
        interpolation=interpolation,
    )


def interpolate_wavelength_axis(
    values: Tensor,
    source_wavelength: Tensor | list[float],
    target_wavelength: Tensor | list[float],
    *,
    axis: int = -1,
) -> Tensor:
    """Linearly interpolate `values` along a wavelength axis."""
    src = torch.as_tensor(source_wavelength, dtype=torch.float64, device=values.device)
    tgt = torch.as_tensor(target_wavelength, dtype=torch.float64, device=values.device)
    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("source_wavelength and target_wavelength must be 1D")
    if src.numel() < 2:
        raise ValueError("source_wavelength must contain at least two values")
    if not bool(torch.all(src[1:] > src[:-1])):
        raise ValueError("source_wavelength must be strictly increasing")

    x = torch.movedim(values, axis, -1)
    if x.shape[-1] != src.numel():
        raise ValueError("values axis size must match source_wavelength length")

    idx_hi = torch.searchsorted(src, tgt, right=False)
    idx_hi = idx_hi.clamp(1, src.numel() - 1)
    idx_lo = idx_hi - 1

    src_lo = src[idx_lo]
    src_hi = src[idx_hi]
    alpha = (tgt - src_lo) / (src_hi - src_lo)

    lo = x.index_select(-1, idx_lo.to(torch.int64))
    hi = x.index_select(-1, idx_hi.to(torch.int64))

    shape = [1] * (x.ndim - 1) + [tgt.shape[0]]
    alpha = alpha.to(dtype=lo.dtype).reshape(shape)
    out = lo + (hi - lo) * alpha
    return torch.movedim(out, -1, axis)


def sample_multiwavelength_healpix(
    cube: Tensor,
    nside: int,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    *,
    source_wavelength: Tensor | list[float] | None = None,
    target_wavelength: Tensor | list[float] | None = None,
    nest: bool = False,
    interpolation: Literal["nearest", "bilinear"] = "bilinear",
) -> Tensor:
    """
    Sample a multiwavelength HEALPix cube and optionally resample wavelengths.

    `cube` shape: `[..., n_wave, npix]`
    Output shape without resampling: `[..., n_wave, *coord_shape]`
    """
    sampled = sample_multiband_healpix(
        cube,
        nside,
        lon_deg,
        lat_deg,
        nest=nest,
        interpolation=interpolation,
    )
    if target_wavelength is None:
        return sampled
    if source_wavelength is None:
        raise ValueError("source_wavelength is required when target_wavelength is provided")

    lon_t = _as_float_tensor(lon_deg)
    lat_t = _as_float_tensor(lat_deg)
    coord_ndim = torch.broadcast_tensors(lon_t, lat_t)[0].ndim
    spectral_axis = sampled.ndim - coord_ndim - 1
    return interpolate_wavelength_axis(
        sampled,
        source_wavelength=source_wavelength,
        target_wavelength=target_wavelength,
        axis=spectral_axis,
    )


def fit_monopole_dipole(
    map_values: Tensor,
    nside: int,
    *,
    nest: bool = False,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Fit monopole+dipole model to a HEALPix map.

    Model: m(pix) = a + d_x*x + d_y*y + d_z*z
    Returns `(monopole, dipole_xyz)`.
    """
    vals = torch.as_tensor(map_values)
    if vals.ndim != 1:
        raise ValueError("map_values must be 1D with shape (npix,)")
    npix = vals.shape[0]
    expected = _healpix.nside2npix(nside)
    if npix != expected:
        raise ValueError(f"map_values length {npix} does not match nside={nside} (npix={expected})")

    vals = vals.to(dtype=torch.float64)
    pix = torch.arange(npix, dtype=torch.int64, device=vals.device)
    x, y, z = _healpix.pix2vec(nside, pix, nest=nest)
    design = torch.stack((torch.ones_like(x), x, y, z), dim=1)

    if valid_mask is None:
        valid = torch.isfinite(vals)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=vals.device)
        if valid.shape != vals.shape:
            raise ValueError("valid_mask must have shape (npix,)")
        valid = valid & torch.isfinite(vals)
    if int(valid.sum().item()) < 4:
        raise ValueError("need at least 4 valid pixels to fit monopole+dipole")

    sol = torch.linalg.lstsq(design[valid], vals[valid].unsqueeze(1)).solution.squeeze(1)
    return sol[0], sol[1:4]


def remove_monopole_dipole(
    map_values: Tensor,
    nside: int,
    *,
    nest: bool = False,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Subtract fitted monopole+dipole model from a HEALPix map."""
    vals = torch.as_tensor(map_values)
    if vals.ndim != 1:
        raise ValueError("map_values must be 1D with shape (npix,)")
    npix = vals.shape[0]
    expected = _healpix.nside2npix(nside)
    if npix != expected:
        raise ValueError(f"map_values length {npix} does not match nside={nside} (npix={expected})")

    mono, dip = fit_monopole_dipole(vals, nside, nest=nest, valid_mask=valid_mask)
    pix = torch.arange(npix, dtype=torch.int64, device=vals.device)
    x, y, z = _healpix.pix2vec(nside, pix, nest=nest)
    model = mono + (dip[0] * x) + (dip[1] * y) + (dip[2] * z)
    return vals.to(dtype=model.dtype) - model


__all__ = [
    "fit_monopole_dipole",
    "great_circle_distance",
    "interpolate_wavelength_axis",
    "lonlat_to_unit_xyz",
    "pairwise_angular_distance",
    "remove_monopole_dipole",
    "sample_healpix_map",
    "sample_multiband_healpix",
    "sample_multiwavelength_healpix",
    "slerp_lonlat",
    "unit_xyz_to_lonlat",
    "wrap_longitude",
]
