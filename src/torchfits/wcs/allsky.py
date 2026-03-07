import torch
from torch import Tensor
from typing import Tuple, Optional
import math

D2R = 0.017453292519943295
R2D = 57.29577951308232
SQRT2 = 1.4142135623730951
_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


def _sincos(x: Tensor) -> Tuple[Tensor, Tensor]:
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)
    return torch.sin(x), torch.cos(x)


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    if angle.requires_grad:
        return torch.remainder(angle + 180.0, 360.0) - 180.0
    mn = torch.amin(angle)
    mx = torch.amax(angle)
    if bool((mn >= -180.0) and (mx < 180.0)):
        return angle
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def project_allsky(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to native spherical coordinates (phi, theta).
    NOTE: This is the INVERSE projection (Plane -> Sphere) relative to standard FITS definition.

    xi, eta: Degrees (Standard FITS Intermediate).
    Returns: phi, theta (Degrees)
    """
    if projection_code == "AIT":
        x = xi * D2R
        y = eta * D2R
        X = x
        Y = y
        r2 = torch.square(X * 0.25) + torch.square(Y * 0.5)
        mask = r2 <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        phi_rad = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0)
        sin_theta = torch.clamp(z * Y, -1.0, 1.0)
        theta_rad = torch.asin(sin_theta)
        phi = phi_rad * R2D
        theta = theta_rad * R2D
        nan_phi = torch.full_like(phi, float("nan"))
        nan_theta = torch.full_like(theta, float("nan"))
        phi = torch.where(mask, phi, nan_phi)
        theta = torch.where(mask, theta, nan_theta)

    elif projection_code == "MOL":
        x = xi * D2R
        y = eta * D2R
        X = x
        Y = y
        valid = torch.abs(Y) <= (SQRT2 + 1e-9)
        sin_gamma = torch.clamp(Y / SQRT2, -1.0, 1.0)
        gamma = torch.asin(sin_gamma)
        cos_gamma = torch.cos(gamma)
        # sin(2x) = 2 sin(x) cos(x)
        t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / math.pi
        theta_rad = torch.asin(torch.clamp(t_val, -1.0, 1.0))
        theta = theta_rad * R2D
        if bool(valid.all()):
            denom = torch.clamp(2.0 * SQRT2 * cos_gamma, min=1e-8)
            phi = (math.pi * X / denom) * R2D
        else:
            denom = 2.0 * SQRT2 * cos_gamma
            mask_pole = torch.abs(denom) < 1e-8
            phi_rad = torch.where(
                mask_pole,
                torch.zeros_like(X),
                math.pi * X / torch.where(mask_pole, torch.ones_like(denom), denom),
            )
            phi = phi_rad * R2D
            if bool((~valid).any()):
                nan_phi = torch.full_like(phi, float("nan"))
                nan_theta = torch.full_like(theta, float("nan"))
                phi = torch.where(valid, phi, nan_phi)
                theta = torch.where(valid, theta, nan_theta)

    elif projection_code == "HPX":
        H = 4.0
        K = 3.0
        if pv1 is not None:
            if pv1[1] != 0:
                H = pv1[1]
            if pv1[2] != 0:
                K = pv1[2]
        if pv2 is not None:
            if pv2[1] != 0:
                H = pv2[1]
            if pv2[2] != 0:
                K = pv2[2]

        eta_scale = 90.0 * (K / H)
        inv_eta_scale = 1.0 / eta_scale
        eta_boundary = eta_scale * (2.0 / 3.0)
        inv_polar_denom = 1.0 / (90.0 - eta_boundary)
        abs_eta = torch.abs(eta)
        mask_eq = abs_eta <= eta_boundary
        sigma = (90.0 - abs_eta) * inv_polar_denom
        s_theta = torch.where(
            mask_eq,
            eta * inv_eta_scale,
            torch.sign(eta) * (1.0 - (sigma * sigma) * (1.0 / 3.0)),
        )
        theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * R2D
        xc = torch.round((xi - 45.0) * (1.0 / 90.0)) * 90.0 + 45.0
        phi = torch.where(mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9))
        invalid = abs_eta > 90.0
        if bool(invalid.any()):
            nan_phi = torch.full_like(phi, float("nan"))
            nan_theta = torch.full_like(theta, float("nan"))
            phi = torch.where(invalid, nan_phi, phi)
            theta = torch.where(invalid, nan_theta, theta)

    elif projection_code == "SFL":
        theta = eta
        # SFL: phi = xi / cos(theta)
        phi = xi / torch.clamp(torch.cos(theta * D2R), min=1e-12)
    else:
        phi, theta = xi, eta

    return phi, theta


def deproject_allsky(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward all-sky projection."""
    if projection_code == "AIT":
        phi_wrapped = _wrap_lon180_checked(phi)
        phi_rad = phi_wrapped * D2R
        theta_rad = theta * D2R
        sin_theta, cos_theta = _sincos(theta_rad)
        sin_half_phi, cos_half_phi = _sincos(phi_rad * 0.5)
        denom = torch.sqrt(0.5 * (1.0 + cos_theta * cos_half_phi))
        xi = 2.0 * cos_theta * sin_half_phi / denom * R2D
        eta = sin_theta / denom * R2D
        return xi, eta

    elif projection_code == "MOL":
        phi_wrapped = _wrap_lon180_checked(phi)
        phi_rad = phi_wrapped * D2R
        theta_rad = theta * D2R
        sin_theta = torch.sin(theta_rad)
        target = math.pi * sin_theta
        gamma = theta_rad.clone()  # Better initial guess than zero
        for _ in range(5):
            sin_2g = torch.sin(2.0 * gamma)
            cos_2g = torch.cos(2.0 * gamma)
            res = 2.0 * gamma + sin_2g - target

            # Convergence check to skip work
            if torch.all(torch.abs(res) < 1e-12):
                break

            gamma = gamma - res / (2.0 + 2.0 * cos_2g + 1e-12)
        xi = 2.0 * SQRT2 * phi_rad * torch.cos(gamma) / math.pi * R2D
        eta = SQRT2 * torch.sin(gamma) * R2D
        return xi, eta

    elif projection_code == "HPX":
        H, K = 4.0, 3.0
        if pv1 is not None:
            if pv1[1] != 0:
                H = pv1[1]
            if pv1[2] != 0:
                K = pv1[2]
        if pv2 is not None:
            if pv2[1] != 0:
                H = pv2[1]
            if pv2[2] != 0:
                K = pv2[2]
        phi_w = _wrap_lon180_checked(phi)
        s_theta = torch.sin(theta * D2R)
        abs_s = torch.abs(s_theta)
        eta_scale = 90.0 * (K / H)
        mask_eq = abs_s <= (2.0 / 3.0)
        sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
        eta_pol = torch.sign(s_theta) * (
            90.0 - (90.0 - (eta_scale * 2.0 / 3.0)) * sigma
        )
        xc = torch.round((phi_w - 45.0) / 90.0) * 90.0 + 45.0
        xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
        eta = torch.where(mask_eq, eta_scale * s_theta, eta_pol)
        return xi, eta

    elif projection_code == "SFL":
        phi_wrapped = _wrap_lon180_checked(phi)
        xi = phi_wrapped * torch.cos(theta * D2R)
        eta = theta
        return xi, eta

    return phi, theta
