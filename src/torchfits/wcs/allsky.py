import torch
from torch import Tensor
from typing import Tuple, Optional
import math


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
    # Constants
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi

    # Convert inputs to radians for calculation
    x = xi * d2r
    y = eta * d2r

    phi = torch.zeros_like(xi)
    theta = torch.zeros_like(eta)

    if projection_code == "AIT":
        X = x
        Y = y
        r2 = (X * 0.25) ** 2 + (Y * 0.5) ** 2
        mask = r2 <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        phi_rad = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0)
        sin_theta = torch.clamp(z * Y, -1.0, 1.0)
        theta_rad = torch.asin(sin_theta)
        phi = phi_rad * r2d
        theta = theta_rad * r2d
        phi = torch.where(mask, phi, torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype))
        theta = torch.where(mask, theta, torch.tensor(float("nan"), device=theta.device, dtype=theta.dtype))

    elif projection_code == "MOL":
        X = x
        Y = y
        sqrt2 = math.sqrt(2.0)
        sin_gamma = torch.clamp(Y / sqrt2, -1.0, 1.0)
        gamma = torch.asin(sin_gamma)
        # sin(2x) = 2 sin(x) cos(x)
        t_val = (2.0 * gamma + torch.sin(2.0 * gamma)) / math.pi
        theta_rad = torch.asin(torch.clamp(t_val, -1.0, 1.0))
        denom = 2.0 * sqrt2 * torch.cos(gamma)
        mask_pole = torch.abs(denom) < 1e-8
        phi_rad = torch.where(mask_pole, torch.zeros_like(X), math.pi * X / torch.where(mask_pole, torch.ones_like(denom), denom))
        phi = phi_rad * r2d
        theta = theta_rad * r2d
        mask_valid = torch.abs(Y) <= (sqrt2 + 1e-9)
        phi = torch.where(mask_valid, phi, torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype))
        theta = torch.where(mask_valid, theta, torch.tensor(float("nan"), device=theta.device, dtype=theta.dtype))

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
        eta_boundary = eta_scale * (2.0 / 3.0)
        abs_eta = torch.abs(eta)
        mask_eq = abs_eta <= eta_boundary
        theta = torch.asin(torch.clamp(torch.where(mask_eq, eta / eta_scale, torch.sign(eta) * (1.0 - ((90.0 - abs_eta)/(90.0-eta_boundary))**2 / 3.0)), -1.0, 1.0)) * r2d
        sigma = (90.0 - abs_eta) / (90.0 - eta_boundary)
        xc = torch.round((xi - 45.0) * (1.0 / 90.0)) * 90.0 + 45.0
        phi = torch.where(mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9))
        invalid = (abs_eta > 90.0)
        phi = torch.where(invalid, torch.tensor(float("nan"), device=phi.device), phi)
        theta = torch.where(invalid, torch.tensor(float("nan"), device=theta.device), theta)

    elif projection_code == "SFL":
        theta = eta
        # SFL: phi = xi / cos(theta)
        phi = xi / torch.clamp(torch.cos(theta * 0.017453292519943295), min=1e-12)

    return phi, theta


def deproject_allsky(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward all-sky projection."""
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi

    if projection_code == "AIT":
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        phi_rad = phi_wrapped * d2r
        theta_rad = theta * d2r
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_half_phi = torch.cos(phi_rad * 0.5)
        sin_half_phi = torch.sin(phi_rad * 0.5)
        denom = torch.sqrt(0.5 * (1.0 + cos_theta * cos_half_phi))
        xi = 2.0 * cos_theta * sin_half_phi / denom * r2d
        eta = sin_theta / denom * r2d
        return xi, eta

    elif projection_code == "MOL":
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        phi_rad = phi_wrapped * d2r
        theta_rad = theta * d2r
        sin_theta = torch.sin(theta_rad)
        target = math.pi * sin_theta
        gamma = theta_rad.clone() # Better initial guess than zero
        for _ in range(5):
            sin_2g = torch.sin(2.0 * gamma)
            cos_2g = torch.cos(2.0 * gamma)
            res = 2.0 * gamma + sin_2g - target
            
            # Convergence check to skip work
            if torch.all(torch.abs(res) < 1e-12):
                break
                
            gamma = gamma - res / (2.0 + 2.0 * cos_2g + 1e-12)
        sqrt2 = math.sqrt(2.0)
        xi = 2.0 * sqrt2 * phi_rad * torch.cos(gamma) / math.pi * r2d
        eta = sqrt2 * torch.sin(gamma) * r2d
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
        phi_w = torch.remainder(phi + 180.0, 360.0) - 180.0
        s_theta = torch.sin(theta * d2r)
        abs_s = torch.abs(s_theta)
        eta_scale = 90.0 * (K / H)
        mask_eq = abs_s <= (2.0 / 3.0)
        sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
        eta_pol = torch.sign(s_theta) * (90.0 - (90.0 - (eta_scale * 2.0/3.0)) * sigma)
        xc = torch.round((phi_w - 45.0) / 90.0) * 90.0 + 45.0
        xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
        eta = torch.where(mask_eq, eta_scale * s_theta, eta_pol)
        return xi, eta

    elif projection_code == "SFL":
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        xi = phi_wrapped * torch.cos(theta * d2r)
        eta = theta
        return xi, eta

    return phi, theta
