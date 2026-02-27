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
        # Hammer-Aitoff
        # FITS Paper II, Sect 5.4
        # Inverse equations (x, y -> phi, theta)
        # z = sqrt(1 - (x/4)^2 - (y/2)^2)
        # phi = 2 * atan2( z*x, 2*z^2 - 1 )
        # theta = asin( z*y )
        # x, y here are normalized?
        # FITS standard:
        # x = 2 * gamma * cos(theta) * sin(phi/2) / sinc(alpha)
        # y = gamma * sin(theta) / sinc(alpha)
        #
        # WCSLIB implementation (inv):
        # rho = sqrt((x/2)^2 + y^2)
        # if rho > 1: return error (outside ellipse)
        # z = sqrt(1 - (x/4)^2 - (y/2)^2)
        # But wait, standard AIT is scaled.
        # X range [-180, 180]? No.
        # X range [-2*sqrt(2), 2*sqrt(2)]?
        # FITS "AIT" assumes x, y are in degrees?
        # Standard formulas assume radians.
        # If xi, eta in degrees, convert to radians.

        # Check boundary
        # ellipse: (x/180)^2/8 + (y/180)^2/2 <= 1 ? No.
        # FITS AIT is usually defined such that at equator, x goes from -2pi to 2pi (if math).
        # But xi, eta are degrees.
        # Standard scaling:
        # At (phi=180, theta=0), x = ?
        # x = 2 * implies x should be ~ 360?
        # Let's assume input x, y are in radians for the formula.

        # Implementation from WCSLIB:
        # u = x/2
        # v = y
        # r2 = u*u + v*v (where x,y in radians?)

        # Rescale for the 2*sqrt(2) vs degrees?
        # Calabretta 2002:
        # x = 180/pi * ...
        # So if we work in radians (dividing input by 180/pi), we use standard math.

        # Let u = x / 2
        # Let v = y
        # term = 1 - (u/2)^2 - v^2 ??
        # Or standard z term:
        # z = sqrt(1 - (x/4)^2 - (y/2)^2)
        # Let's derive from x^2/8 + y^2/2 <= 1 ?

        # Correct Formula (Calabretta & Greisen, Eq 168 inverse):
        # let X = x * (pi/180)
        # let Y = y * (pi/180)
        # Z = sqrt(1 - (X/4)^2 - (Y/2)^2)
        # phi = 2 * atan2(Z * X, 2*Z*Z - 1)
        # theta = asin(Z * Y)

        X = x
        Y = y

        # Term inside sqrt
        # (X/4)^2 + (Y/2)^2 <= 1 must hold.
        # Use simple masking/clamping

        r2 = (X / 4.0) ** 2 + (Y / 2.0) ** 2
        mask = r2 <= 1.0

        # Initialize outputs (already 0)
        # Only compute valid
        # Avoid sqrt negative
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))

        # phi
        # 2 * atan2( z*x, 2*z^2 - 1 )
        # Correct factor for X is 0.5 because X is roughly "2 * phi/2".
        # Calabretta Eq 168 implies x is already normalized?
        # But consistent derivation suggests using 0.5 * Z * X.

        # phi_rad = 2.0 * torch.atan2(z * X, 2.0 * z * z - 1.0)
        phi_rad = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0)

        # theta
        # asin(z * y)
        sin_theta = z * Y
        sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
        theta_rad = torch.asin(sin_theta)

        phi = phi_rad * r2d
        theta = theta_rad * r2d

        # Apply mask for undefined points
        phi = torch.where(
            mask, phi, torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype)
        )
        theta = torch.where(
            mask,
            theta,
            torch.tensor(float("nan"), device=theta.device, dtype=theta.dtype),
        )

    elif projection_code == "MOL":
        # Mollweide
        # Inverse (x,y -> phi, theta) is analytical.
        X = x
        Y = y

        sqrt2 = math.sqrt(2.0)

        # Check domain |Y| <= sqrt(2)
        # Y_clamped = torch.clamp(Y, -sqrt2, sqrt2)
        sin_gamma = torch.clamp(Y / sqrt2, -1.0, 1.0)
        gamma = torch.asin(sin_gamma)
        cos_gamma = torch.cos(gamma)

        # theta
        # (2*gamma + sin(2*gamma)) / pi
        # sin(2x) = 2 sin(x) cos(x)
        # Here sin(gamma) = sin_gamma
        t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / math.pi
        t_val = torch.clamp(t_val, -1.0, 1.0)
        theta_rad = torch.asin(t_val)

        # phi
        # phi = pi * X / (2*sqrt(2) * cos(gamma))
        denom = 2.0 * sqrt2 * cos_gamma

        mask_pole = torch.abs(denom) < 1e-8
        phi_rad = torch.zeros_like(X)

        # Avoid division by zero
        # Safe denom
        denom_safe = torch.where(mask_pole, torch.ones_like(denom), denom)
        phi_rad = torch.where(mask_pole, torch.zeros_like(X), math.pi * X / denom_safe)

        phi = phi_rad * r2d
        theta = theta_rad * r2d

        # Apply mask for undefined points (|Y| > sqrt(2))
        mask_valid = torch.abs(Y) <= sqrt2
        phi = torch.where(
            mask_valid,
            phi,
            torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype),
        )
        theta = torch.where(
            mask_valid,
            theta,
            torch.tensor(float("nan"), device=theta.device, dtype=theta.dtype),
        )

    elif projection_code == "HPX":
        # HEALPix
        # Calabretta & Roukema 2007 (Paper II, Eq 171-177)
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
        eta_pole = 90.0
        inv_polar_denom = 1.0 / (eta_pole - eta_boundary)

        abs_eta = torch.abs(eta)
        mask_eq = abs_eta <= eta_boundary

        # Equatorial zone
        s_theta_eq = eta / eta_scale
        # phi_eq = xi (already correct)

        # Polar caps
        sigma = (eta_pole - abs_eta) * inv_polar_denom
        s_theta_pol = torch.sign(eta) * (1.0 - (sigma * sigma) * (1.0 / 3.0))

        s_theta = torch.where(mask_eq, s_theta_eq, s_theta_pol)
        theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * r2d

        xc = torch.round((xi - 45.0) * (1.0 / 90.0)) * 90.0 + 45.0
        dx = xi - xc
        sigma_safe = torch.where(sigma < 1e-9, torch.ones_like(sigma), sigma)
        phi_pol = xc + dx / sigma_safe

        phi = torch.where(mask_eq, xi, phi_pol)

        # Validity check
        invalid = (abs_eta > eta_pole) | (
            ~mask_eq & (torch.abs(dx) > (45.0 * sigma + 1e-8))
        )
        if invalid.any():
            nan = torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype)
            phi = torch.where(invalid, nan, phi)
            theta = torch.where(invalid, nan, theta)
        return phi, theta

    elif projection_code == "SFL":
        # Sanson-Flamsteed (Sinusoidal)
        # phi = xi / cos(theta)
        # theta = eta
        theta = eta
        cos_theta = torch.cos(theta * d2r)
        # Avoid division by zero at poles
        cos_safe = torch.where(
            torch.abs(cos_theta) < 1e-12, torch.ones_like(cos_theta), cos_theta
        )
        phi = xi / cos_safe

    else:
        raise NotImplementedError(
            f"All-Sky projection {projection_code} not implemented"
        )

    return phi, theta


def deproject_allsky(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Forward all-sky projection: native spherical (phi, theta) -> intermediate (xi, eta).

    phi, theta: Degrees.
    Returns: xi, eta in degrees.
    """
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi

    if projection_code == "AIT":
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        phi_rad = phi_wrapped * d2r
        theta_rad = theta * d2r

        half_phi = phi_rad / 2.0
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_half_phi = torch.cos(half_phi)

        denom = torch.sqrt(0.5 * (1.0 + cos_theta * cos_half_phi))

        xi = 2.0 * cos_theta * torch.sin(half_phi) / denom * r2d
        eta = sin_theta / denom * r2d

        return xi, eta

    elif projection_code == "MOL":
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        phi_rad = phi_wrapped * d2r
        theta_rad = theta * d2r

        sin_theta = torch.sin(theta_rad)

        gamma = torch.zeros_like(theta_rad)
        for _ in range(10):
            f = 2.0 * gamma + torch.sin(2.0 * gamma) - math.pi * sin_theta
            fp = 2.0 + 2.0 * torch.cos(2.0 * gamma)
            gamma = gamma - f / (fp + 1e-12)

        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        sqrt2 = math.sqrt(2.0)
        xi = 2.0 * sqrt2 * phi_rad * cos_gamma / math.pi * r2d
        eta = sqrt2 * sin_gamma * r2d

        return xi, eta

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

        phi_w = torch.remainder(phi + 180.0, 360.0) - 180.0
        s_theta = torch.sin(theta * d2r)
        abs_s = torch.abs(s_theta)

        eta_scale = 90.0 * (K / H)
        eta_boundary_s = 2.0 / 3.0
        mask_eq = abs_s <= eta_boundary_s

        # Polar
        sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
        eta_pol = torch.sign(s_theta) * (
            90.0 - (90.0 - (eta_scale * eta_boundary_s)) * sigma
        )

        xc = torch.round((phi_w - 45.0) * (1.0 / 90.0)) * 90.0 + 45.0
        xi_pol = xc + sigma * (phi_w - xc)

        xi = torch.where(mask_eq, phi_w, xi_pol)
        eta = torch.where(mask_eq, eta_scale * s_theta, eta_pol)

        return xi, eta

    elif projection_code == "SFL":
        # Sanson-Flamsteed (Sinusoidal)
        # xi = phi * cos(theta)
        # eta = theta
        phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0
        xi = phi_wrapped * torch.cos(theta * d2r)
        eta = theta
        return xi, eta

    else:
        raise NotImplementedError(
            f"All-sky forward projection {projection_code} not implemented"
        )
