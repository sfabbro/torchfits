import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
import math


def project_allsky(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to native spherical coordinates (phi, theta).
    NOTE: This is the INVERSE projection (Plane -> Sphere) relative to standard FITS definition.

    xi, eta: Degrees (Standard FITS Intermediate).
    Returns: phi, theta (Degrees)
    """
    params = params or {}

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
        # Standard FITS HPX uses H=4, K=3. We keep H/K scaling for the
        # equatorial zone and use the standard polar-cap relations.
        H = params.get("PV1_1", params.get("PV2_1", 4.0))
        K = params.get("PV1_2", params.get("PV2_2", 3.0))

        phi = torch.zeros_like(xi)
        theta = torch.zeros_like(eta)

        eta_scale = 90.0 * (K / H)
        eta_boundary = eta_scale * (2.0 / 3.0)
        eta_pole = 90.0
        polar_denom = eta_pole - eta_boundary

        abs_eta = torch.abs(eta)
        mask_eq = abs_eta <= eta_boundary
        if mask_eq.any():
            s_theta_eq = eta[mask_eq] / eta_scale
            s_theta_eq = torch.clamp(s_theta_eq, -1.0, 1.0)
            theta[mask_eq] = torch.asin(s_theta_eq) * r2d
            phi[mask_eq] = xi[mask_eq]

        mask_pol = ~mask_eq
        if mask_pol.any():
            xi_p, eta_p = xi[mask_pol], eta[mask_pol]
            abs_eta = torch.abs(eta_p)

            sigma = (eta_pole - abs_eta) / polar_denom
            sigma = torch.clamp(sigma, min=0.0)

            # Standard HPX relation in the polar caps.
            s_theta_pol = torch.sign(eta_p) * (1.0 - (sigma * sigma) / 3.0)
            s_theta_pol = torch.clamp(s_theta_pol, -1.0, 1.0)
            theta[mask_pol] = torch.asin(s_theta_pol) * r2d

            xc = torch.round((xi_p - 45.0) / 90.0) * 90.0 + 45.0
            dx = xi_p - xc
            sigma_safe = torch.where(
                torch.abs(sigma) < 1e-9, torch.ones_like(sigma), sigma
            )
            phi[mask_pol] = xc + dx / sigma_safe

        # Reject points outside the nominal HPX domain.
        invalid_eta = torch.abs(eta) > eta_pole
        if mask_pol.any():
            xi_p, eta_p = xi[mask_pol], eta[mask_pol]
            xc = torch.round((xi_p - 45.0) / 90.0) * 90.0 + 45.0
            sigma = torch.clamp((eta_pole - torch.abs(eta_p)) / polar_denom, min=0.0)
            invalid_x = torch.abs(xi_p - xc) > (45.0 * sigma + 1e-8)
            pol_bad = torch.zeros_like(mask_pol)
            pol_bad[mask_pol] = invalid_x
            invalid_eta = invalid_eta | pol_bad

        nan_like = torch.tensor(float("nan"), device=phi.device, dtype=phi.dtype)
        phi = torch.where(invalid_eta, nan_like, phi)
        theta = torch.where(invalid_eta, nan_like, theta)

    else:
        raise NotImplementedError(
            f"All-Sky projection {projection_code} not implemented"
        )

    return phi, theta


def deproject_allsky(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Forward all-sky projection: native spherical (phi, theta) -> intermediate (xi, eta).

    phi, theta: Degrees.
    Returns: xi, eta in degrees.
    """
    params = params or {}
    d2r = math.pi / 180.0

    if projection_code != "HPX":
        raise NotImplementedError(
            f"All-sky forward projection {projection_code} not implemented"
        )

    H = params.get("PV1_1", params.get("PV2_1", 4.0))
    K = params.get("PV1_2", params.get("PV2_2", 3.0))

    # Wrap to a centered longitude interval to keep facet selection stable.
    phi_w = torch.remainder(phi + 180.0, 360.0) - 180.0
    s_theta = torch.sin(theta * d2r)
    abs_s = torch.abs(s_theta)

    xi = torch.zeros_like(phi_w)
    eta = torch.zeros_like(theta)

    # Equatorial zone.
    # For standard HPX this corresponds to |sin(theta)| <= 2/3.
    mask_eq = abs_s <= (2.0 / 3.0)
    if mask_eq.any():
        eta_scale = 90.0 * (K / H)
        xi[mask_eq] = phi_w[mask_eq]
        eta[mask_eq] = eta_scale * s_theta[mask_eq]

    # Polar zones.
    mask_pol = ~mask_eq
    if mask_pol.any():
        s_pol = s_theta[mask_pol]
        sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - torch.abs(s_pol)), min=0.0))
        eta_scale = 90.0 * (K / H)
        eta_boundary = eta_scale * (2.0 / 3.0)
        eta[mask_pol] = torch.sign(s_pol) * (90.0 - (90.0 - eta_boundary) * sigma)

        # Facet center consistent with project_allsky inverse implementation.
        phi_pol = phi_w[mask_pol]
        xc = torch.round((phi_pol - 45.0) / 90.0) * 90.0 + 45.0
        xi[mask_pol] = xc + sigma * (phi_pol - xc)

    return xi, eta
