import torch
from torch import Tensor
from typing import Tuple, Optional


def project_zenithal(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to spherical coordinates (phi, theta)
    for Zenithal projections.
    """
    # 1. R and Phi
    r = torch.sqrt(xi * xi + eta * eta)

    # Handle R=0 case to avoid division/singularities
    # If R=0, phi is undefined (usually 0), theta = 90.

    # Convert to degrees
    # WCS Paper II: phi increases in the direction of increasing RA (East).
    # For zenithal projections, the native longitude phi is measured from the
    # direction toward the celestial pole. With phi_p = 180°, the formula is:
    # phi = atan2(-xi, -eta)
    phi_rad = torch.atan2(-xi, -eta)
    phi = torch.rad2deg(phi_rad)

    theta = torch.zeros_like(r)

    # Zenithal Functions: R = f(theta) or theta = f_inv(R)
    # Here (xi, eta) -> (phi, theta), so we need theta = f_inv(R).
    # theta is latitude (90 at pole).
    # native coordinates: (phi, theta). Pole at (0, 90).

    # We compute (90 - theta) often called 'u' or colatitude?
    # No, usually we solve for theta directly.

    if projection_code == "TAN":
        r_rad = torch.deg2rad(r)
        theta = torch.rad2deg(torch.atan2(torch.ones_like(r_rad), r_rad))

        # Wait, atan2(1, r_rad) is behavior of cot(theta) = r_rad?
        # If R_true = tan(90-theta)
        # then 90-theta = atan(R_true)
        # theta = 90 - atan(R_true)
        # atan2(1, R) is atan(1/R) = acot(R) = 90 - atan(R).
        # So yes, theta = rad2deg(atan2(1, r_rad)) is correct for TAN.

    elif projection_code == "SIN":
        # Orthographic
        # R = 180/pi * cos(theta) ??
        # Standard: R = sin(90 - theta) = cos(theta)
        # (normalized to unit sphere).
        # scaled by 180/pi?
        # R_true = R_deg * pi/180
        # R_true = cos(theta)
        # theta = acos(R_true)

        # Check boundary: R_true must be <= 1.
        # SIN projection is defined only for sphere hemisphere?
        # Or usually R scales such that R=1 corresponds to... ?
        # Greisen 2002: SIN is Orthographic.
        # "Oblique orthographic".
        # Standard coords (x, y) = (cos theta sin phi, cos theta cos phi).
        # R = cos(theta).
        # So theta = acos(R).
        # Check: at theta=90 (pole), R=0. Correct.
        # At theta=0 (equator), R=1. Correct.
        # Note scale: R here is dimensionless (ratio of radius).
        # But xi, eta are degrees.
        # So R_dim = R_deg / (180/pi).

        r_dim = torch.deg2rad(r)
        # Clamp to [-1, 1] to avoid NaN
        r_dim = torch.clamp(r_dim, -1.0, 1.0)
        theta = torch.rad2deg(torch.acos(r_dim))

        # Wait, standard SIN WCS usually has specific limits.
        # If R > 90 deg, it clips or wraps?
        # SIN is valid for < 90 deg range?

    elif projection_code == "ZEA":
        # Zenithal Equal Area
        # R = 2 * (180/pi) * sin(theta_co / 2)
        # 90 - theta = 2 * asin( R / (2 * 180/pi) )
        r_rad = torch.deg2rad(r)
        val = torch.clamp(r_rad / 2.0, -1.0, 1.0)
        theta = 90.0 - 2.0 * torch.rad2deg(torch.asin(val))

    elif projection_code == "STG":
        # Stereographic
        # R = 2 * (180/pi) * tan(theta_co / 2)
        # 90 - theta = 2 * atan( R / (2 * 180/pi) )
        r_rad = torch.deg2rad(r)
        theta = 90.0 - 2.0 * torch.rad2deg(torch.atan(r_rad / 2.0))

    elif projection_code == "ARC":
        # Zenithal Equidistant
        # R = 90 - theta (linear distance)
        # theta = 90 - R
        theta = 90.0 - r

    elif projection_code == "ZPN":
        if pv2 is None:
            w = torch.deg2rad(r)
        else:
            # PV2_3 is C0, PV2_4 is C1, ...
            # Extract coefficients from PV2 tensor [3:31]
            p_coeffs = pv2[3:31]
            mask_valid = p_coeffs != 0.0
            if not mask_valid.any():
                w = torch.deg2rad(r)
            else:
                # Find last valid coefficient for order
                # Use a small epsilon to detect non-zero coefficients
                valid_indices = torch.nonzero(p_coeffs.abs() > 1e-15, as_tuple=True)[0]
                if valid_indices.numel() == 0:
                    w = torch.deg2rad(r)
                else:
                    max_m = valid_indices[-1].item()
                    coeffs = p_coeffs[: max_m + 1]

                    r_rad_target = torch.deg2rad(r)
                    w = r_rad_target.clone()
                    target_eps = 1e-12

                    # Newton-Raphson (Horner's method)
                    rev_c = torch.flip(coeffs, dims=(0,))
                    for _ in range(12):
                        val = rev_c[0].expand_as(w).clone()
                        der = torch.zeros_like(w)
                        for i in range(1, len(rev_c)):
                            der = der * w + val
                            val = val * w + rev_c[i]

                        diff = val - r_rad_target
                        if diff.abs().max() < target_eps:
                            break
                        mask_nz = der.abs() > 1e-15
                        w = torch.where(mask_nz, w - diff / der, w)

        theta = 90.0 - torch.rad2deg(w)

    else:
        raise NotImplementedError(f"Zenithal code {projection_code} not implemented.")

    return phi, theta


def deproject_zenithal(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Inverse: Spherical (phi, theta) -> Intermediate (xi, eta).
    """
    # R depends on theta
    r = torch.zeros_like(theta)

    # theta is native latitude [-90, 90]
    # colatitude w = 90 - theta?

    if projection_code == "TAN":
        # R = 180/pi * cot(theta)
        # Avoid theta=0 singularity
        # theta in degrees.
        # cot(theta) = 1/tan(theta)
        tan_theta = torch.tan(torch.deg2rad(theta))
        # Mask zeros?
        r = torch.rad2deg(1.0 / (tan_theta + 1e-12))

    elif projection_code == "SIN":
        # R = 180/pi * cos(theta)
        r = torch.rad2deg(torch.cos(torch.deg2rad(theta)))

    elif projection_code == "ARC":
        r = 90.0 - theta

    elif projection_code == "ZEA":
        r = 2.0 * torch.rad2deg(torch.sin(torch.deg2rad(90.0 - theta) / 2.0))

    elif projection_code == "STG":
        r = 2.0 * torch.rad2deg(torch.tan(torch.deg2rad(90.0 - theta) / 2.0))

    elif projection_code == "ZPN":
        if pv2 is not None:
            # PV2_3 is C0, PV2_4 is C1, ...
            p_coeffs = pv2[3:31]
            mask_valid = p_coeffs.abs() > 1e-15
            if not mask_valid.any():
                r = 90.0 - theta
            else:
                # Find last valid coefficient for order
                valid_indices = torch.nonzero(mask_valid, as_tuple=True)[0]
                max_m = valid_indices[-1].item()
                coeffs = p_coeffs[: max_m + 1]

                # Horner's method for P(w)
                # w is colatitude in degrees
                w = 90.0 - theta
                rev_c = torch.flip(coeffs, dims=(0,))
                val = rev_c[0].expand_as(w).clone()
                for i in range(1, len(rev_c)):
                    val = val * w + rev_c[i]
                r = val
        else:
            r = 90.0 - theta

    # xi = -R * sin(phi)
    # eta = -R * cos(phi)  (standard convention for phi measured from North)

    phi_rad = torch.deg2rad(phi)
    xi = -r * torch.sin(phi_rad)
    eta = -r * torch.cos(phi_rad)

    return xi, eta
