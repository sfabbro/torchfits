import torch
from torch import Tensor
from typing import Tuple, Optional

D2R = 0.017453292519943295
R2D = 57.29577951308232
_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


def _sincos(x: Tensor) -> Tuple[Tensor, Tensor]:
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)
    return torch.sin(x), torch.cos(x)


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
    r = torch.hypot(xi, eta)

    # Handle R=0 case to avoid division/singularities
    # If R=0, phi is undefined (usually 0), theta = 90.

    # Convert to degrees
    # WCS Paper II: phi increases in the direction of increasing RA (East).
    # For zenithal projections, the native longitude phi is measured from the
    # direction toward the celestial pole. With phi_p = 180°, the formula is:
    # phi = atan2(-xi, -eta)
    # 57.29577951308232 = 180.0 / math.pi
    phi = torch.atan2(-xi, -eta) * R2D

    theta = torch.zeros_like(r)

    # Zenithal Functions: R = f(theta) or theta = f_inv(R)
    # Here (xi, eta) -> (phi, theta), so we need theta = f_inv(R).
    # theta is latitude (90 at pole).
    # native coordinates: (phi, theta). Pole at (0, 90).

    # We compute (90 - theta) often called 'u' or colatitude?
    # No, usually we solve for theta directly.

    if projection_code == "TAN":
        r_rad = r * D2R
        theta = torch.atan2(torch.ones_like(r_rad), r_rad) * R2D

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

        r_dim = r * D2R
        # SIN is defined only for r <= 1 radian (90 degrees equivalent in dimensionless units)
        # R = cos(theta) = sin(90-theta)
        mask_out = r_dim > 1.0
        theta = torch.acos(torch.clamp(r_dim, -1.0, 1.0)) * R2D
        theta = theta.masked_fill(mask_out, float("nan"))

    elif projection_code == "ZEA":
        # Zenithal Equal Area
        # R = 2 * (180/pi) * sin(theta_co / 2)
        # 90 - theta = 2 * asin( R / (2 * 180/pi) )
        r_rad = r * D2R
        # ZEA is defined for r_rad <= 2.0 (theta_co <= 180 deg)
        mask_out = (r_rad / 2.0) > 1.0
        val = torch.clamp(r_rad / 2.0, -1.0, 1.0)
        theta = 90.0 - 2.0 * torch.asin(val) * R2D
        theta = theta.masked_fill(mask_out, float("nan"))

    elif projection_code == "STG":
        # Stereographic
        # R = 2 * (180/pi) * tan(theta_co / 2)
        # 90 - theta = 2 * atan( R / (2 * 180/pi) )
        r_rad = r * D2R
        theta = 90.0 - 2.0 * torch.atan(r_rad / 2.0) * R2D

    elif projection_code == "ARC":
        # Zenithal Equidistant
        # R = 90 - theta (linear distance)
        # theta = 90 - R
        theta = 90.0 - r

    elif projection_code == "ZPN":
        if pv2 is None:
            w = r * D2R
        else:
            # PV2_3 is C0, PV2_4 is C1, ...
            # Extract coefficients from PV2 tensor [3:31]
            p_coeffs = pv2[3:31]
            mask_valid = p_coeffs != 0.0
            if not mask_valid.any():
                w = r * D2R
            else:
                # Find last valid coefficient for order
                # Use a small epsilon to detect non-zero coefficients
                valid_indices = torch.nonzero(p_coeffs.abs() > 1e-15, as_tuple=True)[0]
                if valid_indices.numel() == 0:
                    w = r * D2R
                else:
                    max_m = valid_indices[-1].item()
                    coeffs = p_coeffs[: max_m + 1]

                    r_rad_target = r * D2R
                    w = r_rad_target.clone()
                    target_eps = 1e-12

                    # Newton-Raphson (Horner's method)
                    rev_c = torch.flip(coeffs, dims=(0,))
                    c0 = rev_c[0]
                    for _ in range(12):
                        val = torch.full_like(w, c0)
                        der = torch.zeros_like(w)
                        for c in rev_c[1:]:
                            der.mul_(w).add_(val)
                            val.mul_(w).add_(c)

                        diff = val.sub_(r_rad_target)
                        if diff.abs().max() < target_eps:
                            break

                        mask_nz = der.abs() > 1e-15
                        der.masked_fill_(~mask_nz, 1.0)
                        diff.masked_fill_(~mask_nz, 0.0)
                        w.sub_(diff.div_(der))

        # 57.29577951308232 = 180.0 / math.pi
        theta = 90.0 - w * R2D

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
        tan_theta = torch.tan(theta * D2R)
        # Mask zeros?
        r = (1.0 / (tan_theta + 1e-12)) * R2D

    elif projection_code == "SIN":
        # R = 180/pi * cos(theta)
        r = torch.cos(theta * D2R) * R2D

    elif projection_code == "ARC":
        r = 90.0 - theta

    elif projection_code == "ZEA":
        r = 2.0 * torch.sin((90.0 - theta) * (0.5 * D2R)) * R2D

    elif projection_code == "STG":
        r = 2.0 * torch.tan((90.0 - theta) * (0.5 * D2R)) * R2D

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
                c0 = rev_c[0]
                val = torch.full_like(w, c0)
                for c in rev_c[1:]:
                    val.mul_(w).add_(c)
                r = val
        else:
            r = 90.0 - theta

    # xi = -R * sin(phi)
    # eta = -R * cos(phi)  (standard convention for phi measured from North)

    phi_rad = phi * D2R
    sin_phi, cos_phi = _sincos(phi_rad)
    xi = -r * sin_phi
    eta = -r * cos_phi

    return xi, eta
