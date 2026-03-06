import torch
from torch import Tensor
from typing import Tuple, Optional

D2R = 0.017453292519943295
R2D = 57.29577951308232


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    if angle.requires_grad:
        return torch.remainder(angle + 180.0, 360.0) - 180.0
    mn = torch.amin(angle)
    mx = torch.amax(angle)
    if bool((mn >= -180.0) and (mx < 180.0)):
        return angle
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def project_cylindrical(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate (xi, eta) to spherical (phi, theta) for Cylindrical projections.
    """
    pv2_1 = pv2[1] if pv2 is not None else 1.0

    if projection_code == "CEA":
        phi = xi
        val = torch.clamp(eta * (pv2_1 * D2R), -1.0, 1.0)
        theta = torch.asin(val) * R2D

    elif projection_code == "CAR":
        phi = xi
        theta = eta

    elif projection_code == "MER":
        phi = xi
        arg = eta * D2R
        arg = torch.clamp(arg, -20.0, 20.0)
        exp_val = torch.exp(arg)
        theta = 2.0 * (torch.atan(exp_val) * R2D - 45.0)

    elif projection_code == "CYP":
        raise NotImplementedError("CYP inversion not yet implemented.")

    else:
        raise NotImplementedError(
            f"Cylindrical code {projection_code} not implemented."
        )

    return phi, theta


def deproject_cylindrical(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Forward cylindrical projection: native spherical (phi, theta) -> intermediate (xi, eta).

    phi, theta: Degrees. For cylindrical projections, these are typically native coords
                where phi is in [0, 360) and needs to be wrapped to [-180, 180] for xi.
    Returns: xi, eta in degrees.
    """
    pv2_1 = pv2[1] if pv2 is not None else 1.0

    # Wrap phi to [-180, 180] range for cylindrical projections
    phi_wrapped = _wrap_lon180_checked(phi)

    if projection_code == "CEA":
        xi = phi_wrapped
        eta = (R2D / pv2_1) * torch.sin(theta * D2R)
        return xi, eta

    elif projection_code == "CAR":
        xi = phi_wrapped
        eta = theta
        return xi, eta

    elif projection_code == "MER":
        xi = phi_wrapped
        half_theta = theta / 2.0
        tan_arg = torch.tan((45.0 + half_theta) * D2R)
        tan_arg = torch.clamp(tan_arg, min=1e-12)
        eta = R2D * torch.log(tan_arg)
        return xi, eta

    elif projection_code == "CYP":
        raise NotImplementedError("CYP forward projection not yet implemented.")

    else:
        raise NotImplementedError(
            f"Cylindrical forward projection {projection_code} not implemented."
        )
