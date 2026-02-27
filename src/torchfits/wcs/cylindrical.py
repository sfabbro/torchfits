import torch
from torch import Tensor
from typing import Tuple, Optional
import math


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
        lambda_val = pv2_1
        val = (lambda_val * eta) * (math.pi / 180.0)
        val = torch.clamp(val, -1.0, 1.0)
        theta = torch.rad2deg(torch.asin(val))

    elif projection_code == "CAR":
        phi = xi
        theta = eta

    elif projection_code == "MER":
        phi = xi
        arg = eta * (math.pi / 180.0)
        arg = torch.clamp(arg, -20.0, 20.0)
        exp_val = torch.exp(arg)
        theta = 2.0 * (torch.rad2deg(torch.atan(exp_val)) - 45.0)

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
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi

    pv2_1 = pv2[1] if pv2 is not None else 1.0

    # Wrap phi to [-180, 180] range for cylindrical projections
    phi_wrapped = torch.remainder(phi + 180.0, 360.0) - 180.0

    if projection_code == "CEA":
        xi = phi_wrapped
        lambda_val = pv2_1
        eta = (r2d / lambda_val) * torch.sin(theta * d2r)
        return xi, eta

    elif projection_code == "CAR":
        xi = phi_wrapped
        eta = theta
        return xi, eta

    elif projection_code == "MER":
        xi = phi_wrapped
        half_theta = theta / 2.0
        tan_arg = torch.tan((45.0 + half_theta) * d2r)
        tan_arg = torch.clamp(tan_arg, min=1e-12)
        eta = r2d * torch.log(tan_arg)
        return xi, eta

    elif projection_code == "CYP":
        raise NotImplementedError("CYP forward projection not yet implemented.")

    else:
        raise NotImplementedError(
            f"Cylindrical forward projection {projection_code} not implemented."
        )
