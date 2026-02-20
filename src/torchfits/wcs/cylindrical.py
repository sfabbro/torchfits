import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
import math


def project_cylindrical(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate (xi, eta) to spherical (phi, theta) for Cylindrical projections.

    Cylindrical projections map the sphere onto a cylinder.
    Standard convention:
    phi = xi / some_scale
    theta = f(eta)

    """
    params = params or {}

    # Defaults
    pv2_1 = params.get("PV2_1", 1.0)  # Lambda parameter for CEA/others usually

    if projection_code == "CEA":
        # Cylindrical Equal Area
        # xi = phi (deg)
        # eta = (180/pi) * sin(theta) / lambda
        # So:
        # phi = xi
        # sin(theta) = lambda * eta * (pi/180)
        # theta = asin( ... )

        phi = xi

        # PV2_1 is lambda. Default 1.0?
        # WCS Paper II: Lambda = PV2_1.
        lambda_val = pv2_1

        val = (lambda_val * eta) * (math.pi / 180.0)
        # Clamp for asin
        val = torch.clamp(val, -1.0, 1.0)
        theta = torch.rad2deg(torch.asin(val))

    elif projection_code == "MER":
        # Mercator
        # xi = phi
        # eta = (180/pi) * ln( tan(45 + theta/2) )
        # To invert:
        # eta * (pi/180) = ln(tan...)
        # exp(...) = tan(45 + theta/2)
        # atan(exp) = 45 + theta/2
        # theta/2 = atan(exp) - 45
        # theta = 2 * (atan(exp) - 45)

        phi = xi

        arg = eta * (math.pi / 180.0)
        # exp of large number -> Infinity. Clamp?
        arg = torch.clamp(arg, -20.0, 20.0)  # exp(20) is huge enough

        exp_val = torch.exp(arg)
        theta = 2.0 * (torch.rad2deg(torch.atan(exp_val)) - 45.0)

    elif projection_code == "CYP":
        # Cylindrical Perspective
        # xi = phi
        # eta = (180/pi) * (mu + lambda) / (mu + cos(theta)) * sin(theta) ... complex.
        # Inversion requires solving?
        # WCS Paper II: eta = (180/pi) * (mu + lambda) sin(theta) / (mu + cos(theta))
        # This is soluble.
        raise NotImplementedError("CYP inversion not yet implemented.")

    else:
        raise NotImplementedError(
            f"Cylindrical code {projection_code} not implemented."
        )

    return phi, theta
