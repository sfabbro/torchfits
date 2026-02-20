import torch
from torch import Tensor
from typing import Callable, Tuple


def solve_newton_raphson(
    func: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    target_ra: Tensor,
    target_dec: Tensor,
    initial_x: Tensor,
    initial_y: Tensor,
    max_iter: int = 10,
    tol: float = 1e-10,
    adaptive_step: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Vectorized Newton-Raphson solver for WCS inverse mapping.

    Finds (x, y) such that func(x, y) = (target_ra, target_dec).
    Uses numerical Jacobian estimation.

    Args:
        func: Forward mapping function (x, y) -> (ra, dec)
        target_ra: Target RA in degrees
        target_dec: Target Dec in degrees
        initial_x: Initial guess for x
        initial_y: Initial guess for y
        max_iter: Maximum iterations
        tol: Convergence tolerance in degrees (default 1e-10 deg ~ 0.36 mas)
        adaptive_step: Use simple dampening if residuals increase

    Returns:
        x, y: Solved pixel coordinates
        residuals: Final distance in degrees
        converged: Boolean mask of converged points
    """
    x = initial_x.clone()
    y = initial_y.clone()

    # Ensure targets are on the same device
    target_ra = target_ra.to(x.device)
    target_dec = target_dec.to(x.device)

    def get_residuals(curr_ra, curr_dec):
        # Calculate angular distance robustly
        d_dec = curr_dec - target_dec
        d_ra = curr_ra - target_ra

        # Handle RA wrap around
        d_ra = (d_ra + 180) % 360 - 180

        # Scaling RA by cos(dec) for linear-ish residuals
        cos_dec = torch.cos(torch.deg2rad(target_dec))
        return d_ra * cos_dec, d_dec

    # Step size for numerical Jacobian (in pixels).
    eps = 1e-3 if x.dtype == torch.float64 else 5e-3

    for i in range(max_iter):
        curr_ra, curr_dec = func(x, y, **kwargs)
        r_ra, r_dec = get_residuals(curr_ra, curr_dec)

        dist = torch.sqrt(r_ra**2 + r_dec**2)

        if torch.all(dist < tol):
            break

        # 1. Compute Numerical Jacobian J
        # df/dx
        ra_dx, dec_dx = func(x + eps, y, **kwargs)
        r_ra_dx, r_dec_dx = get_residuals(ra_dx, dec_dx)
        j11 = (r_ra_dx - r_ra) / eps
        j21 = (r_dec_dx - r_dec) / eps

        # df/dy
        ra_dy, dec_dy = func(x, y + eps, **kwargs)
        r_ra_dy, r_dec_dy = get_residuals(ra_dy, dec_dy)
        j12 = (r_ra_dy - r_ra) / eps
        j22 = (r_dec_dy - r_dec) / eps

        # Determinant
        det = j11 * j22 - j12 * j21

        # Avoid division by zero
        det = torch.where(torch.abs(det) < 1e-15, torch.sign(det) * 1e-15, det)

        # Delta = -J^-1 * residual
        dx = -(j22 * r_ra - j12 * r_dec) / det
        dy = -(-j21 * r_ra + j11 * r_dec) / det

        # 2. Update with simple dampening/clamping
        # Max 1000 pixels jump per iteration to avoid divergence
        dx = torch.clamp(dx, -1000.0, 1000.0)
        dy = torch.clamp(dy, -1000.0, 1000.0)

        x = x + dx
        y = y + dy

    # Final check
    curr_ra, curr_dec = func(x, y, **kwargs)
    r_ra, r_dec = get_residuals(curr_ra, curr_dec)
    dist = torch.sqrt(r_ra**2 + r_dec**2)
    converged = dist < tol

    return x, y, dist, converged
