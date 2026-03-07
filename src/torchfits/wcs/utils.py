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

    target_cos_dec = torch.cos(torch.deg2rad(target_dec))

    def get_residuals(curr_ra, curr_dec):
        # Calculate angular distance robustly
        d_dec = curr_dec - target_dec

        # Handle RA wrap around and scaling in-place for performance
        d_ra = curr_ra.sub(target_ra).add_(180)
        torch.remainder(d_ra, 360, out=d_ra)
        d_ra.sub_(180).mul_(target_cos_dec)

        return d_ra, d_dec

    # Step size for numerical Jacobian (in pixels).
    eps = 1e-3 if x.dtype == torch.float64 else 5e-3

    active_idx = None
    tol2 = tol * tol

    for i in range(max_iter):
        if active_idx is None:
            x_a = x
            y_a = y
            t_ra_a = target_ra
            t_dec_a = target_dec
            cos_dec_a = target_cos_dec
        else:
            if active_idx.numel() == 0:
                break
            x_a = x[active_idx]
            y_a = y[active_idx]
            t_ra_a = target_ra[active_idx]
            t_dec_a = target_dec[active_idx]
            cos_dec_a = target_cos_dec[active_idx]

        curr_ra, curr_dec = func(x_a, y_a, **kwargs)

        def get_residuals_active(curr_ra_a, curr_dec_a):
            d_dec = curr_dec_a - t_dec_a
            d_ra = curr_ra_a.sub(t_ra_a).add_(180)
            torch.remainder(d_ra, 360, out=d_ra)
            d_ra.sub_(180).mul_(cos_dec_a)
            return d_ra, d_dec

        r_ra, r_dec = get_residuals_active(curr_ra, curr_dec)
        dist2 = r_ra**2 + r_dec**2
        keep_active = dist2 >= tol2
        if not torch.any(keep_active):
            break

        x_a = x_a[keep_active]
        y_a = y_a[keep_active]
        t_ra_a = t_ra_a[keep_active]
        t_dec_a = t_dec_a[keep_active]
        cos_dec_a = cos_dec_a[keep_active]
        r_ra = r_ra[keep_active]
        r_dec = r_dec[keep_active]
        if active_idx is None:
            next_active_idx = torch.nonzero(keep_active, as_tuple=False).squeeze(1)
        else:
            next_active_idx = active_idx[keep_active]

        # 1. Compute Numerical Jacobian J
        # df/dx
        ra_dx, dec_dx = func(x_a + eps, y_a, **kwargs)
        r_ra_dx, r_dec_dx = get_residuals_active(ra_dx, dec_dx)
        j11 = (r_ra_dx - r_ra) / eps
        j21 = (r_dec_dx - r_dec) / eps

        # df/dy
        ra_dy, dec_dy = func(x_a, y_a + eps, **kwargs)
        r_ra_dy, r_dec_dy = get_residuals_active(ra_dy, dec_dy)
        j12 = (r_ra_dy - r_ra) / eps
        j22 = (r_dec_dy - r_dec) / eps

        # Determinant
        det = j11 * j22 - j12 * j21

        # Avoid division by zero
        det_sign = torch.where(det >= 0, 1.0, -1.0).to(det.dtype)
        det = torch.where(torch.abs(det) < 1e-15, det_sign * 1e-15, det)

        # Delta = -J^-1 * residual
        dx = -(j22 * r_ra - j12 * r_dec) / det
        dy = -(-j21 * r_ra + j11 * r_dec) / det

        # 2. Update with simple dampening/clamping
        # Max 1000 pixels jump per iteration to avoid divergence
        dx = torch.clamp(dx, -1000.0, 1000.0)
        dy = torch.clamp(dy, -1000.0, 1000.0)

        x[next_active_idx] = x_a + dx
        y[next_active_idx] = y_a + dy
        active_idx = next_active_idx

    # Final check
    curr_ra, curr_dec = func(x, y, **kwargs)
    r_ra, r_dec = get_residuals(curr_ra, curr_dec)
    dist = torch.sqrt(r_ra**2 + r_dec**2)
    converged = dist < tol

    return x, y, dist, converged
