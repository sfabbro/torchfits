
import torch
from torch import Tensor
from typing import Tuple, Optional, Dict

def project_zenithal(xi: Tensor, eta: Tensor, projection_code: str, params: Optional[Dict[str, float]] = None) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to spherical coordinates (phi, theta)
    for Zenithal projections.
    
    Standard algorithm (Calabretta & Greisen 2002):
    1. R = sqrt(xi^2 + eta^2)
    2. phi = atan2(xi, -eta)  (Note: differing conventions on eta sign, checking standard)
       WCSLIB: phi = arg( -eta, xi ) -> atan2(xi, -eta)
    3. theta depends on R via projection specific function.
    
    We align with WCSLIB/Astropy conventions.
    """
    params = params or {}
    
    # 1. R and Phi
    r = torch.sqrt(xi*xi + eta*eta)
    
    # Handle R=0 case to avoid division/singularities
    # If R=0, phi is undefined (usually 0), theta = 90.
    
    # Convert to degrees
    # WCS Paper II: phi increases in the direction of increasing RA (East).
    # Standard projection: xi increases to the West. 
    # So phi = atan2(-xi, -eta)
    phi_rad = torch.atan2(-xi, -eta) 
    phi = torch.rad2deg(phi_rad)
    
    theta = torch.zeros_like(r)
    
    # Zenithal Functions: R = f(theta) or theta = f_inv(R)
    # Here (xi, eta) -> (phi, theta), so we need theta = f_inv(R).
    # theta is latitude (90 at pole).
    # native coordinates: (phi, theta). Pole at (0, 90).
    
    # We compute (90 - theta) often called 'u' or colatitude?
    # No, usually we solve for theta directly.
    
    
    if projection_code == 'TAN':
        # Gnomonic
        # R = tan(90 - theta) = cot(theta)
        # theta = 90 - atan(R) = atan(1/R) ? 
        # Wait, standard TAN: R = tan(theta_native_colatitude)?
        # WCS Paper II: 
        # theta_native = 90 corresponds to R=0.
        # r_theta = 180/pi * cot(theta)  (if R in degrees? No, R is dimensionless in formula, scaled by 180/pi)
        # Usually xi, eta are in degrees.
        # Let's use WCSLIB formulae:
        # TAN: R = 180/pi * cot(theta)
        # -> cot(theta) = R * pi/180
        # -> tan(theta) = 180 / (pi * R)
        # -> theta = atan( 180 / (pi * R) )
        
        # Avoid division by zero
        # If R=0, theta = 90.
        
        # Faster: theta = 90 - degrees(atan(R_rad)) ?
        # If R is in degrees (on tangent plane):
        # r_rad = deg2rad(r)
        # theta = 90 - rad2deg(atan(r_rad)) ?
        # Let's assert: xi, eta in degrees.
        # R in degrees.
        # TAN logic:
        # theta = 90 - atan( R )  (if R was dimensionless tangent)
        # But R is angle on sky? No, R is distance on tangent plane.
        # R_rad = deg2rad(R)
        # theta = rad2deg( atan( 1 / R_rad ) ) ??
        
        # Standard Gnomonic:
        # x = tan(lat_native_co) * sin(lon)
        # y = -tan(lat_native_co) * cos(lon)
        # R = tan(90 - theta)
        # 90 - theta = atan(R)
        # theta = 90 - atan(R) (if R is strictly tan(co-lat))
        
        # BUT xi, eta are usually scaled by CD.
        # If CD gives degrees, then R is in degrees.
        # We must treat R as "degrees on tangent plane".
        # Which corresponds to tan(angle) * (180/pi)?
        # Yes, standard WCS logic assumes projections generate (x,y) in degrees.
        # So x_standard = (180/pi) * x_true
        
        r_rad = torch.deg2rad(r)
        theta = torch.rad2deg(torch.atan2(torch.tensor(1.0, device=r.device, dtype=r.dtype), r_rad))
        
        # Wait, atan2(1, r_rad) is behavior of cot(theta) = r_rad?
        # If R_true = tan(90-theta)
        # then 90-theta = atan(R_true)
        # theta = 90 - atan(R_true)
        # atan2(1, R) is atan(1/R) = acot(R) = 90 - atan(R).
        # So yes, theta = rad2deg(atan2(1, r_rad)) is correct for TAN.

    elif projection_code == 'SIN':
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
        
    elif projection_code == 'ZEA':
        # Zenithal Equal Area
        # R = 2 * (180/pi) * sin(theta_co / 2)
        # 90 - theta = 2 * asin( R / (2 * 180/pi) )
        r_rad = torch.deg2rad(r)
        val = torch.clamp(r_rad / 2.0, -1.0, 1.0)
        theta = 90.0 - 2.0 * torch.rad2deg(torch.asin(val))
        
    elif projection_code == 'STG':
        # Stereographic
        # R = 2 * (180/pi) * tan(theta_co / 2)
        # 90 - theta = 2 * atan( R / (2 * 180/pi) )
        r_rad = torch.deg2rad(r)
        theta = 90.0 - 2.0 * torch.rad2deg(torch.atan(r_rad / 2.0))
        
    elif projection_code == 'ARC':
        # Zenithal Equidistant
        # R = 90 - theta (linear distance)
        # theta = 90 - R
        theta = 90.0 - r
        
    elif projection_code == 'ZPN':
        # Zenithal Polynomial
        # R = 180/pi * P(w)
        # w = 90 - theta (degrees? Check scale).
        # Paper II: P(w) is polynomial in w (radians? No, likely degrees if coeffs are adjusted?
        # Actually standard: "w is the native colatitude in degrees".
        # But R comes out in degrees.
        # PVi_j coeffs: PV2_0 + PV2_1*w + ...
        # Standard: PV2_0, PV2_1 ... match coefficients.
        # 
        # We have R (from xi, eta).
        # We need to solve P(w) - R = 0 for w.
        # w is colatitude [0, 180]. P(w) is monotonic usually.
        # Use Newton-Raphson.
        
        # Get coeffs from params.
        # Keys: 'PV2_0', 'PV2_1', ...
        # Usually up to order 30?
        # Collect PVs
        # ZPN coeffs: PV2_3 ($C_0$), PV2_4 ($C_1$), ...
        # PV2_1 is phi0, PV2_2 is theta0 (not typically used in ZPN dist models but valid)
        
        # Determine max order
        max_m = -1
        for m_idx in range(3, 31):
            if f'PV2_{m_idx}' in params:
                 max_m = m_idx
                 
        if max_m == -1:
             p_coeffs = []
        else:
             order = max_m - 3
             p_coeffs = [0.0] * (order + 1)
             for i in range(order + 1):
                 key = f'PV2_{i+3}'
                 if key in params:
                     p_coeffs[i] = params[key]
            
        # If no coeffs, default ZPN? Identity?
        if not p_coeffs:
            # Fallback to linear? w = R?
            w = r.clone() # Ensure copy
        else:
            # Newton-Raphson Solver for R = P(w)
            # P(w) = sum c_k w^k
            # P'(w) = sum k c_k w^{k-1}
            
            # Pre-process coeffs into tensors for fast eval
            # Poly params (ascending power)
            # coeffs: [c0, c1, c2, ...]
            # deriv_coeffs: [c1, 2*c2, 3*c3, ...]
            # We use Horner's method which is efficiently vectorizable if unrolled 
            # or if we accept sequential kernel launches (still faster than Python loop overhead if cleaner)
            
            # Actually, standard Newton method converges fast (5-10 iter)
            # Loop over iterations
            
            # R_deg = (180/pi) * P(w_rad)
            # Find w_rad such that P(w_rad) = r_deg * (pi/180)
            r_rad_target = torch.deg2rad(r)
            rev_c = p_coeffs[::-1]
            
            # Initial guess: linear w = r_rad_target
            w = r_rad_target.clone()
            
            # Pre-move coeffs to device
            rev_c_t = torch.tensor(rev_c, device=r.device, dtype=r.dtype)
            
            target_eps = 1e-9 if r.dtype == torch.float32 else 1e-12
            for _ in range(12): 
                val = torch.full_like(w, rev_c[0])
                der = torch.zeros_like(w)
                
                for i in range(1, len(rev_c)):
                    der.mul_(w).add_(val)
                    val.mul_(w).add_(rev_c_t[i])
                
                # diff = val - target
                val.sub_(r_rad_target)
                
                if val.abs().max() < target_eps:
                    break
                    
                mask_nz = der.abs() > 1e-15
                w = torch.where(mask_nz, w - val/der, w)
                
                if torch.isnan(w).any():
                     w = torch.nan_to_num(w, nan=1.5708) # Pole
                     break
                
        # w is colatitude in radians
        theta = 90.0 - torch.rad2deg(w)
        
    else:
        raise NotImplementedError(f"Zenithal code {projection_code} not implemented.")

    return phi, theta

def deproject_zenithal(phi: Tensor, theta: Tensor, projection_code: str, params: Optional[Dict[str, float]] = None) -> Tuple[Tensor, Tensor]:
    """
    Inverse: Spherical (phi, theta) -> Intermediate (xi, eta).
    """
    params = params or {}
    
    # R depends on theta
    r = torch.zeros_like(theta)
    
    # theta is native latitude [-90, 90]
    # colatitude w = 90 - theta?
    
    if projection_code == 'TAN':
        # R = 180/pi * cot(theta)
        # Avoid theta=0 singularity
        # theta in degrees.
        # cot(theta) = 1/tan(theta)
        tan_theta = torch.tan(torch.deg2rad(theta))
        # Mask zeros?
        r = torch.rad2deg(1.0 / (tan_theta + 1e-12))
        
    elif projection_code == 'SIN':
        # R = 180/pi * cos(theta)
        r = torch.rad2deg(torch.cos(torch.deg2rad(theta)))
        
    elif projection_code == 'ARC':
        # R = 90 - theta
        r = 90.0 - theta
        
    elif projection_code == 'ZPN':
        # R = 180/pi * P(w)
        # w = 90 - theta (deg? rad?)
        # WCSLIB PVi keywords define P.
        pass
        
    # xi = R * sin(phi)
    # eta = -R * cos(phi)  (standard convention)
    
    xi = r * torch.sin(phi)
    eta = -r * torch.cos(phi)
    
    return xi, eta
