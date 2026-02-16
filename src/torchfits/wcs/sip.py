
import torch
from torch import Tensor
from typing import Dict, Any, Optional

class SIP:
    """
    Simple Imaging Polynomial (SIP) distortion correction.
    
    This class handles the parsing of SIP coefficients from a FITS header
    and applies the distortion correction to pixel coordinates.
    
    References:
    - Shupe et al. (2005): "The SIP Convention for Representing Distortion in FITS Image Headers"
    """
    def __init__(self, header: Dict[str, Any]):
        self.a_order = int(header.get('A_ORDER', 0))
        self.b_order = int(header.get('B_ORDER', 0))
        self.ap_order = int(header.get('AP_ORDER', 0))
        self.bp_order = int(header.get('BP_ORDER', 0))
        
        # Parse A/B coefficients (Forward: Pixel -> Focal Plane)
        self.a_coeffs = self._parse_coeffs(header, 'A', self.a_order)
        self.b_coeffs = self._parse_coeffs(header, 'B', self.b_order)
        
        # Parse AP/BP coefficients (Inverse: Focal Plane -> Pixel)
        self.ap_coeffs = self._parse_coeffs(header, 'AP', self.ap_order)
        self.bp_coeffs = self._parse_coeffs(header, 'BP', self.bp_order)

    def _parse_coeffs(self, header: Dict[str, Any], prefix: str, order: int) -> Dict[str, float]:
        """
        Parse coefficients for a given prefix and order.
        Example: A_2_0, A_0_2, etc.
        Returns a dictionary {(p, q): value}
        """
        coeffs = {}
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                # Skip 0th and 1st order terms if they are typically part of CD matrix?
                # SIP convention says A/B are deviations from the linear term.
                # However, all A_p_q are valid.
                
                key = f'{prefix}_{p}_{q}'
                if key in header:
                    coeffs[(p, q)] = float(header[key])
        return coeffs

    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply forward distortion: (u, v) -> (u + f(u,v), v + g(u,v))
        
        u, v: Relative pixel coordinates (typically x - CRPIX, y - CRPIX)
        """
        # Optimized implementation could use Horner's method or precomputed powers
        # For now, explicit summation
        
        # Compute powers of u and v once if needed, or just compute on the fly
        # Since order is typically small (2-5), we can just loop.
        
        f_uv = torch.zeros_like(u)
        g_uv = torch.zeros_like(v)
        
        # Apply A coefficients to u
        for (p, q), coeff in self.a_coeffs.items():
            # Term is coeff * u^p * v^q
            # Optimization: 
            term = coeff * torch.pow(u, p) * torch.pow(v, q)
            f_uv += term
            
        # Apply B coefficients to v
        for (p, q), coeff in self.b_coeffs.items():
            term = coeff * torch.pow(u, p) * torch.pow(v, q)
            g_uv += term
            
        return u + f_uv, v + g_uv

    def undistort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply inverse distortion (AP/BP): (u_distorted, v_distorted) -> (u_linear, v_linear)
        
        This uses the AP/BP polynomials to map from distorted (focal plane) coordinates 
        back to linear intermediate coordinates? 
        The SIP paper convention is:
        U = u + A(u, v)  (distorted -> rectified) ??
        Actually:
        "The SIP convention defines a transformation from pixel coordinates (u, v)
        to intermediate world coordinates (x, y)." (Shupe et al.)
        
        Let's clarify:
        u, v = pixel coordinates relative to CRPIX
        x, y = "intermediate world coordinates" (linearized)
        
        Forward (Pix -> World):
        x = u + A(u, v) ?? 
        Or:
        [x, y] = CD * [u + f(u,v), v + g(u,v)] 
        
        Let's check Astropy WCS documentation/behavior.
        Astropy: `all_pix2world` applies SIP.
        Logic:
        1. u = pixel - crpix
        2. Apply distortion: u' = u + f(u, v), v' = v + g(u, v)
        3. Apply linear CD/PC: [xi, eta] = CD * [u', v']
        
        So my `distort` method above calculates u', v'.
        
        Inverse (World -> Pix):
        1. [xi, eta] from world coords
        2. [u', v'] = CD_inv * [xi, eta]
        3. Apply reverse distortion: u = u' + AP(u', v'), v = v' + BP(u', v')
           (Note: AP/BP are polynomials of the *linear* coordinates u', v', evaluating to the correction to get back to u, v)
        
        So `undistort` inputs are u', v' (linear/intermediate), outputs u, v (pixel).
        """
        
        delta_u = torch.zeros_like(u)
        delta_v = torch.zeros_like(v)
        
        # Apply AP coefficients
        for (p, q), coeff in self.ap_coeffs.items():
            term = coeff * torch.pow(u, p) * torch.pow(v, q)
            delta_u += term
            
        # Apply BP coefficients
        for (p, q), coeff in self.bp_coeffs.items():
            term = coeff * torch.pow(u, p) * torch.pow(v, q)
            delta_v += term
            
        return u + delta_u, v + delta_v
