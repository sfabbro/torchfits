
import torch
from torch import Tensor
from typing import Dict, Any, Optional

class TPV:
    """
    Tangent PV (TPV) distortion correction.
    
    Used by SCAMP and SWarp. This defines a polynomial distortion on the 
    native tangent plane.
    
    The transformation is:
    xi = P_xi(u, v)
    eta = P_eta(u, v)
    
    where u, v are "intermediate" coordinates (typically linear pixel coords relative to CRPIX).
    
    Coefficients are stored in PV1_j (for xi) and PV2_j (for eta).
    The mapping of j to polynomial terms (1, x, y, r, x^2, xy, y^2, ...) 
    follows the TPV convention (see Calabretta's WCSLIB or SCAMP documentation).
    
    Standard TPV Polynomial Terms (j=0..39 typically):
    0: 1
    1: x
    2: y
    3: r = sqrt(x^2 + y^2)
    4: x^2
    5: xy
    6: y^2
    7: x^3
    8: x^2y
    9: xy^2
    10: y^3
    ... and so on.
    """
    def __init__(self, header: Dict[str, Any]):
        self.pv1 = self._parse_pv(header, 1)
        self.pv2 = self._parse_pv(header, 2)
        
    def _parse_pv(self, header: Dict[str, Any], axis: int) -> Dict[int, float]:
        """Parse PV{axis}_{j} keywords."""
        coeffs = {}
        # TPV coeffs typically go up to j=39 (degree 7?)
        for j in range(40):
            key = f'PV{axis}_{j}'
            if key in header:
                coeffs[j] = float(header[key])
        return coeffs

    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply TPV distortion.
        u, v: Linear intermediate coordinates (relative to CRPIX, scaled by CD?)
        Actually, TPV is usually defined on the "pixel coordinates" (relative to CRPIX)
        OR on the "intermediate coordinates" (after CD matrix).
        
        Most sources (e.g., astropy.wcs) implementation of TPV:
        It's a "Projection" type.
        1. (pixel - CRPIX) -> (U, V) [applying CD if present, or just CDELT]
           Wait, TPV usually *replaces* the standard TAN projection steps.
           
           SCAMP description:
           xi = P_xi(x, y)
           eta = P_eta(x, y)
           where (x,y) are relative pixel coordinates?
           
           Let's verify standard TPV behavior.
           In WCSLIB, TPV is implemented via `dis.c`/`tab.c`? No, it's often a separate registration.
           
           According to SCAMP:
           "TPV... polynomial distortion of the tangent plane."
           Inputs to polynomial are "projected coordinates" x, y ??
           Usually inputs are (u, v) = CD * (pix - crpix).
           
           Let's implement the polynomial evaluator assuming inputs `x, y` are 
           intermediate coordinates.
        """
        
        # Precompute powers 
        x = u
        y = v
        r = torch.sqrt(x*x + y*y)
        
        x2 = x*x
        y2 = y*y
        xy = x*y
        
        # 0: 1
        # 1: x
        # 2: y
        # 3: r
        # 4: x^2
        # 5: xy
        # 6: y^2
        # 7: x^3
        # 8: x^2y
        # 9: xy^2
        # 10: y^3
        # 11: r^3
        
        # We need a robust way to map j -> term.
        # This is a fixed mapping for TPV.
        
        # Let's construct the specialized polynomial evaluator for the standard terms.
        
        
        # Standard TPV mapping (SCAMP/SWarp convention)
        # 0: 1
        # 1: x
        # 2: y
        # 3: r
        # 4: x^2
        # 5: xy
        # 6: y^2
        # 7: x^3
        # 8: x^2y
        # 9: xy^2
        # 10: y^3
        # 11: r^3
        # 12: x^4
        # 13: x^3y
        # 14: x^2y^2
        # 15: xy^3
        # 16: y^4
        # 17: r^4 ... ? No, typically r terms are specific indices
        # Let's simple list based on polynomial order
        
        xi = torch.zeros_like(x)
        eta = torch.zeros_like(y)
        
        # Helper to add term
        def add(out, coeffs, term, j):
            if j in coeffs:
                out += coeffs[j] * term

        # Precompute powers
        x2 = x*x
        y2 = y*y
        xy = x*y
        r = torch.sqrt(x2 + y2)
        
        x3 = x2*x
        y3 = y2*y
        x2y = x2*y
        xy2 = x*y2
        r3 = r*r*r
        
        # Loop for both axes
        for axis_tensor, coeffs in [(xi, self.pv1), (eta, self.pv2)]:
            # 0: 1
            add(axis_tensor, coeffs, 1.0, 0)
            # 1: x
            add(axis_tensor, coeffs, x, 1)
            # 2: y
            add(axis_tensor, coeffs, y, 2)
            # 3: r
            add(axis_tensor, coeffs, r, 3)
            # 4: x^2
            add(axis_tensor, coeffs, x2, 4)
            # 5: xy
            add(axis_tensor, coeffs, xy, 5)
            # 6: y^2
            add(axis_tensor, coeffs, y2, 6)
            # 7: x^3
            add(axis_tensor, coeffs, x3, 7)
            # 8: x^2y
            add(axis_tensor, coeffs, x2y, 8)
            # 9: xy^2
            add(axis_tensor, coeffs, xy2, 9)
            # 10: y^3
            add(axis_tensor, coeffs, y3, 10)
            # 11: r^3
            add(axis_tensor, coeffs, r3, 11)
            
            # Higher orders (up to 39 implemented sparsely for now)
            # We can expand this list as needed for specific benchmarks
            
        return xi, eta
