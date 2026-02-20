
import torch
from torch import Tensor
from typing import Dict, Any

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
        self.power_map = self._build_power_map()
        self.idx1, self.c1 = self._parse_pv(header, 1)
        self.idx2, self.c2 = self._parse_pv(header, 2)
        
    def _build_power_map(self):
        # Map j (0-39) to (px, py, pr)
        # Degree 0
        mapping = {0: (0,0,0)}
        idx = 1
        for deg in range(1, 8):
            # Normal terms: x^{deg-k} y^k
            for k in range(deg + 1):
                mapping[idx] = (deg - k, k, 0)
                idx += 1
            # Radial term if odd
            if deg % 2 == 1:
                mapping[idx] = (0, 0, deg)
                idx += 1
        return mapping
        
    def _parse_pv(self, header: Dict[str, Any], axis: int):
        """Parse PV keywords into tensors."""
        indices = []
        coeffs = []
        
        for j in range(40):
            key = f'PV{axis}_{j}'
            if key in header:
                val = float(header[key])
                if val != 0:
                    if j in self.power_map:
                        indices.append(self.power_map[j])
                        coeffs.append(val)
                    
        if not indices:
            return torch.empty((0, 3), dtype=torch.long), torch.empty((0), dtype=torch.float64)
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(coeffs, dtype=torch.float64)

    def to(self, device: torch.device) -> "TPV":
        """Move TPV coefficient tensors to device."""
        self.idx1 = self.idx1.to(device)
        self.idx2 = self.idx2.to(device)
        self.c1 = self.c1.to(device)
        self.c2 = self.c2.to(device)
        return self

        
    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply TPV distortion with chunking to optimize memory usage.
        """
        if u.numel() == 0:
            return torch.zeros_like(u), torch.zeros_like(v)
            
        # For small N, run directly
        if u.numel() <= 256000:
            return self._distort_impl(u, v)
            
        # Process in chunks to stay within memory/cache limits
        xi = torch.empty_like(u)
        eta = torch.empty_like(v)
        chunk_size = 256000
        
        for i in range(0, u.numel(), chunk_size):
            end = min(i + chunk_size, u.numel())
            u_c = u[i:end]
            v_c = v[i:end]
            xi_c, eta_c = self._distort_impl(u_c, v_c)
            xi[i:end] = xi_c
            eta[i:end] = eta_c
            
        return xi, eta

    def _distort_impl(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """Internal distortion implementation."""
        # Precompute powers and r
        r = torch.sqrt(u*u + v*v)
        
        def make_pow_cache(base, max_deg=7):
            pows = [torch.ones_like(base)]
            curr = base
            pows.append(curr)
            for _ in range(2, max_deg + 1):
                curr = curr * base
                pows.append(curr)
            return torch.stack(pows, dim=0)
            
        x_p_cache = make_pow_cache(u) # (8, N)
        y_p_cache = make_pow_cache(v) # (8, N)
        r_p_cache = make_pow_cache(r) # (8, N)
        
        def eval_poly(p_indices, p_coeffs, xc, yc):
            if len(p_coeffs) == 0:
                return torch.zeros_like(u)
                
            px = p_indices[:, 0]
            py = p_indices[:, 1]
            pr = p_indices[:, 2]
            
            # Gather terms: x^px * y^py * r^pr
            term_val = xc[px] * yc[py] * r_p_cache[pr]
            
            # Dot product (ensure dtypes match)
            coeffs = p_coeffs.to(device=term_val.device, dtype=term_val.dtype)
            return torch.einsum('k, kn -> n', coeffs, term_val)

        # TPV Convention: 
        # Axis 1 uses (u, v)
        # Axis 2 uses (v, u) <-- SWAPPED
        xi = eval_poly(self.idx1, self.c1, x_p_cache, y_p_cache)
        eta = eval_poly(self.idx2, self.c2, y_p_cache, x_p_cache)

        return xi, eta
