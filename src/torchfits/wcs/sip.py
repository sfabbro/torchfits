import torch
from torch import Tensor
from typing import Dict, Any


class SIP:
    """
    Simple Imaging Polynomial (SIP) distortion correction.

    This class handles the parsing of SIP coefficients from a FITS header
    and applies the distortion correction to pixel coordinates.

    References:
    - Shupe et al. (2005): "The SIP Convention for Representing Distortion in FITS Image Headers"
    """

    def __init__(self, header: Dict[str, Any]):
        self.a_order = int(header.get("A_ORDER", 0))
        self.b_order = int(header.get("B_ORDER", 0))
        self.ap_order = int(header.get("AP_ORDER", 0))
        self.bp_order = int(header.get("BP_ORDER", 0))

        # Parse A/B coefficients (Forward: Pixel -> Focal Plane)
        self.a_coeffs = self._parse_coeffs(header, "A", self.a_order)
        self.b_coeffs = self._parse_coeffs(header, "B", self.b_order)

        # Parse AP/BP coefficients (Inverse: Focal Plane -> Pixel)
        self.ap_coeffs = self._parse_coeffs(header, "AP", self.ap_order)
        self.bp_coeffs = self._parse_coeffs(header, "BP", self.bp_order)

    def _parse_coeffs(
        self, header: Dict[str, Any], prefix: str, order: int
    ) -> Dict[str, float]:
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

                key = f"{prefix}_{p}_{q}"
                if key in header:
                    coeffs[(p, q)] = float(header[key])
        return coeffs

    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply forward distortion with power caches.
        """
        if u.numel() == 0:
            return u, v

        # For large N, chunking is handle at the caller level in WCS if needed,
        # but here we just optimize the impl.

        # Max order across all polynomials
        max_order = max(self.a_order, self.b_order, 1)

        def make_pow_cache(base, order):
            pows = [torch.ones_like(base)]
            if order >= 1:
                pows.append(base)
            curr = base
            for _ in range(2, order + 1):
                curr = curr * base
                pows.append(curr)
            return torch.stack(pows, dim=0)

        u_p = make_pow_cache(u, max_order)
        v_p = make_pow_cache(v, max_order)

        f_uv = torch.zeros_like(u)
        g_uv = torch.zeros_like(v)

        for (p, q), coeff in self.a_coeffs.items():
            f_uv += coeff * u_p[p] * v_p[q]

        for (p, q), coeff in self.b_coeffs.items():
            g_uv += coeff * u_p[p] * v_p[q]

        return u + f_uv, v + g_uv

    def undistort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply inverse distortion using power caches.
        """
        if u.numel() == 0:
            return u, v

        max_order = max(self.ap_order, self.bp_order, 1)

        def make_pow_cache(base, order):
            pows = [torch.ones_like(base)]
            if order >= 1:
                pows.append(base)
            curr = base
            for _ in range(2, order + 1):
                curr = curr * base
                pows.append(curr)
            return torch.stack(pows, dim=0)

        u_p = make_pow_cache(u, max_order)
        v_p = make_pow_cache(v, max_order)

        delta_u = torch.zeros_like(u)
        delta_v = torch.zeros_like(v)

        for (p, q), coeff in self.ap_coeffs.items():
            delta_u += coeff * u_p[p] * v_p[q]

        for (p, q), coeff in self.bp_coeffs.items():
            delta_v += coeff * u_p[p] * v_p[q]

        return u + delta_u, v + delta_v
