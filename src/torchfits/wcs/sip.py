import torch
from torch import Tensor
from typing import Dict, Any, Tuple

try:
    from torch.func import vmap, jacrev
except ImportError:
    # Fallback for very old torch versions if needed, though we assume modern torch
    vmap = None
    jacrev = None


class SIP:
    """
    Simple Imaging Polynomial (SIP) distortion correction.

    This class handles the parsing of SIP coefficients from a FITS header
    and applies the distortion correction to pixel coordinates.

    References:
    - Shupe et al. (2005): "The SIP Convention for Representing Distortion in FITS Image Headers"
    """

    _SMALL_VEC_MAX_POINTS = 0

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
        self.a_terms = self._coeff_terms(self.a_coeffs)
        self.b_terms = self._coeff_terms(self.b_coeffs)
        self.ap_terms = self._coeff_terms(self.ap_coeffs)
        self.bp_terms = self._coeff_terms(self.bp_coeffs)
        self.a_du_terms, self.a_dv_terms = self._derivative_terms(self.a_terms)
        self.b_du_terms, self.b_dv_terms = self._derivative_terms(self.b_terms)
        self._a_pack = self._build_term_pack(self.a_terms)
        self._b_pack = self._build_term_pack(self.b_terms)
        self._ap_pack = self._build_term_pack(self.ap_terms)
        self._bp_pack = self._build_term_pack(self.bp_terms)
        self._a_du_pack = self._build_term_pack(self.a_du_terms)
        self._a_dv_pack = self._build_term_pack(self.a_dv_terms)
        self._b_du_pack = self._build_term_pack(self.b_du_terms)
        self._b_dv_pack = self._build_term_pack(self.b_dv_terms)
        # Pre-build tensors for vectorized evaluation
        self._a_c = torch.tensor([t[2] for t in self.a_terms], dtype=torch.float64)
        self._a_pq = torch.tensor(
            [[t[0], t[1]] for t in self.a_terms], dtype=torch.long
        )
        self._b_c = torch.tensor([t[2] for t in self.b_terms], dtype=torch.float64)
        self._b_pq = torch.tensor(
            [[t[0], t[1]] for t in self.b_terms], dtype=torch.long
        )

        self._ap_c = torch.tensor([t[2] for t in self.ap_terms], dtype=torch.float64)
        self._ap_pq = torch.tensor(
            [[t[0], t[1]] for t in self.ap_terms], dtype=torch.long
        )
        self._bp_c = torch.tensor([t[2] for t in self.bp_terms], dtype=torch.float64)
        self._bp_pq = torch.tensor(
            [[t[0], t[1]] for t in self.bp_terms], dtype=torch.long
        )

        self.has_forward = bool(self.a_coeffs) or bool(self.b_coeffs)
        self.has_inverse = bool(self.ap_coeffs) or bool(self.bp_coeffs)

    def to(self, device: torch.device) -> "SIP":
        """Move SIP tensors to device."""
        self._a_c = self._a_c.to(device)
        self._a_pq = self._a_pq.to(device)
        self._b_c = self._b_c.to(device)
        self._b_pq = self._b_pq.to(device)
        self._ap_c = self._ap_c.to(device)
        self._ap_pq = self._ap_pq.to(device)
        self._bp_c = self._bp_c.to(device)
        self._bp_pq = self._bp_pq.to(device)
        return self

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

    @staticmethod
    def _coeff_terms(
        coeffs: Dict[tuple[int, int], float],
    ) -> "list[tuple[int, int, float]]":
        return [(int(p), int(q), float(c)) for (p, q), c in coeffs.items()]

    @staticmethod
    def _derivative_terms(
        terms: "list[tuple[int, int, float]]",
    ) -> "tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]":
        du_terms = []
        dv_terms = []
        for p, q, coeff in terms:
            if p > 0:
                du_terms.append((p - 1, q, coeff * p))
            if q > 0:
                dv_terms.append((p, q - 1, coeff * q))
        return du_terms, dv_terms

    @staticmethod
    def _build_term_pack(terms: "list[tuple[int, int, float]]") -> Dict[str, Tensor]:
        if not terms:
            return {
                "p": torch.empty((0,), dtype=torch.long),
                "q": torch.empty((0,), dtype=torch.long),
                "c": torch.empty((0,), dtype=torch.float64),
            }
        p = torch.tensor([t[0] for t in terms], dtype=torch.long)
        q = torch.tensor([t[1] for t in terms], dtype=torch.long)
        c = torch.tensor([t[2] for t in terms], dtype=torch.float64)
        return {"p": p, "q": q, "c": c}

    @staticmethod
    def _sum_terms_from_pack(
        pack: Dict[str, Tensor], u_stack: Tensor, v_stack: Tensor, out_like: Tensor
    ) -> Tensor:
        c = pack["c"].to(device=out_like.device, dtype=out_like.dtype)
        if c.numel() == 0:
            return torch.zeros_like(out_like)
        p = pack["p"].to(out_like.device)
        q = pack["q"].to(out_like.device)
        vals = u_stack.index_select(0, p) * v_stack.index_select(0, q)
        return (vals * c[:, None]).sum(dim=0)

    def _distort_scalar(self, uv: Tensor) -> Tensor:
        """
        Distort a single (u, v) point. uv shape [2].
        Returns [2].
        Used by vmap and jacrev.
        """
        u, v = uv[0], uv[1]
        
        # Build powers manually for scalar
        max_order = max(self.a_order, self.b_order)
        up = torch.pow(u, torch.arange(max_order + 1, device=uv.device, dtype=uv.dtype))
        vp = torch.pow(v, torch.arange(max_order + 1, device=uv.device, dtype=uv.dtype))
        
        # Evaluation
        def eval_poly_scalar(pq, c):
            if c.numel() == 0:
                return torch.tensor(0.0, device=uv.device, dtype=uv.dtype)
            p = pq[:, 0]
            q = pq[:, 1]
            basis = up[p] * vp[q]
            return torch.dot(c.to(uv.dtype), basis)
            
        f_uv = eval_poly_scalar(self._a_pq, self._a_c)
        g_uv = eval_poly_scalar(self._b_pq, self._b_c)
        
        return uv + torch.stack([f_uv, g_uv])

    def distort_vmap(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply distortion using torch.func.vmap (JAX-style)."""
        uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1) # [N, 2]
        out = vmap(self._distort_scalar)(uv) # [N, 2]
        return out[:, 0].reshape(u.shape), out[:, 1].reshape(v.shape)

    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply forward distortion using vectorized matrix-vector multiplication.
        """
        if u.numel() == 0 or not self.has_forward:
            return u, v

        max_order = max(self.a_order, self.b_order)
        u_p = [torch.ones_like(u)]
        v_p = [torch.ones_like(v)]
        for _ in range(1, max_order + 1):
            u_p.append(u_p[-1] * u)
            v_p.append(v_p[-1] * v)

        u_stack = torch.stack(u_p, dim=0)
        v_stack = torch.stack(v_p, dim=0)

        def eval_poly(pq, c):
            if c.numel() == 0:
                return torch.zeros_like(u)
            # Basis: u^p * v^q
            # pq is (N_terms, 2)
            p = pq[:, 0]
            q = pq[:, 1]
            basis = u_stack.index_select(0, p) * v_stack.index_select(
                0, q
            )  # (N_terms, N_points)
            return torch.matmul(c.to(u.dtype), basis)

        f_uv = eval_poly(self._a_pq, self._a_c)
        g_uv = eval_poly(self._b_pq, self._b_c)

        return u + f_uv, v + g_uv

    def _distort_smallvec(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        max_order = max(self.a_order, self.b_order, 1)

        def make_pow_stack(base, order):
            pows = [torch.ones_like(base)]
            if order >= 1:
                pows.append(base)
            curr = base
            for _ in range(2, order + 1):
                curr = curr * base
                pows.append(curr)
            return torch.stack(pows, dim=0)

        u_s = make_pow_stack(u, max_order)
        v_s = make_pow_stack(v, max_order)
        f_uv = self._sum_terms_from_pack(self._a_pack, u_s, v_s, u)
        g_uv = self._sum_terms_from_pack(self._b_pack, u_s, v_s, v)
        return u + f_uv, v + g_uv

    def _forward_and_jacobian(
        self, u: Tensor, v: Tensor
    ) -> "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]":
        """
        Evaluate SIP forward map and Jacobian in one pass.

        Returns:
            xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv
        """
        if u.numel() <= self._SMALL_VEC_MAX_POINTS:
            return self._forward_and_jacobian_smallvec(u, v)

        max_order = max(self.a_order, self.b_order, 1)

        def make_pow_cache(base, order):
            pows = [torch.ones_like(base)]
            if order >= 1:
                pows.append(base)
            curr = base
            for _ in range(2, order + 1):
                curr = curr * base
                pows.append(curr)
            return pows

        u_p = make_pow_cache(u, max_order)
        v_p = make_pow_cache(v, max_order)

        xd = u.clone()
        yd = v.clone()
        dxd_du = torch.ones_like(u)
        dxd_dv = torch.zeros_like(u)
        dyd_du = torch.zeros_like(v)
        dyd_dv = torch.ones_like(v)

        for p, q, coeff in self.a_terms:
            base = coeff * u_p[p] * v_p[q]
            xd = xd + base

        for p, q, coeff in self.a_du_terms:
            dxd_du = dxd_du + coeff * u_p[p] * v_p[q]
        for p, q, coeff in self.a_dv_terms:
            dxd_dv = dxd_dv + coeff * u_p[p] * v_p[q]

        for p, q, coeff in self.b_terms:
            base = coeff * u_p[p] * v_p[q]
            yd = yd + base
        for p, q, coeff in self.b_du_terms:
            dyd_du = dyd_du + coeff * u_p[p] * v_p[q]
        for p, q, coeff in self.b_dv_terms:
            dyd_dv = dyd_dv + coeff * u_p[p] * v_p[q]

        return xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv

    def _forward_and_jacobian_smallvec(
        self, u: Tensor, v: Tensor
    ) -> "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]":
        max_order = max(self.a_order, self.b_order, 1)

        def make_pow_stack(base, order):
            pows = [torch.ones_like(base)]
            if order >= 1:
                pows.append(base)
            curr = base
            for _ in range(2, order + 1):
                curr = curr * base
                pows.append(curr)
            return torch.stack(pows, dim=0)

        u_s = make_pow_stack(u, max_order)
        v_s = make_pow_stack(v, max_order)

        xd = u + self._sum_terms_from_pack(self._a_pack, u_s, v_s, u)
        yd = v + self._sum_terms_from_pack(self._b_pack, u_s, v_s, v)
        dxd_du = torch.ones_like(u) + self._sum_terms_from_pack(
            self._a_du_pack, u_s, v_s, u
        )
        dxd_dv = self._sum_terms_from_pack(self._a_dv_pack, u_s, v_s, u)
        dyd_du = self._sum_terms_from_pack(self._b_du_pack, u_s, v_s, v)
        dyd_dv = torch.ones_like(v) + self._sum_terms_from_pack(
            self._b_dv_pack, u_s, v_s, v
        )
        return xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv

    def undistort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply inverse distortion using matrix-vectorized evaluation.
        """
        if u.numel() == 0 or not self.has_inverse:
            return u, v

        max_order = max(self.ap_order, self.bp_order)
        u_p = [torch.ones_like(u)]
        v_p = [torch.ones_like(v)]
        for _ in range(1, max_order + 1):
            u_p.append(u_p[-1] * u)
            v_p.append(v_p[-1] * v)

        u_stack = torch.stack(u_p, dim=0)
        v_stack = torch.stack(v_p, dim=0)

        def eval_poly(pq, c):
            if c.numel() == 0:
                return torch.zeros_like(u)
            p = pq[:, 0]
            q = pq[:, 1]
            basis = u_stack.index_select(0, p) * v_stack.index_select(0, q)
            return torch.matmul(c.to(u.dtype), basis)

        delta_u = eval_poly(self._ap_pq, self._ap_c)
        delta_v = eval_poly(self._bp_pq, self._bp_c)

        return u + delta_u, v + delta_v

    def invert_distortion(
        self,
        u_dist: Tensor,
        v_dist: Tensor,
        max_iter: int = 6,
        tol: float = 1e-6,
    ) -> "tuple[Tensor, Tensor]":
        """
        Invert SIP forward distortion map (u, v) -> distort(u, v).
        Uses a vectorized Newton solver with manual Jacobian for high performance.
        """
        if u_dist.numel() == 0 or not self.has_forward:
            return u_dist, v_dist

        u_f = u_dist.reshape(-1)
        v_f = v_dist.reshape(-1)

        # Initial guess from inverse coefficients if available
        if self.has_inverse:
            u_curr, v_curr = self.undistort(u_f, v_f)
        else:
            u_curr, v_curr = u_f.clone(), v_f.clone()

        # Vectorized Newton iteration: x = x - inv(J) * f(x)
        # where f(x) = distort(x) - target_dist
        for _ in range(max_iter):
            # Compute current forward map and its Jacobian at guess (u_curr, v_curr)
            xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv = self._forward_and_jacobian(
                u_curr, v_curr
            )

            # Residuals
            du_res = xd - u_f
            dv_res = yd - v_f

            # Check convergence
            if torch.all(du_res.abs() < tol) and torch.all(dv_res.abs() < tol):
                break

            # Solve linear system J * [du, dv]^T = [du_res, dv_res]^T
            # [ dxd_du  dxd_dv ] [ du ] = [ du_res ]
            # [ dyd_du  dyd_dv ] [ dv ]   [ dv_res ]
            det = dxd_du * dyd_dv - dxd_dv * dyd_du

            # Avoid division by zero
            det_sign = torch.sign(det)
            det_sign = torch.where(det_sign == 0, torch.ones_like(det_sign), det_sign)
            det = torch.where(det.abs() < 1e-15, det_sign * 1e-15, det)

            du_step = (dyd_dv * du_res - dxd_dv * dv_res) / det
            dv_step = (-dyd_du * du_res + dxd_du * dv_res) / det

            u_curr = u_curr - du_step
            v_curr = v_curr - dv_step

        return u_curr.reshape(u_dist.shape), v_curr.reshape(v_dist.shape)
