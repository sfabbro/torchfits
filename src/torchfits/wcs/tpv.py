import torch
from torch import Tensor
from typing import Dict, Any, Tuple

try:
    import torchfits.cpp as _cpp
except Exception:  # pragma: no cover - optional fast path
    _cpp = None


class TPV:
    """
    Tangent PV (TPV) distortion correction.
    """

    # Experimental CPU ATen TPV inverse path.
    _CPP_INVERT_MAX_POINTS = 65536

    def __init__(self, header: Dict[str, Any]):
        self.power_map = self._build_power_map()
        self.idx1, self.c1 = self._parse_pv(header, 1)
        self.idx2, self.c2 = self._parse_pv(header, 2)
        self.terms1 = self._build_terms(self.idx1, self.c1)
        self.terms2 = self._build_terms(self.idx2, self.c2)

        self.terms1_dx_poly, self.terms1_dy_poly, self.terms1_rad = (
            self._build_derivative_terms(self.terms1)
        )
        self.terms2_dx_poly, self.terms2_dy_poly, self.terms2_rad = (
            self._build_derivative_terms(self.terms2)
        )

        (
            self._affine_seed_b1,
            self._affine_seed_b2,
            self._affine_seed_inv00,
            self._affine_seed_inv01,
            self._affine_seed_inv10,
            self._affine_seed_inv11,
            self._has_affine_seed_inverse,
            self._affine_seed_is_identity,
        ) = self._build_affine_seed_params(self.terms1, self.terms2)

        self._invert_trace_enabled = False
        self._last_invert_trace = None

    def _build_power_map(self):
        mapping = {0: (0, 0, 0)}
        idx = 1
        for deg in range(1, 8):
            for k in range(deg + 1):
                mapping[idx] = (deg - k, k, 0)
                idx += 1
            if deg % 2 == 1:
                mapping[idx] = (0, 0, deg)
                idx += 1
        return mapping

    def _parse_pv(self, header: Dict[str, Any], axis: int):
        indices = []
        coeffs = []
        for j in range(40):
            key = f"PV{axis}_{j}"
            if key in header:
                val = float(header[key])
                if val != 0 and j in self.power_map:
                    indices.append(self.power_map[j])
                    coeffs.append(val)
        if not indices:
            return torch.empty((0, 3), dtype=torch.long), torch.empty(
                (0), dtype=torch.float64
            )
        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            coeffs, dtype=torch.float64
        )

    def _build_terms(
        self, idx: Tensor, coeffs: Tensor
    ) -> "list[tuple[int, int, int, float]]":
        return [
            (int(idx[k, 0]), int(idx[k, 1]), int(idx[k, 2]), float(coeffs[k]))
            for k in range(coeffs.numel())
        ]

    def _build_derivative_terms(self, terms):
        dx, dy, dr = [], [], []
        for px, py, pr, c in terms:
            if px > 0:
                dx.append((px - 1, py, pr, c * px))
            if py > 0:
                dy.append((px, py - 1, pr, c * py))
            if pr > 0:
                dr.append((px, py, pr - 1, c * pr))
        return dx, dy, dr

    @staticmethod
    def _build_affine_seed_params(terms1, terms2):
        b1, b2, a11, a12, a21, a22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for px, py, pr, c in terms1:
            if pr != 0:
                continue
            if px == 0 and py == 0:
                b1 += c
            elif px == 1 and py == 0:
                a11 += c
            elif px == 0 and py == 1:
                a12 += c
        for px, py, pr, c in terms2:
            if pr != 0:
                continue
            if px == 0 and py == 0:
                b2 += c
            elif px == 1 and py == 0:
                a22 += c
            elif px == 0 and py == 1:
                a21 += c
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-15:
            return b1, b2, 1.0, 0.0, 0.0, 1.0, False, True
        is_id = (
            abs(b1) < 1e-15
            and abs(b2) < 1e-15
            and abs(a11 - 1.0) < 1e-15
            and abs(a12) < 1e-15
            and abs(a21) < 1e-15
            and abs(a22 - 1.0) < 1e-15
        )
        return b1, b2, a22 / det, -a12 / det, -a21 / det, a11 / det, True, is_id

    def to(self, device: torch.device) -> "TPV":
        self.idx1, self.idx2 = self.idx1.to(device), self.idx2.to(device)
        self.c1, self.c2 = self.c1.to(device), self.c2.to(device)
        return self

    def distort(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        if u.numel() == 0:
            return u, v
        return self._distort_impl(u, v)

    def _distort_impl(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r = torch.sqrt(u * u + v * v)
        xp = [torch.ones_like(u), u]
        yp = [torch.ones_like(v), v]
        rp = [torch.ones_like(r), r]
        for _ in range(2, 8):
            xp.append(xp[-1] * u)
            yp.append(yp[-1] * v)
            rp.append(rp[-1] * r)

        xi = torch.zeros_like(u)
        for px, py, pr, c in self.terms1:
            xi.add_(xp[px] * yp[py] * rp[pr], alpha=c)
        eta = torch.zeros_like(v)
        for px, py, pr, c in self.terms2:
            eta.add_(yp[px] * xp[py] * rp[pr], alpha=c)
        return xi, eta

    def _distort_and_jacobian_impl(
        self, u: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r = torch.sqrt(u * u + v * v)
        xp, yp, rp = (
            [torch.ones_like(u), u],
            [torch.ones_like(v), v],
            [torch.ones_like(r), r],
        )
        for _ in range(2, 8):
            xp.append(xp[-1] * u)
            yp.append(yp[-1] * v)
            rp.append(rp[-1] * r)

        r_safe = torch.where(r > 0, r, torch.ones_like(r))
        dr_du = torch.where(r > 0, u / r_safe, torch.zeros_like(u))
        dr_dv = torch.where(r > 0, v / r_safe, torch.zeros_like(v))

        def eval_poly_jac(terms, dx_p, dy_p, dr_p, xc, yc, dr_da, dr_db):
            out = torch.zeros_like(u)
            for px, py, pr, c in terms:
                out.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            da = torch.zeros_like(u)
            for px, py, pr, c in dx_p:
                da.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            db = torch.zeros_like(u)
            for px, py, pr, c in dy_p:
                db.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            if dr_p:
                pref = torch.zeros_like(u)
                for px, py, pr, c in dr_p:
                    pref.add_(xc[px] * yc[py] * rp[pr], alpha=c)
                da.add_(pref * dr_da)
                db.add_(pref * dr_db)
            return out, da, db

        xi, dxi_du, dxi_dv = eval_poly_jac(
            self.terms1,
            self.terms1_dx_poly,
            self.terms1_dy_poly,
            self.terms1_rad,
            xp,
            yp,
            dr_du,
            dr_dv,
        )
        eta, deta_dv, deta_du = eval_poly_jac(
            self.terms2,
            self.terms2_dx_poly,
            self.terms2_dy_poly,
            self.terms2_rad,
            yp,
            xp,
            dr_dv,
            dr_du,
        )
        return xi, eta, dxi_du, dxi_dv, deta_du, deta_dv

    def invert(
        self, xi_t: Tensor, eta_t: Tensor, max_iter: int = 10, tol: float = 1e-11
    ) -> Tuple[Tensor, Tensor]:
        if xi_t.numel() == 0:
            return xi_t, eta_t
        if self._can_use_cpp_invert(xi_t, eta_t):
            return _cpp.wcs_tpv_invert(
                xi_t.contiguous(),
                eta_t.contiguous(),
                self.idx1,
                self.c1,
                self.idx2,
                self.c2,
                int(max_iter),
                float(tol),
            )
        u, v = self._initial_guess_affine(xi_t, eta_t)
        tol2 = tol * tol
        for _ in range(max_iter):
            xi, eta, j11, j12, j21, j22 = self._distort_and_jacobian_impl(u, v)
            rx, ry = xi - xi_t, eta - eta_t
            if (rx * rx + ry * ry).max() < tol2:
                break
            det = j11 * j22 - j12 * j21
            det = torch.where(det.abs() < 1e-18, torch.sign(det) * 1e-18, det)
            u = u - (j22 * rx - j12 * ry) / det
            v = v - (-j21 * rx + j11 * ry) / det
        return u, v

    def _initial_guess_affine(self, xi_t, eta_t):
        if not self._has_affine_seed_inverse:
            return xi_t.clone(), eta_t.clone()
        xi0, eta0 = xi_t - self._affine_seed_b1, eta_t - self._affine_seed_b2
        u = self._affine_seed_inv00 * xi0 + self._affine_seed_inv01 * eta0
        v = self._affine_seed_inv10 * xi0 + self._affine_seed_inv11 * eta0
        return u, v

    def _can_use_cpp_invert(self, xi_t, eta_t):
        return bool(
            _cpp is not None
            and hasattr(_cpp, "wcs_tpv_invert")
            and xi_t.device.type == "cpu"
            and eta_t.device.type == "cpu"
            and xi_t.dtype == torch.float64
            and eta_t.dtype == torch.float64
            and self.idx1.device.type == "cpu"
            and self.idx2.device.type == "cpu"
            and self.c1.device.type == "cpu"
            and self.c2.device.type == "cpu"
        )
