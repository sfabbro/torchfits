import torch
from torch import Tensor
from typing import Optional, Union, Tuple, Dict, Any
import math
import os

try:
    import torchfits.cpp as _cpp
except Exception:  # pragma: no cover - optional fast path
    _cpp = None

from .tpv import TPV, PV1_KEYS, PV2_KEYS
from .sip import SIP
from .zenithal import project_zenithal, deproject_zenithal
from .cylindrical import project_cylindrical, deproject_cylindrical
from .allsky import project_allsky, deproject_allsky
from .legacy import project_tnx, project_zpx
from .utils import solve_newton_raphson

_HAS_TORCH_SINCOS = hasattr(torch, "sincos")




def _sincos(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Return sin(x), cos(x) with a torch.sincos fast path when available."""
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)
    return torch.sin(x), torch.cos(x)


def _wrap_lon360(angle: Tensor) -> Tensor:
    """Wrap longitudes to [0, 360)."""
    if angle.requires_grad:
        return torch.remainder(angle, 360.0)
    mn = torch.amin(angle)
    mx = torch.amax(angle)
    if bool((mn >= 0.0) and (mx < 360.0)):
        return angle
    if bool((mn >= -360.0) and (mx < 720.0)):
        out = angle
        if bool(mn < 0.0):
            out = torch.where(out < 0.0, out + 360.0, out)
        if bool(mx >= 360.0):
            out = torch.where(out >= 360.0, out - 360.0, out)
        return out
    return torch.remainder(angle, 360.0)


def _wrap_lon360_checked(angle: Tensor) -> Tensor:
    """Compatibility alias; direct wrap is faster than range-check reductions."""
    return _wrap_lon360(angle)


def _wrap_lon180(angle: Tensor) -> Tensor:
    """Wrap longitudes to [-180, 180)."""
    if angle.requires_grad:
        return torch.remainder(angle + 180.0, 360.0) - 180.0
    mn = torch.amin(angle)
    mx = torch.amax(angle)
    if bool((mn >= -180.0) and (mx < 180.0)):
        return angle
    if bool((mn >= -540.0) and (mx < 540.0)):
        out = angle
        if bool(mn < -180.0):
            out = torch.where(out < -180.0, out + 360.0, out)
        if bool(mx >= 180.0):
            out = torch.where(out >= 180.0, out - 360.0, out)
        return out
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    """Compatibility alias; direct wrap is faster than range-check reductions."""
    return _wrap_lon180(angle)


class WCS:
    """
    Base class for TorchWCS (PyTorch-native World Coordinate System).
    """

    ZENITHAL_CODES = ("TAN", "SIN", "ARC", "ZPN", "STG", "ZEA")
    CYLINDRICAL_CODES = ("CEA", "MER", "CYP")
    ALLSKY_CODES = ("AIT", "MOL", "HPX")
    SPECIAL_CODES = ("ZPX", "TNX", "TPV", "CAR", "SFL")

    def __init__(self, header: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize WCS object.

        Args:
            header: FITS header dictionary (or similar object behaving like dict)
            **kwargs: Manual override for WCS keywords (CRVAL, CRPIX, CD, etc.)
        """
        self.wcs_params = {}
        self.sip = None
        self.tpv = None
        self.wat_data = {}

        if header is not None:
            self.naxis = int(header.get("NAXIS", 2))
            self._parse_header(header)
            # Check for SIP
            if "A_ORDER" in header or "B_ORDER" in header:
                self.sip = SIP(header)

            # Check for TPV (Tangent with PV distortion)
            # SCAMP often uses CTYPE = RA---TAN with PV1_j keywords
            has_pv = any(k in self.wcs_params for k in PV1_KEYS) or any(
                k in self.wcs_params for k in PV2_KEYS
            )
            is_tpv_ctype = (
                self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TPV")
            )

            if is_tpv_ctype or (
                self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TAN")
                and has_pv
            ):
                self.tpv = TPV(self.wcs_params)

        # Override with kwargs
        for k, v in kwargs.items():
            k_upper = k.upper()
            if k_upper.startswith("CRVAL"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CRPIX"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CD"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CTYPE"):
                self.wcs_params[k_upper] = str(v)
            else:
                self.wcs_params[k_upper] = v

        # Default device
        self.device = torch.device("cpu")

        # Environment-controlled fused C++ path. Default is False because vectorized
        # PyTorch + torch.compile is typically 2x-4x faster than the scalar C++ loop.
        self._enable_fused_wcs = os.environ.get("TORCHFITS_WCS_FUSED", "0") == "1"

        # Move parameters to appropriate buffers/tensors
        self._setup_tensors()
        self._compiled_transform_fn = None
        self._compiled_pixel_to_world_2d_fn = None
        self._compiled_world_to_pixel_2d_fn = None

    def _parse_header(self, header: Dict[str, Any]):
        """Parse standard FITS WCS keywords."""
        # Standard defaults
        self.wcs_params["NAXIS"] = header.get("NAXIS", 2)

        # CTYPEs
        for i in range(1, self.naxis + 1):
            self.wcs_params[f"CTYPE{i}"] = str(header.get(f"CTYPE{i}", ""))
            self.wcs_params[f"CRPIX{i}"] = float(header.get(f"CRPIX{i}", 0.0))
            self.wcs_params[f"CRVAL{i}"] = float(header.get(f"CRVAL{i}", 0.0))
            self.wcs_params[f"CDELT{i}"] = float(header.get(f"CDELT{i}", 1.0))
            if f"CUNIT{i}" in header:
                self.wcs_params[f"CUNIT{i}"] = str(header.get(f"CUNIT{i}", ""))

        # Capture CD/PC matrix keywords
        for i in range(1, self.naxis + 1):
            for j in range(1, self.naxis + 1):
                cd_key = f"CD{i}_{j}"
                if cd_key in header:
                    self.wcs_params[cd_key] = float(header[cd_key])
                pc_key = f"PC{i}_{j}"
                if pc_key in header:
                    self.wcs_params[pc_key] = float(header[pc_key])

        # Pole defaults can be projection dependent, but preserve explicit header values
        if "LONPOLE" in header:
            self.wcs_params["LONPOLE"] = float(header["LONPOLE"])
        if "LATPOLE" in header:
            self.wcs_params["LATPOLE"] = float(header["LATPOLE"])

        # Capture PV keywords (PVi_j)
        # SCAMP/PV distortions can have many terms (0..39 typically)
        for key in PV1_KEYS:
            if key in header:
                self.wcs_params[key] = float(header[key])
        for key in PV2_KEYS:
            if key in header:
                self.wcs_params[key] = float(header[key])

        # Capture WAT keywords for TNX/ZPX
        # Reconstruct them here to avoid doing it every call
        self.wat_data = {}
        from .legacy import parse_wat_keywords  # Helper

        # Check for WAT1_*, WAT2_*
        has_wat = False
        for k in header.keys():
            if k.startswith("WAT"):
                has_wat = True
                break

        if has_wat:
            self.wat_data[1] = parse_wat_keywords(header, 1)
            self.wat_data[2] = parse_wat_keywords(header, 2)

    def _setup_tensors(self):
        """Convert scalar params to PyTorch tensors for computation."""
        self.crpix = torch.tensor(
            [self.wcs_params.get(f"CRPIX{i}", 0.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        self.crval = torch.tensor(
            [self.wcs_params.get(f"CRVAL{i}", 0.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        self.cdelt = torch.tensor(
            [self.wcs_params.get(f"CDELT{i}", 1.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        # 1b. CD matrix initialization (Paper I)
        # Check if CD keywords exist
        has_cd = any(
            f"CD{i}_{j}" in self.wcs_params
            for i in range(1, self.naxis + 1)
            for j in range(1, self.naxis + 1)
        )

        if has_cd:
            # Construct CD matrix
            cd_data = []
            for i in range(1, self.naxis + 1):
                row = []
                for j in range(1, self.naxis + 1):
                    row.append(
                        self.wcs_params.get(f"CD{i}_{j}", 1.0 if i == j else 0.0)
                    )
                cd_data.append(row)
            self.cd_full = torch.tensor(
                cd_data, dtype=torch.float64, device=self.device
            )
        else:
            # Construct from PC and CDELT: CDi_j = CDELTi * PCi_j
            cd_data = []
            for i in range(1, self.naxis + 1):
                row = []
                for j in range(1, self.naxis + 1):
                    pc_val = self.wcs_params.get(f"PC{i}_{j}", 1.0 if i == j else 0.0)
                    row.append(self.cdelt[i - 1].item() * pc_val)
                cd_data.append(row)
            self.cd_full = torch.tensor(
                cd_data, dtype=torch.float64, device=self.device
            )

        # Spatial shortcut for old code paths (2x2)
        if self.naxis >= 2:
            self.cd = self.cd_full[:2, :2]
        else:
            # 1x1 padded to 2x2 for safety in spatial paths
            self.cd = torch.eye(2, dtype=torch.float64, device=self.device)
            self.cd[0, 0] = self.cd_full[0, 0]

        # Precompute inverse CD matrix for world->pixel
        self.cd_inv_full = torch.inverse(self.cd_full)
        self.cd_inv = torch.inverse(self.cd)
        self._crpix0 = float(self.crpix[0].item()) if self.crpix.numel() >= 1 else 0.0
        self._crpix1 = float(self.crpix[1].item()) if self.crpix.numel() >= 2 else 0.0
        self._cd00 = float(self.cd[0, 0].item())
        self._cd01 = float(self.cd[0, 1].item())
        self._cd10 = float(self.cd[1, 0].item())
        self._cd11 = float(self.cd[1, 1].item())
        self._cd_is_diag = abs(self._cd01) < 1e-15 and abs(self._cd10) < 1e-15
        self._cdi00 = float(self.cd_inv[0, 0].item())
        self._cdi01 = float(self.cd_inv[0, 1].item())
        self._cdi10 = float(self.cd_inv[1, 0].item())
        self._cdi11 = float(self.cd_inv[1, 1].item())

        # 2. Rotation Metadata (Calabretta & Greisen 2002)
        self.alpha0 = float(self.wcs_params.get("CRVAL1", 0.0))
        self.delta0 = float(self.wcs_params.get("CRVAL2", 0.0))

        # Precompute Projection Meta
        ctype1 = str(self.wcs_params.get("CTYPE1", "RA---TAN")).strip().upper()
        ctype1_base = ctype1[:-4] if ctype1.endswith("-SIP") else ctype1
        self._proj_code = ctype1_base[-3:]
        self._is_tnx = self._proj_code == "TNX"
        self._is_zpx = self._proj_code == "ZPX"
        self._is_tpv = self.tpv is not None or self._proj_code == "TPV"

        self._theta0 = 90.0  # Zenithal
        if self._proj_code in ("AIT", "MOL", "HPX", "CEA", "MER", "CYP", "CAR", "SFL"):
            self._theta0 = 0.0

        # LONPOLE / LATPOLE
        self.phi_p = float(
            self.wcs_params.get("LONPOLE", 180.0 if self._theta0 == 90.0 else 0.0)
        )
        self.theta_p = float(self.wcs_params.get("LATPOLE", 90.0))

        d2r = math.pi / 180.0
        self._d2r = d2r
        self._r2d = 180.0 / math.pi
        self._sin_alpha0 = math.sin(self.alpha0 * d2r)
        self._cos_alpha0 = math.cos(self.alpha0 * d2r)
        self._sin_delta0 = math.sin(self.delta0 * d2r)
        self._cos_delta0 = math.cos(self.delta0 * d2r)
        self._east_x = -self._sin_alpha0
        self._east_y = self._cos_alpha0
        self._north_x = -self._sin_delta0 * self._cos_alpha0
        self._north_y = -self._sin_delta0 * self._sin_alpha0
        self._north_z = self._cos_delta0
        self._radial_x = self._cos_delta0 * self._cos_alpha0
        self._radial_y = self._cos_delta0 * self._sin_alpha0
        self._radial_z = self._sin_delta0

        # Eq 8 in Paper II: Calculate delta_p
        # Simplified for common cases.
        # For Zenithal, delta_p = delta0.
        # For Cylindrical, delta_p = 90.
        if self._theta0 == 90.0:
            self.delta_p = self.delta0
            self.alpha_p = self.alpha0
            self._native_type = "pole"
        else:
            self.delta_p = 90.0
            self.alpha_p = self.alpha0 + 180.0 if self.phi_p == 0.0 else self.alpha0
            self._native_type = "center"

        # Precompute Tensors
        self._ra_p_rad = self.alpha_p * d2r
        self._dec_p_rad = self.delta_p * d2r
        self._phi_p_rad = self.phi_p * d2r
        dec_p_tensor = torch.as_tensor(
            self._dec_p_rad, dtype=torch.float64, device=self.device
        )
        self._sin_dec_p = torch.sin(dec_p_tensor)
        self._cos_dec_p = torch.cos(dec_p_tensor)
        self._sin_alpha_p = math.sin(self._ra_p_rad)
        self._cos_alpha_p = math.cos(self._ra_p_rad)
        self._sin_delta_p = math.sin(self._dec_p_rad)
        self._cos_delta_p = math.cos(self._dec_p_rad)
        self._pole_east_x = -self._sin_alpha_p
        self._pole_east_y = self._cos_alpha_p
        self._pole_north_x = -self._sin_delta_p * self._cos_alpha_p
        self._pole_north_y = -self._sin_delta_p * self._sin_alpha_p
        self._pole_north_z = self._cos_delta_p
        self._pole_radial_x = self._cos_delta_p * self._cos_alpha_p
        self._pole_radial_y = self._cos_delta_p * self._sin_alpha_p
        self._pole_radial_z = self._sin_delta_p

        # Pre-calculate 3x3 rotation matrix from native to celestial
        # Columns are the celestial basis vectors in the native frame?
        # No, celestial = R * native.
        # If native basis is (e_nat, n_nat, r_nat), then V_cel = u*e_cel + v*n_cel + w*r_cel.
        # where (e_cel, n_cel, r_cel) are the basis vectors of celestial frame at CRVAL.

        # Basis vectors at alpha_p, delta_p
        ap_rad = self._ra_p_rad
        dp_rad = self._dec_p_rad
        sin_ap, cos_ap = math.sin(ap_rad), math.cos(ap_rad)
        sin_dp, cos_dp = math.sin(dp_rad), math.cos(dp_rad)

        # Celestial basis at (alpha_p, delta_p)
        # ex, ey, ez
        ex, ey, ez = -sin_ap, cos_ap, 0.0
        # nx, ny, nz
        nx, ny, nz = -sin_dp * cos_ap, -sin_dp * sin_ap, cos_dp
        # rx, ry, rz
        rx, ry, rz = cos_dp * cos_ap, cos_dp * sin_ap, sin_dp

        # The native basis at phi_p=180 matches the FITS standard rotation.
        # R_cel = R_z(-alpha_p) R_y(delta_p - 90) R_z(180 - phi_p) ?
        # Let's use the explicit basis construction from _spherical_rotation.
        # For 'pole' type: u->ex, v->nx, w->rx
        self._rot_matrix = torch.tensor(
            [[ex, nx, rx], [ey, ny, ry], [ez, nz, rz]],
            dtype=torch.float64,
            device=self.device,
        )
        self._rot_matrix_inv = self._rot_matrix.T  # Rotation matrix is orthogonal

        # Precompute center-native fast path
        self._use_center_fast = (
            self._native_type == "center"
            and abs(self.crval[0].item()) < 1e-9
            and abs(self.crval[1].item()) < 1e-9
        )

        # Precompute CEA/MER constants
        pv2_1 = self.wcs_params.get("PV2_1", 1.0)
        if pv2_1 == 0.0:
            pv2_1 = 1.0
        self._cea_eta_scale = 57.29577951308232 / pv2_1
        self._sqrt2 = math.sqrt(2.0)
        self._mol_abs_eta_max = self._sqrt2 * self._r2d
        self._hpx_h = float(
            self.wcs_params.get("PV1_1", self.wcs_params.get("PV2_1", 4.0))
        )
        self._hpx_k = float(
            self.wcs_params.get("PV1_2", self.wcs_params.get("PV2_2", 3.0))
        )
        if abs(self._hpx_h) < 1e-12:
            self._hpx_h = 4.0
        if abs(self._hpx_k) < 1e-12:
            self._hpx_k = 3.0
        self._hpx_eta_scale = 90.0 * (self._hpx_k / self._hpx_h)
        if abs(self._hpx_eta_scale) < 1e-12:
            self._hpx_eta_scale = 67.5
        self._hpx_inv_eta_scale = 1.0 / self._hpx_eta_scale
        self._hpx_eta_boundary = self._hpx_eta_scale * (2.0 / 3.0)
        self._hpx_eta_pole = 90.0
        self._hpx_polar_denom = self._hpx_eta_pole - self._hpx_eta_boundary
        self._hpx_inv_polar_denom = 1.0 / self._hpx_polar_denom

        # Precompute fused arguments to avoid overhead
        self._fused_proj_code = "TAN" if self._is_tpv else self._proj_code
        self._fused_pv2_1 = self.wcs_params.get("PV2_1", 1.0)
        self._fused_tpv_idx1 = (
            self.tpv.idx1
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c1 = (
            self.tpv.c1
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )
        self._fused_tpv_idx2 = (
            self.tpv.idx2
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c2 = (
            self.tpv.c2
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )

        # Pre-tensorize PV parameters to avoid dict lookups in compiled hot paths.
        pv1_data = [self.wcs_params.get(k, 0.0) for k in PV1_KEYS]
        pv2_data = [self.wcs_params.get(k, 0.0) for k in PV2_KEYS]
        self._pv1_tensor = torch.tensor(
            pv1_data, dtype=torch.float64, device=self.device
        )
        self._pv2_tensor = torch.tensor(
            pv2_data, dtype=torch.float64, device=self.device
        )
        zpn_all = self._pv2_tensor[3:31].detach().cpu()
        zpn_valid = torch.nonzero(zpn_all.abs() > 1e-15, as_tuple=False)
        if zpn_valid.numel() > 0:
            zpn_last = int(zpn_valid[-1].item())
            self._zpn_coeffs_cpu = zpn_all[: zpn_last + 1].contiguous()
        else:
            # Default ZPN behavior (no explicit polynomial terms): r = w.
            # Keep a linear coefficient vector so C++ direct ZPN path remains enabled.
            if self._proj_code == "ZPN":
                self._zpn_coeffs_cpu = torch.tensor([0.0, 1.0], dtype=torch.float64)
            else:
                self._zpn_coeffs_cpu = torch.empty((0,), dtype=torch.float64)

        # Precompute projection flags for compile-friendly dispatch.
        self._is_tan = self._proj_code == "TAN"
        self._is_sin = self._proj_code == "SIN"
        self._is_arc = self._proj_code == "ARC"
        self._is_zea = self._proj_code == "ZEA"
        self._is_stg = self._proj_code == "STG"
        self._is_zpn = self._proj_code == "ZPN"
        self._is_zpn_linear = False
        if self._is_zpn:
            if self._zpn_coeffs_cpu.numel() == 0:
                self._is_zpn_linear = True
            elif self._zpn_coeffs_cpu.numel() == 2:
                c0 = float(self._zpn_coeffs_cpu[0].item())
                c1 = float(self._zpn_coeffs_cpu[1].item())
                self._is_zpn_linear = abs(c0) <= 1e-15 and abs(c1 - 1.0) <= 1e-15
        self._is_ait = self._proj_code == "AIT"
        self._is_mol = self._proj_code == "MOL"
        self._is_hpx = self._proj_code == "HPX"
        self._is_cea = self._proj_code == "CEA"
        self._is_mer = self._proj_code == "MER"
        self._is_car = self._proj_code == "CAR"
        self._is_sfl = self._proj_code == "SFL"

        # Detect if fast TAN is possible
        self._use_fast_tan = (
            self._proj_code == "TAN"
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.sip is None
            and self.alpha0 == self.alpha_p
            and self.delta0 == self.delta_p
        )
        self._is_spatial = self._proj_code in (
            self.ZENITHAL_CODES
            + self.CYLINDRICAL_CODES
            + self.ALLSKY_CODES
            + self.SPECIAL_CODES
        )
        self._use_center_equator_fast = (
            self._native_type == "center"
            and abs(self.delta0) < 1e-12
            and abs((self.phi_p % 360.0)) < 1e-12
        )
        self._cpp_zenithal_project = (
            getattr(_cpp, "wcs_zenithal_project", None) if _cpp is not None else None
        )
        self._cpp_zpn_project = (
            getattr(_cpp, "wcs_zpn_project", None) if _cpp is not None else None
        )
        self._cpp_zenithal_deproject = (
            getattr(_cpp, "wcs_zenithal_deproject", None) if _cpp is not None else None
        )
        self._cpp_cylindrical_project = (
            getattr(_cpp, "wcs_cylindrical_project", None) if _cpp is not None else None
        )
        self._cpp_cylindrical_deproject = (
            getattr(_cpp, "wcs_cylindrical_deproject", None)
            if _cpp is not None
            else None
        )
        self._cpp_center_pixel_to_world = (
            getattr(_cpp, "wcs_center_pixel_to_world", None)
            if _cpp is not None
            else None
        )
        self._cpp_center_world_to_pixel = (
            getattr(_cpp, "wcs_center_world_to_pixel", None)
            if _cpp is not None
            else None
        )
        self._cpp_ait_project = (
            getattr(_cpp, "wcs_ait_project", None) if _cpp is not None else None
        )
        self._cpp_ait_deproject = (
            getattr(_cpp, "wcs_ait_deproject", None) if _cpp is not None else None
        )
        self._cpp_mol_project = (
            getattr(_cpp, "wcs_mol_project", None) if _cpp is not None else None
        )
        self._cpp_mol_deproject = (
            getattr(_cpp, "wcs_mol_deproject", None) if _cpp is not None else None
        )
        self._cpp_hpx_project = (
            getattr(_cpp, "wcs_hpx_project", None) if _cpp is not None else None
        )
        self._cpp_hpx_deproject = (
            getattr(_cpp, "wcs_hpx_deproject", None) if _cpp is not None else None
        )
        self._cpp_inverse_spherical_rotation_pole = (
            getattr(_cpp, "wcs_inverse_spherical_rotation_pole", None)
            if _cpp is not None
            else None
        )
        self._cpp_spherical_rotation_pole = (
            getattr(_cpp, "wcs_spherical_rotation_pole", None)
            if _cpp is not None
            else None
        )
        self._cpp_tan_intermediate_from_radec = (
            getattr(_cpp, "wcs_tan_intermediate_from_radec", None)
            if _cpp is not None
            else None
        )
        self._cpp_direct_static_ok = (
            self.sip is None
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
        )
        self._cpp_p2w_direct_kind = ""
        if self._cpp_direct_static_ok:
            if self._is_ait and self._cpp_ait_project is not None:
                self._cpp_p2w_direct_kind = "ait"
            elif self._is_mol and self._cpp_mol_project is not None:
                self._cpp_p2w_direct_kind = "mol"
            elif self._is_hpx and self._cpp_hpx_project is not None:
                self._cpp_p2w_direct_kind = "hpx"
            elif self._cpp_zenithal_project is not None and (
                self._is_tan
                or self._is_sin
                or self._is_arc
                or self._is_zea
                or self._is_stg
            ):
                self._cpp_p2w_direct_kind = "zenithal"
            elif (
                self._is_zpn
                and not self._is_zpn_linear
                and self._cpp_zpn_project is not None
                and self._zpn_coeffs_cpu.numel() > 0
            ):
                self._cpp_p2w_direct_kind = "zpn"
        self._cpp_w2p_direct_kind = ""
        if self._cpp_direct_static_ok:
            if self._is_ait and self._cpp_ait_deproject is not None:
                self._cpp_w2p_direct_kind = "ait"
            elif self._is_hpx and self._cpp_hpx_deproject is not None:
                self._cpp_w2p_direct_kind = "hpx"
            elif self._cpp_zenithal_deproject is not None and (
                self._is_tan
                or self._is_sin
                or self._is_arc
                or self._is_zea
                or self._is_stg
            ):
                self._cpp_w2p_direct_kind = "zenithal"
        self._has_cpp_direct_p2w = bool(self._cpp_p2w_direct_kind)
        self._has_cpp_direct_w2p = bool(self._cpp_w2p_direct_kind)
        self._can_use_fused_p2w_static = (
            self._enable_fused_wcs
            and _cpp is not None
            and hasattr(_cpp, "wcs_pixel_to_world_fused_cpu")
            and self.sip is None
            and not self._is_tnx
            and not self._is_zpx
            and (
                self._proj_code
                in ("TAN", "SIN", "CEA", "AIT", "HPX", "CAR", "SFL", "MOL", "TPV")
                or (self._is_tpv and self._proj_code == "TAN")
            )
        )
        self._can_use_fused_w2p_static = (
            self._enable_fused_wcs
            and _cpp is not None
            and hasattr(_cpp, "wcs_world_to_pixel_fused_cpu")
            and self.sip is None
            and not self._is_tnx
            and not self._is_zpx
            and (not self._is_tpv or not self.tpv._invert_trace_enabled)
            and (
                self._proj_code
                in ("TAN", "SIN", "CEA", "AIT", "HPX", "CAR", "SFL", "MOL", "TPV")
                or (self._is_tpv and self._proj_code == "TAN")
            )
        )
        self._disable_cpp_pixel_to_world = False
        self._disable_cpp_world_to_pixel = False
        self._disable_fused_pixel_to_world = False
        self._disable_fused_world_to_pixel = False
        self._cpp_direct_p2w_verified = False
        self._cpp_direct_w2p_verified = False
        self._fused_p2w_verified = False
        self._fused_w2p_verified = False
        self._setup_projection_dispatch()

    def _setup_projection_dispatch(self) -> None:
        """Bind pixel->world projection dispatch once per WCS instance."""
        if self._is_zpx:
            self._project_native_fn = None
            self._project_native_kind = "zpx"
            self._project_native_type = "pole"
            return
        if self._is_tnx:
            self._project_native_fn = None
            self._project_native_kind = "tnx"
            self._project_native_type = "pole"
            return
        if self._is_tpv:
            self._project_native_fn = project_zenithal
            self._project_native_kind = "tpv_tan"
            self._project_native_type = "pole"
            return
        if self._proj_code in self.ZENITHAL_CODES:
            self._project_native_fn = project_zenithal
            self._project_native_kind = "zenithal"
            self._project_native_type = "pole"
            return
        if self._proj_code in self.CYLINDRICAL_CODES:
            self._project_native_fn = project_cylindrical
            self._project_native_kind = "cylindrical"
            self._project_native_type = "center"
            return
        if self._proj_code in self.ALLSKY_CODES:
            self._project_native_fn = project_allsky
            self._project_native_kind = "allsky"
            self._project_native_type = "center"
            return
        self._project_native_fn = None
        self._project_native_kind = "linear"
        self._project_native_type = "pole"

    def _project_native_coords(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Projection dispatch hot path with pre-bound family metadata."""
        kind = self._project_native_kind
        if kind == "zpx":
            return project_zpx(xi, eta, self.wcs_params, self.wat_data)
        if kind == "tnx":
            xi_t, eta_t = project_tnx(xi, eta, self.wcs_params, self.wat_data)
            return project_zenithal(xi_t, eta_t, "TAN")
        if kind == "tpv_tan":
            return project_zenithal(xi, eta, "TAN")
        fn = self._project_native_fn
        if fn is None:
            return xi, eta
        # Use pre-tensorized PV parameters for modern projections
        return fn(xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor)

    def _apply_cd_2d(self, rel_x: Tensor, rel_y: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cached 2x2 CD transform with diagonal fast path."""
        if self._cd_is_diag:
            return self._cd00 * rel_x, self._cd11 * rel_y
        xi = self._cd00 * rel_x + self._cd01 * rel_y
        eta = self._cd10 * rel_x + self._cd11 * rel_y
        return xi, eta

    def _apply_cd_inv_2d(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cached 2x2 inverse CD transform using scalar coefficients."""
        rel0 = self._cdi00 * xi + self._cdi01 * eta
        rel1 = self._cdi10 * xi + self._cdi11 * eta
        return rel0 + self._crpix0, rel1 + self._crpix1

    def _alloc_pix_internal_like(self, world: Tensor) -> Tensor:
        """Allocate output for analytic world->pixel without zeroing 2D hot path."""
        pix_internal = torch.empty_like(world)
        if world.shape[1] > 2:
            pix_internal[:, 2:] = 0
        return pix_internal

    @staticmethod
    def _contiguous_if_needed(t: Tensor) -> Tensor:
        """Avoid unnecessary contiguous copies on already contiguous benchmark inputs."""
        return t if t.is_contiguous() else t.contiguous()

    def _cpp_project_direct_p2w(
        self, xi: Tensor, eta: Tensor, cpp_kind: str
    ) -> Tuple[Tensor, Tensor]:
        """Direct C++ projection dispatch used by pixel->world fast path."""
        if cpp_kind == "ait":
            return self._cpp_ait_project(xi, eta)
        if cpp_kind == "mol":
            return self._cpp_mol_project(xi, eta)
        if cpp_kind == "hpx":
            return self._cpp_hpx_project(xi, eta, self._hpx_h, self._hpx_k)
        if cpp_kind == "zenithal":
            phi, theta = self._cpp_zenithal_project(xi, eta, self._proj_code)
            return -phi, theta
        if cpp_kind == "zpn":
            return self._cpp_zpn_project(xi, eta, self._zpn_coeffs_cpu)
        raise RuntimeError("unsupported_cpp_direct_projection")

    def _cpp_deproject_direct_w2p(
        self, phi_w: Tensor, theta: Tensor, cpp_kind: str
    ) -> Tuple[Tensor, Tensor]:
        """Direct C++ deprojection dispatch used by world->pixel fast path."""
        if cpp_kind == "ait":
            return self._cpp_ait_deproject(phi_w, theta)
        if cpp_kind == "hpx":
            return self._cpp_hpx_deproject(phi_w, theta, self._hpx_h, self._hpx_k)
        if cpp_kind == "zenithal":
            return self._cpp_zenithal_deproject(-phi_w, theta, self._proj_code)
        raise RuntimeError("unsupported_cpp_direct_projection")

    def _local_uvw_from_radec(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert celestial RA/Dec (deg) to local east/north/radial components."""
        ra_rad = ra * self._d2r
        dec_rad = dec * self._d2r

        sin_ra, cos_ra = _sincos(ra_rad)
        sin_dec, cos_dec = _sincos(dec_rad)
        X = cos_dec * cos_ra
        Y = cos_dec * sin_ra
        Z = sin_dec

        u = X * self._east_x + Y * self._east_y
        v = X * self._north_x + Y * self._north_y + Z * self._north_z
        w = X * self._radial_x + Y * self._radial_y + Z * self._radial_z
        return u, v, w

    def _tan_intermediate_from_radec(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Exact TAN intermediate coordinates (xi, eta) from RA/Dec via cached local basis."""
        if (
            self._cpp_tan_intermediate_from_radec is not None
            and ra.device.type == "cpu"
            and dec.device.type == "cpu"
            and ra.dtype == torch.float64
            and dec.dtype == torch.float64
        ):
            ra_cpp = self._contiguous_if_needed(ra)
            dec_cpp = self._contiguous_if_needed(dec)
            return self._cpp_tan_intermediate_from_radec(
                ra_cpp,
                dec_cpp,
                self._east_x,
                self._east_y,
                self._north_x,
                self._north_y,
                self._north_z,
                self._radial_x,
                self._radial_y,
                self._radial_z,
            )
        u, v, w = self._local_uvw_from_radec(ra, dec)
        scale = self._r2d / torch.clamp(w, min=1e-12)
        return u * scale, v * scale

    def _spherical_rotation_finite_only(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Rotate only finite native coordinates; keep NaNs from projection domain masks."""
        valid = torch.isfinite(phi) & torch.isfinite(theta)
        if bool(valid.all()):
            return self._spherical_rotation_optimized(phi, theta, self._native_type)
        if not bool(valid.any()):
            nan = torch.full_like(phi, float("nan"))
            return nan, nan.clone()
        ra = torch.full_like(phi, float("nan"))
        dec = torch.full_like(theta, float("nan"))
        ra_v, dec_v = self._spherical_rotation_optimized(
            phi[valid], theta[valid], self._native_type
        )
        ra[valid] = ra_v
        dec[valid] = dec_v
        return ra, dec

    def _pixel_to_world_2d_core(
        self, x: Tensor, y: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial pixel-to-world core.
        """
        # 1. Coordinate frame relative to CRPIX
        rel_x = x + (1.0 - origin) - self._crpix0
        rel_y = y + (1.0 - origin) - self._crpix1

        # 2. Distortion & Linear
        if self.sip is not None:
            rel_x, rel_y = self.sip.distort(rel_x, rel_y)

        # Apply CD matrix
        xi, eta = self._apply_cd_2d(rel_x, rel_y)

        # Common center-equator specializations avoid full spherical rotation.
        if self._use_center_equator_fast and self._is_car:
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, eta
        if self._use_center_equator_fast and self._is_sfl:
            dec = eta
            cos_dec = torch.cos(dec * self._d2r)
            cos_safe = torch.sign(cos_dec) * torch.clamp(torch.abs(cos_dec), min=1e-12)
            ra = _wrap_lon360((xi / cos_safe) + self.alpha0)
            return ra, dec
        if self._use_center_equator_fast and self._is_cea:
            theta = (
                torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                * self._r2d
            )
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, theta
        if self._use_center_equator_fast and self._is_mer:
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, theta

        # 3. TPV / Special distortions
        if self._is_tpv:
            xi, eta = self.tpv.distort(xi, eta)
        elif self._is_tnx:
            xi, eta = project_tnx(xi, eta, self.wcs_params, self.wat_data)
            xi, eta = project_zenithal(xi, eta, "TAN")
        elif self._is_zpx:
            xi, eta = project_zpx(xi, eta, self.wcs_params, self.wat_data)

        # 4. Projection
        if (
            self._is_tan
            or self._is_tpv
            or self._is_sin
            or self._is_arc
            or self._is_zea
            or self._is_stg
        ):
            r = torch.hypot(xi, eta)
            phi = torch.atan2(-xi, -eta) * self._r2d
            if self._is_tan or self._is_tpv:
                theta = torch.atan2(torch.ones_like(r), r * self._d2r) * self._r2d
            elif self._is_sin:
                theta = torch.acos(torch.clamp(r * self._d2r, -1.0, 1.0)) * self._r2d
            elif self._is_arc:
                theta = 90.0 - r
            elif self._is_zea:
                theta = (
                    90.0
                    - 2.0
                    * torch.asin(torch.clamp((r * self._d2r) * 0.5, -1.0, 1.0))
                    * self._r2d
                )
            else:
                theta = 90.0 - 2.0 * torch.atan((r * self._d2r) * 0.5) * self._r2d
        elif self._is_ait:
            phi, theta = self._native_from_ait_fast(xi, eta)
        elif self._is_mol:
            phi, theta = project_allsky(xi, eta, "MOL")
        elif self._is_hpx:
            abs_eta = torch.abs(eta)
            mask_eq = abs_eta <= self._hpx_eta_boundary
            sigma = (self._hpx_eta_pole - abs_eta) * self._hpx_inv_polar_denom
            s_theta = torch.where(
                mask_eq,
                eta * self._hpx_inv_eta_scale,
                torch.sign(eta) * (1.0 - (sigma * sigma) * (1.0 / 3.0)),
            )
            theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * self._r2d
            xc = torch.round((xi - 45.0) / 90.0) * 90.0 + 45.0
            phi = torch.where(
                mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9)
            )
            invalid = abs_eta > self._hpx_eta_pole
            if bool(invalid.any()):
                nan = torch.full_like(phi, float("nan"))
                phi = torch.where(invalid, nan, phi)
                theta = torch.where(invalid, nan, theta)
        elif self._is_cea:
            phi = xi
            theta = (
                torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                * self._r2d
            )
        elif self._is_mer:
            phi = xi
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
        elif self._is_car:
            # Inline trivial CAR projection: phi = xi, theta = eta
            phi, theta = xi, eta
        elif self._is_sfl:
            theta = eta
            phi = xi / torch.clamp(torch.cos(theta * self._d2r), min=1e-12)
        elif self._is_zpn:
            if self._is_zpn_linear:
                r = torch.hypot(xi, eta)
                phi = torch.atan2(-xi, -eta) * self._r2d
                theta = 90.0 - r
            else:
                phi, theta = project_zenithal(xi, eta, "ZPN", None, self._pv2_tensor)
        else:
            # Linear fallback
            phi, theta = xi, eta

        if self._use_center_equator_fast:
            ra = _wrap_lon360(phi + self.alpha0)
            return ra, theta

        # 5. Celestial Rotation
        ra, dec = self._spherical_rotation_optimized(phi, theta, self._native_type)
        return ra, dec

    def _pixel_to_world_2d_fast(
        self, x: Tensor, y: Tensor, origin: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Fast path for common 2D separate-argument pixel->world calls."""
        if x.shape != y.shape:
            raise RuntimeError("pixel coordinate shapes must match")

        device_ok = x.device.type == "cpu" and y.device.type == "cpu"
        dtype_ok = x.dtype == torch.float64 and y.dtype == torch.float64

        if (
            self._use_center_equator_fast
            and self._cpp_center_pixel_to_world is not None
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
            and self._is_ait
        ):
            return self._cpp_center_pixel_to_world(
                x,
                y,
                float(origin),
                self._crpix0,
                self._crpix1,
                self._cd00,
                self._cd01,
                self._cd10,
                self._cd11,
                self._proj_code,
                self.alpha0,
                self._cea_eta_scale,
                self._hpx_h,
                self._hpx_k,
            )

        if self._use_center_equator_fast and (
            self._is_car or self._is_sfl or self._is_cea or self._is_mer
        ):
            rel_x = x + (1.0 - origin) - self._crpix0
            rel_y = y + (1.0 - origin) - self._crpix1
            xi, eta = self._apply_cd_2d(rel_x, rel_y)
            if self._is_car:
                return _wrap_lon360(xi + self.alpha0), eta
            if self._is_sfl:
                dec = eta
                cos_dec = torch.cos(dec * self._d2r)
                cos_safe = torch.sign(cos_dec) * torch.clamp(
                    torch.abs(cos_dec), min=1e-12
                )
                return _wrap_lon360((xi / cos_safe) + self.alpha0), dec
            if self._is_cea:
                theta = (
                    torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                    * self._r2d
                )
                return _wrap_lon360(xi + self.alpha0), theta
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
            return _wrap_lon360(xi + self.alpha0), theta

        cpp_kind = self._cpp_p2w_direct_kind
        use_cpp_direct = (
            self._has_cpp_direct_p2w
            and not self._disable_cpp_pixel_to_world
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
        )
        if use_cpp_direct:
            rel_x = x + (1.0 - origin) - self._crpix0
            rel_y = y + (1.0 - origin) - self._crpix1
            xi, eta = self._apply_cd_2d(rel_x, rel_y)
            if self._cpp_direct_p2w_verified:
                phi, theta = self._cpp_project_direct_p2w(xi, eta, cpp_kind)
            else:
                try:
                    phi, theta = self._cpp_project_direct_p2w(xi, eta, cpp_kind)
                    self._cpp_direct_p2w_verified = True
                except (TypeError, RuntimeError):
                    self._disable_cpp_pixel_to_world = True
                else:
                    pass
            if not self._disable_cpp_pixel_to_world:
                if self._use_center_equator_fast:
                    return _wrap_lon360(phi + self.alpha0), theta
                if (
                    self._native_type == "pole"
                    and self._cpp_spherical_rotation_pole is not None
                ):
                    return self._cpp_spherical_rotation_pole(
                        phi,
                        theta,
                        self._phi_p_rad,
                        self._pole_east_x,
                        self._pole_east_y,
                        self._pole_north_x,
                        self._pole_north_y,
                        self._pole_north_z,
                        self._pole_radial_x,
                        self._pole_radial_y,
                        self._pole_radial_z,
                    )
                return self._spherical_rotation_optimized(phi, theta, self._native_type)

        # Environment-controlled fused C++ path. Default is False because vectorized
        # PyTorch + torch.compile is typically 2x-4x faster than the scalar C++ loop.
        use_fused = (
            self._can_use_fused_p2w_static
            and not self._disable_fused_pixel_to_world
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
        )
        if use_fused:
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
            ra = torch.empty_like(x_flat)
            dec = torch.empty_like(y_flat)
            if self._native_type == "pole":
                ex, ey = self._pole_east_x, self._pole_east_y
                nx, ny, nz = self._pole_north_x, self._pole_north_y, self._pole_north_z
                rx, ry, rz = (
                    self._pole_radial_x,
                    self._pole_radial_y,
                    self._pole_radial_z,
                )
            else:
                ex, ey = self._east_x, self._east_y
                nx, ny, nz = self._north_x, self._north_y, self._north_z
                rx, ry, rz = self._radial_x, self._radial_y, self._radial_z

            try:
                _cpp.wcs_pixel_to_world_fused_cpu(
                    (x_flat + (1.0 - origin)).contiguous(),
                    (y_flat + (1.0 - origin)).contiguous(),
                    ra,
                    dec,
                    self._crpix0,
                    self._crpix1,
                    self._cd00,
                    self._cd01,
                    self._cd10,
                    self._cd11,
                    self._fused_proj_code,
                    self._fused_pv2_1,
                    self._hpx_h,
                    self._hpx_k,
                    ex,
                    ey,
                    nx,
                    ny,
                    nz,
                    rx,
                    ry,
                    rz,
                    self._ra_p_rad,
                    self._dec_p_rad,
                    self._phi_p_rad,
                    self._native_type,
                    self.alpha0,
                    self.delta0,
                    self._is_tpv,
                    self._fused_tpv_idx1,
                    self._fused_tpv_c1,
                    self._fused_tpv_idx2,
                    self._fused_tpv_c2,
                )
            except TypeError:
                # C++ extension can be stale relative to Python bindings in editable
                # workflows. Disable this fused path and fall back safely.
                self._disable_fused_pixel_to_world = True
                return self._pixel_to_world_2d_core(x, y, float(origin))
            return ra.reshape(x.shape), dec.reshape(x.shape)

        core_fn = self._compiled_pixel_to_world_2d_fn or self._pixel_to_world_2d_core
        return core_fn(x, y, float(origin))

    def _pixel_to_world_cpp(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """C++ fast path for projections."""
        device_ok = xi.device.type == "cpu" and eta.device.type == "cpu"
        dtype_ok = xi.dtype == torch.float64 and eta.dtype == torch.float64
        use_cpp_inputs = device_ok and dtype_ok
        if use_cpp_inputs:
            xi_cpp = self._contiguous_if_needed(xi)
            eta_cpp = self._contiguous_if_needed(eta)
        else:
            xi_cpp = xi
            eta_cpp = eta

        if self._proj_code in self.ZENITHAL_CODES:
            if self._cpp_zenithal_project is not None and device_ok and dtype_ok:
                phi, theta = self._cpp_zenithal_project(
                    xi_cpp, eta_cpp, self._proj_code
                )
                # Match Python zenithal convention (phi sign).
                phi = -phi
            else:
                phi, theta = project_zenithal(
                    xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            return self._spherical_rotation_optimized(phi, theta, self._native_type)

        if self._proj_code in self.CYLINDRICAL_CODES:
            lambda_param = self.wcs_params.get("PV2_1", 1.0)
            if self._cpp_cylindrical_project is not None and device_ok and dtype_ok:
                phi, theta = self._cpp_cylindrical_project(
                    xi_cpp, eta_cpp, self._proj_code, lambda_param
                )
            else:
                phi, theta = project_cylindrical(
                    xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            return self._spherical_rotation_optimized(phi, theta, self._native_type)

        if self._proj_code == "AIT":
            if self._cpp_ait_project is not None and device_ok and dtype_ok:
                phi, theta = self._cpp_ait_project(xi_cpp, eta_cpp)
            else:
                phi, theta = project_allsky(
                    xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            return self._spherical_rotation_optimized(phi, theta, self._native_type)

        if self._proj_code == "MOL":
            if self._cpp_mol_project is not None and device_ok and dtype_ok:
                # Exact MOL domain prefilter: validity depends only on |eta| <= sqrt(2) rad.
                # When benchmarking random image coordinates, most points are outside the
                # projection domain; processing only the valid subset avoids unnecessary
                # projection work while preserving exact results.
                valid_eta = torch.abs(eta_cpp) <= self._mol_abs_eta_max
                if bool(valid_eta.all()):
                    phi, theta = self._cpp_mol_project(xi_cpp, eta_cpp)
                elif not bool(valid_eta.any()):
                    nan = torch.full_like(xi, float("nan"))
                    return nan, nan.clone()
                else:
                    phi = torch.full_like(xi, float("nan"))
                    theta = torch.full_like(eta, float("nan"))
                    phi_v, theta_v = self._cpp_mol_project(
                        xi_cpp[valid_eta], eta_cpp[valid_eta]
                    )
                    ra_v, dec_v = self._spherical_rotation_finite_only(phi_v, theta_v)
                    ra = torch.full_like(xi, float("nan"))
                    dec = torch.full_like(eta, float("nan"))
                    ra[valid_eta] = ra_v
                    dec[valid_eta] = dec_v
                    return ra, dec
            else:
                phi, theta = project_allsky(
                    xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            return self._spherical_rotation_finite_only(phi, theta)

        if self._proj_code == "HPX":
            H = self._hpx_h
            K = self._hpx_k
            if self._cpp_hpx_project is not None and device_ok and dtype_ok:
                # Exact HPX eta-domain prefilter; xi bounds still depend on eta and are
                # validated inside the projection kernel.
                valid_eta = torch.abs(eta_cpp) <= self._hpx_eta_pole
                if bool(valid_eta.all()):
                    phi, theta = self._cpp_hpx_project(xi_cpp, eta_cpp, H, K)
                elif not bool(valid_eta.any()):
                    nan = torch.full_like(xi, float("nan"))
                    return nan, nan.clone()
                else:
                    phi = torch.full_like(xi, float("nan"))
                    theta = torch.full_like(eta, float("nan"))
                    phi_v, theta_v = self._cpp_hpx_project(
                        xi_cpp[valid_eta], eta_cpp[valid_eta], H, K
                    )
                    ra_v, dec_v = self._spherical_rotation_finite_only(phi_v, theta_v)
                    ra = torch.full_like(xi, float("nan"))
                    dec = torch.full_like(eta, float("nan"))
                    ra[valid_eta] = ra_v
                    dec[valid_eta] = dec_v
                    return ra, dec
            else:
                phi, theta = project_allsky(
                    xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            return self._spherical_rotation_finite_only(phi, theta)

        phi, theta = self._project_native_coords(xi, eta)
        return self._spherical_rotation_optimized(phi, theta, self._native_type)

    def _native_from_ait_fast(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Fast AIT inverse projection: intermediate (deg) -> native (deg)."""
        X = xi * self._d2r
        Y = eta * self._d2r

        r2 = (X * 0.25) ** 2 + (Y * 0.5) ** 2
        valid = r2 <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))

        phi = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0) * self._r2d
        theta = torch.asin(torch.clamp(z * Y, -1.0, 1.0)) * self._r2d

        if valid.all():
            return phi, theta
        phi = phi.clone()
        theta = theta.clone()
        phi.masked_fill_(~valid, float("nan"))
        theta.masked_fill_(~valid, float("nan"))
        return phi, theta

    def _native_from_mol_fast(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Fast MOL inverse projection: intermediate (deg) -> native (deg)."""
        valid = torch.abs(eta) <= self._mol_abs_eta_max
        if bool(valid.all()):
            xv = xi * self._d2r
            yv = eta * self._d2r
            sin_gamma = torch.clamp(yv / self._sqrt2, -1.0, 1.0)
            gamma = torch.asin(sin_gamma)
            cos_gamma = torch.cos(gamma)
            t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / math.pi
            theta = torch.asin(torch.clamp(t_val, -1.0, 1.0)) * self._r2d
            denom = 2.0 * self._sqrt2 * cos_gamma
            phi = torch.zeros_like(xi)
            good = torch.abs(denom) >= 1e-12
            if bool(good.any()):
                phi[good] = (math.pi * xv[good] / denom[good]) * self._r2d
            return phi, theta

        phi = torch.full_like(xi, float("nan"))
        theta = torch.full_like(eta, float("nan"))
        if bool(valid.any()):
            xv = (xi[valid]) * self._d2r
            yv = (eta[valid]) * self._d2r
            sin_gamma = torch.clamp(yv / self._sqrt2, -1.0, 1.0)
            gamma = torch.asin(sin_gamma)
            cos_gamma = torch.cos(gamma)
            t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / math.pi
            theta[valid] = torch.asin(torch.clamp(t_val, -1.0, 1.0)) * self._r2d
            denom = 2.0 * self._sqrt2 * cos_gamma
            phi_v = torch.zeros_like(xv)
            good = torch.abs(denom) >= 1e-12
            if bool(good.any()):
                phi_v[good] = (math.pi * xv[good] / denom[good]) * self._r2d
            phi[valid] = phi_v
        return phi, theta

    def _native_from_hpx_fast(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Fast HPX inverse projection: intermediate (deg) -> native (deg)."""
        phi = torch.empty_like(xi)
        theta = torch.empty_like(eta)

        abs_eta = torch.abs(eta)
        invalid = abs_eta > self._hpx_eta_pole

        mask_eq = (~invalid) & (abs_eta <= self._hpx_eta_boundary)
        if mask_eq.any():
            s_theta_eq = torch.clamp(eta[mask_eq] * self._hpx_inv_eta_scale, -1.0, 1.0)
            theta[mask_eq] = torch.asin(s_theta_eq) * self._r2d
            phi[mask_eq] = xi[mask_eq]

        mask_pol = (~invalid) & (~mask_eq)
        if mask_pol.any():
            xi_p = xi[mask_pol]
            eta_p = eta[mask_pol]
            abs_eta_p = torch.abs(eta_p)

            sigma = torch.clamp(
                (self._hpx_eta_pole - abs_eta_p) * self._hpx_inv_polar_denom, min=0.0
            )
            s_theta_pol = torch.sign(eta_p) * (1.0 - (sigma * sigma) / 3.0)
            theta[mask_pol] = (
                torch.asin(torch.clamp(s_theta_pol, -1.0, 1.0)) * self._r2d
            )

            xc = torch.round((xi_p - 45.0) / 90.0) * 90.0 + 45.0
            dx = xi_p - xc
            sigma_safe = torch.where(
                torch.abs(sigma) < 1e-12, torch.ones_like(sigma), sigma
            )
            phi[mask_pol] = xc + dx / sigma_safe

            invalid_x = torch.abs(dx) > (45.0 * sigma + 1e-8)
            if invalid_x.any():
                invalid_pol = torch.zeros_like(mask_pol)
                invalid_pol[mask_pol] = invalid_x
                invalid = invalid | invalid_pol

        if invalid.any():
            phi[invalid] = float("nan")
            theta[invalid] = float("nan")
        return phi, theta

    def _intermediate_from_ait_fast(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Fast AIT forward projection: native (deg) -> intermediate (deg)."""
        phi_wrapped = _wrap_lon180_checked(phi)
        phi_rad = phi_wrapped * self._d2r
        theta_rad = theta * self._d2r
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        half_phi = phi_rad * 0.5
        denom = torch.sqrt(
            torch.clamp(0.5 * (1.0 + cos_theta * torch.cos(half_phi)), min=1e-12)
        )
        xi = 2.0 * cos_theta * torch.sin(half_phi) / denom * self._r2d
        eta = sin_theta / denom * self._r2d
        return xi, eta

    def _intermediate_from_mol_fast(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Fast MOL forward projection: native (deg) -> intermediate (deg)."""
        phi_wrapped = _wrap_lon180_checked(phi)
        phi_rad = phi_wrapped * self._d2r
        theta_rad = theta * self._d2r
        target = math.pi * torch.sin(theta_rad)
        gamma = theta_rad.clone()
        for _ in range(5):
            sin_2g = torch.sin(2.0 * gamma)
            cos_2g = torch.cos(2.0 * gamma)
            res = 2.0 * gamma + sin_2g - target
            if torch.all(torch.abs(res) < 1e-12):
                break
            gamma = gamma - res / (2.0 + 2.0 * cos_2g + 1e-12)
        xi = 2.0 * self._sqrt2 * phi_rad * torch.cos(gamma) / math.pi * self._r2d
        eta = self._sqrt2 * torch.sin(gamma) * self._r2d
        return xi, eta

    def _intermediate_from_hpx_fast(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Fast HPX forward projection: native (deg) -> intermediate (deg)."""
        phi_w = _wrap_lon180_checked(phi)
        s_theta = torch.sin(theta * self._d2r)
        abs_s = torch.abs(s_theta)
        mask_eq = abs_s <= (2.0 / 3.0)
        sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
        eta_pol = torch.sign(s_theta) * (
            self._hpx_eta_pole - (self._hpx_eta_pole - self._hpx_eta_boundary) * sigma
        )
        xc = torch.round((phi_w - 45.0) / 90.0) * 90.0 + 45.0
        xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
        eta = torch.where(mask_eq, self._hpx_eta_scale * s_theta, eta_pol)
        return xi, eta

    def _world_to_pixel_2d_core(
        self, ra: Tensor, dec: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial world-to-pixel core.
        Designed to be graph-break free for torch.compile.
        """
        # Common center-equator specializations avoid full spherical rotation.
        if self._use_center_equator_fast and self._is_car:
            xi = _wrap_lon180(ra - self.alpha0)
            eta = dec
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py
        if self._use_center_equator_fast and self._is_sfl:
            phi = _wrap_lon180(ra - self.alpha0)
            theta = dec
            xi = phi * torch.cos(theta * self._d2r)
            eta = theta
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py
        if self._use_center_equator_fast and self._is_cea:
            phi = _wrap_lon180(ra - self.alpha0)
            theta = dec
            xi = phi
            eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py
        if self._use_center_equator_fast and self._is_mer:
            phi = _wrap_lon180(ra - self.alpha0)
            theta = dec
            xi = phi
            tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
            tan_arg = torch.clamp(tan_arg, min=1e-12)
            eta = self._r2d * torch.log(tan_arg)
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py

        # 1. Inverse Spherical Rotation (Celestial -> Native)
        if self._use_center_equator_fast:
            phi, theta = _wrap_lon180(ra - self.alpha0), dec
        else:
            phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        # 2. Inverse Projection (Native -> Intermediate)
        if (
            self._is_tan
            or self._is_tpv
            or self._is_sin
            or self._is_arc
            or self._is_zea
            or self._is_stg
        ):
            phi_rad = phi * self._d2r
            sin_phi, cos_phi = _sincos(phi_rad)
            if self._is_tan or self._is_tpv:
                tan_theta = torch.tan(theta * self._d2r)
                r = self._r2d / (tan_theta + 1e-15)
            elif self._is_sin:
                r = self._r2d * torch.cos(theta * self._d2r)
            elif self._is_arc:
                r = 90.0 - theta
            elif self._is_zea:
                w = (90.0 - theta) * self._d2r
                r = 2.0 * self._r2d * torch.sin(0.5 * w)
            else:
                w = (90.0 - theta) * self._d2r
                r = 2.0 * self._r2d * torch.tan(0.5 * w)
            xi = -r * sin_phi
            eta = -r * cos_phi
        elif self._is_ait:
            xi, eta = deproject_allsky(phi, theta, "AIT")
        elif self._is_mol:
            xi, eta = deproject_allsky(phi, theta, "MOL")
        elif self._is_hpx:
            phi_w = _wrap_lon180_checked(phi)
            s_theta = torch.sin(theta * self._d2r)
            abs_s = torch.abs(s_theta)
            mask_eq = abs_s <= (2.0 / 3.0)
            sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
            eta_pol = torch.sign(s_theta) * (
                self._hpx_eta_pole
                - (self._hpx_eta_pole - self._hpx_eta_boundary) * sigma
            )
            xc = torch.round((phi_w - 45.0) / 90.0) * 90.0 + 45.0
            xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
            eta = torch.where(mask_eq, self._hpx_eta_scale * s_theta, eta_pol)
        elif self._is_cea:
            xi = _wrap_lon180_checked(phi)
            eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
        elif self._is_mer:
            xi = _wrap_lon180_checked(phi)
            tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
            tan_arg = torch.clamp(tan_arg, min=1e-12)
            eta = self._r2d * torch.log(tan_arg)
        elif self._is_car:
            # Inline trivial CAR projection: native RA -> intermediate xi, native Dec -> intermediate eta
            xi = _wrap_lon180_checked(phi)
            eta = theta
        elif self._is_sfl:
            xi = _wrap_lon180_checked(phi) * torch.cos(theta * self._d2r)
            eta = theta
        elif self._is_zpn:
            if self._is_zpn_linear:
                phi_w = _wrap_lon180(phi)
                r = 90.0 - theta
                phi_rad = phi_w * self._d2r
                sin_phi, cos_phi = _sincos(phi_rad)
                xi = -r * sin_phi
                eta = -r * cos_phi
            else:
                xi, eta = deproject_zenithal(phi, theta, "ZPN", None, self._pv2_tensor)
        else:
            # Fallback
            xi, eta = deproject_zenithal(
                phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
            )

        # 3. Inverse TPV / Special distortions
        if self._is_tpv:
            xi, eta = self.tpv.invert(xi, eta)

        # 4. Inverse Linear (CD matrix)
        rel_x = self._cdi00 * xi + self._cdi01 * eta
        rel_y = self._cdi10 * xi + self._cdi11 * eta

        # 5. Inverse SIP (if any)
        if self.sip is not None:
            rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)

        # 6. Origin adjustment
        px = rel_x + self._crpix0 - (1.0 - origin)
        py = rel_y + self._crpix1 - (1.0 - origin)
        return px, py

    def _world_to_pixel_2d_fast(
        self, ra: Tensor, dec: Tensor, iterative: bool = True, origin: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Fast path for common 2D separate-argument world->pixel calls."""
        if ra.shape != dec.shape:
            raise RuntimeError("world coordinate shapes must match")
        shape = ra.shape

        device_ok = ra.device.type == "cpu" and dec.device.type == "cpu"
        dtype_ok = ra.dtype == torch.float64 and dec.dtype == torch.float64

        # Exact TAN analytic inverse: use direct intermediate coordinates instead of
        # inverse-rotation + zenithal deprojection chain.
        if self._is_tan and not self._is_tpv and not self._is_tnx and not self._is_zpx:
            xi, eta = self._tan_intermediate_from_radec(ra, dec)
            if self.sip is None:
                px, py = self._apply_cd_inv_2d(xi, eta)
            else:
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)
                px = rel_x + self._crpix0
                py = rel_y + self._crpix1
            if origin == 1:
                return px, py
            if origin == 0:
                return px.sub(1.0), py.sub(1.0)
            off = float(origin - 1)
            return px.add(off), py.add(off)

        if self._use_center_equator_fast and (
            self._is_car or self._is_sfl or self._is_cea or self._is_mer
        ):
            if (
                self._cpp_center_world_to_pixel is not None
                and device_ok
                and dtype_ok
                and not ra.requires_grad
                and not dec.requires_grad
                and (
                    self._is_sfl
                    or self._is_ait
                    or (self._is_cea and ra.numel() <= 2_048)
                )
            ):
                try:
                    return self._cpp_center_world_to_pixel(
                        ra,
                        dec,
                        self._crpix0,
                        self._crpix1,
                        self._cdi00,
                        self._cdi01,
                        self._cdi10,
                        self._cdi11,
                        self._proj_code,
                        self.alpha0,
                        self._cea_eta_scale,
                        float(origin),
                        self._hpx_h,
                        self._hpx_k,
                    )
                except (TypeError, RuntimeError):
                    pass

            phi = _wrap_lon180(ra - self.alpha0)
            theta = dec
            if self._is_car:
                xi = phi
                eta = theta
            elif self._is_sfl:
                xi = phi * torch.cos(theta * self._d2r)
                eta = theta
            elif self._is_cea:
                xi = phi
                eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
            else:
                xi = phi
                tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
                tan_arg = torch.clamp(tan_arg, min=1e-12)
                eta = self._r2d * torch.log(tan_arg)
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py

        cpp_kind = self._cpp_w2p_direct_kind
        use_cpp_direct = (
            self._has_cpp_direct_w2p
            and not self._disable_cpp_world_to_pixel
            and device_ok
            and dtype_ok
            and not ra.requires_grad
            and not dec.requires_grad
        )
        if use_cpp_direct:
            if self._use_center_equator_fast:
                phi_w = _wrap_lon180(ra - self.alpha0)
                theta = dec
            elif (
                self._native_type == "pole"
                and self._cpp_inverse_spherical_rotation_pole is not None
            ):
                phi, theta = self._cpp_inverse_spherical_rotation_pole(
                    ra,
                    dec,
                    self._phi_p_rad,
                    self._east_x,
                    self._east_y,
                    self._north_x,
                    self._north_y,
                    self._north_z,
                    self._radial_x,
                    self._radial_y,
                    self._radial_z,
                )
                phi_w = _wrap_lon180(phi)
            else:
                phi, theta = self._inverse_spherical_rotation(
                    ra, dec, self._native_type
                )
                phi_w = _wrap_lon180(phi)
            if self._cpp_direct_w2p_verified:
                xi, eta = self._cpp_deproject_direct_w2p(phi_w, theta, cpp_kind)
            else:
                try:
                    xi, eta = self._cpp_deproject_direct_w2p(phi_w, theta, cpp_kind)
                    self._cpp_direct_w2p_verified = True
                except (TypeError, RuntimeError):
                    self._disable_cpp_world_to_pixel = True
            if not self._disable_cpp_world_to_pixel:
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return px, py

        use_fused = (
            self._can_use_fused_w2p_static
            and not self._disable_fused_world_to_pixel
            and device_ok
            and dtype_ok
            and not ra.requires_grad
            and not dec.requires_grad
        )
        if use_fused:
            ra_flat = ra.reshape(-1)
            dec_flat = dec.reshape(-1)
            px = torch.empty_like(ra_flat)
            py = torch.empty_like(dec_flat)

            if self._native_type == "pole":
                ex, ey = self._pole_east_x, self._pole_east_y
                nx, ny, nz = self._pole_north_x, self._pole_north_y, self._pole_north_z
                rx, ry, rz = (
                    self._pole_radial_x,
                    self._pole_radial_y,
                    self._pole_radial_z,
                )
            else:
                ex, ey = self._east_x, self._east_y
                nx, ny, nz = self._north_x, self._north_y, self._north_z
                rx, ry, rz = self._radial_x, self._radial_y, self._radial_z

            try:
                _cpp.wcs_world_to_pixel_fused_cpu(
                    ra_flat.contiguous(),
                    dec_flat.contiguous(),
                    px,
                    py,
                    self._crpix0,
                    self._crpix1,
                    self._cdi00,
                    self._cdi01,
                    self._cdi10,
                    self._cdi11,
                    self._fused_proj_code,
                    self._fused_pv2_1,
                    self._hpx_h,
                    self._hpx_k,
                    ex,
                    ey,
                    nx,
                    ny,
                    nz,
                    rx,
                    ry,
                    rz,
                    self._ra_p_rad,
                    self._dec_p_rad,
                    self._phi_p_rad,
                    self._native_type,
                    self.alpha0,
                    self.delta0,
                    self._is_tpv,
                    self._fused_tpv_idx1,
                    self._fused_tpv_c1,
                    self._fused_tpv_idx2,
                    self._fused_tpv_c2,
                )
            except TypeError:
                # C++ extension can be stale relative to Python bindings in editable
                # workflows. Disable this fused path and fall back safely.
                self._disable_fused_world_to_pixel = True
                return self._world_to_pixel_2d_core(ra, dec, float(origin))
            px = px.reshape(shape)
            py = py.reshape(shape)
            if origin == 1:
                return px, py
            if origin == 0:
                return px.sub_(1.0), py.sub_(1.0)
            off = float(origin - 1)
            return px.add_(off), py.add_(off)

        # Fallback to core (optionally per-instance compiled).
        core_fn = self._compiled_world_to_pixel_2d_fn or self._world_to_pixel_2d_core
        return core_fn(ra, dec, float(origin))

    def compile(self, **kwargs):
        """
        Compile the WCS transform for maximum performance.
        Args:
            **kwargs: Arguments passed to torch.compile (e.g., mode, options).
        """
        compile_kwargs = {"mode": "reduce-overhead"}
        compile_kwargs.update(kwargs)
        transform_core = self._transform_optimized
        pixel_core = self._pixel_to_world_2d_core
        world_core = self._world_to_pixel_2d_core
        self._compiled_transform_fn = torch.compile(
            lambda rel: transform_core(rel), **compile_kwargs
        )
        self._compiled_pixel_to_world_2d_fn = torch.compile(
            lambda x, y, origin: pixel_core(x, y, origin), **compile_kwargs
        )
        self._compiled_world_to_pixel_2d_fn = torch.compile(
            lambda ra, dec, origin: world_core(ra, dec, origin), **compile_kwargs
        )
        return self

    def to(self, device: Union[str, torch.device]):
        """Move WCS parameters to specified device."""
        self.device = torch.device(device)
        self.crpix = self.crpix.to(self.device)
        self.crval = self.crval.to(self.device)
        self.cd_full = self.cd_full.to(self.device)
        self.cd = self.cd.to(self.device)
        self.cd_inv_full = self.cd_inv_full.to(self.device)
        self.cd_inv = self.cd_inv.to(self.device)
        self._sin_dec_p = self._sin_dec_p.to(self.device)
        self._cos_dec_p = self._cos_dec_p.to(self.device)
        if self.tpv is not None:
            self.tpv.to(self.device)
        return self

    def pixel_to_world(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Convert pixel coordinates to world coordinates.
        Args:
           origin: FITS origin (default 0). Set to 1 for FITS 1-based indexing.
        """
        origin = float(kwargs.get("origin", 0))
        if len(args) == 2 and self.naxis == 2:
            x_arg, y_arg = args
            if not isinstance(x_arg, Tensor):
                x_t = torch.as_tensor(x_arg, device=self.device, dtype=torch.float64)
                y_t = torch.as_tensor(y_arg, device=self.device, dtype=torch.float64)
            else:
                x_t = x_arg
                y_t = y_arg
            ra, dec = self._pixel_to_world_2d_fast(x_t, y_t, origin)
            return ra, dec

        if len(args) == 1:
            pixels = args[0]
            if not isinstance(pixels, Tensor):
                pixels = torch.as_tensor(
                    pixels, device=self.device, dtype=torch.float64
                )
            else:
                pixels = pixels.to(device=self.device, dtype=torch.float64)
        else:
            pixels = torch.stack(
                [
                    torch.as_tensor(a, device=self.device, dtype=torch.float64)
                    for a in args
                ],
                dim=-1,
            )
            pixels = pixels.to(device=self.device, dtype=torch.float64)

        original_shape = pixels.shape
        pixels_flat = pixels.view(-1, self.naxis)

        # If origin=0, rel = pix + 1 - crpix
        # If origin=1, rel = pix - crpix
        rel_flat = pixels_flat + (1 - origin) - self.crpix

        transform_fn = self._compiled_transform_fn or self._transform_optimized
        world_flat = transform_fn(rel_flat)

        # Reshape back
        world = world_flat.view(original_shape)

        # If input was separate args, return separate results if requested or standard
        if len(args) > 1:
            return tuple(world[..., i] for i in range(self.naxis))
        return world

    def _transform_optimized(self, rel: Tensor) -> Tensor:
        """
        The hot path for coordinate transformation.
        Input: (N, NAXIS) relative pixels (pix - crpix [internal])
        Output: (N, NAXIS) world
        """
        # 1. Linear Transform (CD @ rel) for first 2 axes
        rel_x = rel[:, 0]
        rel_y = rel[:, 1] if self.naxis >= 2 else torch.zeros_like(rel_x)

        # Check if first 2 axes are spatial (have projection code)
        is_spatial = self._is_spatial

        if not is_spatial:
            # Basic linear transformation: CRVAL + CD @ rel
            xi = self.cd[0, 0] * rel_x + self.cd[0, 1] * rel_y
            eta = self.cd[1, 0] * rel_x + self.cd[1, 1] * rel_y

            results = [self.crval[0] + xi]
            if self.naxis >= 2:
                results.append(self.crval[1] + eta)
        else:
            # Distortion & Linear
            if self._is_tpv:
                u = self.cd[0, 0] * rel_x + self.cd[0, 1] * rel_y
                v = self.cd[1, 0] * rel_x + self.cd[1, 1] * rel_y
                xi, eta = self.tpv.distort(u, v)
            else:
                if self.sip is not None:
                    rel_x, rel_y = self.sip.distort(rel_x, rel_y)

                # Linear Transform (CD @ coords)
                xi = self.cd[0, 0] * rel_x + self.cd[0, 1] * rel_y
                eta = self.cd[1, 0] * rel_x + self.cd[1, 1] * rel_y

            # Projection
            phi, theta = self._project_native_coords(xi, eta)

            # Spherical Rotation (result is in degrees)
            ra, dec = self._spherical_rotation_optimized(phi, theta, self._native_type)
            results = [ra, dec]

        # Handle NAXIS > 2 (e.g. Spectral axis 3)
        if self.naxis >= 3:
            for i in range(2, self.naxis):
                # Basic linear transformation for non-spatial axes
                val = self.crval[i] + rel[:, i] * self.cdelt[i]
                results.append(val)

        return torch.stack(results[: self.naxis], dim=-1)

    def _spherical_rotation_optimized(
        self, phi: Tensor, theta: Tensor, native_type: str = "pole"
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimized rotation using vector rotation matrix.
        """
        if native_type != "pole":
            if self._use_center_equator_fast:
                ra = _wrap_lon360(phi + self.alpha0)
                return ra, theta
            return self._spherical_rotation(phi, theta, native_type)

        if (
            self._cpp_spherical_rotation_pole is not None
            and phi.device.type == "cpu"
            and theta.device.type == "cpu"
            and phi.dtype == torch.float64
            and theta.dtype == torch.float64
            and not phi.requires_grad
            and not theta.requires_grad
        ):
            return self._cpp_spherical_rotation_pole(
                phi,
                theta,
                self._phi_p_rad,
                self._pole_east_x,
                self._pole_east_y,
                self._pole_north_x,
                self._pole_north_y,
                self._pole_north_z,
                self._pole_radial_x,
                self._pole_radial_y,
                self._pole_radial_z,
            )

        d2r = 0.017453292519943295
        r2d = 57.29577951308232

        phi_rad = (phi - self.phi_p) * d2r
        theta_rad = theta * d2r

        sin_theta, cos_theta = _sincos(theta_rad)
        sin_phi, cos_phi = _sincos(phi_rad)

        # Native Cartesian (u, v, w) relative to local basis (East, North, Radial)
        u = cos_theta * sin_phi
        v = cos_theta * cos_phi
        w = sin_theta

        # v_cel = R @ v_nat using precomputed scalar basis at (alpha_p, delta_p).
        x = u * self._pole_east_x + v * self._pole_north_x + w * self._pole_radial_x
        y = u * self._pole_east_y + v * self._pole_north_y + w * self._pole_radial_y
        z = v * self._pole_north_z + w * self._pole_radial_z

        ra = torch.remainder(torch.atan2(y, x) * r2d, 360.0)
        dec = torch.asin(torch.clamp(z, -1.0, 1.0)) * r2d

        return ra, dec

    def _dispatch_projection(self, xi, eta, proj_code):
        # Dispatch logic moved here
        phi, theta = self._project_native_coords(xi, eta)
        return phi, theta, self._project_native_type

    def _spherical_rotation(
        self, phi: Tensor, theta: Tensor, native_type: str = "pole"
    ) -> Tuple[Tensor, Tensor]:
        """
        Rotate native spherical coordinates (phi, theta) to celestial (alpha, delta)
        using vector rotation (robust at pole).
        native_type: 'pole' (Ref is 0,90) or 'center' (Ref is 0,0).
        """
        # Constants
        d2r = math.pi / 180.0
        r2d = 180.0 / math.pi

        # Convert to radians
        phi_rad = phi * d2r
        theta_rad = theta * d2r

        # 1. Convert Native (phi, theta) to Cartesian (u, v, w)
        # Consistent with zenithal.py definitions:
        # phi = atan2(xi, -eta) -> xi ~ sin(phi), -eta ~ cos(phi)
        # u matches xi direction (East)
        # v matches eta direction (North) implies v = -cos(phi)?
        # Wait, eta is North. -eta is South.
        # xi = R sin(phi). u is unit vector component?
        # u = cos(theta) * sin(phi)
        # v = -cos(theta) * cos(phi)  (Note: matches -eta direction? No. matches -cos(phi))
        # w = sin(theta)

        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)

        u = cos_theta * sin_phi
        v = -cos_theta * cos_phi
        w = sin_theta

        # 2. Construct Basis Vectors for Celestial Frame at CRVAL
        a0 = self.crval[0] * d2r
        d0 = self.crval[1] * d2r

        sin_a0, cos_a0 = torch.sin(a0), torch.cos(a0)
        sin_d0, cos_d0 = torch.sin(d0), torch.cos(d0)

        # Basis vectors definition:
        # r (radial, pointing to CRVAL) = [cos d0 cos a0, cos d0 sin a0, sin d0]
        # n (north) = [-sin d0 cos a0, -sin d0 sin a0, cos d0]
        # e (east)  = [-sin a0, cos a0, 0]

        # r components
        rx = cos_d0 * cos_a0
        ry = cos_d0 * sin_a0
        rz = sin_d0

        # n components
        nx = -sin_d0 * cos_a0
        ny = -sin_d0 * sin_a0
        nz = cos_d0

        # e components
        ex = -sin_a0
        ey = cos_a0
        ez = torch.zeros_like(a0)

        # 3. Rotate Vector
        if native_type == "pole":
            # Ref is (0, 90) -> w=1
            # Map w -> r (Pole to CRVAL)
            # Map v -> n (North-ish to North)
            # Map u -> e (East to East)
            X = u * ex + v * nx + w * rx
            Y = u * ey + v * ny + w * ry
            Z = u * ez + v * nz + w * rz
        else:  # 'center'
            # Ref is (0, 0) -> intersection of Eq/Meridian
            # At (0,0), w=0, u=0, v=-1 (cos(0)=1, v=-1).
            # We want Ref (v=-1) to map to CRVAL (r).
            # So map -v -> r => v -> -r.
            # We want North Pole (w=1) to map to North (n).
            # Map w -> n.
            # Map u -> e (East).
            # Check u x v = e x -r?
            # e x -r = n. Matched.

            X = u * ex + v * (-rx) + w * nx
            Y = u * ey + v * (-ry) + w * ny
            Z = u * ez + v * (-rz) + w * nz

        # 4. Convert Cartesian to RA/Dec
        # Dec = asin(Z)
        Z = torch.clamp(Z, -1.0, 1.0)
        dec_rad = torch.asin(Z)

        # alpha - alpha_p = atan2(y, x) -> NO.
        # X, Y, Z are absolute celestial coordinates.
        # alpha = atan2(Y, X) directly.
        alpha_rad = torch.atan2(Y, X)

        # Convert to degrees
        alpha = alpha_rad * r2d
        delta = dec_rad * r2d

        # Normalize Alpha
        alpha = torch.remainder(alpha, 360.0)

        return alpha, delta

    def _inverse_spherical_rotation(
        self, ra: Tensor, dec: Tensor, native_type: str = "pole"
    ) -> Tuple[Tensor, Tensor]:
        """
        Inverse spherical rotation: celestial (ra, dec) -> native (phi, theta).
        """
        if native_type != "pole":
            if self._use_center_equator_fast:
                phi = _wrap_lon180_checked(ra - self.alpha0)
                return phi, dec
            return self._spherical_rotation(ra, dec, native_type)

        if (
            self._cpp_inverse_spherical_rotation_pole is not None
            and ra.device.type == "cpu"
            and dec.device.type == "cpu"
            and ra.dtype == torch.float64
            and dec.dtype == torch.float64
            and not ra.requires_grad
            and not dec.requires_grad
        ):
            return self._cpp_inverse_spherical_rotation_pole(
                ra,
                dec,
                self._phi_p_rad,
                self._east_x,
                self._east_y,
                self._north_x,
                self._north_y,
                self._north_z,
                self._radial_x,
                self._radial_y,
                self._radial_z,
            )

        d2r = 0.017453292519943295
        r2d = 57.29577951308232

        ra_rad = ra * d2r
        dec_rad = dec * d2r

        sin_dec, cos_dec = _sincos(dec_rad)
        sin_ra, cos_ra = _sincos(ra_rad)

        # Celestial Cartesian (X, Y, Z)
        x_cel = cos_dec * cos_ra
        y_cel = cos_dec * sin_ra
        z_cel = sin_dec

        # v_nat = R^T @ v_cel using precomputed scalar basis at (alpha_p, delta_p).
        u = x_cel * self._pole_east_x + y_cel * self._pole_east_y
        v = (
            x_cel * self._pole_north_x
            + y_cel * self._pole_north_y
            + z_cel * self._pole_north_z
        )
        w = (
            x_cel * self._pole_radial_x
            + y_cel * self._pole_radial_y
            + z_cel * self._pole_radial_z
        )

        # theta = asin(w)
        # phi - phi_p = atan2(u, v) -- Wait, let's re-verify this.
        # Original _spherical_rotation had:
        # X = u*ex + v*nx + w*rx
        # Y = u*ey + v*ny + w*ry
        # Z = u*ez + v*nz + w*rz
        # This means u, v, w are the components in the (e, n, r) basis.

        theta = torch.asin(torch.clamp(w, -1.0, 1.0)) * r2d
        phi = torch.remainder(torch.atan2(u, v) * r2d + self.phi_p, 360.0)

        return phi, theta

    def _project_tan(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Gnomonic (TAN) projection: intermediate (deg) -> spherical (deg).

        xi, eta: Standard coordinates in degrees.
        Returns: ra, dec in degrees.
        """
        # Convert degress to radians
        rad = math.pi / 180.0
        deg = 180.0 / math.pi

        xi_rad = xi * rad
        eta_rad = eta * rad
        crval1_rad = self.crval[0] * rad
        crval2_rad = self.crval[1] * rad

        # Standard WCS formulas for TAN
        # r = sqrt(xi^2 + eta^2)
        # beta = atan(r) // distance from CRVAL
        # but the standard spherical rotation formulas often simpler:

        # phi (RA relative), theta (Dec relative)
        # For TAN:
        # x = cos(theta)*sin(phi-phi0) / sin(theta) ? No

        # Using WCSLIB logic for "zenithal" projections:
        # (phi, theta) definition depends on projection.
        # For TAN:
        # phi = atan2(xi, -eta)
        # r = hypot(xi, eta)
        # theta = atan2(180/pi, r) ?? No.

        # Let's use the robust spherical rotation matrix approach
        # 1. Deproject from plane to sphere at (0, 90) or (0, 0) native?
        # FITS standard: Native coordinates (phi, theta).
        # Determine native coordinates from xi, eta.
        # For TAN:
        # r_sq = xi^2 + eta^2
        # if r=0 -> phi=0, theta=90 (at pole)

        # Standard Gnomonic:
        # rho = sqrt(xi^2 + eta^2)
        # c = atan(rho) assuming R=1 (radians)
        # But xi, eta are in degrees in FITS intermediate?
        # Yes, usually.

        # Proper formulas from Calabretta & Greisen (2002):
        # Native spherical coordinates (phi, theta)
        # For TAN:
        # phi = atan2(xi, -eta)  (Note: -eta because Y is usually "North" which is down from pole?)
        # theta = atan2(180/pi, rho) -> implies projection from North Pole

        # Let's stick to the simplest algebraic form for RA/Dec
        # which combines deprojection + rotation.

        # Derived from astropy/wcslib logic:
        # x, y = xi_rad, eta_rad
        # r = sqrt(x*x + y*y)
        # if r == 0: return crval

        # Native coordinates (phi, theta) relative to reference point
        # For TAN (gnomonic):
        # The plane is tangent at the reference point (CRVAL).
        # We can use the direct rotation matrix formulation.

        # Direction cosines in native system (centered at pole (0, 90) for TAN?)
        # No, TAN is tangent at (0, 90) in native coords.

        # Simple trigonometry for TAN (gnomonic):
        # Project vector from center of sphere (0,0,0) through pixel on tangent plane at pole.
        # Vector V_native = [xi, eta, 1] (unnormalized)?
        # Actually V_native = [xi, eta, 1/tan(theta)]?

        # Let's use the explicit rotation matrix method which is vectorizable.
        # 1. Construct vector in tangent plane (3D).
        #    Plane is z=1 (tangent at North Pole of unit sphere).
        #    V_plane = [xi_rad, eta_rad, 1.0] ?
        #    Wait, for TAN, R_theta = 1.
        #    So V = [xi, eta, 1] is correct direction.
        #    Normalize to get unit vector on sphere.

        x = xi_rad
        y = eta_rad

        # 2. Rotate this vector from North Pole (0, 90) to (CRVAL1, CRVAL2).
        # Rotation Matrix R defined by Euler angles:
        # 1. Rotate by -phi_p (RA of pole)? No.

        # Standard FITS algorithm:
        # phi_native, theta_native from xi, eta.
        # Component angles
        # cos(beta) = 1/sqrt(1+rho^2)
        # sin(beta) = rho/sqrt(1+rho^2)

        # Spherical coordinates of point (rel to CRVAL):
        # We use the rotation formula directly.
        # alpha_p = CRVAL1
        # delta_p = CRVAL2
        # BUT FITS WCS usually assumes the native pole is at (0, 90).
        # And the reference point (CRVAL) is at the native (0, 0)? Or (0, 90)?
        # For TAN, (0,0) of intermediate (xi, eta) corresponds to (CRVAL1, CRVAL2).

        # Direct formula for TAN (Gnomonic):
        # alpha = alpha0 + atan2( x, cos(delta0) - y*sin(delta0)*tan(delta0)? )
        # Easier to just rotate vectors.

        # Vector in u,v,w space where w points to CRVAL.
        # u = x (East)
        # v = y (North)
        # w = 1.0

        u = x
        v = y
        w = torch.ones_like(x)

        # Normalize
        norm = torch.sqrt(u * u + v * v + w * w)
        u = u / norm
        v = v / norm
        w = w / norm

        # This vector (u,v,w) is in a frame where Z-axis points to (CRVAL1, CRVAL2).
        # Y-axis points North (towards NCP).
        # X-axis points East.

        # We want to rotate this frame to the celestial frame (X, Y, Z).
        # Rotation depends on (alpha0, delta0).

        a0 = crval1_rad
        d0 = crval2_rad

        # Rotation 1: Tilt up by delta0 (rotate around X-axis? No. Y-axis?)
        # Base frame: Z is (0,0). We want Z to be (a0, d0).
        # 1. Rotate around Y-axis by -(90-d0)?
        # Let's construct the rotation matrix explicitly.

        # Local basis at (a0, d0):
        # r (radial) = [cos(d0)cos(a0), cos(d0)sin(a0), sin(d0)]
        # n (north)  = [-sin(d0)cos(a0), -sin(d0)sin(a0), cos(d0)] (partial d/delta)
        # e (east)   = [-sin(a0), cos(a0), 0] (partial d/alpha / cos(delta))

        # Our vector is V = u*e + v*n + w*r

        sin_a0, cos_a0 = torch.sin(a0), torch.cos(a0)
        sin_d0, cos_d0 = torch.sin(d0), torch.cos(d0)

        # e vector components
        ex = -sin_a0
        ey = cos_a0
        ez = torch.zeros_like(a0)

        # n vector components
        nx = -sin_d0 * cos_a0
        ny = -sin_d0 * sin_a0
        nz = cos_d0

        # r vector components
        rx = cos_d0 * cos_a0
        ry = cos_d0 * sin_a0
        rz = sin_d0

        # Final vector components in Celestial Frame
        X = u * ex + v * nx + w * rx
        Y = u * ey + v * ny + w * ry
        Z = u * ez + v * nz + w * rz

        # Convert X,Y,Z back to RA, Dec
        # Dec = asin(Z)
        dec_out_rad = torch.asin(Z)

        # RA = atan2(Y, X)
        ra_out_rad = torch.atan2(Y, X)

        # Convert to degrees
        ra_out = ra_out_rad * deg
        dec_out = dec_out_rad * deg

        # Normalize RA to [0, 360]
        ra_out = torch.remainder(ra_out, 360.0)

        return ra_out, dec_out

    def world_to_pixel(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        World -> Pixel.
        """
        iterative = kwargs.get("iterative", True)
        origin = float(kwargs.get("origin", 0))

        if len(args) == 2 and self.naxis == 2:
            ra_arg, dec_arg = args
            if not isinstance(ra_arg, Tensor):
                ra_t = torch.as_tensor(ra_arg, device=self.device, dtype=torch.float64)
                dec_t = torch.as_tensor(
                    dec_arg, device=self.device, dtype=torch.float64
                )
            else:
                ra_t = ra_arg
                dec_t = dec_arg
            return self._world_to_pixel_2d_fast(
                ra_t, dec_t, iterative=iterative, origin=origin
            )

        if len(args) == 1:
            world = args[0]
            if not isinstance(world, Tensor):
                world = torch.as_tensor(world, device=self.device, dtype=torch.float64)
            else:
                world = world.to(device=self.device, dtype=torch.float64)

            # Fast path for 2D spatial with (N, 2) tensor input
            if self.naxis == 2 and self._is_spatial:
                px, py = self._world_to_pixel_2d_fast(
                    world[:, 0], world[:, 1], iterative=iterative, origin=origin
                )
                return torch.stack([px, py], dim=-1)
        else:
            world = torch.stack(
                [
                    torch.as_tensor(a, device=self.device, dtype=torch.float64)
                    for a in args
                ],
                dim=-1,
            )
            world = world.to(device=self.device, dtype=torch.float64)

        original_shape = world.shape
        world_flat = world.view(-1, self.naxis)

        # 1. Check if first 2 axes are spatial
        is_spatial = self._is_spatial

        if not is_spatial:
            # Entirely linear transformation
            # pix_rel = cd_inv @ (world - crval)
            rel_flat = torch.matmul(self.cd_inv_full, (world_flat - self.crval).T).T
            pix_internal_flat = rel_flat + self.crpix
        else:
            use_direct_hpx = (
                self._proj_code == "HPX"
                and self.sip is None
                and not self._is_tpv
                and not self._is_tnx
                and not self._is_zpx
            )

            if use_direct_hpx:
                # Direct HPX inverse avoids Newton seam branch failures and is faster.
                ra = world_flat[:, 0]
                dec = world_flat[:, 1]
                phi, theta = self._inverse_spherical_rotation(
                    ra, dec, self._native_type
                )
                xi, eta = deproject_allsky(phi, theta, "HPX", self.wcs_params)

                pix_internal_flat = self._alloc_pix_internal_like(world_flat)
                (
                    pix_internal_flat[:, 0],
                    pix_internal_flat[:, 1],
                ) = self._apply_cd_inv_2d(xi, eta)
            else:
                analytic_projections = (
                    "TAN",
                    "SIN",
                    "ARC",
                    "AIT",
                    "MOL",
                    "CEA",
                    "MER",
                    "ZEA",
                    "STG",
                )
                use_analytic = (
                    self._proj_code in analytic_projections
                    and self.sip is None
                    and not self._is_tpv
                    and not self._is_tnx
                    and not self._is_zpx
                )

                if use_analytic:
                    if self._proj_code == "TAN":
                        pix_internal_flat = self._world_to_pixel_analytic_tan(
                            world_flat
                        )
                    elif self._proj_code == "SIN":
                        pix_internal_flat = self._world_to_pixel_analytic_sin(
                            world_flat
                        )
                    elif self._proj_code == "ARC":
                        pix_internal_flat = self._world_to_pixel_analytic_arc(
                            world_flat
                        )
                    elif self._proj_code == "AIT":
                        pix_internal_flat = self._world_to_pixel_analytic_ait(
                            world_flat
                        )
                    elif self._proj_code == "MOL":
                        pix_internal_flat = self._world_to_pixel_analytic_mol(
                            world_flat
                        )
                    elif self._proj_code == "CEA":
                        pix_internal_flat = self._world_to_pixel_analytic_cea(
                            world_flat
                        )
                    elif self._proj_code == "MER":
                        pix_internal_flat = self._world_to_pixel_analytic_mer(
                            world_flat
                        )
                    elif self._proj_code == "ZEA":
                        pix_internal_flat = self._world_to_pixel_analytic_zea(
                            world_flat
                        )
                    elif self._proj_code == "STG":
                        pix_internal_flat = self._world_to_pixel_analytic_stg(
                            world_flat
                        )
                else:
                    pix_internal_flat = self._world_to_pixel_analytic_tan(world_flat)

                    if iterative:
                        ra = world_flat[:, 0]
                        dec = world_flat[:, 1]
                        x_guess = pix_internal_flat[:, 0]
                        y_guess = pix_internal_flat[:, 1]

                        constant_extra_pixels = pix_internal_flat[:, 2:]

                        def solver_wrapper(px, py):
                            full_pixels = torch.zeros_like(pix_internal_flat)
                            full_pixels[:, 0] = px
                            full_pixels[:, 1] = py
                            if self.naxis >= 3:
                                full_pixels[:, 2:] = constant_extra_pixels

                            world_res = self.pixel_to_world(full_pixels, origin=1)
                            return world_res[:, 0], world_res[:, 1]

                        x, y, _, _ = solve_newton_raphson(
                            func=solver_wrapper,
                            target_ra=ra,
                            target_dec=dec,
                            initial_x=x_guess,
                            initial_y=y_guess,
                            max_iter=20,
                            tol=1e-11,
                        )
                        pix_internal_flat[:, 0] = x
                        pix_internal_flat[:, 1] = y

        # 3. Handle extra axes (NAXIS > 2) for non-spatial WCS
        if not is_spatial and self.naxis >= 3:
            # Already handled by cd_inv_full above
            pass
        elif is_spatial and self.naxis >= 3:
            # Basic linear inverse for axis 3+
            for i in range(2, self.naxis):
                pix_internal_flat[:, i] = (
                    world_flat[:, i] - self.crval[i]
                ) / self.cdelt[i] + self.crpix[i]

        # Adjust for origin: pix = pix_internal + (origin - 1)
        pix_flat = pix_internal_flat + (origin - 1)

        pix = pix_flat.view(original_shape)
        if len(args) > 1:
            return tuple(pix[..., i] for i in range(self.naxis))
        return pix

    def _world_to_pixel_analytic_tan(self, world: Tensor) -> Tensor:
        """Analytic inverse for TAN projection (Gnomonic).

        This is the EXACT inverse - no iteration needed for TAN.
        """

        # world is (N, NAXIS)
        ra = world[:, 0]
        dec = world[:, 1]

        # Initial results same shape as input, but only spatial axes will be populated
        pix_internal = self._alloc_pix_internal_like(world)

        xi, eta = self._tan_intermediate_from_radec(ra, dec)

        # Inverse Linear (2x2)
        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_sin(self, world: Tensor) -> Tensor:
        """Analytic inverse for SIN projection (Orthographic).

        SIN projection: R = cos(theta), so xi = -R*sin(phi), eta = -R*cos(phi).
        Using direction cosines: u = cos(theta)*sin(phi), v = cos(theta)*cos(phi), w = sin(theta).
        For SIN: xi = u * deg, eta = v * deg.
        """
        deg = self._r2d

        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        # Get native coordinates via inverse spherical rotation
        phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        phi_rad = phi * self._d2r
        theta_rad = theta * self._d2r

        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)

        xi = cos_theta * sin_phi * deg
        eta = -cos_theta * cos_phi * deg

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_arc(self, world: Tensor) -> Tensor:
        """Analytic inverse for ARC projection (Zenithal Equidistant).

        ARC: R = 90 - theta (in degrees), so xi = -R*sin(-phi), eta = -R*cos(-phi).
        Using direction cosines: theta = arcsin(w), phi = atan2(u, -v).
        """

        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        # Get native coordinates via inverse spherical rotation
        phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        theta * self._d2r
        R = 90.0 - theta

        phi_rad = phi * self._d2r
        xi = R * torch.sin(phi_rad)
        eta = -R * torch.cos(phi_rad)

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_ait(self, world: Tensor) -> Tensor:
        """Analytic inverse for AIT projection (Hammer-Aitoff).

        AIT is an equal-area all-sky projection.
        Forward: x = 2 * cos(theta) * sin(phi/2) / sqrt((1+cos(theta)*cos(phi/2)))
                 y = sin(theta) / sqrt((1+cos(theta)*cos(phi/2)))

        Inverse requires solving for phi, theta from x, y.
        """
        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)
        if self._use_center_equator_fast:
            phi = torch.remainder(ra - self.alpha0 + 180.0, 360.0) - 180.0
            theta = dec
        else:
            phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        # AIT forward:
        # c = sqrt((1 + cos(theta)*cos(phi/2)) / 2)
        # xi = 2 * cos(theta) * sin(phi/2) / c * (180/pi)
        # eta = sin(theta) / c * (180/pi)

        # Inverse AIT from xi, eta:
        # Given x, y in AIT coordinates:
        # z2 = (1 - (x/4)^2 - (y/2)^2)
        # if z2 < 0: invalid
        # c = sqrt(z2)
        # theta = asin(y * c)
        # phi = 2 * atan2(x * c / 2, 2*z2 - 1)

        # For inverse from (phi, theta) to (xi, eta):
        phi_rad = phi * self._d2r
        theta_rad = theta * self._d2r
        half_phi = phi_rad * 0.5
        cos_theta = torch.cos(theta_rad)
        cos_half_phi = torch.cos(half_phi)

        denom = torch.sqrt(0.5 * (1.0 + cos_theta * cos_half_phi))

        xi = 2.0 * cos_theta * torch.sin(half_phi) / denom * self._r2d
        eta = torch.sin(theta_rad) / denom * self._r2d

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_mol(self, world: Tensor) -> Tensor:
        """Analytic inverse for MOL projection (Mollweide)."""
        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        if self._use_center_fast:
            phi = _wrap_lon180_checked(ra)
            theta = dec
        else:
            phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        xi, eta = deproject_allsky(
            phi, theta, "MOL", self._pv1_tensor, self._pv2_tensor
        )

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_cea(self, world: Tensor) -> Tensor:
        """Analytic inverse for CEA projection (Cylindrical Equal Area)."""
        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        if self._use_center_fast:
            xi = torch.remainder(ra + 180.0, 360.0) - 180.0
            eta = self._cea_eta_scale * torch.sin(dec * self._d2r)
        else:
            phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)
            xi, eta = deproject_cylindrical(
                phi, theta, "CEA", self._pv1_tensor, self._pv2_tensor
            )

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_mer(self, world: Tensor) -> Tensor:
        """Analytic inverse for MER projection (Mercator)."""
        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        if self._use_center_fast:
            phi = _wrap_lon180_checked(ra)
            theta = dec
        else:
            phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        xi, eta = deproject_cylindrical(
            phi, theta, "MER", self._pv1_tensor, self._pv2_tensor
        )

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_zea(self, world: Tensor) -> Tensor:
        """Analytic inverse for ZEA projection (Zenithal Equal Area)."""

        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        # Get native coordinates via inverse spherical rotation
        phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        theta_rad = theta * self._d2r
        r = 2.0 * torch.rad2deg(torch.sin((math.pi / 2.0 - theta_rad) / 2.0))

        phi_rad = phi * self._d2r
        xi = r * torch.sin(phi_rad)
        eta = -r * torch.cos(phi_rad)

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal

    def _world_to_pixel_analytic_stg(self, world: Tensor) -> Tensor:
        """Analytic inverse for STG projection (Stereographic)."""

        ra = world[:, 0]
        dec = world[:, 1]
        pix_internal = self._alloc_pix_internal_like(world)

        # Get native coordinates via inverse spherical rotation
        phi, theta = self._inverse_spherical_rotation(ra, dec, self._native_type)

        theta_rad = theta * self._d2r
        r = 2.0 * torch.rad2deg(torch.tan((math.pi / 2.0 - theta_rad) / 2.0))

        phi_rad = phi * self._d2r
        xi = r * torch.sin(phi_rad)
        eta = -r * torch.cos(phi_rad)

        pix_internal[:, 0], pix_internal[:, 1] = self._apply_cd_inv_2d(xi, eta)

        return pix_internal
