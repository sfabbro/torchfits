import torch
from torch import Tensor
from typing import Optional, Union, Tuple, Dict, Any
import math

from .tpv import TPV
from .sip import SIP
from .zenithal import project_zenithal
from .cylindrical import project_cylindrical
from .allsky import project_allsky, deproject_allsky
from .legacy import project_tnx, project_zpx
from .utils import solve_newton_raphson


class WCS:
    """
    Base class for TorchWCS (PyTorch-native World Coordinate System).
    """

    ZENITHAL_CODES = ("TAN", "SIN", "ARC", "ZPN", "STG", "ZEA")
    CYLINDRICAL_CODES = ("CEA", "MER", "CYP")
    ALLSKY_CODES = ("AIT", "MOL", "HPX")
    SPECIAL_CODES = ("ZPX", "TNX", "TPV")

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

            # Check for TPV
            if self.wcs_params.get("CTYPE1", "").endswith("TPV"):
                self.tpv = TPV(self.wcs_params)  # Pass parsed params which contain PVs

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

        # Move parameters to appropriate buffers/tensors
        self._setup_tensors()

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
        for i in [1, 2]:
            for j in range(40):
                key = f"PV{i}_{j}"
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
        if self._proj_code in ("AIT", "MOL", "HPX", "CEA", "MER", "CYP"):
            self._theta0 = 0.0

        # LONPOLE / LATPOLE
        self.phi_p = float(
            self.wcs_params.get("LONPOLE", 180.0 if self._theta0 == 90.0 else 0.0)
        )
        self.theta_p = float(self.wcs_params.get("LATPOLE", 90.0))

        d2r = math.pi / 180.0

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
        dec_p_tensor = torch.as_tensor(
            self._dec_p_rad, dtype=torch.float64, device=self.device
        )
        self._sin_dec_p = torch.sin(dec_p_tensor)
        self._cos_dec_p = torch.cos(dec_p_tensor)

        # Detect if fast TAN is possible
        self._use_fast_tan = (
            self._proj_code == "TAN"
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.phi_p == 180.0
            and self.delta0 == self.delta_p
        )

    def compile(self, **kwargs):
        """
        Compile the WCS transform for maximum performance.
        Args:
            **kwargs: Arguments passed to torch.compile (e.g., mode, options).
        """
        self._transform_optimized = torch.compile(self._transform_optimized, **kwargs)
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
        origin = kwargs.get("origin", 0)
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

        world_flat = self._transform_optimized(rel_flat)

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
        is_spatial = self._proj_code in (
            self.ZENITHAL_CODES
            + self.CYLINDRICAL_CODES
            + self.ALLSKY_CODES
            + self.SPECIAL_CODES
        )

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
            phi, theta, _ = self._dispatch_projection(xi, eta, self._proj_code)

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
        Optimized rotation using full 3-angle spherical formula.
        """
        # The fast closed form below assumes pole-native frames (theta0=90).
        # Cylindrical/all-sky projections use center-native frames and require
        # the general vector-basis rotation.
        if native_type != "pole":
            return self._spherical_rotation(phi, theta, native_type)

        d2r = 0.017453292519943295
        r2d = 57.29577951308232

        phi_rad = (phi - self.phi_p) * d2r
        theta_rad = theta * d2r

        sin_theta, cos_theta = torch.sin(theta_rad), torch.cos(theta_rad)
        sin_phi, cos_phi = torch.sin(phi_rad), torch.cos(phi_rad)

        # Eq 2-4:
        # sin delta = sin theta sin delta_p + cos theta cos delta_p cos(phi_rad)
        # cos delta sin(alpha - alpha_p) = cos theta sin phi_rad
        # cos delta cos(alpha - alpha_p) = sin theta cos delta_p - cos theta sin delta_p cos phi_rad

        ct_cp = cos_theta * cos_phi

        sd = sin_theta * self._sin_dec_p + ct_cp * self._cos_dec_p
        sd = torch.clamp(sd, -1.0, 1.0)
        dec_rad = torch.asin(sd)

        y = cos_theta * sin_phi
        x = sin_theta * self._cos_dec_p - ct_cp * self._sin_dec_p
        ra_rad = self._ra_p_rad + torch.atan2(y, x)

        ra = ra_rad * r2d
        dec = dec_rad * r2d

        # Normalize RA to [0, 360]
        ra = torch.remainder(ra, 360.0)

        return ra, dec

    def _dispatch_projection(self, xi, eta, proj_code):
        # Dispatch logic moved here
        zenithal_codes = ("TAN", "SIN", "ARC", "ZPN", "STG", "ZEA")
        cylindrical_codes = ("CEA", "MER", "CYP")
        allsky_codes = ("AIT", "MOL", "HPX")

        if proj_code == "ZPX":
            return (*project_zpx(xi, eta, self.wcs_params, self.wat_data), "pole")
        elif proj_code == "TNX":
            xi, eta = project_tnx(xi, eta, self.wcs_params, self.wat_data)
            return (*project_zenithal(xi, eta, "TAN", self.wcs_params), "pole")
        elif proj_code == "TPV":
            return (*project_zenithal(xi, eta, "TAN", self.wcs_params), "pole")
        elif proj_code in zenithal_codes:
            return (*project_zenithal(xi, eta, proj_code, self.wcs_params), "pole")
        elif proj_code in cylindrical_codes:
            return (*project_cylindrical(xi, eta, proj_code, self.wcs_params), "center")
        elif proj_code in allsky_codes:
            return (*project_allsky(xi, eta, proj_code, self.wcs_params), "center")
        else:
            # Fallback to identity (linear) if unknown projection (Paper I Linear)
            # This handles CTYPEs like 'WAVE', 'FREQ', 'TIME' without specific suffixes.
            return xi, eta, "pole"

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
        d2r = math.pi / 180.0
        r2d = 180.0 / math.pi

        ra_rad = ra * d2r
        dec_rad = dec * d2r

        # Celestial unit vector.
        X = torch.cos(dec_rad) * torch.cos(ra_rad)
        Y = torch.cos(dec_rad) * torch.sin(ra_rad)
        Z = torch.sin(dec_rad)

        # Basis at CRVAL.
        a0 = self.crval[0] * d2r
        d0 = self.crval[1] * d2r
        sin_a0, cos_a0 = torch.sin(a0), torch.cos(a0)
        sin_d0, cos_d0 = torch.sin(d0), torch.cos(d0)

        # e (east), n (north), r (radial).
        ex = -sin_a0
        ey = cos_a0
        ez = torch.zeros_like(a0)

        nx = -sin_d0 * cos_a0
        ny = -sin_d0 * sin_a0
        nz = cos_d0

        rx = cos_d0 * cos_a0
        ry = cos_d0 * sin_a0
        rz = sin_d0

        # Native components in the local basis.
        u = X * ex + Y * ey + Z * ez
        if native_type == "pole":
            v = X * nx + Y * ny + Z * nz
            w = X * rx + Y * ry + Z * rz
        else:
            # Forward center-native mapping uses v * (-r) and w * n.
            v = -(X * rx + Y * ry + Z * rz)
            w = X * nx + Y * ny + Z * nz

        w = torch.clamp(w, -1.0, 1.0)
        theta = torch.asin(w) * r2d
        phi = torch.atan2(u, -v) * r2d
        phi = torch.remainder(phi, 360.0)
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
        origin = kwargs.get("origin", 0)

        if len(args) == 1:
            world = args[0]
            if not isinstance(world, Tensor):
                world = torch.as_tensor(world, device=self.device, dtype=torch.float64)
            else:
                world = world.to(device=self.device, dtype=torch.float64)
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
        is_spatial = self._proj_code in (
            self.ZENITHAL_CODES
            + self.CYLINDRICAL_CODES
            + self.ALLSKY_CODES
            + self.SPECIAL_CODES
        )

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

                coords_2d = torch.stack([xi, eta], dim=0)
                rel_2d = torch.matmul(self.cd_inv.to(coords_2d.dtype), coords_2d)

                pix_internal_flat = torch.zeros_like(world_flat)
                pix_internal_flat[:, 0] = rel_2d[0] + self.crpix[0]
                pix_internal_flat[:, 1] = rel_2d[1] + self.crpix[1]
            else:
                # Initial guess (Analytic TAN for axes 1, 2)
                pix_internal_flat = self._world_to_pixel_analytic_tan(world_flat)

                if iterative:
                    # Solve spatial part
                    ra = world_flat[:, 0]
                    dec = world_flat[:, 1]
                    x_guess = pix_internal_flat[:, 0]
                    y_guess = pix_internal_flat[:, 1]

                    # Wrap pixel_to_world to keep extra axes constant
                    constant_extra_pixels = pix_internal_flat[:, 2:]

                    def solver_wrapper(px, py):
                        # Construct full pixel vector
                        # px, py are (N,)
                        full_pixels = torch.zeros_like(pix_internal_flat)
                        full_pixels[:, 0] = px
                        full_pixels[:, 1] = py
                        if self.naxis >= 3:
                            full_pixels[:, 2:] = constant_extra_pixels

                        # Forward transform
                        world_res = self.pixel_to_world(full_pixels, origin=1)
                        # Return first two columns (RA, Dec)
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
        """Analytic inverse for TAN projection (Gnomonic)."""
        rad = 0.017453292519943295
        deg = 57.29577951308232

        # world is (N, NAXIS)
        ra = world[:, 0]
        dec = world[:, 1]

        # Initial results same shape as input, but only spatial axes will be populated
        pix_internal = torch.zeros_like(world)

        ra_rad = ra * rad
        dec_rad = dec * rad

        # 1. Convert RA/Dec to Cartesian
        X = torch.cos(dec_rad) * torch.cos(ra_rad)
        Y = torch.cos(dec_rad) * torch.sin(ra_rad)
        Z = torch.sin(dec_rad)

        sin_a0 = math.sin(self.alpha0 * rad)
        cos_a0 = math.cos(self.alpha0 * rad)
        sin_d0 = math.sin(self.delta0 * rad)
        cos_d0 = math.cos(self.delta0 * rad)

        # u, v, w in local projection frame
        u = X * (-sin_a0) + Y * (cos_a0)
        v = X * (-sin_d0 * cos_a0) + Y * (-sin_d0 * sin_a0) + Z * (cos_d0)
        w = X * (cos_d0 * cos_a0) + Y * (cos_d0 * sin_a0) + Z * (sin_d0)

        w = torch.clamp(w, min=1e-12)
        xi = (u / w) * deg
        eta = (v / w) * deg

        # Inverse Linear (2x2)
        coords_2d = torch.stack([xi, eta], dim=0)
        rel_2d = torch.matmul(self.cd_inv.to(coords_2d.dtype), coords_2d)

        pix_internal[:, 0] = rel_2d[0] + self.crpix[0]
        pix_internal[:, 1] = rel_2d[1] + self.crpix[1]

        return pix_internal
