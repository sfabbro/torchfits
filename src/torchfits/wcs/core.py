
import torch
from torch import Tensor
from typing import Optional, Union, Tuple, Dict, Any, List
import math

from .tpv import TPV

class WCS:
    """
    Base class for TorchWCS (PyTorch-native World Coordinate System).
    
    This class handles the core logic for WCS transformations, including
    parsing FITS headers and delegating to specific projection implementations.
    """
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
        
        if header is not None:
            self._parse_header(header)
            # Check for SIP
            if 'A_ORDER' in header or 'B_ORDER' in header:
                self.sip = SIP(header)
            
            # Check for TPV
            if self.wcs_params.get('CTYPE1', '').endswith('TPV'):
                self.tpv = TPV(self.wcs_params) # Pass parsed params which contain PVs
        
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

        # Move parameters to appropriate buffers/tensors
        self._setup_tensors()

    def _parse_header(self, header: Dict[str, Any]):
        """Parse standard FITS WCS keywords."""
        # Standard defaults
        self.wcs_params['NAXIS'] = header.get('NAXIS', 2)
        
        # CRPIX: Reference pixel
        self.wcs_params['CRPIX1'] = float(header.get('CRPIX1', 0.0))
        self.wcs_params['CRPIX2'] = float(header.get('CRPIX2', 0.0))
        
        # CRVAL: Reference value
        self.wcs_params['CRVAL1'] = float(header.get('CRVAL1', 0.0))
        self.wcs_params['CRVAL2'] = float(header.get('CRVAL2', 0.0))
        
        # CD matrix or PC matrix + CDELT
        if 'CD1_1' in header:
            self.wcs_params['CD1_1'] = float(header.get('CD1_1', 1.0))
            self.wcs_params['CD1_2'] = float(header.get('CD1_2', 0.0))
            self.wcs_params['CD2_1'] = float(header.get('CD2_1', 0.0))
            self.wcs_params['CD2_2'] = float(header.get('CD2_2', 1.0))
            self.has_cd = True
        else:
            self.has_cd = False
            # simplistic fallback for PC/CDELT is often CDELT1, CDELT2 on diagonal
            # robust implementation would handle PC matrix normalization
            cdelt1 = float(header.get('CDELT1', 1.0))
            cdelt2 = float(header.get('CDELT2', 1.0))
            pc1_1 = float(header.get('PC1_1', 1.0))
            pc1_2 = float(header.get('PC1_2', 0.0))
            pc2_1 = float(header.get('PC2_1', 0.0))
            pc2_2 = float(header.get('PC2_2', 1.0))
            
            self.wcs_params['CD1_1'] = cdelt1 * pc1_1
            self.wcs_params['CD1_2'] = cdelt1 * pc1_2
            self.wcs_params['CD2_1'] = cdelt2 * pc2_1
            self.wcs_params['CD2_2'] = cdelt2 * pc2_2
            
        # CTYPE
        self.wcs_params['CTYPE1'] = str(header.get('CTYPE1', ''))
        self.wcs_params['CTYPE2'] = str(header.get('CTYPE2', ''))

        # Capture PV keywords (PVi_j)
        # SCAMP/PV distortions can have many terms (0..39 typically)
        for i in [1, 2]:
            for j in range(40):
                key = f'PV{i}_{j}'
                if key in header:
                    self.wcs_params[key] = float(header[key])

    def _setup_tensors(self):
        """Convert scalar params to PyTorch tensors for computation."""
        # Convert params to tensors on CPU initially
        # We store them as a flat tensor or dictionary of scalar tensors?
        # A dictionary of scalar tensors is easy to work with
        
        self.crpix = torch.tensor([
            self.wcs_params.get('CRPIX1', 0.0),
            self.wcs_params.get('CRPIX2', 0.0)
        ], dtype=torch.float64)
        
        self.crval = torch.tensor([
            self.wcs_params.get('CRVAL1', 0.0),
            self.wcs_params.get('CRVAL2', 0.0)
        ], dtype=torch.float64)
        
        self.cd = torch.tensor([
            [self.wcs_params.get('CD1_1', 1.0), self.wcs_params.get('CD1_2', 0.0)],
            [self.wcs_params.get('CD2_1', 0.0), self.wcs_params.get('CD2_2', 1.0)]
        ], dtype=torch.float64)
        
        # Precompute inverse CD matrix for world->pixel
        self.cd_inv = torch.inverse(self.cd)

    def to(self, device: Union[str, torch.device]):
        """Move WCS parameters to specified device."""
        self.device = torch.device(device)
        self.crpix = self.crpix.to(self.device)
        self.crval = self.crval.to(self.device)
        self.cd = self.cd.to(self.device)
        self.cd_inv = self.cd_inv.to(self.device)
        return self

    def pixel_to_world(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert pixel coordinates to world coordinates.
        Default implementation assumes simple TAN projection for now (placeholder).
        
        Args:
            x: X pixel coordinates (0-indexed)
            y: Y pixel coordinates (0-indexed)
            
        Returns:
            (ra, dec) tuple of tensors
        """
        # Ensure inputs are tensors on correct device
        if not isinstance(x, Tensor):
            x = torch.tensor(x, device=self.crpix.device, dtype=self.crpix.dtype)
        if not isinstance(y, Tensor):
            y = torch.tensor(y, device=self.crpix.device, dtype=self.crpix.dtype)
            
        # 1. Pixel to Intermediate (Linear Transform)
        # intermediate = CD * (pixel - crpix)
        # x, y = (N,) vectors
        
        # Stack into (2, N)
        # pixel_coords shape: (2, N) or (2, H, W) simplified to (2, -1) for matmul
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Offset from CRPIX (1-based FITS vs 0-based Python adjustment handled here?)
        # FITS uses 1-based indexing for CRPIX. So if CRPIX=100, the 100th pixel is the center.
        # In 0-based, that's index 99.
        # Standard WCSlib logic: intermediate = CD * (p - crpix)
        # where p follows FITS convention (1-based).
        # So if input x,y are 0-based, we pass (x+1, y+1) or adjust CRPIX by -1.
        # Let's verify standard: usually CRPIX is given in FITS coordinates.
        # PyTorch coordinates are 0-based.
        # So we use (x + 1 - CRPIX) or (x - (CRPIX - 1)).
        
        rel_x = x_flat - (self.crpix[0] - 1.0)
        rel_y = y_flat - (self.crpix[1] - 1.0)
        
        # 0. Apply SIP Distortion (if present)
        # u, v -> u', v'
        if self.sip is not None:
            rel_x, rel_y = self.sip.distort(rel_x, rel_y)
        
        # Check for TPV (Tangent + PV distortion)
        # TPV replaces the linear CD transformation + distortion step.
        # It takes pixel offsets (rel_x, rel_y) and outputs intermediate world coords (degrees).
        if self.tpv is not None:
             if self.tpv is None: # Redundant check but keeps linter happy if type inference fails
                 pass
             xi, eta = self.tpv.distort(rel_x, rel_y)
        else:
            # Standard Linear Transform
            # Batched matmul
            # [xi, eta] = CD @ [rel_x, rel_y]
            coords = torch.stack([rel_x, rel_y], dim=0)
            intermediate = torch.matmul(self.cd, coords)
            
            xi = intermediate[0]
            eta = intermediate[1]
             
        ctype1 = self.wcs_params['CTYPE1']
             
        # 2. Intermediate to World (Projection)
        # Dispatch to projection function based on CTYPE
        # TAN and TPV both use gnomonic projection after distortion
        if 'TAN' in ctype1 or 'TPV' in ctype1:
            ra, dec = self._project_tan(xi, eta)
        else:
             # Fallback or error
             raise NotImplementedError(f"Projection {ctype1} not supported yet")
             
        return ra.reshape(original_shape), dec.reshape(original_shape)
    
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
        # R_theta = 180/pi / sqrt(xi^2 + eta^2) ?
        
        rho = torch.sqrt(x*x + y*y)
        beta = torch.atan(rho) # angle from tangent point center
        
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
        norm = torch.sqrt(u*u + v*v + w*w)
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

    def world_to_pixel(self, ra: Tensor, dec: Tensor) -> Tuple[Tensor, Tensor]:
        """
        World (RA, Dec) -> Pixel (x, y).
        Inverse of pixel_to_world.
        """
        # Ensure inputs are tensors
        if not isinstance(ra, Tensor):
            ra = torch.tensor(ra, device=self.crpix.device, dtype=self.crpix.dtype)
        if not isinstance(dec, Tensor):
            dec = torch.tensor(dec, device=self.crpix.device, dtype=self.crpix.dtype)
            
        rad = math.pi / 180.0
        deg = 180.0 / math.pi
        
        ra_rad = ra * rad
        dec_rad = dec * rad
        
        # 1. Convert RA/Dec to Cartesian on Unit Sphere
        X = torch.cos(dec_rad) * torch.cos(ra_rad)
        Y = torch.cos(dec_rad) * torch.sin(ra_rad)
        Z = torch.sin(dec_rad)
        
        # 2. Project onto local basis vectors (e, n, r) defined by CRVAL
        crval1_rad = self.crval[0] * rad
        crval2_rad = self.crval[1] * rad
        
        sin_a0, cos_a0 = torch.sin(crval1_rad), torch.cos(crval1_rad)
        sin_d0, cos_d0 = torch.sin(crval2_rad), torch.cos(crval2_rad)
        
        # Basis vectors (same as above)
        # e = (-sin a0, cos a0, 0)
        # n = (-sin d0 cos a0, -sin d0 sin a0, cos d0)
        # r = (cos d0 cos a0, cos d0 sin a0, sin d0)
        
        # Project vector V=(X,Y,Z) onto these axes
        # u = V . e
        # v = V . n
        # w = V . r
        
        u = X * (-sin_a0) + Y * (cos_a0)
        v = X * (-sin_d0 * cos_a0) + Y * (-sin_d0 * sin_a0) + Z * (cos_d0)
        w = X * (cos_d0 * cos_a0) + Y * (cos_d0 * sin_a0) + Z * (sin_d0)
        
        # 3. Convert (u, v, w) to intermediate (xi, eta)
        # For TAN projection:
        # The plane is tangent at w=1 ?? No, plane is at distance 1.
        # Ray from origin passes through (u,v,w) and hits plane z=1?
        # No, basis is (e, n, r). 'r' is the 'z' axis of the projection.
        # WCS convention: projection plane is tangent to sphere at 'r' axis.
        # So we project from origin (0,0,0) to plane w=1.
        # intersection = (u/w, v/w, 1)
        
        # Standard Gnomonic:
        # xi = u / w
        # eta = v / w
        # (check for w=0 divergence - horizon!)
        
        # Handle w near 0
        mask_valid = w > 1e-10
        
        xi_rad = torch.zeros_like(u)
        eta_rad = torch.zeros_like(v)
        
        xi_rad[mask_valid] = u[mask_valid] / w[mask_valid]
        eta_rad[mask_valid] = v[mask_valid] / w[mask_valid]
        # Invalids remain 0 or should be NaN? FITS WCS usually produces NaN or distinct error.
        # Leavs as 0 for now but maybe mark?
        
        xi = xi_rad * deg
        eta = eta_rad * deg
        
        # 4. Intermediate to Pixel (Inverse Linear)
        # [xi, eta] = CD @ [rel_x, rel_y]
        # => [rel_x, rel_y] = CD_inv @ [xi, eta]
        
        coords = torch.stack([xi, eta], dim=0) # (2, N)
        rel_coords = torch.matmul(self.cd_inv, coords)
        
        rel_x = rel_coords[0]
        rel_y = rel_coords[1]
        
        # 4b. Apply SIP Reverse Distortion (if present)
        # u', v' -> u, v
        if self.sip is not None:
             rel_x, rel_y = self.sip.undistort(rel_x, rel_y)
        
        # 5. Add CRPIX
        # x = rel_x + (crpix - 1)
        x = rel_x + (self.crpix[0] - 1.0)
        y = rel_y + (self.crpix[1] - 1.0)
        
        if ra.ndim == 0:
            return x.item(), y.item()
            
        return x, y
