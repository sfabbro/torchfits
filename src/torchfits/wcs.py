"""
WCS (World Coordinate System) functionality for torchfits.

This module provides efficient coordinate transformations using a hybrid approach:
1. wcslib (via C++) for robust header parsing and normalization.
2. Pure PyTorch for high-performance, GPU-accelerated coordinate transformations.
"""

from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import math

from .hdu import Header

import torchfits.cpp as cpp


class WCS:
    """
    A PyTorch-native WCS implementation with GPU support.

    Uses wcslib for parsing but performs all transformations in PyTorch,
    enabling batch processing, GPU acceleration, and differentiability.
    """

    def __init__(self, header: Header):
        """
        Initialize WCS from FITS header.

        Args:
            header: FITS header containing WCS keywords
        """
        self._header = header

        # 1. Prepare minimal header for C++ wcslib initialization
        wcs_header = {
            "SIMPLE": "T",
            "BITPIX": "-32",
            "NAXIS": "2",
            "NAXIS1": "100",
            "NAXIS2": "100",
        }

        # Update with provided header values, ensuring strings are quoted for wcslib
        header_dict = {}
        for k, v in header.items():
            if isinstance(v, bool):
                val_str = "T" if v else "F"
            elif isinstance(v, (int, float)):
                val_str = str(v)
            elif isinstance(v, str):
                try:
                    float(v)
                    val_str = v
                except ValueError:
                    val_str = str(v)
                    if not (val_str.startswith("'") and val_str.endswith("'")):
                        val_str = f"'{val_str}'"
            else:
                val_str = str(v)
            header_dict[str(k)] = val_str

        wcs_header.update(header_dict)

        # 2. Initialize C++ WCS to parse and normalize main parameters
        self._wcs = cpp.WCS(wcs_header)

        # 3. Extract parameters into PyTorch buffers
        # Linear parameters (from wcslib normalization)
        self._pc = self._wcs.pc.clone()  # (N, N)
        self._cdelt = self._wcs.cdelt.clone()  # (N,)
        self._crpix = self._wcs.crpix.clone()  # (N,) - 0-based
        self._crval = self._wcs.crval.clone()  # (N,)

        self.naxis = self._wcs.naxis
        self._ctype = self._wcs.ctype
        self._cunit = self._wcs.cunit
        self._lonpole = self._wcs.lonpole
        self._latpole = self._wcs.latpole

        # 4. Parse SIP coefficients from Python header (C++ doesn't expose them cleanly yet)
        self._sip_a = self._parse_sip_coeffs(header, "A")
        self._sip_b = self._parse_sip_coeffs(header, "B")
        self._sip_ap = self._parse_sip_coeffs(header, "AP")  # Inverse
        self._sip_bp = self._parse_sip_coeffs(header, "BP")  # Inverse

        self._has_sip = self._sip_a is not None or self._sip_b is not None

        # Check for TPV
        self._has_tpv = False
        self._pv_x = None
        self._pv_y = None

        # Check original header for TPV, as wcslib might normalize CTYPE to TAN
        ctype1_raw = header.get("CTYPE1", "")
        ctype2_raw = header.get("CTYPE2", "")

        if ctype1_raw.endswith("TPV") or ctype2_raw.endswith("TPV"):
            self._has_tpv = True
            pv_x, pv_y = self._parse_pv(header)
            self._pv_x = pv_x
            self._pv_y = pv_y

        # Ensure latpole/lonpole are set
        if not hasattr(self._wcs, "lonpole"):
            self._lonpole = 180.0
        if not hasattr(self._wcs, "latpole"):
            self._latpole = 0.0

    def to(self, device: torch.device) -> "WCS":
        """Move WCS parameters to device."""
        self._pc = self._pc.to(device)
        self._cdelt = self._cdelt.to(device)
        self._crpix = self._crpix.to(device)
        self._crval = self._crval.to(device)

        if self._sip_a is not None:
            self._sip_a = self._sip_a.to(device)
        if self._sip_b is not None:
            self._sip_b = self._sip_b.to(device)
        if self._sip_ap is not None:
            self._sip_ap = self._sip_ap.to(device)
        if self._sip_bp is not None:
            self._sip_bp = self._sip_bp.to(device)
        return self

    def _parse_sip_coeffs(self, header: Header, prefix: str) -> Optional[Tensor]:
        """Parse SIP coefficients (e.g. A_2_0) into a dense tensor."""
        order_key = f"{prefix}_ORDER"
        if order_key not in header:
            return None

        order = int(header[order_key])

        # Create dense tensor (order+1, order+1)
        # coeffs[p, q] corresponds to u^p * v^q
        coeffs = torch.zeros((order + 1, order + 1), dtype=torch.float64)

        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                # key e.g., A_2_0
                key = f"{prefix}_{p}_{q}"
                if key in header:
                    coeffs[p, q] = float(header[key])

        return coeffs

    @property
    def pixel_scale(self) -> Tensor:
        return torch.abs(self._cdelt)

    @property
    def center_coord(self) -> Tensor:
        naxis1 = float(self._header.get("NAXIS1", 100))
        naxis2 = float(self._header.get("NAXIS2", 100))
        center_pix = torch.tensor(
            [[naxis1 / 2.0, naxis2 / 2.0]],
            dtype=torch.float64,
            device=self._crpix.device,
        )
        return self.pixel_to_world(center_pix).squeeze()

    @property
    def crpix(self) -> Tensor:
        return self._crpix

    @property
    def crval(self) -> Tensor:
        return self._crval

    @property
    def cdelt(self) -> Tensor:
        return self._cdelt

    @property
    def pc(self) -> Tensor:
        return self._pc

    @property
    def ctype(self) -> list[str]:
        return self._ctype

    def _parse_pv(self, header: dict) -> tuple[dict, dict]:
        """Parse PV distortion coefficients for TPV projection."""
        pv_x = {}
        pv_y = {}

        # Parse PVi_j
        for key, value in header.items():
            if key.startswith("PV") and "_" in key:
                try:
                    # Format PVi_j where i is axis (1 or 2), j is term
                    parts = key[2:].split("_")
                    axis = int(parts[0])
                    term = int(parts[1])

                    if axis == 1:
                        pv_x[term] = float(value)
                    elif axis == 2:
                        pv_y[term] = float(value)
                except (ValueError, IndexError):
                    continue

        return pv_x, pv_y

    def _apply_sip(
        self, xy: Tensor, coeffs_x: Optional[Tensor], coeffs_y: Optional[Tensor]
    ) -> Tensor:
        """Apply SIP distortion: f(u,v) = u + coeffs(u,v)."""
        if coeffs_x is None and coeffs_y is None:
            return xy

        u = xy[:, 0]
        v = xy[:, 1]

        delta_u = torch.zeros_like(u)
        delta_v = torch.zeros_like(v)

        # Precompute powers
        # u_pow[p] = u^p
        max_order = 0
        if coeffs_x is not None:
            max_order = max(max_order, coeffs_x.shape[0] - 1)
        if coeffs_y is not None:
            max_order = max(max_order, coeffs_y.shape[0] - 1)

        u_pows = [torch.ones_like(u)]
        v_pows = [torch.ones_like(v)]

        for i in range(1, max_order + 1):
            u_pows.append(u_pows[-1] * u)
            v_pows.append(v_pows[-1] * v)

        if coeffs_x is not None:
            rows, cols = coeffs_x.shape
            for p in range(rows):
                for q in range(cols):
                    if coeffs_x[p, q] != 0:
                        delta_u += coeffs_x[p, q] * u_pows[p] * v_pows[q]

        if coeffs_y is not None:
            rows, cols = coeffs_y.shape
            for p in range(rows):
                for q in range(cols):
                    if coeffs_y[p, q] != 0:
                        delta_v += coeffs_y[p, q] * u_pows[p] * v_pows[q]

        xy_out = xy.clone()
        if coeffs_x is not None:
            xy_out[:, 0] += delta_u
        if coeffs_y is not None:
            xy_out[:, 1] += delta_v

        return xy_out

    def _apply_tpv(self, xy: Tensor) -> Tensor:
        """
        Apply TPV distortion.
        Terms: 0:1, 1:x, 2:y, 3:r, 4:x^2, 5:xy, 6:y^2, ...
        """
        x = xy[:, 0]
        # TPV convention: eta axis is inverted relative to standard FITS Y (if derived from CD)
        # Empirical testing vs Astropy confirms y should be inverted.
        y = -xy[:, 1]
        r = torch.sqrt(x * x + y * y)

        # Precompute terms up to needed power (usually low order)
        # We'll implement terms 0-11 commonly used.
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

        terms = {
            0: torch.ones_like(x),
            1: x,
            2: y,
            3: r,
            4: x * x,
            5: x * y,
            6: y * y,
            7: x * x * x,
            8: x * x * y,
            9: x * y * y,
            10: y * y * y,
            11: r * r * r,
        }

        # TPV/PV is a REPLACEMENT, so we start from 0.
        # (Unlike SIP which is additive u + f(u))
        x_new = torch.zeros_like(x)
        y_new = torch.zeros_like(y)

        # Apply X coeffs
        if self._pv_x:
            for k, val in self._pv_x.items():
                if k in terms:
                    x_new += val * terms[k]

        # Apply Y coeffs
        if self._pv_y:
            for k, val in self._pv_y.items():
                if k in terms:
                    y_new += val * terms[k]

        return torch.stack([x_new, y_new], dim=-1)

    def pixel_to_world(
        self, pixels: Tensor, batch_size: Optional[int] = None
    ) -> Tensor:
        """
        Transform pixel coordinates (0-based) to world coordinates.
        Supports TAN and SIP.
        """
        if batch_size is not None and pixels.shape[0] > batch_size:
            return self._wcs._batch_process(self.pixel_to_world, pixels, batch_size)

        # Check basic support conditions first
        if self.naxis != 2:
            # Move to CPU for wcslib
            pixels_cpu = pixels.cpu()
            # Use the parsing-only WCS object which still has the wcslib pointers
            return self._wcs.pixel_to_world(pixels_cpu).to(pixels.device)

        # Fallback to wcslib (CPU) for non-standard cases
        ctype1 = self._ctype[0]
        ctype2 = self._ctype[1]

        is_tan_or_sip = (ctype1.endswith("TAN") or ctype1.endswith("TAN-SIP")) and (
            ctype2.endswith("TAN") or ctype2.endswith("TAN-SIP")
        )

        is_tpv = self._has_tpv  # We set this in init based on CTYPE "TPV"

        is_supported = is_tan_or_sip or is_tpv

        # Check parity: Standard FITS has det < 0 (RA reversed).
        # Flipped parity (det > 0) requires different rotation logic.
        det = self._pc[0, 0] * self._pc[1, 1] - self._pc[0, 1] * self._pc[1, 0]
        det *= self._cdelt[0] * self._cdelt[1]
        is_standard_parity = det < 0.0

        if not is_supported or not is_standard_parity:
            # Move to CPU for wcslib
            pixels_cpu = pixels.cpu()
            # Use the parsing-only WCS object which still has the wcslib pointers
            return self._wcs.pixel_to_world(pixels_cpu).to(pixels.device)

        device = pixels.device
        if device != self._pc.device:
            self.to(device)

        # 1. Translation: intermediate = pixel - crpix
        # Note: pixels should be (N, 2)
        intermediate = pixels - self._crpix

        # 2. SIP Distortion (Applied BEFORE Linear)
        if self._has_sip:
            intermediate = self._apply_sip(intermediate, self._sip_a, self._sip_b)

        # 3. Linear Transformation: intermediate @ PC.T * cdelt

        # Apply PC rotation
        intermediate = intermediate @ self._pc.T
        # Apply CDELT scaling
        intermediate = intermediate * self._cdelt

        # 3.5 TPV Distortion (Applied AFTER Linear)
        if is_tpv:
            intermediate = self._apply_tpv(intermediate)

        # 4. Projection (Deprojection)
        # For TAN / TPV:
        x = -intermediate[:, 0]  # Flip X for standard parity
        y = intermediate[:, 1]

        rad_deg = 180.0 / math.pi

        # Native spherical coordinates (phi, theta) in degrees
        r = torch.sqrt(x * x + y * y)

        # Check for singularity at r=0
        # Use atan2(x, -y) to retrieve phi
        phi = torch.rad2deg(torch.atan2(x, -y))

        # theta is latitude.
        theta = torch.rad2deg(
            torch.atan2(torch.tensor(rad_deg, device=device, dtype=pixels.dtype), r)
        )

        # 5. Spherical Rotation coordinates to Celestial (RA, Dec)

        # Convert to radians
        phi_rad = torch.deg2rad(phi)
        theta_rad = torch.deg2rad(theta)

        crval1_rad = torch.deg2rad(self._crval[0])
        crval2_rad = torch.deg2rad(self._crval[1])

        costhe = torch.cos(theta_rad)
        sinthe = torch.sin(theta_rad)

        ds = torch.sin(crval2_rad)
        dc = torch.cos(crval2_rad)

        # Remove manual shift
        # diff_phi = phi_rad - torch.deg2rad(torch.tensor(self._lonpole, device=device))

        # Remove manual shift
        diff_phi = phi_rad - torch.deg2rad(torch.tensor(self._lonpole, device=device))

        phi_minus_p = diff_phi

        dec_rad = torch.asin(sinthe * ds + costhe * dc * torch.cos(phi_minus_p))

        y_term = costhe * torch.sin(phi_minus_p)
        x_term = sinthe * dc - costhe * ds * torch.cos(phi_minus_p)

        ra_minus_p = torch.atan2(y_term, x_term)
        ra_rad = torch.deg2rad(self._crval[0]) + ra_minus_p

        # Normalize RA to [0, 360)
        ra = torch.rad2deg(ra_rad) % 360.0
        dec = torch.rad2deg(dec_rad)

        return torch.stack([ra, dec], dim=-1)

    def world_to_pixel(
        self, coords: Tensor, batch_size: Optional[int] = None
    ) -> Tensor:
        """Inverse transformation: World -> Pixel."""
        if batch_size is not None and coords.shape[0] > batch_size:
            return self._wcs._batch_process(self.world_to_pixel, coords, batch_size)

        # Check basic support conditions first
        if self.naxis != 2:
            # Move to CPU for wcslib
            coords_cpu = coords.cpu()
            return self._wcs.world_to_pixel(coords_cpu).to(coords.device)

        # Fallback to wcslib (CPU) for non-standard cases
        is_tan = (
            self.naxis == 2
            and self._ctype[0].endswith("TAN")
            and self._ctype[1].endswith("TAN")
        )

        # Check parity
        det = self._pc[0, 0] * self._pc[1, 1] - self._pc[0, 1] * self._pc[1, 0]
        det *= self._cdelt[0] * self._cdelt[1]
        is_standard_parity = det < 0.0

        if not is_tan or not is_standard_parity:
            coords_cpu = coords.cpu()
            return self._wcs.world_to_pixel(coords_cpu).to(coords.device)

        # 1. Rotate Celestial -> Native
        ra = coords[:, 0]
        dec = coords[:, 1]

        ra_rad = torch.deg2rad(ra)
        dec_rad = torch.deg2rad(dec)
        crval1_rad = torch.deg2rad(self._crval[0])
        crval2_rad = torch.deg2rad(self._crval[1])
        lonpole_rad = torch.deg2rad(torch.tensor(self._lonpole, device=coords.device))

        # Inverse rotation (Paper II, Eq 5)
        diff_ra = ra_rad - crval1_rad
        ds = torch.sin(crval2_rad)
        dc = torch.cos(crval2_rad)

        sin_dec = torch.sin(dec_rad)
        cos_dec = torch.cos(dec_rad)

        theta_rad = torch.asin(sin_dec * ds + cos_dec * dc * torch.cos(diff_ra))

        y_n = cos_dec * torch.sin(diff_ra)
        x_n = sin_dec * dc - cos_dec * ds * torch.cos(diff_ra)

        phi_minus_p = torch.atan2(y_n, x_n)

        # 2. Project Native -> Intermediate (x, y)
        rad_deg = 180.0 / math.pi
        r = rad_deg / torch.tan(theta_rad)

        x = r * torch.sin(phi_minus_p)
        y = r * torch.cos(phi_minus_p)

        intermediate = torch.stack([x, y], dim=-1)

        # 3. Inverse Linear
        intermediate = intermediate / self._cdelt

        # Invert PC matrix
        pc_inv = torch.linalg.inv(self._pc)  # (2, 2)
        intermediate = intermediate @ pc_inv.T

        # 4. Inverse SIP
        if self._has_sip:
            intermediate = self._apply_sip(intermediate, self._sip_ap, self._sip_bp)

        # 5. Inverse Translation
        pixels = intermediate + self._crpix

        return pixels

    def pixel_to_world_vectorized(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Vectorized wrapper."""
        pixels = torch.stack([x.flatten(), y.flatten()], dim=1)
        world = self.pixel_to_world(pixels)
        return world[:, 0].reshape(x.shape), world[:, 1].reshape(y.shape)

    def world_to_pixel_vectorized(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Vectorized wrapper."""
        world = torch.stack([ra.flatten(), dec.flatten()], dim=1)
        pixels = self.world_to_pixel(world)
        return pixels[:, 0].reshape(ra.shape), pixels[:, 1].reshape(dec.shape)

    def benchmark_transformation(
        self, n_coords: int = 10000, device: str = "cpu"
    ) -> dict:
        """Benchmark performance."""
        import time

        pixels = torch.randn(n_coords, 2, device=device) * 1000 + 1000

        # Warmup
        self.pixel_to_world(pixels)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        world = self.pixel_to_world(pixels)
        if device == "cuda":
            torch.cuda.synchronize()
        p2w = time.perf_counter() - start

        start = time.perf_counter()
        self.world_to_pixel(world)
        if device == "cuda":
            torch.cuda.synchronize()
        w2p = time.perf_counter() - start

        return {
            "device": device,
            "n": n_coords,
            "p2w_sec": p2w,
            "w2p_sec": w2p,
            "p2w_rate": n_coords / p2w,
            "w2p_rate": n_coords / w2p,
        }
