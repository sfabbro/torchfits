import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, Any, List
import re


def parse_wat_keywords(header: Dict[str, Any], axis: int) -> str:
    """
    Reconstruct the full WAT string for a given axis.
    IRAF splits long strings into WATi_nnn keywords.
    """
    # Collect all WATi_nnn values
    # nnn usually starts at 001
    full_str = ""
    # Check for WATi_001, WATi_002, ...
    # Limit to reasonable number to avoid infinite loops
    for n in range(1, 100):
        key = f"WAT{axis}_{n:03d}"
        if key in header:
            # WAT values often are strings like 'wtype=tnx a_order=...'
            # They are concatenated directly.
            val = str(header[key])
            # Strip quotes if present?
            # FITS string values might have padding.
            # Usually strict concatenation.
            # But header dict values from astropy/fits usually stripped of quotes.
            # However, length limit is 68 chars per card.
            full_str += val
        else:
            break

    return full_str


class LegacyPolynomial:
    """
    Base class for IRAF polynomials (Chebyshev, Legendre, etc.)

    Format string example:
    "wtype=tnx a_order=... " ? No.
    TNX format within WAT:
    "dtype1 = 1  2  1  1.0 1.0  0.5 0.5 ..."

    Structure:
    function_type (1=Chebyshev, 2=Legendre, 3=Polynomial/Simple)
    order_x (int)
    order_y (int)
    cross_terms (0=no, 1=yes, 2=half?)
    x_min (float)
    x_max (float)
    y_min (float)
    y_max (float)
    coefficients...

    """

    def __init__(self, param_str: str):
        self.func_type = 0
        self.order_x = 0
        self.order_y = 0
        self.cross_terms = 0
        self.x_bounds = (0.0, 0.0)
        self.y_bounds = (0.0, 0.0)
        self.coeffs = []

        self._parse(param_str)

    def _parse(self, s: str):
        # s is the full WAT string, e.g. 'wtype=tnx lngcor = "1 2 3..." latcor = "..."'
        # We need to find the specific component this polynomial represents.
        # But wait, LegacyPolynomial is initialized with just the specific attribute string?
        # NO, we pass the generic string. We need to know WHICH attribute to look for.
        # Let's change __init__ to accept the specific value string,
        # OR generic string + key.

        # Actually, let's keep it simple: caller extracts the substring.

        # Split by whitespace
        tokens = s.strip().split()
        if not tokens:
            return

        try:
            # Token 0 might be func_type (1,2,3)
            self.func_type = int(tokens[0])
            self.order_x = int(tokens[1])
            self.order_y = int(tokens[2])
            self.cross_terms = int(tokens[3])

            # Bounds
            self.x_bounds = (float(tokens[4]), float(tokens[5]))
            self.y_bounds = (float(tokens[6]), float(tokens[7]))

            # Coefficients
            self.coeffs = [float(x) for x in tokens[8:]]

        except (ValueError, IndexError) as e:
            # Malformed string
            print(f"Warning: Failed to parse Legacy Polynomial string: {e}")

    def evaluate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate polynomial P(x, y).

        Chebyshev/Legendre require normalizing x, y to [-1, 1].
        """
        # 1. Normalize
        # bounds: x_min, x_max
        nx = (2.0 * x - (self.x_bounds[1] + self.x_bounds[0])) / (
            self.x_bounds[1] - self.x_bounds[0]
        )
        ny = (2.0 * y - (self.y_bounds[1] + self.y_bounds[0])) / (
            self.y_bounds[1] - self.y_bounds[0]
        )

        # 2. Basis Functions
        # Precompute basis polys up to order
        # func_type 1: Chebyshev
        # func_type 2: Legendre

        bx = self._compute_basis(nx, self.order_x, self.func_type)
        by = self._compute_basis(ny, self.order_y, self.func_type)

        # 3. Summation
        result = torch.zeros_like(x)
        k = 0

        for j in range(self.order_y):
            for i in range(self.order_x):
                # Check cross term validity
                if self.cross_terms == 0 and i > 0 and j > 0:
                    continue
                if self.cross_terms == 2 and i + j >= max(self.order_x, self.order_y):
                    continue

                # Conservative: use all coeffs provided
                if k < len(self.coeffs):
                    term = self.coeffs[k] * bx[i] * by[j]
                    result += term
                    k += 1

        return result

    def _compute_basis(self, u: Tensor, order: int, ftype: int) -> List[Tensor]:
        """Generate basis polynomials [P_0(u), ..., P_n(u)]."""
        basis = []
        if order < 1:
            return basis

        # P_0 = 1
        basis.append(torch.ones_like(u))
        if order == 1:
            return basis

        # P_1
        # Chebyshev: x, Legendre: x
        basis.append(u)

        # Recurrence
        for n in range(2, order):
            if ftype == 1:  # Chebyshev T_n = 2x T_{n-1} - T_{n-2}
                bn = 2.0 * u * basis[-1] - basis[-2]
            elif ftype == 2:  # Legendre P_n = ((2n-1)x P_{n-1} - (n-1)P_{n-2}) / n
                k = n
                bn = ((2 * k - 1) * u * basis[-1] - (k - 1) * basis[-2]) / k
            else:
                # Polynomial (simple powers)
                bn = u * basis[-1]

            basis.append(bn)

        return basis


def extract_tnx_coeffs(wat_str: str, key: str) -> Optional[str]:
    """
    Extract value for 'lngcor' or 'latcor' from WAT string.
    Format: key = " value " or key = value
    """
    # Regex to find key = "..." or key = value
    # key e.g. 'lngcor'
    # Look for: lngcor\s*=\s*(?:"([^"]*)"|([^"\s]+))

    pattern = re.compile(rf'{key}\s*=\s*(?:"([^"]*)"|([^"\s]+))')
    match = pattern.search(wat_str)
    if match:
        val_quoted = match.group(1)
        val_unquoted = match.group(2)
        return val_quoted if val_quoted is not None else val_unquoted
    return None


def project_tnx(
    xi: Tensor,
    eta: Tensor,
    params: Optional[Dict[str, float]] = None,
    wat_data: Optional[Dict[int, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    TNX Projection.
    xi, eta: Intermediate coordinates (degrees) from TAN projection.
    """
    # params contain WCS keywords.
    # wat_data is dict: {axis: full_wat_string}

    if wat_data is None:
        # Cannot compute without parsed WAT strings
        return xi, eta

    xi_out = xi.clone()
    eta_out = eta.clone()

    # Apply X correction (lngcor)
    if 1 in wat_data:
        wat1 = wat_data[1]
        poly_str = extract_tnx_coeffs(wat1, "lngcor")
        if poly_str:
            poly_x = LegacyPolynomial(poly_str)
            correction = poly_x.evaluate(xi, eta)
            xi_out += correction

    # Apply Y correction (latcor)
    if 2 in wat_data:
        wat2 = wat_data[2]
        poly_str = extract_tnx_coeffs(wat2, "latcor")
        if poly_str:
            poly_y = LegacyPolynomial(poly_str)
            correction = poly_y.evaluate(xi, eta)
            eta_out += correction

    return xi_out, eta_out


def extract_zpx_params(wat_data: Dict[int, str]) -> Dict[str, float]:
    """
    Extract ZPN coefficients (projp_i) from WAT strings and return as PV2_i.
    """
    params = {}

    # ZPX coefficients usually appear in WAT0_001 or similar,
    # OR distributed across WAT1/WAT2.
    # We should search all available WAT strings.

    # Regex for projp_n = value
    # e.g. "projp1 = 0.5"
    # Case insensitive?

    pattern = re.compile(r"projp(\d+)\s*=\s*(\S+)", re.IGNORECASE)

    for axis, wstr in wat_data.items():
        matches = pattern.findall(wstr)
        for m in matches:
            idx = int(m[0])
            val = float(m[1])
            params[f"PV2_{idx}"] = val

    return params


def project_zpx(
    xi: Tensor,
    eta: Tensor,
    params: Optional[Dict[str, float]] = None,
    wat_data: Optional[Dict[int, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    ZPX: Zenithal Poly + WAT corrections.

    1. Apply WAT polynomial distortions (same as TNX).
    2. Convert to (phi, theta) using ZPN projection with coeffs from WAT.
    """
    from .zenithal import project_zenithal

    if wat_data is None:
        return project_zenithal(xi, eta, "ZPN", params)

    # 1. Distortions (Poly to xi, eta)
    # Identical structure to TNX?
    # IRAF ZPX: "The forward transformation... first applies the polynomial corrections... then the standard projection."
    # Yes.

    xi_out = xi.clone()
    eta_out = eta.clone()

    # Apply X correction (lngcor)
    if 1 in wat_data:
        wat1 = wat_data[1]
        poly_str = extract_tnx_coeffs(wat1, "lngcor")
        if poly_str:
            poly_x = LegacyPolynomial(poly_str)
            correction = poly_x.evaluate(xi, eta)
            xi_out += correction

    # Apply Y correction (latcor)
    if 2 in wat_data:
        wat2 = wat_data[2]
        poly_str = extract_tnx_coeffs(wat2, "latcor")
        if poly_str:
            poly_y = LegacyPolynomial(poly_str)
            correction = poly_y.evaluate(xi, eta)
            eta_out += correction

    # 2. ZPN Projection
    # We need to inject 'projp_n' values from WAT into params as 'PV2_n'.
    zpn_params = params.copy() if params else {}

    # Extract projp
    extra_params = extract_zpx_params(wat_data)
    zpn_params.update(extra_params)

    # Call ZPN
    phi, theta = project_zenithal(xi_out, eta_out, "ZPN", zpn_params)

    return phi, theta
