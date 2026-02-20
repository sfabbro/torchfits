
from torch import Tensor

def transform_spectral(
    freq: Tensor, 
    rest_freq: float,
    convention: str = 'RADIO'
) -> Tensor:
    """
    Spectral transformations (Paper III).
    Currently supports Frequency to Velocity.
    
    Args:
        freq: Input frequencies in Hz
        rest_freq: Rest frequency in Hz
        convention: 'RADIO', 'OPTICAL', or 'RELATIVISTIC'
    """
    c = 299792458.0 # m/s
    
    if convention.upper() == 'RADIO':
        # v = c * (1 - f/f0)
        return c * (1.0 - freq / rest_freq) / 1000.0 # km/s
    elif convention.upper() == 'OPTICAL':
        # v = c * (f0/f - 1)
        return c * (rest_freq / freq - 1.0) / 1000.0 # km/s
    elif convention.upper() == 'RELATIVISTIC':
        # v = c * (f0^2 - f^2) / (f0^2 + f^2)
        f2 = freq * freq
        f02 = rest_freq * rest_freq
        return c * (f02 - f2) / (f02 + f2) / 1000.0 # km/s
    else:
        raise ValueError(f"Unknown spectral convention: {convention}")

# Integration into WCS class would go here, dispatching for CTYPE 'FREQ' etc.
