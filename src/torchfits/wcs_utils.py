"""
Enhanced WCS utilities for torchfits
"""

import torch
from . import fits_reader_cpp


def world_to_pixel(world_coords, header):
    """
    Convert world coordinates to pixel coordinates.
    
    Args:
        world_coords (torch.Tensor or list): World coordinates
        header (dict): FITS header containing WCS information
        
    Returns:
        tuple: (pixel_coords, status) - pixel coordinates and status flags
    """
    if not isinstance(world_coords, torch.Tensor):
        world_coords = torch.tensor(world_coords, dtype=torch.float64)
    
    return fits_reader_cpp.world_to_pixel(world_coords, header)


def pixel_to_world(pixel_coords, header):
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
        pixel_coords (torch.Tensor or list): Pixel coordinates  
        header (dict): FITS header containing WCS information
        
    Returns:
        tuple: (world_coords, status) - world coordinates and status flags
    """
    if not isinstance(pixel_coords, torch.Tensor):
        pixel_coords = torch.tensor(pixel_coords, dtype=torch.float64)
        
    return fits_reader_cpp.pixel_to_world(pixel_coords, header)


def get_wcs_info(header):
    """
    Extract WCS information from header.
    
    Args:
        header (dict): FITS header
        
    Returns:
        dict: WCS information including coordinate types, units, etc.
    """
    wcs_info = {}
    
    # Get coordinate system info
    for i in range(1, 10):  # Check up to 9 dimensions
        ctype_key = f'CTYPE{i}'
        cunit_key = f'CUNIT{i}'
        crval_key = f'CRVAL{i}'
        crpix_key = f'CRPIX{i}'
        cdelt_key = f'CDELT{i}'
        
        if ctype_key in header:
            wcs_info[f'axis{i}'] = {
                'type': header[ctype_key],
                'unit': header.get(cunit_key, ''),
                'reference_value': float(header.get(crval_key, 0)),
                'reference_pixel': float(header.get(crpix_key, 1)),
                'delta': float(header.get(cdelt_key, 1))
            }
    
    return wcs_info


def is_celestial(header):
    """Check if WCS contains celestial coordinates (RA/Dec)."""
    return any('RA' in header.get(f'CTYPE{i}', '') or 'DEC' in header.get(f'CTYPE{i}', '') 
               for i in range(1, 10))


def is_spectral(header):
    """Check if WCS contains spectral coordinates (wavelength, frequency, etc.)."""
    spectral_types = ['WAVE', 'FREQ', 'ENER', 'VELO', 'VRAD', 'VOPT']
    return any(any(stype in header.get(f'CTYPE{i}', '') for stype in spectral_types)
               for i in range(1, 10))


def get_coordinate_names(header):
    """Get human-readable coordinate names."""
    names = []
    for i in range(1, 10):
        ctype = header.get(f'CTYPE{i}', '')
        if ctype:
            if 'RA' in ctype:
                names.append('Right Ascension')
            elif 'DEC' in ctype:
                names.append('Declination')
            elif 'WAVE' in ctype:
                names.append('Wavelength')
            elif 'FREQ' in ctype:
                names.append('Frequency')
            elif 'VELO' in ctype:
                names.append('Velocity')
            else:
                names.append(ctype)
    return names


def transform_cutout_wcs(header, start, shape):
    """
    Update WCS header for a cutout region.
    
    Args:
        header (dict): Original WCS header
        start (list): Starting pixel coordinates (0-based)
        shape (list): Shape of cutout
        
    Returns:
        dict: Updated header with corrected WCS for cutout
    """
    new_header = header.copy()
    
    # Update reference pixels for cutout
    for i, (s, sh) in enumerate(zip(start, shape), 1):
        crpix_key = f'CRPIX{i}'
        if crpix_key in new_header:
            # Adjust reference pixel for cutout offset (convert to 1-based)
            old_crpix = float(new_header[crpix_key])
            new_crpix = old_crpix - s
            new_header[crpix_key] = str(new_crpix)
        
        # Update NAXIS values
        naxis_key = f'NAXIS{i}'
        if naxis_key in new_header:
            new_header[naxis_key] = str(sh)
    
    return new_header
