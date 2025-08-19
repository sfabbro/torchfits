"""
WCS (World Coordinate System) functionality for torchfits.

This module provides efficient coordinate transformations using wcslib
with batch processing capabilities optimized for PyTorch tensors.
"""

from typing import Optional
import torch
from torch import Tensor

from .hdu import Header
from . import cpp


class WCS:
    """A wrapper for the C++ WCS engine with batch coordinate transformations."""
    
    def __init__(self, header: Header):
        """
        Initialize WCS from FITS header.
        
        Args:
            header: FITS header containing WCS keywords
        """
        self._wcs = cpp.WCS(dict(header))
    
    def pixel_to_world(self, pixels: Tensor) -> Tensor:
        """
        Transform pixel coordinates to world coordinates.
        
        Args:
            pixels: Tensor of shape (..., N) where N is the number of dimensions
                   Last dimension contains pixel coordinates (x, y, ...)
        
        Returns:
            Tensor of world coordinates with same shape as input
        """
        return self._wcs.pixel_to_world(pixels)
    
    def world_to_pixel(self, coords: Tensor) -> Tensor:
        """
        Transform world coordinates to pixel coordinates.
        
        Args:
            coords: Tensor of shape (..., N) where N is the number of dimensions
                   Last dimension contains world coordinates (ra, dec, ...)
        
        Returns:
            Tensor of pixel coordinates with same shape as input
        """
        return self._wcs.world_to_pixel(coords)
    
    def footprint(self) -> Tensor:
        """
        Get corner coordinates of the image in world space.
        
        Returns:
            Tensor of shape (4, 2) containing (ra, dec) for each corner
        """
        return self._wcs.get_footprint()
    
    @property
    def naxis(self) -> int:
        """Number of WCS axes."""
        return self._wcs.naxis
    
    @property
    def crpix(self) -> Tensor:
        """Reference pixel coordinates."""
        return self._wcs.crpix
    
    @property
    def crval(self) -> Tensor:
        """Reference world coordinates."""
        return self._wcs.crval
    
    @property
    def cdelt(self) -> Tensor:
        """Coordinate increments."""
        return self._wcs.cdelt
    
    def __repr__(self):
        return f"WCS(naxis={self.naxis}, crpix={self.crpix.tolist()}, crval={self.crval.tolist()})"