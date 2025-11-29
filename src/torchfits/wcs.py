"""
WCS (World Coordinate System) functionality for torchfits.

This module provides efficient coordinate transformations using wcslib
with batch processing capabilities optimized for PyTorch tensors.
"""

from typing import Optional
import torch
from torch import Tensor

from .hdu import Header
import torchfits.cpp as cpp


class WCS:
    """A wrapper for the C++ WCS engine with batch coordinate transformations."""
    
    def __init__(self, header: Header):
        """
        Initialize WCS from FITS header.
        
        Args:
            header: FITS header containing WCS keywords
        """
        self._header = header
        # Add minimal required FITS headers for WCS initialization
        wcs_header = {
            'SIMPLE': 'T',
            'BITPIX': '-32',
            'NAXIS': '2',
            'NAXIS1': '100',
            'NAXIS2': '100'
        }
        # Update with provided header values
        wcs_header.update({str(k): str(v) for k, v in header.items()})
        self._wcs = cpp.WCS(wcs_header)
    
    def pixel_to_world(self, pixels: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        Transform pixel coordinates to world coordinates.
        
        Args:
            pixels: Tensor of shape (..., N) where N is the number of dimensions
                   Last dimension contains pixel coordinates (x, y, ...)
            batch_size: Optional batch size for processing large tensors
        
        Returns:
            Tensor of world coordinates with same shape as input
        """
        # Handle batching for very large tensors
        if batch_size is not None and pixels.shape[0] > batch_size:
            return self._batch_process(self._wcs.pixel_to_world, pixels, batch_size)
        
        return self._wcs.pixel_to_world(pixels)
    
    def world_to_pixel(self, coords: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        Transform world coordinates to pixel coordinates.
        
        Args:
            coords: Tensor of shape (..., N) where N is the number of dimensions
                   Last dimension contains world coordinates (ra, dec, ...)
            batch_size: Optional batch size for processing large tensors
        
        Returns:
            Tensor of pixel coordinates with same shape as input
        """
        # Handle batching for very large tensors
        if batch_size is not None and coords.shape[0] > batch_size:
            return self._batch_process(self._wcs.world_to_pixel, coords, batch_size)
            
        return self._wcs.world_to_pixel(coords)
    
    def pixel_to_world_vectorized(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Vectorized pixel to world transformation for separate x, y tensors.
        
        Args:
            x: X pixel coordinates
            y: Y pixel coordinates
            
        Returns:
            Tuple of (ra, dec) world coordinates
        """
        # Stack coordinates for batch processing
        pixels = torch.stack([x.flatten(), y.flatten()], dim=1)
        world = self.pixel_to_world(pixels)
        
        # Reshape back to original shape
        ra = world[:, 0].reshape(x.shape)
        dec = world[:, 1].reshape(y.shape)
        return ra, dec
    
    def world_to_pixel_vectorized(self, ra: Tensor, dec: Tensor) -> tuple[Tensor, Tensor]:
        """
        Vectorized world to pixel transformation for separate ra, dec tensors.
        
        Args:
            ra: Right ascension coordinates
            dec: Declination coordinates
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        # Stack coordinates for batch processing  
        world = torch.stack([ra.flatten(), dec.flatten()], dim=1)
        pixels = self.world_to_pixel(world)
        
        # Reshape back to original shape
        x = pixels[:, 0].reshape(ra.shape)
        y = pixels[:, 1].reshape(dec.shape)
        return x, y
    
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
        # Temporary workaround: extract from header
        return int(self._header.get('WCSAXES', self._header.get('NAXIS', 2)))
    
    @property
    def crpix(self) -> Tensor:
        """Reference pixel coordinates."""
        # Temporary workaround: extract from header  
        crpix1 = float(self._header.get('CRPIX1', 1.0))
        crpix2 = float(self._header.get('CRPIX2', 1.0))
        return torch.tensor([crpix1, crpix2], dtype=torch.float64)
    
    @property
    def crval(self) -> Tensor:
        """Reference world coordinates."""
        # Temporary workaround: extract from header
        crval1 = float(self._header.get('CRVAL1', 0.0))
        crval2 = float(self._header.get('CRVAL2', 0.0))
        return torch.tensor([crval1, crval2], dtype=torch.float64)
    
    @property
    def cdelt(self) -> Tensor:
        """Coordinate increments."""
        # Temporary workaround: extract from header
        cdelt1 = float(self._header.get('CDELT1', 1.0))
        cdelt2 = float(self._header.get('CDELT2', 1.0))
        return torch.tensor([cdelt1, cdelt2], dtype=torch.float64)
    
    def __repr__(self):
        return f"WCS(naxis={self.naxis}, crpix={self.crpix.tolist()}, crval={self.crval.tolist()})"
    
    def _batch_process(self, func, tensor: Tensor, batch_size: int) -> Tensor:
        """
        Process large tensors in batches to manage memory usage.
        
        Args:
            func: Transformation function to apply
            tensor: Input tensor to process
            batch_size: Size of each batch
            
        Returns:
            Processed tensor
        """
        device = tensor.device
        total_coords = tensor.shape[0]
        
        # Pre-allocate result tensor
        result = torch.empty_like(tensor)
        
        # Process in batches
        for i in range(0, total_coords, batch_size):
            end_idx = min(i + batch_size, total_coords)
            batch = tensor[i:end_idx]
            
            # Apply transformation
            batch_result = func(batch)
            result[i:end_idx] = batch_result
        
        return result
    
    def to_gpu(self) -> 'WCS':
        """
        Move WCS computations to GPU if available.
        
        Returns:
            WCS object optimized for GPU operations
        """
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, staying on CPU")
            return self
            
        # The C++ implementation automatically detects GPU tensors
        # and uses optimized kernels when available
        return self
    
    def benchmark_transformation(self, n_coords: int = 10000, device: str = 'cpu') -> dict:
        """
        Benchmark coordinate transformation performance.
        
        Args:
            n_coords: Number of coordinates to transform
            device: Device to run benchmark on ('cpu' or 'cuda')
            
        Returns:
            Dictionary with timing results
        """
        import time
        
        # Generate test coordinates
        pixels = torch.randn(n_coords, 2, device=device) * 1000 + 1000
        
        # Warm-up
        for _ in range(3):
            _ = self.pixel_to_world(pixels)
        
        # Benchmark pixel to world
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        world_coords = None
        for _ in range(10):
            world_coords = self.pixel_to_world(pixels)
            
        if device == 'cuda':
            torch.cuda.synchronize()
        p2w_time = time.perf_counter() - start_time
        
        # Benchmark world to pixel  
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        if world_coords is not None:
            for _ in range(10):
                pixel_coords = self.world_to_pixel(world_coords)
                
        if device == 'cuda':
            torch.cuda.synchronize()
        w2p_time = time.perf_counter() - start_time
        
        return {
            'device': device,
            'n_coords': n_coords,
            'pixel_to_world_time': p2w_time,
            'world_to_pixel_time': w2p_time,
            'coords_per_second_p2w': n_coords * 10 / p2w_time,
            'coords_per_second_w2p': n_coords * 10 / w2p_time
        }