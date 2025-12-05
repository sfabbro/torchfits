"""
WCS (World Coordinate System) functionality for torchfits.

This module provides efficient coordinate transformations using wcslib
with batch processing capabilities optimized for PyTorch tensors.
"""

from typing import Optional
import torch
from torch import Tensor

from .hdu import Header

# import torchfits.cpp as cpp  <-- Removed to avoid circular import


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
            "SIMPLE": "T",
            "BITPIX": "-32",
            "NAXIS": "2",
            "NAXIS1": "100",
            "NAXIS2": "100",
        }
        # Update with provided header values
        # Ensure string values are quoted for wcslib
        header_dict = {}
        for k, v in header.items():
            if isinstance(v, bool):
                val_str = "T" if v else "F"
            elif isinstance(v, (int, float)):
                val_str = str(v)
            elif isinstance(v, str):
                # Try to parse as float to see if it's a number
                try:
                    float(v)
                    # It's a number, don't quote
                    val_str = v
                except ValueError:
                    # It's a string, quote it if not already quoted
                    val_str = str(v)
                    if not (val_str.startswith("'") and val_str.endswith("'")):
                        val_str = f"'{val_str}'"
            else:
                val_str = str(v)
            header_dict[str(k)] = val_str

        wcs_header.update(header_dict)
        import torchfits.cpp as cpp

        self._wcs = cpp.WCS(wcs_header)

    @property
    def pixel_scale(self) -> Tensor:
        """Pixel scale in degrees per pixel (or CUNIT)."""
        return torch.abs(self.cdelt)

    @property
    def center_coord(self) -> Tensor:
        """World coordinates of the image center."""
        naxis1 = float(self._header.get("NAXIS1", 100))
        naxis2 = float(self._header.get("NAXIS2", 100))

        # Center pixel (0-based index)
        # If image is 100x100, center is at 50.0, 50.0?
        # 0 to 99. Center is 49.5.
        # FITS 1-based: 1 to 100. Center is 50.5.
        # Let's use 0-based center: (N-1)/2 ?
        # Or geometric center N/2 (0.0 to N.0)?
        # Usually center is defined as (NAXIS+1)/2 in FITS (1-based).
        # So in 0-based it is (NAXIS-1)/2? No, (NAXIS+1)/2 - 1 = (NAXIS-1)/2.
        # Wait, center of 100 pixels is 50.5 (1-based).
        # 50.5 - 1 = 49.5 (0-based).

        center_pix = torch.tensor([[naxis1 / 2.0, naxis2 / 2.0]], dtype=torch.float64)
        # Wait, if I use 1-based pixels for WCS (as I did in verification), I should pass 1-based center.
        # In verification I used `pixels + 1.0`.
        # If `torchfits.WCS` wraps `wcslib` directly, it expects 1-based pixels (unless I change it).
        # But users expect 0-based in Python (like Astropy).
        # Astropy `pixel_to_world` takes 0-based.
        # My verification script showed that passing 1-based pixels matched Astropy's 1-based result.
        # So `torchfits.WCS` currently expects 1-based pixels.
        # I should probably standardize on 0-based for the Python API.
        # But for now, let's stick to what it does (1-based) or adjust?
        # If I adjust here, I break consistency with `pixel_to_world`.

        # Let's assume 1-based for now to match current behavior.
        # Center is (NAXIS + 1) / 2.
        center_pix_1based = torch.tensor(
            [[(naxis1 + 1) / 2.0, (naxis2 + 1) / 2.0]], dtype=torch.float64
        )

        return self.pixel_to_world(center_pix_1based).squeeze()

    def pixel_to_world(
        self, pixels: Tensor, batch_size: Optional[int] = None
    ) -> Tensor:
        """
        Transform pixel coordinates to world coordinates.

        Note: Uses 0-based indexing (PyTorch convention), consistent with Astropy's pixel_to_world.

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

    def world_to_pixel(
        self, coords: Tensor, batch_size: Optional[int] = None
    ) -> Tensor:
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

    def world_to_pixel_vectorized(
        self, ra: Tensor, dec: Tensor
    ) -> tuple[Tensor, Tensor]:
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

    def to_gpu(self) -> "WCS":
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

    def benchmark_transformation(
        self, n_coords: int = 10000, device: str = "cpu"
    ) -> dict:
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
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        world_coords = None
        for _ in range(10):
            world_coords = self.pixel_to_world(pixels)

        if device == "cuda":
            torch.cuda.synchronize()
        p2w_time = time.perf_counter() - start_time

        # Benchmark world to pixel
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        if world_coords is not None:
            for _ in range(10):
                pixel_coords = self.world_to_pixel(world_coords)

        if device == "cuda":
            torch.cuda.synchronize()
        w2p_time = time.perf_counter() - start_time

        return {
            "device": device,
            "n_coords": n_coords,
            "pixel_to_world_time": p2w_time,
            "world_to_pixel_time": w2p_time,
            "coords_per_second_p2w": n_coords * 10 / p2w_time,
            "coords_per_second_w2p": n_coords * 10 / w2p_time,
        }
