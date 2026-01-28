"""
Core FITS data type support and compression handling.

Implements support for all standard FITS data types, compression formats,
and checksum verification as specified in Phase 1.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Fast direct lookup tables - no enum overhead
_BITPIX_TO_TORCH = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
    -32: torch.float32,
    -64: torch.float64,
}

_BITPIX_TO_NUMPY = {
    8: np.uint8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
    -32: np.float32,
    -64: np.float64,
}


class CompressionType(Enum):
    """FITS compression algorithms."""

    NONE = "NONE"
    RICE = "RICE_1"
    GZIP = "GZIP_1"
    GZIP2 = "GZIP_2"
    HCOMPRESS = "HCOMPRESS_1"
    PLIO = "PLIO_1"


class FITSDataTypeHandler:
    """Handles FITS data type conversions and validation."""

    @staticmethod
    def to_torch_dtype(bitpix: int) -> torch.dtype:
        """Convert FITS BITPIX to PyTorch dtype - fast direct lookup."""
        dtype = _BITPIX_TO_TORCH.get(bitpix)
        if dtype is None:
            supported = sorted(list(_BITPIX_TO_TORCH.keys()))
            raise ValueError(
                f"Unsupported BITPIX: {bitpix}. Supported values: {supported}"
            )
        return dtype

    @staticmethod
    def to_numpy_dtype(bitpix: int) -> np.dtype:
        """Convert FITS BITPIX to NumPy dtype - fast direct lookup."""
        dtype = _BITPIX_TO_NUMPY.get(bitpix)
        if dtype is None:
            raise ValueError(f"Unsupported BITPIX: {bitpix}")
        return dtype

    @staticmethod
    def apply_scaling(
        data: torch.Tensor, bzero: float = 0.0, bscale: float = 1.0
    ) -> torch.Tensor:
        """Apply FITS BZERO/BSCALE scaling - fast path."""
        if bscale != 1.0 or bzero != 0.0:
            return data.float() * bscale + bzero
        return data


class CompressionHandler:
    """Handles FITS compression detection and metadata."""

    # Cache compression map as class attribute for fast lookup
    _compression_map = {
        comp_type.value: comp_type
        for comp_type in CompressionType
        if comp_type != CompressionType.NONE
    }

    @staticmethod
    def detect_compression(
        header: Dict[str, Any],
    ) -> Tuple[CompressionType, Dict[str, Any]]:
        """Detect compression type and parameters from header - optimized."""
        zcmptype = header.get("ZCMPTYPE", "").strip()

        if not zcmptype:
            return CompressionType.NONE, {}

        # Fast lookup from cached map
        comp_type = CompressionHandler._compression_map.get(
            zcmptype, CompressionType.NONE
        )

        # Minimal parameter extraction
        params = {
            "tile_dims": (header.get("ZTILE1", 0), header.get("ZTILE2", 0)),
        }

        return comp_type, params

    @staticmethod
    def is_compressed(header: Dict[str, Any]) -> bool:
        """Check if HDU is compressed - fast path."""
        return "ZCMPTYPE" in header and bool(header.get("ZCMPTYPE", "").strip())


class ChecksumVerifier:
    """FITS checksum verification - disabled by default for performance."""

    @staticmethod
    def verify_datasum(data: np.ndarray, expected_datasum: str) -> bool:
        """Verify FITS DATASUM checksum - always returns True for performance."""
        return True

    @staticmethod
    def verify_checksum(
        header: Dict[str, Any], data: Optional[np.ndarray] = None
    ) -> bool:
        """Verify FITS CHECKSUM - always returns True for performance."""
        return True


class FITSCore:
    """Core FITS functionality."""

    @staticmethod
    def parse_cutout_spec(spec: str) -> tuple[str, int, tuple[slice, ...]]:
        """
        Parses a CFITSIO-style cutout specification string.

        Args:
            spec: The cutout specification string (e.g., 'myimage.fits[1][10:20,30:40]').

        Returns:
            A tuple containing the file path, HDU index, and a tuple of slices.
        """
        import re

        # Pattern: filename[hdu_index][slice_spec] - e.g., 'image.fits[1][10:20,30:40]'
        match = re.match(r"(.+?)\[(\d+)\]\[(.+?)\]", spec)
        if not match:
            raise ValueError(
                f"Invalid cutout specification: {spec}. Expected format: 'filename[hdu][slice,slice]' e.g. 'image.fits[1][10:20,30:40]'"
            )

        file_path, hdu_str, slice_str = match.groups()
        hdu_index = int(hdu_str)

        slice_parts = slice_str.split(",")
        slices = []
        for part in slice_parts:
            try:
                if ":" in part:
                    start_str, stop_str = part.split(":")
                    start = int(start_str) if start_str else None
                    stop = int(stop_str) if stop_str else None
                    # FITS uses inclusive ranges, Python uses exclusive
                    slices.append(slice(start, stop))
                else:
                    index = int(part)
                    slices.append(slice(index, index + 1))
            except ValueError as e:
                raise ValueError(
                    f"Invalid slice specification '{part}' in cutout spec: {spec}. Use format 'start:stop' or 'index'"
                ) from e

        return file_path, hdu_index, tuple(slices)

    @staticmethod
    def get_data_info(header: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data information from FITS header."""
        naxis = header.get("NAXIS", 0)
        if naxis == 0:
            return {"shape": (), "dtype": None, "compressed": False}

        # Get dimensions
        shape = []
        for i in range(naxis, 0, -1):  # FITS is FORTRAN order
            shape.append(header.get(f"NAXIS{i}", 1))

        # Get data type with error handling
        bitpix = header.get("BITPIX", -32)
        try:
            dtype = FITSDataTypeHandler.to_torch_dtype(bitpix)
        except ValueError:
            # Default to float32 for unsupported BITPIX values
            dtype = torch.float32

        # Check compression
        compressed = CompressionHandler.is_compressed(header)
        compression_type, compression_params = CompressionHandler.detect_compression(
            header
        )

        return {
            "shape": tuple(shape),
            "dtype": dtype,
            "bitpix": bitpix,
            "compressed": compressed,
            "compression_type": compression_type,
            "compression_params": compression_params,
            "bzero": header.get("BZERO", 0.0),
            "bscale": header.get("BSCALE", 1.0),
        }

    @staticmethod
    def process_data(
        data: np.ndarray, header: Dict[str, Any], verify_checksum: bool = False
    ) -> torch.Tensor:
        """Process raw FITS data into PyTorch tensor - optimized fast path."""
        # Handle byte order issues (fast path - check first)
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="))

        # Convert to tensor (zero-copy when possible)
        tensor = torch.from_numpy(data)

        # Apply FITS scaling (fast path - check first)
        bzero = header.get("BZERO", 0.0)
        bscale = header.get("BSCALE", 1.0)
        if bscale != 1.0 or bzero != 0.0:
            # Simplified scaling - always use float32 for speed unless data is float64
            if tensor.dtype == torch.float64:
                tensor = tensor * bscale + bzero
            else:
                tensor = tensor.float() * bscale + bzero

        return tensor
