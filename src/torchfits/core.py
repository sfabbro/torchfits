"""
Core FITS data type support and compression handling.

Implements support for all standard FITS data types, compression formats,
and checksum verification as specified in Phase 1.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

import re
import torch

if TYPE_CHECKING:
    import numpy as np

_CUTOUT_SPEC_RE = re.compile(r"(.+?)\[(\d+)\]\[(.+?)\]")

# Fast direct lookup tables - no enum overhead
_BITPIX_TO_TORCH = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
    -32: torch.float32,
    -64: torch.float64,
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
            raise ValueError(f"Unsupported BITPIX: {bitpix}")
        return dtype

    _BITPIX_TO_NUMPY = None

    @classmethod
    def to_numpy_dtype(cls, bitpix: int) -> "np.dtype":
        """Convert FITS BITPIX to NumPy dtype - fast direct lookup."""
        if cls._BITPIX_TO_NUMPY is None:
            import numpy as np

            cls._BITPIX_TO_NUMPY = {
                8: np.uint8,
                16: np.int16,
                32: np.int32,
                64: np.int64,
                -32: np.float32,
                -64: np.float64,
            }
        dtype = cls._BITPIX_TO_NUMPY.get(bitpix)
        if dtype is None:
            raise ValueError(f"Unsupported BITPIX: {bitpix}")
        return dtype

    @staticmethod
    def apply_scaling(
        data: torch.Tensor, bzero: float = 0.0, bscale: float = 1.0
    ) -> torch.Tensor:
        """Apply FITS BZERO/BSCALE scaling - fast path."""
        if bscale != 1.0 or bzero != 0.0:
            data = data.to(dtype=torch.float32, copy=True)
            if bscale != 1.0:
                data.mul_(bscale)
            if bzero != 0.0:
                data.add_(bzero)
            return data
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
        header: dict[str, Any],
    ) -> tuple[CompressionType, dict[str, Any]]:
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
    def is_compressed(header: dict[str, Any]) -> bool:
        """Check if HDU is compressed - fast path."""
        return "ZCMPTYPE" in header and bool(header.get("ZCMPTYPE", "").strip())


class ChecksumVerifier:
    """CFITSIO-backed FITS checksum helpers.

    FITS checksums are defined over the HDU bytes, not just an already-materialized
    ndarray. Use the file/HDU methods for exact validation.
    """

    @staticmethod
    def _validate_hdu(hdu: int) -> int:
        if isinstance(hdu, bool) or not isinstance(hdu, int):
            raise TypeError("hdu must be a non-negative integer")
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        return int(hdu)

    @staticmethod
    def write_hdu_checksums(path: str, hdu: int = 0) -> None:
        """Compute and write DATASUM/CHECKSUM for one HDU."""
        import torchfits._C as cpp

        cpp.write_hdu_checksums(str(path), ChecksumVerifier._validate_hdu(hdu))

    @staticmethod
    def verify_hdu_checksums(path: str, hdu: int = 0) -> dict[str, int | bool]:
        """Verify DATASUM/CHECKSUM for one HDU using CFITSIO semantics."""
        import torchfits._C as cpp

        datastatus, hdustatus = cpp.verify_hdu_checksums(
            str(path), ChecksumVerifier._validate_hdu(hdu)
        )
        data_i = int(datastatus)
        hdu_i = int(hdustatus)
        return {
            "datastatus": data_i,
            "hdustatus": hdu_i,
            "ok": data_i == 1 and hdu_i == 1,
        }

    @staticmethod
    def verify_datasum(data: np.ndarray, expected_datasum: str) -> bool:
        """Deprecated in-memory stub kept only to fail closed."""
        raise NotImplementedError(
            "FITS DATASUM verification requires the source HDU bytes; use "
            "ChecksumVerifier.verify_hdu_checksums(path, hdu) instead."
        )

    @staticmethod
    def verify_checksum(header: dict[str, Any], data: np.ndarray | None = None) -> bool:
        """Deprecated in-memory stub kept only to fail closed."""
        raise NotImplementedError(
            "FITS CHECKSUM verification requires the source HDU bytes; use "
            "ChecksumVerifier.verify_hdu_checksums(path, hdu) instead."
        )


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
        # Pattern: filename[hdu_index][slice_spec] - e.g., 'image.fits[1][10:20,30:40]'
        match = _CUTOUT_SPEC_RE.match(spec)
        if not match:
            raise ValueError(f"Invalid cutout specification: {spec}")

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
                    f"Invalid slice specification '{part}' in cutout spec: {spec}"
                ) from e

        return file_path, hdu_index, tuple(slices)

    @staticmethod
    def get_data_info(header: dict[str, Any]) -> dict[str, Any]:
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
        data: np.ndarray, header: dict[str, Any], verify_checksum: bool = False
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
            if tensor.dtype == torch.float64:
                tensor = tensor.clone()
            else:
                tensor = tensor.to(dtype=torch.float32, copy=True)
            if bscale != 1.0:
                tensor.mul_(bscale)
            if bzero != 0.0:
                tensor.add_(bzero)

        return tensor
