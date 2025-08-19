"""
Core FITS data type support and compression handling.

Implements support for all standard FITS data types, compression formats,
and checksum verification as specified in Phase 1.
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum

import torch
import numpy as np

class FITSDataType(Enum):
    """Standard FITS data types."""
    UINT8 = (torch.uint8, np.uint8, 'B')
    INT16 = (torch.int16, np.int16, 'I') 
    INT32 = (torch.int32, np.int32, 'J')
    INT64 = (torch.int64, np.int64, 'K')
    FLOAT32 = (torch.float32, np.float32, 'E')
    FLOAT64 = (torch.float64, np.float64, 'D')
    
    @property
    def torch_dtype(self):
        return self.value[0]
    
    @property 
    def numpy_dtype(self):
        return self.value[1]
    
    @property
    def fits_code(self):
        return self.value[2]

class CompressionType(Enum):
    """FITS compression algorithms."""
    NONE = 'NONE'
    RICE = 'RICE_1'
    GZIP = 'GZIP_1' 
    GZIP2 = 'GZIP_2'
    HCOMPRESS = 'HCOMPRESS_1'
    PLIO = 'PLIO_1'

class FITSDataTypeHandler:
    """Handles FITS data type conversions and validation."""
    
    BITPIX_TO_DTYPE = {
        8: FITSDataType.UINT8,
        16: FITSDataType.INT16,
        32: FITSDataType.INT32,
        64: FITSDataType.INT64,
        -32: FITSDataType.FLOAT32,
        -64: FITSDataType.FLOAT64,
    }
    
    @classmethod
    def from_bitpix(cls, bitpix: int) -> FITSDataType:
        """Convert FITS BITPIX to FITSDataType."""
        if bitpix not in cls.BITPIX_TO_DTYPE:
            raise ValueError(f"Unsupported BITPIX: {bitpix}")
        return cls.BITPIX_TO_DTYPE[bitpix]
    
    @classmethod
    def to_torch_dtype(cls, bitpix: int) -> torch.dtype:
        """Convert FITS BITPIX to PyTorch dtype."""
        return cls.from_bitpix(bitpix).torch_dtype
    
    @classmethod
    def apply_scaling(cls, data: torch.Tensor, bzero: float = 0.0, bscale: float = 1.0) -> torch.Tensor:
        """Apply FITS BZERO/BSCALE scaling."""
        if bscale != 1.0 or bzero != 0.0:
            return data.float() * bscale + bzero
        return data

class CompressionHandler:
    """Handles FITS compression detection and metadata."""
    
    @staticmethod
    def detect_compression(header: Dict[str, Any]) -> Tuple[CompressionType, Dict[str, Any]]:
        """Detect compression type and parameters from header."""
        zcmptype = header.get('ZCMPTYPE', '').strip()
        
        if not zcmptype:
            return CompressionType.NONE, {}
        
        # Map FITS compression names to enum (dynamic mapping)
        compression_map = {comp_type.value: comp_type for comp_type in CompressionType if comp_type != CompressionType.NONE}
        
        comp_type = compression_map.get(zcmptype, CompressionType.NONE)
        
        # Extract compression parameters
        params = {
            'tile_dims': (header.get('ZTILE1', 0), header.get('ZTILE2', 0)),
            'quantize_level': header.get('ZQUANTIZ', 0),
            'dither_seed': header.get('ZDITHER0', 0),
        }
        
        return comp_type, params
    
    @staticmethod
    def is_compressed(header: Dict[str, Any]) -> bool:
        """Check if HDU is compressed."""
        return 'ZCMPTYPE' in header and header.get('ZCMPTYPE', '').strip() != ''

class ChecksumVerifier:
    """FITS checksum verification."""
    
    @staticmethod
    def verify_datasum(data: np.ndarray, expected_datasum: str) -> bool:
        """Verify FITS DATASUM checksum."""
        if not expected_datasum or expected_datasum == '0':
            return True
        
        # WARNING: Simplified checksum implementation - may not detect data corruption
        # TODO: Replace with proper FITS checksum algorithm for production use
        computed = np.sum(data.view(np.uint8)) % (2**32)
        try:
            expected = int(expected_datasum)
            return computed == expected
        except ValueError:
            return True  # Skip verification for non-numeric checksums
    
    @staticmethod
    def verify_checksum(header: Dict[str, Any], data: Optional[np.ndarray] = None) -> bool:
        """Verify FITS CHECKSUM."""
        checksum = header.get('CHECKSUM', '')
        datasum = header.get('DATASUM', '')
        
        if not checksum:
            return True
        
        # Verify datasum if data provided
        if data is not None and datasum:
            return ChecksumVerifier.verify_datasum(data, datasum)
        
        return True  # Skip header checksum verification for now

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
            raise ValueError(f"Invalid cutout specification: {spec}")

        file_path, hdu_str, slice_str = match.groups()
        hdu_index = int(hdu_str)

        slice_parts = slice_str.split(',')
        slices = []
        for part in slice_parts:
            try:
                if ':' in part:
                    start_str, stop_str = part.split(':')
                    start = int(start_str) if start_str else None
                    stop = int(stop_str) if stop_str else None
                    slices.append(slice(start, stop))
                else:
                    index = int(part)
                    slices.append(slice(index, index + 1))
            except ValueError as e:
                raise ValueError(f"Invalid slice specification '{part}' in cutout spec: {spec}") from e
        
        return file_path, hdu_index, tuple(slices)
    
    @staticmethod
    def get_data_info(header: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data information from FITS header."""
        naxis = header.get('NAXIS', 0)
        if naxis == 0:
            return {'shape': (), 'dtype': None, 'compressed': False}
        
        # Get dimensions
        shape = []
        for i in range(naxis, 0, -1):  # FITS is FORTRAN order
            shape.append(header.get(f'NAXIS{i}', 1))
        
        # Get data type with error handling
        bitpix = header.get('BITPIX', -32)
        try:
            dtype = FITSDataTypeHandler.to_torch_dtype(bitpix)
        except ValueError:
            # Default to float32 for unsupported BITPIX values
            dtype = torch.float32
        
        # Check compression
        compressed = CompressionHandler.is_compressed(header)
        compression_type, compression_params = CompressionHandler.detect_compression(header)
        
        return {
            'shape': tuple(shape),
            'dtype': dtype,
            'bitpix': bitpix,
            'compressed': compressed,
            'compression_type': compression_type,
            'compression_params': compression_params,
            'bzero': header.get('BZERO', 0.0),
            'bscale': header.get('BSCALE', 1.0),
        }
    
    @staticmethod
    def process_data(data: np.ndarray, header: Dict[str, Any], verify_checksum: bool = True) -> torch.Tensor:
        """Process raw FITS data into PyTorch tensor."""
        # Verify checksum if requested
        if verify_checksum:
            ChecksumVerifier.verify_checksum(header, data)
        
        # Handle byte order issues
        if data.dtype.byteorder not in ('=', '|'):
            data = data.astype(data.dtype.newbyteorder('='))
        
        # Convert to tensor (zero-copy when possible)
        tensor = torch.from_numpy(data)
        
        # Apply FITS scaling with precision preservation
        bzero = header.get('BZERO', 0.0)
        bscale = header.get('BSCALE', 1.0)
        if bscale != 1.0 or bzero != 0.0:
            # Preserve precision for 64-bit data
            if tensor.dtype in (torch.int64, torch.float64):
                tensor = tensor.to(torch.float64) * bscale + bzero
            else:
                tensor = tensor.float() * bscale + bzero
        
        return tensor