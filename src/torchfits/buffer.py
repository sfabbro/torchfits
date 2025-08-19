"""
Zero-copy buffer management for torchfits.

Implements FITSManagedBuffer for efficient memory management and zero-copy
tensor operations as specified in Phase 1.
"""

import torch
import numpy as np
import weakref
from typing import Optional, Any, Callable
import threading

class FITSManagedBuffer:
    """
    Managed buffer for zero-copy tensor operations.
    
    Uses shared_ptr-like reference counting for thread-safe memory management.
    """
    
    def __init__(self, data: np.ndarray, cleanup_callback: Optional[Callable] = None):
        """
        Initialize managed buffer.
        
        Args:
            data: NumPy array containing the data
            cleanup_callback: Optional callback for cleanup when buffer is released
        """
        self._data = data
        self._cleanup_callback = cleanup_callback
        self._ref_count = 0  # Start with 0, increment when tensors are created
        self._lock = threading.RLock()
        self._released = False
    
    def get_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """
        Create zero-copy PyTorch tensor from buffer.
        
        Args:
            device: Target device for tensor
            
        Returns:
            PyTorch tensor sharing memory with buffer
        """
        with self._lock:
            if self._released:
                raise RuntimeError("Buffer has been released")
            
            # Create tensor with shared memory
            tensor = torch.from_numpy(self._data)
            
            # Keep reference to buffer to prevent cleanup
            tensor._fits_buffer_ref = self
            self._add_ref()
            
            # Add finalizer to release reference when tensor is deleted
            weakref.finalize(tensor, self._release_ref)
            
            # Move to device if needed (breaks zero-copy but necessary)
            if device != 'cpu':
                import warnings
                warnings.warn("Moving tensor to GPU breaks zero-copy optimization", UserWarning)
                tensor = tensor.to(device)
            
            return tensor
    
    def _add_ref(self):
        """Add reference to buffer."""
        with self._lock:
            if not self._released:
                self._ref_count += 1
    
    def _release_ref(self):
        """Release reference to buffer."""
        with self._lock:
            if self._released:
                return
            
            self._ref_count -= 1
            if self._ref_count <= 0:
                self._cleanup()
    
    def _cleanup(self):
        """Cleanup buffer resources."""
        if self._released:
            return
        
        self._released = True
        
        if self._cleanup_callback:
            try:
                self._cleanup_callback()
            except Exception as e:
                import logging
                logging.warning(f"Buffer cleanup failed: {e}")
        
        # Clear data reference
        self._data = None
    
    def release(self):
        """Explicitly release buffer."""
        self._release_ref()
    
    @property
    def shape(self):
        """Get buffer shape."""
        return self._data.shape if self._data is not None else ()
    
    @property
    def dtype(self):
        """Get buffer dtype."""
        return self._data.dtype if self._data is not None else None
    
    @property
    def size(self):
        """Get buffer size in bytes."""
        return self._data.nbytes if self._data is not None else 0

class BufferPool:
    """
    Memory pool for efficient buffer allocation.
    
    Reduces allocation overhead for frequent small allocations.
    """
    
    def __init__(self, initial_size: int = 1024 * 1024):  # 1MB default
        self.initial_size = initial_size
        self.pools = {}  # size -> list of available buffers
        self.lock = threading.RLock()
    
    def get_buffer(self, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Get buffer from pool or allocate new one.
        
        Args:
            size: Buffer size in elements
            dtype: NumPy dtype
            
        Returns:
            NumPy array buffer
        """
        buffer_key = (size, dtype)
        
        with self.lock:
            if buffer_key in self.pools and self.pools[buffer_key]:
                return self.pools[buffer_key].pop()
            else:
                # Allocate new buffer
                return np.empty(size, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """
        Return buffer to pool for reuse.
        
        Args:
            buffer: Buffer to return
        """
        buffer_key = (buffer.size, buffer.dtype)
        
        with self.lock:
            if buffer_key not in self.pools:
                self.pools[buffer_key] = []
            
            # Limit pool size to prevent memory bloat
            if len(self.pools[buffer_key]) < 10:
                self.pools[buffer_key].append(buffer)
    
    def clear(self):
        """Clear all pools."""
        with self.lock:
            self.pools.clear()

class SIMDConverter:
    """
    SIMD-optimized data type conversions.
    
    Placeholder for SIMD optimizations - would use compiler intrinsics
    or vectorized operations in full implementation.
    """
    
    @staticmethod
    def convert_with_scaling(data: np.ndarray, bzero: float = 0.0, bscale: float = 1.0, 
                           target_dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Convert data with FITS scaling using vectorized operations.
        
        Args:
            data: Input data array
            bzero: FITS BZERO parameter
            bscale: FITS BSCALE parameter
            target_dtype: Target data type
            
        Returns:
            Converted and scaled array
        """
        if bscale == 1.0 and bzero == 0.0 and data.dtype == target_dtype:
            return data
        
        # Use vectorized operations (would be SIMD in C++)
        if bscale != 1.0 or bzero != 0.0:
            # Convert to float for scaling
            result = data.astype(np.float64)
            if bscale != 1.0:
                result *= bscale
            if bzero != 0.0:
                result += bzero
            
            # Convert to target type
            if target_dtype != np.float64:
                result = result.astype(target_dtype)
        else:
            result = data.astype(target_dtype)
        
        return result
    
    @staticmethod
    def byte_swap_if_needed(data: np.ndarray, native_endian: bool = True) -> np.ndarray:
        """
        Byte swap data if endianness doesn't match system.
        
        Args:
            data: Input data array
            native_endian: Whether data is in native endianness
            
        Returns:
            Data with correct endianness
        """
        if not native_endian and data.dtype.byteorder not in ('=', '|'):
            return data.byteswap().newbyteorder()
        return data

# Global buffer pool
_buffer_pool = BufferPool()

def get_buffer_pool() -> BufferPool:
    """Get global buffer pool."""
    return _buffer_pool

def create_managed_buffer(data: np.ndarray, cleanup_callback: Optional[Callable] = None) -> FITSManagedBuffer:
    """
    Create managed buffer for zero-copy operations.
    
    Args:
        data: NumPy array data
        cleanup_callback: Optional cleanup callback
        
    Returns:
        Managed buffer instance
    """
    return FITSManagedBuffer(data, cleanup_callback)