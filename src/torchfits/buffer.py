"""
Buffer management for torchfits.

This module provides efficient buffer management for streaming datasets,
memory-mapped file access, and optimized data loading pipelines.
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None
import torch
from torch import Tensor


class MemoryPool:
    """Thread-safe memory pool for tensor buffers."""

    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._pool = OrderedDict()  # LRU cache
        self._lock = threading.RLock()
        self._current_size = 0

    def get_buffer(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cpu"
    ) -> Tensor:
        """Get a buffer from the pool or create a new one."""
        key = (shape, dtype, device)

        with self._lock:
            if key in self._pool:
                # Move to end (most recently used)
                buffer = self._pool.pop(key)
                self._pool[key] = buffer
                return buffer

            # Create new buffer
            buffer = torch.empty(shape, dtype=dtype, device=device)
            buffer_size = buffer.numel() * buffer.element_size()

            # Evict old buffers if necessary
            while self._current_size + buffer_size > self.max_size_bytes and self._pool:
                old_key, old_buffer = self._pool.popitem(last=False)
                self._current_size -= old_buffer.numel() * old_buffer.element_size()

            # Add new buffer
            self._pool[key] = buffer
            self._current_size += buffer_size

            return buffer

    def clear(self):
        """Clear the memory pool."""
        with self._lock:
            self._pool.clear()
            self._current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                "num_buffers": len(self._pool),
                "current_size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": (
                    self._current_size / self.max_size_bytes
                    if self.max_size_bytes > 0
                    else 0
                ),
            }


class StreamingBuffer:
    """Circular buffer for streaming data."""

    def __init__(self, capacity: int, item_shape: Tuple[int, ...], dtype: torch.dtype):
        self.capacity = capacity
        self.item_shape = item_shape
        self.dtype = dtype

        # Pre-allocate buffer
        full_shape = (capacity,) + item_shape
        self.buffer = torch.empty(full_shape, dtype=dtype)

        self._write_idx = 0
        self._read_idx = 0
        self._size = 0
        self._lock = threading.RLock()

    def put(self, item: Tensor, zero_copy: bool = False) -> bool:
        """Put an item into the buffer. Returns False if buffer is full."""
        with self._lock:
            if self._size >= self.capacity:
                return False

            if (
                zero_copy
                and item.is_contiguous()
                and item.shape == self.item_shape
                and item.dtype == self.dtype
            ):
                # Zero-copy: swap tensor storage
                self.buffer[self._write_idx], item = item, self.buffer[self._write_idx]
            else:
                # Standard copy
                self.buffer[self._write_idx].copy_(item)

            self._write_idx = (self._write_idx + 1) % self.capacity
            self._size += 1
            return True

    def get(self) -> Optional[Tensor]:
        """Get an item from the buffer. Returns None if buffer is empty."""
        with self._lock:
            if self._size == 0:
                return None

            item = self.buffer[self._read_idx].clone()
            self._read_idx = (self._read_idx + 1) % self.capacity
            self._size -= 1
            return item

    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return self._size >= self.capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return self._size == 0

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return self._size


class BufferManager:
    """Advanced buffer manager with multiple strategies."""

    def __init__(
        self,
        buffer_size_mb: int = 256,
        num_buffers: int = 4,
        enable_memory_pool: bool = True,
    ):
        self.buffer_size_mb = buffer_size_mb
        self.num_buffers = num_buffers
        self.enable_memory_pool = enable_memory_pool

        # Initialize memory pool
        if enable_memory_pool:
            pool_size = min(
                buffer_size_mb * num_buffers, self._get_available_memory_mb() // 4
            )
            self.memory_pool = MemoryPool(pool_size)
        else:
            self.memory_pool = None

        # Buffer registry
        self._buffers = {}
        self._buffer_usage = {}
        self._streaming_buffers = {}
        self._lock = threading.RLock()

    def _get_available_memory_mb(self) -> int:
        """Get available system memory in MB."""
        if psutil:
            return int(psutil.virtual_memory().available / (1024 * 1024))
        # Fallback to safe default (4GB) if psutil/os check fails
        return 4096

    def get_buffer(
        self, key: str, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cpu"
    ) -> Tensor:
        """Get or create a buffer for the given specifications."""
        if self.memory_pool:
            return self.memory_pool.get_buffer(shape, dtype, device)

        # Fallback to simple buffer management
        buffer_key = f"{key}_{shape}_{dtype}_{device}"

        with self._lock:
            if buffer_key not in self._buffers:
                buffer = torch.empty(shape, dtype=dtype, device=device)
                self._buffers[buffer_key] = buffer
                self._buffer_usage[buffer_key] = 0

            self._buffer_usage[buffer_key] += 1
            return self._buffers[buffer_key]

    def create_streaming_buffer(
        self, key: str, capacity: int, item_shape: Tuple[int, ...], dtype: torch.dtype
    ) -> StreamingBuffer:
        """Create a streaming buffer for continuous data flow."""
        with self._lock:
            if key in self._streaming_buffers:
                return self._streaming_buffers[key]

            buffer = StreamingBuffer(capacity, item_shape, dtype)
            self._streaming_buffers[key] = buffer
            return buffer

    def release_buffer(
        self, key: str, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cpu"
    ):
        """Release a buffer (decrease usage count)."""
        if self.memory_pool:
            # Memory pool handles its own lifecycle
            return

        buffer_key = f"{key}_{shape}_{dtype}_{device}"

        with self._lock:
            if buffer_key in self._buffer_usage:
                self._buffer_usage[buffer_key] -= 1

                # Clean up unused buffers
                if self._buffer_usage[buffer_key] <= 0:
                    del self._buffers[buffer_key]
                    del self._buffer_usage[buffer_key]

    def clear_buffers(self):
        """Clear all buffers."""
        with self._lock:
            if self.memory_pool:
                self.memory_pool.clear()

            self._buffers.clear()
            self._buffer_usage.clear()
            self._streaming_buffers.clear()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        with self._lock:
            stats = {
                "num_buffers": len(self._buffers),
                "num_streaming_buffers": len(self._streaming_buffers),
                "buffer_size_mb": self.buffer_size_mb,
                "num_buffers_config": self.num_buffers,
                "memory_pool_enabled": self.enable_memory_pool,
            }

            if self.memory_pool:
                stats["memory_pool"] = self.memory_pool.get_stats()

            # Calculate total memory usage
            total_bytes = 0
            for buf in self._buffers.values():
                total_bytes += buf.numel() * buf.element_size()

            for stream_buf in self._streaming_buffers.values():
                total_bytes += (
                    stream_buf.buffer.numel() * stream_buf.buffer.element_size()
                )

            stats["total_memory_mb"] = total_bytes / (1024 * 1024)
            stats["available_memory_mb"] = self._get_available_memory_mb()

            return stats

    def optimize_for_workload(
        self, expected_file_size_mb: float, concurrent_files: int
    ):
        """Optimize buffer configuration for expected workload."""
        # Calculate optimal buffer size
        total_memory_needed = expected_file_size_mb * concurrent_files
        available_memory = self._get_available_memory_mb()

        # Use up to 50% of available memory
        max_memory = available_memory * 0.5

        if total_memory_needed <= max_memory:
            # Can fit everything in memory
            self.buffer_size_mb = int(expected_file_size_mb * 1.2)  # 20% overhead
            self.num_buffers = concurrent_files + 2  # Extra buffers for prefetch
        else:
            # Need to be more conservative
            self.buffer_size_mb = int(max_memory / (concurrent_files + 2))
            self.num_buffers = concurrent_files + 1

        # Recreate memory pool with new settings
        if self.enable_memory_pool:
            pool_size = min(
                self.buffer_size_mb * self.num_buffers, available_memory // 4
            )
            self.memory_pool = MemoryPool(pool_size)


# Global buffer manager
_global_buffer_manager = None


def get_buffer_manager() -> BufferManager:
    """Get the global buffer manager instance."""
    global _global_buffer_manager
    if _global_buffer_manager is None:
        _global_buffer_manager = BufferManager()
    return _global_buffer_manager


def configure_buffers(
    buffer_size_mb: int = 256, num_buffers: int = 4, enable_memory_pool: bool = True
):
    """Configure global buffer settings."""
    global _global_buffer_manager
    _global_buffer_manager = BufferManager(
        buffer_size_mb, num_buffers, enable_memory_pool
    )


def get_buffer_stats() -> Dict[str, Any]:
    """Get comprehensive buffer usage statistics."""
    return get_buffer_manager().get_memory_usage()


def clear_buffers():
    """Clear all buffers."""
    get_buffer_manager().clear_buffers()


def optimize_for_workload(expected_file_size_mb: float, concurrent_files: int = 4):
    """Optimize buffer configuration for expected workload."""
    get_buffer_manager().optimize_for_workload(expected_file_size_mb, concurrent_files)


def create_streaming_buffer(
    key: str, capacity: int, item_shape: Tuple[int, ...], dtype: torch.dtype
) -> StreamingBuffer:
    """Create a streaming buffer for continuous data flow."""
    return get_buffer_manager().create_streaming_buffer(
        key, capacity, item_shape, dtype
    )


def get_optimal_buffer_config(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get optimal buffer configuration for a dataset."""
    file_size_mb = dataset_info.get("avg_file_size_mb", 10.0)
    num_files = dataset_info.get("num_files", 1000)
    concurrent_workers = dataset_info.get("num_workers", 4)

    # Calculate optimal settings
    if psutil:
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
    else:
        available_memory = 4096.0  # Fallback default

    # Conservative memory usage (25% of available)
    max_buffer_memory = available_memory * 0.25

    # Calculate buffer size and count
    optimal_buffer_size = min(
        file_size_mb * 2, max_buffer_memory / (concurrent_workers + 2)
    )
    optimal_num_buffers = concurrent_workers + 2

    return {
        "buffer_size_mb": int(optimal_buffer_size),
        "num_buffers": optimal_num_buffers,
        "enable_memory_pool": True,
        "streaming_buffer_capacity": min(100, num_files // 10),
        "total_memory_mb": optimal_buffer_size * optimal_num_buffers,
        "memory_utilization": (optimal_buffer_size * optimal_num_buffers)
        / available_memory,
    }
