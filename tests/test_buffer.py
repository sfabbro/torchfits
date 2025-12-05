"""
Tests for torchfits buffer management module.
"""

import pytest
import torch
import threading
import time
from unittest.mock import patch, Mock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchfits.buffer import (
    MemoryPool,
    StreamingBuffer,
    BufferManager,
    get_buffer_manager,
    configure_buffers,
    get_buffer_stats,
    clear_buffers,
    optimize_for_workload,
    create_streaming_buffer,
    get_optimal_buffer_config,
)


class TestMemoryPool:
    """Test MemoryPool functionality."""

    def test_memory_pool_creation(self):
        """Test memory pool creation and basic functionality."""
        pool = MemoryPool(max_size_mb=100)

        assert pool.max_size_bytes == 100 * 1024 * 1024
        assert pool._current_size == 0
        assert len(pool._pool) == 0

    def test_get_buffer_new(self):
        """Test getting a new buffer from pool."""
        pool = MemoryPool(max_size_mb=100)

        buffer = pool.get_buffer((10, 10), torch.float32, "cpu")

        assert buffer.shape == (10, 10)
        assert buffer.dtype == torch.float32
        assert buffer.device.type == "cpu"
        assert len(pool._pool) == 1

    def test_get_buffer_existing(self):
        """Test getting an existing buffer from pool."""
        pool = MemoryPool(max_size_mb=100)

        # Get buffer first time
        buffer1 = pool.get_buffer((10, 10), torch.float32, "cpu")
        buffer1.fill_(42.0)

        # Get same buffer specification
        buffer2 = pool.get_buffer((10, 10), torch.float32, "cpu")

        # Should be the same buffer object
        assert torch.equal(buffer1, buffer2)
        assert len(pool._pool) == 1

    def test_buffer_eviction(self):
        """Test buffer eviction when pool is full."""
        # Small pool that can only hold one small buffer
        pool = MemoryPool(max_size_mb=1)  # Very small

        # Create first buffer
        buffer1 = pool.get_buffer((100, 100), torch.float32, "cpu")
        initial_size = len(pool._pool)

        # Create second buffer that should cause eviction
        buffer2 = pool.get_buffer((200, 200), torch.float32, "cpu")

        # Pool should still exist but may have evicted old buffer
        assert len(pool._pool) >= 1

    def test_clear_pool(self):
        """Test clearing the memory pool."""
        pool = MemoryPool(max_size_mb=100)

        # Add some buffers
        pool.get_buffer((10, 10), torch.float32, "cpu")
        pool.get_buffer((20, 20), torch.float64, "cpu")

        assert len(pool._pool) == 2

        pool.clear()

        assert len(pool._pool) == 0
        assert pool._current_size == 0

    def test_get_stats(self):
        """Test getting pool statistics."""
        pool = MemoryPool(max_size_mb=100)

        stats = pool.get_stats()

        assert "num_buffers" in stats
        assert "current_size_mb" in stats
        assert "max_size_mb" in stats
        assert "utilization" in stats

        assert stats["num_buffers"] == 0
        assert stats["max_size_mb"] == 100
        assert stats["utilization"] == 0


class TestStreamingBuffer:
    """Test StreamingBuffer functionality."""

    def test_streaming_buffer_creation(self):
        """Test streaming buffer creation."""
        buffer = StreamingBuffer(capacity=10, item_shape=(5, 5), dtype=torch.float32)

        assert buffer.capacity == 10
        assert buffer.item_shape == (5, 5)
        assert buffer.dtype == torch.float32
        assert buffer.buffer.shape == (10, 5, 5)
        assert buffer.size() == 0
        assert buffer.is_empty()
        assert not buffer.is_full()

    def test_put_and_get(self):
        """Test putting and getting items."""
        buffer = StreamingBuffer(capacity=3, item_shape=(2, 2), dtype=torch.float32)

        # Put items
        item1 = torch.ones(2, 2)
        item2 = torch.ones(2, 2) * 2

        assert buffer.put(item1) == True
        assert buffer.size() == 1

        assert buffer.put(item2) == True
        assert buffer.size() == 2

        # Get items
        retrieved1 = buffer.get()
        assert torch.equal(retrieved1, item1)
        assert buffer.size() == 1

        retrieved2 = buffer.get()
        assert torch.equal(retrieved2, item2)
        assert buffer.size() == 0
        assert buffer.is_empty()

    def test_buffer_full(self):
        """Test buffer full condition."""
        buffer = StreamingBuffer(capacity=2, item_shape=(3, 3), dtype=torch.float32)

        item = torch.ones(3, 3)

        # Fill buffer
        assert buffer.put(item) == True
        assert buffer.put(item) == True
        assert buffer.is_full()

        # Should reject additional items
        assert buffer.put(item) == False
        assert buffer.size() == 2

    def test_circular_behavior(self):
        """Test circular buffer behavior."""
        buffer = StreamingBuffer(capacity=2, item_shape=(1,), dtype=torch.float32)

        # Fill and empty multiple times
        for i in range(5):
            item = torch.tensor([float(i)])
            assert buffer.put(item) == True
            retrieved = buffer.get()
            assert torch.equal(retrieved, item)

    def test_empty_buffer_get(self):
        """Test getting from empty buffer."""
        buffer = StreamingBuffer(capacity=2, item_shape=(1,), dtype=torch.float32)

        result = buffer.get()
        assert result is None


class TestBufferManager:
    """Test BufferManager functionality."""

    def test_buffer_manager_creation(self):
        """Test buffer manager creation."""
        manager = BufferManager(
            buffer_size_mb=128, num_buffers=4, enable_memory_pool=True
        )

        assert manager.buffer_size_mb == 128
        assert manager.num_buffers == 4
        assert manager.enable_memory_pool == True
        assert manager.memory_pool is not None

    def test_buffer_manager_without_pool(self):
        """Test buffer manager without memory pool."""
        manager = BufferManager(enable_memory_pool=False)

        assert manager.memory_pool is None

    def test_get_buffer_with_pool(self):
        """Test getting buffer with memory pool enabled."""
        manager = BufferManager(enable_memory_pool=True)

        buffer = manager.get_buffer("test", (10, 10), torch.float32, "cpu")

        assert buffer.shape == (10, 10)
        assert buffer.dtype == torch.float32

    def test_get_buffer_without_pool(self):
        """Test getting buffer without memory pool."""
        manager = BufferManager(enable_memory_pool=False)

        buffer = manager.get_buffer("test", (10, 10), torch.float32, "cpu")

        assert buffer.shape == (10, 10)
        assert buffer.dtype == torch.float32
        assert len(manager._buffers) == 1

    def test_create_streaming_buffer(self):
        """Test creating streaming buffer."""
        manager = BufferManager()

        stream_buf = manager.create_streaming_buffer(
            "stream1", 5, (3, 3), torch.float32
        )

        assert isinstance(stream_buf, StreamingBuffer)
        assert stream_buf.capacity == 5
        assert stream_buf.item_shape == (3, 3)
        assert "stream1" in manager._streaming_buffers

    def test_release_buffer_without_pool(self):
        """Test releasing buffer without memory pool."""
        manager = BufferManager(enable_memory_pool=False)

        # Get buffer
        buffer = manager.get_buffer("test", (10, 10), torch.float32, "cpu")
        assert len(manager._buffers) == 1

        # Release buffer
        manager.release_buffer("test", (10, 10), torch.float32, "cpu")
        assert len(manager._buffers) == 0

    def test_clear_buffers(self):
        """Test clearing all buffers."""
        manager = BufferManager(enable_memory_pool=False)

        # Create some buffers
        manager.get_buffer("test1", (10, 10), torch.float32, "cpu")
        manager.create_streaming_buffer("stream1", 5, (3, 3), torch.float32)

        assert len(manager._buffers) == 1
        assert len(manager._streaming_buffers) == 1

        manager.clear_buffers()

        assert len(manager._buffers) == 0
        assert len(manager._streaming_buffers) == 0

    def test_get_memory_usage(self):
        """Test getting memory usage statistics."""
        manager = BufferManager()

        stats = manager.get_memory_usage()

        required_keys = [
            "num_buffers",
            "num_streaming_buffers",
            "buffer_size_mb",
            "num_buffers_config",
            "memory_pool_enabled",
            "total_memory_mb",
            "available_memory_mb",
        ]

        for key in required_keys:
            assert key in stats

    @patch("psutil.virtual_memory")
    def test_optimize_for_workload(self, mock_memory):
        """Test workload optimization."""
        # Mock available memory
        mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB

        manager = BufferManager()

        # Test optimization for small workload
        manager.optimize_for_workload(expected_file_size_mb=10.0, concurrent_files=2)

        assert manager.buffer_size_mb > 0
        assert manager.num_buffers > 0


class TestGlobalFunctions:
    """Test global buffer management functions."""

    def test_get_buffer_manager(self):
        """Test getting global buffer manager."""
        manager1 = get_buffer_manager()
        manager2 = get_buffer_manager()

        # Should return same instance
        assert manager1 is manager2
        assert isinstance(manager1, BufferManager)

    def test_configure_buffers(self):
        """Test configuring global buffers."""
        configure_buffers(buffer_size_mb=256, num_buffers=8, enable_memory_pool=True)

        manager = get_buffer_manager()
        assert manager.buffer_size_mb == 256
        assert manager.num_buffers == 8
        assert manager.enable_memory_pool == True

    def test_get_buffer_stats(self):
        """Test getting global buffer stats."""
        stats = get_buffer_stats()

        assert isinstance(stats, dict)
        assert "num_buffers" in stats

    def test_clear_buffers_global(self):
        """Test clearing global buffers."""
        # Create some buffers first
        manager = get_buffer_manager()
        manager.get_buffer("test", (5, 5), torch.float32, "cpu")

        clear_buffers()

        stats = get_buffer_stats()
        assert stats["num_buffers"] == 0

    def test_optimize_for_workload_global(self):
        """Test global workload optimization."""
        optimize_for_workload(expected_file_size_mb=20.0, concurrent_files=4)

        manager = get_buffer_manager()
        assert manager.buffer_size_mb > 0

    def test_create_streaming_buffer_global(self):
        """Test creating global streaming buffer."""
        stream_buf = create_streaming_buffer("global_stream", 10, (4, 4), torch.float32)

        assert isinstance(stream_buf, StreamingBuffer)
        assert stream_buf.capacity == 10

    @patch("psutil.virtual_memory")
    def test_get_optimal_buffer_config(self, mock_memory):
        """Test getting optimal buffer configuration."""
        # Mock system memory
        mock_memory.return_value.available = 16 * 1024 * 1024 * 1024  # 16GB

        dataset_info = {"avg_file_size_mb": 50.0, "num_files": 1000, "num_workers": 4}

        config = get_optimal_buffer_config(dataset_info)

        required_keys = [
            "buffer_size_mb",
            "num_buffers",
            "enable_memory_pool",
            "streaming_buffer_capacity",
            "total_memory_mb",
            "memory_utilization",
        ]

        for key in required_keys:
            assert key in config

        assert config["buffer_size_mb"] > 0
        assert config["num_buffers"] > 0
        assert 0 <= config["memory_utilization"] <= 1


class TestThreadSafety:
    """Test thread safety of buffer operations."""

    def test_memory_pool_thread_safety(self):
        """Test memory pool thread safety."""
        pool = MemoryPool(max_size_mb=100)
        results = []
        errors = []

        def worker():
            try:
                for i in range(10):
                    buffer = pool.get_buffer((5, 5), torch.float32, "cpu")
                    buffer.fill_(float(threading.current_thread().ident))
                    results.append(buffer.mean().item())
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have any errors
        assert len(errors) == 0
        assert len(results) > 0

    def test_streaming_buffer_thread_safety(self):
        """Test streaming buffer thread safety."""
        buffer = StreamingBuffer(capacity=20, item_shape=(2,), dtype=torch.float32)
        put_count = 0
        get_count = 0
        errors = []

        def producer():
            nonlocal put_count
            try:
                for i in range(10):
                    item = torch.tensor([float(i), float(i * 2)])
                    if buffer.put(item):
                        put_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def consumer():
            nonlocal get_count
            try:
                for _ in range(10):
                    item = buffer.get()
                    if item is not None:
                        get_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Create producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # Should not have any errors
        assert len(errors) == 0


@pytest.mark.performance
class TestBufferPerformance:
    """Test buffer performance characteristics."""

    def test_memory_pool_performance(self):
        """Test memory pool performance."""
        pool = MemoryPool(max_size_mb=500)

        start_time = time.time()

        # Create many buffers
        for i in range(100):
            buffer = pool.get_buffer((100, 100), torch.float32, "cpu")
            buffer.fill_(float(i))

        end_time = time.time()

        # Should be reasonably fast
        assert end_time - start_time < 1.0  # Less than 1 second

    def test_streaming_buffer_performance(self):
        """Test streaming buffer performance."""
        buffer = StreamingBuffer(
            capacity=1000, item_shape=(50, 50), dtype=torch.float32
        )

        start_time = time.time()

        # Fill and empty buffer multiple times
        for i in range(100):
            item = torch.randn(50, 50)
            buffer.put(item)
            retrieved = buffer.get()
            assert retrieved is not None

        end_time = time.time()

        # Should be reasonably fast
        assert end_time - start_time < 2.0  # Less than 2 seconds
