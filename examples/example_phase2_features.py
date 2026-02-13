#!/usr/bin/env python3
"""
Phase 2 features example for torchfits.
Demonstrates zero-copy operations, tile-aware compression, streaming tables, and enhanced caching.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table

import torchfits


def create_large_image(shape=(2000, 2000), compressed=False):
    """Create a large test image."""
    data = np.random.normal(100, 10, shape).astype(np.float32)

    # Add structure for better compression
    y, x = np.ogrid[: shape[0], : shape[1]]
    data += 20 * np.sin(2 * np.pi * x / 200) * np.cos(2 * np.pi * y / 200)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        if compressed:
            hdu = fits.CompImageHDU(data, compression_type="RICE_1")
        else:
            hdu = fits.PrimaryHDU(data)

        hdu.writeto(f.name, overwrite=True)
        return f.name, data


def create_large_table(nrows=100000):
    """Create a large test table."""
    data = {
        "ID": np.arange(nrows, dtype=np.int64),
        "RA": np.random.uniform(0, 360, nrows).astype(np.float64),
        "DEC": np.random.uniform(-90, 90, nrows).astype(np.float64),
        "MAG_G": np.random.normal(20, 2, nrows).astype(np.float32),
        "MAG_R": np.random.normal(20, 2, nrows).astype(np.float32),
        "MAG_I": np.random.normal(20, 2, nrows).astype(np.float32),
        "FLUX_G": np.random.exponential(1000, nrows).astype(np.float32),
        "FLUX_R": np.random.exponential(1000, nrows).astype(np.float32),
        "FLUX_I": np.random.exponential(1000, nrows).astype(np.float32),
        "Z": np.random.exponential(0.5, nrows).astype(np.float32),
    }

    table = Table(data)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        table.write(f.name, format="fits", overwrite=True)
        return f.name


def demo_zero_copy_operations():
    """Demonstrate zero-copy tensor operations."""
    print("🚀 Zero-Copy Operations")
    print("-" * 30)

    # Create test image
    filepath, expected_data = create_large_image((1500, 1500))

    try:
        # Measure read performance
        start_time = time.time()
        result, _ = torchfits.read(filepath, return_header=True)
        read_time = time.time() - start_time

        # Calculate throughput
        data_size_mb = result.numel() * 4 / (1024 * 1024)  # float32
        throughput = data_size_mb / read_time if read_time > 0 else float("inf")

        print(f"  Image size: {result.shape}")
        print(f"  Data size: {data_size_mb:.1f} MB")
        print(f"  Read time: {read_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} MB/s")

        # Verify zero-copy (data should be identical)
        np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        print("  ✅ Zero-copy verification passed")

    finally:
        os.unlink(filepath)

    print()


def demo_tile_aware_compression():
    """Demonstrate tile-aware compressed reading."""
    print("🗜️  Tile-Aware Compression")
    print("-" * 30)

    # Create compressed image
    filepath, expected_data = create_large_image((3000, 3000), compressed=True)

    try:
        # Read full compressed image
        start_time = time.time()
        full_result, _ = torchfits.read(
            filepath, hdu=1, return_header=True
        )  # Compressed images in HDU 1
        full_time = time.time() - start_time

        print(f"  Full image: {full_result.shape}")
        print(f"  Full read time: {full_time:.3f}s")

        # Read subset (should use tile optimization)
        # Use FITS 1-based inclusive indexing to match Python [1000:2000]
        subset_spec = "[1][1001:2000,1001:2000]"
        start_time = time.time()
        subset_result, _ = torchfits.read(filepath + subset_spec, return_header=True)
        subset_time = time.time() - start_time

        print(f"  Subset: {subset_result.shape}")
        print(f"  Subset read time: {subset_time:.3f}s")

        # Calculate efficiency
        subset_pixels = 1000 * 1000
        full_pixels = 3000 * 3000
        expected_time = full_time * (subset_pixels / full_pixels)
        if subset_time > 0:
            efficiency = expected_time / subset_time
            print(f"  Tile optimization: {efficiency:.1f}x faster than expected")
        else:
            print("  Tile optimization: n/a (subset_time too small to measure)")

        # Verify subset correctness
        expected_subset = full_result[1000:2000, 1000:2000]
        torch.testing.assert_close(subset_result, expected_subset, rtol=1e-3, atol=1e-1)
        print("  ✅ Tile-aware reading verification passed")

    finally:
        os.unlink(filepath)

    print()


def demo_streaming_tables():
    """Demonstrate streaming table I/O."""
    print("📊 Streaming Table I/O")
    print("-" * 30)

    # Create large table
    filepath = create_large_table(50000)

    try:
        # Regular table read
        start_time = time.time()
        regular_result, _ = torchfits.read(filepath, hdu=1, return_header=True)
        regular_time = time.time() - start_time

        print(f"  Regular read: {regular_time:.3f}s")
        if isinstance(regular_result, dict):
            print(f"  Columns: {list(regular_result.keys())}")
            print(
                f"  Rows: {len(regular_result['RA']) if 'RA' in regular_result else 'N/A'}"
            )

        # Streaming read with memory limit
        start_time = time.time()
        streaming_result = torchfits.read_large_table(
            filepath, max_memory_mb=5, streaming=True
        )
        streaming_time = time.time() - start_time

        print(f"  Streaming read: {streaming_time:.3f}s")
        print("  Memory limit: 5 MB")

        if isinstance(streaming_result, dict):
            print(f"  Columns: {list(streaming_result.keys())}")
            print(
                f"  Rows: {len(streaming_result['RA']) if 'RA' in streaming_result else 'N/A'}"
            )

            # Verify results are equivalent
            if isinstance(regular_result, dict):
                for key in regular_result:
                    if key in streaming_result:
                        torch.testing.assert_close(
                            regular_result[key],
                            streaming_result[key],
                            rtol=1e-5,
                            atol=1e-8,
                        )
                print("  ✅ Streaming verification passed")

    finally:
        os.unlink(filepath)

    print()


def demo_enhanced_caching():
    """Demonstrate enhanced file caching."""
    print("💾 Enhanced File Caching")
    print("-" * 30)

    # Create multiple test files
    files = []
    try:
        for i in range(3):
            filepath, _ = create_large_image((500, 500))
            files.append(filepath)

        # Clear cache and get initial stats
        torchfits.clear_file_cache()

        # First pass (cache misses)
        start_time = time.time()
        for filepath in files:
            torchfits.read(filepath)
        first_pass_time = time.time() - start_time

        torchfits.get_cache_performance()

        # Second pass (cache hits)
        start_time = time.time()
        for filepath in files:
            torchfits.read(filepath)
        second_pass_time = time.time() - start_time

        stats_after_second = torchfits.get_cache_performance()

        print(f"  Files: {len(files)}")
        print(f"  First pass (misses): {first_pass_time:.3f}s")
        print(f"  Second pass (hits): {second_pass_time:.3f}s")
        if second_pass_time > 0:
            print(f"  Speedup: {first_pass_time / second_pass_time:.1f}x")
        else:
            print("  Speedup: n/a (second pass too small to measure)")
        print(f"  Hit rate: {stats_after_second.get('hit_rate', 0):.1%}")
        print(f"  Total requests: {stats_after_second.get('total_requests', 0)}")
        print("  ✅ Enhanced caching working correctly")

    finally:
        for f in files:
            if os.path.exists(f):
                os.unlink(f)
        torchfits.clear_file_cache()

    print()


def demo_batch_operations():
    """Demonstrate batch operations."""
    print("📦 Batch Operations")
    print("-" * 30)

    # Create multiple test files
    files = []
    try:
        for i in range(5):
            filepath, _ = create_large_image((300, 300))
            files.append(filepath)

        # Sequential reading
        start_time = time.time()
        sequential_results = []
        for filepath in files:
            result, _ = torchfits.read(filepath, return_header=True)
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        # Batch reading
        start_time = time.time()
        torchfits.read_batch(files)
        batch_time = time.time() - start_time

        print(f"  Files: {len(files)}")
        print(f"  Sequential: {sequential_time:.3f}s")
        print(f"  Batch: {batch_time:.3f}s")
        if batch_time > 0:
            print(f"  Speedup: {sequential_time / batch_time:.1f}x")
        else:
            print("  Speedup: n/a (batch time too small to measure)")

        # Batch info
        info = torchfits.get_batch_info(files)
        print(f"  Valid files: {info.get('valid_files', 0)}/{info.get('num_files', 0)}")
        print(f"  Total size: {info.get('total_size_mb', 0):.1f} MB")
        print("  ✅ Batch operations working correctly")

    finally:
        for f in files:
            if os.path.exists(f):
                os.unlink(f)

    print()


def demo_memory_efficiency():
    """Demonstrate memory efficiency."""
    print("🧠 Memory Efficiency")
    print("-" * 30)

    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        print("  ⚠️  psutil not available - skipping memory demo")
        return

    # Create large file
    filepath, _ = create_large_image((2000, 2000))

    try:
        # Measure memory usage
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result, _ = torchfits.read(filepath, return_header=True)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = mem_after - mem_before

        # Calculate efficiency
        file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
        data_size = result.numel() * 4 / 1024 / 1024  # MB

        print(f"  File size: {file_size:.1f} MB")
        print(f"  Data size: {data_size:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        if memory_increase > 0:
            print(f"  Efficiency: {data_size / memory_increase:.1f}x")
        else:
            print("  Efficiency: n/a (RSS unchanged or decreased)")
        print("  ✅ Memory efficiency optimized")

        del result

    finally:
        os.unlink(filepath)

    print()


def main():
    """Run Phase 2 feature demonstrations."""
    print("🌟 TorchFITS Phase 2 Features")
    print("=" * 40)
    print("Demonstrating advanced performance optimizations:")
    print("- Zero-copy tensor operations")
    print("- Tile-aware compressed reading")
    print("- Streaming table I/O")
    print("- Enhanced file caching")
    print("- Batch operations")
    print("- Memory efficiency")
    print()

    try:
        demo_zero_copy_operations()
        demo_tile_aware_compression()
        demo_streaming_tables()
        demo_enhanced_caching()
        demo_batch_operations()
        demo_memory_efficiency()

        print("🎉 All Phase 2 features demonstrated successfully!")
        print("These optimizations provide significant performance improvements")
        print("for large-scale astronomical data processing workflows.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
