"""
Basic benchmarks for torchfits performance.

This module provides benchmark tests to measure performance
against other FITS libraries like astropy and fitsio.
"""


import time
import torch
import numpy as np
import pytest
import torchfits
from astropy.io import fits as astropy_fits
import fitsio


def create_test_data(shape=(1000, 1000), dtype=np.float32):
    """Create test data for benchmarking."""
    return np.random.randn(*shape).astype(dtype)


def benchmark_tensor_creation():
    """Benchmark tensor creation from numpy arrays."""
    print("=== Tensor Creation Benchmark ===")
    
    # Create test data
    data = create_test_data((2000, 2000))
    
    # Benchmark torch.from_numpy
    start_time = time.time()
    for _ in range(100):
        tensor = torch.from_numpy(data)
    torch_time = time.time() - start_time
    
    print(f"torch.from_numpy (100 iterations): {torch_time:.4f}s")
    print(f"Average per iteration: {torch_time/100*1000:.2f}ms")
    print()


def benchmark_header_operations():
    """Benchmark header operations."""
    print("=== Header Operations Benchmark ===")
    
    # Create test header
    header_data = {f'KEY{i:03d}': f'VALUE{i}' for i in range(1000)}
    
    # Benchmark torchfits Header
    start_time = time.time()
    for _ in range(1000):
        header = torchfits.Header(header_data)
        _ = header['KEY500']
        header['NEWKEY'] = 'NEWVALUE'
    torchfits_time = time.time() - start_time
    
    print(f"torchfits Header (1000 iterations): {torchfits_time:.4f}s")
    
    # Benchmark astropy Header
    astropy_header = astropy_fits.Header(header_data)
    start_time = time.time()
    for _ in range(1000):
        _ = astropy_header['KEY500']
        astropy_header['NEWKEY'] = 'NEWVALUE'
    astropy_time = time.time() - start_time
        
    print(f"astropy Header (1000 iterations): {astropy_time:.4f}s")
    print(f"Speedup: {astropy_time/torchfits_time:.2f}x")
    
    print()


def benchmark_wcs_transformations():
    """Benchmark WCS coordinate transformations."""
    print("=== WCS Transformations Benchmark ===")
    
    # Create test WCS
    header = torchfits.Header({
        'CRPIX1': 1000.0,
        'CRPIX2': 1000.0,
        'CRVAL1': 180.0,
        'CRVAL2': 0.0,
        'CDELT1': -0.0001,
        'CDELT2': 0.0001
    })
    
    wcs = torchfits.WCS(header)
    
    # Create test coordinates
    n_coords = 10000
    pixels = torch.randn(n_coords, 2) * 100 + 1000
    
    # Benchmark pixel to world transformation
    start_time = time.time()
    for _ in range(10):
        world_coords = wcs.pixel_to_world(pixels)
    torchfits_time = time.time() - start_time
    
    print(f"torchfits WCS ({n_coords} coords, 10 iterations): {torchfits_time:.4f}s")
    print(f"Coords per second: {n_coords * 10 / torchfits_time:.0f}")
    
    # Note: astropy WCS comparison would require actual WCS setup
    print("Note: astropy WCS comparison requires full WCS initialization")
    print()


def benchmark_data_loading_simulation():
    """Simulate data loading performance."""
    print("=== Data Loading Simulation ===")
    
    # Simulate different data sizes
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for size in sizes:
        data = create_test_data(size)
        
        # Simulate torchfits loading (tensor creation + device transfer)
        start_time = time.time()
        for _ in range(10):
            tensor = torch.from_numpy(data.copy())
            if torch.cuda.is_available():
                tensor = tensor.cuda()
        torchfits_time = time.time() - start_time
        
        data_size_mb = data.nbytes / 1024 / 1024
        throughput = data_size_mb * 10 / torchfits_time
        
        print(f"Size {size}: {torchfits_time:.4f}s, "
              f"{data_size_mb:.1f}MB, {throughput:.1f}MB/s")
    
    print()


def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("=== Memory Usage Benchmark ===")
    
    # Test different tensor operations
    data = create_test_data((1000, 1000))
    
    # Memory-efficient operations
    start_time = time.time()
    tensor = torch.from_numpy(data)
    stats = {
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'min': float(tensor.min()),
        'max': float(tensor.max())
    }
    stats_time = time.time() - start_time
    
    print(f"Statistics computation: {stats_time:.4f}s")
    print(f"Stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    print()


@pytest.mark.benchmark
def test_tensor_creation_benchmark(benchmark):
    """Pytest benchmark for tensor creation."""
    data = create_test_data((1000, 1000))
    
    def create_tensor():
        return torch.from_numpy(data.copy())
    
    result = benchmark(create_tensor)
    assert result.shape == (1000, 1000)


@pytest.mark.benchmark
def test_header_access_benchmark(benchmark):
    """Pytest benchmark for header access."""
    header_data = {f'KEY{i:03d}': f'VALUE{i}' for i in range(100)}
    header = torchfits.Header(header_data)
    
    def access_header():
        return header['KEY050']
    
    result = benchmark(access_header)
    assert result == 'VALUE50'


def main():
    """Run all benchmarks."""
    print("torchfits Performance Benchmarks")
    print("=" * 40)
    print()
    
    benchmark_tensor_creation()
    benchmark_header_operations()
    benchmark_wcs_transformations()
    benchmark_data_loading_simulation()
    benchmark_memory_usage()
    
    print("Benchmarks completed!")
    print("\nNote: These are synthetic benchmarks using fallback implementations.")
    print("Real performance gains will be seen with the compiled C++ extension.")


if __name__ == "__main__":
    main()