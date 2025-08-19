"""
Core feature benchmarks for Phase 1 completion.
Tests data type handling, scaling, compression detection, and caching.
"""

import time
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torchfits
from torchfits.core import FITSCore, FITSDataTypeHandler, CompressionHandler
from torchfits.cache import clear_cache, configure_for_environment, get_cache_stats
from torchfits.buffer import create_managed_buffer
from astropy.io import fits

def benchmark_data_types():
    """Benchmark FITS data type handling."""
    print("=== Data Type Handling Benchmarks ===")
    
    dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for dtype in dtypes:
        for size in sizes:
            data = np.random.randn(*size).astype(dtype)
            
            # Write test file
            filename = f'bench_{dtype.__name__}_{size[0]}x{size[1]}.fits'
            fits.writeto(filename, data, overwrite=True)
            
            # Benchmark torchfits
            start = time.perf_counter()
            tensor = torchfits.read(filename)
            torchfits_time = time.perf_counter() - start
            
            # Benchmark astropy
            start = time.perf_counter()
            with fits.open(filename) as hdul:
                astropy_data = hdul[0].data
                if astropy_data.dtype.byteorder not in ('=', '|'):
                    astropy_data = astropy_data.astype(astropy_data.dtype.newbyteorder('='))
                astropy_tensor = torch.from_numpy(astropy_data.copy())
            astropy_time = time.perf_counter() - start
            
            speedup = astropy_time / torchfits_time
            print(f"{dtype.__name__} {size}: {speedup:.1f}x speedup ({torchfits_time:.4f}s vs {astropy_time:.4f}s)")
            
            Path(filename).unlink()

def benchmark_scaling():
    """Benchmark FITS scaling operations."""
    print("\n=== FITS Scaling Benchmarks ===")
    
    
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        data = np.random.randint(0, 32767, size, dtype=np.int16)
        bzero, bscale = 32768.0, 0.1
        
        # Create FITS with scaling
        hdu = fits.PrimaryHDU(data)
        hdu.header['BZERO'] = bzero
        hdu.header['BSCALE'] = bscale
        filename = f'scaling_{size}.fits'
        fits.writeto(filename, hdu.data, header=hdu.header, overwrite=True)
        
        # Benchmark torchfits (includes scaling)
        start = time.perf_counter()
        tensor = torchfits.read(filename)
        torchfits_time = time.perf_counter() - start
        
        # Benchmark manual scaling
        start = time.perf_counter()
        manual_scaled = data.astype(np.float32) * bscale + bzero
        manual_tensor = torch.from_numpy(manual_scaled)
        manual_time = time.perf_counter() - start
        
        throughput = size / torchfits_time / 1e6
        print(f"Size {size}: {throughput:.1f}M elements/sec ({torchfits_time:.4f}s)")
        
        Path(filename).unlink()

def benchmark_compression_detection():
    """Benchmark compression detection."""
    print("\n=== Compression Detection Benchmarks ===")
    
    headers = [
        {'SIMPLE': True},  # Uncompressed
        {'ZCMPTYPE': 'RICE_1', 'ZTILE1': 256, 'ZTILE2': 256},
        {'ZCMPTYPE': 'GZIP_1', 'ZTILE1': 128, 'ZTILE2': 128},
        {'ZCMPTYPE': 'HCOMPRESS_1', 'ZQUANTIZ': 4},
    ]
    
    n_iterations = 100000
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        for header in headers:
            CompressionHandler.detect_compression(header)
            CompressionHandler.is_compressed(header)
    total_time = time.perf_counter() - start
    
    ops_per_sec = (n_iterations * len(headers)) / total_time
    print(f"Compression detection: {ops_per_sec:.0f} ops/sec")

def benchmark_caching():
    """Benchmark caching system."""
    print("\n=== Caching System Benchmarks ===")
    
    # Configure cache for testing
    configure_for_environment()
    
    # Test cache statistics
    start = time.perf_counter()
    for _ in range(1000):
        stats = get_cache_stats()
    stats_time = time.perf_counter() - start
    
    print(f"Cache stats operations: {1000/stats_time:.0f} ops/sec")
    print(f"Current cache size: {stats['size']}")
    print(f"Cache configured: {stats['configured']}")
    
    # Test cache clearing
    start = time.perf_counter()
    clear_cache()
    clear_time = time.perf_counter() - start
    
    print(f"Cache clear time: {clear_time*1000:.2f} ms")

def benchmark_buffer_management():
    """Benchmark buffer management."""
    print("\n=== Buffer Management Benchmarks ===")
    
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        
        # Benchmark managed buffer creation
        start = time.perf_counter()
        buffer = create_managed_buffer(data)
        creation_time = time.perf_counter() - start
        
        # Benchmark tensor creation
        start = time.perf_counter()
        tensor = buffer.get_tensor()
        tensor_time = time.perf_counter() - start
        
        # Benchmark GPU transfer if available
        if torch.cuda.is_available():
            start = time.perf_counter()
            gpu_tensor = buffer.get_tensor('cuda')
            gpu_time = time.perf_counter() - start
            print(f"Size {size}: buffer={creation_time:.6f}s, tensor={tensor_time:.6f}s, gpu={gpu_time:.4f}s")
        else:
            print(f"Size {size}: buffer={creation_time:.6f}s, tensor={tensor_time:.6f}s")

def benchmark_core_processing():
    """Benchmark core FITS processing pipeline."""
    print("\n=== Core Processing Pipeline Benchmarks ===")
    
    
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for size in sizes:
        # Create test data with scaling
        data = np.random.randint(0, 1000, size, dtype=np.int16)
        header = {'BZERO': 1000.0, 'BSCALE': 0.1, 'CHECKSUM': 'test'}
        
        # Benchmark full processing pipeline
        start = time.perf_counter()
        tensor = FITSCore.process_data(data, header)
        processing_time = time.perf_counter() - start
        
        # Compare with manual processing
        start = time.perf_counter()
        manual_data = data.astype(data.dtype.newbyteorder('=')) if data.dtype.byteorder not in ('=', '|') else data
        manual_tensor = torch.from_numpy(manual_data.copy())
        manual_scaled = manual_tensor.float() * 0.1 + 1000.0
        manual_time = time.perf_counter() - start
        
        elements = size[0] * size[1]
        throughput = elements / processing_time / 1e6
        
        print(f"Size {size}: {throughput:.1f}M elements/sec ({processing_time:.4f}s)")

def main():
    """Run all core benchmarks."""
    print("torchfits Core Features Benchmark Suite")
    print("=" * 50)
    
    benchmark_data_types()
    benchmark_scaling()
    benchmark_compression_detection()
    benchmark_caching()
    benchmark_buffer_management()
    benchmark_core_processing()
    
    print("\n" + "=" * 50)
    print("Core benchmarks completed")

if __name__ == "__main__":
    main()