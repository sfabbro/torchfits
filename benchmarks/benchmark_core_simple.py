#!/usr/bin/env python3
"""
Simplified core benchmarks that work around tensor conversion issues.
"""

import time
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torchfits.core import FITSDataTypeHandler, CompressionHandler
from torchfits.cache import clear_cache, configure_for_environment, get_cache_stats
from torchfits.buffer import get_buffer_manager, get_buffer_stats, clear_buffers
from astropy.io import fits

def benchmark_data_types():
    """Benchmark FITS data type handling."""
    print("=== Data Type Handling Benchmarks ===")
    
    dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    sizes = [(100, 100), (500, 500)]
    
    for dtype in dtypes:
        for size in sizes:
            data = np.random.randn(*size).astype(dtype)
            
            # Write test file
            filename = f'bench_{dtype.__name__}_{size[0]}x{size[1]}.fits'
            fits.writeto(filename, data, overwrite=True)
            
            # Benchmark astropy baseline
            start = time.perf_counter()
            with fits.open(filename) as hdul:
                astropy_data = hdul[0].data
                if astropy_data.dtype.byteorder not in ('=', '|'):
                    astropy_data = astropy_data.astype(astropy_data.dtype.newbyteorder('='))
                astropy_tensor = torch.from_numpy(astropy_data.copy())
            astropy_time = time.perf_counter() - start
            
            # Test data type conversion
            start = time.perf_counter()
            torch_dtype = FITSDataTypeHandler.to_torch_dtype(-32)  # float32
            conversion_time = time.perf_counter() - start
            
            print(f"{dtype.__name__} {size}: astropy={astropy_time:.4f}s, dtype_conversion={conversion_time:.6f}s")
            
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
    try:
        configure_for_environment()
        
        # Test cache statistics
        start = time.perf_counter()
        for _ in range(1000):
            stats = get_cache_stats()
        stats_time = time.perf_counter() - start
        
        print(f"Cache stats operations: {1000/stats_time:.0f} ops/sec")
        print(f"Cache configured: {stats.get('config', {}).get('max_files', 'unknown')}")
        
        # Test cache clearing
        start = time.perf_counter()
        clear_cache()
        clear_time = time.perf_counter() - start
        
        print(f"Cache clear time: {clear_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"Cache benchmark failed: {e}")

def benchmark_buffer_management():
    """Benchmark buffer management."""
    print("\n=== Buffer Management Benchmarks ===")
    
    try:
        buffer_manager = get_buffer_manager()
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            shape = (size,)
            dtype = torch.float32
            
            # Benchmark buffer creation
            start = time.perf_counter()
            buffer = buffer_manager.get_buffer(f"test_{size}", shape, dtype)
            creation_time = time.perf_counter() - start
            
            # Benchmark tensor operations
            start = time.perf_counter()
            buffer.fill_(1.0)
            tensor_time = time.perf_counter() - start
            
            print(f"Size {size}: buffer={creation_time:.6f}s, tensor={tensor_time:.6f}s")
        
        # Print buffer statistics
        stats = get_buffer_stats()
        print(f"Buffer stats: {stats['num_buffers']} buffers, {stats['total_memory_mb']:.1f} MB total")
        
    except Exception as e:
        print(f"Buffer benchmark failed: {e}")

def benchmark_scaling():
    """Benchmark FITS scaling operations."""
    print("\n=== FITS Scaling Benchmarks ===")
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        data = np.random.randint(0, 32767, size, dtype=np.int16)
        bzero, bscale = 32768.0, 0.1
        
        # Benchmark manual scaling
        start = time.perf_counter()
        manual_scaled = data.astype(np.float32) * bscale + bzero
        manual_tensor = torch.from_numpy(manual_scaled)
        manual_time = time.perf_counter() - start
        
        # Benchmark FITSDataTypeHandler scaling
        start = time.perf_counter()
        tensor = torch.from_numpy(data.astype(np.int16))
        scaled_tensor = FITSDataTypeHandler.apply_scaling(tensor, bzero, bscale)
        handler_time = time.perf_counter() - start
        
        throughput = size / handler_time / 1e6
        print(f"Size {size}: handler={handler_time:.4f}s ({throughput:.1f}M elements/sec), manual={manual_time:.4f}s")

def main():
    """Run all core benchmarks."""
    print("torchfits Core Features Benchmark Suite (Simplified)")
    print("=" * 60)
    
    try:
        benchmark_data_types()
        benchmark_scaling()
        benchmark_compression_detection()
        benchmark_caching()
        benchmark_buffer_management()
        
        print("\n" + "=" * 60)
        print("Core benchmarks completed successfully")
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        try:
            clear_cache()
            clear_buffers()
        except:
            pass

if __name__ == "__main__":
    main()