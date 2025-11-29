#!/usr/bin/env python3
"""
Minimal core benchmarks that avoid segfaults.
"""

import time
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def benchmark_compression_detection():
    """Benchmark compression detection."""
    print("=== Compression Detection Benchmarks ===")
    
    from torchfits.core import CompressionHandler
    
    headers = [
        {'SIMPLE': True},  # Uncompressed
        {'ZCMPTYPE': 'RICE_1', 'ZTILE1': 256, 'ZTILE2': 256},
        {'ZCMPTYPE': 'GZIP_1', 'ZTILE1': 128, 'ZTILE2': 128},
        {'ZCMPTYPE': 'HCOMPRESS_1', 'ZQUANTIZ': 4},
    ]
    
    n_iterations = 50000  # Reduced for safety
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        for header in headers:
            CompressionHandler.detect_compression(header)
            CompressionHandler.is_compressed(header)
    total_time = time.perf_counter() - start
    
    ops_per_sec = (n_iterations * len(headers)) / total_time
    print(f"Compression detection: {ops_per_sec:.0f} ops/sec")

def benchmark_scaling():
    """Benchmark FITS scaling operations."""
    print("\n=== FITS Scaling Benchmarks ===")
    
    from torchfits.core import FITSDataTypeHandler
    
    sizes = [1000, 10000, 100000]  # Reduced for safety
    
    for size in sizes:
        data = np.random.randint(0, 32767, size, dtype=np.int16)
        bzero, bscale = 32768.0, 0.1
        
        # Benchmark FITSDataTypeHandler scaling
        start = time.perf_counter()
        tensor = torch.from_numpy(data.astype(np.int16))
        scaled_tensor = FITSDataTypeHandler.apply_scaling(tensor, bzero, bscale)
        handler_time = time.perf_counter() - start
        
        # Benchmark manual scaling
        start = time.perf_counter()
        manual_scaled = data.astype(np.float32) * bscale + bzero
        manual_tensor = torch.from_numpy(manual_scaled)
        manual_time = time.perf_counter() - start
        
        throughput = size / handler_time / 1e6 if handler_time > 0 else float('inf')
        print(f"Size {size}: {throughput:.1f}M elements/sec ({handler_time:.4f}s vs {manual_time:.4f}s)")

def benchmark_data_type_conversion():
    """Benchmark data type conversion without file I/O."""
    print("\n=== Data Type Conversion Benchmarks ===")
    
    from torchfits.core import FITSDataTypeHandler
    
    bitpix_values = [8, 16, 32, -32, -64]
    
    for bitpix in bitpix_values:
        try:
            start = time.perf_counter()
            for _ in range(10000):
                dtype = FITSDataTypeHandler.to_torch_dtype(bitpix)
            conversion_time = time.perf_counter() - start
            
            ops_per_sec = 10000 / conversion_time
            print(f"BITPIX {bitpix}: {ops_per_sec:.0f} conversions/sec -> {dtype}")
        except Exception as e:
            print(f"BITPIX {bitpix}: Failed - {e}")

def main():
    """Run minimal core benchmarks."""
    print("torchfits Core Features Benchmark Suite (Minimal)")
    print("=" * 60)
    
    try:
        benchmark_data_type_conversion()
        benchmark_scaling()
        benchmark_compression_detection()
        
        print("\n" + "=" * 60)
        print("Core benchmarks completed successfully")
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()