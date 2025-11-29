"""
Performance Regression Analysis

Quick benchmark to check if recent zero-copy optimizations
introduced any performance regressions compared to baseline.
"""

import time
import sys
from pathlib import Path
import numpy as np
import torch
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torchfits
except ImportError as e:
    print(f"⚠️  torchfits import failed: {e}")
    torchfits = None

try:
    from astropy.io import fits as astropy_fits
except ImportError:
    astropy_fits = None

try:
    import fitsio
except ImportError:
    fitsio = None


def create_test_file(shape, dtype):
    """Create a simple test FITS file."""
    data = np.random.randn(*shape).astype(dtype)
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        if astropy_fits:
            astropy_fits.writeto(f.name, data, overwrite=True)
            return f.name, data
    return None, None


def benchmark_simple_read(filepath, label, num_runs=10):
    """Benchmark simple read operations."""
    if not torchfits:
        return None
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        tensor = torchfits.read(filepath)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  {label}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    return avg_time


def benchmark_baseline_methods(filepath, num_runs=10):
    """Benchmark baseline methods for comparison."""
    results = {}
    
    # Benchmark astropy
    if astropy_fits:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with astropy_fits.open(filepath) as hdul:
                array = hdul[0].data
                if array is not None:
                    # Handle byte order
                    if array.dtype.byteorder not in ('=', '|'):
                        array = array.astype(array.dtype.newbyteorder('='))
                    tensor = torch.from_numpy(array.copy())
            end = time.perf_counter()
            times.append(end - start)
        
        results['astropy'] = np.mean(times)
        print(f"  astropy:    {results['astropy']*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
    
    # Benchmark fitsio
    if fitsio:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            array = fitsio.read(filepath)
            tensor = torch.from_numpy(array)
            end = time.perf_counter()
            times.append(end - start)
        
        results['fitsio'] = np.mean(times)
        print(f"  fitsio:     {results['fitsio']*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
    
    return results


def main():
    """Run performance regression analysis."""
    print("🔍 Performance Regression Analysis")
    print("=" * 50)
    
    if not torchfits:
        print("❌ torchfits not available")
        return
    
    # Test different configurations
    test_configs = [
        {"shape": (100, 100), "dtype": np.float32, "name": "small_float32"},
        {"shape": (1000, 1000), "dtype": np.float32, "name": "medium_float32"},
        {"shape": (1000, 1000), "dtype": np.int16, "name": "medium_int16"},
        {"shape": (2000, 2000), "dtype": np.float32, "name": "large_float32"},
    ]
    
    for config in test_configs:
        shape = config["shape"]
        dtype = config["dtype"]
        name = config["name"]
        
        print(f"\n📊 Testing {name} {shape}")
        
        # Create test file
        filepath, data = create_test_file(shape, dtype)
        if not filepath:
            print("  ❌ Failed to create test file")
            continue
        
        try:
            # Test different read modes
            torchfits_cpu = benchmark_simple_read(filepath, "torchfits CPU")
            
            # Test with explicit device specification
            if torchfits_cpu:
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    tensor = torchfits.read(filepath, device='cpu')
                    end = time.perf_counter()
                    times.append(end - start)
                
                torchfits_device = np.mean(times)
                print(f"  torchfits device='cpu': {torchfits_device*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
                
                # Check for device parameter overhead
                if torchfits_device > torchfits_cpu * 1.1:  # 10% overhead threshold
                    print(f"  ⚠️  Device parameter adds {((torchfits_device/torchfits_cpu-1)*100):.1f}% overhead")
            
            # Benchmark baseline methods
            baseline_results = benchmark_baseline_methods(filepath)
            
            # Calculate speedups
            if torchfits_cpu and baseline_results:
                if 'astropy' in baseline_results:
                    speedup = baseline_results['astropy'] / torchfits_cpu
                    print(f"  Speedup vs astropy: {speedup:.2f}x")
                
                if 'fitsio' in baseline_results:
                    speedup = baseline_results['fitsio'] / torchfits_cpu
                    print(f"  Speedup vs fitsio:  {speedup:.2f}x")
                    
                    if speedup < 1.0:
                        print(f"  ⚠️  REGRESSION: {speedup:.2f}x slower than fitsio")
        
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    print("\n" + "=" * 50)
    print("Analysis complete")


if __name__ == "__main__":
    main()