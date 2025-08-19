#!/usr/bin/env python3
"""
Memory mapping benchmarks comparing torchfits, astropy, and fitsio.
"""

import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import torchfits

try:
    from astropy.io import fits as astropy_fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

def create_test_files():
    """Create test files of different sizes."""
    sizes = [
        (1000, 1000),    # 4MB
        (2000, 2000),    # 16MB  
        (4000, 4000),    # 64MB
        (8000, 8000),    # 256MB
    ]
    
    files = {}
    temp_dir = Path(tempfile.mkdtemp())
    
    for shape in sizes:
        data = np.random.randn(*shape).astype(np.float32)
        filename = temp_dir / f"mmap_test_{shape[0]}x{shape[1]}.fits"
        
        if HAS_ASTROPY:
            astropy_fits.writeto(filename, data, overwrite=True)
            files[shape] = filename
            print(f"Created {shape[0]}x{shape[1]} ({data.nbytes/1024**2:.1f}MB): {filename}")
    
    return files

def benchmark_memory_mapping(files):
    """Benchmark memory mapping vs regular reading."""
    print("\\n" + "="*60)
    print("MEMORY MAPPING BENCHMARKS")
    print("="*60)
    
    for shape, filename in files.items():
        size_mb = shape[0] * shape[1] * 4 / 1024**2
        print(f"\\n{shape[0]}x{shape[1]} ({size_mb:.1f}MB):")
        
        results = {}
        
        # torchfits regular
        times = []
        for i in range(3):
            start = time.perf_counter()
            tensor = torchfits.read(str(filename))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results['torchfits_regular'] = sum(times) / len(times)
        print(f"  torchfits (regular): {results['torchfits_regular']:.4f}s")
        
        # torchfits mmap
        times = []
        for i in range(3):
            start = time.perf_counter()
            tensor = torchfits.read(str(filename), mmap=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results['torchfits_mmap'] = sum(times) / len(times)
        print(f"  torchfits (mmap):    {results['torchfits_mmap']:.4f}s")
        
        # astropy memmap=False
        if HAS_ASTROPY:
            times = []
            for i in range(3):
                start = time.perf_counter()
                with astropy_fits.open(filename, memmap=False) as hdul:
                    data = hdul[0].data.copy()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            results['astropy_regular'] = sum(times) / len(times)
            print(f"  astropy (regular):   {results['astropy_regular']:.4f}s")
            
            # astropy memmap=True
            times = []
            for i in range(3):
                start = time.perf_counter()
                with astropy_fits.open(filename, memmap=True) as hdul:
                    data = hdul[0].data[:]  # Force read
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            results['astropy_mmap'] = sum(times) / len(times)
            print(f"  astropy (mmap):      {results['astropy_mmap']:.4f}s")
            
            # astropy mmap -> torch
            times = []
            for i in range(3):
                start = time.perf_counter()
                with astropy_fits.open(filename, memmap=True) as hdul:
                    np_data = hdul[0].data[:]
                    if np_data.dtype.byteorder not in ('=', '|'):
                        np_data = np_data.astype(np_data.dtype.newbyteorder('='))
                    tensor = torch.from_numpy(np_data)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            results['astropy_mmap_torch'] = sum(times) / len(times)
            print(f"  astropy mmap->torch: {results['astropy_mmap_torch']:.4f}s")
        
        # fitsio regular vs torch (paired timing)
        if HAS_FITSIO:
            import gc
            fitsio_times = []
            fitsio_torch_times = []
            
            for i in range(3):
                gc.collect()
                
                # Time fitsio read
                start = time.perf_counter()
                np_data = fitsio.read(str(filename))
                fitsio_time = time.perf_counter() - start
                fitsio_times.append(fitsio_time)
                
                # Time conversion (same data)
                start = time.perf_counter()
                tensor = torch.from_numpy(np_data)
                conv_time = time.perf_counter() - start
                
                fitsio_torch_times.append(fitsio_time + conv_time)
            
            results['fitsio_regular'] = sum(fitsio_times) / len(fitsio_times)
            results['fitsio_torch'] = sum(fitsio_torch_times) / len(fitsio_torch_times)
            
            print(f"  fitsio (regular):    {results['fitsio_regular']:.4f}s")
            print(f"  fitsio->torch:       {results['fitsio_torch']:.4f}s")
            
            # Note: fitsio does not support memory mapping
            # All data is always copied (OWNDATA=True)
        
        # Calculate speedups
        if HAS_ASTROPY and 'astropy_regular' in results:
            speedup = results['astropy_regular'] / results['torchfits_regular']
            print(f"  torchfits vs astropy: {speedup:.2f}x speedup")
            
            if 'astropy_mmap' in results:
                mmap_speedup = results['astropy_mmap'] / results['torchfits_mmap']
                print(f"  torchfits mmap vs astropy mmap: {mmap_speedup:.2f}x speedup")

def benchmark_memory_usage():
    """Test memory usage patterns."""
    print("\\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    
    # Create large test file
    shape = (4000, 4000)
    data = np.random.randn(*shape).astype(np.float32)
    filename = "memory_test.fits"
    
    if HAS_ASTROPY:
        astropy_fits.writeto(filename, data, overwrite=True)
        
        print(f"\\nFile size: {data.nbytes/1024**2:.1f}MB")
        
        # Test memory usage (simplified - would need psutil for real measurement)
        print("\\nMemory usage patterns:")
        print("  torchfits (regular): Allocates full tensor")
        print("  torchfits (mmap):    Fallback to regular (CFITSIO limitation)")
        print("  astropy (memmap):    Memory-mapped view")
        print("  astropy (regular):   Allocates full array")
        
        import os
        os.remove(filename)

def main():
    print("Memory Mapping Benchmark Suite")
    print("Testing torchfits vs astropy vs fitsio")
    
    if not HAS_ASTROPY:
        print("WARNING: astropy not available")
    if not HAS_FITSIO:
        print("WARNING: fitsio not available")
    
    # Create test files
    files = create_test_files()
    
    if files:
        # Run benchmarks
        benchmark_memory_mapping(files)
        benchmark_memory_usage()
        
        # Cleanup
        for filename in files.values():
            filename.unlink()
        filename.parent.rmdir()
    else:
        print("No test files created (astropy required)")

if __name__ == "__main__":
    main()