"""
Zero-Copy Core I/O Optimization Benchmark

This benchmark tests and validates the Core I/O optimizations implemented
as part of the OPTMIZE.md work package #2: True Zero-Copy C++ Core Reader.

Tests the following optimizations:
1. Enhanced tensor allocation with alignment optimization
2. True zero-copy with direct cfitsio pointer passing
3. Device-aware tensor allocation (CPU/CUDA)
4. Memory alignment and contiguous layout optimization
5. Optimized scaling operations
"""

import time
import sys
import tracemalloc
import traceback
from pathlib import Path
import numpy as np
import torch
import tempfile
import os
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torchfits
    from torchfits.core import FITSCore
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


class ZeroCopyBenchmark:
    """Benchmark zero-copy optimizations."""
    
    def __init__(self):
        self.results = []
        self.temp_files = []
    
    def create_test_files(self):
        """Create test files for benchmarking."""
        test_configs = [
            # Small files
            {"shape": (100, 100), "dtype": np.float32, "name": "small_float32"},
            {"shape": (100, 100), "dtype": np.int16, "name": "small_int16"},
            
            # Medium files
            {"shape": (1000, 1000), "dtype": np.float32, "name": "medium_float32"},
            {"shape": (1000, 1000), "dtype": np.int16, "name": "medium_int16"},
            {"shape": (1000, 1000), "dtype": np.float64, "name": "medium_float64"},
            
            # Large files
            {"shape": (2000, 2000), "dtype": np.float32, "name": "large_float32"},
            {"shape": (4000, 2000), "dtype": np.int16, "name": "large_int16"},
            
            # Files with scaling
            {"shape": (1000, 1000), "dtype": np.int16, "name": "scaled_int16", "bscale": 0.01, "bzero": 1000.0},
            {"shape": (2000, 1000), "dtype": np.int16, "name": "large_scaled_int16", "bscale": 0.1, "bzero": 32768.0},
        ]
        
        for config in test_configs:
            shape = config["shape"]
            dtype = config["dtype"]
            name = config["name"]
            
            # Create random data
            if dtype == np.int16:
                if "bscale" in config:
                    # Create data that will use scaling
                    data = np.random.randint(-32768, 32767, shape, dtype=dtype)
                else:
                    data = np.random.randint(0, 1000, shape, dtype=dtype)
            else:
                data = np.random.randn(*shape).astype(dtype)
            
            # Create FITS file
            with tempfile.NamedTemporaryFile(suffix=f'_{name}.fits', delete=False) as f:
                if astropy_fits:
                    hdu = astropy_fits.PrimaryHDU(data)
                    
                    # Add scaling if specified
                    if "bscale" in config:
                        hdu.header["BSCALE"] = config["bscale"]
                        hdu.header["BZERO"] = config["bzero"]
                    
                    hdu.writeto(f.name, overwrite=True)
                    self.temp_files.append((f.name, config))
    
    def benchmark_memory_usage(self, filepath, config):
        """Benchmark memory usage during read operations."""
        if not torchfits:
            return {}
        
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark torchfits memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_memory = process.memory_info().rss / 1024 / 1024
        
        tensor = torchfits.read(filepath, device='cpu')
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = peak_memory - start_memory
        
        # Calculate theoretical minimum (just the tensor size)
        tensor_size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
        memory_efficiency = tensor_size_mb / memory_usage if memory_usage > 0 else 0
        
        del tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "memory_usage_mb": memory_usage,
            "tensor_size_mb": tensor_size_mb,
            "memory_efficiency": memory_efficiency,
            "baseline_memory_mb": baseline_memory
        }
    
    def benchmark_read_performance(self, filepath, config):
        """Benchmark read performance with zero-copy optimizations."""
        name = config["name"]
        shape = config["shape"]
        dtype = config["dtype"]
        
        print(f"\\n🚀 Benchmarking {name} ({shape}, {dtype.__name__})")
        
        results = {
            "name": name,
            "shape": shape,
            "dtype": dtype.__name__,
            "file_size_mb": os.path.getsize(filepath) / 1024 / 1024,
            "has_scaling": "bscale" in config
        }
        
        # Benchmark torchfits CPU performance
        if torchfits:
            times = []
            for _ in range(5):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                start = time.perf_counter()
                tensor_cpu = torchfits.read(filepath, device='cpu')
                end = time.perf_counter()
                times.append(end - start)
            
            results["torchfits_cpu_time"] = np.mean(times)
            results["torchfits_cpu_std"] = np.std(times)
            results["torchfits_shape"] = tuple(tensor_cpu.shape)
            results["torchfits_dtype"] = str(tensor_cpu.dtype)
            
            # Test tensor properties for zero-copy validation
            results["tensor_is_contiguous"] = tensor_cpu.is_contiguous()
            results["tensor_is_pinned"] = tensor_cpu.is_pinned()
            
            # Benchmark torchfits GPU performance if available
            if torch.cuda.is_available():
                times = []
                for _ in range(5):
                    torch.cuda.empty_cache()
                    start = time.perf_counter()
                    tensor_gpu = torchfits.read(filepath, device='cuda')
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                results["torchfits_gpu_time"] = np.mean(times)
                results["torchfits_gpu_std"] = np.std(times)
                results["gpu_tensor_device"] = str(tensor_gpu.device)
        
        # Benchmark astropy for comparison
        if astropy_fits:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                with astropy_fits.open(filepath) as hdul:
                    array = hdul[0].data
                    if array is not None:
                        # Handle byte order issues
                        if array.dtype.byteorder not in ('=', '|'):
                            array = array.astype(array.dtype.newbyteorder('='))
                        tensor = torch.from_numpy(array.copy())
                end = time.perf_counter()
                times.append(end - start)
            
            results["astropy_time"] = np.mean(times)
            results["astropy_std"] = np.std(times)
        
        # Benchmark fitsio for comparison
        if fitsio:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                array = fitsio.read(filepath)
                tensor = torch.from_numpy(array)
                end = time.perf_counter()
                times.append(end - start)
            
            results["fitsio_time"] = np.mean(times)
            results["fitsio_std"] = np.std(times)
        
        # Calculate performance ratios
        if "torchfits_cpu_time" in results:
            if "astropy_time" in results:
                results["speedup_vs_astropy"] = results["astropy_time"] / results["torchfits_cpu_time"]
            if "fitsio_time" in results:
                results["speedup_vs_fitsio"] = results["fitsio_time"] / results["torchfits_cpu_time"]
        
        # Benchmark memory usage
        memory_results = self.benchmark_memory_usage(filepath, config)
        results.update(memory_results)
        
        # Calculate throughput
        total_elements = np.prod(shape)
        if "torchfits_cpu_time" in results:
            results["throughput_cpu_mpixels_sec"] = total_elements / results["torchfits_cpu_time"] / 1e6
        if "torchfits_gpu_time" in results:
            results["throughput_gpu_mpixels_sec"] = total_elements / results["torchfits_gpu_time"] / 1e6
        
        # Print results
        print(f"  File size: {results['file_size_mb']:.1f} MB")
        if "torchfits_cpu_time" in results:
            print(f"  torchfits CPU:  {results['torchfits_cpu_time']*1000:.1f}ms ± {results['torchfits_cpu_std']*1000:.1f}ms")
            print(f"  CPU throughput: {results['throughput_cpu_mpixels_sec']:.1f} Mpixels/sec")
            print(f"  Memory efficiency: {results['memory_efficiency']:.2f} (1.0 = perfect)")
            print(f"  Tensor contiguous: {results['tensor_is_contiguous']}")
        
        if "torchfits_gpu_time" in results:
            print(f"  torchfits GPU:  {results['torchfits_gpu_time']*1000:.1f}ms ± {results['torchfits_gpu_std']*1000:.1f}ms")
            print(f"  GPU throughput: {results['throughput_gpu_mpixels_sec']:.1f} Mpixels/sec")
        
        if "astropy_time" in results:
            print(f"  astropy:        {results['astropy_time']*1000:.1f}ms ± {results['astropy_std']*1000:.1f}ms")
        
        if "fitsio_time" in results:
            print(f"  fitsio:         {results['fitsio_time']*1000:.1f}ms ± {results['fitsio_std']*1000:.1f}ms")
        
        if "speedup_vs_astropy" in results:
            print(f"  Speedup vs astropy: {results['speedup_vs_astropy']:.2f}x")
        if "speedup_vs_fitsio" in results:
            print(f"  Speedup vs fitsio:  {results['speedup_vs_fitsio']:.2f}x")
        
        self.results.append(results)
        return results
    
    def benchmark_device_transfers(self):
        """Benchmark device-aware tensor allocation."""
        if not torch.cuda.is_available() or not torchfits:
            print("\\n⚠️  CUDA not available, skipping device transfer benchmarks")
            return
        
        print("\\n🔄 Benchmarking Device Transfers")
        
        # Create a medium-sized test file
        shape = (1000, 1000)
        data = np.random.randn(*shape).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='_device_test.fits', delete=False) as f:
            if astropy_fits:
                astropy_fits.writeto(f.name, data, overwrite=True)
                
                # Benchmark CPU → GPU transfer
                times = []
                for _ in range(10):
                    torch.cuda.empty_cache()
                    start = time.perf_counter()
                    
                    # Read to CPU first, then transfer
                    tensor_cpu = torchfits.read(f.name, device='cpu')
                    tensor_gpu = tensor_cpu.to('cuda')
                    torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    times.append(end - start)
                
                cpu_gpu_time = np.mean(times)
                
                # Benchmark direct GPU read
                times = []
                for _ in range(10):
                    torch.cuda.empty_cache()
                    start = time.perf_counter()
                    
                    tensor_gpu = torchfits.read(f.name, device='cuda')
                    torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    times.append(end - start)
                
                direct_gpu_time = np.mean(times)
                
                print(f"  CPU → GPU transfer:   {cpu_gpu_time*1000:.1f}ms")
                print(f"  Direct GPU read:      {direct_gpu_time*1000:.1f}ms")
                print(f"  Direct GPU speedup:   {cpu_gpu_time/direct_gpu_time:.2f}x")
                
                os.unlink(f.name)
    
    def analyze_optimization_impact(self):
        """Analyze the impact of zero-copy optimizations."""
        print("\\n📊 Zero-Copy Optimization Analysis")
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Calculate average performance improvements
        speedups_astropy = [r.get("speedup_vs_astropy", 1.0) for r in self.results if "speedup_vs_astropy" in r]
        speedups_fitsio = [r.get("speedup_vs_fitsio", 1.0) for r in self.results if "speedup_vs_fitsio" in r]
        memory_efficiencies = [r.get("memory_efficiency", 0.0) for r in self.results if "memory_efficiency" in r]
        throughputs = [r.get("throughput_cpu_mpixels_sec", 0.0) for r in self.results if "throughput_cpu_mpixels_sec" in r]
        
        print(f"\\n🎯 Performance Summary:")
        if speedups_astropy:
            print(f"  Average speedup vs astropy: {np.mean(speedups_astropy):.2f}x (best: {np.max(speedups_astropy):.2f}x)")
        if speedups_fitsio:
            print(f"  Average speedup vs fitsio:  {np.mean(speedups_fitsio):.2f}x (best: {np.max(speedups_fitsio):.2f}x)")
        if memory_efficiencies:
            print(f"  Average memory efficiency:  {np.mean(memory_efficiencies):.2f} (best: {np.max(memory_efficiencies):.2f})")
        if throughputs:
            print(f"  Average throughput:         {np.mean(throughputs):.1f} Mpixels/sec (peak: {np.max(throughputs):.1f})")
        
        # Identify optimization opportunities
        slow_cases = [r for r in self.results if r.get("speedup_vs_fitsio", 1.0) < 1.0]
        if slow_cases:
            print(f"\\n⚠️  Performance Issues Detected:")
            for result in slow_cases:
                print(f"    {result['name']}: {result.get('speedup_vs_fitsio', 0):.2f}x slower than fitsio")
        
        # Check zero-copy indicators
        contiguous_tensors = sum(1 for r in self.results if r.get("tensor_is_contiguous", False))
        total_tests = len(self.results)
        
        print(f"\\n✅ Zero-Copy Validation:")
        print(f"  Contiguous tensors: {contiguous_tensors}/{total_tests} ({100*contiguous_tensors/total_tests:.1f}%)")
        
        # Memory efficiency analysis
        efficient_cases = len([r for r in memory_efficiencies if r > 0.8])
        print(f"  High memory efficiency (>80%): {efficient_cases}/{len(memory_efficiencies)} cases")
        
        return {
            "avg_speedup_astropy": np.mean(speedups_astropy) if speedups_astropy else 0,
            "avg_speedup_fitsio": np.mean(speedups_fitsio) if speedups_fitsio else 0,
            "avg_memory_efficiency": np.mean(memory_efficiencies) if memory_efficiencies else 0,
            "avg_throughput": np.mean(throughputs) if throughputs else 0,
            "contiguous_percentage": 100*contiguous_tensors/total_tests if total_tests > 0 else 0
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        for filepath, _ in self.temp_files:
            try:
                os.unlink(filepath)
            except:
                pass
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive zero-copy optimization benchmark."""
        print("🚀 Zero-Copy Core I/O Optimization Benchmark")
        print("=" * 60)
        
        if not torchfits:
            print("❌ torchfits not available - cannot run benchmark")
            return None
        
        try:
            # Create test files
            print("📝 Creating test files...")
            self.create_test_files()
            print(f"Created {len(self.temp_files)} test files")
            
            # Run benchmarks for each file
            for filepath, config in self.temp_files:
                self.benchmark_read_performance(filepath, config)
            
            # Test device transfers
            self.benchmark_device_transfers()
            
            # Analyze results
            summary = self.analyze_optimization_impact()
            
            return summary
            
        finally:
            self.cleanup()


def main():
    """Run zero-copy optimization benchmark."""
    benchmark = ZeroCopyBenchmark()
    
    try:
        summary = benchmark.run_comprehensive_benchmark()
        
        if summary:
            print(f"\\n💾 Benchmark completed successfully!")
            print(f"Key metrics:")
            print(f"  - Average speedup vs astropy: {summary['avg_speedup_astropy']:.2f}x")
            print(f"  - Average speedup vs fitsio: {summary['avg_speedup_fitsio']:.2f}x") 
            print(f"  - Average memory efficiency: {summary['avg_memory_efficiency']:.2f}")
            print(f"  - Average throughput: {summary['avg_throughput']:.1f} Mpixels/sec")
            print(f"  - Contiguous tensor rate: {summary['contiguous_percentage']:.1f}%")
            
            # Determine if optimizations are successful
            success_criteria = {
                "speedup_vs_fitsio": summary['avg_speedup_fitsio'] >= 1.0,  # At least as fast as fitsio
                "memory_efficiency": summary['avg_memory_efficiency'] >= 0.7,  # Good memory efficiency
                "contiguous_tensors": summary['contiguous_percentage'] >= 90.0  # Most tensors contiguous
            }
            
            if all(success_criteria.values()):
                print("\\n✅ Zero-copy optimizations are working correctly!")
            else:
                print("\\n⚠️  Some optimization issues detected:")
                for criterion, passed in success_criteria.items():
                    if not passed:
                        print(f"    - {criterion}: NEEDS IMPROVEMENT")
                        
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()