"""
GPU Memory Usage Validation Benchmark

Tests GPU memory efficiency and validates direct GPU loading capabilities.
"""

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

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


class GPUMemoryBenchmark:
    """Benchmark GPU memory usage and efficiency."""

    def __init__(self):
        self.results = []
        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.device = torch.device("cuda")
            print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
            print(
                f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            print("⚠️  CUDA not available - GPU tests will be skipped")

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB."""
        if not self.cuda_available:
            return 0, 0

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2  # MB
        return allocated, cached

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            gc.collect()

    def create_test_file(self, shape, dtype=np.float32):
        """Create test FITS file."""
        data = np.random.normal(0, 1, shape).astype(dtype)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            if astropy_fits:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.writeto(f.name, overwrite=True)
                return f.name, data.nbytes / 1024**2  # Return filepath and size in MB
            else:
                raise ImportError("astropy required for test file creation")

    def benchmark_direct_gpu_loading(self):
        """Benchmark direct GPU loading vs CPU->GPU transfer."""
        if not self.cuda_available or not torchfits:
            print("⚠️  Skipping GPU loading test (CUDA or torchfits not available)")
            return

        print("\\n🎯 Testing Direct GPU Loading")

        test_sizes = [
            (1000, 1000),  # 4MB
            (2000, 2000),  # 16MB
            (3000, 3000),  # 36MB
        ]

        for shape in test_sizes:
            filepath, file_size_mb = self.create_test_file(shape)

            try:
                print(f"\\n  Testing {shape} ({file_size_mb:.1f} MB)")

                # Clear memory before test
                self.clear_gpu_memory()
                mem_before, _ = self.get_gpu_memory_usage()

                # Method 1: Direct GPU loading (if supported)
                try:
                    import time

                    start_time = time.perf_counter()
                    gpu_tensor = torchfits.read(filepath, device="cuda")
                    direct_time = time.perf_counter() - start_time

                    mem_after_direct, _ = self.get_gpu_memory_usage()
                    direct_memory = mem_after_direct - mem_before

                    print(
                        f"    Direct GPU load: {direct_time*1000:.1f}ms, {direct_memory:.1f}MB GPU memory"
                    )

                    # Verify data is on GPU
                    assert gpu_tensor.device.type == "cuda"

                    del gpu_tensor
                    self.clear_gpu_memory()

                except Exception as e:
                    print(f"    Direct GPU load failed: {e}")
                    direct_time = None
                    direct_memory = None

                # Method 2: CPU load + GPU transfer
                self.clear_gpu_memory()
                mem_before, _ = self.get_gpu_memory_usage()

                start_time = time.perf_counter()
                cpu_tensor = torchfits.read(filepath)
                gpu_tensor = cpu_tensor.cuda()
                transfer_time = time.perf_counter() - start_time

                mem_after_transfer, _ = self.get_gpu_memory_usage()
                transfer_memory = mem_after_transfer - mem_before

                print(
                    f"    CPU->GPU transfer: {transfer_time*1000:.1f}ms, {transfer_memory:.1f}MB GPU memory"
                )

                # Compare methods
                if direct_time is not None:
                    speedup = transfer_time / direct_time
                    memory_efficiency = (
                        transfer_memory / direct_memory if direct_memory > 0 else 1.0
                    )
                    print(
                        f"    Direct loading: {speedup:.2f}x faster, {memory_efficiency:.2f}x memory efficient"
                    )

                del cpu_tensor, gpu_tensor
                self.clear_gpu_memory()

            finally:
                os.unlink(filepath)

    def benchmark_gpu_memory_efficiency(self):
        """Test GPU memory efficiency with large files."""
        if not self.cuda_available or not torchfits:
            print("⚠️  Skipping GPU memory efficiency test")
            return

        print("\\n💾 Testing GPU Memory Efficiency")

        # Test with progressively larger files
        shapes = [
            (2000, 2000),  # 16MB
            (4000, 4000),  # 64MB
            (6000, 6000),  # 144MB
        ]

        for shape in shapes:
            filepath, file_size_mb = self.create_test_file(shape)

            try:
                print(f"\\n  Testing {shape} ({file_size_mb:.1f} MB file)")

                self.clear_gpu_memory()
                mem_before, _ = self.get_gpu_memory_usage()

                # Load to GPU
                tensor = torchfits.read(
                    filepath, device="cuda" if self.cuda_available else "cpu"
                )

                mem_after, _ = self.get_gpu_memory_usage()
                memory_used = mem_after - mem_before

                # Calculate efficiency
                theoretical_memory = file_size_mb  # Minimum memory needed
                efficiency = theoretical_memory / memory_used if memory_used > 0 else 0

                print(f"    File size: {file_size_mb:.1f} MB")
                print(f"    GPU memory used: {memory_used:.1f} MB")
                print(f"    Memory efficiency: {efficiency:.2f} (1.0 = perfect)")

                # Test operations on GPU tensor
                if tensor.device.type == "cuda":
                    # Test basic operations
                    mean_val = tensor.mean()
                    std_val = tensor.std()
                    print(
                        f"    GPU operations work: mean={mean_val:.3f}, std={std_val:.3f}"
                    )

                del tensor
                self.clear_gpu_memory()

            finally:
                os.unlink(filepath)

    def benchmark_gpu_transforms(self):
        """Test GPU transforms performance."""
        if not self.cuda_available:
            print("⚠️  Skipping GPU transforms test")
            return

        print("\\n🎨 Testing GPU Transforms")

        try:
            from torchfits.transforms import Compose, RandomFlip, ZScale
        except ImportError:
            print("⚠️  Transforms not available")
            return

        shape = (2000, 2000)
        filepath, file_size_mb = self.create_test_file(shape)

        try:
            # Create transform pipeline
            transform = Compose([ZScale(), RandomFlip(p=0.5)])

            print(f"  Testing transforms on {shape} ({file_size_mb:.1f} MB)")

            # Test CPU transforms
            self.clear_gpu_memory()
            cpu_tensor = torchfits.read(filepath)

            import time

            start_time = time.perf_counter()
            cpu_result = transform(cpu_tensor)
            cpu_time = time.perf_counter() - start_time

            print(f"    CPU transforms: {cpu_time*1000:.1f}ms")

            # Test GPU transforms
            self.clear_gpu_memory()
            mem_before, _ = self.get_gpu_memory_usage()

            gpu_tensor = cpu_tensor.cuda()

            start_time = time.perf_counter()
            gpu_result = transform(gpu_tensor)
            gpu_time = time.perf_counter() - start_time

            mem_after, _ = self.get_gpu_memory_usage()
            gpu_memory = mem_after - mem_before

            print(
                f"    GPU transforms: {gpu_time*1000:.1f}ms, {gpu_memory:.1f}MB GPU memory"
            )

            # Compare performance
            speedup = cpu_time / gpu_time
            print(f"    GPU speedup: {speedup:.2f}x")

            # Verify results are similar (transforms may be stochastic)
            assert gpu_result.device.type == "cuda"
            assert gpu_result.shape == cpu_result.shape

        finally:
            os.unlink(filepath)

    def benchmark_memory_leaks(self):
        """Test for GPU memory leaks."""
        if not self.cuda_available or not torchfits:
            print("⚠️  Skipping memory leak test")
            return

        print("\\n🔍 Testing GPU Memory Leaks")

        shape = (1000, 1000)
        filepath, file_size_mb = self.create_test_file(shape)

        try:
            self.clear_gpu_memory()
            initial_memory, _ = self.get_gpu_memory_usage()

            # Perform multiple load/delete cycles
            n_cycles = 10
            for i in range(n_cycles):
                tensor = torchfits.read(
                    filepath, device="cuda" if self.cuda_available else "cpu"
                )

                # Do some operations
                _ = tensor.mean()
                _ = tensor.std()

                del tensor

                if i % 3 == 0:  # Periodic cleanup
                    self.clear_gpu_memory()

            # Final cleanup
            self.clear_gpu_memory()
            final_memory, _ = self.get_gpu_memory_usage()

            memory_leak = final_memory - initial_memory
            print(f"    Initial GPU memory: {initial_memory:.1f} MB")
            print(f"    Final GPU memory: {final_memory:.1f} MB")
            print(f"    Memory leak: {memory_leak:.1f} MB after {n_cycles} cycles")

            if memory_leak > 10:  # More than 10MB leak is concerning
                print("    ⚠️  Potential memory leak detected!")
            else:
                print("    ✅ No significant memory leak detected")

        finally:
            os.unlink(filepath)

    def run_comprehensive_benchmark(self):
        """Run comprehensive GPU memory benchmark."""
        print("🚀 GPU Memory Usage Validation Benchmark")
        print("=" * 50)

        if not self.cuda_available:
            print("❌ CUDA not available - skipping GPU tests")
            return

        if not torchfits:
            print("❌ torchfits not available - cannot run benchmark")
            return

        # Run all GPU tests
        self.benchmark_direct_gpu_loading()
        self.benchmark_gpu_memory_efficiency()
        self.benchmark_gpu_transforms()
        self.benchmark_memory_leaks()

        print("\\n✅ GPU Memory Benchmark Complete")


def main():
    """Run GPU memory benchmark."""
    benchmark = GPUMemoryBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
