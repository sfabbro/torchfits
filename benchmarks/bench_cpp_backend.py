"""
C++ Backend Performance Benchmark

Focused benchmark to identify and fix C++ backend performance issues.
Compares current implementation against astropy/fitsio to find bottlenecks.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torchfits
    from torchfits.core import FITSCore  # noqa: F401
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


class CPPBackendBenchmark:
    """Benchmark C++ backend performance against reference implementations."""

    def __init__(self):
        self.results = []

    def create_test_data(self, shape, dtype=np.float32, add_scaling=False):
        """Create test data with optional FITS scaling."""
        data = np.random.normal(1000, 100, shape).astype(dtype)

        header_kwargs = {}
        if add_scaling:
            header_kwargs["BSCALE"] = 0.01
            header_kwargs["BZERO"] = 1000.0
            # Convert to integer for scaling test
            data = ((data - 1000.0) / 0.01).astype(np.int16)

        return data, header_kwargs

    def write_test_file(self, data, header_kwargs=None, compressed=False):
        """Write test FITS file."""
        header_kwargs = header_kwargs or {}

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            if astropy_fits:
                if compressed:
                    hdu = astropy_fits.CompImageHDU(data, compression_type="RICE_1")
                else:
                    hdu = astropy_fits.PrimaryHDU(data)

                for key, value in header_kwargs.items():
                    hdu.header[key] = value

                hdu.writeto(f.name, overwrite=True)
                return f.name
            else:
                raise ImportError("astropy required for test file creation")

    def bench_read_performance(
        self, shape, dtype=np.float32, compressed=False, scaled=False
    ):
        """Benchmark read performance for specific configuration."""
        print(
            f"\\n📊 Benchmarking {shape} {dtype.__name__} {'compressed' if compressed else 'uncompressed'} {'scaled' if scaled else 'raw'}"
        )

        # Create test data
        data, header_kwargs = self.create_test_data(shape, dtype, scaled)
        filepath = self.write_test_file(data, header_kwargs, compressed)

        results = {
            "shape": shape,
            "dtype": dtype.__name__,
            "compressed": compressed,
            "scaled": scaled,
            "file_size_mb": os.path.getsize(filepath) / 1024 / 1024,
        }

        try:
            # Benchmark torchfits
            if torchfits:
                times = []
                tensor = None
                for _ in range(3):
                    start = time.perf_counter()
                    res = torchfits.read(filepath)
                    if isinstance(res, tuple):
                        tensor = res[0]
                    else:
                        tensor = res
                    end = time.perf_counter()
                    times.append(end - start)

                results["torchfits_time"] = np.mean(times)
                results["torchfits_std"] = np.std(times)
                if tensor is not None:
                    results["torchfits_shape"] = tuple(tensor.shape)
                    results["torchfits_dtype"] = str(tensor.dtype)

            # Benchmark astropy
            if astropy_fits:
                times = []
                for _ in range(3):
                    start = time.perf_counter()
                    with astropy_fits.open(filepath) as hdul:
                        array = hdul[0].data
                        if array is not None:
                            # Ensure native byte order for PyTorch
                            if array.dtype.byteorder not in ("=", "|"):
                                array = array.astype(array.dtype.newbyteorder("="))
                            tensor = torch.from_numpy(array)
                    end = time.perf_counter()
                    times.append(end - start)

                results["astropy_time"] = np.mean(times)
                results["astropy_std"] = np.std(times)

            # Benchmark fitsio
            if fitsio:
                times = []
                for _ in range(3):
                    start = time.perf_counter()
                    array = fitsio.read(filepath)
                    # Ensure native byte order for PyTorch
                    if array.dtype.byteorder not in ("=", "|"):
                        array = array.astype(array.dtype.newbyteorder("="))
                    tensor = torch.from_numpy(array)
                    end = time.perf_counter()
                    times.append(end - start)

                results["fitsio_time"] = np.mean(times)
                results["fitsio_std"] = np.std(times)

            # Calculate performance ratios
            if "torchfits_time" in results:
                if "astropy_time" in results:
                    results["vs_astropy"] = (
                        results["astropy_time"] / results["torchfits_time"]
                    )
                if "fitsio_time" in results:
                    results["vs_fitsio"] = (
                        results["fitsio_time"] / results["torchfits_time"]
                    )

            # Print results
            print(f"  File size: {results['file_size_mb']:.1f} MB")
            if "torchfits_time" in results:
                print(
                    f"  torchfits: {results['torchfits_time'] * 1000:.1f}ms ± {results['torchfits_std'] * 1000:.1f}ms"
                )
            if "astropy_time" in results:
                print(
                    f"  astropy:   {results['astropy_time'] * 1000:.1f}ms ± {results['astropy_std'] * 1000:.1f}ms"
                )
            if "fitsio_time" in results:
                print(
                    f"  fitsio:    {results['fitsio_time'] * 1000:.1f}ms ± {results['fitsio_std'] * 1000:.1f}ms"
                )

            if "vs_astropy" in results:
                print(
                    f"  torchfits vs astropy: {results['vs_astropy']:.2f}x {'faster' if results['vs_astropy'] > 1 else 'slower'}"
                )
            if "vs_fitsio" in results:
                print(
                    f"  torchfits vs fitsio:  {results['vs_fitsio']:.2f}x {'faster' if results['vs_fitsio'] > 1 else 'slower'}"
                )

            self.results.append(results)

        finally:
            os.unlink(filepath)

    def bench_cutout_performance(self):
        """Benchmark cutout/subset reading performance."""
        print("\\n🔍 Benchmarking Cutout Performance")

        # Large image for cutout testing
        shape = (4000, 4000)
        data, _ = self.create_test_data(shape, np.float32)
        filepath = self.write_test_file(data)

        cutout_spec = f"{filepath}[0][1000:2000,1000:2000]"

        try:
            if torchfits:
                times = []
                subset = None
                for _ in range(5):
                    start = time.perf_counter()
                    res = torchfits.read(cutout_spec)
                    if isinstance(res, tuple):
                        subset = res[0]
                    else:
                        subset = res
                    end = time.perf_counter()
                    times.append(end - start)

                print(
                    f"  torchfits cutout: {np.mean(times) * 1000:.1f}ms ± {np.std(times) * 1000:.1f}ms"
                )
                if subset is not None:
                    print(f"  cutout shape: {subset.shape}")

            # Compare with full read + slice
            if astropy_fits:
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    with astropy_fits.open(filepath) as hdul:
                        full_data = hdul[0].data
                        subset = full_data[1000:2000, 1000:2000]
                        if subset.dtype.byteorder not in ("=", "|"):
                            subset = subset.astype(subset.dtype.newbyteorder("="))
                        torch.from_numpy(subset)
                    end = time.perf_counter()
                    times.append(end - start)

                print(
                    f"  astropy full+slice: {np.mean(times) * 1000:.1f}ms ± {np.std(times) * 1000:.1f}ms"
                )

        finally:
            os.unlink(filepath)

    def bench_scaling_performance(self):
        """Benchmark BSCALE/BZERO scaling performance."""
        print("\\n⚖️  Benchmarking Scaling Performance")

        shape = (2000, 2000)

        # Test with scaling
        self.bench_read_performance(shape, np.int16, compressed=False, scaled=True)

        # Test without scaling for comparison
        self.bench_read_performance(
            shape, np.float32, compressed=False, scaled=False
        )

    def identify_bottlenecks(self):
        """Analyze results to identify performance bottlenecks."""
        print("\\n🔍 Performance Analysis")

        slow_cases = []
        for result in self.results:
            if "vs_astropy" in result and result["vs_astropy"] < 1.0:
                slow_cases.append((result, "astropy", result["vs_astropy"]))
            if "vs_fitsio" in result and result["vs_fitsio"] < 1.0:
                slow_cases.append((result, "fitsio", result["vs_fitsio"]))

        if slow_cases:
            print("\\n⚠️  Performance Issues Identified:")
            for result, vs_lib, ratio in slow_cases:
                print(
                    f"  {result['shape']} {result['dtype']} {'compressed' if result['compressed'] else 'uncompressed'}: "
                    f"{ratio:.2f}x slower than {vs_lib}"
                )

        # Identify patterns
        compressed_slow = any(
            r["compressed"] and "vs_fitsio" in r and r["vs_fitsio"] < 1.0
            for r in self.results
        )
        large_file_slow = any(
            r["file_size_mb"] > 50 and "vs_fitsio" in r and r["vs_fitsio"] < 1.0
            for r in self.results
        )
        scaling_slow = any(
            r["scaled"] and "vs_fitsio" in r and r["vs_fitsio"] < 1.0
            for r in self.results
        )

        print("\\n🎯 Optimization Targets:")
        if compressed_slow:
            print("  - Compressed file reading needs optimization")
        if large_file_slow:
            print("  - Large file I/O needs optimization")
        if scaling_slow:
            print("  - BSCALE/BZERO scaling needs optimization")

        # Calculate overall performance
        if self.results:
            avg_vs_astropy = np.mean(
                [r.get("vs_astropy", 1.0) for r in self.results if "vs_astropy" in r]
            )
            avg_vs_fitsio = np.mean(
                [r.get("vs_fitsio", 1.0) for r in self.results if "vs_fitsio" in r]
            )

            print("\\n📈 Overall Performance:")
            print(f"  Average vs astropy: {avg_vs_astropy:.2f}x")
            print(f"  Average vs fitsio:  {avg_vs_fitsio:.2f}x")

    def run_comprehensive_benchmark(self):
        """Run comprehensive C++ backend benchmark."""
        print("🚀 C++ Backend Performance Benchmark")
        print("=" * 50)

        if not torchfits:
            print("❌ torchfits not available - cannot run benchmark")
            return

        # Test different data sizes and types
        test_configs = [
            # Small files
            ((100, 100), np.float32, False),
            ((100, 100), np.int16, False),
            # Medium files
            ((1000, 1000), np.float32, False),
            ((1000, 1000), np.int16, False),
            ((1000, 1000), np.float64, False),
            # Large files
            ((2000, 2000), np.float32, False),
            ((2000, 2000), np.int16, False),
            # Compressed files
            ((1000, 1000), np.float32, True),
            ((1000, 1000), np.int16, True),
        ]

        for shape, dtype, compressed in test_configs:
            self.bench_read_performance(shape, dtype, compressed)

        # Test cutouts
        self.bench_cutout_performance()

        # Test scaling
        self.bench_scaling_performance()

        # Analyze results
        self.identify_bottlenecks()

        return self.results


def main():
    """Run C++ backend benchmark."""
    benchmark = CPPBackendBenchmark()
    results = benchmark.run_comprehensive_benchmark()

    # Save results
    import json

    output_file = (
        Path(__file__).parent.parent / "bench_results" / "cpp_backend_results.json"
    )
    output_file.parent.mkdir(exist_ok=True)

    # Convert numpy types to JSON serializable
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif hasattr(value, "item"):  # numpy scalar
                json_result[key] = value.item()
            else:
                json_result[key] = value
        json_results.append(json_result)

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
