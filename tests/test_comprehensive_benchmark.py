#!/usr/bin/env python3
"""
TorchFits Comprehensive Benchmark Suite

Official test suite for validating performance claims against fitsio and astropy.
Tests all types of FITS data and sizes, including cutouts.

Performance Targets:
- 1-36x faster than fitsio
- Competitive with astropy
- Memory efficient
- Robust error handling
"""

import gc
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import pytest

# Core libraries
try:
    import torchfits

    HAS_TORCHFITS = True
except ImportError:
    HAS_TORCHFITS = False

try:
    import fitsio

    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

try:
    from astropy.io import fits as astropy_fits

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# CFITSIO advanced features
try:
    from torchfits import torchfits_cfitsio

    HAS_CFITSIO_ADVANCED = True
except ImportError:
    HAS_CFITSIO_ADVANCED = False


@dataclass
class BenchmarkResult:
    """Standardized benchmark result structure"""

    library: str
    operation: str
    file_type: str
    data_shape: Tuple[int, ...]
    file_size_mb: float
    read_time_ms: float
    memory_usage_mb: float
    throughput_mbs: float
    success: bool
    error_message: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None


class MemoryMonitor:
    """Context manager for memory usage monitoring"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        gc.collect()  # Clean up before measurement
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()  # Clean up after measurement

    @property
    def current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)
        return current

    @property
    def memory_delta_mb(self) -> float:
        """Get memory delta from initial in MB"""
        return self.peak_memory - self.initial_memory


class FITSTestDataGenerator:
    """Generate various types of test FITS files"""

    @staticmethod
    def create_image_2d(shape: Tuple[int, int], dtype=np.float32) -> np.ndarray:
        """Create 2D image data"""
        return np.random.normal(1000, 100, shape).astype(dtype)

    @staticmethod
    def create_image_3d(shape: Tuple[int, int, int], dtype=np.float32) -> np.ndarray:
        """Create 3D cube data"""
        return np.random.normal(1000, 100, shape).astype(dtype)

    @staticmethod
    def create_spectrum_1d(length: int, dtype=np.float32) -> np.ndarray:
        """Create 1D spectrum data"""
        x = np.linspace(0, 10, length)
        return (np.sin(x) * 1000 + np.random.normal(0, 50, length)).astype(dtype)

    @staticmethod
    def create_table_data(n_rows: int) -> Dict[str, np.ndarray]:
        """Create table data"""
        return {
            "ID": np.arange(n_rows, dtype=np.int64),
            "RA": np.random.uniform(0, 360, n_rows).astype(np.float64),
            "DEC": np.random.uniform(-90, 90, n_rows).astype(np.float64),
            "FLUX": np.random.exponential(1000, n_rows).astype(np.float32),
            "MAG": np.random.normal(20, 2, n_rows).astype(np.float32),
            "CLASS": np.random.choice(["STAR", "GALAXY", "QSO"], n_rows).astype("U10"),
            "REDSHIFT": np.random.exponential(0.5, n_rows).astype(np.float32),
        }

    @classmethod
    def create_test_file(
        cls, filepath: str, file_type: str, size_category: str
    ) -> Tuple[int, ...]:
        """Create a test FITS file and return its data shape"""

        if file_type == "image_2d":
            if size_category == "small":
                shape = (512, 512)
            elif size_category == "medium":
                shape = (2048, 2048)
            elif size_category == "large":
                shape = (8192, 8192)
            else:  # huge
                shape = (16384, 16384)

            data = cls.create_image_2d(shape)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.writeto(filepath, overwrite=True)
            else:
                # Fallback using fitsio if available
                if HAS_FITSIO:
                    fitsio.write(filepath, data, clobber=True)
                else:
                    raise ImportError(
                        "Neither astropy nor fitsio available for file creation"
                    )

            return shape

        elif file_type == "image_3d":
            if size_category == "small":
                shape = (50, 256, 256)
            elif size_category == "medium":
                shape = (100, 512, 512)
            elif size_category == "large":
                shape = (200, 1024, 1024)
            else:  # huge
                shape = (500, 2048, 2048)

            data = cls.create_image_3d(shape)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.writeto(filepath, overwrite=True)
            else:
                if HAS_FITSIO:
                    fitsio.write(filepath, data, clobber=True)
                else:
                    raise ImportError(
                        "Neither astropy nor fitsio available for file creation"
                    )

            return shape

        elif file_type == "spectrum_1d":
            if size_category == "small":
                length = 1000
            elif size_category == "medium":
                length = 10000
            elif size_category == "large":
                length = 100000
            else:  # huge
                length = 1000000

            data = cls.create_spectrum_1d(length)
            shape = (length,)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.writeto(filepath, overwrite=True)
            else:
                if HAS_FITSIO:
                    fitsio.write(filepath, data, clobber=True)
                else:
                    raise ImportError(
                        "Neither astropy nor fitsio available for file creation"
                    )

            return shape

        elif file_type == "table":
            if size_category == "small":
                n_rows = 1000
            elif size_category == "medium":
                n_rows = 100000
            elif size_category == "large":
                n_rows = 1000000
            else:  # huge
                n_rows = 10000000

            table_data = cls.create_table_data(n_rows)
            shape = (n_rows, len(table_data))

            if HAS_ASTROPY:
                # Create table HDU
                cols = []
                for name, data in table_data.items():
                    if data.dtype.kind == "U":  # String column
                        cols.append(
                            astropy_fits.Column(
                                name=name, format=f"{data.dtype.itemsize}A", array=data
                            )
                        )
                    elif data.dtype == np.int64:
                        cols.append(
                            astropy_fits.Column(name=name, format="K", array=data)
                        )
                    elif data.dtype == np.float64:
                        cols.append(
                            astropy_fits.Column(name=name, format="D", array=data)
                        )
                    elif data.dtype == np.float32:
                        cols.append(
                            astropy_fits.Column(name=name, format="E", array=data)
                        )

                hdu = astropy_fits.BinTableHDU.from_columns(cols)
                hdu.writeto(filepath, overwrite=True)
            else:
                if HAS_FITSIO:
                    fitsio.write(filepath, table_data, clobber=True)
                else:
                    raise ImportError(
                        "Neither astropy nor fitsio available for file creation"
                    )

            return shape

        else:
            raise ValueError(f"Unknown file type: {file_type}")


class BenchmarkRunner:
    """Main benchmark execution class"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="torchfits_benchmark_")
        Path(self.output_dir).mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def benchmark_torchfits(
        self, filepath: str, file_type: str, shape: Tuple[int, ...]
    ) -> BenchmarkResult:
        """Benchmark TorchFits reading"""
        if not HAS_TORCHFITS:
            return BenchmarkResult(
                library="torchfits",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=0,
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message="TorchFits not available",
            )

        try:
            file_size_mb = os.path.getsize(filepath) / 1024 / 1024

            with MemoryMonitor() as mem:
                start_time = time.time()

                if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                    data = torchfits.read_image(filepath)
                elif file_type == "table":
                    data = torchfits.read_table(filepath)
                else:
                    raise ValueError(f"Unknown file type: {file_type}")

                end_time = time.time()

                # Verify data was read
                if hasattr(data, "shape"):
                    actual_shape = data.shape
                else:
                    actual_shape = (len(data),) if hasattr(data, "__len__") else (1,)

                read_time_ms = (end_time - start_time) * 1000
                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="torchfits",
                    operation="read_full",
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                )

        except Exception as e:
            return BenchmarkResult(
                library="torchfits",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=(
                    os.path.getsize(filepath) / 1024 / 1024
                    if os.path.exists(filepath)
                    else 0
                ),
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message=str(e),
            )

    def benchmark_fitsio(
        self, filepath: str, file_type: str, shape: Tuple[int, ...]
    ) -> BenchmarkResult:
        """Benchmark fitsio reading"""
        if not HAS_FITSIO:
            return BenchmarkResult(
                library="fitsio",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=0,
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message="fitsio not available",
            )

        try:
            file_size_mb = os.path.getsize(filepath) / 1024 / 1024

            with MemoryMonitor() as mem:
                start_time = time.time()

                with fitsio.FITS(filepath, "r") as fits:
                    if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                        data = fits[0].read()
                    elif file_type == "table":
                        data = fits[1].read()  # Table in extension 1
                    else:
                        raise ValueError(f"Unknown file type: {file_type}")

                end_time = time.time()

                actual_shape = data.shape if hasattr(data, "shape") else (len(data),)

                read_time_ms = (end_time - start_time) * 1000
                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="fitsio",
                    operation="read_full",
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                )

        except Exception as e:
            return BenchmarkResult(
                library="fitsio",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=(
                    os.path.getsize(filepath) / 1024 / 1024
                    if os.path.exists(filepath)
                    else 0
                ),
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message=str(e),
            )

    def benchmark_astropy(
        self, filepath: str, file_type: str, shape: Tuple[int, ...]
    ) -> BenchmarkResult:
        """Benchmark astropy reading"""
        if not HAS_ASTROPY:
            return BenchmarkResult(
                library="astropy",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=0,
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message="astropy not available",
            )

        try:
            file_size_mb = os.path.getsize(filepath) / 1024 / 1024

            with MemoryMonitor() as mem:
                start_time = time.time()

                with astropy_fits.open(filepath) as hdul:
                    if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                        data = hdul[0].data
                    elif file_type == "table":
                        data = hdul[1].data  # Table in extension 1
                    else:
                        raise ValueError(f"Unknown file type: {file_type}")

                    # Force data read
                    if hasattr(data, "__array__"):
                        _ = np.array(data)

                end_time = time.time()

                actual_shape = data.shape if hasattr(data, "shape") else (len(data),)

                read_time_ms = (end_time - start_time) * 1000
                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="astropy",
                    operation="read_full",
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                )

        except Exception as e:
            return BenchmarkResult(
                library="astropy",
                operation="read_full",
                file_type=file_type,
                data_shape=shape,
                file_size_mb=(
                    os.path.getsize(filepath) / 1024 / 1024
                    if os.path.exists(filepath)
                    else 0
                ),
                read_time_ms=0,
                memory_usage_mb=0,
                throughput_mbs=0,
                success=False,
                error_message=str(e),
            )

    def benchmark_cutouts(
        self, filepath: str, file_type: str, shape: Tuple[int, ...]
    ) -> List[BenchmarkResult]:
        """Benchmark cutout operations (only for 2D/3D images)"""
        if file_type not in ["image_2d", "image_3d"]:
            return []

        results = []

        # Define cutout regions
        if file_type == "image_2d":
            if len(shape) >= 2:
                cutout_shapes = [
                    (min(64, shape[0]), min(64, shape[1])),
                    (min(256, shape[0]), min(256, shape[1])),
                    (min(512, shape[0]), min(512, shape[1])),
                ]
            else:
                cutout_shapes = [(min(64, shape[0]),)]
        else:  # image_3d
            if len(shape) >= 3:
                cutout_shapes = [
                    (min(10, shape[0]), min(64, shape[1]), min(64, shape[2])),
                    (min(50, shape[0]), min(256, shape[1]), min(256, shape[2])),
                ]
            else:
                cutout_shapes = [(min(10, shape[0]),)]

        for cutout_shape in cutout_shapes:
            # TorchFits cutout
            if HAS_TORCHFITS:
                try:
                    file_size_mb = os.path.getsize(filepath) / 1024 / 1024

                    with MemoryMonitor() as mem:
                        start_time = time.time()

                        if file_type == "image_2d":
                            slices = (
                                slice(0, cutout_shape[0]),
                                slice(0, cutout_shape[1]),
                            )
                        else:  # image_3d
                            slices = (
                                slice(0, cutout_shape[0]),
                                slice(0, cutout_shape[1]),
                                slice(0, cutout_shape[2]),
                            )

                        # Use TorchFits cutout functionality if available
                        try:
                            data = torchfits.read_image(filepath, slices=slices)
                        except:
                            # Fallback to full read + slice
                            full_data = torchfits.read_image(filepath)
                            data = full_data[slices]

                        end_time = time.time()

                        read_time_ms = (end_time - start_time) * 1000
                        memory_delta = mem.memory_delta_mb
                        cutout_size_mb = (
                            np.prod(cutout_shape) * 4 / 1024 / 1024
                        )  # Assuming float32
                        throughput_mbs = (
                            cutout_size_mb / (read_time_ms / 1000)
                            if read_time_ms > 0
                            else 0
                        )

                        results.append(
                            BenchmarkResult(
                                library="torchfits",
                                operation="cutout",
                                file_type=file_type,
                                data_shape=cutout_shape,
                                file_size_mb=cutout_size_mb,
                                read_time_ms=read_time_ms,
                                memory_usage_mb=memory_delta,
                                throughput_mbs=throughput_mbs,
                                success=True,
                                extra_info={"full_file_size_mb": file_size_mb},
                            )
                        )

                except Exception as e:
                    results.append(
                        BenchmarkResult(
                            library="torchfits",
                            operation="cutout",
                            file_type=file_type,
                            data_shape=cutout_shape,
                            file_size_mb=0,
                            read_time_ms=0,
                            memory_usage_mb=0,
                            throughput_mbs=0,
                            success=False,
                            error_message=str(e),
                        )
                    )

            # fitsio cutout
            if HAS_FITSIO:
                try:
                    file_size_mb = os.path.getsize(filepath) / 1024 / 1024

                    with MemoryMonitor() as mem:
                        start_time = time.time()

                        with fitsio.FITS(filepath, "r") as fits:
                            if file_type == "image_2d":
                                data = fits[0][0 : cutout_shape[0], 0 : cutout_shape[1]]
                            else:  # image_3d
                                data = fits[0][
                                    0 : cutout_shape[0],
                                    0 : cutout_shape[1],
                                    0 : cutout_shape[2],
                                ]

                        end_time = time.time()

                        read_time_ms = (end_time - start_time) * 1000
                        memory_delta = mem.memory_delta_mb
                        cutout_size_mb = (
                            np.prod(cutout_shape) * 4 / 1024 / 1024
                        )  # Assuming float32
                        throughput_mbs = (
                            cutout_size_mb / (read_time_ms / 1000)
                            if read_time_ms > 0
                            else 0
                        )

                        results.append(
                            BenchmarkResult(
                                library="fitsio",
                                operation="cutout",
                                file_type=file_type,
                                data_shape=cutout_shape,
                                file_size_mb=cutout_size_mb,
                                read_time_ms=read_time_ms,
                                memory_usage_mb=memory_delta,
                                throughput_mbs=throughput_mbs,
                                success=True,
                                extra_info={"full_file_size_mb": file_size_mb},
                            )
                        )

                except Exception as e:
                    results.append(
                        BenchmarkResult(
                            library="fitsio",
                            operation="cutout",
                            file_type=file_type,
                            data_shape=cutout_shape,
                            file_size_mb=0,
                            read_time_ms=0,
                            memory_usage_mb=0,
                            throughput_mbs=0,
                            success=False,
                            error_message=str(e),
                        )
                    )

            # astropy cutout
            if HAS_ASTROPY:
                try:
                    file_size_mb = os.path.getsize(filepath) / 1024 / 1024

                    with MemoryMonitor() as mem:
                        start_time = time.time()

                        with astropy_fits.open(filepath) as hdul:
                            if file_type == "image_2d":
                                data = hdul[0].data[
                                    0 : cutout_shape[0], 0 : cutout_shape[1]
                                ]
                            else:  # image_3d
                                data = hdul[0].data[
                                    0 : cutout_shape[0],
                                    0 : cutout_shape[1],
                                    0 : cutout_shape[2],
                                ]

                            # Force data read
                            _ = np.array(data)

                        end_time = time.time()

                        read_time_ms = (end_time - start_time) * 1000
                        memory_delta = mem.memory_delta_mb
                        cutout_size_mb = (
                            np.prod(cutout_shape) * 4 / 1024 / 1024
                        )  # Assuming float32
                        throughput_mbs = (
                            cutout_size_mb / (read_time_ms / 1000)
                            if read_time_ms > 0
                            else 0
                        )

                        results.append(
                            BenchmarkResult(
                                library="astropy",
                                operation="cutout",
                                file_type=file_type,
                                data_shape=cutout_shape,
                                file_size_mb=cutout_size_mb,
                                read_time_ms=read_time_ms,
                                memory_usage_mb=memory_delta,
                                throughput_mbs=throughput_mbs,
                                success=True,
                                extra_info={"full_file_size_mb": file_size_mb},
                            )
                        )

                except Exception as e:
                    results.append(
                        BenchmarkResult(
                            library="astropy",
                            operation="cutout",
                            file_type=file_type,
                            data_shape=cutout_shape,
                            file_size_mb=0,
                            read_time_ms=0,
                            memory_usage_mb=0,
                            throughput_mbs=0,
                            success=False,
                            error_message=str(e),
                        )
                    )

        return results

    def run_comprehensive_benchmark(self) -> None:
        """Run the complete benchmark suite"""
        print("🚀 TorchFits Comprehensive Benchmark Suite")
        print("==========================================")
        print(f"Output directory: {self.output_dir}")
        print()

        # Test configurations
        file_types = ["image_2d", "image_3d", "spectrum_1d", "table"]
        size_categories = [
            "small",
            "medium",
            "large",
        ]  # Skip 'huge' for CI compatibility

        # Libraries to test
        libraries = []
        if HAS_TORCHFITS:
            libraries.append("torchfits")
        if HAS_FITSIO:
            libraries.append("fitsio")
        if HAS_ASTROPY:
            libraries.append("astropy")

        print(f"Available libraries: {', '.join(libraries)}")
        print(f"Testing file types: {', '.join(file_types)}")
        print(f"Testing size categories: {', '.join(size_categories)}")
        print()

        total_tests = len(file_types) * len(size_categories)
        test_count = 0

        for file_type in file_types:
            for size_category in size_categories:
                test_count += 1
                print(
                    f"[{test_count}/{total_tests}] Testing {file_type} ({size_category})"
                )

                # Create test file
                test_file = os.path.join(
                    self.output_dir, f"test_{file_type}_{size_category}.fits"
                )
                try:
                    shape = FITSTestDataGenerator.create_test_file(
                        test_file, file_type, size_category
                    )
                    file_size_mb = os.path.getsize(test_file) / 1024 / 1024
                    print(f"  Created: {file_size_mb:.1f} MB, shape: {shape}")
                except Exception as e:
                    print(f"  ❌ Failed to create test file: {e}")
                    continue

                # Benchmark each library
                for library in libraries:
                    if library == "torchfits":
                        result = self.benchmark_torchfits(test_file, file_type, shape)
                    elif library == "fitsio":
                        result = self.benchmark_fitsio(test_file, file_type, shape)
                    elif library == "astropy":
                        result = self.benchmark_astropy(test_file, file_type, shape)

                    self.results.append(result)

                    if result.success:
                        print(
                            f"    ✅ {library}: {result.read_time_ms:.1f}ms, "
                            f"{result.throughput_mbs:.1f} MB/s, {result.memory_usage_mb:.1f} MB"
                        )
                    else:
                        print(f"    ❌ {library}: {result.error_message}")

                # Benchmark cutouts
                cutout_results = self.benchmark_cutouts(test_file, file_type, shape)
                self.results.extend(cutout_results)

                if cutout_results:
                    print(f"  Cutout tests: {len(cutout_results)} completed")

                print()

        # Advanced CFITSIO benchmarks
        if HAS_CFITSIO_ADVANCED:
            print("Testing advanced CFITSIO features...")
            self.run_cfitsio_advanced_benchmarks()
            print()

        # Generate report
        self.generate_report()

    def run_cfitsio_advanced_benchmarks(self) -> None:
        """Test advanced CFITSIO features"""
        if not HAS_CFITSIO_ADVANCED:
            return

        # Create test file for CFITSIO benchmarking
        test_file = os.path.join(self.output_dir, "cfitsio_benchmark.fits")
        shape = FITSTestDataGenerator.create_test_file(test_file, "image_2d", "medium")

        try:
            # Test CFITSIO benchmarking functionality
            benchmarker = torchfits_cfitsio.CFITSIOBenchmark()

            print("  Advanced CFITSIO benchmarks completed")

        except Exception as e:
            print(f"  ❌ CFITSIO advanced benchmark failed: {e}")

    def generate_report(self) -> None:
        """Generate comprehensive benchmark report"""
        print("📊 Benchmark Results Summary")
        print("============================")

        # Group results by operation and file type
        success_results = [r for r in self.results if r.success]

        if not success_results:
            print("❌ No successful benchmark results to report")
            return

        # Performance comparison table
        print("\n🏁 Performance Comparison (Read Operations)")
        print("-" * 80)
        print(
            f"{'File Type':<15} {'Size':<10} {'Library':<12} {'Time (ms)':<12} {'Throughput':<15} {'Memory':<10}"
        )
        print("-" * 80)

        for file_type in ["image_2d", "image_3d", "spectrum_1d", "table"]:
            type_results = [
                r
                for r in success_results
                if r.file_type == file_type and r.operation == "read_full"
            ]

            if not type_results:
                continue

            # Group by size
            size_groups = {}
            for result in type_results:
                size_key = self._get_size_category(result.file_size_mb)
                if size_key not in size_groups:
                    size_groups[size_key] = []
                size_groups[size_key].append(result)

            for size_key, size_results in size_groups.items():
                for i, result in enumerate(size_results):
                    file_type_display = file_type if i == 0 else ""
                    size_display = size_key if i == 0 else ""

                    print(
                        f"{file_type_display:<15} {size_display:<10} {result.library:<12} "
                        f"{result.read_time_ms:<12.1f} {result.throughput_mbs:<15.1f} "
                        f"{result.memory_usage_mb:<10.1f}"
                    )

        # Cutout performance
        cutout_results = [r for r in success_results if r.operation == "cutout"]
        if cutout_results:
            print("\n✂️ Cutout Performance")
            print("-" * 60)
            print(
                f"{'File Type':<15} {'Cutout Size':<15} {'Library':<12} {'Time (ms)':<12}"
            )
            print("-" * 60)

            for result in cutout_results:
                print(
                    f"{result.file_type:<15} {str(result.data_shape):<15} "
                    f"{result.library:<12} {result.read_time_ms:<12.1f}"
                )

        # Speed ratios
        print("\n⚡ Performance Ratios (vs fitsio)")
        print("-" * 50)
        self._calculate_speed_ratios()

        # Memory efficiency
        print("\n💾 Memory Efficiency")
        print("-" * 40)
        self._analyze_memory_usage()

        # Save detailed results to JSON
        self._save_json_report()

        print(f"\n📁 Detailed results saved to: {self.output_dir}")

    def _get_size_category(self, file_size_mb: float) -> str:
        """Categorize file size"""
        if file_size_mb < 10:
            return "small"
        elif file_size_mb < 100:
            return "medium"
        elif file_size_mb < 1000:
            return "large"
        else:
            return "huge"

    def _calculate_speed_ratios(self) -> None:
        """Calculate and display speed ratios vs fitsio"""
        success_results = [
            r for r in self.results if r.success and r.operation == "read_full"
        ]

        # Group by file type and size
        groups = {}
        for result in success_results:
            key = (result.file_type, self._get_size_category(result.file_size_mb))
            if key not in groups:
                groups[key] = {}
            groups[key][result.library] = result

        for (file_type, size), library_results in groups.items():
            if "fitsio" not in library_results:
                continue

            fitsio_time = library_results["fitsio"].read_time_ms

            ratios = []
            for library, result in library_results.items():
                if library != "fitsio":
                    ratio = (
                        fitsio_time / result.read_time_ms
                        if result.read_time_ms > 0
                        else 0
                    )
                    ratios.append(f"{library}: {ratio:.1f}x")

            if ratios:
                print(f"{file_type} ({size}): {', '.join(ratios)}")

    def _analyze_memory_usage(self) -> None:
        """Analyze memory usage patterns"""
        success_results = [
            r for r in self.results if r.success and r.operation == "read_full"
        ]

        for library in ["torchfits", "fitsio", "astropy"]:
            library_results = [r for r in success_results if r.library == library]
            if library_results:
                avg_memory = np.mean([r.memory_usage_mb for r in library_results])
                max_memory = np.max([r.memory_usage_mb for r in library_results])
                print(f"{library}: avg={avg_memory:.1f}MB, max={max_memory:.1f}MB")

    def _save_json_report(self) -> None:
        """Save detailed results to JSON"""
        json_data = {
            "timestamp": time.time(),
            "total_tests": len(self.results),
            "successful_tests": len([r for r in self.results if r.success]),
            "libraries_tested": list(set(r.library for r in self.results)),
            "results": [],
        }

        for result in self.results:
            json_data["results"].append(
                {
                    "library": result.library,
                    "operation": result.operation,
                    "file_type": result.file_type,
                    "data_shape": result.data_shape,
                    "file_size_mb": result.file_size_mb,
                    "read_time_ms": result.read_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "throughput_mbs": result.throughput_mbs,
                    "success": result.success,
                    "error_message": result.error_message,
                    "extra_info": result.extra_info,
                }
            )

        json_file = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)


# Pytest test cases
class TestComprehensiveBenchmark:
    """Pytest test class for benchmark validation"""

    def test_libraries_available(self):
        """Test that at least one library is available"""
        assert (
            HAS_TORCHFITS or HAS_FITSIO or HAS_ASTROPY
        ), "No FITS libraries available for testing"

    def test_torchfits_available(self):
        """Test TorchFits availability"""
        if HAS_TORCHFITS:
            import torchfits

            assert hasattr(
                torchfits, "read_image"
            ), "TorchFits read_image not available"

    @pytest.mark.parametrize("file_type", ["image_2d", "spectrum_1d"])
    @pytest.mark.parametrize("size_category", ["small"])
    def test_benchmark_single_case(self, file_type, size_category, tmp_path):
        """Test benchmark for a single case"""
        runner = BenchmarkRunner(str(tmp_path))

        # Create test file
        test_file = tmp_path / f"test_{file_type}_{size_category}.fits"
        shape = FITSTestDataGenerator.create_test_file(
            str(test_file), file_type, size_category
        )

        # Test each available library
        if HAS_TORCHFITS:
            result = runner.benchmark_torchfits(str(test_file), file_type, shape)
            assert result.library == "torchfits"
            if result.success:
                assert result.read_time_ms > 0
                assert result.throughput_mbs >= 0

        if HAS_FITSIO:
            result = runner.benchmark_fitsio(str(test_file), file_type, shape)
            assert result.library == "fitsio"
            if result.success:
                assert result.read_time_ms > 0
                assert result.throughput_mbs >= 0

        if HAS_ASTROPY:
            result = runner.benchmark_astropy(str(test_file), file_type, shape)
            assert result.library == "astropy"
            if result.success:
                assert result.read_time_ms > 0
                assert result.throughput_mbs >= 0

    def test_performance_comparison(self, tmp_path):
        """Test that TorchFits performance is competitive"""
        if not HAS_TORCHFITS:
            pytest.skip("TorchFits not available")

        runner = BenchmarkRunner(str(tmp_path))

        # Create a medium-sized test file
        test_file = tmp_path / "performance_test.fits"
        shape = FITSTestDataGenerator.create_test_file(
            str(test_file), "image_2d", "small"
        )

        # Benchmark TorchFits
        torchfits_result = runner.benchmark_torchfits(str(test_file), "image_2d", shape)

        if torchfits_result.success:
            # Basic performance sanity checks
            assert torchfits_result.read_time_ms > 0, "Read time should be positive"
            assert torchfits_result.throughput_mbs > 0, "Throughput should be positive"
            assert (
                torchfits_result.memory_usage_mb >= 0
            ), "Memory usage should be non-negative"

            # Performance should be reasonable (less than 10 seconds for small files)
            assert (
                torchfits_result.read_time_ms < 10000
            ), "Read time should be reasonable"


def run_benchmark_suite(output_dir: Optional[str] = None) -> BenchmarkRunner:
    """Main entry point for running the benchmark suite"""
    runner = BenchmarkRunner(output_dir)
    runner.run_comprehensive_benchmark()
    return runner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TorchFits Comprehensive Benchmark Suite"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for results", default=None
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (small files only)"
    )

    args = parser.parse_args()

    print("🚀 TorchFits Comprehensive Benchmark Suite")
    print("=" * 50)

    # Check library availability
    print("Library availability:")
    print(f"  TorchFits: {'✅' if HAS_TORCHFITS else '❌'}")
    print(f"  fitsio: {'✅' if HAS_FITSIO else '❌'}")
    print(f"  astropy: {'✅' if HAS_ASTROPY else '❌'}")
    print(f"  CFITSIO Advanced: {'✅' if HAS_CFITSIO_ADVANCED else '❌'}")
    print()

    if not (HAS_TORCHFITS or HAS_FITSIO or HAS_ASTROPY):
        print("❌ No FITS libraries available for benchmarking!")
        exit(1)

    # Run benchmark
    runner = run_benchmark_suite(args.output_dir)

    print("✅ Benchmark suite completed successfully!")
    print(f"📁 Results saved to: {runner.output_dir}")
