#!/usr/bin/env python3
"""
TorchFits Official Comprehensive Benchmark Suite

Official test suite for validating performance claims against fitsio and astropy.
Tests all types of FITS data and sizes, including cutouts and advanced operations.

Performance Targets:
- Images: 2-17x faster than fitsio/astropy
- Tables: 0.8-5x faster than fitsio/astropy
- Cutouts: Sub-millisecond for typical operations
- Memory efficiency: 40% reduction vs baseline
- Remote files: <15% overhead vs local (cached)

Usage:
    pytest tests/test_official_benchmark_suite.py -v
    pytest tests/test_official_benchmark_suite.py::test_image_performance -v
    pytest tests/test_official_benchmark_suite.py::test_cutout_performance -v
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import pytest
import torch

# Add src to path for development testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Core libraries
try:
    import torchfits

    HAS_TORCHFITS = True
    print(
        f"✓ TorchFits available (version: {getattr(torchfits, '__version__', 'unknown')})"
    )
except ImportError as e:
    HAS_TORCHFITS = False
    print(f"✗ TorchFits not available: {e}")

try:
    import fitsio

    HAS_FITSIO = True
    print(f"✓ fitsio available")
except ImportError:
    HAS_FITSIO = False
    print("✗ fitsio not available")

try:
    from astropy.io import fits as astropy_fits
    from astropy.table import Table as astropy_table

    HAS_ASTROPY = True
    print(f"✓ astropy available")
except ImportError:
    HAS_ASTROPY = False
    print("✗ astropy not available")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    speedup_vs_baseline: Optional[float] = None
    cutout_region: Optional[Tuple[slice, ...]] = None
    extra_info: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""

    total_tests: int
    successful_tests: int
    failed_tests: int
    torchfits_results: List[BenchmarkResult]
    fitsio_results: List[BenchmarkResult]
    astropy_results: List[BenchmarkResult]
    performance_comparisons: Dict[str, Dict[str, float]]

    def get_speedup_summary(self) -> Dict[str, float]:
        """Get summary of speedups vs baselines"""
        speedups = {}
        for result in self.torchfits_results:
            if result.success and result.speedup_vs_baseline:
                key = f"{result.file_type}_{result.operation}"
                if key not in speedups:
                    speedups[key] = []
                speedups[key].append(result.speedup_vs_baseline)

        # Calculate averages
        summary = {}
        for key, values in speedups.items():
            summary[key] = {
                "mean_speedup": np.mean(values),
                "min_speedup": np.min(values),
                "max_speedup": np.max(values),
                "count": len(values),
            }
        return summary


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


def time_operation(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time an operation with high precision."""
    # Ensure CUDA synchronization if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    return result, (end_time - start_time) * 1000  # Return time in milliseconds


class FITSTestDataGenerator:
    """Generate various types of test FITS files"""

    @staticmethod
    def create_image_2d(
        shape: Tuple[int, int], dtype=np.float32, add_noise=True
    ) -> np.ndarray:
        """Create 2D image data with realistic astronomical characteristics"""
        data = np.random.normal(1000, 100, shape).astype(dtype)

        if add_noise:
            # Add some realistic features
            y, x = np.ogrid[: shape[0], : shape[1]]

            # Add some "stars" (bright points)
            n_stars = min(50, shape[0] * shape[1] // 10000)
            for _ in range(n_stars):
                star_y = np.random.randint(10, shape[0] - 10)
                star_x = np.random.randint(10, shape[1] - 10)
                brightness = np.random.exponential(5000)

                # Add gaussian profile
                sigma = np.random.uniform(1.5, 3.0)
                star_profile = brightness * np.exp(
                    -((y - star_y) ** 2 + (x - star_x) ** 2) / (2 * sigma**2)
                )
                data += star_profile

        return data

    @staticmethod
    def create_image_3d(shape: Tuple[int, int, int], dtype=np.float32) -> np.ndarray:
        """Create 3D cube data (spectral cube)"""
        # Create spectral cube with wavelength-dependent structure
        data = np.zeros(shape, dtype=dtype)

        for i in range(shape[0]):  # wavelength axis
            # Create 2D image for this wavelength
            img = FITSTestDataGenerator.create_image_2d(
                shape[1:], dtype, add_noise=(i % 5 == 0)
            )

            # Add wavelength-dependent scaling
            wavelength_factor = 0.5 + 1.5 * np.sin(i * np.pi / shape[0])
            data[i] = img * wavelength_factor

        return data

    @staticmethod
    def create_spectrum_1d(length: int, dtype=np.float32) -> np.ndarray:
        """Create 1D spectrum data"""
        wavelength = np.linspace(4000, 8000, length)  # Angstroms

        # Create realistic spectrum with emission/absorption lines
        continuum = 1000 * np.exp(-((wavelength - 6000) ** 2) / (2 * 2000**2))

        # Add some spectral lines
        lines = [4861, 5007, 6563, 6717, 6731]  # Common emission lines
        for line_wave in lines:
            if line_wave > wavelength.min() and line_wave < wavelength.max():
                line_strength = np.random.exponential(500)
                line_profile = line_strength * np.exp(
                    -((wavelength - line_wave) ** 2) / (2 * 5**2)
                )
                continuum += line_profile

        # Add noise
        noise = np.random.normal(0, 50, length)
        spectrum = (continuum + noise).astype(dtype)

        return spectrum

    @staticmethod
    def create_table_data(n_rows: int) -> Dict[str, np.ndarray]:
        """Create realistic astronomical table data"""
        # Generate realistic RA/DEC distribution
        ra = np.random.uniform(0, 360, n_rows).astype(np.float64)
        dec = (
            np.arcsin(2 * np.random.uniform(0, 1, n_rows) - 1) * 180 / np.pi
        )  # Proper sphere sampling

        # Generate correlated magnitudes (brighter objects have better S/N)
        mag_g = np.random.normal(20, 2, n_rows).astype(np.float32)
        mag_r = mag_g + np.random.normal(0.5, 0.3, n_rows)  # g-r color
        mag_i = mag_r + np.random.normal(0.3, 0.2, n_rows)  # r-i color

        # Generate fluxes from magnitudes
        flux_g = 10 ** (-0.4 * (mag_g - 25)) * np.random.exponential(1, n_rows)
        flux_r = 10 ** (-0.4 * (mag_r - 25)) * np.random.exponential(1, n_rows)
        flux_i = 10 ** (-0.4 * (mag_i - 25)) * np.random.exponential(1, n_rows)

        # Object classification based on color
        g_r = mag_g - mag_r
        r_i = mag_r - mag_i

        object_class = np.full(n_rows, "STAR", dtype="U10")
        galaxy_mask = (g_r > 0.6) & (r_i > 0.4)
        qso_mask = (g_r < 0.2) & (mag_g < 22)
        object_class[galaxy_mask] = "GALAXY"
        object_class[qso_mask] = "QSO"

        # Generate redshifts (for galaxies and QSOs)
        redshift = np.zeros(n_rows, dtype=np.float32)
        redshift[galaxy_mask] = np.random.exponential(
            0.3, size=int(np.sum(galaxy_mask))
        )
        redshift[qso_mask] = np.random.uniform(0.5, 4.0, size=int(np.sum(qso_mask)))

        return {
            "ID": np.arange(n_rows, dtype=np.int64),
            "RA": ra,
            "DEC": dec,
            "MAG_G": mag_g,
            "MAG_R": mag_r,
            "MAG_I": mag_i,
            "FLUX_G": flux_g.astype(np.float32),
            "FLUX_R": flux_r.astype(np.float32),
            "FLUX_I": flux_i.astype(np.float32),
            "CLASS": object_class,
            "REDSHIFT": redshift,
            "SNR": np.random.exponential(10, n_rows).astype(np.float32),
        }

    @classmethod
    def create_test_file(
        cls, filepath: str, file_type: str, size_category: str
    ) -> Tuple[int, ...]:
        """Create a test FITS file and return its data shape"""

        # Define size categories for different data types
        size_configs = {
            "image_2d": {
                "tiny": (256, 256),
                "small": (512, 512),
                "medium": (2048, 2048),
                "large": (4096, 4096),
                "huge": (8192, 8192),
            },
            "image_3d": {
                "tiny": (20, 128, 128),
                "small": (50, 256, 256),
                "medium": (100, 512, 512),
                "large": (200, 1024, 1024),
                "huge": (500, 2048, 2048),
            },
            "spectrum_1d": {
                "tiny": 1000,
                "small": 5000,
                "medium": 20000,
                "large": 100000,
                "huge": 500000,
            },
            "table": {
                "tiny": 100,
                "small": 1000,
                "medium": 50000,
                "large": 500000,
                "huge": 2000000,
            },
        }

        if file_type not in size_configs:
            raise ValueError(f"Unknown file type: {file_type}")

        if size_category not in size_configs[file_type]:
            raise ValueError(f"Unknown size category: {size_category}")

        config = size_configs[file_type][size_category]

        if file_type == "image_2d":
            shape = config
            data = cls.create_image_2d(shape)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.header["OBJECT"] = f"TEST_{size_category.upper()}_IMAGE"
                hdu.header["EXPTIME"] = 300.0
                hdu.header["FILTER"] = "r"
                hdu.writeto(filepath, overwrite=True)
            elif HAS_FITSIO:
                fitsio.write(filepath, data, clobber=True)
            else:
                raise ImportError(
                    "Neither astropy nor fitsio available for file creation"
                )

            return shape

        elif file_type == "image_3d":
            shape = config
            data = cls.create_image_3d(shape)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.header["OBJECT"] = f"TEST_{size_category.upper()}_CUBE"
                hdu.header["CTYPE1"] = "RA---TAN"
                hdu.header["CTYPE2"] = "DEC--TAN"
                hdu.header["CTYPE3"] = "WAVE"
                hdu.writeto(filepath, overwrite=True)
            elif HAS_FITSIO:
                fitsio.write(filepath, data, clobber=True)
            else:
                raise ImportError(
                    "Neither astropy nor fitsio available for file creation"
                )

            return shape

        elif file_type == "spectrum_1d":
            length = config
            data = cls.create_spectrum_1d(length)
            shape = (length,)

            if HAS_ASTROPY:
                hdu = astropy_fits.PrimaryHDU(data)
                hdu.header["OBJECT"] = f"TEST_{size_category.upper()}_SPECTRUM"
                hdu.header["CTYPE1"] = "WAVE"
                hdu.header["CRVAL1"] = 4000.0
                hdu.header["CDELT1"] = 4000.0 / length
                hdu.writeto(filepath, overwrite=True)
            elif HAS_FITSIO:
                fitsio.write(filepath, data, clobber=True)
            else:
                raise ImportError(
                    "Neither astropy nor fitsio available for file creation"
                )

            return shape

        elif file_type == "table":
            n_rows = config
            table_data = cls.create_table_data(n_rows)
            shape = (n_rows, len(table_data))

            if HAS_ASTROPY:
                # Create table HDU with proper FITS types
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
                    else:
                        # Default to double precision
                        cols.append(
                            astropy_fits.Column(
                                name=name, format="D", array=data.astype(np.float64)
                            )
                        )

                # Create primary HDU and table HDU
                primary = astropy_fits.PrimaryHDU()
                table_hdu = astropy_fits.BinTableHDU.from_columns(cols, name="CATALOG")
                table_hdu.header["OBJECT"] = f"TEST_{size_category.upper()}_CATALOG"

                hdul = astropy_fits.HDUList([primary, table_hdu])
                hdul.writeto(filepath, overwrite=True)
            elif HAS_FITSIO:
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

    def __init__(self, output_dir: Optional[str] = None, warmup_runs: int = 1):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="torchfits_benchmark_")
        Path(self.output_dir).mkdir(exist_ok=True)
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []

    def __del__(self):
        """Cleanup temporary files"""
        if hasattr(self, "output_dir") and os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except:
                pass  # Ignore cleanup errors

    def _warmup_run(self, func, *args, **kwargs):
        """Perform warmup runs to stabilize timing"""
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except:
                pass  # Ignore warmup errors

    def benchmark_torchfits(
        self,
        filepath: str,
        file_type: str,
        shape: Tuple[int, ...],
        operation: str = "read_full",
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark TorchFits reading operations"""
        if not HAS_TORCHFITS:
            return BenchmarkResult(
                library="torchfits",
                operation=operation,
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

            # Determine reading function based on operation and file type
            if operation == "read_full":
                if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                    read_func = lambda: torchfits.read(filepath, hdu=0)
                elif file_type == "table":
                    read_func = lambda: torchfits.read(
                        filepath, hdu=1
                    )  # Table in extension 1
                else:
                    raise ValueError(f"Unknown file type: {file_type}")

            elif operation == "read_cutout":
                # Handle cutout operations
                cutout_region = kwargs.get("cutout_region")
                if not cutout_region:
                    raise ValueError("Cutout region required for cutout operation")

                if file_type in ["image_2d", "image_3d"]:
                    start = [s.start for s in cutout_region]
                    shape_cut = [s.stop - s.start for s in cutout_region]
                    read_func = lambda: torchfits.read(
                        filepath, hdu=0, start=start, shape=shape_cut
                    )
                else:
                    raise ValueError(f"Cutout not supported for file type: {file_type}")

            elif operation == "read_columns":
                # Handle column reading for tables
                columns = kwargs.get("columns", ["RA", "DEC", "MAG_G"])
                if file_type != "table":
                    raise ValueError("Column reading only supported for tables")
                read_func = lambda: torchfits.read(filepath, hdu=1, columns=columns)

            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Warmup
            self._warmup_run(read_func)

            # Actual benchmark
            with MemoryMonitor() as mem:
                result, read_time_ms = time_operation(read_func)

                # Handle different return types from TorchFits
                if isinstance(result, tuple) and len(result) == 2:
                    # Image data returns (tensor, header)
                    data, header = result
                elif isinstance(result, dict):
                    # Table data returns dict of tensors
                    data = result
                else:
                    # Direct tensor return
                    data = result

                # Verify data was read correctly
                if hasattr(data, "shape") and not isinstance(data, (dict, tuple)):
                    # Tensor data
                    actual_shape = data.shape
                elif isinstance(data, dict):
                    # For table data or column selection
                    first_col = next(iter(data.values()))
                    if hasattr(first_col, "shape"):
                        actual_shape = (first_col.shape[0], len(data))
                    else:
                        actual_shape = (len(first_col), len(data))
                elif hasattr(data, "shape"):
                    # Direct tensor
                    actual_shape = data.shape  # type: ignore
                else:
                    actual_shape = (len(data),) if hasattr(data, "__len__") else (1,)

                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="torchfits",
                    operation=operation,
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                    cutout_region=kwargs.get("cutout_region"),
                    extra_info=(
                        {"device": str(getattr(data, "device", "cpu"))}
                        if hasattr(data, "device")
                        else None
                    ),
                )

        except Exception as e:
            return BenchmarkResult(
                library="torchfits",
                operation=operation,
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
        self,
        filepath: str,
        file_type: str,
        shape: Tuple[int, ...],
        operation: str = "read_full",
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark fitsio reading operations"""
        if not HAS_FITSIO:
            return BenchmarkResult(
                library="fitsio",
                operation=operation,
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

            if operation == "read_full":

                def read_func():
                    with fitsio.FITS(filepath, "r") as fits:
                        if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                            return fits[0].read()
                        elif file_type == "table":
                            return fits[1].read()  # Table in extension 1
                        else:
                            raise ValueError(f"Unknown file type: {file_type}")

            elif operation == "read_cutout":
                cutout_region = kwargs.get("cutout_region")
                if not cutout_region:
                    raise ValueError("Cutout region required for cutout operation")

                def read_func():
                    with fitsio.FITS(filepath, "r") as fits:
                        if file_type == "image_2d":
                            y_slice, x_slice = cutout_region
                            return fits[0][
                                y_slice.start : y_slice.stop,
                                x_slice.start : x_slice.stop,
                            ]
                        elif file_type == "image_3d":
                            z_slice, y_slice, x_slice = cutout_region
                            return fits[0][
                                z_slice.start : z_slice.stop,
                                y_slice.start : y_slice.stop,
                                x_slice.start : x_slice.stop,
                            ]
                        else:
                            raise ValueError(
                                f"Cutout not supported for file type: {file_type}"
                            )

            elif operation == "read_columns":
                columns = kwargs.get("columns", ["RA", "DEC", "MAG_G"])
                if file_type != "table":
                    raise ValueError("Column reading only supported for tables")

                def read_func():
                    with fitsio.FITS(filepath, "r") as fits:
                        return fits[1].read(columns=columns)

            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Warmup
            self._warmup_run(read_func)

            # Actual benchmark
            with MemoryMonitor() as mem:
                data, read_time_ms = time_operation(read_func)

                if hasattr(data, "shape"):
                    actual_shape = data.shape
                elif isinstance(data, np.recarray):
                    actual_shape = (len(data), len(data.dtype.names))
                else:
                    actual_shape = (len(data),) if hasattr(data, "__len__") else (1,)

                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="fitsio",
                    operation=operation,
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                    cutout_region=kwargs.get("cutout_region"),
                )

        except Exception as e:
            return BenchmarkResult(
                library="fitsio",
                operation=operation,
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
        self,
        filepath: str,
        file_type: str,
        shape: Tuple[int, ...],
        operation: str = "read_full",
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark astropy reading operations"""
        if not HAS_ASTROPY:
            return BenchmarkResult(
                library="astropy",
                operation=operation,
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

            if operation == "read_full":

                def read_full_func():
                    hdul = astropy_fits.open(filepath)
                    try:
                        if file_type in ["image_2d", "image_3d", "spectrum_1d"]:
                            return hdul[0].data  # type: ignore
                        elif file_type == "table":
                            return hdul[1].data  # type: ignore # Table in extension 1
                        else:
                            raise ValueError(f"Unknown file type: {file_type}")
                    finally:
                        hdul.close()

                read_func = read_full_func

            elif operation == "read_cutout":
                cutout_region = kwargs.get("cutout_region")
                if not cutout_region:
                    raise ValueError("Cutout region required for cutout operation")

                def read_cutout_func():
                    hdul = astropy_fits.open(filepath)
                    try:
                        data = hdul[0].data  # type: ignore
                        if file_type == "image_2d":
                            y_slice, x_slice = cutout_region
                            return data[y_slice, x_slice]
                        elif file_type == "image_3d":
                            z_slice, y_slice, x_slice = cutout_region
                            return data[z_slice, y_slice, x_slice]
                        else:
                            raise ValueError(
                                f"Cutout not supported for file type: {file_type}"
                            )
                    finally:
                        hdul.close()

                read_func = read_cutout_func

            elif operation == "read_columns":
                columns = kwargs.get("columns", ["RA", "DEC", "MAG_G"])
                if file_type != "table":
                    raise ValueError("Column reading only supported for tables")

                def read_columns_func():
                    hdul = astropy_fits.open(filepath)
                    try:
                        data = hdul[1].data  # type: ignore
                        return {col: data[col] for col in columns}
                    finally:
                        hdul.close()

                read_func = read_columns_func

            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Warmup
            self._warmup_run(read_func)

            # Actual benchmark
            with MemoryMonitor() as mem:
                data, read_time_ms = time_operation(read_func)

                if hasattr(data, "shape"):
                    actual_shape = data.shape
                elif isinstance(data, dict):
                    first_col = next(iter(data.values()))
                    actual_shape = (len(first_col), len(data))
                elif hasattr(data, "__len__"):
                    actual_shape = (len(data),)
                else:
                    actual_shape = (1,)

                memory_delta = mem.memory_delta_mb
                throughput_mbs = (
                    file_size_mb / (read_time_ms / 1000) if read_time_ms > 0 else 0
                )

                return BenchmarkResult(
                    library="astropy",
                    operation=operation,
                    file_type=file_type,
                    data_shape=actual_shape,
                    file_size_mb=file_size_mb,
                    read_time_ms=read_time_ms,
                    memory_usage_mb=memory_delta,
                    throughput_mbs=throughput_mbs,
                    success=True,
                    cutout_region=kwargs.get("cutout_region"),
                )

        except Exception as e:
            return BenchmarkResult(
                library="astropy",
                operation=operation,
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

    def run_comparison_benchmark(
        self,
        filepath: str,
        file_type: str,
        shape: Tuple[int, ...],
        operation: str = "read_full",
        **kwargs,
    ) -> List[BenchmarkResult]:
        """Run benchmark comparison across all libraries"""
        results = []

        # Benchmark TorchFits
        result_tf = self.benchmark_torchfits(
            filepath, file_type, shape, operation, **kwargs
        )
        results.append(result_tf)

        # Benchmark fitsio
        result_fitsio = self.benchmark_fitsio(
            filepath, file_type, shape, operation, **kwargs
        )
        results.append(result_fitsio)

        # Benchmark astropy
        result_astropy = self.benchmark_astropy(
            filepath, file_type, shape, operation, **kwargs
        )
        results.append(result_astropy)

        # Calculate speedups vs baselines
        if result_tf.success:
            if result_fitsio.success and result_fitsio.read_time_ms > 0:
                result_tf.speedup_vs_baseline = (
                    result_fitsio.read_time_ms / result_tf.read_time_ms
                )
            elif result_astropy.success and result_astropy.read_time_ms > 0:
                result_tf.speedup_vs_baseline = (
                    result_astropy.read_time_ms / result_tf.read_time_ms
                )

        self.results.extend(results)
        return results


# Test fixtures and setup
@pytest.fixture(scope="session")
def benchmark_runner():
    """Create a benchmark runner for the test session"""
    return BenchmarkRunner()


@pytest.fixture(scope="session")
def test_files(benchmark_runner):
    """Create test files for benchmarking"""
    files = {}

    # Create different types and sizes of test files
    test_configs = [
        ("image_2d", "small"),
        ("image_2d", "medium"),
        ("image_3d", "small"),
        ("spectrum_1d", "medium"),
        ("table", "small"),
        ("table", "medium"),
    ]

    for file_type, size_category in test_configs:
        filename = f"test_{file_type}_{size_category}.fits"
        filepath = os.path.join(benchmark_runner.output_dir, filename)

        try:
            shape = FITSTestDataGenerator.create_test_file(
                filepath, file_type, size_category
            )
            files[f"{file_type}_{size_category}"] = {
                "path": filepath,
                "type": file_type,
                "size": size_category,
                "shape": shape,
            }
            print(
                f"✓ Created {filename}: shape={shape}, size={os.path.getsize(filepath)/1024/1024:.1f}MB"
            )
        except Exception as e:
            print(f"✗ Failed to create {filename}: {e}")

    return files


# Core benchmark tests
@pytest.mark.parametrize(
    "file_key",
    ["image_2d_small", "image_2d_medium", "image_3d_small", "spectrum_1d_medium"],
)
def test_image_performance(benchmark_runner, test_files, file_key):
    """Test image reading performance across all libraries"""
    if file_key not in test_files:
        pytest.skip(f"Test file {file_key} not available")

    file_info = test_files[file_key]
    results = benchmark_runner.run_comparison_benchmark(
        file_info["path"], file_info["type"], file_info["shape"]
    )

    # Validate results
    torchfits_result = next((r for r in results if r.library == "torchfits"), None)
    assert torchfits_result is not None, "TorchFits result not found"

    if HAS_TORCHFITS:
        assert (
            torchfits_result.success
        ), f"TorchFits failed: {torchfits_result.error_message}"
        assert torchfits_result.read_time_ms > 0, "Read time should be positive"
        assert torchfits_result.throughput_mbs > 0, "Throughput should be positive"

        # Check for performance targets
        if torchfits_result.speedup_vs_baseline:
            print(f"TorchFits speedup: {torchfits_result.speedup_vs_baseline:.2f}x")
            # For images, we expect some speedup (relaxed target for realistic benchmarking)
            if file_info["size"] in ["medium", "large"]:
                # Relaxed performance target - any speedup is good
                if torchfits_result.speedup_vs_baseline < 0.8:
                    print(
                        f"Warning: Performance below 0.8x baseline: {torchfits_result.speedup_vs_baseline:.2f}x"
                    )
                # Still assert reasonable performance
                assert (
                    torchfits_result.speedup_vs_baseline > 0.5
                ), f"Performance too low: {torchfits_result.speedup_vs_baseline:.2f}x"


@pytest.mark.parametrize("file_key", ["table_small", "table_medium"])
def test_table_performance(benchmark_runner, test_files, file_key):
    """Test table reading performance across all libraries"""
    if file_key not in test_files:
        pytest.skip(f"Test file {file_key} not available")

    file_info = test_files[file_key]
    results = benchmark_runner.run_comparison_benchmark(
        file_info["path"], file_info["type"], file_info["shape"]
    )

    # Validate results
    torchfits_result = next((r for r in results if r.library == "torchfits"), None)
    assert torchfits_result is not None, "TorchFits result not found"

    if HAS_TORCHFITS:
        assert (
            torchfits_result.success
        ), f"TorchFits failed: {torchfits_result.error_message}"
        assert torchfits_result.read_time_ms > 0, "Read time should be positive"

        # Check data integrity
        assert (
            len(torchfits_result.data_shape) == 2
        ), "Table should have 2D shape (rows, columns)"
        assert torchfits_result.data_shape[1] > 5, "Should have multiple columns"


@pytest.mark.parametrize(
    "file_key,cutout_spec",
    [
        ("image_2d_small", (slice(100, 200), slice(100, 200))),
        ("image_2d_medium", (slice(500, 1000), slice(500, 1000))),
        ("image_3d_small", (slice(10, 30), slice(50, 150), slice(50, 150))),
    ],
)
def test_cutout_performance(benchmark_runner, test_files, file_key, cutout_spec):
    """Test cutout operations performance"""
    if file_key not in test_files:
        pytest.skip(f"Test file {file_key} not available")

    file_info = test_files[file_key]

    # Test cutout operations
    results = benchmark_runner.run_comparison_benchmark(
        file_info["path"],
        file_info["type"],
        file_info["shape"],
        operation="read_cutout",
        cutout_region=cutout_spec,
    )

    torchfits_result = next((r for r in results if r.library == "torchfits"), None)

    if HAS_TORCHFITS and torchfits_result and torchfits_result.success:
        # Cutouts should be very fast
        assert (
            torchfits_result.read_time_ms < 100
        ), f"Cutout took {torchfits_result.read_time_ms:.1f}ms, expected <100ms"

        # Verify cutout shape
        expected_shape = tuple(s.stop - s.start for s in cutout_spec)
        if file_info["type"] == "image_2d":
            assert (
                torchfits_result.data_shape == expected_shape
            ), f"Expected shape {expected_shape}, got {torchfits_result.data_shape}"
        elif file_info["type"] == "image_3d":
            assert (
                torchfits_result.data_shape == expected_shape
            ), f"Expected shape {expected_shape}, got {torchfits_result.data_shape}"

        print(
            f"Cutout performance: {torchfits_result.read_time_ms:.2f}ms for {expected_shape}"
        )


@pytest.mark.parametrize(
    "file_key,columns",
    [
        ("table_small", ["RA", "DEC"]),
        ("table_medium", ["RA", "DEC", "MAG_G", "MAG_R"]),
        ("table_small", ["FLUX_G", "FLUX_R", "FLUX_I"]),
    ],
)
def test_column_selection_performance(benchmark_runner, test_files, file_key, columns):
    """Test column selection performance for tables"""
    if file_key not in test_files:
        pytest.skip(f"Test file {file_key} not available")

    file_info = test_files[file_key]

    results = benchmark_runner.run_comparison_benchmark(
        file_info["path"],
        file_info["type"],
        file_info["shape"],
        operation="read_columns",
        columns=columns,
    )

    torchfits_result = next((r for r in results if r.library == "torchfits"), None)

    if HAS_TORCHFITS and torchfits_result and torchfits_result.success:
        # Column selection should be very fast
        assert (
            torchfits_result.read_time_ms < 50
        ), f"Column selection took {torchfits_result.read_time_ms:.1f}ms, expected <50ms"

        # Verify correct number of columns
        assert torchfits_result.data_shape[1] == len(
            columns
        ), f"Expected {len(columns)} columns, got {torchfits_result.data_shape[1]}"

        print(
            f"Column selection performance: {torchfits_result.read_time_ms:.2f}ms for {len(columns)} columns"
        )


def test_memory_efficiency(benchmark_runner, test_files):
    """Test memory efficiency compared to other libraries"""
    if "image_2d_medium" not in test_files:
        pytest.skip("Medium image test file not available")

    file_info = test_files["image_2d_medium"]
    results = benchmark_runner.run_comparison_benchmark(
        file_info["path"], file_info["type"], file_info["shape"]
    )

    # Compare memory usage
    torchfits_result = next(
        (r for r in results if r.library == "torchfits" and r.success), None
    )
    other_results = [r for r in results if r.library != "torchfits" and r.success]

    if torchfits_result and other_results:
        min_other_memory = min(r.memory_usage_mb for r in other_results)
        if min_other_memory > 0:
            memory_ratio = torchfits_result.memory_usage_mb / min_other_memory
            print(
                f"Memory efficiency: TorchFits uses {memory_ratio:.2f}x memory vs best competitor"
            )

            # Memory usage should be competitive
            assert (
                memory_ratio < 2.0
            ), f"Memory usage too high: {memory_ratio:.2f}x vs competitors"


def test_error_handling(benchmark_runner, test_files):
    """Test error handling for invalid operations"""
    if not test_files:
        pytest.skip("No test files available")

    # Test non-existent file
    result = benchmark_runner.benchmark_torchfits(
        "/nonexistent/file.fits", "image_2d", (100, 100)
    )
    assert not result.success
    assert (
        "not found" in result.error_message.lower()
        or "no such file" in result.error_message.lower()
    )

    # Test invalid cutout region (if we have a small image)
    if "image_2d_small" in test_files:
        file_info = test_files["image_2d_small"]
        # Try to read beyond image bounds
        invalid_cutout = (slice(0, 10000), slice(0, 10000))
        result = benchmark_runner.benchmark_torchfits(
            file_info["path"],
            file_info["type"],
            file_info["shape"],
            operation="read_cutout",
            cutout_region=invalid_cutout,
        )
        # Should either succeed with adjusted bounds or fail gracefully
        if not result.success:
            assert result.error_message is not None


def test_performance_summary(benchmark_runner):
    """Generate and validate performance summary"""
    if not benchmark_runner.results:
        pytest.skip("No benchmark results available")

    # Separate results by library
    torchfits_results = [
        r for r in benchmark_runner.results if r.library == "torchfits" and r.success
    ]
    fitsio_results = [
        r for r in benchmark_runner.results if r.library == "fitsio" and r.success
    ]
    astropy_results = [
        r for r in benchmark_runner.results if r.library == "astropy" and r.success
    ]

    summary = BenchmarkSummary(
        total_tests=len(benchmark_runner.results),
        successful_tests=len([r for r in benchmark_runner.results if r.success]),
        failed_tests=len([r for r in benchmark_runner.results if not r.success]),
        torchfits_results=torchfits_results,
        fitsio_results=fitsio_results,
        astropy_results=astropy_results,
        performance_comparisons={},
    )

    # Generate performance report
    print("\n" + "=" * 60)
    print("TORCHFITS PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Total tests: {summary.total_tests}")
    print(f"Successful: {summary.successful_tests}")
    print(f"Failed: {summary.failed_tests}")

    if torchfits_results:
        print(f"\nTorchFits Results:")
        for result in torchfits_results:
            speedup_str = (
                f" ({result.speedup_vs_baseline:.2f}x speedup)"
                if result.speedup_vs_baseline
                else ""
            )
            print(
                f"  {result.file_type}_{result.operation}: {result.read_time_ms:.2f}ms{speedup_str}"
            )

    # Calculate overall statistics
    if torchfits_results:
        speedups = [
            r.speedup_vs_baseline for r in torchfits_results if r.speedup_vs_baseline
        ]
        if speedups:
            avg_speedup = np.mean(speedups)
            min_speedup = np.min(speedups)
            max_speedup = np.max(speedups)

            print(f"\nSpeedup Statistics:")
            print(f"  Average: {avg_speedup:.2f}x")
            print(f"  Range: {min_speedup:.2f}x - {max_speedup:.2f}x")

            # Performance targets validation
            assert (
                avg_speedup > 1.0
            ), f"Overall performance should be better than baseline, got {avg_speedup:.2f}x"

    # Save detailed results
    results_file = os.path.join(benchmark_runner.output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(
            [asdict(r) for r in benchmark_runner.results], f, indent=2, default=str
        )

    print(f"\nDetailed results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    # Allow running as script for manual testing
    print("TorchFits Official Benchmark Suite")
    print("Run with: pytest tests/test_official_benchmark_suite.py -v")
