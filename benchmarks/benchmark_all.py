#!/usr/bin/env python3
"""
Exhaustive torchfits benchmark suite.

Tests all data types, formats, and operations with detailed reporting.
Covers: spectra, images, cubes, tables, MEFs, multi-MEFs, cutouts,
multi-cutouts, multi-files, compression, WCS, scaling, all sizes.

Produces comprehensive tables, plots, and summaries.
"""

import csv
import gc
import os
import random
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev, median
from typing import Any, Dict, List, Optional

# Add benchmarks and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mpl_config import configure

import fitsio
import numpy as np
import pandas as pd
import psutil
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits as astropy_fits
from astropy.io.fits import CompImageHDU

import torchfits


class ExhaustiveBenchmarkSuite:
    """
    Exhaustive benchmark suite for torchfits covering all use cases.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        use_mmap: bool = True,
        include_tables: bool = False,
        cache_capacity: int = 10,
        hot_cache_capacity: int = 10,
        handle_cache_capacity: int = 16,
        profile: str = "user",
        payload_min_ratio: float = 0.60,
    ):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="torchfits_exhaustive_"))
        self.output_dir = output_dir or Path("benchmark_results")
        # Allow nested output dirs like benchmark_results/exhaustive_YYYYmmdd_HHMMSS
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.csv_file = self.output_dir / "exhaustive_results.csv"
        self.summary_file = self.output_dir / "exhaustive_summary.md"
        self.focused_csv_file = self.output_dir / "focused_results.csv"
        self.focused_summary_file = self.output_dir / "focused_summary.md"
        self.use_mmap = use_mmap
        self.include_tables = include_tables
        self.cache_capacity = cache_capacity
        self.hot_cache_capacity = hot_cache_capacity
        self.handle_cache_capacity = handle_cache_capacity
        self.profile = str(profile).strip().lower() or "user"
        if self.profile not in {"user", "lab"}:
            raise ValueError("profile must be 'user' or 'lab'")
        self.payload_min_ratio = max(0.0, min(1.0, float(payload_min_ratio)))

        # Test configurations
        self.data_types = {
            "int8": (np.int8, "BYTE_IMG"),
            "int16": (np.int16, "SHORT_IMG"),
            "int32": (np.int32, "LONG_IMG"),
            # Common in catalogs / integer timestamp images; BITPIX=64
            "int64": (np.int64, "LONGLONG_IMG"),
            "float32": (np.float32, "FLOAT_IMG"),
            "float64": (np.float64, "DOUBLE_IMG"),
        }

        self.size_categories = {
            "tiny": {"1d": 1000, "2d": (64, 64), "3d": (5, 32, 32)},
            "small": {"1d": 10000, "2d": (256, 256), "3d": (10, 128, 128)},
            "medium": {"1d": 100000, "2d": (1024, 1024), "3d": (25, 256, 256)},
            "large": {"1d": 1000000, "2d": (2048, 2048), "3d": (50, 512, 512)},
        }

        self.compression_types = ["RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"]

    def _torchfits_mmap_mode(self):
        """TorchFits mmap mode used by benchmark methods."""
        if not self.use_mmap:
            return False
        # User profile should benchmark library-default behavior.
        if self.profile == "user":
            return "auto"
        # Lab profile can force explicit mmap to study low-level effects.
        return True

    def _torchfits_effective_mmap_bool(self, path: Path, hdu_num: int) -> bool:
        """Resolve to a bool mmap flag for C++ methods that require bool."""
        mode = self._torchfits_mmap_mode()
        if isinstance(mode, bool):
            return mode
        try:
            return bool(
                torchfits._resolve_image_mmap(  # benchmark-only internal use
                    str(path), hdu_num, mode, self.cache_capacity
                )
            )
        except Exception:
            return True

    def create_test_files(self) -> Dict[str, Path]:
        """Create comprehensive test dataset covering all scenarios."""
        print("Creating exhaustive test dataset...")
        files = {}

        # 1. Single HDU files (all combinations)
        files.update(self._create_single_hdu_files())

        # 2. MEF files (Multiple Extension FITS)
        files.update(self._create_mef_files())

        # 3. Multi-MEF files (MEFs with many extensions)
        files.update(self._create_multi_mef_files())

        # 4. Table files
        if self.include_tables:
            files.update(self._create_table_files())

        # 5. Scaled data files (BSCALE/BZERO)
        files.update(self._create_scaled_files())

        # 6. WCS-enabled files
        files.update(self._create_wcs_files())

        # 7. Compressed files
        files.update(self._create_compressed_files())

        # 8. Multi-file collections
        files.update(self._create_multi_file_collections())

        print(f"✓ Created {len(files)} test files in {self.temp_dir}")
        return files

    def _create_single_hdu_files(self) -> Dict[str, Path]:
        """Create single HDU files for all data types and sizes."""
        files = {}

        for size_name, size_specs in self.size_categories.items():
            for dtype_name, (np_dtype, fits_bitpix) in self.data_types.items():
                for dim_name, shape in size_specs.items():
                    # Skip huge 3D arrays to avoid memory issues
                    if size_name == "large" and dim_name == "3d":
                        continue

                    data = self._generate_data(shape, np_dtype)
                    filename = (
                        self.temp_dir / f"{size_name}_{dtype_name}_{dim_name}.fits"
                    )

                    astropy_fits.PrimaryHDU(data).writeto(filename, overwrite=True)
                    files[f"{size_name}_{dtype_name}_{dim_name}"] = filename

        return files

    def _create_mef_files(self) -> Dict[str, Path]:
        """Create Multi-Extension FITS files."""
        files = {}

        for size_name in ["small", "medium"]:
            # Standard MEF with mixed data types
            hdu_list = [astropy_fits.PrimaryHDU()]

            shape = self.size_categories[size_name]["2d"]
            for i, (dtype_name, (np_dtype, _)) in enumerate(self.data_types.items()):
                if i >= 3:  # Limit to 3 extensions
                    break
                data = self._generate_data(shape, np_dtype)
                hdu = astropy_fits.ImageHDU(data, name=f"EXT_{dtype_name.upper()}")
                hdu_list.append(hdu)

            mef_filename = self.temp_dir / f"mef_{size_name}.fits"
            astropy_fits.HDUList(hdu_list).writeto(mef_filename, overwrite=True)
            files[f"mef_{size_name}"] = mef_filename

        return files

    def _create_multi_mef_files(self) -> Dict[str, Path]:
        """Create MEF files with many extensions."""
        files = {}

        # Create MEF with 10 extensions
        hdu_list = [astropy_fits.PrimaryHDU()]
        shape = (256, 256)

        for i in range(10):
            dtype_name, (np_dtype, _) = list(self.data_types.items())[
                i % len(self.data_types)
            ]
            data = self._generate_data(shape, np_dtype)
            hdu = astropy_fits.ImageHDU(data, name=f"EXT_{i:02d}_{dtype_name.upper()}")
            hdu_list.append(hdu)

        multi_mef_filename = self.temp_dir / "multi_mef_10ext.fits"
        astropy_fits.HDUList(hdu_list).writeto(multi_mef_filename, overwrite=True)
        files["multi_mef_10ext"] = multi_mef_filename

        return files

    def _create_table_files(self) -> Dict[str, Path]:
        """Create comprehensive table FITS files showing torchfits superiority."""
        files = {}

        # Comprehensive table configurations matching benchmark_table_comprehensive.py
        table_configs = [
            # Basic configurations
            {
                "nrows": 1000,
                "ncols": 7,
                "types": ["f4"] * 5 + ["S10", "bool"],
                "name": "basic_mixed",
            },
            {
                "nrows": 10000,
                "ncols": 12,
                "types": ["f4"] * 5 + ["i4"] * 5 + ["S8", "bool"],
                "name": "mixed_basic",
            },
            {
                "nrows": 100000,
                "ncols": 22,
                "types": ["f4"] * 10 + ["i4"] * 5 + ["f8"] * 5 + ["S16", "bool"],
                "name": "mixed_medium",
            },
            # Astronomical catalog configurations
            {
                "nrows": 10000,
                "ncols": 17,
                "types": ["f8"] * 6 + ["f4"] * 6 + ["i4"] * 3 + ["S12", "bool"],
                "name": "astrometry_catalog",
            },
            {
                "nrows": 50000,
                "ncols": 27,
                "types": ["f8"] * 10 + ["f4"] * 10 + ["i4"] * 5 + ["S20", "bool"],
                "name": "photometry_catalog",
            },
            {
                "nrows": 100000,
                "ncols": 52,
                "types": ["f4"] * 30 + ["f8"] * 10 + ["i4"] * 10 + ["S32", "bool"],
                "name": "survey_catalog",
            },
        ]

        for config in table_configs:
            nrows = config["nrows"]
            ncols = config["ncols"]
            types = config["types"]
            name = config["name"]

            # Create table data with proper column names
            data = {}
            for i, dtype in enumerate(types[:ncols]):
                col_name = f"col_{i:02d}"

                if dtype.startswith("f"):  # float types
                    if dtype == "f4":
                        data[col_name] = np.random.randn(nrows).astype(np.float32)
                    else:  # f8
                        data[col_name] = np.random.randn(nrows).astype(np.float64)
                elif dtype.startswith("i"):  # integer types
                    if dtype == "i4":
                        data[col_name] = np.random.randint(
                            -1000000, 1000000, nrows, dtype=np.int32
                        )
                    else:  # i8
                        data[col_name] = np.random.randint(
                            -1000000, 1000000, nrows, dtype=np.int64
                        )
                elif dtype == "bool":
                    data[col_name] = np.random.choice([True, False], nrows)
                elif dtype.startswith("U") or dtype.startswith("S"):  # string types
                    str_len = int(dtype[1:]) if len(dtype) > 1 else 10
                    # Create fixed length strings
                    data[col_name] = np.array(
                        [
                            f"s{j:0{min(str_len - 1, 6)}d}"[:str_len]
                            for j in range(nrows)
                        ],
                        dtype=f"S{str_len}",
                    )

            # Create FITS table using astropy
            from astropy.table import Table

            table = Table(data)
            table_hdu = astropy_fits.BinTableHDU(table, name=f"TABLE_{name.upper()}")
            hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), table_hdu])

            table_filename = self.temp_dir / f"table_{name}.fits"
            hdul.writeto(table_filename, overwrite=True)
            files[f"table_{name}"] = table_filename

        return files

    def _create_scaled_files(self) -> Dict[str, Path]:
        """Create files with BSCALE/BZERO scaling."""
        files = {}

        for size_name in ["small", "medium", "large"]:
            shape = self.size_categories[size_name]["2d"]

            # Create float data that will be scaled to int16
            float_data = np.random.randn(*shape).astype(np.float32) * 1000 + 32768

            hdu = astropy_fits.PrimaryHDU()
            hdu.data = float_data.astype(np.int16)
            hdu.header["BSCALE"] = 0.1
            hdu.header["BZERO"] = 32768
            hdu.header["COMMENT"] = "Scaled data test"

            scaled_filename = self.temp_dir / f"scaled_{size_name}.fits"
            hdu.writeto(scaled_filename, overwrite=True)
            files[f"scaled_{size_name}"] = scaled_filename

        return files

    def _create_wcs_files(self) -> Dict[str, Path]:
        """Create files with WCS information."""
        files = {}

        shape = (512, 512)
        data = self._generate_data(shape, np.float32)

        # Create WCS header
        hdu = astropy_fits.PrimaryHDU(data)
        hdu.header["CRPIX1"] = shape[1] / 2
        hdu.header["CRPIX2"] = shape[0] / 2
        hdu.header["CRVAL1"] = 180.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CDELT1"] = -0.0001
        hdu.header["CDELT2"] = 0.0001
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CUNIT2"] = "deg"

        wcs_filename = self.temp_dir / "wcs_image.fits"
        hdu.writeto(wcs_filename, overwrite=True)
        files["wcs_image"] = wcs_filename

        return files

    def _create_compressed_files(self) -> Dict[str, Path]:
        """Create compressed FITS files."""
        files = {}

        shape = (1024, 1024)
        data = self._generate_data(shape, np.float32)

        for comp_type in self.compression_types:
            try:
                comp_hdu = CompImageHDU(data, compression_type=comp_type)
                hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), comp_hdu])

                comp_filename = self.temp_dir / f"compressed_{comp_type.lower()}.fits"
                hdul.writeto(comp_filename, overwrite=True)
                files[f"compressed_{comp_type.lower()}"] = comp_filename
            except Exception as e:
                print(f"Warning: Could not create {comp_type} compressed file: {e}")

        return files

    def _create_multi_file_collections(self) -> Dict[str, Path]:
        """Create collections for multi-file operations."""
        files = {}

        # Create a set of related files (simulating a time series)
        collection_dir = self.temp_dir / "timeseries"
        collection_dir.mkdir(exist_ok=True)

        shape = (256, 256)
        for i in range(5):
            data = (
                self._generate_data(shape, np.float32) + i * 100
            )  # Add offset per file
            filename = collection_dir / f"frame_{i:03d}.fits"
            astropy_fits.PrimaryHDU(data).writeto(filename, overwrite=True)
            files[f"timeseries_frame_{i:03d}"] = filename

        return files

    def _generate_data(self, shape, dtype):
        """Generate test data with appropriate values for data type."""
        if dtype == np.int8:
            return np.random.randint(-100, 100, shape, dtype=dtype)
        elif dtype == np.int16:
            return np.random.randint(-1000, 1000, shape, dtype=dtype)
        elif dtype == np.int32:
            return np.random.randint(-10000, 10000, shape, dtype=dtype)
        elif dtype == np.int64:
            return np.random.randint(-100000, 100000, shape, dtype=dtype)
        else:
            if isinstance(shape, tuple):
                return np.random.randn(*shape).astype(dtype)
            else:
                return np.random.randn(shape).astype(dtype)

    def run_exhaustive_benchmarks(self, files: Dict[str, Path]) -> List[Dict]:
        """Run benchmarks on all test files."""
        print("\n" + "=" * 100)
        print("EXHAUSTIVE BENCHMARK SUITE")
        print("=" * 100)
        print(
            "Methods:\n"
            "- torchfits: torchfits.read -> torch.Tensor\n"
            "- torchfits_numpy: torchfits.read -> numpy (fair compare vs numpy libs)\n"
            "- fitsio: fitsio.read -> numpy\n"
            "- fitsio_torch: fitsio.read -> numpy -> torch.Tensor\n"
            "- astropy: astropy fits -> numpy\n"
            "- astropy_torch: astropy fits -> numpy -> torch.Tensor\n"
            "- torchfits_hot [diag]: torchfits.read with hot torchfits cache\n"
            "- torchfits_handle_cache [diag]: torchfits.read with data cache OFF but handle cache ON\n"
            "- torchfits_cpp_open_once [diag]: reuse open C++ FITS handle"
        )
        print(
            "Legend (ranked table columns):\n"
            "- spread_s: p90(time) - p10(time) across runs (robust jitter / outliers indicator)\n"
            "- rssΔ_MB: (final RSS - initial RSS) while the method runs (best-effort; allocator reuse can hide deltas)\n"
            "- peakΔ_MB: (max sampled RSS - initial RSS) while the method runs (sampling can miss brief peaks)\n"
            "- payload_MB: deterministic size of the returned data (tensor/ndarray nbytes; dict/list summed)\n"
        )

        csv_headers = [
            "filename",
            "file_type",
            "operation",
            "size_mb",
            "data_type",
            "dimensions",
            "compression",
            "torchfits_mean",
            "torchfits_std",
            "torchfits_median",
            "torchfits_mb_s",
            "torchfits_memory",
            "torchfits_peak_memory",
            "torchfits_payload_mb",
            "torchfits_numpy_mean",
            "torchfits_numpy_std",
            "torchfits_numpy_median",
            "torchfits_numpy_mb_s",
            "torchfits_numpy_memory",
            "torchfits_numpy_peak_memory",
            "torchfits_numpy_payload_mb",
            "torchfits_hot_mean",
            "torchfits_hot_std",
            "torchfits_hot_median",
            "torchfits_hot_mb_s",
            "torchfits_hot_memory",
            "torchfits_hot_peak_memory",
            "torchfits_hot_payload_mb",
            "torchfits_handle_cache_mean",
            "torchfits_handle_cache_std",
            "torchfits_handle_cache_median",
            "torchfits_handle_cache_mb_s",
            "torchfits_handle_cache_memory",
            "torchfits_handle_cache_peak_memory",
            "torchfits_handle_cache_payload_mb",
            "torchfits_mmap_mean",
            "torchfits_mmap_std",
            "torchfits_mmap_median",
            "torchfits_mmap_mb_s",
            "torchfits_mmap_memory",
            "torchfits_mmap_peak_memory",
            "torchfits_mmap_payload_mb",
            "astropy_mean",
            "astropy_std",
            "astropy_median",
            "astropy_mb_s",
            "astropy_memory",
            "astropy_peak_memory",
            "astropy_payload_mb",
            "fitsio_mean",
            "fitsio_std",
            "fitsio_median",
            "fitsio_mb_s",
            "fitsio_memory",
            "fitsio_peak_memory",
            "fitsio_payload_mb",
            "astropy_torch_mean",
            "astropy_torch_std",
            "astropy_torch_median",
            "astropy_torch_mb_s",
            "astropy_torch_memory",
            "astropy_torch_peak_memory",
            "astropy_torch_payload_mb",
            "fitsio_torch_mean",
            "fitsio_torch_std",
            "fitsio_torch_median",
            "fitsio_torch_mb_s",
            "fitsio_torch_memory",
            "fitsio_torch_peak_memory",
            "fitsio_torch_payload_mb",
            "torchfits_cpp_open_once_mean",
            "torchfits_cpp_open_once_std",
            "torchfits_cpp_open_once_median",
            "torchfits_cpp_open_once_mb_s",
            "torchfits_cpp_open_once_memory",
            "torchfits_cpp_open_once_peak_memory",
            "torchfits_cpp_open_once_payload_mb",
            "best_method",
            "torchfits_rank",
            "speedup_vs_best",
            "best_method_torch",
            "torchfits_rank_torch",
            "speedup_vs_best_torch",
            "best_method_numpy",
            "torchfits_numpy_rank",
            "speedup_vs_best_numpy",
        ]

        detailed_results = []

        for name, filepath in files.items():
            try:
                result = self._benchmark_single_file(name, filepath)
                if result:
                    detailed_results.append(result)
            except Exception as e:
                print(f"❌ Benchmark failed for {name}: {e}")
                import traceback

                traceback.print_exc()

        # Run cutout benchmark
        cutout_results = self._benchmark_cutouts(files)
        detailed_results.extend(cutout_results)

        # Random extension access benchmark (MEF user workflow)
        ext_results = self._benchmark_random_extension_reads(files)
        detailed_results.extend(ext_results)

        # Table out-of-core scan benchmark (Arrow streaming)
        scan_results = self._benchmark_table_scan_arrow(files)
        detailed_results.extend(scan_results)

        # Write CSV results
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(detailed_results)

        print(f"\n✓ Detailed results saved to: {self.csv_file}")
        return detailed_results

    def run_focused_benchmarks(self, files: Dict[str, Path]) -> List[Dict]:
        """Run targeted benchmarks to isolate overheads and scaling costs."""
        print("\n" + "=" * 100)
        print("FOCUSED BENCHMARKS")
        print("=" * 100)

        targets = [
            "tiny_int16_1d",
            "tiny_float32_1d",
            "scaled_small",
            "scaled_medium",
            "scaled_large",
            "compressed_rice_1",
            "large_float32_2d",
            "large_float64_2d",
            "medium_float32_3d",
        ]

        results = []
        for name in targets:
            path = files.get(name)
            if not path:
                continue

            size_mb = path.stat().st_size / 1024 / 1024
            use_median = size_mb < 0.1
            runs = 30 if size_mb < 0.1 else 10 if size_mb < 1.0 else 3

            file_type = self._get_file_type(name)
            hdu_num = (
                1 if file_type in {"compressed", "table", "mef", "multi_mef"} else 0
            )
            tf_mmap_mode = self._torchfits_mmap_mode()
            tf_mmap_cpp = self._torchfits_effective_mmap_bool(path, hdu_num)

            print(f"\n{name} ({size_mb:.2f} MB) - focused")
            print("-" * 80)

            def tf_open_close():
                fh = torchfits.cpp.open_fits_file(str(path), "r")
                fh.close()

            def tf_header_fast():
                fh = torchfits.cpp.open_fits_file(str(path), "r")
                _ = torchfits.cpp.read_header_string(fh, hdu_num)
                fh.close()

            def tf_header_full():
                fh = torchfits.cpp.open_fits_file(str(path), "r")
                _ = torchfits.cpp.read_header(fh, hdu_num)
                fh.close()

            def tf_scaled():
                return torchfits.read(
                    str(path),
                    hdu=hdu_num,
                    mmap=tf_mmap_mode,
                    scale_on_device=True,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                )

            def tf_scaled_cpu():
                return torchfits.cpp.read_full_scaled_cpu(
                    str(path), hdu_num, tf_mmap_cpp
                )

            def tf_raw_with_scale_direct():
                data, scaled, bscale, bzero = torchfits.cpp.read_full_raw_with_scale(
                    str(path), hdu_num, tf_mmap_cpp
                )
                if scaled:
                    data = data.to(dtype=torch.float32)
                    if bscale != 1.0:
                        data.mul_(bscale)
                    if bzero != 0.0:
                        data.add_(bzero)
                return data

            def tf_raw():
                return torchfits.read(
                    str(path),
                    hdu=hdu_num,
                    mmap=tf_mmap_mode,
                    raw_scale=True,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                )

            def tf_raw_scale_cpu():
                data = torchfits.read(
                    str(path),
                    hdu=hdu_num,
                    mmap=tf_mmap_mode,
                    raw_scale=True,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                )
                try:
                    header = torchfits.read_header(str(path), hdu=hdu_num)
                    bscale = float(header.get("BSCALE", 1.0))
                    bzero = float(header.get("BZERO", 0.0))
                    data = data.to(torch.float32)
                    if bscale != 1.0:
                        data.mul_(bscale)
                    if bzero != 0.0:
                        data.add_(bzero)
                except Exception:
                    pass
                return data

            def tf_read_mmap_off():
                return torchfits.read(
                    str(path),
                    hdu=hdu_num,
                    mmap=False,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                )

            def fitsio_read():
                return fitsio.read(str(path), ext=hdu_num)

            methods = {
                "torchfits_open_close": tf_open_close,
                "torchfits_header_fast": tf_header_fast,
                "torchfits_header_full": tf_header_full,
                "torchfits_scaled": tf_scaled,
                "torchfits_scaled_cpu": tf_scaled_cpu,
                "torchfits_raw_with_scale_direct": tf_raw_with_scale_direct,
                "torchfits_raw": tf_raw,
                "torchfits_raw_scale_cpu": tf_raw_scale_cpu,
                "torchfits_mmap_off": tf_read_mmap_off,
                "fitsio_read": fitsio_read,
            }

            per_file = []
            for method_name, method_func in methods.items():
                method_result = self._time_method(
                    method_func, method_name, runs=runs, use_median=use_median
                )
                if not method_result:
                    per_file.append((method_name, None, None))
                    results.append(
                        {
                            "filename": name,
                            "size_mb": size_mb,
                            "file_type": file_type,
                            "method": method_name,
                            "time_s": None,
                            "std_s": None,
                            "median_s": None,
                        }
                    )
                    continue

                time_value = (
                    method_result["median"] if use_median else method_result["mean"]
                )
                per_file.append((method_name, time_value, method_result["std"]))
                results.append(
                    {
                        "filename": name,
                        "size_mb": size_mb,
                        "file_type": file_type,
                        "method": method_name,
                        "time_s": time_value,
                        "std_s": method_result["std"],
                        "median_s": method_result["median"],
                    }
                )

            # Display ranked times (fastest first). Execution order is left intact.
            ranked = [(m, t, s) for (m, t, s) in per_file if t is not None]
            ranked.sort(key=lambda x: x[1])
            if ranked:
                best_m, best_t, _ = ranked[0]
                stat_label = "median" if use_median else "mean"
                print(f"\nRanked (focused, {stat_label}, fastest first):")
                for i, (m, t, s) in enumerate(ranked, start=1):
                    ratio = t / best_t if best_t else float("inf")
                    print(f"  {i:2d}. {m:22s} {t:.6f}s ± {s:.6f}s  ({ratio:.2f}x best)")
            failed = [m for (m, t, _s) in per_file if t is None]
            if failed:
                print("\nFailed:")
                for m in failed:
                    print(f"  - {m}")

        if results:
            with open(self.focused_csv_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "filename",
                        "size_mb",
                        "file_type",
                        "method",
                        "time_s",
                        "std_s",
                        "median_s",
                    ],
                )
                writer.writeheader()
                writer.writerows(results)

            self._generate_focused_summary(results)

        return results

    def _generate_focused_summary(self, results: List[Dict]) -> None:
        df = pd.DataFrame(results)
        if df.empty:
            return

        with open(self.focused_summary_file, "w") as f:
            f.write("# torchfits Focused Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overhead Breakdown (torchfits)\n\n")
            for name, g in df.groupby("filename"):
                f.write(f"### {name}\n\n")
                pivot = g.pivot_table(
                    index="method", values="time_s", aggfunc="min"
                ).sort_values("time_s")
                f.write(pivot.to_string())
                f.write("\n\n")

            f.write("## Scaling Cost (torchfits)\n\n")
            for name, g in df.groupby("filename"):
                tf_scaled = g[g["method"] == "torchfits_scaled"]["time_s"]
                tf_raw = g[g["method"] == "torchfits_raw"]["time_s"]
                tf_raw_scale = g[g["method"] == "torchfits_raw_scale_cpu"]["time_s"]
                if not tf_scaled.empty and not tf_raw.empty:
                    f.write(
                        f"- {name}: scaled={tf_scaled.iloc[0]:.6f}s raw={tf_raw.iloc[0]:.6f}s"
                    )
                    if not tf_raw_scale.empty:
                        f.write(f" raw+scale_cpu={tf_raw_scale.iloc[0]:.6f}s")
                    f.write("\n")
            f.write("\n")

            f.write("## Header Overhead (torchfits)\n\n")
            for name, g in df.groupby("filename"):
                open_close = g[g["method"] == "torchfits_open_close"]["time_s"]
                header_fast = g[g["method"] == "torchfits_header_fast"]["time_s"]
                header_full = g[g["method"] == "torchfits_header_full"]["time_s"]
                if not open_close.empty:
                    line = f"- {name}: open_close={open_close.iloc[0]:.6f}s"
                    if not header_fast.empty:
                        line += f" header_fast={header_fast.iloc[0]:.6f}s"
                    if not header_full.empty:
                        line += f" header_full={header_full.iloc[0]:.6f}s"
                    f.write(line + "\n")
            f.write("\n")

    def _benchmark_single_file(self, name: str, filepath: Path) -> Optional[Dict]:
        """Benchmark a single file with all methods."""
        size_mb = filepath.stat().st_size / 1024 / 1024
        # Use robust statistics for all files; means are noisy in cold I/O.
        use_median = True
        runs = 30 if size_mb < 0.1 else 12 if size_mb < 1.0 else 5
        # Tight races on a few large image buckets are sensitive to scheduling/cache
        # jitter; use more samples so ranking reflects steady behavior.
        if name in {"large_int8_2d", "large_float64_2d", "large_int64_2d"}:
            runs = max(runs, 15)
        # WCS benchmark is a tight race on a small-ish file and can occasionally
        # flip on scheduler noise; sample more for stable ranking.
        if name == "wcs_image":
            runs = max(runs, 15)

        # Parse file characteristics
        parts = name.split("_")
        file_type = self._get_file_type(name)
        data_type = next((p for p in parts if p in self.data_types.keys()), "unknown")
        dimensions = next((p for p in parts if p in ["1d", "2d", "3d"]), "unknown")
        compression = self._get_compression_type(name)

        print(
            f"\n{name} ({size_mb:.2f} MB) - {file_type} {data_type} {dimensions} {compression}"
        )
        print("-" * 80)

        result = {
            "filename": name,
            "operation": "read_full",
            "file_type": file_type,
            "size_mb": size_mb,
            "data_type": data_type,
            "dimensions": dimensions,
            "compression": compression,
        }

        # Determine HDU index for files with data in extensions
        if file_type in {"compressed", "table", "mef", "multi_mef"}:
            hdu_num = 1
        else:
            hdu_num = 0
        expected_payload_mb = (
            self._expected_image_payload_mb(filepath, hdu_num)
            if file_type != "table"
            else None
        )

        # Compression timings are noisy; use more samples for stability.
        if file_type == "compressed":
            runs = max(runs, 30)
        # Very small table files are dominated by scheduler/open jitter.
        # Use higher sample count so sub-ms rankings are stable.
        if file_type == "table" and size_mb < 0.1:
            runs = max(runs, 60)
        tf_mmap_mode = self._torchfits_mmap_mode()
        tf_mmap_cpp = self._torchfits_effective_mmap_bool(filepath, hdu_num)

        # Define benchmark methods
        methods = {}
        diagnostic_methods = {}

        # Always test torchfits
        methods["torchfits"] = lambda: torchfits.read(
            str(filepath),
            hdu=hdu_num,
            mmap=tf_mmap_mode,
            cache_capacity=self.cache_capacity,
            handle_cache_capacity=self.handle_cache_capacity,
        )
        # Fair comparison against numpy-returning methods (fitsio/astropy):
        # return a numpy payload (CPU) instead of a torch.Tensor.
        try:
            import torchfits.cpp as cpp

            if file_type == "table":
                if hasattr(cpp, "read_fits_table_rows_numpy"):
                    methods["torchfits_numpy"] = lambda: cpp.read_fits_table_rows_numpy(
                        str(filepath),
                        hdu_num,
                        [],
                        1,
                        -1,
                        self.use_mmap,
                    )
                elif hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):

                    def _table_numpy_from_handle():
                        fh = cpp.open_fits_file(str(filepath), "r")
                        try:
                            return cpp.read_fits_table_rows_numpy_from_handle(
                                fh, hdu_num, [], 1, -1
                            )
                        finally:
                            try:
                                fh.close()
                            except Exception:
                                pass

                    methods["torchfits_numpy"] = _table_numpy_from_handle
            else:
                # In practice, direct numpy path is best for most dtypes, while the
                # cached-numpy bridge can be slightly better on 64-bit payloads and
                # some 64-bit payloads.
                prefer_cached_numpy = data_type in {"int8", "int64", "float64"}
                if prefer_cached_numpy and hasattr(cpp, "read_full_numpy_cached"):
                    methods["torchfits_numpy"] = lambda: cpp.read_full_numpy_cached(
                        str(filepath), hdu_num, tf_mmap_cpp
                    )
                elif hasattr(cpp, "read_full_numpy"):
                    # Avoid torch->numpy conversion overhead when comparing against
                    # numpy-native libraries like fitsio/astropy.
                    methods["torchfits_numpy"] = lambda: cpp.read_full_numpy(
                        str(filepath), hdu_num, tf_mmap_cpp
                    )
                elif hasattr(cpp, "read_full_numpy_cached"):
                    # Fallback for older builds that only expose the cached variant.
                    methods["torchfits_numpy"] = lambda: cpp.read_full_numpy_cached(
                        str(filepath), hdu_num, tf_mmap_cpp
                    )
                else:
                    methods["torchfits_numpy"] = lambda: torchfits.read(
                        str(filepath),
                        hdu=hdu_num,
                        mmap=tf_mmap_mode,
                        cache_capacity=self.cache_capacity,
                        handle_cache_capacity=self.handle_cache_capacity,
                    ).numpy()
        except Exception:
            if file_type != "table":
                methods["torchfits_numpy"] = lambda: torchfits.read(
                    str(filepath),
                    hdu=hdu_num,
                    mmap=tf_mmap_mode,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                ).numpy()
        if file_type == "table" and "torchfits_numpy" not in methods:

            def _table_numpy_fallback():
                table = torchfits.read(
                    str(filepath),
                    hdu=hdu_num,
                    mmap=tf_mmap_mode,
                    cache_capacity=self.cache_capacity,
                    handle_cache_capacity=self.handle_cache_capacity,
                )
                out = {}
                for k, v in table.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = v.numpy()
                    elif isinstance(v, list):
                        out[k] = [
                            item.numpy() if isinstance(item, torch.Tensor) else item
                            for item in v
                        ]
                    elif isinstance(v, tuple):
                        out[k] = tuple(
                            item.numpy() if isinstance(item, torch.Tensor) else item
                            for item in v
                        )
                    else:
                        out[k] = v
                return out

            methods["torchfits_numpy"] = _table_numpy_fallback

        # Test torchfits mmap explicitly for tables
        if file_type == "table":
            methods["torchfits_mmap"] = lambda: torchfits.read(
                str(filepath),
                hdu=hdu_num,
                mmap=True,
                cache_capacity=self.cache_capacity,
                handle_cache_capacity=self.handle_cache_capacity,
            )

        # Test astropy if available
        methods["astropy"] = lambda: self._astropy_read(filepath, hdu_num)
        methods["astropy_torch"] = lambda: self._astropy_to_torch(filepath, hdu_num)

        # Test fitsio if available
        methods["fitsio"] = lambda: self._fitsio_read(filepath, hdu_num)
        methods["fitsio_torch"] = lambda: self._fitsio_to_torch(filepath, hdu_num)

        # Run benchmarks (shuffle order to reduce cache bias). We buffer output and
        # print a single ranked table per benchmark case (no duplicated unranked view).
        case_label = f"{name} ({size_mb:.2f} MB) - {file_type} {data_type} {dimensions} {compression}"

        progress_enabled = (
            sys.stdout.isatty() or os.getenv("TORCHFITS_BENCH_PROGRESS") == "1"
        )
        progress_log_started = False

        def _progress(method: str) -> None:
            """
            One-line progress indicator.
            - TTY: update in-place on the same line (cleared).
            - Non-TTY (e.g. piped to tee): default OFF to avoid log spam. If enabled via
              `TORCHFITS_BENCH_PROGRESS=1`, emit a *single* line by appending method names,
              then finish with 'done'.
            """
            nonlocal progress_log_started
            if not progress_enabled:
                return

            try:
                if sys.stdout.isatty():
                    # In-place update for interactive runs (clear line first).
                    line = f"{case_label} : timing {method}"
                    sys.stdout.write("\r\033[2K" + line)
                    sys.stdout.flush()
                    return

                # Non-interactive logs: keep everything on one line (no carriage returns).
                if method == "done":
                    if not progress_log_started:
                        sys.stdout.write(f"{case_label} : done\n")
                    else:
                        sys.stdout.write(" done\n")
                    sys.stdout.flush()
                    return

                if not progress_log_started:
                    sys.stdout.write(f"{case_label} : timing {method}")
                    progress_log_started = True
                else:
                    sys.stdout.write(f" {method}")
                sys.stdout.flush()
            except Exception:
                pass

        method_results = {}
        method_items = list(methods.items())
        random.shuffle(method_items)
        for method_name, method_func in method_items:
            _progress(method_name)

            method_result = self._time_method(
                method_func,
                method_name,
                runs=runs,
                use_median=use_median,
                expected_payload_mb=expected_payload_mb,
            )
            method_results[method_name] = method_result

            if method_result:
                time_value = (
                    method_result["median"] if use_median else method_result["mean"]
                )
                result[f"{method_name}_mean"] = method_result["mean"]
                result[f"{method_name}_std"] = method_result["std"]
                result[f"{method_name}_median"] = method_result["median"]
                result[f"{method_name}_mb_s"] = (
                    size_mb / time_value if time_value else None
                )
                result[f"{method_name}_memory"] = method_result["memory"]
                result[f"{method_name}_peak_memory"] = method_result["peak_memory"]
                payload_mb = method_result.get("payload_memory", 0.0)
                result[f"{method_name}_payload_mb"] = payload_mb
            else:
                result[f"{method_name}_mean"] = None
                result[f"{method_name}_std"] = None
                result[f"{method_name}_median"] = None
                result[f"{method_name}_mb_s"] = None
                result[f"{method_name}_memory"] = None
                result[f"{method_name}_peak_memory"] = None
                result[f"{method_name}_payload_mb"] = None

        # Run diagnostic methods (useful for debugging overhead; included in the single
        # ranked output but marked as diagnostic).
        # Open after the main loop so cache clears don't invalidate the handle.
        try:
            file_handle = torchfits.cpp.open_fits_file(str(filepath), "r")
        except Exception:
            file_handle = None
        if file_handle is not None:
            # Reuse an already-open file handle. For image HDUs this benchmarks
            # the C++ image read without open/close overhead; for table HDUs it
            # benchmarks the handle-based table reader.
            if file_type == "table":
                diagnostic_methods["torchfits_cpp_open_once"] = (
                    lambda fh=file_handle: torchfits.cpp.read_fits_table_from_handle(
                        fh, hdu_num
                    )
                )
            else:
                diagnostic_methods["torchfits_cpp_open_once"] = (
                    lambda fh=file_handle: torchfits.cpp.read_full(
                        fh, hdu_num, tf_mmap_cpp
                    )
                )
        diagnostic_methods["torchfits_hot"] = lambda: torchfits.read(
            str(filepath),
            hdu=hdu_num,
            mmap=tf_mmap_mode,
            cache_capacity=self.hot_cache_capacity,
            handle_cache_capacity=self.handle_cache_capacity,
        )
        diagnostic_methods["torchfits_handle_cache"] = lambda: torchfits.read(
            str(filepath),
            hdu=hdu_num,
            mmap=tf_mmap_mode,
            cache_capacity=0,
            handle_cache_capacity=self.hot_cache_capacity,
        )
        diagnostic_results = {}
        for method_name, method_func in diagnostic_methods.items():
            _progress(method_name)

            method_result = self._time_method(
                method_func,
                method_name,
                runs=runs,
                use_median=use_median,
                expected_payload_mb=expected_payload_mb,
            )
            diagnostic_results[method_name] = method_result
            if method_result:
                time_value = (
                    method_result["median"] if use_median else method_result["mean"]
                )
                result[f"{method_name}_mean"] = method_result["mean"]
                result[f"{method_name}_std"] = method_result["std"]
                result[f"{method_name}_median"] = method_result["median"]
                result[f"{method_name}_mb_s"] = (
                    size_mb / time_value if time_value else None
                )
                result[f"{method_name}_memory"] = method_result["memory"]
                result[f"{method_name}_peak_memory"] = method_result["peak_memory"]
                result[f"{method_name}_payload_mb"] = method_result.get(
                    "payload_memory", 0.0
                )
            else:
                result[f"{method_name}_mean"] = None
                result[f"{method_name}_std"] = None
                result[f"{method_name}_median"] = None
                result[f"{method_name}_mb_s"] = None
                result[f"{method_name}_memory"] = None
                result[f"{method_name}_peak_memory"] = None
                result[f"{method_name}_payload_mb"] = None

        # Finish progress indicator cleanly.
        try:
            if progress_enabled:
                _progress("done")
                if sys.stdout.isatty():
                    # Clear the in-place progress line before printing the ranked table.
                    sys.stdout.write("\r\033[2K\n")
                    sys.stdout.flush()
        except Exception:
            pass

        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass

        # Single ranked display for this benchmark case.
        # Include both main methods and diagnostics, but clearly tag diagnostics.
        stat_label = "median" if use_median else "mean"
        display_rows = []
        for mname, res in method_results.items():
            if not res or res.get("mean") is None:
                display_rows.append((mname, None, None, None, None, None, "main"))
                continue
            t = res["median"] if use_median else res["mean"]
            display_rows.append(
                (
                    mname,
                    float(t),
                    float(res.get("spread", 0.0)),
                    float(res.get("memory", 0.0)),
                    float(res.get("peak_memory", 0.0)),
                    float(res.get("payload_memory", 0.0)),
                    "main",
                )
            )
        for mname, res in diagnostic_results.items():
            if not res or res.get("mean") is None:
                display_rows.append((mname, None, None, None, None, None, "diag"))
                continue
            t = res["median"] if use_median else res["mean"]
            display_rows.append(
                (
                    mname,
                    float(t),
                    float(res.get("spread", 0.0)),
                    float(res.get("memory", 0.0)),
                    float(res.get("peak_memory", 0.0)),
                    float(res.get("payload_memory", 0.0)),
                    "diag",
                )
            )

        ok = [r for r in display_rows if r[1] is not None]
        ok.sort(key=lambda x: x[1])
        print(f"\nRanked results ({stat_label}, fastest first):")
        best_time = ok[0][1] if ok else None

        # Print a single aligned table.
        # Columns: rank, method, time, std, x_best, MB/s, rssΔ, peakΔ, payload
        # Note: RSS deltas are best-effort process-level signals; payload is deterministic nbytes.
        method_w = 28
        header = (
            f"{'#':>2s}  {'method':<{method_w}s}  {stat_label + '_s':>10s}  {'spread_s':>10s}  "
            f"{'x_best':>8s}  {'MB/s':>10s}  {'rssΔ_MB':>8s}  {'peakΔ_MB':>8s}  {'payload_MB':>10s}"
        )
        print(header)
        print("-" * len(header))

        ranked = 1
        for mname, t, s, rssd, peakd, payload, kind in ok:
            ratio = (t / best_time) if (best_time and t is not None) else float("inf")
            mb_s = (size_mb / t) if (t and t > 0) else float("nan")
            label = mname + (" [diag]" if kind != "main" else "")
            print(
                f"{ranked:2d}  {label:<{method_w}.{method_w}s}  {t:10.6f}  {s:10.6f}  "
                f"{ratio:8.2f}  {mb_s:10.1f}  {rssd:8.2f}  {peakd:8.2f}  {payload:10.2f}"
            )
            ranked += 1

        failed = [r for r in display_rows if r[1] is None]
        for mname, *_rest in failed:
            label = mname + (
                " [diag]" if any(mname == k for k in diagnostic_results) else ""
            )
            print(f"{'--':>2s}  {label:<{method_w}.{method_w}s}  {'FAILED':>10s}")

        # Analyze results
        valid_methods = {}
        for k, v in method_results.items():
            if v and v["mean"] is not None:
                valid_methods[k] = v["median"] if use_median else v["mean"]
        if valid_methods:
            best_method = min(valid_methods.keys(), key=lambda k: valid_methods[k])
            sorted_methods = sorted(valid_methods.items(), key=lambda x: x[1])
            torchfits_rank = next(
                (i + 1 for i, (k, v) in enumerate(sorted_methods) if k == "torchfits"),
                len(sorted_methods) + 1,
            )

            result["best_method"] = best_method
            result["torchfits_rank"] = torchfits_rank

            # Calculate speedup vs best
            if "torchfits" in valid_methods:
                best_time = valid_methods[best_method]
                tf_time = valid_methods["torchfits"]
                speedup = (
                    best_time / tf_time
                    if best_method != "torchfits"
                    else tf_time
                    / min(v for k, v in valid_methods.items() if k != "torchfits")
                )
                result["speedup_vs_best"] = speedup

                # Keep the console output to a single ranked table; store summary in CSV only.
            else:
                result["speedup_vs_best"] = None

            # Split rankings by return type (torch tensor vs numpy array) for fair comparisons.
            torch_methods = {"torchfits", "fitsio_torch", "astropy_torch"}
            numpy_methods = {"torchfits_numpy", "fitsio", "astropy"}

            torch_valid = {k: v for k, v in valid_methods.items() if k in torch_methods}
            if torch_valid:
                best_torch = min(torch_valid.keys(), key=lambda k: torch_valid[k])
                sorted_torch = sorted(torch_valid.items(), key=lambda x: x[1])
                tf_rank_torch = next(
                    (
                        i + 1
                        for i, (k, _) in enumerate(sorted_torch)
                        if k == "torchfits"
                    ),
                    len(sorted_torch) + 1,
                )
                result["best_method_torch"] = best_torch
                result["torchfits_rank_torch"] = tf_rank_torch
                if "torchfits" in torch_valid:
                    result["speedup_vs_best_torch"] = (
                        torch_valid[best_torch] / torch_valid["torchfits"]
                    )
                else:
                    result["speedup_vs_best_torch"] = None
            else:
                result["best_method_torch"] = "none"
                result["torchfits_rank_torch"] = 999
                result["speedup_vs_best_torch"] = None

            numpy_valid = {k: v for k, v in valid_methods.items() if k in numpy_methods}
            if numpy_valid:
                best_numpy = min(numpy_valid.keys(), key=lambda k: numpy_valid[k])
                sorted_numpy = sorted(numpy_valid.items(), key=lambda x: x[1])
                tf_np_rank = next(
                    (
                        i + 1
                        for i, (k, _) in enumerate(sorted_numpy)
                        if k == "torchfits_numpy"
                    ),
                    len(sorted_numpy) + 1,
                )
                result["best_method_numpy"] = best_numpy
                result["torchfits_numpy_rank"] = tf_np_rank
                if "torchfits_numpy" in numpy_valid:
                    result["speedup_vs_best_numpy"] = (
                        numpy_valid[best_numpy] / numpy_valid["torchfits_numpy"]
                    )
                else:
                    result["speedup_vs_best_numpy"] = None
            else:
                result["best_method_numpy"] = "none"
                result["torchfits_numpy_rank"] = 999
                result["speedup_vs_best_numpy"] = None

            # Explicitly report the "switch to torch" goal metric in the CSV only.
        else:
            result["best_method"] = "none"
            result["torchfits_rank"] = 999
            result["speedup_vs_best"] = None
            result["best_method_torch"] = "none"
            result["torchfits_rank_torch"] = 999
            result["speedup_vs_best_torch"] = None
            result["best_method_numpy"] = "none"
            result["torchfits_numpy_rank"] = 999
            result["speedup_vs_best_numpy"] = None

        return result

    def _estimate_data_size_mb(self, data) -> float:
        """Estimate payload size (MB) of benchmark return values."""
        visited = set()

        def _sizeof(obj) -> int:
            obj_id = id(obj)
            if obj_id in visited:
                return 0
            visited.add(obj_id)

            if obj is None:
                return 0

            if hasattr(obj, "element_size") and hasattr(obj, "numel"):
                try:
                    return int(obj.element_size() * obj.numel())
                except Exception:
                    pass

            if hasattr(obj, "nbytes"):
                try:
                    return int(obj.nbytes)
                except Exception:
                    pass

            if isinstance(obj, dict):
                return sum(_sizeof(v) for v in obj.values())

            if isinstance(obj, (list, tuple)):
                return sum(_sizeof(v) for v in obj)

            return 0

        return _sizeof(data) / 1024 / 1024

    def _expected_image_payload_mb(
        self, filepath: Path, hdu_num: int
    ) -> Optional[float]:
        """Estimate expected uncompressed image payload size (MB) from FITS headers."""

        def _to_int(value, default=0) -> int:
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return int(default)

        try:
            header = fitsio.read_header(str(filepath), ext=hdu_num)
        except Exception:
            return None

        use_compressed_axes = _to_int(header.get("ZNAXIS", 0), 0) > 0
        axis_prefix = "ZNAXIS" if use_compressed_axes else "NAXIS"
        naxis = _to_int(header.get(axis_prefix, 0), 0)
        if naxis <= 0:
            return None

        bitpix_key = "ZBITPIX" if use_compressed_axes else "BITPIX"
        bitpix = abs(_to_int(header.get(bitpix_key, header.get("BITPIX", 0)), 0))
        if bitpix <= 0:
            return None

        elements = 1
        for i in range(1, naxis + 1):
            axis_len = _to_int(header.get(f"{axis_prefix}{i}", 0), 0)
            if axis_len <= 0:
                return None
            elements *= axis_len

        return (elements * (bitpix / 8.0)) / (1024.0 * 1024.0)

    def _time_method(
        self,
        method_func,
        method_name: str,
        runs: int = 3,
        use_median: bool = False,
        expected_payload_mb: Optional[float] = None,
    ) -> Optional[Dict]:
        """Time a method with robust RSS memory tracking."""
        times = []
        rss_increase_mb = []
        peak_rss_increase_mb = []
        payload_memory_mb = []

        process = psutil.Process()
        # RSS sampling is best-effort. It can under-report brief peaks, and RSS deltas
        # may be near-zero due to allocator reuse. Treat it as a coarse signal.
        sample_interval_s = float(os.getenv("TORCHFITS_BENCH_RSS_INTERVAL_S", "0.001"))
        warmup_runs = max(0, int(os.getenv("TORCHFITS_BENCH_WARMUP_RUNS", "1")))

        # Clear torchfits caches once per timed method (not per run). Clearing inside
        # the run loop makes the benchmark highly noisy and unrealistically penalizes
        # torchfits relative to other libs that don't expose/trigger similar cache clears.
        try:
            if (
                "torchfits" in method_name
                and "open_once" not in method_name
                and "hot" not in method_name
                and "handle_cache" not in method_name
            ):
                torchfits.clear_file_cache()
        except Exception:
            pass

        # Warmup/discard-first pass to reduce cold-start noise in medians.
        for _ in range(warmup_runs):
            try:
                gc.collect()
                for _ in range(2):
                    gc.collect()
                data = method_func()
                del data
            except Exception as e:
                print(f"Error in {method_name} warmup: {e}")
                return None

        for _ in range(runs):
            sampler_stop = None
            sampler_thread = None
            initial_rss_mb = 0.0
            peak_rss_mb = 0.0

            try:
                gc.collect()
                for _ in range(3):  # Extra cleanup
                    gc.collect()

                initial_rss_mb = process.memory_info().rss / 1024 / 1024
                peak_rss_mb = initial_rss_mb
                sampler_stop = threading.Event()

                def _sample_rss(stop_event):
                    nonlocal peak_rss_mb
                    while not stop_event.is_set():
                        try:
                            current_rss_mb = process.memory_info().rss / 1024 / 1024
                        except Exception:
                            break
                        if current_rss_mb > peak_rss_mb:
                            peak_rss_mb = current_rss_mb
                        stop_event.wait(sample_interval_s)

                sampler_thread = threading.Thread(
                    target=_sample_rss, args=(sampler_stop,), daemon=True
                )
                sampler_thread.start()

                # Time the operation
                start_time = time.perf_counter()
                data = method_func()
                elapsed = time.perf_counter() - start_time

                final_rss_mb = process.memory_info().rss / 1024 / 1024
                peak_rss_mb = max(peak_rss_mb, final_rss_mb)
                payload_mb = self._estimate_data_size_mb(data)
                if expected_payload_mb is not None and expected_payload_mb > 0:
                    min_expected_mb = expected_payload_mb * self.payload_min_ratio
                    if payload_mb < min_expected_mb:
                        raise RuntimeError(
                            "payload check failed "
                            f"(observed={payload_mb:.4f}MB, "
                            f"expected>={min_expected_mb:.4f}MB, "
                            f"ratio={self.payload_min_ratio:.2f})"
                        )

                times.append(elapsed)
                rss_increase_mb.append(max(0.0, final_rss_mb - initial_rss_mb))
                peak_rss_increase_mb.append(max(0.0, peak_rss_mb - initial_rss_mb))
                payload_memory_mb.append(payload_mb)

                del data
                gc.collect()

            except Exception as e:
                print(f"Error in {method_name}: {e}")
                return None
            finally:
                if sampler_stop is not None:
                    sampler_stop.set()
                if sampler_thread is not None:
                    sampler_thread.join(timeout=0.05)

        def _percentile(sorted_vals, q: float) -> float:
            # Linear interpolation between closest ranks (like numpy default).
            if not sorted_vals:
                return 0.0
            if len(sorted_vals) == 1:
                return float(sorted_vals[0])
            q = float(min(1.0, max(0.0, q)))
            pos = (len(sorted_vals) - 1) * q
            lo = int(pos)
            hi = min(lo + 1, len(sorted_vals) - 1)
            if hi == lo:
                return float(sorted_vals[lo])
            frac = pos - lo
            return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)

        if times:
            sorted_times = sorted(times)
            p10 = _percentile(sorted_times, 0.10)
            p90 = _percentile(sorted_times, 0.90)
            return {
                "mean": mean(times),
                "median": median(times),
                "std": stdev(times) if len(times) > 1 else 0,
                # Robust spread: helps spot outliers without being dominated by them.
                "spread": p90 - p10,
                "memory": mean(rss_increase_mb),
                "peak_memory": mean(peak_rss_increase_mb),
                "payload_memory": mean(payload_memory_mb),
            }
        return None

    def _benchmark_cutouts(self, files: Dict[str, Path]) -> List[Dict]:
        """Benchmark random access cutouts on large files."""
        print("\n" + "=" * 100)
        print("CUTOUT BENCHMARK")
        print("=" * 100)

        results: List[Dict] = []

        def _run_case(
            target_name: str,
            target_file: Path,
            hdu_idx: int,
            compression: str,
        ) -> Dict:
            print(
                f"Benchmarking cutouts on {target_name} (hdu={hdu_idx}, {compression})..."
            )

            cutout_size = (100, 100)
            n_iter = 50

            x1, y1 = 100, 100
            x2 = x1 + cutout_size[1]
            y2 = y1 + cutout_size[0]

            methods = {
                "torchfits": lambda: torchfits.read_subset(
                    str(target_file), hdu_idx, x1, y1, x2, y2
                ),
                "astropy": lambda: self._astropy_cutout(
                    target_file, hdu_idx, x1, y1, x2, y2
                ),
                "fitsio": lambda: self._fitsio_cutout(
                    target_file, hdu_idx, x1, y1, x2, y2
                ),
            }

            row = {
                "filename": target_name,
                "operation": "cutout_100x100",
                "file_type": self._get_file_type(target_name),
                "size_mb": target_file.stat().st_size / 1024 / 1024,
                "data_type": "mixed",
                "dimensions": "2d",
                "compression": compression,
            }

            for name, func in methods.items():
                res = self._time_method(func, name, runs=n_iter)
                if res:
                    row[f"{name}_mean"] = res["mean"]
                    row[f"{name}_std"] = res["std"]
                    row[f"{name}_median"] = res["median"]
                    row[f"{name}_mb_s"] = (
                        row["size_mb"] / res["median"] if res["median"] else None
                    )
                    row[f"{name}_memory"] = res["memory"]
                    row[f"{name}_peak_memory"] = res["peak_memory"]
                    print(
                        f"{name:15s}: {res['mean'] * 1e6:.2f}us ± {res['std'] * 1e6:.2f}us"
                    )
                else:
                    row[f"{name}_mean"] = None
                    row[f"{name}_std"] = None
                    row[f"{name}_median"] = None
                    row[f"{name}_mb_s"] = None
                    row[f"{name}_memory"] = None
                    row[f"{name}_peak_memory"] = None

            valid_methods = {
                method_name: row[f"{method_name}_median"]
                for method_name in methods
                if row.get(f"{method_name}_median") is not None
            }
            if valid_methods:
                best_method = min(valid_methods, key=valid_methods.get)
                sorted_methods = sorted(valid_methods.items(), key=lambda kv: kv[1])
                torchfits_rank = next(
                    (
                        idx + 1
                        for idx, (nm, _) in enumerate(sorted_methods)
                        if nm == "torchfits"
                    ),
                    len(sorted_methods) + 1,
                )
                row["best_method"] = best_method
                row["torchfits_rank"] = torchfits_rank

                tf_time = valid_methods.get("torchfits")
                if tf_time:
                    if best_method == "torchfits":
                        competitor_times = [
                            v for k, v in valid_methods.items() if k != "torchfits"
                        ]
                        row["speedup_vs_best"] = (
                            tf_time / min(competitor_times)
                            if competitor_times
                            else None
                        )
                    else:
                        row["speedup_vs_best"] = valid_methods[best_method] / tf_time
                else:
                    row["speedup_vs_best"] = None
            else:
                row["best_method"] = "none"
                row["torchfits_rank"] = 999
                row["speedup_vs_best"] = None

            return row

        # Find a suitable large file (MEF or large image)
        target_file = None
        target_name = None

        # Prefer multi_mef, then mef, then large single
        for name, path in files.items():
            if "multi_mef" in name:
                target_file = path
                target_name = name
                break

        if not target_file:
            for name, path in files.items():
                if "mef" in name and "large" in name:
                    target_file = path
                    target_name = name
                    break

        if target_file and target_name:
            hdu_idx = 5 if "multi_mef" in target_name else 1
            results.append(_run_case(target_name, target_file, hdu_idx, "uncompressed"))
        else:
            print("No suitable file found for uncompressed cutout benchmark.")

        # Compressed cutouts (extension 1)
        comp_name = None
        comp_file = None
        for key in ["compressed_rice_1", "compressed_gzip_1", "compressed_hcompress_1"]:
            if key in files:
                comp_name = key
                comp_file = files[key]
                break
        if comp_name and comp_file:
            results.append(_run_case(comp_name, comp_file, 1, "compressed"))
        else:
            print("No suitable file found for compressed cutout benchmark.")

        return results

    def _benchmark_random_extension_reads(self, files: Dict[str, Path]) -> List[Dict]:
        """Benchmark random extension full reads on a multi-extension FITS."""
        print("\n" + "=" * 100)
        print("RANDOM EXTENSION READ BENCHMARK")
        print("=" * 100)

        target = files.get("multi_mef_10ext")
        if target is None:
            print("No multi_mef_10ext file found for extension benchmark.")
            return []

        target_name = "multi_mef_10ext"
        size_mb = target.stat().st_size / 1024 / 1024
        n_iter = 200
        ext_seq = [((i * 3) % 10) + 1 for i in range(n_iter)]  # 1..10
        # This benchmark targets extension-switch overhead. For repeated random
        # HDU hops, mmap data-path setup dominates and is not representative of
        # the intended comparison, so we force non-mmap here.
        ext_use_mmap = False

        def tf_cached_handles():
            if hasattr(torchfits.cpp, "read_hdus_sequence_last"):
                return torchfits.cpp.read_hdus_sequence_last(
                    str(target), ext_seq, ext_use_mmap
                )
            out = None
            for ext in ext_seq:
                out = torchfits.read(
                    str(target),
                    hdu=ext,
                    mmap=ext_use_mmap,
                    cache_capacity=0,
                    handle_cache_capacity=self.handle_cache_capacity,
                )
            return out

        def tf_cpp_open_once():
            fh = torchfits.cpp.open_fits_file(str(target), "r")
            try:
                out = None
                for ext in ext_seq:
                    out = torchfits.cpp.read_full(fh, ext, ext_use_mmap)
                return out
            finally:
                try:
                    fh.close()
                except Exception:
                    pass

        def fitsio_open_once():
            f = fitsio.FITS(str(target))
            try:
                out = None
                for ext in ext_seq:
                    out = f[ext].read()
                return out
            finally:
                try:
                    f.close()
                except Exception:
                    pass

        def astropy_open_once():
            # astropy may refuse memmap when BZERO/BSCALE keywords are present; for this
            # benchmark we care about extension access overhead, so keep it robust.
            with self._astropy_open(target, False) as hdul:
                out = None
                for ext in ext_seq:
                    # Force materialization to match torchfits/fitsio read semantics.
                    out = self._ensure_native_endian_numpy(
                        np.array(hdul[ext].data, copy=True)
                    )
                return out

        methods = {
            "torchfits": tf_cached_handles,
            "torchfits_cpp_open_once": tf_cpp_open_once,
            "fitsio": fitsio_open_once,
            "astropy": astropy_open_once,
        }

        row: Dict[str, Any] = {
            "filename": target_name,
            "operation": "random_ext_full_reads_200",
            "file_type": "multi_mef",
            "size_mb": size_mb,
            "data_type": "mixed",
            "dimensions": "2d",
            "compression": "uncompressed",
        }

        # Each method already does 200 reads; keep repeat count small.
        for name, func in methods.items():
            res = self._time_method(func, name, runs=5, use_median=True)
            if res:
                row[f"{name}_mean"] = res["mean"]
                row[f"{name}_std"] = res["std"]
                row[f"{name}_median"] = res["median"]
                row[f"{name}_mb_s"] = size_mb / res["median"] if res["median"] else None
                row[f"{name}_memory"] = res["memory"]
                row[f"{name}_peak_memory"] = res["peak_memory"]
                row[f"{name}_payload_mb"] = res["payload_memory"]
                print(f"{name:22s}: {res['median'] * 1e3:.3f}ms (median)")
            else:
                row[f"{name}_mean"] = None
                row[f"{name}_std"] = None
                row[f"{name}_median"] = None
                row[f"{name}_mb_s"] = None
                row[f"{name}_memory"] = None
                row[f"{name}_peak_memory"] = None
                row[f"{name}_payload_mb"] = None

        # Ensure all known columns exist for CSV compatibility.
        for missing in [
            "fitsio_torch",
            "astropy_torch",
            "torchfits_numpy",
            "torchfits_hot",
            "torchfits_handle_cache",
        ]:
            if f"{missing}_median" not in row:
                row[f"{missing}_mean"] = None
                row[f"{missing}_std"] = None
                row[f"{missing}_median"] = None
                row[f"{missing}_mb_s"] = None
                row[f"{missing}_memory"] = None
                row[f"{missing}_peak_memory"] = None
                row[f"{missing}_payload_mb"] = None

        return [row]

    def _benchmark_table_scan_arrow(self, files: Dict[str, Path]) -> List[Dict]:
        """Benchmark Arrow out-of-core scan path (torchfits only)."""
        if not self.include_tables:
            return []
        target = files.get("table_survey_catalog")
        if target is None:
            return []

        print("\n" + "=" * 100)
        print("TABLE SCAN (ARROW) BENCHMARK")
        print("=" * 100)

        import pyarrow as pa  # noqa: F401

        def tf_scan_count():
            total = 0
            for batch in torchfits.table.scan(
                str(target),
                hdu=1,
                batch_size=65536,
                mmap=True,
                backend="cpp_numpy",
                include_fits_metadata=False,
                apply_fits_nulls=False,
            ):
                total += int(batch.num_rows)
            return total

        res = self._time_method(tf_scan_count, "torchfits", runs=5, use_median=True)
        size_mb = target.stat().st_size / 1024 / 1024
        row: Dict[str, Any] = {
            "filename": "table_survey_catalog",
            "operation": "table_scan_arrow_count",
            "file_type": "table",
            "size_mb": size_mb,
            "data_type": "mixed",
            "dimensions": "2d",
            "compression": "uncompressed",
        }

        if res:
            row["torchfits_mean"] = res["mean"]
            row["torchfits_std"] = res["std"]
            row["torchfits_median"] = res["median"]
            row["torchfits_mb_s"] = size_mb / res["median"] if res["median"] else None
            row["torchfits_memory"] = res["memory"]
            row["torchfits_peak_memory"] = res["peak_memory"]
            row["torchfits_payload_mb"] = res["payload_memory"]

        # Fill other columns with None.
        for prefix in [
            "torchfits_numpy",
            "torchfits_hot",
            "torchfits_handle_cache",
            "torchfits_mmap",
            "astropy",
            "fitsio",
            "astropy_torch",
            "fitsio_torch",
            "torchfits_cpp_open_once",
        ]:
            for suffix in [
                "mean",
                "std",
                "median",
                "mb_s",
                "memory",
                "peak_memory",
                "payload_mb",
            ]:
                row.setdefault(f"{prefix}_{suffix}", None)
        row.setdefault("best_method", "torchfits")
        row.setdefault("torchfits_rank", 1)
        row.setdefault("speedup_vs_best", None)
        row.setdefault("best_method_torch", "torchfits")
        row.setdefault("torchfits_rank_torch", 1)
        row.setdefault("speedup_vs_best_torch", None)
        row.setdefault("best_method_numpy", "none")
        row.setdefault("torchfits_numpy_rank", 999)
        row.setdefault("speedup_vs_best_numpy", None)

        if res:
            print(f"torchfits (scan count): {res['median']:.6f}s (median)")

        return [row]

    def _astropy_cutout(self, path, hdu, x1, y1, x2, y2):
        try:
            with self._astropy_open(path, True) as hdul:
                return hdul[hdu].section[y1:y2, x1:x2]
        except Exception:
            with self._astropy_open(path, False) as hdul:
                return hdul[hdu].section[y1:y2, x1:x2]

    def _fitsio_cutout(self, path, hdu, x1, y1, x2, y2):
        with fitsio.FITS(str(path)) as f:
            return f[hdu][y1:y2, x1:x2]

    def _get_file_type(self, name: str) -> str:
        """Determine file type from name."""
        if "multi_mef" in name:
            return "multi_mef"
        elif "mef" in name:
            return "mef"
        elif "table" in name:
            return "table"
        elif "scaled" in name:
            return "scaled"
        elif "wcs" in name:
            return "wcs"
        elif "compressed" in name:
            return "compressed"
        elif "timeseries" in name:
            return "timeseries"
        else:
            return "single"

    def _get_compression_type(self, name: str) -> str:
        """Determine compression type from name."""
        for comp in self.compression_types:
            if comp.lower() in name:
                return comp.lower()
        return "uncompressed"

    @contextmanager
    def _astropy_open(self, filepath: Path, memmap: bool):
        """Open FITS with astropy, falling back to non-mmap when scaling blocks memmap."""
        hdul = None
        try:
            try:
                hdul = astropy_fits.open(filepath, memmap=memmap)
            except Exception:
                if memmap:
                    hdul = astropy_fits.open(filepath, memmap=False)
                else:
                    raise
            yield hdul
        finally:
            if hdul is not None:
                hdul.close()

    def _ensure_native_endian_numpy(self, arr):
        """Return a contiguous numpy array with native byte order."""
        out = np.ascontiguousarray(np.asarray(arr))
        if out.dtype.byteorder not in ("=", "|"):
            out = np.ascontiguousarray(out.astype(out.dtype.newbyteorder("=")))
        return out

    def _table_to_numpy_dict(self, table_data):
        """Convert table-like column accessor to dict[str, np.ndarray]."""
        return {
            col: self._ensure_native_endian_numpy(table_data[col])
            for col in table_data.names
        }

    def _astropy_read(self, filepath: Path, hdu_num: int):
        """Pure astropy read - handles both images and tables."""
        try:
            with self._astropy_open(filepath, self.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        # Table: convert to dict for fair comparison
                        return self._table_to_numpy_dict(hdu.data)
                    else:
                        # Image: force materialization so this benchmark measures
                        # full-read latency (not lazy memmap view creation).
                        return self._ensure_native_endian_numpy(
                            np.array(hdu.data, copy=True)
                        )
                return None
        except Exception:
            if self.use_mmap:
                with self._astropy_open(filepath, False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            return self._table_to_numpy_dict(hdu.data)
                        return self._ensure_native_endian_numpy(
                            np.array(hdu.data, copy=True)
                        )
                    return None
            raise

    def _fitsio_read(self, filepath: Path, hdu_num: int):
        """Pure fitsio read; table payload is normalized to dict[str, np.ndarray]."""
        data = fitsio.read(str(filepath), ext=hdu_num)
        if isinstance(data, np.ndarray) and data.dtype.names:
            return {
                col: self._ensure_native_endian_numpy(data[col])
                for col in data.dtype.names
            }
        return data

    def _astropy_to_torch(self, filepath: Path, hdu_num: int):
        """Astropy read + torch conversion - handles both images and tables."""
        try:
            with self._astropy_open(filepath, self.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        # Table: convert to dict of tensors
                        data = hdu.data
                        result = {}
                        for col in data.names:
                            col_data = data[col]
                            if col_data.dtype.byteorder not in ("=", "|"):
                                col_data = col_data.astype(
                                    col_data.dtype.newbyteorder("=")
                                )
                            try:
                                if col_data.dtype.kind in ("S", "U"):
                                    # Convert strings to uint8 tensor
                                    if col_data.dtype.kind == "U":
                                        col_data = np.char.encode(col_data, "ascii")
                                    # View as uint8
                                    # Need to ensure contiguous
                                    col_data = np.ascontiguousarray(col_data)
                                    # View as uint8 (shape will be (rows, len))
                                    col_data = col_data.view("uint8").reshape(
                                        len(col_data), -1
                                    )
                                    result[col] = torch.from_numpy(col_data)
                                elif col_data.dtype.kind == "b":
                                    # Boolean
                                    result[col] = torch.from_numpy(
                                        col_data.astype(bool)
                                    )
                                else:
                                    result[col] = torch.from_numpy(col_data)
                            except Exception:
                                # Skip problematic columns
                                continue
                        return result
                    else:
                        # Image: convert to tensor
                        np_data = hdu.data.copy()
                        if np_data.dtype.byteorder not in ("=", "|"):
                            np_data = np_data.astype(np_data.dtype.newbyteorder("="))
                        return torch.from_numpy(np_data)
                return None
        except Exception:
            if self.use_mmap:
                with self._astropy_open(filepath, False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            data = hdu.data
                            result = {}
                            for col in data.names:
                                col_data = data[col]
                                if col_data.dtype.byteorder not in ("=", "|"):
                                    col_data = col_data.astype(
                                        col_data.dtype.newbyteorder("=")
                                    )
                                try:
                                    if col_data.dtype.kind in ("S", "U"):
                                        if col_data.dtype.kind == "U":
                                            col_data = np.char.encode(col_data, "ascii")
                                        col_data = np.ascontiguousarray(col_data)
                                        col_data = col_data.view("uint8").reshape(
                                            len(col_data), -1
                                        )
                                        result[col] = torch.from_numpy(col_data)
                                    elif col_data.dtype.kind == "b":
                                        result[col] = torch.from_numpy(
                                            col_data.astype(bool)
                                        )
                                    else:
                                        result[col] = torch.from_numpy(col_data)
                                except Exception:
                                    continue
                            return result
                        np_data = hdu.data.copy()
                        if np_data.dtype.byteorder not in ("=", "|"):
                            np_data = np_data.astype(np_data.dtype.newbyteorder("="))
                        return torch.from_numpy(np_data)
                    return None
            raise

    def _fitsio_to_torch(self, filepath: Path, hdu_num: int):
        """Fitsio read + torch conversion."""
        try:
            np_data = fitsio.read(str(filepath), ext=hdu_num)
            if np_data is None:
                return None
            return torch.from_numpy(np_data)
        except Exception:
            return None

    def generate_plots(self, results: List[Dict]):
        """Generate comprehensive plots from benchmark results."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        globals()["plt"] = plt
        globals()["sns"] = sns

        print("\nGenerating exhaustive plots...")
        df = pd.DataFrame(results)
        if df.empty:
            print("No results to plot.")
            return

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Performance comparison by file type
        self._plot_performance_by_type(df)

        # 2. Memory usage analysis
        self._plot_memory_usage(df)

        # 3. Speedup analysis
        self._plot_speedup_analysis(df)

        # 4. Data type performance
        self._plot_data_type_performance(df)

        # 5. File size vs performance
        self._plot_size_performance(df)

        # 6. Compression analysis
        self._plot_compression_analysis(df)

        print(f"✓ Plots saved to {self.output_dir}")

    def _plot_performance_by_type(self, df):
        """Plot performance comparison by file type."""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Performance Comparison by File Type", fontsize=16)

        # Filter valid results
        methods = ["torchfits_mean", "astropy_mean", "fitsio_mean"]

        for i, method in enumerate(methods):
            if method in df.columns:
                valid_df = df[df[method].notna()]
                if not valid_df.empty:
                    ax = axes[i // 2, i % 2]
                    sns.boxplot(data=valid_df, x="file_type", y=method, ax=ax)
                    ax.set_title(f"{method.replace('_mean', '').title()} Performance")
                    ax.set_ylabel("Time (seconds)")
                    plt.setp(ax.get_xticklabels(), rotation=45)

        # Remove empty subplot
        if len(methods) < 4:
            axes[1, 1].remove()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_by_type.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_memory_usage(self, df: pd.DataFrame):
        """Plot RSS memory growth analysis."""
        if (
            "torchfits_memory" not in df.columns
            or "torchfits_peak_memory" not in df.columns
        ):
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("RSS Memory Growth Analysis", fontsize=16)

        # RSS growth vs file size
        valid_df = df[
            df["torchfits_memory"].notna() & df["torchfits_peak_memory"].notna()
        ]
        if not valid_df.empty:
            axes[0].scatter(
                valid_df["size_mb"],
                valid_df["torchfits_memory"],
                alpha=0.6,
                label="Steady RSS increase",
            )
            axes[0].scatter(
                valid_df["size_mb"],
                valid_df["torchfits_peak_memory"],
                alpha=0.4,
                label="Peak RSS increase",
            )
            axes[0].set_xlabel("File Size (MB)")
            axes[0].set_ylabel("RSS Increase (MB)")
            axes[0].set_title("RSS Increase vs File Size")
            axes[0].legend()

        # Peak memory by data type
        if not valid_df.empty:
            sns.boxplot(
                data=valid_df, x="data_type", y="torchfits_peak_memory", ax=axes[1]
            )
            axes[1].set_title("Peak RSS Increase by Data Type")
            axes[1].set_ylabel("Peak RSS Increase (MB)")
            plt.setp(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_usage.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_speedup_analysis(self, df: pd.DataFrame):
        """Plot speedup analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create speedup comparison
        methods = [
            "astropy_mean",
            "fitsio_mean",
            "astropy_torch_mean",
            "fitsio_torch_mean",
        ]
        speedups = {}

        for method in methods:
            if method in df.columns:
                valid_df = df[(df["torchfits_mean"].notna()) & (df[method].notna())]
                if not valid_df.empty:
                    speedup = valid_df[method] / valid_df["torchfits_mean"]
                    speedups[method.replace("_mean", "")] = speedup

        if speedups:
            # Create box plot of speedups
            data_for_plot = []
            labels = []
            for method, speeds in speedups.items():
                data_for_plot.append(speeds)
                labels.append(method.replace("_", " ").title())

            ax.boxplot(data_for_plot, tick_labels=labels)
            ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="No speedup")
            ax.set_ylabel("Speedup Factor (other/torchfits)")
            ax.set_title("torchfits Speedup vs Other Methods")
            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "speedup_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_data_type_performance(self, df: pd.DataFrame):
        """Plot performance by data type."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        valid_df = df[df["torchfits_mean"].notna()]
        if not valid_df.empty:
            # Group by data type and dimensions
            perf_data = (
                valid_df.groupby(["data_type", "dimensions"])["torchfits_mean"]
                .mean()
                .reset_index()
            )

            # Create heatmap
            pivot_data = perf_data.pivot(
                index="data_type", columns="dimensions", values="torchfits_mean"
            )
            sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="viridis", ax=ax)
            ax.set_title("Average Performance by Data Type and Dimensions")
            ax.set_ylabel("Data Type")
            ax.set_xlabel("Dimensions")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "data_type_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_size_performance(self, df: pd.DataFrame):
        """Plot performance vs file size."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        valid_df = df[df["torchfits_mean"].notna()]
        if not valid_df.empty:
            # Color by file type
            file_types = valid_df["file_type"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(file_types)))

            for file_type, color in zip(file_types, colors):
                type_df = valid_df[valid_df["file_type"] == file_type]
                ax.scatter(
                    type_df["size_mb"],
                    type_df["torchfits_mean"],
                    label=file_type,
                    alpha=0.7,
                    color=color,
                )

            ax.set_xlabel("File Size (MB)")
            ax.set_ylabel("Performance (seconds)")
            ax.set_title("Performance vs File Size by Type")
            ax.legend()
            ax.set_xscale("log")
            ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "size_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_compression_analysis(self, df: pd.DataFrame):
        """Plot compression analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Compression Analysis", fontsize=16)

        # Performance by compression type
        valid_df = df[df["torchfits_mean"].notna()]
        if not valid_df.empty:
            sns.boxplot(data=valid_df, x="compression", y="torchfits_mean", ax=axes[0])
            axes[0].set_title("Performance by Compression Type")
            axes[0].set_ylabel("Time (seconds)")
            plt.setp(axes[0].get_xticklabels(), rotation=45)

            # File size reduction
            if len(valid_df["compression"].unique()) > 1:
                size_by_comp = valid_df.groupby("compression")["size_mb"].mean()
                axes[1].bar(size_by_comp.index, size_by_comp.values)
                axes[1].set_title("Average File Size by Compression")
                axes[1].set_ylabel("File Size (MB)")
                plt.setp(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "compression_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_summary_report(self, results: List[Dict]):
        """Generate a comprehensive summary report."""
        print("\nGenerating exhaustive summary report...")

        df = pd.DataFrame(results)
        tf_col = (
            "torchfits_median" if "torchfits_median" in df.columns else "torchfits_mean"
        )
        fi_col = "fitsio_median" if "fitsio_median" in df.columns else "fitsio_mean"
        tf_hot_col = (
            "torchfits_hot_median"
            if "torchfits_hot_median" in df.columns
            else "torchfits_hot_mean"
        )

        # Keep the core summary focused on full-read benchmarks.
        if "operation" in df.columns:
            op_series = df["operation"].fillna("read_full")
            primary_df = df[op_series == "read_full"].copy()
            additional_df = df[op_series != "read_full"].copy()
        else:
            primary_df = df.copy()
            additional_df = pd.DataFrame(columns=df.columns)

        def _is_valid_number(value) -> bool:
            return value is not None and pd.notna(value)

        def _format_unique(series: pd.Series) -> str:
            values = [str(v) for v in series.dropna().unique()]
            return ", ".join(sorted(values)) if values else "-"

        with open(self.summary_file, "w") as f:
            f.write("# torchfits Exhaustive Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # System information
            f.write("## System Information\n\n")
            f.write(f"- Python: {sys.version.split()[0]}\n")
            f.write(f"- PyTorch: {torch.__version__}\n")
            f.write(f"- CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"- CUDA device: {torch.cuda.get_device_name()}\n")
            f.write("- astropy available: True\n")
            f.write("- fitsio available: True\n")
            f.write(
                f"- System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB\n"
            )
            f.write(
                "- Memory metrics: process RSS deltas (mean rssΔ and peakΔ while the method runs); "
                "best-effort sampling, may under-report brief peaks and may read as ~0 due to allocator reuse.\n"
            )
            f.write("\n")

            # Test coverage summary
            f.write("## Test Coverage Summary\n\n")
            f.write(f"- Total benchmark rows: {len(df)}\n")
            f.write(f"- Primary read benchmarks: {len(primary_df)}\n")
            f.write(f"- Additional operation benchmarks: {len(additional_df)}\n")
            if primary_df.empty:
                f.write("\nNo primary read benchmarks matched the current filter.\n")
                f.write(
                    "Tip: some cases are intentionally not generated (e.g. 'large_*_3d' is skipped "
                    "to avoid large memory usage). Adjust --filter accordingly.\n"
                )
                return
            f.write(f"- File types tested: {_format_unique(primary_df['file_type'])}\n")
            f.write(f"- Data types tested: {_format_unique(primary_df['data_type'])}\n")
            f.write(f"- MMap enabled: {self.use_mmap}\n")
            f.write(f"- TorchFits mmap mode: {self._torchfits_mmap_mode()}\n")
            f.write(f"- Tables included: {self.include_tables}\n")
            f.write(f"- Benchmark profile: {self.profile}\n\n")
            f.write(f"- Cache capacity (cold): {self.cache_capacity}\n")
            f.write(f"- Handle cache capacity: {self.handle_cache_capacity}\n")
            f.write(f"- Cache capacity (hot): {self.hot_cache_capacity}\n\n")
            f.write(f"- Payload sanity min ratio: {self.payload_min_ratio:.2f}\n\n")

            # Winner summary (avoids misreading 'torchfits_rank' when multiple TorchFits variants exist)
            f.write("## Winner Summary\n\n")
            best_overall = primary_df["best_method"].fillna("none").astype(str)
            best_torch = primary_df["best_method_torch"].fillna("none").astype(str)
            best_numpy = primary_df["best_method_numpy"].fillna("none").astype(str)

            tf_family_wins = int(best_overall.str.startswith("torchfits").sum())
            tf_default_wins = int((best_overall == "torchfits").sum())
            tf_torch_wins = int((best_torch == "torchfits").sum())
            tf_numpy_wins = int((best_numpy == "torchfits_numpy").sum())
            n_primary = len(primary_df)

            f.write(
                f"- TorchFits family best overall: {tf_family_wins}/{n_primary} ({(tf_family_wins / max(n_primary, 1)):.1%})\n"
            )
            f.write(
                f"- TorchFits (torch return) best: {tf_torch_wins}/{n_primary} ({(tf_torch_wins / max(n_primary, 1)):.1%})\n"
            )
            f.write(
                f"- TorchFits (numpy return) best: {tf_numpy_wins}/{n_primary} ({(tf_numpy_wins / max(n_primary, 1)):.1%})\n"
            )
            f.write(
                f"- TorchFits default (`torchfits`) best overall: {tf_default_wins}/{n_primary} ({(tf_default_wins / max(n_primary, 1)):.1%})\n"
            )

            # Show torch-returning misses explicitly (these are the true gaps vs fitsio_torch/astropy_torch)
            misses = primary_df[best_torch != "torchfits"].copy()
            if not misses.empty:
                misses = misses.sort_values("speedup_vs_best_torch").head(10)
                f.write("\nTop cases where TorchFits (torch) is not best:\n\n")
                f.write(
                    "| File | Type | Size (MB) | Best (torch) | Speedup vs Best (torch) |\n"
                )
                f.write("|---|---|---:|---|---:|\n")
                for _, r in misses.iterrows():
                    speedup = r.get("speedup_vs_best_torch")
                    speedup_str = (
                        f"{float(speedup):.2f}x" if _is_valid_number(speedup) else "-"
                    )
                    f.write(
                        f"| {r['filename']} | {r['file_type']} | {r['size_mb']:.2f} | {r.get('best_method_torch', '-')} | {speedup_str} |\n"
                    )
            f.write("\n\n")

            # Performance Summary Table
            f.write("## Performance Table\n\n")
            f.write(
                "| File | Operation | Type | Size (MB) | TorchFits (s) | TorchFits Hot (s) | Best (torch) | Rank (torch) | Best (numpy) | Rank (numpy) | Fitsio / TorchFits |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")

            for _, r in primary_df.iterrows():
                name = r["filename"]
                operation = (
                    r.get("operation") if pd.notna(r.get("operation")) else "read_full"
                )
                ftype = r["file_type"]
                size = f"{r['size_mb']:.2f}"

                tf_time = r.get(tf_col)
                tf_hot = r.get(tf_hot_col)
                if not _is_valid_number(tf_time):
                    tf_str = "FAIL"
                    tf_hot_str = "-"
                    best_torch = "-"
                    rank_torch = "-"
                    best_numpy = "-"
                    rank_numpy = "-"
                    fitsio_over_tf = "-"
                else:
                    tf_str = f"{tf_time:.4f}"
                    tf_hot_str = f"{tf_hot:.4f}" if _is_valid_number(tf_hot) else "-"
                    best_torch = r.get("best_method_torch", "-")
                    rank_torch = r.get("torchfits_rank_torch", "-")
                    best_numpy = r.get("best_method_numpy", "-")
                    rank_numpy = r.get("torchfits_numpy_rank", "-")
                    fitsio_time = r.get(fi_col)
                    fitsio_over_tf = (
                        f"{fitsio_time / tf_time:.2f}x"
                        if _is_valid_number(fitsio_time)
                        else "-"
                    )

                f.write(
                    f"| {name} | {operation} | {ftype} | {size} | {tf_str} | {tf_hot_str} | {best_torch} | {rank_torch} | {best_numpy} | {rank_numpy} | {fitsio_over_tf} |\n"
                )

            f.write("\n")
            f.write(
                f"- Dimensions tested: {_format_unique(primary_df['dimensions'])}\n"
            )
            f.write(
                f"- Compression types: {_format_unique(primary_df['compression'])}\n"
            )
            f.write("\n")

            if not additional_df.empty:
                f.write("## Additional Operation Benchmarks\n\n")
                op_counts = additional_df["operation"].fillna("unknown").value_counts()
                for op_name, count in op_counts.items():
                    f.write(f"- {op_name}: {count} row(s)\n")
                f.write("\n")

            # Performance summary
            if tf_col in df.columns:
                f.write("## Performance Summary\n\n")
                valid_df = primary_df[primary_df[tf_col].notna()]

                if not valid_df.empty:
                    f.write(
                        f"- Fastest torchfits time: {valid_df[tf_col].min():.6f}s\n"
                    )
                    f.write(
                        f"- Slowest torchfits time: {valid_df[tf_col].max():.6f}s\n"
                    )
                    f.write(
                        f"- Average torchfits time: {valid_df[tf_col].mean():.6f}s\n"
                    )
                    f.write(
                        f"- Median torchfits time: {valid_df[tf_col].median():.6f}s\n"
                    )
                if "torchfits_median" in primary_df.columns:
                    valid_med = primary_df[primary_df["torchfits_median"].notna()]
                    if not valid_med.empty:
                        f.write(
                            f"- Median of torchfits median-times: {valid_med['torchfits_median'].median():.6f}s\n"
                        )
                f.write(
                    "- Note: for files <0.1 MB, rankings and MB/s are computed using median timings to reduce noise.\n"
                )
                f.write("\n")

            # File type breakdown
            f.write("## Performance by File Type\n\n")
            if tf_col in df.columns:
                type_stats = (
                    primary_df.groupby("file_type")[tf_col]
                    .agg(["count", "mean", "min", "max"])
                    .round(6)
                )
                f.write(type_stats.to_string())
                f.write("\n\n")

            # Ranking analysis
            if "torchfits_rank" in df.columns:
                f.write("## Ranking Analysis\n\n")
                rank_series = pd.to_numeric(
                    primary_df["torchfits_rank"], errors="coerce"
                )
                rank_series = rank_series[(rank_series >= 1) & (rank_series < 999)]
                if not rank_series.empty:
                    rank_counts = rank_series.value_counts().sort_index()
                    total_valid = rank_counts.sum()
                    rank3_plus = sum(rank_counts[rank_counts.index >= 3])

                    f.write(
                        f"- Times torchfits ranked #1: {rank_counts.get(1, 0)} ({rank_counts.get(1, 0) / total_valid * 100:.1f}%)\n"
                    )
                    f.write(
                        f"- Times torchfits ranked #2: {rank_counts.get(2, 0)} ({rank_counts.get(2, 0) / total_valid * 100:.1f}%)\n"
                    )
                    f.write(
                        f"- Times torchfits ranked #3+: {rank3_plus} ({rank3_plus / total_valid * 100:.1f}%)\n"
                    )
                    f.write(f"- Average ranking: {rank_series.mean():.2f}\n")
                else:
                    f.write("- No valid ranking data available.\n")
                f.write("\n")

            # Top regressions vs best
            if "speedup_vs_best" in df.columns and "best_method" in df.columns:
                f.write("## Top Regressions (torchfits vs best)\n\n")
                regressions = primary_df.copy()
                regressions["torchfits_rank"] = pd.to_numeric(
                    regressions["torchfits_rank"], errors="coerce"
                )
                regressions["speedup_vs_best"] = pd.to_numeric(
                    regressions["speedup_vs_best"], errors="coerce"
                )
                regressions = regressions[
                    regressions["speedup_vs_best"].notna()
                    & (regressions["torchfits_rank"] > 1)
                ].copy()
                if not regressions.empty:
                    regressions = regressions.sort_values("speedup_vs_best").head(10)
                    f.write("| File | Type | Size (MB) | Best | Speedup vs Best |\n")
                    f.write("|---|---|---|---|---|\n")
                    for _, r in regressions.iterrows():
                        f.write(
                            f"| {r['filename']} | {r['file_type']} | {r['size_mb']:.2f} | {r['best_method']} | {r['speedup_vs_best']:.2f}x |\n"
                        )
                    f.write("\n")
                else:
                    f.write("No regressions found.\n\n")

            # Top regressions vs best (torch-returning methods only)
            if (
                "speedup_vs_best_torch" in df.columns
                and "best_method_torch" in df.columns
            ):
                f.write("## Top Regressions (torch-returning methods)\n\n")
                regressions = primary_df.copy()
                regressions["torchfits_rank_torch"] = pd.to_numeric(
                    regressions["torchfits_rank_torch"], errors="coerce"
                )
                regressions["speedup_vs_best_torch"] = pd.to_numeric(
                    regressions["speedup_vs_best_torch"], errors="coerce"
                )
                regressions = regressions[
                    regressions["speedup_vs_best_torch"].notna()
                    & (regressions["torchfits_rank_torch"] > 1)
                ].copy()
                if not regressions.empty:
                    regressions = regressions.sort_values("speedup_vs_best_torch").head(
                        10
                    )
                    f.write(
                        "| File | Type | Size (MB) | Best (torch) | Speedup vs Best |\n"
                    )
                    f.write("|---|---|---|---|---|\n")
                    for _, r in regressions.iterrows():
                        f.write(
                            f"| {r['filename']} | {r['file_type']} | {r['size_mb']:.2f} | {r['best_method_torch']} | {r['speedup_vs_best_torch']:.2f}x |\n"
                        )
                    f.write("\n")
                else:
                    f.write("No regressions found.\n\n")

            # Top regressions vs best (numpy-returning methods only)
            if (
                "speedup_vs_best_numpy" in df.columns
                and "best_method_numpy" in df.columns
            ):
                f.write("## Top Regressions (numpy-returning methods)\n\n")
                regressions = primary_df.copy()
                regressions["torchfits_numpy_rank"] = pd.to_numeric(
                    regressions["torchfits_numpy_rank"], errors="coerce"
                )
                regressions["speedup_vs_best_numpy"] = pd.to_numeric(
                    regressions["speedup_vs_best_numpy"], errors="coerce"
                )
                regressions = regressions[
                    regressions["speedup_vs_best_numpy"].notna()
                    & (regressions["torchfits_numpy_rank"] > 1)
                ].copy()
                if not regressions.empty:
                    regressions = regressions.sort_values("speedup_vs_best_numpy").head(
                        10
                    )
                    f.write(
                        "| File | Type | Size (MB) | Best (numpy) | Speedup vs Best |\n"
                    )
                    f.write("|---|---|---|---|---|\n")
                    for _, r in regressions.iterrows():
                        f.write(
                            f"| {r['filename']} | {r['file_type']} | {r['size_mb']:.2f} | {r['best_method_numpy']} | {r['speedup_vs_best_numpy']:.2f}x |\n"
                        )
                    f.write("\n")
                else:
                    f.write("No regressions found.\n\n")

            # Memory analysis
            if "torchfits_memory" in df.columns:
                f.write("## Memory Analysis\n\n")
                mem_df = primary_df[
                    primary_df["torchfits_memory"].notna()
                    & primary_df["torchfits_peak_memory"].notna()
                ].copy()
                if not mem_df.empty:
                    mem_df["torchfits_memory"] = (
                        pd.to_numeric(mem_df["torchfits_memory"], errors="coerce")
                        .fillna(0)
                        .clip(lower=0)
                    )
                    mem_df["torchfits_peak_memory"] = (
                        pd.to_numeric(mem_df["torchfits_peak_memory"], errors="coerce")
                        .fillna(0)
                        .clip(lower=0)
                    )
                    peak_lt_steady = int(
                        (
                            mem_df["torchfits_peak_memory"] < mem_df["torchfits_memory"]
                        ).sum()
                    )
                    nonzero_peak_df = mem_df[mem_df["torchfits_peak_memory"] > 0]

                    f.write(
                        f"- Average steady RSS increase: {mem_df['torchfits_memory'].mean():.2f} MB\n"
                    )
                    f.write(
                        f"- Median steady RSS increase: {mem_df['torchfits_memory'].median():.2f} MB\n"
                    )
                    f.write(
                        f"- Average peak RSS increase: {mem_df['torchfits_peak_memory'].mean():.2f} MB\n"
                    )
                    f.write(
                        f"- Median peak RSS increase: {mem_df['torchfits_peak_memory'].median():.2f} MB\n"
                    )
                    f.write(
                        f"- P95 peak RSS increase: {mem_df['torchfits_peak_memory'].quantile(0.95):.2f} MB\n"
                    )
                    f.write(f"- Rows with peak < steady RSS: {peak_lt_steady}\n")
                    if not nonzero_peak_df.empty:
                        steady_over_peak = (
                            nonzero_peak_df["torchfits_memory"]
                            / nonzero_peak_df["torchfits_peak_memory"]
                        ).mean()
                        size_over_peak = (
                            nonzero_peak_df["size_mb"]
                            / nonzero_peak_df["torchfits_peak_memory"]
                        ).mean()
                        f.write(
                            f"- Mean steady/peak RSS ratio: {steady_over_peak:.2f}\n"
                        )
                        f.write(
                            f"- Mean file-size/peak RSS ratio: {size_over_peak:.2f}\n"
                        )

                    # Payload size is deterministic (tensor/ndarray nbytes) and is the most
                    # reliable per-call memory signal we can compute without forking a subprocess.
                    if "torchfits_payload_mb" in mem_df.columns:
                        payload = (
                            pd.to_numeric(
                                mem_df["torchfits_payload_mb"], errors="coerce"
                            )
                            .fillna(0)
                            .clip(lower=0)
                        )
                        f.write(
                            f"- Average payload size (TorchFits return): {payload.mean():.2f} MB\n"
                        )
                        f.write(
                            f"- Median payload size (TorchFits return): {payload.median():.2f} MB\n"
                        )
                else:
                    f.write("- No valid torchfits RSS memory data collected.\n")
                f.write("\n")

            # Top performers
            f.write("## Best Performing Files\n\n")
            if tf_col in df.columns:
                fastest = valid_df.nsmallest(10, tf_col)[
                    ["filename", tf_col, "size_mb", "file_type"]
                ]
                f.write("### Fastest Files:\n")
                for _, row in fastest.iterrows():
                    f.write(
                        f"- {row['filename']}: {row[tf_col]:.6f}s ({row['size_mb']:.2f} MB, {row['file_type']})\n"
                    )
                f.write("\n")

            # Issues and failures
            failed_files = primary_df[primary_df["torchfits_mean"].isna()]
            if not failed_files.empty:
                f.write("## Failed Tests\n\n")
                for _, row in failed_files.iterrows():
                    f.write(f"- {row['filename']}: Failed to benchmark\n")
                f.write("\n")

            # Comprehensive recommendations
            f.write("## Comprehensive Recommendations\n\n")
            f.write("Based on the exhaustive benchmark results:\n\n")

            if "torchfits_rank" in df.columns:
                rank_series = pd.to_numeric(
                    primary_df["torchfits_rank"], errors="coerce"
                )
                rank_series = rank_series[(rank_series >= 1) & (rank_series < 999)]
                if rank_series.empty:
                    f.write("⚠️ **insufficient ranking data for performance verdict**\n")
                else:
                    avg_rank = rank_series.mean()
                    if avg_rank <= 2:
                        f.write(
                            "✅ **torchfits shows excellent performance** across primary read scenarios\n"
                        )
                    elif avg_rank <= 3:
                        f.write(
                            "⚠️ **torchfits shows good performance** with opportunities for optimization\n"
                        )
                    else:
                        f.write(
                            "❌ **torchfits performance needs significant improvement**\n"
                        )

            # Specific findings
            f.write("\n### Specific Findings:\n\n")

            # Best file types
            if "torchfits_rank" in df.columns and "file_type" in df.columns:
                rank_by_type = primary_df.copy()
                rank_by_type["torchfits_rank"] = pd.to_numeric(
                    rank_by_type["torchfits_rank"], errors="coerce"
                )
                rank_by_type = rank_by_type[
                    rank_by_type["torchfits_rank"].notna()
                    & (rank_by_type["torchfits_rank"] < 999)
                ]
                if not rank_by_type.empty:
                    best_types = (
                        rank_by_type.groupby("file_type")["torchfits_rank"]
                        .mean()
                        .sort_values()
                    )
                    f.write(
                        f"- **Best file types**: {', '.join(best_types.head(3).index)}\n"
                    )
                    f.write(
                        f"- **Challenging file types**: {', '.join(best_types.tail(3).index)}\n"
                    )

            # Data type performance
            if "data_type" in df.columns and tf_col in df.columns:
                dtype_perf = (
                    primary_df.groupby("data_type")[tf_col].mean().sort_values()
                )
                f.write(
                    f"- **Fastest data types**: {', '.join(dtype_perf.head(3).index)}\n"
                )
                f.write(
                    f"- **Slowest data types**: {', '.join(dtype_perf.tail(3).index)}\n"
                )
            f.write("\n")
            f.write("## Files Generated\n\n")
            f.write(f"- Detailed results: `{self.csv_file.name}`\n")
            f.write("- Performance plots: `*.png` files\n")
            f.write(f"- This summary: `{self.summary_file.name}`\n")
            f.write("\n")
            f.write("## Next Steps\n\n")
            f.write(
                "1. Review detailed CSV results for specific performance bottlenecks\n"
            )
            f.write("2. Examine plots for visual performance patterns\n")
            f.write("3. Focus optimization efforts on underperforming file types\n")
            f.write(
                "4. Consider implementing specialized paths for best-performing scenarios\n"
            )

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"✓ Cleaned up temporary directory: {self.temp_dir}")

    def run_additional_benchmarks(self):
        """Run additional focused benchmarks for new features."""
        print("\n" + "=" * 60)
        print("RUNNING ADDITIONAL FEATURE BENCHMARKS")
        print("=" * 60)

        # Run C++ backend performance benchmark
        try:
            print("\n🎯 Running C++ Backend Performance Benchmark...")
            from benchmark_cpp_backend import CPPBackendBenchmark

            cpp_benchmark = CPPBackendBenchmark()
            cpp_benchmark.run_comprehensive_benchmark()
            print("✅ C++ backend benchmark completed")
        except Exception as e:
            print(f"⚠️  C++ backend benchmark failed: {e}")

        # Run GPU memory validation
        try:
            print("\n🚀 Running GPU Memory Validation...")
            from benchmark_gpu_memory import GPUMemoryBenchmark

            gpu_benchmark = GPUMemoryBenchmark()
            gpu_benchmark.run_comprehensive_benchmark()
            print("✅ GPU memory benchmark completed")
        except Exception as e:
            print(f"⚠️  GPU memory benchmark failed: {e}")

        # Run transform benchmarks
        try:
            print("\n🎨 Running Transform Benchmarks...")
            import subprocess

            result = subprocess.run(
                [sys.executable, "benchmark_transforms.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("✅ Transform benchmarks completed")
            else:
                print(f"⚠️  Transform benchmarks failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Transform benchmarks not available: {e}")

        # Run buffer benchmarks
        try:
            print("\n💾 Running Buffer Benchmarks...")
            import subprocess

            result = subprocess.run(
                [sys.executable, "benchmark_buffer.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("✅ Buffer benchmarks completed")
            else:
                print(f"⚠️  Buffer benchmarks failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Buffer benchmarks not available: {e}")

        # Run cache benchmarks
        try:
            print("\n🗄️  Running Cache Benchmarks...")
            import subprocess

            result = subprocess.run(
                [sys.executable, "benchmark_cache.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("✅ Cache benchmarks completed")
            else:
                print(f"⚠️  Cache benchmarks failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Cache benchmarks not available: {e}")

        # Run focused cold-target benchmarks
        try:
            print("\n🧊 Running Cold Target Benchmarks...")
            import subprocess

            result = subprocess.run(
                [sys.executable, "benchmark_cold_targets.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("✅ Cold target benchmarks completed")
                print(result.stdout)
            else:
                print(f"⚠️  Cold target benchmarks failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Cold target benchmarks not available: {e}")

    def run_phase3_benchmarks(self):
        """Run Phase 3 benchmarks (Scaled Data & Parallel I/O)."""
        print("\n🚀 Running Phase 3 Benchmarks (Scaled & Parallel)...")
        import subprocess

        # Scaled Data
        try:
            print("  Running Scaled Data Benchmark...")
            result = subprocess.run(
                [sys.executable, "benchmark_scaled.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("  ✅ Scaled Data Benchmark Passed")
                print(result.stdout)
            else:
                print(f"  ⚠️  Scaled Data Benchmark Failed: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️  Scaled Data Benchmark Error: {e}")

        # Parallel I/O
        try:
            print("  Running Parallel I/O Benchmark...")
            result = subprocess.run(
                [sys.executable, "benchmark_parallel.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("  ✅ Parallel I/O Benchmark Passed")
                print(result.stdout)
            else:
                print(f"  ⚠️  Parallel I/O Benchmark Failed: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️  Parallel I/O Benchmark Error: {e}")

        # MMap & Safety
        try:
            print("  Running MMap & Safety Benchmark...")
            result = subprocess.run(
                [sys.executable, "benchmark_mmap.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                print("  ✅ MMap & Safety Benchmark Passed")
                print(result.stdout)
            else:
                print(f"  ⚠️  MMap & Safety Benchmark Failed: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️  MMap & Safety Benchmark Error: {e}")

    def run_full_suite(
        self,
        *,
        filter_regex: str = "",
        core_only: bool = False,
        generate_plots: bool = True,
    ):
        """Run the complete exhaustive benchmark suite."""
        try:
            print("Starting exhaustive torchfits benchmark suite...", flush=True)
            print(f"Output directory: {self.output_dir}", flush=True)
            print(
                "Benchmark profile: "
                f"{self.profile} (cache={self.cache_capacity}, "
                f"handle_cache={self.handle_cache_capacity}, "
                f"hot_cache={self.hot_cache_capacity}, "
                f"payload_min_ratio={self.payload_min_ratio:.2f})",
                flush=True,
            )
            configure()

            # Create test files
            files = self.create_test_files()
            if filter_regex:
                import re

                rx = re.compile(filter_regex)
                files = {k: v for k, v in files.items() if rx.search(k)}

            # Run core benchmarks
            results = self.run_exhaustive_benchmarks(files)

            # Run focused benchmarks
            self.run_focused_benchmarks(files)

            # Optional: run extra feature/phase3 benches (these are informative, but slow).
            # Keep core I/O results stable by defaulting to core-only in CI-like workflows.
            if not core_only:
                self.run_additional_benchmarks()
                self.run_phase3_benchmarks()

            # Generate visualizations (optional; expensive during iteration)
            if generate_plots:
                self.generate_plots(results)

            # Generate summary report
            self.generate_summary_report(results)

            print("\n" + "=" * 80)
            print("EXHAUSTIVE BENCHMARK SUITE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved to: {self.output_dir}")
            print(f"- CSV data: {self.csv_file}")
            print(f"- Summary: {self.summary_file}")
            print(f"- Focused CSV: {self.focused_csv_file}")
            print(f"- Focused Summary: {self.focused_summary_file}")
            if generate_plots:
                print(f"- Plots: {self.output_dir}/*.png")

        finally:
            self.cleanup()


def main():
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # `pixi run <task> -- <args>` injects a literal `--` into argv. That is fine
    # for many CLIs, but argparse treats it as "end of options", which would make
    # all subsequent `--flags` look like positionals. Strip a leading separator.
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        del sys.argv[1]

    parser = argparse.ArgumentParser(description="Run exhaustive torchfits benchmarks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Enable memory mapping (default)",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory mapping",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Keep temporary files"
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Include table benchmarks (off by default)",
    )
    parser.add_argument(
        "--focused-only",
        action="store_true",
        help="Run only focused benchmarks (skip full suite)",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run core exhaustive + focused benchmarks only (skip extra feature/phase3 benches)",
    )
    parser.add_argument(
        "--profile",
        choices=["user", "lab"],
        default="user",
        help=(
            "Benchmark profile: 'user' uses library-like defaults "
            "(cache enabled), 'lab' is stricter cold-I/O style defaults"
        ),
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=None,
        help="torchfits in-memory cache entries (overrides profile default)",
    )
    parser.add_argument(
        "--handle-cache-capacity",
        type=int,
        default=None,
        help="torchfits open-handle cache entries (overrides profile default)",
    )
    parser.add_argument(
        "--hot-cache-capacity",
        type=int,
        default=None,
        help="torchfits cache entries for hot path benchmark (overrides profile default)",
    )
    parser.add_argument(
        "--payload-min-ratio",
        type=float,
        default=0.60,
        help=(
            "minimum observed/expected payload ratio before marking a method invalid "
            "(image-like read_full only)"
        ),
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Regex filter for benchmark case names (e.g. 'int8|compressed_rice_1')",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots (slow). Defaults to off when using --filter.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate plots",
    )
    args = parser.parse_args()

    use_mmap = True
    if args.no_mmap:
        use_mmap = False
    elif args.mmap:
        use_mmap = True

    profile_defaults = {
        "user": {
            "cache_capacity": 10,
            "handle_cache_capacity": 16,
            "hot_cache_capacity": 10,
        },
        "lab": {
            "cache_capacity": 0,
            "handle_cache_capacity": 16,
            "hot_cache_capacity": 10,
        },
    }
    defaults = profile_defaults[args.profile]
    cache_capacity = (
        args.cache_capacity
        if args.cache_capacity is not None
        else defaults["cache_capacity"]
    )
    handle_cache_capacity = (
        args.handle_cache_capacity
        if args.handle_cache_capacity is not None
        else defaults["handle_cache_capacity"]
    )
    hot_cache_capacity = (
        args.hot_cache_capacity
        if args.hot_cache_capacity is not None
        else defaults["hot_cache_capacity"]
    )

    suite = ExhaustiveBenchmarkSuite(
        output_dir=args.output_dir,
        use_mmap=use_mmap,
        include_tables=args.include_tables,
        cache_capacity=cache_capacity,
        hot_cache_capacity=hot_cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        profile=args.profile,
        payload_min_ratio=args.payload_min_ratio,
    )

    # Plot policy:
    # - If user explicitly sets a flag, honor it.
    # - Otherwise, skip plots when running filtered iterations.
    if args.no_plots:
        generate_plots = False
    elif args.plots:
        generate_plots = True
    elif args.filter:
        generate_plots = False
    else:
        generate_plots = True

    if args.no_cleanup:
        # Override cleanup method (monkey patch)
        suite.cleanup = lambda: print(f"Temporary files kept in: {suite.temp_dir}")

    if args.focused_only:
        print("Starting focused torchfits benchmark suite...")
        print(
            "Benchmark profile: "
            f"{suite.profile} (cache={suite.cache_capacity}, "
            f"handle_cache={suite.handle_cache_capacity}, "
            f"hot_cache={suite.hot_cache_capacity}, "
            f"payload_min_ratio={suite.payload_min_ratio:.2f})",
            flush=True,
        )
        files = suite.create_test_files()
        if args.filter:
            import re

            rx = re.compile(args.filter)
            files = {k: v for k, v in files.items() if rx.search(k)}
        suite.run_focused_benchmarks(files)
        suite.cleanup()
    else:
        suite.run_full_suite(
            filter_regex=args.filter,
            core_only=args.core_only,
            generate_plots=generate_plots,
        )


if __name__ == "__main__":
    main()
