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
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev, median
from typing import Dict, List, Optional

# Add benchmarks and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mpl_config import configure

import fitsio
import numpy as np
import pandas as pd
import psutil
import torch
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
        cache_capacity: int = 0,
        hot_cache_capacity: int = 10,
    ):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="torchfits_exhaustive_"))
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.csv_file = self.output_dir / "exhaustive_results.csv"
        self.summary_file = self.output_dir / "exhaustive_summary.md"
        self.focused_csv_file = self.output_dir / "focused_results.csv"
        self.focused_summary_file = self.output_dir / "focused_summary.md"
        self.use_mmap = use_mmap
        self.include_tables = include_tables
        self.cache_capacity = cache_capacity
        self.hot_cache_capacity = hot_cache_capacity

        # Test configurations
        self.data_types = {
            "int8": (np.int8, "BYTE_IMG"),
            "int16": (np.int16, "SHORT_IMG"),
            "int32": (np.int32, "LONG_IMG"),
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

        for size_name in ["small", "medium"]:
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
            "torchfits_hot_mean",
            "torchfits_hot_std",
            "torchfits_hot_median",
            "torchfits_hot_mb_s",
            "torchfits_mmap_mean",
            "torchfits_mmap_std",
            "torchfits_mmap_median",
            "torchfits_mmap_mb_s",
            "torchfits_mmap_memory",
            "torchfits_mmap_peak_memory",
            "astropy_mean",
            "astropy_std",
            "astropy_median",
            "astropy_mb_s",
            "astropy_memory",
            "astropy_peak_memory",
            "fitsio_mean",
            "fitsio_std",
            "fitsio_median",
            "fitsio_mb_s",
            "fitsio_memory",
            "fitsio_peak_memory",
            "astropy_torch_mean",
            "astropy_torch_std",
            "astropy_torch_median",
            "astropy_torch_mb_s",
            "astropy_torch_memory",
            "astropy_torch_peak_memory",
            "fitsio_torch_mean",
            "fitsio_torch_std",
            "fitsio_torch_median",
            "fitsio_torch_mb_s",
            "fitsio_torch_memory",
            "fitsio_torch_peak_memory",
            "torchfits_cpp_open_once_mean",
            "torchfits_cpp_open_once_std",
            "torchfits_cpp_open_once_median",
            "torchfits_cpp_open_once_mb_s",
            "best_method",
            "torchfits_rank",
            "speedup_vs_best",
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
            "compressed_rice_1",
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
            hdu_num = 1 if file_type in {"compressed", "table", "mef", "multi_mef"} else 0

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
                    mmap=self.use_mmap,
                    scale_on_device=True,
                    cache_capacity=self.cache_capacity,
                )

            def tf_scaled_cpu():
                return torchfits.cpp.read_full_scaled_cpu(str(path), hdu_num, self.use_mmap)

            def tf_raw_with_scale_direct():
                data, scaled, bscale, bzero = torchfits.cpp.read_full_raw_with_scale(
                    str(path), hdu_num, self.use_mmap
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
                    mmap=self.use_mmap,
                    raw_scale=True,
                    cache_capacity=self.cache_capacity,
                )

            def tf_raw_scale_cpu():
                data = torchfits.read(
                    str(path),
                    hdu=hdu_num,
                    mmap=self.use_mmap,
                    raw_scale=True,
                    cache_capacity=self.cache_capacity,
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

            for method_name, method_func in methods.items():
                method_result = self._time_method(
                    method_func, method_name, runs=runs, use_median=use_median
                )
                if not method_result:
                    print(f"{method_name:22s}: FAILED")
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
                print(
                    f"{method_name:22s}: {time_value:.6f}s ± {method_result['std']:.6f}s"
                )
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
                        f.write(
                            f" raw+scale_cpu={tf_raw_scale.iloc[0]:.6f}s"
                        )
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

        # Compression timings are noisy; use more samples for stability.
        if file_type == "compressed":
            runs = max(runs, 20)

        # Define benchmark methods
        methods = {}
        diagnostic_methods = {}

        # Always test torchfits
        methods["torchfits"] = lambda: torchfits.read(
            str(filepath),
            hdu=hdu_num,
            mmap=self.use_mmap,
            cache_capacity=self.cache_capacity,
        )

        # Test torchfits mmap explicitly for tables
        if file_type == "table":
            methods["torchfits_mmap"] = lambda: torchfits.read(
                str(filepath),
                hdu=hdu_num,
                mmap=True,
                cache_capacity=self.cache_capacity,
            )

        # Test astropy if available
        methods["astropy"] = lambda: self._astropy_read(filepath, hdu_num)
        methods["astropy_torch"] = lambda: self._astropy_to_torch(filepath, hdu_num)

        # Test fitsio if available
        methods["fitsio"] = lambda: fitsio.read(
            str(filepath), ext=hdu_num
        )  # fitsio uses mmap by default usually? Or we can't control it easily here.
        methods["fitsio_torch"] = lambda: self._fitsio_to_torch(filepath, hdu_num)

        # Run benchmarks (shuffle order to reduce cache bias)
        method_results = {}
        method_items = list(methods.items())
        random.shuffle(method_items)
        for method_name, method_func in method_items:
            method_result = self._time_method(
                method_func, method_name, runs=runs, use_median=use_median
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

                stat_label = "median" if use_median else "mean"
                print(
                    f"{method_name:15s}: {time_value:.6f}s ± {method_result['std']:.6f}s ({stat_label})  "
                    f"mem: {method_result['memory']:.1f}MB  peak: {method_result['peak_memory']:.1f}MB"
                )
            else:
                result[f"{method_name}_mean"] = None
                result[f"{method_name}_std"] = None
                result[f"{method_name}_median"] = None
                result[f"{method_name}_mb_s"] = None
                result[f"{method_name}_memory"] = None
                result[f"{method_name}_peak_memory"] = None
                print(f"{method_name:15s}: FAILED")

        # Run diagnostic methods (not included in ranking)
        # Open after the main loop so cache clears don't invalidate the handle.
        try:
            file_handle = torchfits.cpp.open_fits_file(str(filepath), "r")
        except Exception:
            file_handle = None
        if file_handle is not None:
            diagnostic_methods["torchfits_cpp_open_once"] = lambda: torchfits.cpp.read_full(
                file_handle, hdu_num, self.use_mmap
            )
        diagnostic_methods["torchfits_hot"] = lambda: torchfits.read(
            str(filepath),
            hdu=hdu_num,
            mmap=self.use_mmap,
            cache_capacity=self.hot_cache_capacity,
        )
        for method_name, method_func in diagnostic_methods.items():
            method_result = self._time_method(
                method_func, method_name, runs=runs, use_median=use_median
            )
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
                stat_label = "median" if use_median else "mean"
                print(f"{method_name:15s}: {time_value:.6f}s ± {method_result['std']:.6f}s ({stat_label})")
            else:
                result[f"{method_name}_mean"] = None
                result[f"{method_name}_std"] = None
                result[f"{method_name}_median"] = None
                result[f"{method_name}_mb_s"] = None
                print(f"{method_name:15s}: FAILED")

        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass

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

                print(
                    f"\nBest method: {best_method} ({valid_methods[best_method]:.6f}s)"
                )
                print(f"torchfits rank: {torchfits_rank}/{len(valid_methods)}")
                if best_method != "torchfits":
                    print(
                        f"torchfits vs best: {tf_time / valid_methods[best_method]:.2f}x"
                    )
            else:
                result["speedup_vs_best"] = None
        else:
            result["best_method"] = "none"
            result["torchfits_rank"] = 999
            result["speedup_vs_best"] = None

        return result

    def _time_method(
        self, method_func, method_name: str, runs: int = 3, use_median: bool = False
    ) -> Optional[Dict]:
        """Time a method with memory monitoring."""
        times = []
        memory_usage = []
        peak_memory_usage = []

        # Import psutil for memory measurement
        try:
            import psutil

            process = psutil.Process()
            memory_available = True
        except ImportError:
            memory_available = False

        for i in range(runs):
            try:
                gc.collect()
                for _ in range(3):  # Extra cleanup
                    gc.collect()

                # Clear torchfits cache if applicable
                if (
                    "torchfits" in method_name
                    and "open_once" not in method_name
                    and "hot" not in method_name
                ):
                    torchfits.clear_file_cache()

                # Get initial memory usage
                if memory_available:
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

                # Time the operation
                start_time = time.perf_counter()
                data = method_func()
                elapsed = time.perf_counter() - start_time

                # Get final memory usage
                if memory_available:
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    peak_memory_increase = max(0, final_memory - initial_memory)
                else:
                    peak_memory_increase = 0

                # Calculate data memory usage
                if hasattr(data, "element_size") and hasattr(data, "numel"):
                    # PyTorch tensor
                    data_size_mb = (data.element_size() * data.numel()) / 1024 / 1024
                elif hasattr(data, "nbytes"):
                    # NumPy array
                    data_size_mb = data.nbytes / 1024 / 1024
                else:
                    data_size_mb = peak_memory_increase

                times.append(elapsed)
                memory_usage.append(data_size_mb)
                peak_memory_usage.append(peak_memory_increase)

                del data
                gc.collect()

            except Exception as e:
                print(f"Error in {method_name}: {e}")
                return None

        if times:
            return {
                "mean": mean(times),
                "median": median(times),
                "std": stdev(times) if len(times) > 1 else 0,
                "memory": mean(memory_usage),
                "peak_memory": mean(peak_memory_usage),
            }
        return None

    def _benchmark_cutouts(self, files: Dict[str, Path]) -> List[Dict]:
        """Benchmark random access cutouts on large files."""
        print("\n" + "=" * 100)
        print("CUTOUT BENCHMARK")
        print("=" * 100)

        results = []

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

        if not target_file:
            print("No suitable file found for cutout benchmark.")
            return []

        print(f"Benchmarking cutouts on {target_name}...")

        # Define cutout parameters
        cutout_size = (100, 100)
        n_iter = 50

        # Random position
        x1, y1 = 100, 100
        x2 = x1 + cutout_size[1]
        y2 = y1 + cutout_size[0]

        # Target a middle extension
        hdu_idx = 5 if "multi_mef" in target_name else 1

        # Define methods
        methods = {}

        methods["torchfits"] = lambda: torchfits.read_subset(
            str(target_file), hdu_idx, x1, y1, x2, y2
        )

        methods["astropy"] = lambda: self._astropy_cutout(
            target_file, hdu_idx, x1, y1, x2, y2
        )

        methods["fitsio"] = lambda: self._fitsio_cutout(
            target_file, hdu_idx, x1, y1, x2, y2
        )

        # Run benchmark
        row = {
            "filename": target_name,
            "operation": "cutout_100x100",
            "file_type": self._get_file_type(target_name),
            "size_mb": target_file.stat().st_size / 1024 / 1024,
            "data_type": "mixed",
            "dimensions": "2d",
            "compression": "uncompressed",
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
                print(
                    f"{name:15s}: {res['mean'] * 1e6:.2f}us ± {res['std'] * 1e6:.2f}us"
                )
            else:
                row[f"{name}_mean"] = None

        results.append(row)
        return results

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

    def _astropy_read(self, filepath: Path, hdu_num: int):
        """Pure astropy read - handles both images and tables."""
        try:
            with self._astropy_open(filepath, self.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        # Table: convert to dict for fair comparison
                        data = hdu.data
                        return {col: data[col] for col in data.names}
                    else:
                        # Image: return numpy array
                        return hdu.data.copy()
                return None
        except Exception:
            if self.use_mmap:
                with self._astropy_open(filepath, False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            data = hdu.data
                            return {col: data[col] for col in data.names}
                        return hdu.data.copy()
                    return None
            raise

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
                                col_data = col_data.astype(col_data.dtype.newbyteorder("="))
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
                                    result[col] = torch.from_numpy(col_data.astype(bool))
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
                                    col_data = col_data.astype(col_data.dtype.newbyteorder("="))
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
                                        result[col] = torch.from_numpy(col_data.astype(bool))
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
        """Plot memory usage analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Memory Usage Analysis", fontsize=16)

        # Memory vs file size
        valid_df = df[df["torchfits_memory"].notna()]
        if not valid_df.empty:
            axes[0].scatter(
                valid_df["size_mb"], valid_df["torchfits_memory"], alpha=0.6
            )
            axes[0].plot(
                [0, valid_df["size_mb"].max()],
                [0, valid_df["size_mb"].max()],
                "r--",
                alpha=0.5,
            )
            axes[0].set_xlabel("File Size (MB)")
            axes[0].set_ylabel("Memory Usage (MB)")
            axes[0].set_title("Memory Usage vs File Size")

        # Peak memory by data type
        if not valid_df.empty:
            sns.boxplot(
                data=valid_df, x="data_type", y="torchfits_peak_memory", ax=axes[1]
            )
            axes[1].set_title("Peak Memory by Data Type")
            axes[1].set_ylabel("Peak Memory (MB)")
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
        tf_col = "torchfits_median" if "torchfits_median" in df.columns else "torchfits_mean"
        fi_col = "fitsio_median" if "fitsio_median" in df.columns else "fitsio_mean"
        astro_col = "astropy_median" if "astropy_median" in df.columns else "astropy_mean"
        tf_hot_col = (
            "torchfits_hot_median"
            if "torchfits_hot_median" in df.columns
            else "torchfits_hot_mean"
        )

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
            f.write("\n")

            # Test coverage summary
            f.write("## Test Coverage Summary\n\n")
            f.write(f"- Total test files: {len(results)}\n")
            f.write(
                f"- File types tested: {', '.join(sorted(df['file_type'].unique()))}\n"
            )
            f.write(
                f"- Data types tested: {', '.join(sorted(df['data_type'].unique()))}\n"
            )
            f.write(f"- MMap enabled: {self.use_mmap}\n")
            f.write(f"- Tables included: {self.include_tables}\n\n")
            f.write(f"- Cache capacity (cold): {self.cache_capacity}\n")
            f.write(f"- Cache capacity (hot): {self.hot_cache_capacity}\n\n")

            # Performance Summary Table
            f.write("## Performance Summary (Ratio vs TorchFits)\n\n")
            f.write(
                "| File | Type | Size (MB) | TorchFits (s) | TorchFits Hot (s) | Astropy (x) | Fitsio (x) | Fitsio vs Hot (x) | Best |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|\n")

            for r in results:
                name = r["filename"]
                ftype = r["file_type"]
                size = f"{r['size_mb']:.2f}"

                tf_time = r.get(tf_col)
                tf_hot = r.get(tf_hot_col)
                if tf_time is None:
                    tf_str = "FAIL"
                    tf_hot_str = "-"
                    astro_ratio = "-"
                    fitsio_ratio = "-"
                    fitsio_hot_ratio = "-"
                else:
                    tf_str = f"{tf_time:.4f}"
                    tf_hot_str = f"{tf_hot:.4f}" if tf_hot else "-"

                    astro_time = r.get(astro_col)
                    if astro_time:
                        astro_ratio = f"{astro_time / tf_time:.2f}x"
                    else:
                        astro_ratio = "-"

                    fitsio_time = r.get(fi_col)
                    if fitsio_time:
                        fitsio_ratio = f"{fitsio_time / tf_time:.2f}x"
                        if tf_hot:
                            fitsio_hot_ratio = f"{fitsio_time / tf_hot:.2f}x"
                        else:
                            fitsio_hot_ratio = "-"
                    else:
                        fitsio_ratio = "-"
                        fitsio_hot_ratio = "-"

                best = r.get("best_method", "-")

                f.write(
                    f"| {name} | {ftype} | {size} | {tf_str} | {tf_hot_str} | {astro_ratio} | {fitsio_ratio} | {fitsio_hot_ratio} | {best} |\n"
                )

            f.write("\n")
            f.write(
                f"- Dimensions tested: {', '.join(sorted(df['dimensions'].unique()))}\n"
            )
            f.write(
                f"- Compression types: {', '.join(sorted(df['compression'].unique()))}\n"
            )
            f.write("\n")

            # Performance summary
            if tf_col in df.columns:
                f.write("## Performance Summary\n\n")
                valid_df = df[df[tf_col].notna()]

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
                if "torchfits_median" in df.columns:
                    valid_med = df[df["torchfits_median"].notna()]
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
                    df.groupby("file_type")[tf_col]
                    .agg(["count", "mean", "min", "max"])
                    .round(6)
                )
                f.write(type_stats.to_string())
                f.write("\n\n")

            # Ranking analysis
            if "torchfits_rank" in df.columns:
                f.write("## Ranking Analysis\n\n")
                rank_counts = df["torchfits_rank"].value_counts().sort_index()
                total_valid = rank_counts.sum()

                f.write(
                    f"- Times torchfits ranked #1: {rank_counts.get(1, 0)} ({rank_counts.get(1, 0) / total_valid * 100:.1f}%)\n"
                )
                f.write(
                    f"- Times torchfits ranked #2: {rank_counts.get(2, 0)} ({rank_counts.get(2, 0) / total_valid * 100:.1f}%)\n"
                )
                f.write(
                    f"- Times torchfits ranked #3+: {sum(rank_counts[rank_counts.index >= 3])} ({sum(rank_counts[rank_counts.index >= 3]) / total_valid * 100:.1f}%)\n"
                )
                f.write(f"- Average ranking: {df['torchfits_rank'].mean():.2f}\n")
                f.write("\n")

            # Top regressions vs best
            if "speedup_vs_best" in df.columns and "best_method" in df.columns:
                f.write("## Top Regressions (torchfits vs best)\n\n")
                regressions = df[
                    df["speedup_vs_best"].notna() & (df["torchfits_rank"] > 1)
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

            # Memory analysis
            if "torchfits_memory" in df.columns:
                f.write("## Memory Analysis\n\n")
                mem_df = df[df["torchfits_memory"].notna()]
                if not mem_df.empty:
                    f.write(
                        f"- Average memory usage: {mem_df['torchfits_memory'].mean():.1f} MB\n"
                    )
                    f.write(
                        f"- Peak memory usage: {mem_df['torchfits_peak_memory'].mean():.1f} MB\n"
                    )
                    f.write(
                        f"- Memory efficiency (data/peak): {(mem_df['torchfits_memory'] / mem_df['torchfits_peak_memory']).mean():.2f}\n"
                    )
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
            failed_files = df[df["torchfits_mean"].isna()]
            if not failed_files.empty:
                f.write("## Failed Tests\n\n")
                for _, row in failed_files.iterrows():
                    f.write(f"- {row['filename']}: Failed to benchmark\n")
                f.write("\n")

            # Comprehensive recommendations
            f.write("## Comprehensive Recommendations\n\n")
            f.write("Based on the exhaustive benchmark results:\n\n")

            if "torchfits_rank" in df.columns:
                avg_rank = df["torchfits_rank"].mean()
                if avg_rank <= 2:
                    f.write(
                        "✅ **torchfits shows excellent performance** across all test scenarios\n"
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
                best_types = (
                    df.groupby("file_type")["torchfits_rank"].mean().sort_values()
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
                    df.groupby("data_type")[tf_col].mean().sort_values()
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

    def run_full_suite(self):
        """Run the complete exhaustive benchmark suite."""
        try:
            print("Starting exhaustive torchfits benchmark suite...", flush=True)
            print(f"Output directory: {self.output_dir}", flush=True)
            configure()

            # Create test files
            files = self.create_test_files()

            # Run core benchmarks
            results = self.run_exhaustive_benchmarks(files)

            # Run focused benchmarks
            self.run_focused_benchmarks(files)

            # Run additional feature benchmarks
            self.run_additional_benchmarks()

            # Run Phase 3 benchmarks
            self.run_phase3_benchmarks()

            # Generate visualizations
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
        "--cache-capacity",
        type=int,
        default=0,
        help="torchfits in-memory cache entries (0 disables cache for fair I/O timing)",
    )
    parser.add_argument(
        "--hot-cache-capacity",
        type=int,
        default=10,
        help="torchfits cache entries for hot path benchmark (torchfits_hot)",
    )
    args = parser.parse_args()

    use_mmap = True
    if args.no_mmap:
        use_mmap = False
    elif args.mmap:
        use_mmap = True

    suite = ExhaustiveBenchmarkSuite(
        output_dir=args.output_dir,
        use_mmap=use_mmap,
        include_tables=args.include_tables,
        cache_capacity=args.cache_capacity,
        hot_cache_capacity=args.hot_cache_capacity,
    )

    if args.no_cleanup:
        # Override cleanup method (monkey patch)
        suite.cleanup = lambda: print(f"Temporary files kept in: {suite.temp_dir}")

    if args.focused_only:
        print("Starting focused torchfits benchmark suite...")
        files = suite.create_test_files()
        suite.run_focused_benchmarks(files)
        suite.cleanup()
    else:
        suite.run_full_suite()


if __name__ == "__main__":
    main()
