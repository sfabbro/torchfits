import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'benchmarks' package
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

import argparse
import gc
import os
import re
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from contextlib import contextmanager
from typing import Any

import fitsio
import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits

from benchmarks.bench_contract import (
    RESULT_COLUMNS,
    annotate_rankings,
    write_csv,
    write_json,
)  # noqa: E402


SMART_METHODS = [
    ("torchfits", "torchfits", "torchfits"),
    ("astropy_torch", "astropy", "astropy_torch"),
    ("fitsio_torch", "fitsio", "fitsio_torch"),
]

SPECIALIZED_METHODS = [
    ("torchfits_specialized", "torchfits", "torchfits_specialized"),
    ("astropy", "astropy", "astropy"),
    ("fitsio", "fitsio", "fitsio"),
]


class FITSBenchmarkSuite:
    """FITS-only version of the pre-extraction exhaustive fixture suite.

    This preserves the historical 84 image workflows without retaining the
    WCS/sphere benchmark implementation that moved out of torchfits.
    """

    EXPECTED_FILE_COUNT = 87
    EXPECTED_WORKFLOW_COUNT = 91

    def __init__(
        self,
        *,
        output_dir: Path,
        use_mmap: bool,
        include_tables: bool = False,
        include_wcs: bool = False,
        include_sphere: bool = False,
        profile: str = "user",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.use_mmap = bool(use_mmap)
        self.profile = profile
        self.temp_dir = self.output_dir / "fixtures"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_types = {
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float64": np.float64,
        }
        self.size_categories = {
            "tiny": {"1d": 1000, "2d": (64, 64), "3d": (5, 32, 32)},
            "small": {"1d": 10000, "2d": (256, 256), "3d": (10, 128, 128)},
            "medium": {"1d": 100000, "2d": (1024, 1024), "3d": (25, 256, 256)},
            "large": {"1d": 1000000, "2d": (2048, 2048), "3d": (50, 512, 512)},
        }

    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_files(self) -> dict[str, Path]:
        files: dict[str, Path] = {}
        rng = np.random.default_rng(20260318)

        for size_name, dimensions in self.size_categories.items():
            for dtype_name, dtype in self.data_types.items():
                for dim_name, shape in dimensions.items():
                    if size_name == "large" and dim_name == "3d":
                        continue
                    name = f"{size_name}_{dtype_name}_{dim_name}"
                    path = self.temp_dir / f"{name}.fits"
                    astropy_fits.PrimaryHDU(
                        self._generate_data(rng, shape, dtype)
                    ).writeto(path, overwrite=True)
                    files[name] = path

        for size_name in ("small", "medium"):
            shape = self.size_categories[size_name]["2d"]
            hdus = [astropy_fits.PrimaryHDU()]
            for dtype_name, dtype in list(self.data_types.items())[:3]:
                hdus.append(
                    astropy_fits.ImageHDU(
                        self._generate_data(rng, shape, dtype),
                        name=f"EXT_{dtype_name.upper()}",
                    )
                )
            path = self.temp_dir / f"mef_{size_name}.fits"
            astropy_fits.HDUList(hdus).writeto(path, overwrite=True)
            files[f"mef_{size_name}"] = path

        hdus = [astropy_fits.PrimaryHDU()]
        for index in range(10):
            dtype_name, dtype = list(self.data_types.items())[
                index % len(self.data_types)
            ]
            hdus.append(
                astropy_fits.ImageHDU(
                    self._generate_data(rng, (256, 256), dtype),
                    name=f"EXT_{index:02d}_{dtype_name.upper()}",
                )
            )
        multi_path = self.temp_dir / "multi_mef_10ext.fits"
        astropy_fits.HDUList(hdus).writeto(multi_path, overwrite=True)
        files["multi_mef_10ext"] = multi_path

        for size_name in ("small", "medium", "large"):
            shape = self.size_categories[size_name]["2d"]
            data = (rng.normal(size=shape).astype(np.float32) * 1000 + 32768).astype(
                np.int16
            )
            hdu = astropy_fits.PrimaryHDU(data)
            hdu.header["BSCALE"] = 0.1
            hdu.header["BZERO"] = 32768
            path = self.temp_dir / f"scaled_{size_name}.fits"
            hdu.writeto(path, overwrite=True)
            files[f"scaled_{size_name}"] = path

        for size_name in ("small", "medium", "large"):
            shape = self.size_categories[size_name]["2d"]
            for dtype_name, np_dtype in (("uint16", np.uint16), ("uint32", np.uint32)):
                name = f"{size_name}_{dtype_name}_2d"
                path = self.temp_dir / f"{name}.fits"
                high = min(np.iinfo(np_dtype).max, 100_000)
                data = rng.integers(0, high, size=shape, dtype=np_dtype)
                astropy_fits.PrimaryHDU(data).writeto(path, overwrite=True)
                files[name] = path

        compressed = self._generate_data(rng, (1024, 1024), np.float32)
        for compression in ("RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"):
            name = f"compressed_{compression.lower()}"
            path = self.temp_dir / f"{name}.fits"
            astropy_fits.HDUList(
                [
                    astropy_fits.PrimaryHDU(),
                    astropy_fits.CompImageHDU(
                        compressed,
                        compression_type=compression,
                    ),
                ]
            ).writeto(path, overwrite=True)
            files[name] = path

        for index in range(5):
            name = f"timeseries_frame_{index:03d}"
            path = self.temp_dir / f"{name}.fits"
            astropy_fits.PrimaryHDU(
                self._generate_data(rng, (256, 256), np.float32) + index * 100
            ).writeto(path, overwrite=True)
            files[name] = path
        if len(files) != self.EXPECTED_FILE_COUNT:
            raise RuntimeError(
                "FITS benchmark fixture contract changed: "
                f"expected {self.EXPECTED_FILE_COUNT} files, created {len(files)}"
            )
        return files

    @staticmethod
    def _generate_data(
        rng: np.random.Generator,
        shape: int | tuple[int, ...],
        dtype: type[np.generic],
    ) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            bounds = {
                np.dtype(np.int8): (-100, 100),
                np.dtype(np.int16): (-1000, 1000),
                np.dtype(np.int32): (-10000, 10000),
                np.dtype(np.int64): (-100000, 100000),
            }
            low, high = bounds[np.dtype(dtype)]
            return rng.integers(low, high, size=shape, dtype=dtype)
        return rng.normal(size=shape).astype(dtype)

    def _get_file_type(self, name: str) -> str:
        lowered = name.lower()
        if "compressed" in lowered:
            return "compressed"
        if "multi_mef" in lowered:
            return "multi_mef"
        if "mef" in lowered:
            return "mef"
        if "scaled" in lowered:
            return "scaled"
        if "uint" in lowered:
            return "uint"
        return "image"

    def _get_compression_type(self, name: str) -> str:
        lowered = name.lower()
        if "compressed_" in lowered:
            return lowered.split("compressed_", 1)[1]
        return "none"

    @staticmethod
    def _ensure_native_endian_numpy(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.byteorder not in ("=", "|"):
            return arr.astype(arr.dtype.newbyteorder("="))
        return arr

    @staticmethod
    def _table_to_numpy_dict(data: Any) -> dict[str, np.ndarray]:
        return {name: np.asarray(data[name]) for name in data.names}

    def _astropy_read(self, path: Path, hdu_num: int) -> np.ndarray:
        with astropy_fits.open(path, memmap=self.use_mmap) as hdul:
            return self._ensure_native_endian_numpy(
                np.array(hdul[hdu_num].data, copy=True)
            )

    def _astropy_to_torch(self, path: Path, hdu_num: int) -> torch.Tensor:
        return torch.from_numpy(self._astropy_read(path, hdu_num))

    @staticmethod
    def _mb_per_second(path: Path, seconds: float | None) -> float | None:
        if seconds is None or seconds <= 0:
            return None
        return path.stat().st_size / (1024.0 * 1024.0) / seconds

    def run_exhaustive_benchmarks(self, files: dict[str, Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        runs = 3 if self.profile == "user" else 7
        warmup = 1 if self.profile == "user" else 2
        torchfits_mmap: bool | str = (
            "auto" if self.profile == "user" and self.use_mmap else self.use_mmap
        )

        for name, path in sorted(files.items()):
            file_type = self._get_file_type(name)
            hdu = _hdu_for_file_type(file_type)
            print(f"[fits] case={name} file_type={file_type} runs={runs}", flush=True)

            methods = {
                "torchfits": lambda p=path, h=hdu: torchfits.read(
                    str(p), hdu=h, mmap=torchfits_mmap
                ),
                "astropy_torch": lambda p=path, h=hdu: self._astropy_to_torch(p, h),
                "fitsio_torch": lambda p=path, h=hdu: torch.from_numpy(
                    self._ensure_native_endian_numpy(fitsio.read(str(p), ext=h))
                ),
                "torchfits_specialized": lambda p=path, h=hdu: torchfits.read_tensor(
                    str(p), hdu=h, mmap=self.use_mmap
                ),
                "astropy": lambda p=path, h=hdu: self._astropy_read(p, h),
                "fitsio": lambda p=path, h=hdu: self._ensure_native_endian_numpy(
                    fitsio.read(str(p), ext=h)
                ),
            }

            row: dict[str, Any] = {
                "filename": name,
                "operation": "read_full",
                "file_type": file_type,
                "data_type": "image",
                "dimensions": "2d",
                "compression": self._get_compression_type(name),
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
            }
            for method_name, fn in methods.items():
                median_s, _err = _time_median(fn, runs=runs, warmup=warmup)
                row[f"{method_name}_median"] = median_s
                row[f"{method_name}_mb_s"] = self._mb_per_second(path, median_s)
            rows.append(row)

        rows.extend(
            self._benchmark_cutout_rows(
                files,
                runs=runs,
                warmup=warmup,
            )
        )
        rows.extend(
            self._benchmark_repeated_cutout_rows(
                files,
                runs=runs,
                warmup=warmup,
            )
        )
        random_extension_row = self._benchmark_random_extensions(
            files,
            runs=runs,
            warmup=warmup,
        )
        if random_extension_row is not None:
            rows.append(random_extension_row)
        return rows

    def _benchmark_cutout_rows(
        self,
        files: dict[str, Path],
        *,
        runs: int,
        warmup: int,
    ) -> list[dict[str, Any]]:
        targets = [
            ("multi_mef_10ext", 5, "uncompressed"),
            ("compressed_rice_1", 1, "compressed"),
        ]
        rows: list[dict[str, Any]] = []
        x1, y1, x2, y2 = 100, 100, 200, 200
        for name, hdu, compression in targets:
            path = files.get(name)
            if path is None:
                continue

            def astropy_cutout(p=path, h=hdu):
                with astropy_fits.open(p, memmap=self.use_mmap) as hdul:
                    return self._ensure_native_endian_numpy(
                        np.array(hdul[h].section[y1:y2, x1:x2], copy=True)
                    )

            def fitsio_cutout(p=path, h=hdu):
                with fitsio.FITS(str(p)) as handle:
                    return self._ensure_native_endian_numpy(handle[h][y1:y2, x1:x2])

            methods = {
                "torchfits": lambda p=path, h=hdu: torchfits.read_subset(
                    str(p), h, x1, y1, x2, y2
                ),
                "astropy_torch": lambda: torch.from_numpy(astropy_cutout()),
                "fitsio_torch": lambda: torch.from_numpy(fitsio_cutout()),
                "torchfits_specialized": lambda p=path, h=hdu: torchfits.read_subset(
                    str(p), h, x1, y1, x2, y2
                ),
                "astropy": astropy_cutout,
                "fitsio": fitsio_cutout,
            }
            row: dict[str, Any] = {
                "filename": name,
                "operation": "cutout_100x100",
                "file_type": self._get_file_type(name),
                "data_type": "mixed",
                "dimensions": "2d",
                "compression": compression,
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
            }
            for method_name, fn in methods.items():
                median_s, _err = _time_median(fn, runs=runs, warmup=warmup)
                row[f"{method_name}_median"] = median_s
                row[f"{method_name}_mb_s"] = self._mb_per_second(path, median_s)
            rows.append(row)
        return rows

    def _benchmark_repeated_cutout_rows(
        self,
        files: dict[str, Path],
        *,
        runs: int,
        warmup: int,
    ) -> list[dict[str, Any]]:
        path = files.get("medium_float32_2d")
        if path is None:
            for k in sorted(files.keys()):
                if "2d" in k:
                    path = files[k]
                    break
        if path is None:
            return []

        file_type = self._get_file_type(path.stem)
        hdu = _hdu_for_file_type(file_type)

        with fitsio.FITS(str(path)) as f:
            header = f[hdu].read_header()
            naxis1 = header.get("NAXIS1", 1024)
            naxis2 = header.get("NAXIS2", 1024)

        cutout_size = min(100, naxis1 // 2, naxis2 // 2)
        if cutout_size < 2:
            cutout_size = 2

        coords_rng = np.random.default_rng(42)
        cutouts_coords = []
        for _ in range(50):
            x1 = int(coords_rng.integers(0, max(1, naxis1 - cutout_size)))
            y1 = int(coords_rng.integers(0, max(1, naxis2 - cutout_size)))
            cutouts_coords.append((x1, y1, x1 + cutout_size, y1 + cutout_size))

        print(
            f"[fits] case=repeated_cutouts_50x_{cutout_size}x{cutout_size} runs={runs}",
            flush=True,
        )

        def astropy_repeated_cutout():
            with astropy_fits.open(path, memmap=self.use_mmap) as hdul:
                results = []
                for x1, y1, x2, y2 in cutouts_coords:
                    results.append(
                        self._ensure_native_endian_numpy(
                            np.array(hdul[hdu].section[y1:y2, x1:x2], copy=True)
                        )
                    )
                return results

        def fitsio_repeated_cutout():
            with fitsio.FITS(str(path)) as handle:
                results = []
                for x1, y1, x2, y2 in cutouts_coords:
                    results.append(
                        self._ensure_native_endian_numpy(handle[hdu][y1:y2, x1:x2])
                    )
                return results

        def tf_repeated_cutout_persistent():
            with torchfits.open_subset_reader(str(path), hdu=hdu) as reader:
                results = []
                for x1, y1, x2, y2 in cutouts_coords:
                    results.append(reader.read_subset(x1, y1, x2, y2))
                return results

        def tf_repeated_cutout_naive():
            results = []
            for x1, y1, x2, y2 in cutouts_coords:
                results.append(torchfits.read_subset(str(path), hdu, x1, y1, x2, y2))
            return results

        methods = {
            "torchfits": tf_repeated_cutout_naive,
            "astropy_torch": lambda: torch.from_numpy(
                np.array(astropy_repeated_cutout())
            ),
            "fitsio_torch": lambda: torch.from_numpy(
                np.array(fitsio_repeated_cutout())
            ),
            "torchfits_specialized": tf_repeated_cutout_persistent,
            "astropy": astropy_repeated_cutout,
            "fitsio": fitsio_repeated_cutout,
        }

        row: dict[str, Any] = {
            "filename": f"repeated_cutouts_50x_{cutout_size}x{cutout_size}",
            "operation": f"repeated_cutouts_50x_{cutout_size}x{cutout_size}",
            "file_type": "image",
            "data_type": "mixed",
            "dimensions": "2d",
            "compression": "none",
            "size_mb": path.stat().st_size / (1024.0 * 1024.0),
        }
        for method_name, fn in methods.items():
            median_s, _err = _time_median(fn, runs=runs, warmup=warmup)
            row[f"{method_name}_median"] = median_s
            row[f"{method_name}_mb_s"] = self._mb_per_second(path, median_s)
        return [row]

    def _benchmark_random_extensions(
        self,
        files: dict[str, Path],
        *,
        runs: int,
        warmup: int,
    ) -> dict[str, Any] | None:
        path = files.get("multi_mef_10ext")
        if path is None:
            return None
        ext_sequence = [((index * 3) % 10) + 1 for index in range(200)]

        def torchfits_sequence():
            handle = torchfits.cpp.open_fits_file(str(path), "r")
            try:
                result = None
                for hdu in ext_sequence:
                    result = torchfits.cpp.read_full(handle, hdu, False)
                return result
            finally:
                handle.close()

        def astropy_sequence():
            with astropy_fits.open(path, memmap=False) as hdul:
                result = None
                for hdu in ext_sequence:
                    result = self._ensure_native_endian_numpy(
                        np.array(hdul[hdu].data, copy=True)
                    )
                return result

        def fitsio_sequence():
            with fitsio.FITS(str(path)) as handle:
                result = None
                for hdu in ext_sequence:
                    result = self._ensure_native_endian_numpy(handle[hdu].read())
                return result

        methods = {
            "torchfits": torchfits_sequence,
            "astropy_torch": lambda: torch.from_numpy(astropy_sequence()),
            "fitsio_torch": lambda: torch.from_numpy(fitsio_sequence()),
            "torchfits_specialized": torchfits_sequence,
            "astropy": astropy_sequence,
            "fitsio": fitsio_sequence,
        }
        row: dict[str, Any] = {
            "filename": "multi_mef_10ext",
            "operation": "random_ext_full_reads_200",
            "file_type": "multi_mef",
            "data_type": "mixed",
            "dimensions": "2d",
            "compression": "uncompressed",
            "size_mb": path.stat().st_size / (1024.0 * 1024.0),
        }
        for method_name, fn in methods.items():
            median_s, _err = _time_median(fn, runs=runs, warmup=warmup)
            row[f"{method_name}_median"] = median_s
            row[f"{method_name}_mb_s"] = self._mb_per_second(path, median_s)
        return row


def _hdu_for_file_type(file_type: str) -> int:
    if file_type in {"compressed", "mef", "multi_mef"}:
        return 1
    return 0


def _time_median(fn, *, runs: int, warmup: int) -> tuple[float | None, str | None]:
    for _ in range(max(0, warmup)):
        try:
            _ = fn()
        except Exception as exc:
            return None, str(exc)

    times: list[float] = []
    for _ in range(max(1, runs)):
        gc.collect()
        t0 = time.perf_counter()
        try:
            _ = fn()
        except Exception as exc:
            return None, str(exc)
        times.append(time.perf_counter() - t0)
    if not times:
        return None, "no_samples"
    return float(np.median(times)), None


def _strict_patch_astropy(suite: FITSBenchmarkSuite) -> None:
    fallback_paths: set[str] = set()

    @contextmanager
    def _astropy_open_strict(filepath: Path, memmap: bool):
        hdul = None
        try:
            hdul = astropy_fits.open(filepath, memmap=bool(memmap))
        except Exception:
            if bool(memmap):
                # Astropy cannot memmap some scaled/byte-packed payloads; read via
                # non-mmap fallback but mark these rows non-comparable later.
                hdul = astropy_fits.open(filepath, memmap=False)
                fallback_paths.add(str(filepath))
            else:
                raise
        try:
            yield hdul
        finally:
            if hdul is not None:
                hdul.close()

    def _astropy_read_strict(filepath: Path, hdu_num: int):
        try:
            with _astropy_open_strict(filepath, suite.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        return suite._table_to_numpy_dict(hdu.data)
                    return suite._ensure_native_endian_numpy(
                        np.array(hdu.data, copy=True)
                    )
                return None
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(filepath))
                with astropy_fits.open(filepath, memmap=False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            return suite._table_to_numpy_dict(hdu.data)
                        return suite._ensure_native_endian_numpy(
                            np.array(hdu.data, copy=True)
                        )
                    return None
            raise

    def _astropy_to_torch_strict(filepath: Path, hdu_num: int):
        try:
            with _astropy_open_strict(filepath, suite.use_mmap) as hdul:
                hdu = hdul[hdu_num]
                if hasattr(hdu, "data") and hdu.data is not None:
                    if isinstance(hdu, astropy_fits.BinTableHDU):
                        out = {}
                        data = hdu.data
                        for col in data.names:
                            col_data = np.ascontiguousarray(np.asarray(data[col]))
                            if col_data.dtype.byteorder not in ("=", "|"):
                                col_data = col_data.astype(
                                    col_data.dtype.newbyteorder("=")
                                )
                            if col_data.dtype.kind in {"S", "U"}:
                                if col_data.dtype.kind == "U":
                                    col_data = np.char.encode(col_data, "ascii")
                                col_data = (
                                    np.ascontiguousarray(col_data)
                                    .view("uint8")
                                    .reshape(len(col_data), -1)
                                )
                            elif col_data.dtype.kind == "b":
                                col_data = col_data.astype(bool)
                            out[col] = torch.from_numpy(col_data)
                        return out
                    arr = np.array(hdu.data, copy=True)
                    if arr.dtype.byteorder not in ("=", "|"):
                        arr = arr.astype(arr.dtype.newbyteorder("="))
                    return torch.from_numpy(arr)
                return None
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(filepath))
                with astropy_fits.open(filepath, memmap=False) as hdul:
                    hdu = hdul[hdu_num]
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if isinstance(hdu, astropy_fits.BinTableHDU):
                            out = {}
                            data = hdu.data
                            for col in data.names:
                                col_data = np.ascontiguousarray(np.asarray(data[col]))
                                if col_data.dtype.byteorder not in ("=", "|"):
                                    col_data = col_data.astype(
                                        col_data.dtype.newbyteorder("=")
                                    )
                                if col_data.dtype.kind in {"S", "U"}:
                                    if col_data.dtype.kind == "U":
                                        col_data = np.char.encode(col_data, "ascii")
                                    col_data = (
                                        np.ascontiguousarray(col_data)
                                        .view("uint8")
                                        .reshape(len(col_data), -1)
                                    )
                                elif col_data.dtype.kind == "b":
                                    col_data = col_data.astype(bool)
                                out[col] = torch.from_numpy(col_data)
                            return out
                        arr = np.array(hdu.data, copy=True)
                        if arr.dtype.byteorder not in ("=", "|"):
                            arr = arr.astype(arr.dtype.newbyteorder("="))
                        return torch.from_numpy(arr)
                    return None
            raise

    def _astropy_cutout_strict(path, hdu, x1, y1, x2, y2):
        try:
            with _astropy_open_strict(path, suite.use_mmap) as hdul:
                return hdul[hdu].section[y1:y2, x1:x2]
        except Exception:
            if suite.use_mmap:
                fallback_paths.add(str(path))
                with astropy_fits.open(path, memmap=False) as hdul:
                    return hdul[hdu].section[y1:y2, x1:x2]
            raise

    suite._astropy_open = _astropy_open_strict  # type: ignore[attr-defined]
    suite._astropy_read = _astropy_read_strict  # type: ignore[attr-defined]
    suite._astropy_to_torch = _astropy_to_torch_strict  # type: ignore[attr-defined]
    suite._astropy_cutout = _astropy_cutout_strict  # type: ignore[attr-defined]
    suite._astropy_memmap_fallback_paths = fallback_paths  # type: ignore[attr-defined]


def _normalize_legacy_rows(
    *,
    run_id: str,
    raw_rows: list[dict[str, Any]],
    mmap_target: str,
    files: dict[str, Path],
    astropy_fallback_paths: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        domain_file_type = str(raw.get("file_type", ""))
        if domain_file_type in {"table", "wcs", "sphere"}:
            continue

        case_id = f"{raw.get('filename')}::{raw.get('operation', 'read_full')}"
        case_label = f"{raw.get('filename')} [{raw.get('operation', 'read_full')}]"
        metadata = {
            "file_type": raw.get("file_type"),
            "data_type": raw.get("data_type"),
            "dimensions": raw.get("dimensions"),
            "compression": raw.get("compression"),
        }

        for family, methods in (
            ("smart", SMART_METHODS),
            ("specialized", SPECIALIZED_METHODS),
        ):
            for method_key, library, method_label in methods:
                t_col = f"{method_key}_median"
                tp_col = f"{method_key}_mb_s"
                t_val = raw.get(t_col)
                tp_val = raw.get(tp_col)

                status = "OK" if t_val is not None else "FAILED"
                comparable = status == "OK"
                skip_reason = ""

                if library == "fitsio":
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = (
                        "strict_mmap_fairness: comparator mmap mode is not controllable"
                    )
                    t_val = None
                    tp_val = None
                elif library == "astropy" and status != "OK":
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = "strict_mmap_fairness: astropy mmap mode unavailable for this case"
                    t_val = None
                    tp_val = None
                elif (
                    library == "astropy"
                    and mmap_target == "on"
                    and str(files.get(str(raw.get("filename")), ""))
                    in astropy_fallback_paths
                ):
                    comparable = False
                    status = "SKIPPED"
                    skip_reason = (
                        "strict_mmap_fairness: astropy required memmap=False fallback "
                        "for this payload"
                    )
                    t_val = None
                    tp_val = None

                row = {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "fits_io",
                    "case_id": case_id,
                    "case_label": case_label,
                    "operation": raw.get("operation", "read_full"),
                    "family": family,
                    "library": library,
                    "method": method_label,
                    "mode": "smart" if family == "smart" else "specialized",
                    "status": status,
                    "skip_reason": skip_reason,
                    "comparable": comparable,
                    "mmap_target": mmap_target,
                    "time_s": t_val,
                    "throughput": tp_val,
                    "unit": "MB/s",
                    "size_mb": raw.get("size_mb"),
                    "n_points": "",
                    "metadata": metadata,
                }
                rows.append(row)

    return rows


def _benchmark_headers(
    *,
    run_id: str,
    files: dict[str, Path],
    suite: FITSBenchmarkSuite,
    mmap_target: str,
    runs: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_memmap = mmap_target == "on"

    for name, path in sorted(files.items()):
        file_type = suite._get_file_type(name)
        if file_type in {"table", "wcs", "sphere"}:
            continue
        hdu = _hdu_for_file_type(file_type)
        case_id = f"{name}::header_read"
        case_label = f"{name} [header_read]"

        def _tf_header():
            return torchfits.get_header(str(path), hdu=hdu)

        def _astropy_header():
            with astropy_fits.open(path, memmap=target_memmap) as hdul:
                return dict(hdul[hdu].header)

        def _fitsio_header():
            return fitsio.read_header(str(path), ext=hdu)

        methods = [
            ("torchfits", "torchfits_specialized", _tf_header),
            ("astropy", "astropy", _astropy_header),
            ("fitsio", "fitsio", _fitsio_header),
        ]

        for library, method_label, fn in methods:
            t_val, err = _time_median(fn, runs=runs, warmup=warmup)
            status = "OK" if t_val is not None else "FAILED"
            comparable = status == "OK"
            skip_reason = ""
            if library == "fitsio":
                status = "SKIPPED"
                comparable = False
                skip_reason = (
                    "strict_mmap_fairness: comparator mmap mode is not controllable"
                )
                t_val = None

            if library == "astropy" and err and target_memmap:
                status = "SKIPPED"
                comparable = False
                skip_reason = f"strict_mmap_fairness: astropy memmap={target_memmap} unavailable ({err})"
                t_val = None

            rows.append(
                {
                    "run_id": run_id,
                    "domain": "fits",
                    "suite": "fits_io",
                    "case_id": case_id,
                    "case_label": case_label,
                    "operation": "header_read",
                    "family": "specialized",
                    "library": library,
                    "method": method_label,
                    "mode": "specialized",
                    "status": status,
                    "skip_reason": skip_reason,
                    "comparable": comparable,
                    "mmap_target": mmap_target,
                    "time_s": t_val,
                    "throughput": "",
                    "unit": "ops/s",
                    "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                    "n_points": "",
                    "metadata": {
                        "file_type": file_type,
                        "data_type": "header",
                        "dimensions": "n/a",
                        "compression": suite._get_compression_type(name),
                    },
                }
            )

    return rows


def run_fits_domain(
    *,
    run_id: str,
    output_dir: Path,
    profile: str = "user",
    use_mmap: bool = True,
    case_filter: str = "",
    header_runs: int = 7,
    header_warmup: int = 2,
    keep_temp: bool = False,
) -> list[dict[str, Any]]:
    mmap_target = "on" if use_mmap else "off"
    raw_dir = output_dir / "_raw" / "fits"
    raw_dir.mkdir(parents=True, exist_ok=True)

    suite = FITSBenchmarkSuite(
        output_dir=raw_dir,
        use_mmap=use_mmap,
        include_tables=False,
        include_wcs=False,
        include_sphere=False,
        profile=profile,
    )
    _strict_patch_astropy(suite)

    try:
        files = suite.create_test_files()
        files = {
            k: v
            for k, v in files.items()
            if (not k.startswith("table_")) and ("wcs" not in k.lower())
        }
        if case_filter:
            rx = re.compile(case_filter)
            files = {k: v for k, v in files.items() if rx.search(k)}

        print(
            f"[fits] cases={len(files)} mmap={mmap_target} profile={profile}",
            flush=True,
        )

        raw_rows = suite.run_exhaustive_benchmarks(files)
        if not case_filter and len(raw_rows) != suite.EXPECTED_WORKFLOW_COUNT:
            raise RuntimeError(
                "FITS benchmark workflow contract changed: "
                f"expected {suite.EXPECTED_WORKFLOW_COUNT} workflows, "
                f"ran {len(raw_rows)}"
            )
        astropy_fallback_paths = set(
            getattr(suite, "_astropy_memmap_fallback_paths", set())  # type: ignore[attr-defined]
        )
        rows = _normalize_legacy_rows(
            run_id=run_id,
            raw_rows=raw_rows,
            mmap_target=mmap_target,
            files=files,
            astropy_fallback_paths=astropy_fallback_paths,
        )
        rows.extend(
            _benchmark_headers(
                run_id=run_id,
                files=files,
                suite=suite,
                mmap_target=mmap_target,
                runs=header_runs,
                warmup=header_warmup,
            )
        )

        annotate_rankings(rows)
        return rows
    finally:
        if keep_temp:
            print(f"[fits] temp files kept: {suite.temp_dir}", flush=True)
        else:
            # Do not unlink benchmark fixtures while the global CFITSIO handle
            # cache still owns descriptors for them. The next benchmark domain
            # runs in the same process and may otherwise inherit stale native
            # handle state.
            torchfits.clear_file_cache()
            suite.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--header-runs", type=int, default=7)
    parser.add_argument("--header-warmup", type=int, default=2)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_mmap = not args.no_mmap
    if args.mmap:
        use_mmap = True

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    rows = run_fits_domain(
        run_id=run_id,
        output_dir=run_dir,
        profile=args.profile,
        use_mmap=use_mmap,
        case_filter=args.filter,
        header_runs=args.header_runs,
        header_warmup=args.header_warmup,
        keep_temp=args.keep_temp,
    )

    out_csv = run_dir / "fits_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    if args.json_out:
        write_json(args.json_out, rows)
    print(f"[fits] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
