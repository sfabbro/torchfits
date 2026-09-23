import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'benchmarks' package
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

import argparse
import os
import re
import socket
import time
from contextlib import contextmanager
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402

_BENCH_HOST = socket.gethostname()

SMART_METHODS = [
    ("torchfits", "torchfits", "torchfits"),
    ("astropy_torch", "astropy", "astropy_torch"),
    ("fitsio_torch", "fitsio", "fitsio_torch"),
]

SPECIALIZED_METHODS = [
    # Tensor APIs: specialized torchfits vs fitsio/astropy + from_numpy(+clone).
    # Bare ndarray peers invent wrap-only losses on CFITSIO-parity CompImage.
    ("torchfits_specialized", "torchfits", "torchfits_specialized"),
    ("astropy_torch", "astropy", "astropy_torch"),
    ("fitsio_torch", "fitsio", "fitsio_torch"),
]

# Diagnostic ndarray peers (timed separately). Own family so Tensor ranking /
# zero-deficit scorecard never invents wrap-only losses vs bare NumPy.
NUMPY_METHODS = [
    ("astropy", "astropy", "astropy"),
    ("fitsio", "fitsio", "fitsio"),
]


def pixel_itemsize_from_header(header: Any) -> int:
    """Uncompressed pixel width. Tile-compressed HDUs store it in ZBITPIX."""
    raw = header.get("ZBITPIX")
    if raw in (None, ""):
        raw = header.get("BITPIX", 8)
    return max(1, abs(int(raw)) // 8)


def window_payload_bytes(
    header: Any, width: int, height: int, *, count: int = 1
) -> int:
    """Bytes moved by ``count`` cutouts of ``width``×``height`` pixels."""
    return int(count) * int(width) * int(height) * pixel_itemsize_from_header(header)


def plane_payload_bytes(header: Any) -> int:
    """Uncompressed image bytes. Compressed HDUs use ZNAXIS*, not the tile table."""
    if header.get("ZNAXIS") not in (None, ""):
        naxis = int(header.get("ZNAXIS") or 0)
        prefix = "ZNAXIS"
    else:
        naxis = int(header.get("NAXIS") or 0)
        prefix = "NAXIS"
    pixels = 1
    for axis in range(1, naxis + 1):
        pixels *= int(header.get(f"{prefix}{axis}") or 1)
    return pixels * pixel_itemsize_from_header(header)


def mb_from_bytes(nbytes: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return nbytes / (1024.0 * 1024.0) / seconds


class FITSBenchmarkSuite:
    """FITS-only version of the pre-extraction exhaustive fixture suite."""

    EXPECTED_FILE_COUNT = 87
    # Full buffered pass: 87 read_full + 2 cutout + 1 repeated + 1 random_ext.
    EXPECTED_WORKFLOW_COUNT = 91
    EXPECTED_WORKFLOW_COUNT_MMAP_ON = 87

    def __init__(
        self,
        *,
        output_dir: Path,
        use_mmap: bool,
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

    def create_test_files(self, *, fixture_profile: str = "full") -> dict[str, Path]:
        files: dict[str, Path] = {}
        rng = np.random.default_rng(20260318)
        # gpu_core: skip large* to keep modular GPU-only suites cheap while
        # still covering cutouts + compressed + MEF transports.
        size_names = list(self.size_categories.keys())
        if fixture_profile == "gpu_core":
            size_names = [s for s in size_names if s != "large"]

        for size_name in size_names:
            dimensions = self.size_categories[size_name]
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

        scaled_sizes = (
            ("small", "medium")
            if fixture_profile == "gpu_core"
            else ("small", "medium", "large")
        )
        for size_name in scaled_sizes:
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

        for size_name in scaled_sizes:
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
        if fixture_profile == "full" and len(files) != self.EXPECTED_FILE_COUNT:
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

    def run_exhaustive_benchmarks(
        self,
        files: dict[str, Path],
        *,
        runs: int | None = None,
        warmup: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if runs is None:
            runs = 3 if self.profile == "user" else 7
        if warmup is None:
            warmup = 1 if self.profile == "user" else 2
        # Scored runs must honor the pass Boolean — never "auto" under mmap-on.
        torchfits_mmap = bool(self.use_mmap)

        for name, path in sorted(files.items()):
            file_type = self._get_file_type(name)
            hdu = _hdu_for_file_type(file_type)
            print(f"[fits] case={name} file_type={file_type} runs={runs}", flush=True)

            methods = {
                # Peers reopen every iter — disable TorchFits caches for fair cold I/O.
                "torchfits": lambda p=path, h=hdu: torchfits.read(
                    str(p), hdu=h, mmap=torchfits_mmap, use_cache=False
                ),
                "astropy_torch": lambda p=path, h=hdu: self._astropy_to_torch(p, h),
                "fitsio_torch": lambda p=path, h=hdu: torch.from_numpy(
                    self._ensure_native_endian_numpy(fitsio.read(str(p), ext=h))
                ).clone(),
                "torchfits_specialized": lambda p=path, h=hdu: torchfits.read(
                    str(p), hdu=h, mmap=torchfits_mmap, use_cache=False
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
            # One Tensor-peer schedule so smart/specialized share fitsio_torch
            # samples (no cross-schedule invented losses). Clone so the peer owns
            # storage like torchfits. Astropy CompImageHDU uses Python-level
            # decompression (~10× slower than CFITSIO tile decode) — excluded
            # from tensor peers on compressed files to avoid poisoning the
            # CFITSIO-parity round-robin.
            smart_runs = max(runs, 21) if file_type == "compressed" else runs
            tensor_peers = {
                "torchfits": methods["torchfits"],
                "torchfits_specialized": methods["torchfits_specialized"],
                "fitsio_torch": methods["fitsio_torch"],
            }
            if file_type != "compressed":
                tensor_peers["astropy_torch"] = methods["astropy_torch"]
            timed = time_medians_interleaved(
                tensor_peers, runs=smart_runs, warmup=warmup
            )
            timed.update(
                time_medians_interleaved(
                    {
                        "fitsio": methods["fitsio"],
                        "astropy": methods["astropy"],
                    },
                    runs=runs,
                    warmup=warmup,
                )
            )
            for method_name, (median_s, peak_rss, peak_cuda, _err) in timed.items():
                row[f"{method_name}_median"] = median_s
                row[f"{method_name}_mb_s"] = self._mb_per_second(path, median_s)
                row[f"{method_name}_peak_rss_mb"] = peak_rss
                row[f"{method_name}_peak_cuda_alloc_mb"] = peak_cuda
            rows.append(row)

        # Cutout / random_ext paths do not honor the mmap matrix (subset reader
        # has no mmap knob; random_ext peers force buffered opens). Emit once
        # on the buffered pass; normalize labels them mmap_target=n/a.
        if not self.use_mmap:
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

            def tf_cutout(p=path, h=hdu):
                with torchfits.open_subset_reader(str(p), hdu=h) as reader:
                    return reader.read_subset(x1, y1, x2, y2)

            methods = {
                "torchfits": tf_cutout,
                "astropy_torch": lambda fn=astropy_cutout: torch.from_numpy(fn()),
                "fitsio_torch": lambda fn=fitsio_cutout: torch.from_numpy(fn()),
                "torchfits_specialized": tf_cutout,
                "astropy": astropy_cutout,
                "fitsio": fitsio_cutout,
            }
            with fitsio.FITS(str(path)) as handle:
                cutout_header = handle[hdu].read_header()
            payload_bytes = window_payload_bytes(cutout_header, x2 - x1, y2 - y1)
            row: dict[str, Any] = {
                "filename": name,
                "operation": "cutout_100x100",
                "file_type": self._get_file_type(name),
                "data_type": "mixed",
                "dimensions": "2d",
                "compression": compression,
                "size_mb": payload_bytes / (1024.0 * 1024.0),
            }
            timed = time_medians_interleaved(
                {
                    "torchfits": methods["torchfits"],
                    "torchfits_specialized": methods["torchfits_specialized"],
                    "fitsio_torch": methods["fitsio_torch"],
                    "astropy_torch": methods["astropy_torch"],
                },
                runs=runs,
                warmup=warmup,
            )
            # Bare ndarray schedule separate from Tensor peers (CompImage poison).
            timed.update(
                time_medians_interleaved(
                    {
                        "fitsio": methods["fitsio"],
                        "astropy": methods["astropy"],
                    },
                    runs=runs,
                    warmup=warmup,
                )
            )
            for method_name, (median_s, peak_rss, peak_cuda, _err) in timed.items():
                row[f"{method_name}_median"] = median_s
                row[f"{method_name}_mb_s"] = mb_from_bytes(payload_bytes, median_s)
                row[f"{method_name}_peak_rss_mb"] = peak_rss
                row[f"{method_name}_peak_cuda_alloc_mb"] = peak_cuda
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
        payload_bytes = window_payload_bytes(header, cutout_size, cutout_size, count=50)

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

        methods = {
            # Match fitsio's open-once FITS handle: persistent subset reader.
            "torchfits": tf_repeated_cutout_persistent,
            "astropy_torch": lambda fn=astropy_repeated_cutout: torch.from_numpy(
                np.array(fn())
            ),
            "fitsio_torch": lambda fn=fitsio_repeated_cutout: torch.from_numpy(
                np.array(fn())
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
            "size_mb": payload_bytes / (1024.0 * 1024.0),
        }
        timed = time_medians_interleaved(
            {
                "torchfits": methods["torchfits"],
                "torchfits_specialized": methods["torchfits_specialized"],
                "fitsio_torch": methods["fitsio_torch"],
                "astropy_torch": methods["astropy_torch"],
            },
            runs=runs,
            warmup=warmup,
        )
        timed.update(
            time_medians_interleaved(
                {
                    "fitsio": methods["fitsio"],
                    "astropy": methods["astropy"],
                },
                runs=runs,
                warmup=warmup,
            )
        )
        for method_name, (median_s, peak_rss, peak_cuda, _err) in timed.items():
            row[f"{method_name}_median"] = median_s
            row[f"{method_name}_mb_s"] = mb_from_bytes(payload_bytes, median_s)
            row[f"{method_name}_peak_rss_mb"] = peak_rss
            row[f"{method_name}_peak_cuda_alloc_mb"] = peak_cuda
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
        with fitsio.FITS(str(path)) as handle:
            plane = {
                hdu: plane_payload_bytes(handle[hdu].read_header())
                for hdu in set(ext_sequence)
            }
        payload_bytes = sum(plane[hdu] for hdu in ext_sequence)

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
            "astropy_torch": lambda fn=astropy_sequence: torch.from_numpy(fn()),
            "fitsio_torch": lambda fn=fitsio_sequence: torch.from_numpy(fn()),
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
            "size_mb": payload_bytes / (1024.0 * 1024.0),
        }
        timed = time_medians_interleaved(
            {
                "torchfits": methods["torchfits"],
                "torchfits_specialized": methods["torchfits_specialized"],
                "fitsio_torch": methods["fitsio_torch"],
                "astropy_torch": methods["astropy_torch"],
            },
            runs=runs,
            warmup=warmup,
        )
        timed.update(
            time_medians_interleaved(
                {
                    "fitsio": methods["fitsio"],
                    "astropy": methods["astropy"],
                },
                runs=runs,
                warmup=warmup,
            )
        )
        for method_name, (median_s, peak_rss, peak_cuda, _err) in timed.items():
            row[f"{method_name}_median"] = median_s
            row[f"{method_name}_mb_s"] = mb_from_bytes(payload_bytes, median_s)
            row[f"{method_name}_peak_rss_mb"] = peak_rss
            row[f"{method_name}_peak_cuda_alloc_mb"] = peak_cuda
        return row


def _hdu_for_file_type(file_type: str) -> int:
    if file_type in {"compressed", "mef", "multi_mef"}:
        return 1
    return 0


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
        if domain_file_type == "table":
            continue

        operation = str(raw.get("operation", "read_full"))
        case_id = f"{raw.get('filename')}:{operation}"
        case_label = f"{raw.get('filename')} [{operation}]"
        metadata = {
            "file_type": raw.get("file_type"),
            "data_type": raw.get("data_type"),
            "dimensions": raw.get("dimensions"),
            "compression": raw.get("compression"),
        }
        # Ops that cannot honor mmap toggling are scored under n/a, not on/off.
        row_mmap = mmap_target
        if (
            operation.startswith("cutout")
            or "repeated_cutouts" in operation
            or operation.startswith("random_ext")
        ):
            row_mmap = "n/a"

        for family, methods in (
            ("smart", SMART_METHODS),
            ("specialized", SPECIALIZED_METHODS),
            ("numpy", NUMPY_METHODS),
        ):
            for method_key, library, method_label in methods:
                t_col = f"{method_key}_median"
                tp_col = f"{method_key}_mb_s"
                t_val = raw.get(t_col)
                tp_val = raw.get(tp_col)

                status = "OK" if t_val is not None else "FAILED"
                comparable = status == "OK"
                skip_reason = ""

                if library == "astropy" and status != "OK":
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
                elif library == "fitsio" and mmap_target == "on":
                    # fitsio has no mmap toggle — ranking it under mmap-on
                    # compares buffered CFITSIO to torchfits honoring mmap=True.
                    # Exception: compressed images ignore mmap (tile decode), so
                    # fitsio remains a comparable CFITSIO peer.
                    compression = str(raw.get("compression") or "").strip().lower()
                    if compression in {"", "none", "uncompressed", "n/a", "nan"}:
                        comparable = False
                        skip_reason = "fitsio_no_mmap: not comparable under mmap-on"

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
                    "mode": family,
                    "status": status,
                    "skip_reason": skip_reason,
                    "comparable": comparable,
                    "mmap_target": row_mmap,
                    "host": _BENCH_HOST,
                    "time_s": t_val,
                    "peak_rss_mb": raw.get(f"{method_key}_peak_rss_mb"),
                    "peak_cuda_alloc_mb": raw.get(f"{method_key}_peak_cuda_alloc_mb"),
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
        if file_type == "table":
            continue
        hdu = _hdu_for_file_type(file_type)
        case_id = f"{name}:header_read"
        case_label = f"{name} [header_read]"

        def _tf_header():
            return torchfits.read_header(str(path), hdu=hdu)

        def _astropy_header():
            with astropy_fits.open(path, memmap=target_memmap) as hdul:
                return dict(hdul[hdu].header)

        def _fitsio_header():
            return fitsio.read_header(str(path), ext=hdu)

        methods = {
            "torchfits_specialized": _tf_header,
            "astropy": _astropy_header,
            "fitsio": _fitsio_header,
        }
        timed = time_medians_interleaved(methods, runs=runs, warmup=warmup)
        for method_label, (t_val, peak_rss, peak_cuda, err) in timed.items():
            library = (
                "torchfits"
                if method_label.startswith("torchfits")
                else "fitsio"
                if method_label.startswith("fitsio")
                else "astropy"
            )
            status = "OK" if t_val is not None else "FAILED"
            comparable = status == "OK"
            skip_reason = ""
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
                    # Header parse does not toggle image mmap — score once as n/a.
                    "mmap_target": "n/a",
                    "host": _BENCH_HOST,
                    "time_s": t_val,
                    "peak_rss_mb": peak_rss,
                    "peak_cuda_alloc_mb": peak_cuda,
                    "throughput": "",
                    "unit": "",
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
    operation_filter: str = "",
    header_runs: int = 7,
    header_warmup: int = 2,
    keep_temp: bool = False,
    runs: int | None = None,
    warmup: int | None = None,
) -> list[dict[str, Any]]:
    mmap_target = "on" if use_mmap else "off"
    raw_dir = output_dir / "_raw" / "fits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    op_rx = re.compile(operation_filter) if operation_filter else None

    suite = FITSBenchmarkSuite(
        output_dir=raw_dir,
        use_mmap=use_mmap,
        profile=profile,
    )
    _strict_patch_astropy(suite)

    try:
        files = suite.create_test_files()
        files = {k: v for k, v in files.items() if not k.startswith("table_")}
        if case_filter:
            rx = re.compile(case_filter)
            files = {k: v for k, v in files.items() if rx.search(k)}

        print(
            f"[fits] cases={len(files)} mmap={mmap_target} profile={profile}"
            + (f" op_filter={operation_filter!r}" if operation_filter else ""),
            flush=True,
        )

        raw_rows = suite.run_exhaustive_benchmarks(files, runs=runs, warmup=warmup)
        if op_rx is not None:
            raw_rows = [
                r for r in raw_rows if op_rx.search(str(r.get("operation", "")))
            ]
        expected_workflows = (
            suite.EXPECTED_WORKFLOW_COUNT_MMAP_ON
            if use_mmap
            else suite.EXPECTED_WORKFLOW_COUNT
        )
        if (
            not case_filter
            and not operation_filter
            and runs is None
            and len(raw_rows) != expected_workflows
        ):
            raise RuntimeError(
                "FITS benchmark workflow contract changed: "
                f"expected {expected_workflows} workflows, "
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
        # Headers are mmap-irrelevant — emit once on the buffered pass.
        if (not use_mmap) and (op_rx is None or op_rx.search("header_read")):
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
    parser.add_argument(
        "--operation", type=str, default="", help="Regex operation filter"
    )
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
        operation_filter=args.operation,
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
