"""
C++ Backend Performance Benchmark

Diagnostic microbenchmark for C++/image read paths.
This script is informational (non-gating) and uses robust timing statistics.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torchfits
    from torchfits.core import FITSCore  # noqa: F401
except ImportError as e:
    print(f"[diagnostic] torchfits import failed: {e}")
    torchfits = None

try:
    from astropy.io import fits as astropy_fits
except ImportError:
    astropy_fits = None

try:
    import fitsio
except ImportError:
    fitsio = None


def _to_native_endian(array: np.ndarray) -> np.ndarray:
    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))
    return array


class CPPBackendBenchmark:
    """Benchmark C++ backend performance against reference implementations."""

    def __init__(
        self,
        runs: int = 9,
        warmup: int = 3,
        seed: int = 123,
        issue_ratio_threshold: float = 0.97,
        max_relative_spread: float = 0.25,
    ):
        self.results: List[Dict[str, Any]] = []
        self.runs = max(3, int(runs))
        self.warmup = max(0, int(warmup))
        self.issue_ratio_threshold = float(issue_ratio_threshold)
        self.max_relative_spread = float(max_relative_spread)
        np.random.seed(seed)
        random.seed(seed)

    def create_test_data(
        self, shape: tuple[int, int], dtype=np.float32, add_scaling: bool = False
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """Create test data with optional FITS scaling headers."""
        data = np.random.normal(1000, 100, shape).astype(dtype)

        header_kwargs: Dict[str, float] = {}
        if add_scaling:
            header_kwargs["BSCALE"] = 0.01
            header_kwargs["BZERO"] = 1000.0
            data = ((data - 1000.0) / 0.01).astype(np.int16)

        return data, header_kwargs

    def write_test_file(
        self,
        data: np.ndarray,
        header_kwargs: Optional[Dict[str, float]] = None,
        compressed: bool = False,
    ) -> str:
        """Write temporary FITS file and return path."""
        header_kwargs = header_kwargs or {}
        if not astropy_fits:
            raise ImportError("astropy required for test file creation")

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            if compressed:
                hdu = astropy_fits.CompImageHDU(data, compression_type="RICE_1")
            else:
                hdu = astropy_fits.PrimaryHDU(data)

            for key, value in header_kwargs.items():
                hdu.header[key] = value

            hdu.writeto(f.name, overwrite=True)
            return f.name

    def _time_callable(self, func: Callable[[], Any]) -> Dict[str, float]:
        """Robust timing with warmup and median/percentile stats."""
        for _ in range(self.warmup):
            _ = func()

        samples: List[float] = []
        for _ in range(self.runs):
            t0 = time.perf_counter()
            _ = func()
            samples.append(time.perf_counter() - t0)

        arr = np.array(samples, dtype=np.float64)
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        spread = p90 - p10
        rel_spread = (spread / median) if median > 0 else float("inf")
        return {
            "mean_s": mean,
            "median_s": median,
            "std_s": std,
            "p10_s": p10,
            "p90_s": p90,
            "spread_s": spread,
            "rel_spread": rel_spread,
        }

    def _case_label(
        self, shape: tuple[int, int], dtype: np.dtype, compressed: bool, scaled: bool
    ) -> str:
        compression = "compressed" if compressed else "uncompressed"
        scale_mode = "scaled" if scaled else "raw"
        return f"shape={shape} dtype={dtype.__name__} compression={compression} mode={scale_mode}"

    def bench_read_performance(
        self,
        shape: tuple[int, int],
        dtype: np.dtype = np.float32,
        compressed: bool = False,
        scaled: bool = False,
    ) -> None:
        """Benchmark read performance for one case."""
        case_label = self._case_label(shape, dtype, compressed, scaled)
        print(f"\n[diagnostic] Case: {case_label}")

        data, header_kwargs = self.create_test_data(shape, dtype, scaled)
        filepath = self.write_test_file(data, header_kwargs, compressed)
        hdu = 1 if compressed else 0

        result: Dict[str, Any] = {
            "case_label": case_label,
            "shape": list(shape),
            "dtype": dtype.__name__,
            "compressed": bool(compressed),
            "scaled": bool(scaled),
            "hdu": int(hdu),
            "file_size_mb": os.path.getsize(filepath) / 1024 / 1024,
            "runs": self.runs,
            "warmup": self.warmup,
        }

        try:
            if torchfits:
                tf_last = {"tensor_shape": None, "tensor_dtype": None}

                def read_torchfits() -> torch.Tensor:
                    # High-level comparison path.
                    out = torchfits.read(filepath, hdu=hdu, policy="smart")
                    tensor = out[0] if isinstance(out, tuple) else out
                    tf_last["tensor_shape"] = list(tensor.shape)
                    tf_last["tensor_dtype"] = str(tensor.dtype)
                    return tensor

                tf_stats = self._time_callable(read_torchfits)
                result.update({f"torchfits_{k}": v for k, v in tf_stats.items()})
                result["torchfits_shape"] = tf_last["tensor_shape"]
                result["torchfits_dtype"] = tf_last["tensor_dtype"]

                def read_torchfits_specialized() -> torch.Tensor:
                    # Direct/specialized image path.
                    return torchfits.read_image(
                        filepath,
                        hdu=hdu,
                        mmap=(not compressed),
                        handle_cache=True,
                    )

                tf_spec_stats = self._time_callable(read_torchfits_specialized)
                result.update(
                    {f"torchfits_specialized_{k}": v for k, v in tf_spec_stats.items()}
                )

            if astropy_fits:

                def read_astropy() -> torch.Tensor:
                    with astropy_fits.open(filepath) as hdul:
                        array = hdul[hdu].data
                        if array is None:
                            raise RuntimeError("astropy returned no data")
                        return torch.from_numpy(_to_native_endian(array))

                ast_stats = self._time_callable(read_astropy)
                result.update({f"astropy_{k}": v for k, v in ast_stats.items()})

                def read_astropy_specialized() -> np.ndarray:
                    with astropy_fits.open(filepath) as hdul:
                        array = hdul[hdu].data
                        if array is None:
                            raise RuntimeError("astropy returned no data")
                        return _to_native_endian(array)

                ast_spec_stats = self._time_callable(read_astropy_specialized)
                result.update(
                    {f"astropy_specialized_{k}": v for k, v in ast_spec_stats.items()}
                )

            if fitsio:

                def read_fitsio() -> torch.Tensor:
                    array = fitsio.read(filepath, ext=hdu)
                    return torch.from_numpy(_to_native_endian(array))

                fi_stats = self._time_callable(read_fitsio)
                result.update({f"fitsio_{k}": v for k, v in fi_stats.items()})

                def read_fitsio_specialized() -> np.ndarray:
                    return _to_native_endian(fitsio.read(filepath, ext=hdu))

                fi_spec_stats = self._time_callable(read_fitsio_specialized)
                result.update(
                    {f"fitsio_specialized_{k}": v for k, v in fi_spec_stats.items()}
                )

            if "torchfits_median_s" in result and "astropy_median_s" in result:
                result["vs_astropy_median"] = (
                    result["astropy_median_s"] / result["torchfits_median_s"]
                )
            if "torchfits_median_s" in result and "fitsio_median_s" in result:
                result["vs_fitsio_median"] = (
                    result["fitsio_median_s"] / result["torchfits_median_s"]
                )
            if (
                "torchfits_specialized_median_s" in result
                and "fitsio_median_s" in result
            ):
                result["vs_fitsio_median_specialized"] = (
                    result["fitsio_median_s"] / result["torchfits_specialized_median_s"]
                )
            if (
                "torchfits_specialized_median_s" in result
                and "fitsio_specialized_median_s" in result
            ):
                result["vs_fitsio_median_specialized_direct"] = (
                    result["fitsio_specialized_median_s"]
                    / result["torchfits_specialized_median_s"]
                )
            if (
                "torchfits_specialized_median_s" in result
                and "astropy_specialized_median_s" in result
            ):
                result["vs_astropy_median_specialized_direct"] = (
                    result["astropy_specialized_median_s"]
                    / result["torchfits_specialized_median_s"]
                )

            print(f"  File size: {result['file_size_mb']:.2f} MB")
            if "torchfits_median_s" in result:
                print(
                    "  torchfits (smart): "
                    f"median={result['torchfits_median_s'] * 1000:.2f}ms "
                    f"(p10={result['torchfits_p10_s'] * 1000:.2f}ms, "
                    f"p90={result['torchfits_p90_s'] * 1000:.2f}ms, "
                    f"spread={result['torchfits_spread_s'] * 1000:.2f}ms)"
                )
            if "torchfits_specialized_median_s" in result:
                print(
                    "  torchfits (specialized): "
                    f"median={result['torchfits_specialized_median_s'] * 1000:.2f}ms "
                    f"(spread={result['torchfits_specialized_spread_s'] * 1000:.2f}ms)"
                )
            if "astropy_median_s" in result:
                print(
                    "  astropy (smart):   "
                    f"median={result['astropy_median_s'] * 1000:.2f}ms "
                    f"(spread={result['astropy_spread_s'] * 1000:.2f}ms)"
                )
            if "astropy_specialized_median_s" in result:
                print(
                    "  astropy (specialized): "
                    f"median={result['astropy_specialized_median_s'] * 1000:.2f}ms "
                    f"(spread={result['astropy_specialized_spread_s'] * 1000:.2f}ms)"
                )
            if "fitsio_median_s" in result:
                print(
                    "  fitsio (smart):    "
                    f"median={result['fitsio_median_s'] * 1000:.2f}ms "
                    f"(spread={result['fitsio_spread_s'] * 1000:.2f}ms)"
                )
            if "fitsio_specialized_median_s" in result:
                print(
                    "  fitsio (specialized): "
                    f"median={result['fitsio_specialized_median_s'] * 1000:.2f}ms "
                    f"(spread={result['fitsio_specialized_spread_s'] * 1000:.2f}ms)"
                )

            if "vs_astropy_median" in result:
                speed = result["vs_astropy_median"]
                print(
                    f"  torchfits vs astropy (median): {speed:.2f}x {'faster' if speed > 1 else 'slower'}"
                )
            if "vs_fitsio_median" in result:
                speed = result["vs_fitsio_median"]
                print(
                    f"  torchfits smart vs fitsio (median):  {speed:.2f}x {'faster' if speed > 1 else 'slower'}"
                )
            if "vs_fitsio_median_specialized" in result:
                speed = result["vs_fitsio_median_specialized"]
                print(
                    f"  torchfits specialized vs fitsio smart (median):  {speed:.2f}x {'faster' if speed > 1 else 'slower'}"
                )
            if "vs_fitsio_median_specialized_direct" in result:
                speed = result["vs_fitsio_median_specialized_direct"]
                print(
                    f"  torchfits specialized vs fitsio specialized (median):  {speed:.2f}x {'faster' if speed > 1 else 'slower'}"
                )
            if "vs_astropy_median_specialized_direct" in result:
                speed = result["vs_astropy_median_specialized_direct"]
                print(
                    f"  torchfits specialized vs astropy specialized (median): {speed:.2f}x {'faster' if speed > 1 else 'slower'}"
                )

            self.results.append(result)
        finally:
            os.unlink(filepath)

    def bench_cutout_performance(self) -> None:
        """Benchmark cutout/subset reading performance."""
        print("\n[diagnostic] Cutout Performance")
        shape = (4000, 4000)
        data, _ = self.create_test_data(shape, np.float32)
        filepath = self.write_test_file(data)
        cutout_spec = f"{filepath}[0][1000:2000,1000:2000]"

        try:
            if torchfits:

                def tf_cutout() -> torch.Tensor:
                    out = torchfits.read(cutout_spec, policy="smart")
                    return out[0] if isinstance(out, tuple) else out

                tf_stats = self._time_callable(tf_cutout)
                print(
                    "  torchfits cutout: "
                    f"median={tf_stats['median_s'] * 1000:.2f}ms "
                    f"(spread={tf_stats['spread_s'] * 1000:.2f}ms)"
                )

            if astropy_fits:

                def astropy_cutout() -> torch.Tensor:
                    with astropy_fits.open(filepath) as hdul:
                        arr = hdul[0].data
                        subset = arr[1000:2000, 1000:2000]
                        return torch.from_numpy(_to_native_endian(subset))

                ast_stats = self._time_callable(astropy_cutout)
                print(
                    "  astropy full+slice: "
                    f"median={ast_stats['median_s'] * 1000:.2f}ms "
                    f"(spread={ast_stats['spread_s'] * 1000:.2f}ms)"
                )
        finally:
            os.unlink(filepath)

    def bench_scaling_performance(self) -> None:
        """Benchmark BSCALE/BZERO scaling performance."""
        print("\n[diagnostic] Scaling Performance")
        shape = (2000, 2000)
        self.bench_read_performance(shape, np.int16, compressed=False, scaled=True)
        self.bench_read_performance(shape, np.float32, compressed=False, scaled=False)

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Analyze stable issues and print grouped medians."""
        print("\n[diagnostic] Performance Analysis")

        stable_issues: List[Dict[str, Any]] = []
        stable_issues_specialized: List[Dict[str, Any]] = []
        for result in self.results:
            ratio = result.get("vs_fitsio_median")
            tf_rel_spread = result.get("torchfits_rel_spread")
            fi_rel_spread = result.get("fitsio_rel_spread")
            if ratio is None:
                continue
            stable = (
                ratio < self.issue_ratio_threshold
                and tf_rel_spread is not None
                and fi_rel_spread is not None
                and tf_rel_spread <= self.max_relative_spread
                and fi_rel_spread <= self.max_relative_spread
            )
            if stable:
                stable_issues.append(
                    {
                        "case_label": result["case_label"],
                        "mode": "smart",
                        "vs_fitsio_median": ratio,
                        "torchfits_rel_spread": tf_rel_spread,
                        "fitsio_rel_spread": fi_rel_spread,
                    }
                )

            ratio_spec = result.get("vs_fitsio_median_specialized_direct")
            tf_spec_rel_spread = result.get("torchfits_specialized_rel_spread")
            fi_spec_rel_spread = result.get("fitsio_specialized_rel_spread")
            stable_spec = (
                ratio_spec is not None
                and ratio_spec < self.issue_ratio_threshold
                and tf_spec_rel_spread is not None
                and fi_spec_rel_spread is not None
                and tf_spec_rel_spread <= self.max_relative_spread
                and fi_spec_rel_spread <= self.max_relative_spread
            )
            if stable_spec:
                stable_issues_specialized.append(
                    {
                        "case_label": result["case_label"],
                        "mode": "specialized",
                        "vs_fitsio_median": ratio_spec,
                        "torchfits_rel_spread": tf_spec_rel_spread,
                        "fitsio_rel_spread": fi_spec_rel_spread,
                    }
                )

        if stable_issues:
            print("\n[diagnostic] Stable Performance Issues (non-gating):")
            for issue in stable_issues:
                print(
                    "  "
                    f"{issue['case_label']}: "
                    f"vs_fitsio={issue['vs_fitsio_median']:.2f}x, "
                    f"tf_rel_spread={issue['torchfits_rel_spread']:.2f}, "
                    f"fitsio_rel_spread={issue['fitsio_rel_spread']:.2f}"
                )
        else:
            print("\n[diagnostic] No stable slow-vs-fitsio issues detected.")

        if stable_issues_specialized:
            print("\n[diagnostic] Stable Performance Issues (specialized, non-gating):")
            for issue in stable_issues_specialized:
                print(
                    "  "
                    f"{issue['case_label']}: "
                    f"vs_fitsio={issue['vs_fitsio_median']:.2f}x, "
                    f"tf_rel_spread={issue['torchfits_rel_spread']:.2f}, "
                    f"fitsio_rel_spread={issue['fitsio_rel_spread']:.2f}"
                )
        else:
            print("\n[diagnostic] No stable specialized slow-vs-fitsio issues detected.")

        grouped: Dict[str, List[float]] = {
            "compressed": [],
            "uncompressed": [],
            "scaled": [],
        }
        grouped_specialized: Dict[str, List[float]] = {
            "compressed": [],
            "uncompressed": [],
            "scaled": [],
        }
        for result in self.results:
            ratio = result.get("vs_fitsio_median")
            ratio_spec = result.get("vs_fitsio_median_specialized_direct")
            if ratio is None:
                ratio = None
            if result.get("compressed"):
                if ratio is not None:
                    grouped["compressed"].append(ratio)
                if ratio_spec is not None:
                    grouped_specialized["compressed"].append(ratio_spec)
            else:
                if ratio is not None:
                    grouped["uncompressed"].append(ratio)
                if ratio_spec is not None:
                    grouped_specialized["uncompressed"].append(ratio_spec)
            if result.get("scaled"):
                if ratio is not None:
                    grouped["scaled"].append(ratio)
                if ratio_spec is not None:
                    grouped_specialized["scaled"].append(ratio_spec)

        print("\n[diagnostic] Ratio Summary vs fitsio (smart mode, median-based):")
        summary: Dict[str, Any] = {
            "stable_issues": stable_issues + stable_issues_specialized,
            "group_summary_smart": {},
            "group_summary_specialized": {},
        }
        for group, vals in grouped.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            group_stats = {
                "count": int(arr.size),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
            summary["group_summary_smart"][group] = group_stats
            print(
                f"  {group:12s} n={group_stats['count']:2d} "
                f"median={group_stats['median']:.2f}x "
                f"p10={group_stats['p10']:.2f}x "
                f"p90={group_stats['p90']:.2f}x "
                f"min={group_stats['min']:.2f}x"
            )

        print("\n[diagnostic] Ratio Summary vs fitsio (specialized mode, median-based):")
        for group, vals in grouped_specialized.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            group_stats = {
                "count": int(arr.size),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
            summary["group_summary_specialized"][group] = group_stats
            print(
                f"  {group:12s} n={group_stats['count']:2d} "
                f"median={group_stats['median']:.2f}x "
                f"p10={group_stats['p10']:.2f}x "
                f"p90={group_stats['p90']:.2f}x "
                f"min={group_stats['min']:.2f}x"
            )

        return summary

    def save_results_json(self, output_file: Path, summary: Dict[str, Any]) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "runs": self.runs,
                "warmup": self.warmup,
                "issue_ratio_threshold": self.issue_ratio_threshold,
                "max_relative_spread": self.max_relative_spread,
            },
            "summary": summary,
            "results": self.results,
        }
        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)

    def run_comprehensive_benchmark(self, json_out: Optional[Path] = None) -> Dict[str, Any]:
        """Run comprehensive C++ backend benchmark."""
        print("[diagnostic] C++ Backend Performance Benchmark (non-gating)")
        print("=" * 50)

        if not torchfits:
            print("[diagnostic] SKIPPED: torchfits not available")
            return {"status": "SKIPPED", "reason": "torchfits_not_available", "results": []}

        test_configs = [
            ((100, 100), np.float32, False),
            ((100, 100), np.int16, False),
            ((1000, 1000), np.float32, False),
            ((1000, 1000), np.int16, False),
            ((1000, 1000), np.float64, False),
            ((2000, 2000), np.float32, False),
            ((2000, 2000), np.int16, False),
            ((1000, 1000), np.float32, True),
            ((1000, 1000), np.int16, True),
        ]

        for shape, dtype, compressed in test_configs:
            self.bench_read_performance(shape, dtype, compressed)

        self.bench_cutout_performance()
        self.bench_scaling_performance()
        summary = self.identify_bottlenecks()

        output_file = (
            json_out
            if json_out is not None
            else Path(__file__).parent.parent / "bench_results" / "cpp_backend_results.json"
        )
        self.save_results_json(output_file, summary)
        print(f"\n[diagnostic] Results saved to: {output_file}")

        return {
            "status": "WARN" if summary["stable_issues"] else "PASS",
            "results": self.results,
            "summary": summary,
            "json_out": str(output_file),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C++ backend diagnostic benchmark (informational, non-gating)."
    )
    parser.add_argument("--runs", type=int, default=9, help="Timed repeats per method")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup repeats per method")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--issue-ratio-threshold",
        type=float,
        default=0.97,
        help="Flag only if fitsio/torchfits median ratio is below this threshold",
    )
    parser.add_argument(
        "--max-relative-spread",
        type=float,
        default=0.25,
        help="Require both methods to have spread/median below this value for stable issue flags",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results") / "cpp_backend_results.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark = CPPBackendBenchmark(
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        issue_ratio_threshold=args.issue_ratio_threshold,
        max_relative_spread=args.max_relative_spread,
    )
    benchmark.run_comprehensive_benchmark(json_out=args.json_out)


if __name__ == "__main__":
    main()
