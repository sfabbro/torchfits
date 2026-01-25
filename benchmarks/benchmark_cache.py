"""
Cache performance benchmarks for torchfits.

This module provides focused benchmarks for cache system performance,
including environment detection, configuration optimization, and cache operations.
"""

# Add src to path for imports
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchfits import cache


class CacheBenchmark:
    """Benchmark suite for cache performance."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Test configurations
        self.cache_configs = [
            {"max_files": 50, "max_memory_mb": 512, "name": "small"},
            {"max_files": 100, "max_memory_mb": 1024, "name": "medium"},
            {"max_files": 200, "max_memory_mb": 2048, "name": "large"},
        ]

    def run_configuration_benchmarks(self) -> Dict[str, Any]:
        """Benchmark cache configuration performance."""
        print("  ⚙️  Testing cache configuration...")

        results = {}

        for config in self.cache_configs:
            config_name = config["name"]

            # Test configuration time
            start_time = time.time()
            cache.configure_cache(
                max_files=config["max_files"], max_memory_mb=config["max_memory_mb"]
            )
            config_time = time.time() - start_time

            # Test stats access performance
            start_time = time.time()
            for _ in range(100):
                cache.get_cache_stats()
            stats_time = time.time() - start_time

            # Test cache clearing
            start_time = time.time()
            cache.clear_cache()
            clear_time = time.time() - start_time

            results[config_name] = {
                "config_time_ms": config_time * 1000,
                "stats_time_ms": stats_time * 1000,
                "clear_time_ms": clear_time * 1000,
                "config": config,
            }

            print(
                f"    {config_name.title()} config: "
                f"setup={config_time * 1000:.2f}ms, "
                f"stats={stats_time * 1000:.2f}ms, "
                f"clear={clear_time * 1000:.2f}ms"
            )

        return results

    def run_environment_detection_benchmarks(self) -> Dict[str, Any]:
        """Benchmark environment detection performance."""
        print("  🔍 Testing environment detection...")

        results = {}

        # Test environment detection
        start_time = time.time()
        optimal_config = cache.get_optimal_cache_config()
        detection_time = time.time() - start_time

        # Test auto-configuration
        start_time = time.time()
        cache.configure_for_environment()
        auto_config_time = time.time() - start_time

        results = {
            "detection_time_ms": detection_time * 1000,
            "auto_config_time_ms": auto_config_time * 1000,
            "detected_config": optimal_config,
        }

        print(f"    Environment detection: {detection_time * 1000:.2f}ms")
        print(f"    Auto-configuration: {auto_config_time * 1000:.2f}ms")
        print(
            f"    Detected environment: {optimal_config.get('environment', 'unknown')}"
        )

        return results

    def run_cache_operations_benchmarks(self) -> Dict[str, Any]:
        """Benchmark cache operations performance."""
        print("  🗃️  Testing cache operations...")

        results = {}

        # Configure cache for testing
        cache.configure_cache(max_files=100, max_memory_mb=1024)

        # Test repeated stats access
        iterations = [10, 100, 1000]
        for num_iter in iterations:
            start_time = time.time()
            for _ in range(num_iter):
                cache.get_cache_stats()
            total_time = time.time() - start_time

            results[f"stats_access_{num_iter}"] = {
                "total_time_ms": total_time * 1000,
                "avg_time_per_call_us": (total_time / num_iter) * 1e6,
                "calls_per_sec": num_iter / total_time if total_time > 0 else 0,
            }

            print(
                f"    Stats access ({num_iter} calls): "
                f"{(total_time / num_iter) * 1e6:.2f}μs per call"
            )

        # Test cache manager lifecycle
        start_time = time.time()
        for _ in range(10):
            manager = cache.get_cache_manager()
            manager.clear()
        lifecycle_time = time.time() - start_time

        results["manager_lifecycle"] = {
            "total_time_ms": lifecycle_time * 1000,
            "avg_time_per_cycle_ms": (lifecycle_time / 10) * 1000,
        }

        print(
            f"    Manager lifecycle (10 cycles): {(lifecycle_time / 10) * 1000:.2f}ms per cycle"
        )

        return results

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all cache benchmarks."""
        print("📦 Running Cache Performance Benchmarks...")

        results = {
            "configuration": self.run_configuration_benchmarks(),
            "environment_detection": self.run_environment_detection_benchmarks(),
            "operations": self.run_cache_operations_benchmarks(),
        }

        return results

    def generate_plots(self, results: Dict[str, Any]):
        """Generate cache performance plots."""
        print("  📊 Generating cache plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Cache Performance Analysis", fontsize=16)

        # Configuration performance
        if "configuration" in results:
            config_data = []
            for config_name, metrics in results["configuration"].items():
                config_data.append(
                    {
                        "config": config_name,
                        "setup_time": metrics["config_time_ms"],
                        "stats_time": metrics["stats_time_ms"],
                        "clear_time": metrics["clear_time_ms"],
                    }
                )

            if config_data:
                df_config = pd.DataFrame(config_data)
                df_config.set_index("config")[
                    ["setup_time", "stats_time", "clear_time"]
                ].plot(kind="bar", ax=axes[0, 0])
                axes[0, 0].set_title("Cache Configuration Performance")
                axes[0, 0].set_ylabel("Time (ms)")
                axes[0, 0].tick_params(axis="x", rotation=45)

        # Environment detection
        if "environment_detection" in results:
            env_data = results["environment_detection"]
            categories = ["Detection", "Auto-Config"]
            times = [env_data["detection_time_ms"], env_data["auto_config_time_ms"]]

            axes[0, 1].bar(categories, times)
            axes[0, 1].set_title("Environment Detection Performance")
            axes[0, 1].set_ylabel("Time (ms)")

        # Operations performance
        if "operations" in results:
            ops_data = []
            for key, metrics in results["operations"].items():
                if key.startswith("stats_access_"):
                    num_calls = key.split("_")[-1]
                    ops_data.append(
                        {
                            "num_calls": int(num_calls),
                            "avg_time_us": metrics["avg_time_per_call_us"],
                            "calls_per_sec": metrics["calls_per_sec"],
                        }
                    )

            if ops_data:
                df_ops = pd.DataFrame(ops_data)
                axes[1, 0].plot(df_ops["num_calls"], df_ops["avg_time_us"], marker="o")
                axes[1, 0].set_title("Stats Access Performance")
                axes[1, 0].set_xlabel("Number of Calls")
                axes[1, 0].set_ylabel("Avg Time per Call (μs)")
                axes[1, 0].set_xscale("log")
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].plot(
                    df_ops["num_calls"], df_ops["calls_per_sec"], marker="s"
                )
                axes[1, 1].set_title("Stats Access Throughput")
                axes[1, 1].set_xlabel("Number of Calls")
                axes[1, 1].set_ylabel("Calls per Second")
                axes[1, 1].set_xscale("log")
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "cache_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"    ✅ Cache plots saved to {self.output_dir}")

    def save_results(self, results: Dict[str, Any]):
        """Save results to CSV."""
        csv_data = []

        # Configuration results
        if "configuration" in results:
            for config_name, metrics in results["configuration"].items():
                csv_data.append(
                    {
                        "benchmark_type": "configuration",
                        "test_name": config_name,
                        "config_time_ms": metrics["config_time_ms"],
                        "stats_time_ms": metrics["stats_time_ms"],
                        "clear_time_ms": metrics["clear_time_ms"],
                        "max_files": metrics["config"]["max_files"],
                        "max_memory_mb": metrics["config"]["max_memory_mb"],
                    }
                )

        # Environment detection results
        if "environment_detection" in results:
            env_data = results["environment_detection"]
            csv_data.append(
                {
                    "benchmark_type": "environment_detection",
                    "test_name": "detection",
                    "detection_time_ms": env_data["detection_time_ms"],
                    "auto_config_time_ms": env_data["auto_config_time_ms"],
                    "detected_environment": env_data["detected_config"].get(
                        "environment", "unknown"
                    ),
                }
            )

        # Operations results
        if "operations" in results:
            for key, metrics in results["operations"].items():
                if key.startswith("stats_access_"):
                    num_calls = key.split("_")[-1]
                    csv_data.append(
                        {
                            "benchmark_type": "operations",
                            "test_name": f"stats_access_{num_calls}",
                            "total_time_ms": metrics["total_time_ms"],
                            "avg_time_per_call_us": metrics["avg_time_per_call_us"],
                            "calls_per_sec": metrics["calls_per_sec"],
                        }
                    )
                elif key == "manager_lifecycle":
                    csv_data.append(
                        {
                            "benchmark_type": "operations",
                            "test_name": "manager_lifecycle",
                            "total_time_ms": metrics["total_time_ms"],
                            "avg_time_per_cycle_ms": metrics["avg_time_per_cycle_ms"],
                        }
                    )

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "cache_benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  📄 Results saved to {csv_path}")


def main():
    """Run cache benchmarks."""
    benchmark = CacheBenchmark()
    results = benchmark.run_benchmarks()
    benchmark.generate_plots(results)
    benchmark.save_results(results)
    print("✅ Cache benchmarks completed!")


if __name__ == "__main__":
    main()
