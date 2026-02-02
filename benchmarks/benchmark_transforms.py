"""
Transform performance benchmarks for torchfits.

This module provides focused benchmarks for the transform pipeline,
measuring performance across different image sizes and transform types.
"""

# Add benchmarks and src to path for imports
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mpl_config import configure

configure()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torchfits import transforms


class TransformBenchmark:
    """Benchmark suite for transform performance."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Test configurations
        self.test_sizes = [
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
        self.transform_configs = {
            "ZScale": transforms.ZScale(),
            "AsinhStretch": transforms.AsinhStretch(),
            "LogStretch": transforms.LogStretch(),
            "PowerStretch": transforms.PowerStretch(),
            "Normalize": transforms.Normalize(),
            "RandomCrop224": transforms.RandomCrop(224),
            "CenterCrop224": transforms.CenterCrop(224),
            "RandomFlip": transforms.RandomFlip(),
            "GaussianNoise": transforms.GaussianNoise(std=0.01),
            "Compose": transforms.Compose(
                [
                    transforms.ZScale(),
                    transforms.RandomFlip(),
                    transforms.GaussianNoise(std=0.01),
                ]
            ),
        }

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run transform benchmarks."""
        print("🎨 Running Transform Benchmarks...")

        results = {}

        for size in self.test_sizes:
            size_key = f"{size[0]}x{size[1]}"
            results[size_key] = {}

            # Create test data
            data = torch.randn(size, dtype=torch.float32)

            print(f"  Testing size {size_key}...")

            for transform_name, transform in self.transform_configs.items():
                # Skip crop transforms for small images
                if "Crop" in transform_name and min(size) < 224:
                    continue

                times = []
                for _ in range(10):  # Multiple runs for accuracy
                    start_time = time.time()
                    try:
                        transform(data)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    ⚠️  {transform_name} failed on {size_key}: {e}")
                        times.append(float("inf"))
                        break

                if times and times[0] != float("inf"):
                    avg_time = np.mean(times[1:])  # Skip first run (warmup)
                    results[size_key][transform_name] = {
                        "avg_time_ms": avg_time * 1000,
                        "throughput_mpixels_per_sec": (size[0] * size[1] / 1e6)
                        / avg_time,
                    }
                    print(
                        f"    {transform_name:15s}: {avg_time * 1000:.2f}ms, {(size[0] * size[1] / 1e6) / avg_time:.2f} MPix/s"
                    )

        return results

    def generate_plots(self, results: Dict[str, Any]):
        """Generate transform performance plots."""
        print("  📊 Generating transform plots...")

        # Prepare data for plotting
        plot_data = []
        for size_key, transforms_data in results.items():
            for transform_name, metrics in transforms_data.items():
                plot_data.append(
                    {
                        "size": size_key,
                        "transform": transform_name,
                        "time_ms": metrics["avg_time_ms"],
                        "throughput": metrics["throughput_mpixels_per_sec"],
                    }
                )

        if not plot_data:
            print("    ⚠️  No data to plot")
            return

        df = pd.DataFrame(plot_data)

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Throughput plot
        pivot_throughput = df.pivot(
            index="transform", columns="size", values="throughput"
        )
        pivot_throughput.plot(kind="bar", ax=ax1)
        ax1.set_title("Transform Throughput by Size")
        ax1.set_ylabel("Throughput (MPix/s)")
        ax1.set_xlabel("Transform")
        ax1.legend(title="Image Size")
        ax1.tick_params(axis="x", rotation=45)

        # Time plot
        pivot_time = df.pivot(index="transform", columns="size", values="time_ms")
        pivot_time.plot(kind="bar", ax=ax2)
        ax2.set_title("Transform Processing Time by Size")
        ax2.set_ylabel("Time (ms)")
        ax2.set_xlabel("Transform")
        ax2.legend(title="Image Size")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "transform_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"    ✅ Transform plots saved to {self.output_dir}")

    def save_results(self, results: Dict[str, Any]):
        """Save results to CSV."""
        csv_data = []
        for size_key, transforms_data in results.items():
            for transform_name, metrics in transforms_data.items():
                csv_data.append(
                    {
                        "size": size_key,
                        "transform": transform_name,
                        "avg_time_ms": metrics["avg_time_ms"],
                        "throughput_mpixels_per_sec": metrics[
                            "throughput_mpixels_per_sec"
                        ],
                    }
                )

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "transform_benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  📄 Results saved to {csv_path}")


def main():
    """Run transform benchmarks."""
    benchmark = TransformBenchmark()
    results = benchmark.run_benchmarks()
    benchmark.generate_plots(results)
    benchmark.save_results(results)
    print("✅ Transform benchmarks completed!")


if __name__ == "__main__":
    main()
