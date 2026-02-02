#!/usr/bin/env python3
"""
Benchmark script to compare torchfits performance on CPU vs MPS.
"""

import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits


def benchmark_mps_vs_cpu():
    print("=" * 60)
    print("Torchfits MPS vs CPU Benchmark")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("WARNING: MPS is not available on this machine.")
        print("This benchmark requires a Mac with Apple Silicon.")
        return

    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Create a large image file (float32)
        print("\n1. Creating test data (2k x 2k float32 image)...")
        shape = (2048, 2048)
        data = np.random.randn(*shape).astype(np.float32)
        img_path = temp_path / "test_image.fits"
        astropy_fits.PrimaryHDU(data).writeto(img_path)

        # Warmup MPS
        print("\nWarming up MPS...")
        warmup_data = torch.randn(1000, 1000, device="mps")
        torch.mps.synchronize()
        del warmup_data

        # 2. Benchmark Image Read
        print("\n2. Benchmarking Image Read:")

        # CPU
        gc.collect()
        start = time.perf_counter()
        cpu_data, _ = torchfits.read(
            str(img_path), device="cpu", return_header=True
        )
        cpu_time = time.perf_counter() - start
        print(f"  CPU Read: {cpu_time:.4f}s")

        # MPS
        gc.collect()
        torch.mps.empty_cache()
        start = time.perf_counter()
        mps_data, _ = torchfits.read(
            str(img_path), device="mps", return_header=True
        )
        torch.mps.synchronize()  # Ensure operation is complete
        mps_time = time.perf_counter() - start
        print(f"  MPS Read: {mps_time:.4f}s")
        print(f"  Speedup: {cpu_time / mps_time:.2f}x")

        # Verify data
        print("  Verifying data match...", end="")
        if torch.allclose(cpu_data, mps_data.cpu()):
            print(" OK")
        else:
            print(" FAILED")

        # 3. Benchmark Simple Transform (e.g. AsinhStretch)
        print("\n3. Benchmarking Transform (AsinhStretch):")
        from torchfits.transforms import AsinhStretch

        transform = AsinhStretch()

        # Warmup transform
        _ = transform(torch.randn(100, 100, device="mps"))
        torch.mps.synchronize()

        # CPU
        gc.collect()
        start = time.perf_counter()
        _ = transform(cpu_data)
        cpu_trans_time = time.perf_counter() - start
        print(f"  CPU Transform: {cpu_trans_time:.4f}s")

        # MPS
        gc.collect()
        torch.mps.empty_cache()
        start = time.perf_counter()
        transform(mps_data)
        torch.mps.synchronize()
        mps_trans_time = time.perf_counter() - start
        print(f"  MPS Transform: {mps_trans_time:.4f}s")
        print(f"  Speedup: {cpu_trans_time / mps_trans_time:.2f}x")

        # 4. Benchmark Large Table Read (if applicable)
        print("\n4. Benchmarking Table Read (100k rows):")
        nrows = 100000
        table_data = {
            "col1": np.random.randn(nrows).astype(np.float32),
            "col2": np.random.randint(0, 100, nrows).astype(np.int32),
        }
        table_path = temp_path / "test_table.fits"
        t = astropy_fits.BinTableHDU.from_columns(
            [
                astropy_fits.Column(name="col1", format="E", array=table_data["col1"]),
                astropy_fits.Column(name="col2", format="J", array=table_data["col2"]),
            ]
        )
        t.writeto(table_path)

        # CPU
        gc.collect()
        start = time.perf_counter()
        cpu_table, _ = torchfits.read(
            str(table_path), device="cpu", return_header=True
        )
        cpu_table_time = time.perf_counter() - start
        print(f"  CPU Table Read: {cpu_table_time:.4f}s")

        gc.collect()
        torch.mps.empty_cache()
        start = time.perf_counter()
        mps_table, _ = torchfits.read(
            str(table_path), device="mps", return_header=True
        )
        # Synchronize all tensors
        if isinstance(mps_table, dict):
            for v in mps_table.values():
                if isinstance(v, torch.Tensor) and v.device.type == "mps":
                    torch.mps.synchronize()
        elif isinstance(mps_table, torch.Tensor) and mps_table.device.type == "mps":
            torch.mps.synchronize()

        mps_table_time = time.perf_counter() - start
        print(f"  MPS Table Read: {mps_table_time:.4f}s")
        print(f"  Speedup: {cpu_table_time / mps_table_time:.2f}x")


if __name__ == "__main__":
    benchmark_mps_vs_cpu()
