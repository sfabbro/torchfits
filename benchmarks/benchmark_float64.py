import os
import time

import fitsio
import numpy as np
import torch

import torchfits


def create_float64_fits(filename, shape=(4096, 4096)):
    print(f"Creating {filename} with shape {shape} (float64)...")
    data = np.random.randn(*shape).astype(np.float64)
    if os.path.exists(filename):
        os.remove(filename)
    fitsio.write(filename, data)
    return filename


def benchmark(filename):
    print(f"\nBenchmarking {filename}...")

    # Warmup
    torchfits.read(filename)
    fitsio.read(filename)

    torchfits.clear_file_cache()

    # TorchFits
    start = time.time()
    t_data, _ = torchfits.read(filename)
    end = time.time()
    tf_time = end - start
    print(f"TorchFits: {tf_time*1000:.2f} ms")
    print(f"TorchFits dtype: {t_data.dtype}")

    # Fitsio
    start = time.time()
    f_data = fitsio.read(filename)
    end = time.time()
    f_time = end - start
    print(f"Fitsio:    {f_time*1000:.2f} ms")
    print(f"Fitsio dtype: {f_data.dtype}")

    print(f"Speedup:   {f_time/tf_time:.2f}x")


if __name__ == "__main__":
    filename = "float64_test.fits"
    create_float64_fits(filename)
    benchmark(filename)
    os.remove(filename)
