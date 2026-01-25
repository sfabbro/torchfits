import gc
import os
import timeit

import fitsio
import numpy as np
import psutil

import torchfits


def get_open_fds():
    process = psutil.Process()
    return process.num_fds()


def benchmark_mmap():
    filename = "large_mmap.fits"
    shape = (4096, 4096)  # 16M pixels * 4 bytes = 64MB (not huge but enough to test)
    dtype = np.float32

    if not os.path.exists(filename):
        print(f"Creating {filename} with shape {shape}...")
        data = np.random.randn(*shape).astype(dtype)
        fitsio.write(filename, data, clobber=True)

    print("Benchmarking mmap read...")

    initial_fds = get_open_fds()
    print(f"Initial open FDs: {initial_fds}")

    def read_torchfits():
        # Force mmap usage (it's default now for read_image)
        return torchfits.open(filename)[0].to_tensor()

    def read_fitsio():
        return fitsio.read(filename)

    # Warmup
    read_torchfits()

    n_iter = 10
    t_tf = timeit.timeit(read_torchfits, number=n_iter) / n_iter
    t_fitsio = timeit.timeit(read_fitsio, number=n_iter) / n_iter

    print(f"TorchFits (mmap): {t_tf * 1000:.2f} ms")
    print(f"Fitsio:           {t_fitsio * 1000:.2f} ms")
    print(f"Speedup:          {t_fitsio / t_tf:.2f}x")

    # Check for FD leaks
    gc.collect()
    final_fds = get_open_fds()
    print(f"Final open FDs: {final_fds}")

    if final_fds > initial_fds:
        print(f"⚠️  WARNING: Potential FD leak! {final_fds - initial_fds} extra FDs.")
    else:
        print("✅ No FD leak detected.")

    # Test Error Handling
    print("\nTesting Error Handling...")
    try:
        torchfits.open("non_existent_file.fits")
    except FileNotFoundError:
        print("✅ Correctly caught FileNotFoundError")
    except Exception as e:
        print(f"⚠️  Unexpected error for missing file: {type(e).__name__}: {e}")


if __name__ == "__main__":
    benchmark_mmap()
