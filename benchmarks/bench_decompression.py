import os
import time
import numpy as np
import torch
import fitsio
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import torchfits
from astropy.io import fits


def create_compressed_fits(filename, shape=(4096, 4096), dtype="int32"):
    """Create a Rice-compressed FITS file for benchmarking."""
    print(f"Generating synthetic data {shape}...")
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    # Add some noise to make compression non-trivial
    data += np.random.randint(0, 100, size=shape, dtype=dtype)

    if os.path.exists(filename):
        os.remove(filename)

    print(f"Writing compressed FITS to {filename} (RICE_1)...")
    fq = fits.CompImageHDU(data=data, compression_type="RICE_1")
    fq.writeto(filename)

    # Verify file size
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB (Uncompressed: {data.nbytes / 1024**2:.2f} MB)")
    return filename


def bench_fitsio(filename):
    print("\n--- fitsio (C-Engine) ---")
    start = time.time()
    with fitsio.FITS(filename) as f:
        data = f[1].read()
    end = time.time()
    t = end - start
    print(f"Time: {t:.4f} s")
    print(f"Throughput: {data.nbytes / 1024**2 / t:.2f} MB/s")
    return t


def bench_torchfits(filename):
    print("\n--- torchfits (Native) ---")

    start = time.time()
    # Use public read API
    data = torchfits.read(filename, 1)

    # Force sync
    if isinstance(data, torch.Tensor):
        _ = data[0, 0].item()

    end = time.time()
    t = end - start
    print(f"Time: {t:.4f} s")
    print(f"Throughput: {data.nbytes / 1024**2 / t:.2f} MB/s")
    return t


def bench_rice_parallel(filename, threads=4):
    import torchfits.cpp

    print(f"\n--- torchfits (Rice Parallel {threads} threads) ---")
    start = time.time()
    # Direct C++ call
    data = torchfits.cpp.read_rice_parallel(filename, 1, threads)

    # Force sync (it returns tensor so it is already in memory, but just in case)
    if isinstance(data, torch.Tensor):
        _ = data[0, 0].item()

    end = time.time()
    t = end - start
    print(f"Time: {t:.4f} s")
    print(f"Throughput: {data.nbytes / 1024**2 / t:.2f} MB/s")
    return t


def main():
    filename = "bench_rice.fits"
    # Use larger file to amortize thread startup cost (8k x 8k = ~256MB uncompressed)
    # create_compressed_fits(filename, shape=(8192, 8192))
    if not os.path.exists(filename):
        create_compressed_fits(filename, shape=(8192, 8192))

    # Baseline
    t_fitsio = bench_fitsio(filename)

    # Scaling test
    print("\n--- torchfits Scaling ---")
    for threads in [1, 2, 4, 8]:
        # Legacy path via ENV (no scaling expected)
        # os.environ["TORCHFITS_COMPRESSED_PARALLEL_MAX_THREADS"] = str(threads)
        # print(f"\nThreads: {threads} (Legacy)")
        # t = bench_torchfits(filename)

        # New Parallel Path
        t_new = bench_rice_parallel(filename, threads)
        print(f"Speedup vs fitsio: {t_fitsio / t_new:.2f}x")

    print("\n--- Environment ---")
    print(sys.version)

    print("\n--- Raw Table Access Test ---")
    try:
        # Try to read the underlying BINTABLE
        t = torchfits.read(filename, 1, mode="table")
        print("Success! Keys:", t.keys())
        if "COMPRESSED_DATA" in t:
            data = t["COMPRESSED_DATA"]
            print(f"COMPRESSED_DATA found, type: {type(data)}")
            if isinstance(data, list):
                print(f"Length: {len(data)}")
                if len(data) > 0:
                    print(f"First tile bytes: {len(data[0])}")
    except Exception as e:
        print("Failed to read as table:", e)

    # if os.path.exists(filename):
    #     os.remove(filename)


if __name__ == "__main__":
    main()
