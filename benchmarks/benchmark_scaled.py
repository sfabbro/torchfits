import os
import timeit

import fitsio
import numpy as np

import torchfits


def benchmark_scaled():
    filename = "scaled.fits"
    shape = (4096, 4096)
    dtype = np.int16
    bscale = 1.5
    bzero = 32768.0

    if not os.path.exists(filename):
        print(f"Creating {filename} with shape {shape} and scaling...")
        data = np.random.randint(-32768, 32767, size=shape, dtype=dtype)
        # fitsio handles scaling automatically if header keywords are present?
        # Or we write raw and set keywords manually.
        # Let's write raw data then update header.
        fitsio.write(filename, data, clobber=True)

        with fitsio.FITS(filename, "rw") as f:
            f[0].write_key("BSCALE", bscale)
            f[0].write_key("BZERO", bzero)

    print("Benchmarking scaled read...")

    def read_torchfits():
        return torchfits.read(filename, scale_on_device=True)

    def read_fitsio():
        return fitsio.read(filename)

    # Warmup
    read_torchfits()
    read_fitsio()

    n_iter = 10
    t_tf = timeit.timeit(read_torchfits, number=n_iter) / n_iter
    t_fitsio = timeit.timeit(read_fitsio, number=n_iter) / n_iter

    print(f"TorchFits: {t_tf * 1000:.2f} ms")
    print(f"Fitsio:    {t_fitsio * 1000:.2f} ms")
    print(f"Speedup:   {t_fitsio / t_tf:.2f}x")

    # Verify correctness
    tf_data = read_torchfits()
    fitsio_data = read_fitsio()

    # fitsio returns float64 for scaled data by default? or float32?
    # torchfits returns float32.
    # Check max difference.
    diff = np.abs(tf_data.numpy() - fitsio_data).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-4


if __name__ == "__main__":
    benchmark_scaled()
