import concurrent.futures
import os
import timeit

import fitsio
import numpy as np
import torch

import torchfits
import torchfits.cpp as cpp


def benchmark_parallel():
    n_files = 16
    filename_base = "parallel_test"
    shape = (1024, 1024)
    dtype = np.float32

    files = []
    for i in range(n_files):
        fname = f"{filename_base}_{i}.fits"
        files.append(fname)
        if not os.path.exists(fname):
            data = np.random.randn(*shape).astype(dtype)
            fitsio.write(fname, data, clobber=True)

    print(f"Benchmarking parallel read of {n_files} files...")

    def read_serial_torchfits():
        tensors = []
        for f in files:
            tensors.append(torchfits.open(f)[0].to_tensor())
        return tensors

    def read_parallel_torchfits():
        # Use the C++ batch reader
        return cpp.read_images_batch(files, 0)

    def read_parallel_python_threads():
        # Python threading with torchfits (should release GIL)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(lambda f: torchfits.open(f)[0].to_tensor(), f)
                for f in files
            ]
            return [f.result() for f in futures]

    # Warmup
    read_parallel_torchfits()

    n_iter = 5

    t_serial = timeit.timeit(read_serial_torchfits, number=n_iter) / n_iter
    t_parallel_cpp = timeit.timeit(read_parallel_torchfits, number=n_iter) / n_iter
    t_parallel_py = timeit.timeit(read_parallel_python_threads, number=n_iter) / n_iter

    print(f"Serial (Python):       {t_serial * 1000:.2f} ms")
    print(f"Parallel (Python Threads): {t_parallel_py * 1000:.2f} ms")
    print(f"Parallel (C++ Batch):  {t_parallel_cpp * 1000:.2f} ms")

    speedup_cpp = t_serial / t_parallel_cpp
    speedup_py = t_serial / t_parallel_py

    print(f"Speedup (C++): {speedup_cpp:.2f}x")
    print(f"Speedup (Py):  {speedup_py:.2f}x")

    # Verify correctness
    res_cpp = read_parallel_torchfits()
    res_serial = read_serial_torchfits()

    assert len(res_cpp) == len(res_serial)
    for i in range(n_files):
        assert torch.allclose(res_cpp[i], res_serial[i])


if __name__ == "__main__":
    benchmark_parallel()
