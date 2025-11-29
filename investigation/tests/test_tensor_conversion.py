#!/usr/bin/env python3
"""
Test if tensor_to_python (THPVariable_Wrap) has dtype-specific overhead.
"""
import time
import statistics
import torch
from torchfits import cpp

def benchmark_tensor_return(dtype, shape, iterations=1000):
    """Benchmark returning a tensor from C++."""
    # Create a simple C++ function that creates and returns a tensor
    times = []

    for _ in range(iterations):
        # This will create a tensor in C++ and convert it via tensor_to_python
        start = time.perf_counter()

        # Create tensor in Python (to avoid C++ overhead), then pass through C++
        tensor = torch.zeros(shape, dtype=dtype)

        end = time.perf_counter()
        times.append((end - start) * 1e6)  # microseconds

    return statistics.median(times)

def main():
    shape = (1000, 1000)
    print(f"Testing tensor allocation overhead ({shape[0]}x{shape[1]} tensors)...")
    print()

    uint8_time = benchmark_tensor_return(torch.uint8, shape)
    int16_time = benchmark_tensor_return(torch.int16, shape)
    int32_time = benchmark_tensor_return(torch.int32, shape)
    float32_time = benchmark_tensor_return(torch.float32, shape)

    print(f"uint8:   {uint8_time:.1f}μs")
    print(f"int16:   {int16_time:.1f}μs")
    print(f"int32:   {int32_time:.1f}μs")
    print(f"float32: {float32_time:.1f}μs")
    print()

    print("Ratios vs uint8:")
    print(f"int16:   {int16_time/uint8_time:.2f}x")
    print(f"int32:   {int32_time/uint8_time:.2f}x")
    print(f"float32: {float32_time/uint8_time:.2f}x")
    print()

    # Now test the actual read path to see where the overhead is
    print("=" * 60)
    print("Testing actual read path overhead:")
    print("=" * 60)

    import tempfile
    from pathlib import Path
    import numpy as np
    from astropy.io import fits

    # Create test files
    tmpdir = Path(tempfile.gettempdir())

    uint8_file = tmpdir / f"tensor_test_uint8_{time.time_ns()}.fits"
    data_uint8 = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    fits.writeto(uint8_file, data_uint8, overwrite=True)

    int16_file = tmpdir / f"tensor_test_int16_{time.time_ns()}.fits"
    data_int16 = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    fits.writeto(int16_file, data_int16, overwrite=True)

    # Test with fresh reads each time
    print("\nFresh reads (no handle reuse):")

    times_uint8 = []
    for _ in range(100):
        start = time.perf_counter()
        h = cpp.open_fits_file(str(uint8_file), 0)
        t = cpp.read_full(h, 0)
        cpp.close_fits_file(h)
        times_uint8.append((time.perf_counter() - start) * 1000)

    times_int16 = []
    for _ in range(100):
        start = time.perf_counter()
        h = cpp.open_fits_file(str(int16_file), 0)
        t = cpp.read_full(h, 0)
        cpp.close_fits_file(h)
        times_int16.append((time.perf_counter() - start) * 1000)

    uint8_median = statistics.median(times_uint8)
    int16_median = statistics.median(times_int16)

    print(f"  uint8:  {uint8_median:.4f}ms")
    print(f"  int16:  {int16_median:.4f}ms")
    print(f"  Ratio:  {int16_median/uint8_median:.2f}x")

    # Cleanup
    uint8_file.unlink()
    int16_file.unlink()

if __name__ == "__main__":
    main()
