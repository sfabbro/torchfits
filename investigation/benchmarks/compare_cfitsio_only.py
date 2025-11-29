#!/usr/bin/env python3
"""
Compare pure CFITSIO read performance (no tensor creation) between
our code and what fitsio likely experiences.
"""
import time
import statistics
import tempfile
from pathlib import Path
import numpy as np
from astropy.io import fits
import ctypes

# Load CFITSIO directly
libcfitsio = ctypes.CDLL('.pixi/envs/default/lib/libcfitsio.dylib')

# Define CFITSIO types
class fitsfile(ctypes.Structure):
    pass

fitsfile_p = ctypes.POINTER(fitsfile)

# CFITSIO functions
fits_open_file = libcfitsio.fits_open_file
fits_open_file.argtypes = [ctypes.POINTER(fitsfile_p), ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fits_open_file.restype = ctypes.c_int

fits_read_pixll = libcfitsio.fits_read_pixll
fits_read_pixll.argtypes = [fitsfile_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong),
                            ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
fits_read_pixll.restype = ctypes.c_int

fits_close_file = libcfitsio.fits_close_file
fits_close_file.argtypes = [fitsfile_p, ctypes.POINTER(ctypes.c_int)]
fits_close_file.restype = ctypes.c_int

READONLY = 0
TBYTE = 11
TSHORT = 21

def benchmark_pure_cfitsio(filepath, datatype, size, iterations=100):
    """Benchmark pure CFITSIO read without any wrapper."""
    times = []

    itemsize = 1 if datatype == TBYTE else 2
    buffer = np.zeros(size, dtype=np.uint8 if datatype == TBYTE else np.int16)

    for _ in range(iterations):
        fptr = fitsfile_p()
        status = ctypes.c_int(0)

        # Open file
        fits_open_file(ctypes.byref(fptr), filepath.encode(), READONLY, ctypes.byref(status))
        if status.value != 0:
            raise RuntimeError(f"Failed to open file: {status.value}")

        # Setup for read
        firstpix = (ctypes.c_longlong * 2)(1, 1)
        anynul = ctypes.c_int(0)

        start = time.perf_counter()

        # Read data
        fits_read_pixll(fptr, datatype, firstpix, size, None,
                       buffer.ctypes.data_as(ctypes.c_void_p),
                       ctypes.byref(anynul), ctypes.byref(status))

        end = time.perf_counter()

        if status.value != 0:
            raise RuntimeError(f"Failed to read: {status.value}")

        times.append((end - start) * 1000)

        # Close file
        fits_close_file(fptr, ctypes.byref(status))

    return statistics.median(times)

def create_test_file(dtype_str):
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"cfitsio_compare_{dtype_str}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def main():
    print("=" * 80)
    print("PURE CFITSIO PERFORMANCE TEST")
    print("Comparing raw CFITSIO read without any Python/C++ wrapper overhead")
    print("=" * 80)
    print()

    # Create test files
    uint8_file = create_test_file('uint8')
    int16_file = create_test_file('int16')
    size = 1000 * 1000

    print("Testing pure CFITSIO via ctypes (100 iterations):")
    print("-" * 80)

    uint8_time = benchmark_pure_cfitsio(uint8_file, TBYTE, size, 100)
    int16_time = benchmark_pure_cfitsio(int16_file, TSHORT, size, 100)

    print(f"  uint8:  {uint8_time:.4f}ms")
    print(f"  int16:  {int16_time:.4f}ms")
    print(f"  Ratio:  {int16_time/uint8_time:.2f}x")
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    print(f"Pure CFITSIO int16/uint8 ratio: {int16_time/uint8_time:.2f}x")
    print()
    print("This is the inherent CFITSIO overhead for int16.")
    print("Any ratio higher than this in our code indicates wrapper overhead.")
    print()

    # Compare with fitsio
    import fitsio

    fitsio_times_uint8 = []
    fitsio_times_int16 = []

    for _ in range(100):
        start = time.perf_counter()
        _ = fitsio.read(uint8_file)
        fitsio_times_uint8.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        _ = fitsio.read(int16_file)
        fitsio_times_int16.append((time.perf_counter() - start) * 1000)

    fitsio_uint8 = statistics.median(fitsio_times_uint8)
    fitsio_int16 = statistics.median(fitsio_times_int16)

    print("fitsio (CFITSIO + numpy wrapper):")
    print(f"  uint8:  {fitsio_uint8:.4f}ms")
    print(f"  int16:  {fitsio_int16:.4f}ms")
    print(f"  Ratio:  {fitsio_int16/fitsio_uint8:.2f}x")
    print()

    wrapper_overhead_uint8 = fitsio_uint8 - uint8_time
    wrapper_overhead_int16 = fitsio_int16 - int16_time

    print("fitsio wrapper overhead:")
    print(f"  uint8:  {wrapper_overhead_uint8:.4f}ms")
    print(f"  int16:  {wrapper_overhead_int16:.4f}ms")
    print(f"  Extra int16 overhead: {wrapper_overhead_int16 - wrapper_overhead_uint8:.4f}ms")
    print()

    if abs(wrapper_overhead_int16 - wrapper_overhead_uint8) < 0.05:
        print("✅ fitsio has consistent wrapper overhead across types")
    else:
        print("⚠️  fitsio has dtype-specific wrapper overhead")

    # Cleanup
    Path(uint8_file).unlink()
    Path(int16_file).unlink()

if __name__ == "__main__":
    main()
