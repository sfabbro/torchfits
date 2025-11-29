#!/usr/bin/env python3
"""
Detailed profiling of int16 read path to identify remaining overhead.
Breaks down the read operation into individual components.
"""
import time
import tempfile
import statistics
from pathlib import Path
import numpy as np
from astropy.io import fits
import torchfits
from torchfits import cpp

def create_test_file(dtype_str):
    """Create a test FITS file."""
    tmpdir = Path(tempfile.gettempdir())
    filepath = tmpdir / f"profile_{dtype_str}_{time.time_ns()}.fits"

    if dtype_str == 'uint8':
        data = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    elif dtype_str == 'int16':
        data = np.random.randint(-32768, 32767, (1000, 1000), dtype=np.int16)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    fits.writeto(filepath, data, overwrite=True)
    return str(filepath)

def profile_component(func, iterations=100):
    """Profile a single component."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)

def main():
    print("=" * 80)
    print("DETAILED INT16 PROFILING")
    print("=" * 80)
    print()

    # Create test files
    print("Creating test files...")
    uint8_file = create_test_file('uint8')
    int16_file = create_test_file('int16')
    print()

    # Profile component by component
    print("Profiling individual components (100 iterations each):")
    print("-" * 80)

    # 1. File open/close overhead
    print("\n1. File open/close overhead:")
    uint8_open = profile_component(lambda: (h := cpp.open_fits_file(uint8_file, 'r'), cpp.close_fits_file(h)))
    int16_open = profile_component(lambda: (h := cpp.open_fits_file(int16_file, 'r'), cpp.close_fits_file(h)))
    print(f"   uint8:  {uint8_open:.4f}ms")
    print(f"   int16:  {int16_open:.4f}ms")
    print(f"   Ratio:  {int16_open/uint8_open:.2f}x")

    # 2. Read with pre-opened handle (no file open overhead)
    print("\n2. Read operation (with pre-opened handle):")
    h_uint8 = cpp.open_fits_file(uint8_file, 'r')
    h_int16 = cpp.open_fits_file(int16_file, 'r')

    uint8_read = profile_component(lambda: cpp.read_full(h_uint8, 0))
    int16_read = profile_component(lambda: cpp.read_full(h_int16, 0))

    print(f"   uint8:  {uint8_read:.4f}ms")
    print(f"   int16:  {int16_read:.4f}ms")
    print(f"   Ratio:  {int16_read/uint8_read:.2f}x")

    cpp.close_fits_file(h_uint8)
    cpp.close_fits_file(h_int16)

    # 3. Full read path (open + read + close)
    print("\n3. Full read path (open + read + close):")
    def full_read_uint8():
        h = cpp.open_fits_file(uint8_file, 'r')
        t = cpp.read_full(h, 0)
        cpp.close_fits_file(h)

    def full_read_int16():
        h = cpp.open_fits_file(int16_file, 'r')
        t = cpp.read_full(h, 0)
        cpp.close_fits_file(h)

    uint8_full = profile_component(full_read_uint8)
    int16_full = profile_component(full_read_int16)

    print(f"   uint8:  {uint8_full:.4f}ms")
    print(f"   int16:  {int16_full:.4f}ms")
    print(f"   Ratio:  {int16_full/uint8_full:.2f}x")

    # 4. Python wrapper overhead (torchfits.read vs cpp.read_full)
    print("\n4. Python wrapper overhead:")
    torchfits.clear_file_cache()

    def wrapper_uint8():
        torchfits.clear_file_cache()
        data, hdr = torchfits.read(uint8_file)

    def wrapper_int16():
        torchfits.clear_file_cache()
        data, hdr = torchfits.read(int16_file)

    uint8_wrapper = profile_component(wrapper_uint8, iterations=50)
    int16_wrapper = profile_component(wrapper_int16, iterations=50)

    print(f"   uint8:  {uint8_wrapper:.4f}ms")
    print(f"   int16:  {int16_wrapper:.4f}ms")
    print(f"   Ratio:  {int16_wrapper/uint8_wrapper:.2f}x")

    # 5. Compare with fitsio
    print("\n5. Comparison with fitsio:")
    import fitsio

    uint8_fitsio = profile_component(lambda: fitsio.read(uint8_file))
    int16_fitsio = profile_component(lambda: fitsio.read(int16_file))

    print(f"   uint8:  {uint8_fitsio:.4f}ms")
    print(f"   int16:  {int16_fitsio:.4f}ms")
    print(f"   Ratio:  {int16_fitsio/uint8_fitsio:.2f}x (fitsio baseline)")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\nFile open/close overhead:")
    print(f"  uint8: {uint8_open:.4f}ms, int16: {int16_open:.4f}ms ({int16_open/uint8_open:.2f}x)")

    print(f"\nPure read operation (handle reuse):")
    print(f"  uint8: {uint8_read:.4f}ms, int16: {int16_read:.4f}ms ({int16_read/uint8_read:.2f}x)")
    print(f"  This is the CFITSIO-level overhead: {int16_read/uint8_read:.2f}x")

    print(f"\nFull C++ path:")
    print(f"  uint8: {uint8_full:.4f}ms, int16: {int16_full:.4f}ms ({int16_full/uint8_full:.2f}x)")

    print(f"\nPython wrapper (torchfits.read):")
    print(f"  uint8: {uint8_wrapper:.4f}ms, int16: {int16_wrapper:.4f}ms ({int16_wrapper/uint8_wrapper:.2f}x)")
    print(f"  Wrapper overhead vs C++: uint8={uint8_wrapper-uint8_full:.4f}ms, int16={int16_wrapper-int16_full:.4f}ms")

    print(f"\nfitsio (reference):")
    print(f"  uint8: {uint8_fitsio:.4f}ms, int16: {int16_fitsio:.4f}ms ({int16_fitsio/uint8_fitsio:.2f}x)")

    print(f"\nPerformance comparison:")
    print(f"  torchfits vs fitsio (uint8): {uint8_wrapper/uint8_fitsio:.2f}x")
    print(f"  torchfits vs fitsio (int16): {int16_wrapper/int16_fitsio:.2f}x")

    print(f"\nWhere we're losing performance (int16):")
    cfitsio_overhead = int16_read / uint8_read
    fitsio_cfitsio_overhead = int16_fitsio / uint8_fitsio
    our_extra = cfitsio_overhead - fitsio_cfitsio_overhead

    print(f"  CFITSIO baseline (our measurements): {cfitsio_overhead:.2f}x")
    print(f"  CFITSIO baseline (fitsio): {fitsio_cfitsio_overhead:.2f}x")
    print(f"  Extra overhead in our C++ layer: {our_extra:.2f}x ({our_extra/cfitsio_overhead*100:.1f}% of total)")

    # Cleanup
    Path(uint8_file).unlink()
    Path(int16_file).unlink()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if our_extra < 0.1:
        print("✅ Our C++ layer has minimal overhead")
        print("   The remaining difference is purely CFITSIO's int16 handling")
        print()
        print("Next steps:")
        print("  1. Profile CFITSIO internals with Instruments")
        print("  2. Consider SIMD byte swapping")
        print("  3. Explore custom mmap-based reader for uncompressed data")
    else:
        print("⚠️  Our C++ layer has measurable overhead")
        print(f"   Extra overhead: {our_extra:.2f}x ({our_extra/cfitsio_overhead*100:.1f}% of total)")
        print()
        print("Next steps:")
        print("  1. Profile the C++ read path to find the bottleneck")
        print("  2. Check tensor allocation/memory patterns")
        print("  3. Verify THPVariable_Wrap performance for int16")

if __name__ == "__main__":
    main()
