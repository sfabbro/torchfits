#!/usr/bin/env python3
"""
Test if DLPack conversion has dtype-specific overhead.
"""
import torch
import time
import statistics

def test_dlpack_conversion(dtype, shape, iterations=1000):
    """Test torch.to_dlpack + torch.from_dlpack overhead."""
    times = []

    for _ in range(iterations):
        # Create tensor
        tensor = torch.zeros(shape, dtype=dtype)

        # Measure DLPack round-trip
        start = time.perf_counter()
        capsule = torch.utils.dlpack.to_dlpack(tensor)
        result = torch.utils.dlpack.from_dlpack(capsule)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times) * 1e6  # Convert to microseconds

def main():
    shape = (1000, 1000)
    print(f"Testing DLPack conversion overhead ({shape[0]}x{shape[1]} tensors)...")
    print()

    uint8_time = test_dlpack_conversion(torch.uint8, shape)
    int16_time = test_dlpack_conversion(torch.int16, shape)
    int32_time = test_dlpack_conversion(torch.int32, shape)
    float32_time = test_dlpack_conversion(torch.float32, shape)

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

    if int16_time > uint8_time * 1.5:
        print(f"❌ DLPack has int16-specific overhead ({int16_time/uint8_time:.2f}x)")
    else:
        print(f"✅ DLPack overhead is similar across types")

if __name__ == "__main__":
    main()
