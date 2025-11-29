#!/usr/bin/env python3
"""Test DLPack conversion overhead."""
import torch
import time
import numpy as np

# Create test tensor
data = np.random.randn(2000, 2000).astype(np.float32)

print("Testing DLPack conversion overhead")
print("=" * 60)

# Test 1: DLPack round-trip
times = []
for _ in range(100):
    tensor = torch.from_numpy(data)
    start = time.perf_counter()

    # Simulate what our C++ code does
    capsule = torch.utils.dlpack.to_dlpack(tensor)
    result = torch.utils.dlpack.from_dlpack(capsule)

    elapsed = time.perf_counter() - start
    times.append(elapsed)

dlpack_time = np.median(times) * 1000000  # microseconds

# Test 2: Direct numpy->torch (baseline)
times = []
for _ in range(100):
    start = time.perf_counter()
    tensor = torch.from_numpy(data)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

direct_time = np.median(times) * 1000000  # microseconds

print(f"DLPack round-trip:  {dlpack_time:.1f}μs")
print(f"Direct numpy->torch: {direct_time:.1f}μs")
print(f"DLPack overhead:     {dlpack_time - direct_time:.1f}μs ({(dlpack_time/direct_time):.1f}x slower)")
print()
print(f"For a 4MB file taking ~1ms to read:")
print(f"  - DLPack adds {dlpack_time/1000:.2f}ms ({dlpack_time/1000/1.0*100:.1f}% overhead)")
