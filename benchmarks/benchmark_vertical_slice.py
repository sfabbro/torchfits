import time
import numpy as np
import torch
import fitsio
import torchfits
import os

def create_test_file(filename, size=(4096, 4096)):
    data = np.random.randint(-32768, 32767, size=size, dtype=np.int16)
    if os.path.exists(filename):
        os.remove(filename)
    fitsio.write(filename, data)
    return data

def benchmark():
    filename = "test_int16.fits"
    size = (4096, 4096)
    print(f"Creating test file {filename} with size {size}...")
    data = create_test_file(filename, size)
    
    # Warmup
    print("Warming up...")
    fitsio.read(filename)
    torchfits.read_image_fast_int16(filename)
    
    n_iters = 20
    
    # Benchmark fitsio (to float tensor)
    start = time.time()
    for _ in range(n_iters):
        arr = fitsio.read(filename)
        # Convert to float tensor (common use case)
        _ = torch.from_numpy(arr.astype(np.float32))
    end = time.time()
    fitsio_time = (end - start) / n_iters
    print(f"fitsio (to float tensor): {fitsio_time*1000:.2f} ms")
    
    # Benchmark torchfits fast
    start = time.time()
    for _ in range(n_iters):
        _ = torchfits.read_image_fast_int16(filename)
    end = time.time()
    torchfits_time = (end - start) / n_iters
    print(f"torchfits (fast): {torchfits_time*1000:.2f} ms")
    
    speedup = fitsio_time / torchfits_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify correctness
    ref = fitsio.read(filename)
    res = torchfits.read_image_fast_int16(filename).numpy()
    
    if np.allclose(ref.astype(np.float32), res):
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")
        print("Max diff:", np.max(np.abs(ref.astype(np.float32) - res)))
        
    # Verify BSCALE/BZERO
    print("Verifying BSCALE/BZERO...")
    filename_scaled = "test_int16_scaled.fits"
    if os.path.exists(filename_scaled):
        os.remove(filename_scaled)
    
    # Write raw data, but tell reader to scale it
    # We write int16 data. We want it to be interpreted as val * 2.0 + 10.0
    fitsio.write(filename_scaled, data, header={'BSCALE': 2.0, 'BZERO': 10.0}, clobber=True)
    
    # fitsio read applies scaling by default
    ref_scaled = fitsio.read(filename_scaled)
    res_scaled = torchfits.read_image_fast_int16(filename_scaled).numpy()
    
    if np.allclose(ref_scaled, res_scaled):
        print("Verification (Scaled): PASSED")
    else:
        print("Verification (Scaled): FAILED")
        print("Max diff:", np.max(np.abs(ref_scaled - res_scaled)))
        print("Ref sample:", ref_scaled[0,0])
        print("Res sample:", res_scaled[0,0])

    os.remove(filename)
    if os.path.exists(filename_scaled):
        os.remove(filename_scaled)

if __name__ == "__main__":
    benchmark()
