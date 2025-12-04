# TorchFits: Benchmark & Functionality Comparison

This document provides a detailed comparison of `torchfits` against `astropy`, `fitsio`, and their PyTorch wrappers (`astropy_pytorch`, `fitsio_pytorch`). It highlights the performance advantages of `torchfits` and identifies missing functionality (e.g., WCS) to guide future development.

## 1. Functionality Comparison

| Feature | TorchFits | Fitsio | Astropy | Fitsio (PyTorch) | Astropy (PyTorch) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Core I/O** |
| Read Images | ✅ (Tensor) | ✅ (Numpy) | ✅ (Numpy) | ✅ (Tensor) | ✅ (Tensor) |
| Read Tables | ✅ (Dict[Tensor]) | ✅ (Numpy) | ✅ (Table) | ✅ (Dict[Tensor]) | ✅ (Dict[Tensor]) |
| Read Headers | ✅ (Dict) | ✅ (Dict) | ✅ (Header) | ✅ (Dict) | ✅ (Header) |
| Write Images | ✅ | ✅ | ✅ | ❌ | ❌ |
| Write Tables | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Advanced I/O** |
| Memory Mapping | ✅ (Native) | ✅ | ✅ | ❌ (Copy) | ❌ (Copy) |
| Compression (Read) | ✅ (Rice, Gzip, HComp) | ✅ | ✅ | ✅ | ✅ |
| Compression (Write) | ⚠️ (Rice only) | ✅ | ✅ | ❌ | ❌ |
| Multi-Extension FITS | ✅ | ✅ | ✅ | ✅ | ✅ |
| Variable Length Arrays | ✅ | ✅ | ✅ | ❌ | ❌ |
| Scaled Data (BSCALE) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Integration** |
| **Pure Torch Return** | ✅ (Direct) | ❌ | ❌ | ✅ (Converted) | ✅ (Converted) |
| **Device Support** | ✅ (CPU/CUDA) | ❌ | ❌ | ✅ (via copy) | ✅ (via copy) |
| **TorchFrame Support** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Missing / Limited** |
| **WCS Support** | ❌ (Header only) | ❌ | ✅ (Full) | ❌ | ❌ |
| **Coordinate Systems** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Detailed Validation** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **ASCII Tables** | ❌ | ✅ | ✅ | ❌ | ❌ |

### Key Gaps in TorchFits
1.  **WCS (World Coordinate System)**: `torchfits` currently parses headers into a dictionary but does not provide a WCS object for coordinate transformations (pixel <-> sky). Users must use `astropy.wcs` separately.
2.  **ASCII Tables**: Only Binary Tables (`XTENSION=BINTABLE`) are supported. ASCII tables (`XTENSION=TABLE`) are not supported.
3.  **Complex Compression Writing**: Only Rice compression is supported for writing. Gzip and HCompress are read-only.

## 2. Performance Comparison

Benchmarks were run on a high-performance system using the exhaustive `benchmark_all.py` suite.
**System**: MacBook Pro M1 Max (ARM64)

### Table Reading (100k rows, Mixed Types)
*The "Pure Torch" advantage is most visible here.*

| Method | Time (s) | Speedup (vs Fitsio) | Speedup (vs Astropy) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TorchFits (MMap)** | **0.000017** | **51.6x** | **54.8x** | Zero-copy, instant return |
| **TorchFits (Standard)** | 0.000224 | 3.9x | 4.1x | Full read |
| Fitsio | 0.000878 | 1.0x | 1.06x | Numpy return |
| Astropy | 0.000933 | 0.94x | 1.0x | Table object |
| Fitsio (PyTorch) | 0.000607 | 1.4x | 1.5x | Conversion overhead |

### Large Image Reading (4k x 4k Float64)
*Raw I/O throughput test.*

| Method | Time (s) | Speedup (vs Fitsio) | Speedup (vs Astropy) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TorchFits** | **0.001585** | **1.8x** | **2.6x** | Direct to Tensor (1D) |
| Fitsio | 0.002841 | 1.0x | 1.45x | Fast C I/O |
| Astropy | 0.004145 | 0.68x | 1.0x | Python overhead |

### Compressed Image Reading (Rice)
*Decompression performance.*

| Method | Time (s) | Speedup (vs Fitsio) | Speedup (vs Astropy) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TorchFits** | **0.005734** | **2.5x** | **8.0x** | Optimized decompression |
| Fitsio | 0.014174 | 1.0x | 3.2x | CFITSIO based |
| Astropy | 0.046224 | 0.3x | 1.0x | Python overhead |

### Scaled Data (Int16 -> Float32)
*On-the-fly scaling (BSCALE/BZERO).*

| Method | Time (s) | Speedup (vs Fitsio) | Speedup (vs Astropy) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TorchFits** | **0.02057** | **1.6x** | **2.2x** | SIMD scaling |
| Fitsio | 0.03233 | 1.0x | 1.4x | |
| Astropy | 0.04500 | 0.7x | 1.0x | |

## 3. Performance Weaknesses & Gaps

While `torchfits` excels at large data and tables, it has notable weaknesses:

### 1. Small File Overhead
For very small files (e.g., 64x64 images), `fitsio` is often faster due to lower Python/C++ dispatch overhead.
*   **Small Int16 2D**: `fitsio` (0.00027s) vs `torchfits` (0.00057s) -> **Fitsio is 2x faster**.
*   **Timeseries Frames**: `fitsio` consistently beats `torchfits` by ~20-30% for small frame iteration.

### 2. Uncompressed Large 2D Float64
In specific uncompressed large array scenarios, `fitsio`'s memory mapping or reading strategy outperforms `torchfits`.
*   **Large Float64 2D**: `fitsio` (0.009s) vs `torchfits` (0.021s) -> **Fitsio is 2.3x faster**.
    *   *Investigation needed*: Likely due to memory layout optimization or stride handling in `fitsio`.

### 3. Writing Performance
Writing performance has **not been benchmarked** exhaustively.
*   `torchfits` currently converts Tensors to CPU/NumPy before writing using `cfitsio`.
*   This likely incurs a copy overhead compared to `fitsio` which might write directly from numpy buffers.
*   **Gap**: No direct GPU-to-Disk writing support.

## 4. Summary & Recommendations

### When to use TorchFits
*   **Deep Learning Pipelines**: If you need data as PyTorch Tensors (especially on GPU), `torchfits` is the undisputed winner.
*   **Large Tables**: The `mmap=True` mode for tables provides instant access and minimal memory footprint (**50x speedup**), ideal for random access in large catalogs.
*   **High Throughput**: For batch processing of images (compressed or uncompressed), `torchfits` offers significant speedups.

### When to use Astropy
*   **WCS Transformations**: If you need to convert pixel coordinates to sky coordinates.
*   **ASCII Tables**: If you are dealing with legacy ASCII FITS tables.
*   **Complex Metadata**: If you need to modify complex headers or use standards that `torchfits` might not fully support yet.
*   **Validation**: If you need strict FITS standard compliance checking.

### When to use Fitsio
*   **General Numpy Work**: If you are working purely in the NumPy ecosystem and don't need Tensors.
*   **Small Files / Metadata Access**: `fitsio` has very low overhead for quick header peeking or small file reading.
*   **Writing**: Likely faster for writing standard FITS files from NumPy arrays.
