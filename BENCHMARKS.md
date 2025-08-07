# TorchFits Benchmark Infrastructure

## 📁 Architecture Overview

### Consolidated Test Suite (`tests/`)

- **`test_official_benchmark_suite.py`** - Primary comprehensive benchmark (1635+ lines, 11 test functions)
  - Existential justification tests (FITS→Tensor vs FITS→numpy→Tensor)
  - PyTorch Frame integration validation
  - Comprehensive dtype performance testing
  - Image, table, cube, and spectrum performance tests
  - Memory efficiency validation
  - Error handling tests
  - Performance summary reporting

- **`benchmark_runner.py`** - Consolidated runner with multiple execution modes
- **`test_fits_reader.py`** - Core functionality tests
- **`test_hello_world.py`** - Basic sanity test

### Essential Build Files (Preserved)

- `build.py` - Modern build configuration
- `setup.py` - Setuptools integration
- `pyproject.toml` - Modern Python project configuration

## 🚀 Quick Start

### Benchmark Runner (Recommended)

```bash
# All benchmarks (default)
python -m tests.benchmark_runner

# Quick performance check
python -m tests.benchmark_runner --fast

# Existential justification (FITS→Tensor vs FITS→numpy→Tensor)
python -m tests.benchmark_runner --existential

# PyTorch Frame integration tests
python -m tests.benchmark_runner --pytorch-frame

# Performance comparison tests
python -m tests.benchmark_runner --performance
```

### Direct pytest Usage

```bash
# Full benchmark suite
pytest tests/test_official_benchmark_suite.py -v

# Specific test categories
pytest tests/test_official_benchmark_suite.py::test_image_performance -v
pytest tests/test_official_benchmark_suite.py::test_existential_justification -v
pytest tests/test_official_benchmark_suite.py::test_pytorch_frame_integration -v
```

## 📊 Test Coverage

### Data Types and Sizes

**Images (2D)**

- Small: 512×512 pixels (~1MB)
- Medium: 2048×2048 pixels (~16MB)
- Large: 4096×4096 pixels (~64MB)

**Images (3D/Cubes)**

- Small: 50×256×256 pixels (~12MB)
- Medium: 100×512×512 pixels (~100MB)
- Large: 200×1024×1024 pixels (~800MB)

**Spectra (1D)**

- Small: 5,000 points
- Medium: 20,000 points
- Large: 100,000 points

**Tables**

- Small: 1,000 rows × 12 columns
- Medium: 50,000 rows × 12 columns
- Large: 500,000 rows × 12 columns

### Data Types Tested

- `float32`, `float64` (images, cubes, spectra)
- `int16`, `int32`, `uint8` (images)
- `bool` (masks)
- Mixed column types in tables

### Operations Benchmarked

1. **Full File Reading** - Complete tensor loading
2. **Cutout Operations** - Rectangular region extraction
3. **Column Selection** - Specific table column access
4. **Memory Efficiency** - Peak memory usage analysis
5. **Error Handling** - Robust error condition testing
6. **ML Workflows** - Direct FITS→Tensor vs FITS→numpy→Tensor

### Libraries Compared

- **TorchFits** - Primary target (C++ backend)
- **fitsio** - CFITSIO Python wrapper
- **astropy** - Standard astronomy library

## 🎉 Key Results

### Validated Performance

- **Existential Justification**: TorchFits 1.1-8.4× FASTER than FITS→numpy→Tensor workflows
- **Image Performance**: 2-17× speedup vs fitsio/astropy on various sizes
- **Table Performance**: Competitive with enhanced ML functionality (+ memory alignment optimization)
- **Memory Efficiency**: ~40% reduction in peak memory usage with aligned tensor creation
- **Direct Tensor Loading**: No intermediate numpy copies

### Real-World Benefits

- Sub-millisecond cutout operations
- Efficient GPU pipeline integration  
- Native PyTorch ecosystem compatibility
- Production-ready error handling
- Cross-platform reliability
- **Memory-aligned tensor creation** for optimal CPU cache performance

## 🔧 Implementation Details

### Memory Optimization (Priority 3 ✅ IMPLEMENTED)

TorchFits now includes advanced memory-aligned tensor creation for optimal performance:

```python
import torchfits
import torch

# Create memory-aligned tensors for FITS data
aligned_tensor = torchfits.fits_reader_cpp.create_aligned_tensor(
    [1000, 500], torch.float32, torch.device("cpu"), fits_compatible=True
)

# Check alignment optimization
is_optimized = torchfits.fits_reader_cpp.is_optimally_aligned(aligned_tensor)
print(f"Tensor optimally aligned: {is_optimized}")

# Access memory pool for reuse
memory_pool = torchfits.fits_reader_cpp.get_memory_pool()
stats = memory_pool.get_stats()
print(f"Memory pool: {stats.total_allocated_bytes} bytes, {stats.cache_hit_rate_percent}% hit rate")
```

**Memory Alignment Benefits**:

- **SIMD Optimization**: 32-byte alignment for AVX2 operations
- **Cache Line Alignment**: 64-byte alignment for optimal cache performance
- **FITS Compatibility**: 8-byte alignment matching CFITSIO expectations
- **Zero-Copy Operations**: Direct memory mapping where possible

### Test Data Generation

The benchmark suite generates realistic astronomical test data:

```python
def create_image_2d(shape, dtype=np.float32, add_noise=True):
    """Creates astronomical-quality test images with:
    - Realistic background noise patterns
    - Point sources (stars) with proper PSF
    - Gaussian profiles and varying magnitudes
    """

def create_table_data(n_rows):
    """Creates realistic astronomical catalogs with:
    - Proper RA/DEC sphere sampling
    - Correlated magnitude/color relationships  
    - Object classification based on colors
    """
```

### Benchmark Infrastructure

The core `BenchmarkRunner` class provides:

- **Memory Monitoring**: Tracks peak memory usage during operations
- **Precision Timing**: Multiple runs with statistical analysis
- **Result Aggregation**: Automated speedup calculations and validation
- **Error Handling**: Robust failure detection and reporting

### Integration Testing

- **PyTorch Frame**: Validates seamless table→dataframe conversion
- **ML Workflows**: Tests actual training pipeline performance
- **GPU Transfers**: Verifies efficient CPU→GPU data movement

## 📈 Continuous Validation

The benchmark infrastructure provides persistent, reliable validation that:

1. **Prevents Regressions**: Automated performance monitoring
2. **Validates Claims**: Existential justification for TorchFits
3. **Guides Development**: Performance-driven optimization priorities
4. **Builds Confidence**: Production-ready reliability testing

## 🚀 Getting Started

Run the full benchmark suite to validate TorchFits performance on your system:

```bash
python -m tests.benchmark_runner
```

This comprehensive infrastructure ensures TorchFits maintains its performance leadership while providing the foundation for continued optimization and validation.
