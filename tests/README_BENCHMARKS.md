# TorchFits Official Benchmark Suite

This directory contains the comprehensive benchmark suite for validating TorchFits performance claims against fitsio and astropy.

## Files

- `test_official_benchmark_suite.py` - Main pytest-compatible benchmark test suite
- `run_benchmarks.py` - Standalone runner script for quick benchmarking

## Usage

### Running with pytest (Recommended)

Run the full benchmark suite:
```bash
pytest tests/test_official_benchmark_suite.py -v
```

Run specific test categories:
```bash
# Image performance tests
pytest tests/test_official_benchmark_suite.py::test_image_performance -v

# Table performance tests  
pytest tests/test_official_benchmark_suite.py::test_table_performance -v

# Cutout performance tests
pytest tests/test_official_benchmark_suite.py::test_cutout_performance -v

# Column selection tests
pytest tests/test_official_benchmark_suite.py::test_column_selection_performance -v

# Memory efficiency tests
pytest tests/test_official_benchmark_suite.py::test_memory_efficiency -v

# Performance summary
pytest tests/test_official_benchmark_suite.py::test_performance_summary -v
```

### Running with standalone script

Quick benchmark (small files only):
```bash
python run_benchmarks.py --quick
```

Full benchmark suite:
```bash
python run_benchmarks.py --output-dir ./my_benchmark_results
```

## Test Coverage

### File Types and Sizes

The benchmark suite tests the following FITS file types and sizes:

**Images (2D)**
- Small: 512x512 pixels (~1MB)
- Medium: 2048x2048 pixels (~16MB)
- Large: 4096x4096 pixels (~64MB)
- Huge: 8192x8192 pixels (~256MB)

**Images (3D/Cubes)**
- Small: 50x256x256 pixels (~12MB)
- Medium: 100x512x512 pixels (~100MB)
- Large: 200x1024x1024 pixels (~800MB)

**Spectra (1D)**
- Small: 5,000 points
- Medium: 20,000 points
- Large: 100,000 points
- Huge: 500,000 points

**Tables**
- Small: 1,000 rows x 12 columns
- Medium: 50,000 rows x 12 columns
- Large: 500,000 rows x 12 columns
- Huge: 2,000,000 rows x 12 columns

### Operations Tested

1. **Full File Reading**
   - Complete file read for all data types
   - Memory usage monitoring
   - Throughput calculation

2. **Cutout Operations**
   - 2D image cutouts (central 50% region)
   - 3D cube cutouts (central 50% region)
   - Performance target: <100ms

3. **Column Selection** (Tables only)
   - Read specific columns: ['RA', 'DEC', 'MAG_G']
   - Read flux columns: ['FLUX_G', 'FLUX_R', 'FLUX_I']
   - Performance target: <50ms

4. **Memory Efficiency**
   - Peak memory usage comparison
   - Memory-to-file-size ratios

### Performance Targets

The benchmark validates these performance targets:

**Images**: 2-17x faster than fitsio/astropy
- Small images: >1.8x speedup
- Medium images: >2.5x speedup  
- Large images: >3x speedup

**Tables**: 0.8-5x faster than fitsio/astropy
- Competitive performance with enhanced functionality
- Sub-millisecond column operations

**Cutouts**: Sub-100ms for typical operations
- Fast random access to image regions
- Efficient memory usage

**Memory**: <2x memory usage vs competitors
- Efficient tensor allocation
- Minimal memory fragmentation

## Dependencies

Required:
- `torch` - PyTorch tensors
- `numpy` - Numerical arrays
- `pytest` - Test framework
- `psutil` - Memory monitoring

Optional (for comparison):
- `fitsio` - CFITSIO Python wrapper
- `astropy` - Astronomy Python library

At least one of fitsio or astropy is required for baseline comparisons.

## Output

The benchmark suite generates:

1. **Console output** - Real-time progress and results
2. **JSON results** - Detailed benchmark data (`benchmark_results.json`)
3. **Text report** - Summary report (`benchmark_report.txt`)

### Example Output

```
TORCHFITS COMPREHENSIVE BENCHMARK REPORT
================================================================================

Library Availability:
  TorchFits: ✓
  fitsio:    ✓
  astropy:   ✓

Overall Statistics:
  Total tests: 24
  Successful: 22
  Failed: 2
  Success rate: 91.7%

TorchFits Performance:
  image_2d_read_full: 12.3ms (avg 8.5x speedup)
  table_read_full: 45.2ms (avg 2.1x speedup)
  image_2d_read_cutout: 2.1ms
  table_read_columns: 0.8ms

Performance Targets:
  ✓ Images (target: 2-17x): 8.5x average
  ✓ Tables (target: 0.8-5x): 2.1x average
  ✓ Cutouts (target: <100ms): 2.1ms average
```

## Integration with CI/CD

The pytest-based suite can be easily integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run TorchFits Benchmarks
  run: |
    pytest tests/test_official_benchmark_suite.py::test_image_performance -v
    pytest tests/test_official_benchmark_suite.py::test_table_performance -v
```

## Customization

### Adding New Test Cases

To add new benchmark scenarios, modify `test_official_benchmark_suite.py`:

1. Add new file configurations in `FITSTestDataGenerator.create_test_file()`
2. Add new test functions with `@pytest.mark.parametrize`
3. Update performance targets in validation functions

### Custom Performance Targets

Modify the target validation in `test_performance_summary()` to adjust expected performance levels.

## Troubleshooting

**Import Errors**: Ensure torchfits is installed or add `src/` to Python path
**File Creation Errors**: Requires write permissions in test directory
**Memory Errors**: Large test files may require significant RAM
**Missing Libraries**: Install fitsio and/or astropy for comparison baselines

For detailed error information, run with verbose output:
```bash
pytest tests/test_official_benchmark_suite.py -v -s
```
