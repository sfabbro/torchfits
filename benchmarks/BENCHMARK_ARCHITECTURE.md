# torchfits Comprehensive Benchmark Suite

This document describes the final comprehensive benchmark architecture for torchfits.

## Overview

The benchmark suite provides two tiers of testing:

1. **Core Benchmarks** - Essential development validation
2. **Exhaustive Benchmarks** - Comprehensive performance analysis

## Architecture

### Core Benchmarks (Fast CI/CD)

- **benchmark_basic.py** - Smoke tests and basic functionality
- **benchmark_core.py** - Phase 1 validation features  
- **benchmark_ml.py** - ML workflow and PyTorch integration
- **benchmark_table.py** - Table operations and pytorch-frame integration

### Exhaustive Benchmark Suite

- **benchmark_exhaustive.py** - Complete comprehensive testing

#### Test Coverage

The exhaustive suite covers:

**Data Types:**
- int8, int16, int32, float32, float64

**Dimensions:**
- 1D (spectra): 1K, 10K, 100K, 1M elements
- 2D (images): 64x64, 256x256, 1024x1024, 2048x2048
- 3D (cubes): 5x32x32, 10x128x128, 25x256x256, 50x512x512

**File Types:**
- Single HDU files (all data type/size combinations)
- MEF files (Multiple Extension FITS)
- Multi-MEF files (10+ extensions)
- Table files (1K, 10K, 100K rows)
- Scaled data files (BSCALE/BZERO)
- WCS-enabled files
- Compressed files (RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1)
- Multi-file collections (time series)

**Comparisons:**
- torchfits vs astropy.io.fits
- torchfits vs fitsio
- Memory usage analysis
- PyTorch conversion performance

#### Output

The exhaustive suite generates:

**CSV Data:**
- `benchmark_results/exhaustive_results.csv` - Detailed performance metrics

**Plots:**
- `performance_by_type.png` - Performance comparison by file type
- `memory_usage.png` - Memory usage analysis
- `speedup_analysis.png` - Speedup comparison vs other methods
- `data_type_performance.png` - Performance heatmap by data type/dimensions
- `size_performance.png` - Performance vs file size scatter plot
- `compression_analysis.png` - Compression performance analysis

**Summary:**
- `exhaustive_summary.md` - Comprehensive markdown report with findings and recommendations

## Usage

### pixi Commands

```bash
# Core benchmarks (fast, for CI/CD)
pixi run bench-basic      # Basic smoke tests
pixi run bench-core       # Core feature validation
pixi run bench-ml         # ML workflow tests
pixi run bench-table      # Table operation tests

# Run all core benchmarks
pixi run bench-all

# Comprehensive benchmarks (slow, detailed analysis)
pixi run bench-exhaustive           # Full exhaustive suite with cleanup
pixi run bench-exhaustive-keep      # Keep temporary files for debugging
pixi run bench-comprehensive        # Alias for exhaustive with custom output

# Interactive orchestrator
pixi run bench-all                  # Runs core + asks about comprehensive
```

### Environment Requirements

**Core benchmarks:**
- Only require base dependencies (python, pytorch, numpy)

**Exhaustive benchmarks:**
- Require `bench` feature: `pixi install -e bench`
- Optional dependencies: astropy, fitsio, matplotlib, seaborn, pandas, psutil

## Design Principles

1. **Two-tier approach** - Fast core tests for CI/CD, comprehensive analysis when needed
2. **Rich reporting** - CSV data, plots, and markdown summaries
3. **Memory monitoring** - Track memory usage with tracemalloc
4. **Statistical rigor** - Multiple runs with mean/std reporting
5. **Comparison fairness** - Test all methods with same data and conditions
6. **Practical scenarios** - Test real-world FITS file types and operations

## Dependencies

### Core
```toml
python = ">=3.11"
pytorch = "2.8.0"
numpy = ">=1.24"
```

### Benchmark Feature
```toml
astropy = ">=5.0"
fitsio = ">=1.0"
matplotlib = "*"
seaborn = "*"
pandas = "*"
psutil = "*"
pytest-benchmark = "*"
```

## Integration

The benchmark suite integrates with:
- **pixi tasks** - Easy command-line execution
- **CI/CD** - Fast core benchmarks for continuous testing
- **Development workflow** - Performance validation during development
- **Performance analysis** - Detailed reporting for optimization

## File Organization

```
benchmarks/
├── benchmark_basic.py          # Core: Smoke tests
├── benchmark_core.py           # Core: Phase 1 validation
├── benchmark_ml.py             # Core: ML workflows  
├── benchmark_table.py          # Core: Table operations
├── benchmark_exhaustive.py     # Comprehensive: Full suite
└── run_all_benchmarks.py       # Orchestrator
```

This architecture provides both fast essential testing and comprehensive analysis capabilities while maintaining clear separation of concerns and easy usability.
