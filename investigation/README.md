# Investigation Files

This directory contains all files generated during the int16 performance investigation (November 2024).

## Structure

- `reports/` - Markdown documentation of investigation phases and findings
- `benchmarks/` - Python benchmark scripts and results
- `tests/` - Test scripts (Python and C/C++) and test data files
- `binaries/` - Compiled C/C++ test programs
- `baseline_results/` - Baseline benchmark results for validation

## Main Report

See `../COMPREHENSIVE_INVESTIGATION_REPORT.md` for the complete technical documentation of all investigation phases, tests, benchmarks, profiling, and findings.

## Key Files

### Reports
- Phase-by-phase analysis documenting the investigation progress
- Optimization summaries and performance analysis
- Final diagnosis and bottleneck identification

### Benchmarks
- Component breakdown profiling
- Pure CFITSIO baseline benchmarks
- Comparison with fitsio library
- Comprehensive performance testing

### Tests
- Minimal C++ tests proving torch::Tensor memory causes slowdown
- CFITSIO file handle reuse tests
- Various C/C++ source files used for isolated testing

### Binaries
- Compiled versions of C/C++ test programs
- Can be recompiled from source files in `tests/` directory
