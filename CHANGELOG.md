# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-02-14

### Performance Improvements
- **Int32 Optimization**: Routed `int32` image reads through the efficient CFITSIO path, achieving a **1.45x** speedup (vs `fitsio`) and fixing a previous regression.
- **Intelligent Chunking**: Implemented 128MB chunking for large file reads. This prevents memory spikes during type conversion, maintaining high throughput for multi-GB files without OOM risks.
- **Table Reads**: Verified massive speedups (**~1500x**) for table operations using memory mapping.
- **No Regressions**: Confirmed 100% win rate against `fitsio` across 88 benchmark cases in the exhaustive suite.

### Documentation
- Updated `docs/benchmarks.md` with the latest exhaustive benchmark results (100% win rate).
- Updated `docs/performance_attempts.md` with successful optimization logs (Int32 routing, Chunking).

### Maintenance
- Cleaned up internal benchmark scripts and temporary logs.
- Updated CI workflows and pre-commit hooks.
