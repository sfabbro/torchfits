# torchfits Benchmarks

This directory contains performance benchmarks for `torchfits`.

## Running Benchmarks

Benchmarks are managed via `pixi` tasks. You can see all available benchmark tasks in `pixi.toml` or by running `pixi task list`.

### Basic Benchmarks

Run basic I/O and WCS benchmarks:

```bash
pixi run bench-basic
```

### Comprehensive Benchmarks

Run the full suite of benchmarks (may take several minutes):

```bash
pixi run bench-all
```

Use benchmark profiles:

```bash
pixi run bench-all -- --profile user  # representative defaults (cache on)
pixi run bench-all -- --profile lab   # stricter cold-I/O defaults
```

To run detailed benchmarks for specific components:

```bash
pixi run bench-core       # Core I/O
pixi run bench-table      # Table operations
pixi run bench-transforms # Image transforms (GPU/CPU)
pixi run bench-buffer     # Buffer performance
pixi run bench-cache      # Caching strategies
```

## Methodology

Benchmarks measure:
1.  **Cold Start**: First access time (no caching).
2.  **Warm Acccess**: Subsequent access time (L1/L2 cache hits).
3.  **Throughput**: MB/s for large file operations.
4.  **CPU/GPU Transfer**: Time to move stats between host and device.

Results are compared against:
- `astropy.io.fits`
- `fitsio` (python wrapper for cfitsio)

## Files

- `benchmark_all.py`: Master runner for all benchmarks.
- `benchmark_basic.py`: Simple read/write and WCS timing.
- `benchmark_core.py`: Detailed core I/O performance analysis.
- `benchmark_table.py`: Table reading/writing performance.
- `benchmark_transforms.py`: GPU-accelerated transform benchmarks.
- `benchmark_*.py`: Specialized micro-benchmarks for specific features.
