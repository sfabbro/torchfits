# Pre-extraction FITS Benchmark Audit — 2026-06-26

## Outcome

The current `torchfits` build completed the restored full benchmark command
with no runtime failures after remediation. It was first in every scorecard
case:

| Domain | Family | Scorecard cases | torchfits wins | Deficits |
|---|---|---:|---:|---:|
| FITS images | smart | 84 | 84 | 0 |
| FITS images | specialized | 84 | 84 | 0 |
| FITS tables | smart | 70 | 70 | 0 |
| FITS tables | specialized | 70 | 70 | 0 |

The definitive maintained-suite run produced 1,197 normalized rows: 747 image
rows and 450 table rows.

The skipped rows are intentional benchmark-policy exclusions, primarily fitsio
rows whose mmap mode cannot be controlled independently and the disabled
compressed-table placeholder. They are not execution failures.

The complete correctness suite passes: **336 passed, 10 skipped**.

## Benchmark source and method

The exact pre-extraction FITS image and table harness was recovered from commit
`fd6d58c` (`e9a9a48^`) and run with the current editable `torchfits` build.
WCS and sphere lanes were not run because those domains moved to `torchsky` and
are not FITS I/O benchmarks.

The image run covered the historical 81 generated files plus two 100×100
cutout workflows and one 200-read random-HDU workflow. To make the full
remediation run practical, image timings used three measured samples per
method. The table harness retained its historical 3–9 samples by table size.

Environment:

- macOS 14.8.2 arm64
- Python 3.13.14
- PyTorch 2.10.0
- NumPy 2.5.0
- Astropy 7.2.0
- fitsio 1.3.0
- torchfits 0.3.2 editable build

Local raw evidence is generated under
`benchmarks_results/20260626_postfix_full_zero_deficit/`. Benchmark artifacts
are intentionally gitignored; this report records the durable scorecard and
environment.

## Comparison with the published pre-extraction snapshot

The old release snapshot was generated on 2026-03-18 with a different
dependency state and more image repetitions. The following comparison is a
trend check, not a statistically controlled before/after experiment.

| Representative workflow | Old snapshot | Current |
|---|---:|---:|
| tiny int16 1D image | 73 µs | 79 µs |
| small float64 2D image | 145 µs | 138 µs |
| medium int16 1D image | 125 µs | 129 µs |
| large int16 2D image | 1.3 ms | 1.39 ms |
| large float64 2D image | 6.0 ms | 6.08 ms |
| Rice compressed image | 6.6 ms | 6.98 ms |
| GZIP-2 compressed image | 7.7 ms | 7.74 ms |
| HCOMPRESS image | 21.9 ms | 22.03 ms |
| narrow 1M-row table read | 131 µs | 70 µs |
| mixed 1M-row table read | 78 µs | 78 µs |
| mixed 100K-row table slice | 61 µs | 64 µs |
| mixed 100K-row scan count | 142 µs | 143 µs |
| narrow 1K-row predicate | 1.3 ms | 426 µs |

Large and compressed image timings are close to the published snapshot.
Several tiny-image timings are noisier and slower, while high-row-count table
reads and predicate filtering improved. No current comparator beat torchfits
in a comparable group.

## Issues found and fixed

1. **Native abort in streamed table reads.** A `TableReader` created from a
   caller-owned CFITSIO handle closed that handle in its destructor, and the
   Python owner then closed it again. External handles are now borrowed; cached
   path handles still release their cache reference.

2. **Corrupt complex mmap columns.** Complex values advanced by packed element
   width instead of the FITS row stride. Reads now use `NAXIS1` row stride and
   each column's physical byte offset.

3. **Incorrect mmap bit columns.** FITS `X` columns were treated as unpacked
   bytes and returned uninitialized values. They are now decoded from packed,
   most-significant-bit-first storage into boolean tensors.

4. **Incorrect physical offsets for packed and variable columns.** Table layout
   now accounts for `ceil(bits/8)` for `X`, 8-byte `P` descriptors, and 16-byte
   `Q` descriptors. The declared `NAXIS1` value is used for row stride.

5. **Historical row-slice benchmark failures.** Extraction dropped the
   `policy` argument from `read_table_rows`, although the underlying table API
   still supports it. The wrapper and regression test now preserve it.

6. **The current “full” image suite had silently shrunk.** It generated only
   five files after extraction. The current runner now restores the historical
   FITS-only 81-file / 84-workflow matrix, including cutouts and random-HDU
   access. Explicit cardinality guards fail the run if coverage shrinks.

7. **Whitespace gate failure.** A trailing-whitespace defect in
   `compression.cpp` was removed.

8. **Stale native benchmark build.** The `bench-all` Pixi environment had an
   older editable extension than the default development environment. Primary
   benchmark tasks now rebuild the editable package in the target environment
   before running.

9. **Cross-domain native-state contamination.** Image and table domains now run
   in separate processes when invoked together. This prevents allocator,
   CFITSIO, and extension-global state from affecting the following domain.

10. **Benchmark-policy regressions.** The user-profile image suite again uses
    torchfits' public `mmap="auto"` behavior, and row-count workflows read
    `NAXIS2` instead of materializing an HDU list.

## Remaining benchmark limitations

- This run proves current cross-library leadership and broad regression
  coverage, but the three-sample image run is not noise-equivalent to the old
  release snapshot.
- fitsio timings are recorded, but mmap-fairness policy excludes them from
  rankings because fitsio does not expose an equivalent independently
  controllable mmap mode.
- The table suite still marks compressed-table benchmarking as a disabled
  placeholder. Compressed image coverage is active.
- The archived benchmark measures CPU/macOS behavior. CUDA and Linux require
  separate runs before making platform-wide performance claims.
