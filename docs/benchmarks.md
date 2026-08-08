# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio. CPU↔GPU comparisons are published
when hardware was available; GPU deficits are listed, not hidden.

**Honesty:** torchfits cuts **1.2.3** next (not yet released; **1.0.0rc5** is on
PyPI). Headline ratios below are
lab medians from named scorecard runs — not guarantees on your filesystem,
file mix, or PyTorch version. Check [Performance deficits](#performance-deficits)
before assuming torchfits wins every case.

## How to read this page

| If you want… | Jump to |
|---|---|
| Headline wins | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 (CPU and GPU) | [Performance deficits](#performance-deficits) |
| GPU transport rows | [I/O transport and backend](#io-transport-and-backend) |
| Reproduce numbers | [Reproducing](#reproducing) |
| Every measured configuration | [Exhaustive benchmark results](#exhaustive-benchmark-results) |
| Raw CSV | [Published CSVs](#published-csvs) |

Published GPU/CPU numbers come from the multi-host release scorecard
(`exhaustive_mps_*`, `exhaustive_cpu_*`, `exhaustive_cuda_*`). Manual
`workflow_dispatch` on `.github/workflows/bench-report.yml` is CPU-only and
does not refresh GPU cells.

## Comparison targets

| Domain | torchfits API | Compared against |
|---|---|---|
| Tensor (IMAGE HDU) | `read` / `read_tensor` / `write` | `astropy.io.fits`, `fitsio` |
| Table (dataframe) | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions, plus
**peak process RSS** (and peak CUDA alloc when on CUDA). Deficit ranking is
**time-based**; RSS is reported alongside times.

Cases are grouped into two families:

- **default** — high-level API (`torchfits.read` / `table.read`, etc.).
- **specialized** — `torchfits_specialized` methods (open-once handle /
  `open_subset_reader` paths). Empty specialized cells mean that path was not
  measured for the case.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- **Why fitsio has no mmap rows:** fitsio does not expose a comparable mmap
  toggle. Under `mmap_target=on` / `strict_mmap_fairness`, fitsio rows are
  non-comparable and show as skipped in transport tables (see
  `scripts/render_bench_iopath_table.py`).
- Warm-cache and cold-cache profiles are kept separate.

### Disk to GPU

True **disk→GPU** (GPUDirect Storage / cuFile, or a CFITSIO path that never
touches host RAM) is **not** implemented. Every Python FITS stack here
decodes on the host, then copies with `.to(device)`. Exploring a direct path
is a **1.1** candidate (see [Roadmap](roadmap.md)) — not a 1.0 claim.

### Tables on GPU transports

Table GPU transport rows compare `table.read_torch(..., device=cpu)` against
`device=cuda` / `device=mps` on a medium mixed catalog case
(`mixed_100000`). Decode still happens on the host; the GPU column measures
host decode plus H2D copy into tensor columns.

### Small-payload CUDA overhead measurement (2026-08-04 → refreshed 2026-08-07)

The development host has no CUDA device, so these follow-ups use matched CUDA
scorecard artifacts rather than a CPU proxy. The original 2026-08-04 analysis
(21 CUDA lanes, 3,654 paired `read_full` / `read_full_gpu` rows from the
Round-3 scorecard, mmap-on, same case and host) found torchfits' host-to-device
path added more fixed time than fitsio's:

| Case group | torchfits host | torchfits GPU | torchfits added | fitsio host | fitsio GPU | fitsio added |
|---|---:|---:|---:|---:|---:|---:|
| tiny images | 0.0656 ms | 0.1158 ms | 0.0509 ms | 0.0911 ms | 0.1094 ms | 0.0179 ms |
| small images | 0.1127 ms | 0.1801 ms | 0.0732 ms | 0.1588 ms | 0.1857 ms | 0.0262 ms |
| `large_uint16_2d` | 3.0271 ms | 3.3141 ms | 0.3606 ms | 2.4247 ms | 2.2133 ms | -0.1251 ms |

That supported a combined launch/H2D/conversion hypothesis (which component
dominates still needs Nsight or CUDA-event instrumentation; no speculative
CUDA-graph or stream-manager change was landed).

**Refreshed on the 2026-08-07 CANFAR CUDA exhaustive
(`exhaustive_cuda_20260807_013736`):** tiny/small GPU rows are now at parity —
`tiny_float32_1d` GPU 0.104 vs 0.106 ms (fitsio), `small_float32_2d` GPU
0.180 vs 0.194 ms, all tiny-lane ratios ≤1.10 and below significance. The
`large_uint16_2d` GPU row flipped to a win (1.543 vs 2.204 ms, 1.43×) once the
host-decode fix below removed the BZERO scalar second pass. No remaining GPU
deficit is above noise on the refreshed host; the residual rows are the
compressed-hcompress lanes (~2–3.5%, noise) and `narrow_1000000::read_full`
GPU (8.3%, significant).

**uint16 host-decode gap fixed (2026-08-05, `4652476`).** The mmap fast path
applied the BZERO=32768 unsigned offset with a scalar second pass over the
buffer after the vectorized byte-swap copy; folding the offset into the SIMD
loops removed that pass (wraps mod 2^16 exactly like the scalar cast). Same
host, mmap-on, median of 3 interleaved runs: `large_uint16_2d`
6.180 → 1.399 ms (3.4× vs standalone fitsio; 1.32 → 0.57× vs the in-family
tensor peer `fitsio` + `torch.from_numpy`), `medium_uint16_2d`
1.830 → 0.408 ms, `small_uint16_2d` 0.178 → 0.124 ms. Bitwise-verified against
the CFITSIO path on full-range uint16 data; parity suite and `ci-local` green.

## Published CSVs

Exhaustive `results.csv` / `torchfits_deficits.csv` for the scorecard runs are
linked from GitHub Release assets when published, and mirrored under
`docs/assets/bench/<run-id>/` when size allows. The newest CPU/CUDA
exhaustives (`exhaustive_cpu_20260807_082931_reader_cache`,
`exhaustive_cuda_20260807_013736`, refreshed 2026-08-07) feed
the generated tables above; the surviving 2026-08-07 CPU/CUDA run CSVs are
mirrored alongside the earlier Round-3 soak runs:

- `docs/assets/bench/exhaustive_cpu_20260807_013736/results.csv`
- `docs/assets/bench/exhaustive_cuda_20260807_013736/results.csv`
- `docs/assets/bench/exhaustive_mps_20260719_143706/results.csv`
- `docs/assets/bench/exhaustive_cpu_20260719_144337/results.csv`
- `docs/assets/bench/exhaustive_cuda_20260719_144457/results.csv`

(the `_082931_reader_cache` run's CSVs were lost with a local workspace reset;
its scorecard numbers are preserved in the generated tables above)


Deficit floors (same-mmap peers):

- **Images / cubes / spectra / cutouts** (`domain=fits`): any lag above float-timer
  ε counts — including rice/hcompress (no percent floor).
- **Arrow table interchange** (`domain=fitstable`): allow up to **1.05×**.

### Modular suites and release exhaustives

Named suites live in `benchmarks/suites.py` and resolve to `bench_all.py` flags
(`--scope` / `--filter` / `--operation` / GPU / mmap / profile):

```bash
pixi run bench-suite hcompress
pixi run bench-suite compressed_rice -- --no-mmap
pixi run bench-suite fitstable_predicate
pixi run bench-deficit-focus          # registry-driven deficit clusters
```

Release composition is the `release` suite (full fits + fitstable, mmap matrix,
GPU when present). Host recipes:

| Task | Host | Run ID prefix |
|---|---|---|
| `pixi run bench-exhaustive-local` | Mac CPU + MPS | `exhaustive_mps_*` |
| `pixi run bench-exhaustive-canfar-cpu` | CANFAR multicore CPU | `exhaustive_cpu_*` |
| `pixi run bench-exhaustive-canfar-cuda` | CANFAR CUDA | `exhaustive_cuda_*` |
| `pixi run bench-release-scorecard -- <run_dir>...` | meta | patches multi-host docs |
| `pixi run bench-cfitsio-direct` | local C | full-suite pure vendored CFITSIO (`--profile full`) |
| `pixi run bench-megacam` | local | CFHT MegaCam MEF cutouts (requires fetched sample data) |
| `pixi run bench-ml` | local | PyTorch DataLoader throughput vs fitsio |

### CFHT MegaCam cutout suite

Public CFHT MegaCam MEF samples (CADC Direct Data Service) exercise **Rice
`.fz`** repeated cutouts with peer ranking:

| Method | Role |
|---|---|
| `torchfits_cached` / `fitsio_cached` | Open once + N× subset (comparable family) |
| `torchfits_materialize` | Decompress plane once, then host slices (isolates Rice vs cutout API) |
| `torchfits_naive` | Re-open per cutout (pathological baseline; not ranked) |

Uses `ZNAXIS*` for tile-compressed sizes; throughput is cutout **payload** MB/s.

```bash
bash scripts/fetch_cfht_megacam_sample.sh   # once; idempotent
pixi run bench-megacam
```

Outputs land in `benchmarks_results/<run-id>/megacam_results.csv`. Sample
FITS files are gitignored under `benchmarks_data/cfht_megacam/`.

The 2026-08-05 same-host repro (`megacam-wave4/20260805_004215`, two files,
four HDUs, 40 256x256 cutouts per HDU) did **not** reproduce a torchfits lag:
`torchfits_cached` led `fitsio_cached` on every sampled HDU by approximately
7.5–15.2%. `torchfits_materialize` was faster still, but it is a separate
full-plane algorithm and used higher peak RSS on the larger sample. No Rice
decompression or materialization change is justified without a new host/file
combination that reverses this result.

For **uncompressed** survey mosaics (e.g. CFHTLS MegaPipe float32 stacks),
`open_subset_reader` maps the data segment once and slices cutouts with
endian swap into torch tensors — see
[ML with FITS](examples-ml.md#survey-mosaic-cutouts-cfht-megapipe). Rice `.fz`
MegaCam cutouts remain a separate comparison (tile decompress inside CFITSIO).

## Correctness Gates

| Gate | Command | Validates |
|---|---|---|
| fitsio parity | `pixi run pytest tests/test_fitsio_upstream_smoke.py -q` | Common fitsio image, header, table, compression, and checksum workflows |
| Astropy parity | `pixi run pytest tests/test_astropy_upstream_smoke.py -q` | Common Astropy HDU, header, image, compressed-image, table, and scaled-data workflows |
| Package isolation | `pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q` | Clean FITS-only package boundary and docs contract |

## Reproducing

```bash
pixi run bench-fits
pixi run bench-fitstable
pixi run bench-all
pixi run bench-ml
bash scripts/fetch_cfht_megacam_sample.sh && pixi run bench-megacam
# Full transport matrix (mmap on + off, doubles CPU rows; GPU rows for both when CUDA/MPS):
pixi run -e bench-gpu python benchmarks/bench_all.py --profile lab --scope all --mmap-matrix
```

For focused FITS partitions:

```bash
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(tiny_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(small_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(medium_|large_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(scaled_|compressed_|mef_)'
```

Named **deficit-cluster** recipes (mmap on+off, no unrelated GPU matrix when scoped to tables):

```bash
pixi run bench-deficit-focus              # hcompress + tiny_int8 + narrow predicates
pixi run bench-deficit-focus hcompress
pixi run bench-deficit-focus tiny_int8
pixi run bench-deficit-focus predicate
```

Rankings and deficit scoring group by `(domain, case_id, family, mmap_target)` so
mmap-on and mmap-off peers are never cross-compared.
## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | fits / fitstable | FITS benchmark orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression, scaling, MEF, and cutouts |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, projection, row slicing, predicates, and streaming |
| `bench_all.py` / `bench-fits` | fits | Scorecard path |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_gpu_transports.py` | fits (GPU) | CUDA/MPS image reads, cutouts, repeated cutouts (`disk→CPU→GPU` / `disk→RAM→GPU` rows) |
| `bench_ml_loader.py` | fits (diagnostic) | PyTorch `DataLoader` throughput (not merged into `bench-all` CSV) |
| `bench_gpu_memory.py` | fits (diagnostic) | GPU memory/leak checks (non-gating) |

## Coverage matrix

What the exhaustive `bench-all` suite measures today, and what is intentionally out of
scope or not yet wired into the published tables.

| Dimension | Covered? | Where | Gap / caveat |
|---|---|---|---|
| Backends (torchfits / astropy / fitsio) | Yes | `bench_fits_io.py`, `bench_fitstable_io.py` | `fitsio` often excluded from mmap-fairness summaries; **uint** image comparators may be torchfits-only when astropy requires buffered fallback |
| CPU vs GPU device | Partial | CPU: full matrix; GPU: tensor reads | GPU requires CUDA/MPS (`pixi run -e bench-gpu`); manual CI bench is CPU-only |
| I/O transport `disk→RAM→CPU` | Yes | `bench-all` mmap-on pass | Median mixes many ops/sizes — coarse aggregate |
| I/O transport `disk→CPU` (non-mmap) | Yes | `bench-all --mmap-matrix` mmap-off pass | Buffered host decode |
| I/O transport `disk→RAM→GPU` | Partial | `bench_gpu_transports.py` (mmap on) | Tensor `read_full`, cutouts, repeated cutouts; tables until suite lands |
| I/O transport `disk→CPU→GPU` | Partial | `bench_gpu_transports.py` (mmap off) | Same with buffered host decode + H2D |
| I/O transport `disk→GPU` | No | — | No host-bypass path yet (see Methodology); 1.1 candidate |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D fixtures; unsigned via BZERO in `scaled_*` |
| Tensor dimensions / sizes | Yes | tiny → large; 1D–3D (4D where fixtures exist) | Large 3D cubes may hit size caps |
| Compression (read) | Yes | gzip, rice, hcompress, plio | Write→compress cases are being added to the suite |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` | MEF random ext reads on selected fixtures |
| Multi-extension (MEF) | Yes | `mef_*`, `multi_mef_10ext`, MegaCam suite | — |
| Table full read / projection / slice | Yes | `bench_fitstable_io.py` | — |
| Table predicate / scan | Yes | `predicate_filter` (dense ~50% keep), `predicate_filter_selective` (~5–7%), `scan_count` | Both keep-rate regimes; fused gather ≠ project+mask |
| Table schemas | Partial | mixed / narrow / wide / varlen | typed / ascii at selected row counts |
| Table GPU vs CPU | Partial | GPU transports / fitstable | Expanding into published tables |
| Writes / write→compress | Partial | suite expansion | Read-heavy historically; write parity also in tests |
| ML DataLoader | Yes | `bench_ml_loader.py` | Reported in highlights / dedicated section |

### Why the I/O transport table looks sparse on GPU

1. **`disk→GPU` is always empty** — backends decode on the host first, then
   `.to(device)`. See [Disk to GPU](#disk-to-gpu).
2. **`disk→CPU→GPU` vs `disk→RAM→GPU`** — mmap-off vs mmap-on host decode + H2D.
3. **GPU rows need CUDA/MPS hardware** — published CUDA numbers come from
   CANFAR staging (`exhaustive_cuda_20260807_013736`, refreshed 2026-08-07).
4. **Tables** — see [Tables on GPU transports](#tables-on-gpu-transports).

### GPU integer dtype comparisons

The **deficit table** compares default
`torchfits.read(..., scale_on_device=True)` against
`torch.from_numpy(fitsio.read(...)).to(cuda)`. That pairing is not
dtype-equivalent for every scaled integer FITS file.

| FITS convention | fitsio @ CUDA | default `read` @ CUDA |
|---|---|---|
| Signed byte (BITPIX=8, BZERO=-128) | native `int8` H2D | narrow `int8` H2D + offset on device |
| Unsigned uint16/uint32 (BZERO) | native uint H2D | narrow storage H2D, offset on device |
| Generic BSCALE/BZERO | often native storage | `float32` on device (ML-friendly) |

For apples-to-apples integer GPU timing, the suite also records
`torchfits_dtype_fair_device` (`read_tensor(..., raw_scale=True)`).

**Training loops:** call
`torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=…)` before
`DataLoader` epochs so handle caches stay warm.

### Refreshing GPU numbers (CANFAR staging)

CUDA lab numbers come from a headless GPU session on `@staging`. From a
machine with `canfar` x509 auth:

```bash
bash scripts/selfcheck_canfar_launcher.sh
TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest TORCHFITS_BENCH_MODE=exhaustive \
  pixi run bench-canfar-gpu
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_<stamp>
bash scripts/patch_canfar_exhaustive_docs.sh exhaustive_cuda_<stamp>
```

```bash
# Local CI + docs before push
bash scripts/ci_local.sh
# Apple Silicon (MPS transport rows)
pixi run bench-mps
```

## I/O transport and backend

> **GPU summary:** Tensor **`disk→CPU→GPU`** / **`disk→RAM→GPU`** rows appear
> only when the CSV was produced on CUDA or MPS. **`disk→GPU`** stays empty
> (unsupported). Table GPU cells stay empty until the table-GPU suite lands.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_cpu_20260807_082931_reader_cache/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.12 ms` (n=174) | `0.77 ms` (n=253) | `0.22 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.22 ms` (n=174) | `0.86 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.37 ms` (n=216) | `4.43 ms` (n=184) | `0.94 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.26 ms` (n=216) | `2.89 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- Pure-C CFITSIO (vendored): `pixi run bench-cfitsio-direct` runs the **full**
  image+table scorecard fixture set with op→API mapping in
  `benchmarks/cfitsio_direct/bench_cfitsio_direct.c`
  (`fits_read_img` / `fits_read_subset` / `fits_read_record` /
  `fits_read_tblbytes` / `fits_read_col`). CSV:
  `benchmarks_results/<run-id>/cfitsio_direct.csv`.
- Cell `n=` counts comparable OK rows in the bucket; `—` indicates the
  bucket is empty (no rows match, or rows were excluded under
  `strict_mmap_fairness` in the original `bench-all` summary).
- Median is computed over heterogeneous operations (`read_full`,
  `cutout_100x100`, `header_read`, `predicate_filter`, `projection`,
  `row_slice`, etc.) and payload sizes; treat the per-cell ms as a
  coarse representative number, not a precise benchmark.

## Performance highlights

<!-- BENCH_HIGHLIGHTS_BEGIN -->
The following table showcases median wall-clock times for key FITS tensor and table cases. The **specialized** column is `torchfits_specialized` (open-once / subset-reader paths); it is empty when that path was not measured.

| Benchmark Case | Device | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Table read (100k rows, 8 cols, mixed) | CPU | **3.48 ms** | 3.36 ms | 52.52 ms | 16.87 ms | **15.62x** | **5.02x** |
| Varlen table read (100k rows, 3 cols) | CPU | **19.36 ms** | 19.11 ms | 930.67 ms | 197.17 ms | **48.70x** | **10.32x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

CPU category rows aggregate the CPU exhaustive
(`exhaustive_cpu_20260807_082931_reader_cache`, source of the generated
[highlights](#performance-highlights) and [full table](#exhaustive-benchmark-results)
above); the GPU (CUDA) rows come from `exhaustive_cuda_20260807_013736`
(see host scorecard for deficit honesty — all lags listed, floors label
noise vs significant). Category ranges are the last regenerated aggregation
shape; for absolute times prefer the generated tables above.

### FITS image I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **1D** (float32/64, int8–int64, tiny–large) | 24 | 29 μs – 1.28 ms | 302 μs – 2.57 ms | 61 μs – 1.69 ms | **2.0–13.5×** | **1.30–2.4×** |
| **2D** (float32/64, int8–int64, uint16/32, tiny–large) | 30 | 37 μs – 7.07 ms | 361 μs – 13.10 ms | 75 μs – 8.92 ms | **1.8–12.9×** | **1.26–2.2×** |
| **3D** (float32/64, int8–int64, tiny–medium) | 18 | 45 μs – 1.96 ms | 423 μs – 4.35 ms | 83 μs – 2.98 ms | **2.2–15.4×** | **1.41–2.1×** |
| **Compressed** (gzip, hcompress, rice) | 5 | 1.33–45.56 ms | 11.43–72.27 ms | 1.38–44.34 ms | **1.1–8.6×** | **0.58–1.1×** |
| **Scaled** (BSCALE/BZERO, small–large) | 3 | 76 μs – 2.93 ms | 496 μs – 6.09 ms | 128 μs – 3.87 ms | **2.1–6.6×** | **1.32–1.7×** |
| **MEF** (multi-extension, small/medium) | 2 | 69–220 μs | 614–954 μs | 155–342 μs | **4.3–8.9×** | **1.55–2.2×** |
| **Multi-MEF** (10 extensions, cutouts + random reads) | 3 | 64 μs – 8.15 ms | 634 μs – 12.01 ms | 194 μs – 11.04 ms | **1.5–40.4×** | **1.36–3.0×** |
| **Repeated cutouts** (50× 100×100) | 1 | 698 μs | 88.49 ms | 5.33 ms | **126.7×** | **7.63×** |
| **Time series frames** (5 frames) | 5 | 64–91 μs | 492–663 μs | 143–195 μs | **6.5–7.8×** | **1.95–2.3×** |
| **Header read** (all fixture types) | 87 | 14–51 μs | 257 μs – 2.46 ms | 25–267 μs | **15.1–48.2×** | **1.58–5.2×** |

**GPU (CUDA) device lanes** — 85 comparable `read_full` / cutout cases:

| Category | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|
| **1D** (tiny–large) | 107 μs – 1.57 ms | 438 μs – 2.90 ms | 106 μs – 2.05 ms | **1.8–6.8×** | **0.88–1.5×** |
| **2D** (tiny–large) | 114 μs – 11.92 ms | 450 μs – 22.88 ms | 112 μs – 13.09 ms | **1.9–6.9×** | **0.91–1.5×** |
| **3D** (tiny–medium) | 111 μs – 2.76 ms | 479 μs – 5.75 ms | 110 μs – 3.55 ms | **1.8–7.4×** | **0.95–1.6×** |
| **Compressed** (gzip, hcompress, rice) | 989 μs – 30.66 ms | 9.63–67.38 ms | 1.03–29.60 ms | **1.2–9.7×** | **0.97–1.1×** |
| **Scaled** | 198 μs – 4.34 ms | 920 μs – 10.93 ms | 203 μs – 4.91 ms | **1.9–4.6×** | **1.03–1.1×** |
| **MEF + Multi-MEF** | 143–391 μs | 1.14–2.80 ms | 173–451 μs | **4.0–17.0×** | **1.16–1.7×** |
| **Repeated cutouts (GPU)** | 1.47 ms | 61.22 ms | 6.19 ms | **41.7×** | **4.21×** |

### FITS table I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **read_full** (all schemas incl. 1M rows, varlen) | 18 | 175 μs – 74.74 ms | 2.40–681.18 ms | 0.26–485.33 ms | **1.3–32×** | **0.73–22×** |
| **projection** (column subset) | 18 | 171 μs – 72.73 ms | 2.29–526.19 ms | 0.30–107.85 ms | **1.4–38×** | **1.48–13×** |
| **row_slice** (row range) | 18 | 113 μs – 6.96 ms | 1.99–59.12 ms | 0.25–31.15 ms | **7.8–55×** | **1.55–17×** |
| **predicate_filter** (WHERE clause, dense + selective) | 36 | 71 μs – 11.13 ms | 1.51–2.64 ms | 0.12–20.44 ms | **6.9–11×** | **0.98–3×** |
| **scan_count** (streaming) | 18 | 26–65 μs | 0.40–0.98 ms | 0.06–0.43 ms | **14.4–17×** | **2.12–7×** |

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured configurations. Empty cells mean that method was not run for the case (for example `torchfits_specialized` is only used for open-once / subset-reader paths). Domain `tensor` = IMAGE HDU payloads (1D–4D); `table` = binary/ASCII tables.

| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | cfitsio (direct) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 59.7 μs | 2.47 ms | 259.3 μs | — | **41.42x** | **4.34x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 59.4 μs | 2.44 ms | 266.7 μs | — | **41.06x** | **4.49x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 56.2 μs | 2.53 ms | 286.8 μs | — | **44.97x** | **5.10x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.31 ms** | 1.41 ms | 12.12 ms | 1.44 ms | — | **9.24x** | **1.10x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 59.1 μs | 2.55 ms | 295.5 μs | — | **43.10x** | **5.00x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 26.1 μs | 475.3 μs | 53.6 μs | — | **18.24x** | **2.06x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 34.3 μs | 505.7 μs | 57.3 μs | — | **14.73x** | **1.67x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 25.5 μs | 469.5 μs | 54.9 μs | — | **18.43x** | **2.15x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 29.1 μs | 518.7 μs | 61.4 μs | — | **17.80x** | **2.11x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 30.2 μs | 449.7 μs | 54.3 μs | — | **14.89x** | **1.80x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 27.3 μs | 520.9 μs | 56.9 μs | — | **19.05x** | **2.08x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 26.5 μs | 458.6 μs | 52.1 μs | — | **17.33x** | **1.97x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 27.4 μs | 515.6 μs | 56.3 μs | — | **18.80x** | **2.05x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 27.4 μs | 473.3 μs | 51.9 μs | — | **17.28x** | **1.90x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 28.2 μs | 525.7 μs | 56.0 μs | — | **18.62x** | **1.98x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 62.0 μs | 894.3 μs | 119.3 μs | — | **14.43x** | **1.93x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 30.2 μs | 607.5 μs | 66.4 μs | — | **20.14x** | **2.20x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 29.6 μs | 589.5 μs | 73.3 μs | — | **19.89x** | **2.47x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 29.5 μs | 609.1 μs | 66.0 μs | — | **20.63x** | **2.23x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 29.0 μs | 473.6 μs | 54.0 μs | — | **16.31x** | **1.86x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 29.1 μs | 516.6 μs | 55.9 μs | — | **17.78x** | **1.92x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 29.7 μs | 549.2 μs | 63.2 μs | — | **18.46x** | **2.12x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 27.1 μs | 464.9 μs | 51.6 μs | — | **17.17x** | **1.91x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 29.9 μs | 518.7 μs | 56.2 μs | — | **17.34x** | **1.88x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 29.8 μs | 537.6 μs | 58.4 μs | — | **18.02x** | **1.96x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 28.0 μs | 473.1 μs | 51.9 μs | — | **16.88x** | **1.85x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 29.3 μs | 555.0 μs | 58.2 μs | — | **18.96x** | **1.99x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 27.6 μs | 535.7 μs | 55.8 μs | — | **19.43x** | **2.02x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 26.0 μs | 465.3 μs | 53.0 μs | — | **17.90x** | **2.04x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 28.5 μs | 502.7 μs | 55.9 μs | — | **17.67x** | **1.96x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 27.3 μs | 543.6 μs | 58.7 μs | — | **19.90x** | **2.15x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 26.9 μs | 494.7 μs | 53.6 μs | — | **18.38x** | **1.99x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 28.9 μs | 520.3 μs | 61.7 μs | — | **17.97x** | **2.13x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 29.1 μs | 542.6 μs | 59.3 μs | — | **18.62x** | **2.03x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 32.4 μs | 552.2 μs | 62.5 μs | — | **17.05x** | **1.93x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 31.3 μs | 596.1 μs | 68.7 μs | — | **19.04x** | **2.20x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 32.2 μs | 640.0 μs | 68.9 μs | — | **19.88x** | **2.14x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 29.4 μs | 601.9 μs | 67.5 μs | — | **20.48x** | **2.30x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 29.8 μs | 598.8 μs | 64.9 μs | — | **20.09x** | **2.18x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 37.2 μs | 940.6 μs | 89.7 μs | — | **25.26x** | **2.41x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 33.9 μs | 954.3 μs | 81.0 μs | — | **28.17x** | **2.39x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **180.4 μs** | 116.4 μs | 4.02 ms | 316.6 μs | — | **34.55x** | **2.72x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 36.3 μs | 948.7 μs | 88.7 μs | — | **26.13x** | **2.44x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **10.85 ms** | 11.16 ms | 12.53 ms | 13.42 ms | — | **1.16x** | **1.24x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **861.3 μs** | 852.5 μs | 94.66 ms | 6.21 ms | — | **111.04x** | **7.29x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 31.0 μs | 599.0 μs | 68.2 μs | — | **19.35x** | **2.20x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 32.8 μs | 601.3 μs | 69.8 μs | — | **18.31x** | **2.12x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 31.6 μs | 598.8 μs | 69.2 μs | — | **18.94x** | **2.19x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 28.6 μs | 484.6 μs | 51.9 μs | — | **16.95x** | **1.82x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 31.6 μs | 529.6 μs | 57.2 μs | — | **16.74x** | **1.81x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 29.4 μs | 558.9 μs | 57.3 μs | — | **19.01x** | **1.95x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 28.2 μs | 471.8 μs | 53.3 μs | — | **16.74x** | **1.89x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 30.7 μs | 521.9 μs | 56.9 μs | — | **16.98x** | **1.85x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 65.5 μs | 1.02 ms | 131.2 μs | — | **15.63x** | **2.00x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 26.9 μs | 475.5 μs | 55.3 μs | — | **17.65x** | **2.05x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 28.2 μs | 539.7 μs | 57.5 μs | — | **19.17x** | **2.04x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 29.4 μs | 569.7 μs | 63.6 μs | — | **19.36x** | **2.16x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 30.4 μs | 470.3 μs | 53.0 μs | — | **15.47x** | **1.74x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 29.4 μs | 514.6 μs | 56.5 μs | — | **17.52x** | **1.92x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 29.3 μs | 548.8 μs | 60.1 μs | — | **18.70x** | **2.05x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 29.3 μs | 472.5 μs | 53.0 μs | — | **16.11x** | **1.81x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 27.1 μs | 511.8 μs | 55.5 μs | — | **18.89x** | **2.05x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 33.5 μs | 565.4 μs | 63.0 μs | — | **16.86x** | **1.88x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 29.3 μs | 548.9 μs | 66.5 μs | — | **18.71x** | **2.27x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 32.5 μs | 605.8 μs | 67.3 μs | — | **18.64x** | **2.07x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 30.4 μs | 641.8 μs | 77.0 μs | — | **21.11x** | **2.53x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 29.9 μs | 597.0 μs | 66.8 μs | — | **19.95x** | **2.23x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 18.3 μs | 371.0 μs | 40.6 μs | — | **20.30x** | **2.22x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 17.9 μs | 296.3 μs | 34.4 μs | — | **16.52x** | **1.92x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.2 μs | 294.9 μs | 33.6 μs | — | **18.20x** | **2.08x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 18.2 μs | 289.6 μs | 31.2 μs | — | **15.93x** | **1.72x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 17.4 μs | 292.4 μs | 30.6 μs | — | **16.85x** | **1.76x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 17.0 μs | 289.7 μs | 32.4 μs | — | **17.05x** | **1.91x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.6 μs | 264.2 μs | 29.8 μs | — | **16.96x** | **1.91x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 14.8 μs | 279.2 μs | 32.2 μs | — | **18.82x** | **2.17x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.1 μs | 304.7 μs | 33.3 μs | — | **18.89x** | **2.07x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.7 μs | 269.3 μs | 29.3 μs | — | **17.12x** | **1.86x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 16.5 μs | 286.3 μs | 29.9 μs | — | **17.35x** | **1.81x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.1 μs | 329.3 μs | 32.8 μs | — | **20.43x** | **2.04x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 15.4 μs | 275.9 μs | 28.3 μs | — | **17.94x** | **1.84x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.9 μs | 290.1 μs | 29.1 μs | — | **19.48x** | **1.96x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.0 μs | 314.3 μs | 32.6 μs | — | **19.64x** | **2.04x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.9 μs | 274.5 μs | 27.9 μs | — | **17.27x** | **1.76x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 16.3 μs | 286.8 μs | 32.6 μs | — | **17.64x** | **2.01x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.2 μs | 310.6 μs | 35.1 μs | — | **19.13x** | **2.16x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.1 μs | 265.5 μs | 30.5 μs | — | **18.77x** | **2.16x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 15.3 μs | 284.4 μs | 31.8 μs | — | **18.55x** | **2.07x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.6 μs | 300.8 μs | 34.1 μs | — | **18.17x** | **2.06x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 16.9 μs | 303.6 μs | 31.6 μs | — | **17.97x** | **1.87x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 17.4 μs | 328.4 μs | 36.2 μs | — | **18.83x** | **2.08x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 17.5 μs | 348.1 μs | 38.4 μs | — | **19.83x** | **2.19x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.20 ms** | — | 58.60 ms | — | — | **1.22x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.66 ms** | — | 68.56 ms | — | — | **1.82x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **14.26 ms** | 14.42 ms | 26.67 ms | 15.29 ms | — | **1.87x** | **1.07x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **12.16 ms** | 12.12 ms | 42.34 ms | 13.33 ms | — | **3.49x** | **1.10x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **27.18 ms** | 29.13 ms | 29.78 ms | 25.63 ms | — | **1.10x** | **0.94x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **7.44 ms** | 7.44 ms | 19.10 ms | 7.35 ms | — | **2.57x** | **0.99x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **497.4 μs** | 828.1 μs | 1.17 ms | 782.4 μs | — | **2.36x** | **1.57x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **5.40 ms** | 2.00 ms | 12.26 ms | 5.99 ms | — | **6.13x** | **3.00x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **977.8 μs** | 943.9 μs | 2.05 ms | 1.35 ms | — | **2.18x** | **1.43x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **15.34 ms** | 14.82 ms | 29.47 ms | 12.50 ms | — | **1.99x** | **0.84x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **404.9 μs** | 268.1 μs | 748.4 μs | 352.9 μs | — | **2.79x** | **1.32x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.18 ms** | 1.27 ms | 1.99 ms | 1.31 ms | — | **1.69x** | **1.11x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **517.1 μs** | 491.3 μs | 1.26 ms | 825.3 μs | — | **2.56x** | **1.68x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **6.48 ms** | 7.49 ms | 11.65 ms | 6.03 ms | — | **1.80x** | **0.93x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.03 ms** | 949.3 μs | 2.57 ms | 1.63 ms | — | **2.71x** | **1.72x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **20.81 ms** | 19.29 ms | 48.15 ms | 19.45 ms | — | **2.50x** | **1.01x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **269.1 μs** | 222.7 μs | 999.6 μs | 280.0 μs | — | **4.49x** | **1.26x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **788.7 μs** | 817.1 μs | 2.14 ms | 988.1 μs | — | **2.72x** | **1.25x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.76 ms** | 1.79 ms | 6.45 ms | 2.19 ms | — | **3.66x** | **1.24x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **8.49 ms** | 10.21 ms | 15.17 ms | 9.56 ms | — | **1.79x** | **1.13x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **93.0 μs** | 109.7 μs | 583.5 μs | 179.5 μs | — | **6.28x** | **1.93x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **798.0 μs** | 710.5 μs | 1.73 ms | 1.17 ms | — | **2.44x** | **1.65x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.22 ms** | 1.26 ms | 2.46 ms | 1.78 ms | — | **2.01x** | **1.46x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **200.4 μs** | 197.3 μs | 707.7 μs | 235.4 μs | — | **3.59x** | **1.19x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.55 ms** | 1.28 ms | 2.93 ms | 1.84 ms | — | **2.28x** | **1.43x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **6.10 ms** | 3.18 ms | 13.29 ms | 6.32 ms | — | **4.18x** | **1.99x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **94.5 μs** | 110.7 μs | 623.1 μs | 138.0 μs | — | **6.60x** | **1.46x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **430.9 μs** | 488.9 μs | 1.17 ms | 522.1 μs | — | **2.72x** | **1.21x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **723.5 μs** | 664.1 μs | 1.51 ms | 786.1 μs | — | **2.28x** | **1.18x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **117.4 μs** | 131.5 μs | 619.1 μs | 187.9 μs | — | **5.27x** | **1.60x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **878.9 μs** | 933.5 μs | 1.76 ms | 1.19 ms | — | **2.00x** | **1.36x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.14 ms** | 1.23 ms | 3.17 ms | 2.38 ms | — | **2.78x** | **2.08x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **179.5 μs** | 194.5 μs | 826.4 μs | 280.8 μs | — | **4.60x** | **1.56x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.31 ms** | 1.32 ms | 2.92 ms | 1.85 ms | — | **2.23x** | **1.41x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.62 ms** | 2.66 ms | 13.43 ms | 6.24 ms | — | **5.12x** | **2.38x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **93.5 μs** | 62.7 μs | 741.9 μs | 108.9 μs | — | **11.84x** | **1.74x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **259.6 μs** | 298.9 μs | 1.06 ms | 306.1 μs | — | **4.10x** | **1.18x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **378.8 μs** | 338.9 μs | 1.30 ms | 434.7 μs | — | **3.85x** | **1.28x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **498.9 μs** | 535.5 μs | 2.95 ms | 800.9 μs | — | **5.92x** | **1.61x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **872.1 μs** | 933.3 μs | 2.72 ms | 1.39 ms | — | **3.12x** | **1.59x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **273.8 μs** | 298.7 μs | 1.46 ms | 374.3 μs | — | **5.33x** | **1.37x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **83.0 μs** | 92.9 μs | 962.2 μs | 159.0 μs | — | **11.59x** | **1.91x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **95.6 μs** | 69.5 μs | 986.4 μs | 248.1 μs | — | **14.19x** | **3.57x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **9.80 ms** | 9.82 ms | 12.56 ms | 9.16 ms | — | **1.28x** | **0.94x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **1.12 ms** | 1.14 ms | 2.11 ms | 1.34 ms | — | **1.87x** | **1.19x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **142.8 μs** | 118.8 μs | 744.1 μs | 164.6 μs | — | **6.26x** | **1.39x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **84.8 μs** | 65.0 μs | 468.4 μs | 81.2 μs | — | **7.21x** | **1.25x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **95.6 μs** | 93.7 μs | 563.1 μs | 145.7 μs | — | **6.01x** | **1.55x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **116.5 μs** | 169.6 μs | 716.9 μs | 252.9 μs | — | **6.15x** | **2.17x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **93.3 μs** | 99.8 μs | 463.3 μs | 85.5 μs | — | **4.97x** | **0.92x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **115.7 μs** | 144.1 μs | 654.9 μs | 184.7 μs | — | **5.66x** | **1.60x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **328.3 μs** | 303.5 μs | 941.9 μs | 362.4 μs | — | **3.10x** | **1.19x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **53.6 μs** | 47.3 μs | 446.4 μs | 82.4 μs | — | **9.44x** | **1.74x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **88.4 μs** | 64.7 μs | 536.9 μs | 107.3 μs | — | **8.30x** | **1.66x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **111.4 μs** | 133.0 μs | 676.3 μs | 155.3 μs | — | **6.07x** | **1.39x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **67.2 μs** | 53.9 μs | 459.9 μs | 84.9 μs | — | **8.53x** | **1.57x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **98.9 μs** | 79.6 μs | 562.0 μs | 146.7 μs | — | **7.06x** | **1.84x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **158.6 μs** | 140.4 μs | 735.1 μs | 251.8 μs | — | **5.23x** | **1.79x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **59.5 μs** | 78.5 μs | 470.8 μs | 86.4 μs | — | **7.92x** | **1.45x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **119.2 μs** | 146.7 μs | 755.5 μs | 205.7 μs | — | **6.34x** | **1.73x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **274.3 μs** | 273.8 μs | 941.6 μs | 363.5 μs | — | **3.44x** | **1.33x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **71.5 μs** | 54.5 μs | 618.5 μs | 77.2 μs | — | **11.35x** | **1.42x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **83.7 μs** | 62.0 μs | 662.0 μs | 91.4 μs | — | **10.67x** | **1.47x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **101.4 μs** | 96.5 μs | 720.6 μs | 114.1 μs | — | **7.47x** | **1.18x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **70.3 μs** | 106.7 μs | 712.7 μs | 122.7 μs | — | **10.14x** | **1.75x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **111.7 μs** | 91.6 μs | 929.7 μs | 201.8 μs | — | **10.15x** | **2.20x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **74.3 μs** | 90.6 μs | 597.1 μs | 148.0 μs | — | **8.04x** | **1.99x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **94.4 μs** | 97.9 μs | 574.8 μs | 139.3 μs | — | **6.09x** | **1.48x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **93.4 μs** | 99.5 μs | 588.2 μs | 140.8 μs | — | **6.29x** | **1.51x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **102.1 μs** | 90.1 μs | 565.1 μs | 140.3 μs | — | **6.27x** | **1.56x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **87.2 μs** | 100.7 μs | 630.4 μs | 152.2 μs | — | **7.23x** | **1.75x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **59.1 μs** | 47.1 μs | 449.0 μs | 77.9 μs | — | **9.54x** | **1.65x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **67.3 μs** | 51.4 μs | 495.0 μs | 80.9 μs | — | **9.64x** | **1.58x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **60.8 μs** | 55.3 μs | 501.7 μs | 83.9 μs | — | **9.08x** | **1.52x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **49.3 μs** | 42.4 μs | 450.8 μs | 80.5 μs | — | **10.64x** | **1.90x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **48.9 μs** | 71.4 μs | 485.1 μs | 79.4 μs | — | **9.93x** | **1.63x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **71.6 μs** | 57.3 μs | 535.0 μs | 85.8 μs | — | **9.34x** | **1.50x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **64.6 μs** | 53.2 μs | 455.1 μs | 82.2 μs | — | **8.56x** | **1.55x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **90.5 μs** | 101.5 μs | 493.9 μs | 82.2 μs | — | **5.46x** | **0.91x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **50.8 μs** | 71.2 μs | 535.7 μs | 88.2 μs | — | **10.55x** | **1.74x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **60.1 μs** | 66.1 μs | 475.7 μs | 83.9 μs | — | **7.92x** | **1.40x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **70.2 μs** | 91.5 μs | 526.3 μs | 88.9 μs | — | **7.50x** | **1.27x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **51.3 μs** | 75.5 μs | 538.0 μs | 88.7 μs | — | **10.50x** | **1.73x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **50.3 μs** | 85.6 μs | 473.8 μs | 93.3 μs | — | **9.42x** | **1.85x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **82.1 μs** | 60.2 μs | 538.6 μs | 95.4 μs | — | **8.95x** | **1.59x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **94.3 μs** | 85.2 μs | 548.3 μs | 99.1 μs | — | **6.44x** | **1.16x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **61.9 μs** | 57.5 μs | 633.2 μs | 83.9 μs | — | **11.02x** | **1.46x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **63.3 μs** | 97.8 μs | 677.3 μs | 95.7 μs | — | **10.70x** | **1.51x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **82.0 μs** | 57.8 μs | 721.1 μs | 94.0 μs | — | **12.47x** | **1.63x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **25.26 ms** | 27.24 ms | 46.92 ms | 27.04 ms | — | **1.86x** | **1.07x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **22.96 ms** | 22.74 ms | 74.09 ms | 23.63 ms | — | **3.26x** | **1.04x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **47.78 ms** | 46.38 ms | 52.81 ms | 44.99 ms | — | **1.14x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **14.53 ms** | 13.98 ms | 34.13 ms | 13.18 ms | — | **2.44x** | **0.94x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **1.36 ms** | 1.34 ms | 2.22 ms | — | — | **1.66x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **11.22 ms** | 4.61 ms | 13.50 ms | — | — | **2.93x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **2.50 ms** | 2.53 ms | 3.70 ms | — | — | **1.48x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **27.95 ms** | 28.23 ms | 42.23 ms | — | — | **1.51x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **646.7 μs** | 740.8 μs | 1.56 ms | — | — | **2.41x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **2.06 ms** | 2.13 ms | 4.00 ms | — | — | **1.94x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **1.27 ms** | 1.19 ms | 2.36 ms | — | — | **1.98x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **4.54 ms** | 10.92 ms | 14.20 ms | — | — | **3.13x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.97 ms** | 2.14 ms | 3.98 ms | — | — | **2.02x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **23.10 ms** | 22.85 ms | 42.13 ms | — | — | **1.84x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **397.4 μs** | 398.7 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **1.87 ms** | 1.39 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **2.41 ms** | 2.22 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **6.12 ms** | 13.54 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **199.7 μs** | 171.7 μs | 786.4 μs | — | — | **4.58x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.58 ms** | 1.26 ms | 2.43 ms | — | — | **1.93x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.59 ms** | 1.48 ms | 3.36 ms | — | — | **2.27x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **298.8 μs** | 262.5 μs | 1.18 ms | — | — | **4.51x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **2.18 ms** | 2.79 ms | 3.52 ms | — | — | **1.62x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **2.66 ms** | 2.87 ms | 4.76 ms | — | — | **1.79x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **151.0 μs** | 104.7 μs | 666.9 μs | — | — | **6.37x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **837.5 μs** | 704.6 μs | 1.63 ms | — | — | **2.32x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **1.06 ms** | 1.01 ms | 2.11 ms | — | — | **2.09x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **254.3 μs** | 222.3 μs | 773.6 μs | — | — | **3.48x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.14 ms** | 1.15 ms | 2.64 ms | — | — | **2.32x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.64 ms** | 1.70 ms | 3.40 ms | — | — | **2.07x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **290.0 μs** | 331.6 μs | 994.7 μs | — | — | **3.43x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **2.06 ms** | 2.01 ms | 4.04 ms | — | — | **2.01x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **3.47 ms** | 3.66 ms | 5.36 ms | — | — | **1.55x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **119.3 μs** | 129.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **372.7 μs** | 500.3 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **667.7 μs** | 593.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **656.2 μs** | 729.7 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.65 ms** | 2.15 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **447.7 μs** | 430.5 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **152.5 μs** | 61.0 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **77.2 μs** | 142.6 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **5.68 ms** | 12.76 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **1.56 ms** | 1.56 ms | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **179.3 μs** | 192.4 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **60.0 μs** | 97.6 μs | 522.5 μs | — | — | **8.70x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **142.9 μs** | 160.3 μs | 776.3 μs | — | — | **5.43x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **232.7 μs** | 225.5 μs | 969.2 μs | — | — | **4.30x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **54.9 μs** | 91.2 μs | 542.6 μs | — | — | **9.89x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **215.3 μs** | 223.6 μs | 870.0 μs | — | — | **4.04x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **496.9 μs** | 491.7 μs | 1.32 ms | — | — | **2.68x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **60.3 μs** | 95.7 μs | 503.7 μs | — | — | **8.36x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **162.0 μs** | 87.9 μs | 680.4 μs | — | — | **7.74x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **197.0 μs** | 225.5 μs | 809.9 μs | — | — | **4.11x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **119.4 μs** | 67.0 μs | 509.7 μs | — | — | **7.60x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **186.2 μs** | 191.7 μs | 735.7 μs | — | — | **3.95x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **220.3 μs** | 311.8 μs | 975.8 μs | — | — | **4.43x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **116.2 μs** | 95.3 μs | 531.1 μs | — | — | **5.57x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **199.0 μs** | 251.1 μs | 889.5 μs | — | — | **4.47x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **477.9 μs** | 470.9 μs | 1.31 ms | — | — | **2.78x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **55.9 μs** | 120.7 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **132.6 μs** | 59.6 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **139.7 μs** | 156.0 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **160.7 μs** | 152.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **219.5 μs** | 231.8 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **89.7 μs** | 148.8 μs | 805.4 μs | — | — | **8.98x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **148.5 μs** | 71.8 μs | 724.9 μs | — | — | **10.10x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **152.7 μs** | 153.2 μs | 722.5 μs | — | — | **4.73x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **74.3 μs** | 147.1 μs | 741.1 μs | — | — | **9.98x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **138.6 μs** | 150.7 μs | 756.9 μs | — | — | **5.46x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **46.7 μs** | 66.1 μs | 494.6 μs | — | — | **10.60x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **103.7 μs** | 55.7 μs | 529.7 μs | — | — | **9.51x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **110.4 μs** | 72.1 μs | 558.5 μs | — | — | **7.75x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **85.9 μs** | 57.2 μs | 528.2 μs | — | — | **9.24x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **65.8 μs** | 97.9 μs | 554.4 μs | — | — | **8.43x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **60.2 μs** | 88.5 μs | 580.4 μs | — | — | **9.63x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **79.9 μs** | 70.7 μs | 501.3 μs | — | — | **7.09x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **107.2 μs** | 72.4 μs | 532.7 μs | — | — | **7.36x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **62.1 μs** | 86.9 μs | 550.2 μs | — | — | **8.86x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **73.8 μs** | 96.1 μs | 498.4 μs | — | — | **6.75x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **93.4 μs** | 56.0 μs | 523.5 μs | — | — | **9.36x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **68.5 μs** | 101.3 μs | 569.6 μs | — | — | **8.31x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **83.7 μs** | 67.3 μs | 509.4 μs | — | — | **7.57x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **111.4 μs** | 119.5 μs | 558.2 μs | — | — | **5.01x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **108.1 μs** | 86.6 μs | 562.2 μs | — | — | **6.49x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **49.3 μs** | 100.0 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **124.6 μs** | 65.8 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **51.8 μs** | 111.9 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **598.9 μs** | 609.4 μs | 4.45 ms | 600.5 μs | — | **7.43x** | **1.00x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **611.3 μs** | 622.6 μs | 4.47 ms | 591.4 μs | — | **7.32x** | **0.97x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.76 ms** | 1.69 ms | 13.76 ms | 3.38 ms | — | **8.15x** | **2.00x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.76 ms** | 1.69 ms | 14.57 ms | 3.45 ms | — | **8.61x** | **2.04x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **355.5 μs** | 268.3 μs | 4.38 ms | 890.1 μs | — | **16.34x** | **3.32x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **46.8 μs** | 45.8 μs | 696.9 μs | 108.3 μs | — | **15.23x** | **2.37x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **217.1 μs** | 225.2 μs | 2.51 ms | 266.3 μs | — | **11.55x** | **1.23x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **228.0 μs** | 225.4 μs | 2.50 ms | 265.4 μs | — | **11.10x** | **1.18x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **356.6 μs** | 351.7 μs | 4.76 ms | 721.6 μs | — | **13.54x** | **2.05x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **357.3 μs** | 276.1 μs | 3.65 ms | 539.4 μs | — | **13.20x** | **1.95x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **276.6 μs** | 151.3 μs | 3.26 ms | 317.5 μs | — | **21.54x** | **2.10x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **44.4 μs** | 44.5 μs | 688.1 μs | 108.4 μs | — | **15.48x** | **2.44x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **21.39 ms** | 23.77 ms | 65.33 ms | 42.34 ms | — | **3.05x** | **1.98x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **19.17 ms** | 23.20 ms | 58.60 ms | 32.96 ms | — | **3.06x** | **1.72x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **40.67 ms** | 19.39 ms | 50.59 ms | 60.20 ms | — | **2.61x** | **3.10x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **77.93 ms** | 88.43 ms | 711.46 ms | 301.09 ms | — | **9.13x** | **3.86x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **782.5 μs** | 579.2 μs | 61.29 ms | 2.94 ms | — | **105.80x** | **5.08x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **68.0 μs** | 69.0 μs | 821.5 μs | 143.7 μs | — | **12.08x** | **2.11x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **2.08 ms** | 2.02 ms | 5.21 ms | 3.66 ms | — | **2.58x** | **1.81x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **2.25 ms** | 1.83 ms | 4.67 ms | 3.01 ms | — | **2.55x** | **1.64x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.82 ms** | 1.73 ms | 5.10 ms | 5.36 ms | — | **2.94x** | **3.09x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **3.48 ms** | 3.36 ms | 52.52 ms | 16.87 ms | — | **15.62x** | **5.02x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **521.8 μs** | 444.5 μs | 9.99 ms | 2.56 ms | — | **22.47x** | **5.75x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **52.4 μs** | 51.4 μs | 780.9 μs | 131.0 μs | — | **15.19x** | **2.55x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **404.0 μs** | 456.4 μs | 3.28 ms | 623.3 μs | — | **8.12x** | **1.54x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **304.1 μs** | 373.0 μs | 3.22 ms | 543.8 μs | — | **10.59x** | **1.79x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **329.7 μs** | 253.1 μs | 3.28 ms | 784.5 μs | — | **12.95x** | **3.10x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **511.6 μs** | 438.9 μs | 7.80 ms | 1.82 ms | — | **17.78x** | **4.14x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **274.6 μs** | 192.1 μs | 5.08 ms | 537.4 μs | — | **26.46x** | **2.80x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **47.6 μs** | 47.8 μs | 743.5 μs | 128.1 μs | — | **15.62x** | **2.69x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **138.2 μs** | 216.4 μs | 3.12 ms | 311.9 μs | — | **22.56x** | **2.26x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **130.7 μs** | 209.7 μs | 3.11 ms | 300.4 μs | — | **23.79x** | **2.30x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **213.4 μs** | 143.7 μs | 3.12 ms | 322.3 μs | — | **21.70x** | **2.24x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **271.0 μs** | 198.9 μs | 3.71 ms | 440.3 μs | — | **18.67x** | **2.21x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **248.3 μs** | 166.9 μs | 4.57 ms | 338.3 μs | — | **27.40x** | **2.03x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **47.4 μs** | 49.2 μs | 750.7 μs | 130.5 μs | — | **15.85x** | **2.76x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **8.26 ms** | 8.05 ms | 16.37 ms | 29.47 ms | — | **2.03x** | **3.66x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **7.53 ms** | 7.58 ms | 9.79 ms | 20.58 ms | — | **1.30x** | **2.73x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **7.41 ms** | 7.33 ms | 8.82 ms | 40.70 ms | — | **1.20x** | **5.55x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **10.26 ms** | 9.97 ms | 10.74 ms | 9.62 ms | — | **1.08x** | **0.96x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **334.6 μs** | 226.3 μs | 6.91 ms | 994.9 μs | — | **30.54x** | **4.40x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **60.1 μs** | 52.5 μs | 773.9 μs | 116.7 μs | — | **14.74x** | **2.22x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.34 ms** | 1.24 ms | 3.96 ms | 2.93 ms | — | **3.20x** | **2.36x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **1.01 ms** | 1.02 ms | 2.98 ms | 2.15 ms | — | **2.96x** | **2.14x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **926.6 μs** | 849.8 μs | 2.97 ms | 4.33 ms | — | **3.49x** | **5.09x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **1.20 ms** | 1.12 ms | 3.18 ms | 1.10 ms | — | **2.84x** | **0.98x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **320.4 μs** | 226.5 μs | 3.51 ms | 993.4 μs | — | **15.52x** | **4.39x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **42.9 μs** | 42.7 μs | 731.9 μs | 109.3 μs | — | **17.16x** | **2.56x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **300.1 μs** | 378.5 μs | 2.37 ms | 515.8 μs | — | **7.91x** | **1.72x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **223.4 μs** | 296.4 μs | 2.30 ms | 440.1 μs | — | **10.30x** | **1.97x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **273.9 μs** | 186.3 μs | 2.33 ms | 660.1 μs | — | **12.51x** | **3.54x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **297.2 μs** | 231.8 μs | 2.41 ms | 321.3 μs | — | **10.39x** | **1.39x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **223.4 μs** | 139.3 μs | 3.06 ms | 325.3 μs | — | **21.99x** | **2.34x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **45.0 μs** | 57.3 μs | 962.1 μs | 138.3 μs | — | **21.38x** | **3.07x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **134.0 μs** | 212.3 μs | 2.25 ms | 281.2 μs | — | **16.81x** | **2.10x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **120.7 μs** | 206.4 μs | 2.25 ms | 272.7 μs | — | **18.62x** | **2.26x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **205.1 μs** | 132.5 μs | 2.24 ms | 288.8 μs | — | **16.87x** | **2.18x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **208.8 μs** | 136.1 μs | 2.31 ms | 242.1 μs | — | **16.96x** | **1.78x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **211.8 μs** | 128.6 μs | 2.99 ms | 254.2 μs | — | **23.23x** | **1.98x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **46.9 μs** | 45.6 μs | 763.3 μs | 108.8 μs | — | **16.75x** | **2.39x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **1.32 ms** | 1.33 ms | 3.12 ms | 2.30 ms | — | **2.37x** | **1.74x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **1.30 ms** | 1.34 ms | 3.15 ms | 2.30 ms | — | **2.43x** | **1.77x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **6.23 ms** | 5.85 ms | 51.82 ms | 22.84 ms | — | **8.86x** | **3.91x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **9.13 ms** | 9.04 ms | 52.30 ms | 25.39 ms | — | **5.79x** | **2.81x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **1.11 ms** | 1.02 ms | 7.94 ms | 3.26 ms | — | **7.75x** | **3.18x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **44.7 μs** | 47.8 μs | 764.9 μs | 118.1 μs | — | **17.11x** | **2.64x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **349.4 μs** | 404.8 μs | 2.51 ms | 506.2 μs | — | **7.19x** | **1.45x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **304.5 μs** | 357.7 μs | 2.38 ms | 473.5 μs | — | **7.82x** | **1.55x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **950.9 μs** | 828.3 μs | 7.29 ms | 2.64 ms | — | **8.80x** | **3.19x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **1.23 ms** | 1.12 ms | 7.37 ms | 2.89 ms | — | **6.60x** | **2.58x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **342.2 μs** | 265.6 μs | 3.82 ms | 695.8 μs | — | **14.39x** | **2.62x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **45.8 μs** | 49.2 μs | 717.6 μs | 116.5 μs | — | **15.66x** | **2.54x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.42 ms** | 1.28 ms | 3.06 ms | 2.21 ms | — | **2.39x** | **1.73x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **1.11 ms** | 1.17 ms | 2.80 ms | 2.19 ms | — | **2.53x** | **1.97x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **19.22 ms** | 18.85 ms | 923.27 ms | 193.03 ms | — | **48.98x** | **10.24x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **19.36 ms** | 19.11 ms | 930.67 ms | 197.17 ms | — | **48.70x** | **10.32x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **1.97 ms** | 1.92 ms | 95.56 ms | 21.11 ms | — | **49.74x** | **10.99x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **50.9 μs** | 62.0 μs | 759.7 μs | 136.0 μs | — | **14.92x** | **2.67x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **338.5 μs** | 440.2 μs | 2.50 ms | 548.0 μs | — | **7.38x** | **1.62x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **316.4 μs** | 455.7 μs | 2.54 ms | 547.6 μs | — | **8.01x** | **1.73x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **2.10 ms** | 2.08 ms | 93.84 ms | 19.44 ms | — | **45.18x** | **9.36x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **2.18 ms** | 2.06 ms | 93.78 ms | 19.35 ms | — | **45.49x** | **9.39x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **503.5 μs** | 368.5 μs | 12.44 ms | 2.47 ms | — | **33.76x** | **6.71x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **60.2 μs** | 50.8 μs | 770.8 μs | 117.7 μs | — | **15.18x** | **2.32x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **153.1 μs** | 264.1 μs | 2.29 ms | 314.3 μs | — | **14.93x** | **2.05x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **141.6 μs** | 261.8 μs | 2.27 ms | 317.3 μs | — | **16.05x** | **2.24x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **470.7 μs** | 340.8 μs | 11.66 ms | 2.28 ms | — | **34.23x** | **6.70x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **458.4 μs** | 375.8 μs | 11.72 ms | 2.24 ms | — | **31.19x** | **5.96x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **294.1 μs** | 220.7 μs | 4.04 ms | 568.6 μs | — | **18.29x** | **2.58x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **55.0 μs** | 51.8 μs | 792.2 μs | 111.8 μs | — | **15.30x** | **2.16x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **5.04 ms** | 5.44 ms | 14.14 ms | 6.96 ms | — | **2.81x** | **1.38x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **5.43 ms** | 4.83 ms | 14.61 ms | 6.50 ms | — | **3.02x** | **1.34x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **4.74 ms** | 4.65 ms | 14.12 ms | 8.70 ms | — | **3.04x** | **1.87x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **29.87 ms** | 29.04 ms | 237.01 ms | 95.92 ms | — | **8.16x** | **3.30x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **2.26 ms** | 2.19 ms | 37.01 ms | 8.35 ms | — | **16.89x** | **3.81x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **67.0 μs** | 64.3 μs | 951.9 μs | 447.3 μs | — | **14.79x** | **6.95x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **768.7 μs** | 847.7 μs | 10.68 ms | 1.27 ms | — | **13.90x** | **1.65x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **718.9 μs** | 754.6 μs | 10.09 ms | 1.19 ms | — | **14.03x** | **1.65x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **670.5 μs** | 588.8 μs | 10.16 ms | 1.41 ms | — | **17.26x** | **2.40x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **2.26 ms** | 2.18 ms | 28.67 ms | 7.44 ms | — | **13.16x** | **3.42x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **932.8 μs** | 863.4 μs | 18.12 ms | 1.50 ms | — | **20.99x** | **1.74x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **62.9 μs** | 60.9 μs | 959.8 μs | 427.7 μs | — | **15.75x** | **7.02x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **217.5 μs** | 287.3 μs | 9.74 ms | 657.4 μs | — | **44.77x** | **3.02x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **214.1 μs** | 300.5 μs | 9.66 ms | 649.2 μs | — | **45.12x** | **3.03x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **299.4 μs** | 228.1 μs | 9.73 ms | 668.5 μs | — | **42.66x** | **2.93x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **930.1 μs** | 862.8 μs | 12.33 ms | 1.37 ms | — | **14.29x** | **1.59x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **815.5 μs** | 740.4 μs | 16.15 ms | 860.3 μs | — | **21.82x** | **1.16x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **63.4 μs** | 64.3 μs | 956.7 μs | 432.1 μs | — | **15.08x** | **6.81x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **430.0 μs** | 419.2 μs | 2.65 ms | — | — | **6.31x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **424.4 μs** | 436.5 μs | 2.70 ms | — | — | **6.37x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.09 ms** | 1.04 ms | 8.06 ms | — | — | **7.74x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.08 ms** | 1.04 ms | 8.09 ms | — | — | **7.79x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **258.5 μs** | 214.2 μs | 2.56 ms | — | — | **11.98x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **27.6 μs** | 27.7 μs | 407.8 μs | — | — | **14.76x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **187.2 μs** | 189.5 μs | 1.51 ms | — | — | **8.06x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **189.5 μs** | 198.3 μs | 1.51 ms | — | — | **7.95x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **259.0 μs** | 223.5 μs | 2.18 ms | — | — | **9.75x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **255.9 μs** | 222.8 μs | 2.18 ms | — | — | **9.76x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **183.8 μs** | 140.3 μs | 1.94 ms | — | — | **13.84x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **29.4 μs** | 32.0 μs | 413.1 μs | — | — | **14.07x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.02 ms** | 5.10 ms | 12.22 ms | — | — | **2.43x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.05 ms** | 3.95 ms | 8.94 ms | — | — | **2.26x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **10.03 ms** | 7.38 ms | 17.84 ms | — | — | **2.42x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **37.63 ms** | 37.23 ms | 399.79 ms | — | — | **10.74x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **293.0 μs** | 223.5 μs | 9.52 ms | — | — | **42.61x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **29.3 μs** | 30.4 μs | 460.6 μs | — | — | **15.74x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.05 ms** | 781.3 μs | 3.08 ms | — | — | **3.94x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **655.6 μs** | 652.1 μs | 2.90 ms | — | — | **4.45x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **892.9 μs** | 828.2 μs | 3.01 ms | — | — | **3.64x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **1.95 ms** | 1.84 ms | 30.74 ms | — | — | **16.70x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **272.7 μs** | 209.7 μs | 5.73 ms | — | — | **27.33x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **31.5 μs** | 28.1 μs | 443.9 μs | — | — | **15.79x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **212.4 μs** | 237.1 μs | 1.94 ms | — | — | **9.15x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **193.1 μs** | 187.1 μs | 1.90 ms | — | — | **10.15x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **178.2 μs** | 126.5 μs | 1.92 ms | — | — | **15.15x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **271.3 μs** | 218.3 μs | 4.59 ms | — | — | **21.01x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **168.6 μs** | 120.1 μs | 2.97 ms | — | — | **24.74x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **28.4 μs** | 28.6 μs | 439.7 μs | — | — | **15.47x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **181.7 μs** | 295.2 μs | 3.36 ms | — | — | **18.47x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **167.3 μs** | 271.6 μs | 3.30 ms | — | — | **19.74x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **306.3 μs** | 184.8 μs | 3.34 ms | — | — | **18.07x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **351.1 μs** | 229.7 μs | 3.99 ms | — | — | **17.35x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **350.6 μs** | 225.1 μs | 4.85 ms | — | — | **21.53x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **58.9 μs** | 63.1 μs | 793.0 μs | — | — | **13.47x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **2.49 ms** | 1.91 ms | 9.15 ms | — | — | **4.80x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.47 ms** | 1.65 ms | 5.02 ms | — | — | **3.41x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.27 ms** | 2.17 ms | 5.30 ms | — | — | **2.44x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.32 ms** | 3.28 ms | 6.56 ms | — | — | **2.00x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **170.5 μs** | 121.6 μs | 3.86 ms | — | — | **31.76x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **37.7 μs** | 29.1 μs | 436.7 μs | — | — | **15.01x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **768.6 μs** | 608.4 μs | 2.33 ms | — | — | **3.84x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **396.3 μs** | 378.8 μs | 1.86 ms | — | — | **4.90x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **326.5 μs** | 276.7 μs | 1.75 ms | — | — | **6.33x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **444.8 μs** | 375.8 μs | 1.89 ms | — | — | **5.03x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **174.8 μs** | 112.9 μs | 2.05 ms | — | — | **18.13x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **36.4 μs** | 29.3 μs | 437.5 μs | — | — | **14.93x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **185.9 μs** | 233.3 μs | 1.60 ms | — | — | **8.59x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **122.4 μs** | 164.0 μs | 1.37 ms | — | — | **11.15x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **143.9 μs** | 95.0 μs | 1.37 ms | — | — | **14.40x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **153.3 μs** | 112.0 μs | 1.41 ms | — | — | **12.61x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **138.3 μs** | 88.1 μs | 1.81 ms | — | — | **20.51x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **27.0 μs** | 26.0 μs | 420.3 μs | — | — | **16.14x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **155.8 μs** | 282.4 μs | 2.44 ms | — | — | **15.68x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **144.1 μs** | 279.7 μs | 2.39 ms | — | — | **16.56x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **255.1 μs** | 156.5 μs | 2.43 ms | — | — | **15.52x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **261.8 μs** | 169.2 μs | 2.46 ms | — | — | **14.52x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **265.1 μs** | 167.8 μs | 3.15 ms | — | — | **18.79x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **49.9 μs** | 53.2 μs | 778.8 μs | — | — | **15.60x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **673.7 μs** | 571.4 μs | 1.78 ms | — | — | **3.12x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **547.2 μs** | 518.1 μs | 1.78 ms | — | — | **3.44x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.28 ms** | 1.18 ms | 29.04 ms | — | — | **24.71x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.41 ms** | 1.29 ms | 29.14 ms | — | — | **22.63x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **275.7 μs** | 205.9 μs | 4.54 ms | — | — | **22.05x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **32.1 μs** | 32.4 μs | 426.8 μs | — | — | **13.30x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **182.0 μs** | 223.7 μs | 1.62 ms | — | — | **8.91x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **168.9 μs** | 204.0 μs | 1.44 ms | — | — | **8.55x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **255.8 μs** | 201.8 μs | 4.05 ms | — | — | **20.09x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **263.5 μs** | 207.8 μs | 4.10 ms | — | — | **19.73x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **156.0 μs** | 104.3 μs | 2.15 ms | — | — | **20.63x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **34.4 μs** | 29.0 μs | 429.3 μs | — | — | **14.81x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **579.6 μs** | 453.1 μs | 1.59 ms | — | — | **3.50x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **444.9 μs** | 458.1 μs | 1.59 ms | — | — | **3.57x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **11.58 ms** | 11.19 ms | 535.24 ms | — | — | **47.83x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **11.11 ms** | 11.13 ms | 535.95 ms | — | — | **48.24x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **1.15 ms** | 1.16 ms | 54.66 ms | — | — | **47.40x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **29.4 μs** | 29.7 μs | 449.0 μs | — | — | **15.26x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **169.8 μs** | 211.4 μs | 1.37 ms | — | — | **8.07x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **161.9 μs** | 206.2 μs | 1.33 ms | — | — | **8.19x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **1.15 ms** | 1.14 ms | 53.98 ms | — | — | **47.43x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **1.15 ms** | 1.18 ms | 54.07 ms | — | — | **47.10x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **279.4 μs** | 243.9 μs | 7.03 ms | — | — | **28.84x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **29.9 μs** | 28.6 μs | 420.8 μs | — | — | **14.70x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **82.1 μs** | 131.0 μs | 1.28 ms | — | — | **15.54x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **85.7 μs** | 131.4 μs | 1.27 ms | — | — | **14.80x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **281.2 μs** | 237.8 μs | 6.68 ms | — | — | **28.08x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **276.8 μs** | 247.1 μs | 6.68 ms | — | — | **27.04x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **201.4 μs** | 165.3 μs | 2.24 ms | — | — | **13.57x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **25.7 μs** | 26.8 μs | 413.4 μs | — | — | **16.09x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.55 ms** | 1.51 ms | 7.94 ms | — | — | **5.26x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.46 ms** | 1.41 ms | 8.21 ms | — | — | **5.83x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.65 ms** | 1.60 ms | 7.52 ms | — | — | **4.70x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **16.67 ms** | 15.59 ms | 138.87 ms | — | — | **8.91x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.06 ms** | 1.02 ms | 21.35 ms | — | — | **21.02x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **39.1 μs** | 41.3 μs | 555.4 μs | — | — | **14.21x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **315.7 μs** | 346.8 μs | 6.10 ms | — | — | **19.34x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **281.1 μs** | 336.5 μs | 7.15 ms | — | — | **25.45x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **281.3 μs** | 228.5 μs | 5.74 ms | — | — | **25.12x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.05 ms** | 989.2 μs | 16.60 ms | — | — | **16.79x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **501.4 μs** | 450.8 μs | 10.36 ms | — | — | **22.98x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **38.1 μs** | 44.5 μs | 649.1 μs | — | — | **17.02x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **272.2 μs** | 352.1 μs | 8.50 ms | — | — | **31.21x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **136.9 μs** | 169.4 μs | 5.58 ms | — | — | **40.75x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **377.9 μs** | 267.7 μs | 9.96 ms | — | — | **37.22x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **945.7 μs** | 858.1 μs | 12.60 ms | — | — | **14.69x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **858.3 μs** | 771.3 μs | 16.41 ms | — | — | **21.28x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **39.9 μs** | 37.7 μs | 549.1 μs | — | — | **14.55x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | medium_float64_3d [read_full] | off | 6.10 ms | 300.3 | fitsio/fitsio_torch | 1.58× |
| Linux x86_64 / CPU | tensor | scaled_large [read_full] | off | 9.80 ms | 303.8 | fitsio/fitsio_torch | 1.20× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 47.78 ms | 299.2 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CPU | tensor | compressed_rice_1 [read_full] | on | 14.53 ms | 299.2 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CPU | tensor | large_uint32_2d [read_full] | off | 8.49 ms | 303.8 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | large_uint32_2d [read_full] | off | 10.21 ms | 303.8 | fitsio/fitsio_torch | 1.21× |
| Linux x86_64 / CPU | tensor | scaled_large [read_full] | off | 9.82 ms | 303.8 | fitsio/fitsio_torch | 1.20× |
| Linux x86_64 / CPU | tensor | large_int32_2d [read_full] | off | 7.49 ms | 303.8 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 29.13 ms | 315.9 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 46.38 ms | 299.2 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | compressed_rice_1 [read_full] | on | 13.98 ms | 299.2 | fitsio/fitsio_torch | 1.00× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 1.20 ms | 369.8 | fitsio/fitsio_torch | 1.36× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 10.26 ms | 389.1 | fitsio/fitsio_torch | 1.32× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter_selective] | off | 611.3 μs | 378.9 | fitsio/fitsio_torch | 1.00× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter_selective] | off | 622.6 μs | 378.9 | fitsio/fitsio | 1.05× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 9.97 ms | 393.2 | fitsio/fitsio | 1.04× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 1.12 ms | 369.8 | fitsio/fitsio | 1.02× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter] | off | 609.4 μs | 378.9 | fitsio/fitsio | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 118.1 μs | 766.8 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_float64_3d [read_full @ cuda] | off | 131.0 μs | 766.8 | fitsio/fitsio_torch_device | 1.08× |
| Linux x86_64 / CUDA | tensor | tiny_int16_2d [read_full @ cuda] | off | 117.4 μs | 766.8 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_int64_1d [read_full @ cuda] | off | 111.0 μs | 766.8 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.72 ms | 729.4 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | medium_int8_1d [read_full @ cuda] | off | 145.8 μs | 766.8 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.60 ms | 699.1 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int64_2d [read_full @ cuda] | off | 125.7 μs | 766.8 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int8_2d [read_full @ cuda] | off | 117.5 μs | 766.8 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.27 ms | 606.7 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_float64_1d [read_full @ cuda] | off | 112.6 μs | 766.8 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | small_int8_1d [read_full @ cuda] | off | 113.2 μs | 766.8 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.24 ms | 728.3 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CUDA | tensor | small_uint16_2d [read_full @ cuda] | off | 153.5 μs | 766.8 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_float64_2d [read_full @ cuda] | off | 119.2 μs | 766.8 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | small_int64_1d [read_full @ cuda] | off | 133.1 μs | 766.8 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | medium_int8_2d [read_full] | off | 321.0 μs | 765.8 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_int32_2d [read_full @ cuda] | off | 114.4 μs | 766.8 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.73 ms | 729.4 | fitsio/fitsio_torch_device_specialized | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.58 ms | 699.1 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.31 ms | 606.7 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.33 ms | 728.3 | fitsio/fitsio_torch | 1.02× |

_…and 4 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260807_082931_reader_cache` | 3057 | 18 | 300.3 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260807_013736` | 4315 | 26 | 719.3 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

Round-3 soak (post thin-I/O): MPS `exhaustive_mps_20260719_143706` (local);
CANFAR staging CPU `exhaustive_cpu_20260719_144337` and CUDA
`exhaustive_cuda_20260719_144457` (clone `bench/thin-io-scorecard` @ 9b9e7cf).
ML loader: `ml_20260719_145743`. MegaCam: `20260719_075555`.



Latest local quick benchmark evidence:

<!-- BENCH_QUICK_BEGIN -->
| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| — | FITS image I/O | _(no run yet)_ | — | — |
| — | FITS table I/O | _(no run yet)_ | — | — |
<!-- BENCH_QUICK_END -->

### ML DataLoader throughput

<!-- BENCH_ML_BEGIN -->
_Run `pixi run bench-ml` to populate ML loader throughput._
<!-- BENCH_ML_END -->

### CFHT MegaCam MEF cutouts (local)

<!-- BENCH_MEGACAM_BEGIN -->
Source: `docs/assets/bench/20260719_075555/megacam_results.csv` (160 OK rows).
Median throughput over OK rows (earlier table values were copy-paste μs from unrelated suites).

| Method | Median throughput |
|---|---:|
| `fitsio_cached` | 52.7 MB/s |
| `torchfits_cached` | 49.3 MB/s |
| `torchfits_materialize` | 119.4 MB/s |
| `torchfits_naive` | 50.6 MB/s |
<!-- BENCH_MEGACAM_END -->


Keep this page current with the latest tensor and table benchmark run before
making performance claims.
