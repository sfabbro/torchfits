# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio. CPU↔GPU comparisons are published
when hardware was available; GPU deficits are listed, not hidden.

**Honesty:** torchfits is a **1.0.0rc5** prerelease. Headline ratios below are
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
exhaustives (`exhaustive_cpu_20260807_013736`,
`exhaustive_cuda_20260807_013736`, refreshed 2026-08-07) feed
the generated tables above; their CSVs are archived under
`benchmarks_results/<run-id>/` locally and on the benchmark hosts until the
next release mirrors them as assets. Example local paths from the Round-3
soak that are already mirrored:

- `docs/assets/bench/exhaustive_mps_20260719_143706/results.csv`
- `docs/assets/bench/exhaustive_cpu_20260719_144337/results.csv`
- `docs/assets/bench/exhaustive_cuda_20260719_144457/results.csv`

(also under `benchmarks_results/<run-id>/` locally)


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
Source: `benchmarks_results/exhaustive_cpu_20260807_013736/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.09 ms` (n=174) | `0.59 ms` (n=253) | `0.17 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.11 ms` (n=174) | `0.46 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.29 ms` (n=216) | `3.30 ms` (n=184) | `0.72 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.26 ms` (n=216) | `3.25 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Table read (100k rows, 8 cols, mixed) | CPU | **3.38 ms** | 3.30 ms | 52.55 ms | 16.85 ms | **15.93x** | **5.11x** |
| Varlen table read (100k rows, 3 cols) | CPU | **74.74 ms** | 10.80 ms | 528.67 ms | 107.37 ms | **48.93x** | **9.94x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

CPU category rows aggregate the CPU exhaustive
(`exhaustive_cpu_20260807_013736`, source of the generated
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
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 50.8 μs | 2.34 ms | 237.1 μs | — | **46.08x** | **4.66x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 49.3 μs | 2.36 ms | 233.4 μs | — | **47.97x** | **4.74x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 51.1 μs | 2.46 ms | 267.0 μs | — | **48.20x** | **5.22x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.33 ms** | 1.27 ms | 11.55 ms | 1.37 ms | — | **9.08x** | **1.08x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 37.9 μs | 1.77 ms | 192.8 μs | — | **46.68x** | **5.09x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 16.8 μs | 272.8 μs | 30.1 μs | — | **16.23x** | **1.79x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.9 μs | 299.1 μs | 30.4 μs | — | **17.69x** | **1.80x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 15.1 μs | 273.8 μs | 26.7 μs | — | **18.08x** | **1.76x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 17.5 μs | 301.7 μs | 30.3 μs | — | **17.28x** | **1.73x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 16.3 μs | 266.6 μs | 27.8 μs | — | **16.32x** | **1.70x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.5 μs | 297.0 μs | 30.5 μs | — | **18.04x** | **1.85x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 16.8 μs | 266.2 μs | 27.2 μs | — | **15.86x** | **1.62x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 15.6 μs | 295.3 μs | 28.2 μs | — | **18.91x** | **1.81x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 14.9 μs | 271.0 μs | 28.3 μs | — | **18.13x** | **1.90x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 17.1 μs | 286.1 μs | 28.3 μs | — | **16.75x** | **1.66x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 16.4 μs | 317.2 μs | 33.0 μs | — | **19.40x** | **2.02x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.2 μs | 339.1 μs | 32.9 μs | — | **19.72x** | **1.91x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 17.5 μs | 339.0 μs | 35.4 μs | — | **19.36x** | **2.02x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 18.6 μs | 347.8 μs | 42.0 μs | — | **18.74x** | **2.26x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 17.9 μs | 269.2 μs | 29.9 μs | — | **15.05x** | **1.67x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.6 μs | 294.1 μs | 29.4 μs | — | **17.67x** | **1.77x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 18.2 μs | 320.0 μs | 31.4 μs | — | **17.57x** | **1.72x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 16.8 μs | 282.8 μs | 28.9 μs | — | **16.82x** | **1.72x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 18.2 μs | 303.4 μs | 28.7 μs | — | **16.71x** | **1.58x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 16.1 μs | 315.5 μs | 31.1 μs | — | **19.56x** | **1.93x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 15.7 μs | 269.5 μs | 27.8 μs | — | **17.16x** | **1.77x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 18.0 μs | 295.8 μs | 32.2 μs | — | **16.43x** | **1.79x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 17.8 μs | 317.1 μs | 31.8 μs | — | **17.85x** | **1.79x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 14.9 μs | 273.2 μs | 28.0 μs | — | **18.28x** | **1.87x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.3 μs | 294.6 μs | 29.0 μs | — | **18.03x** | **1.77x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 15.5 μs | 312.1 μs | 29.4 μs | — | **20.10x** | **1.90x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 16.6 μs | 278.7 μs | 26.5 μs | — | **16.75x** | **1.59x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 17.0 μs | 300.8 μs | 29.9 μs | — | **17.66x** | **1.75x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 15.6 μs | 314.6 μs | 32.3 μs | — | **20.19x** | **2.08x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 16.9 μs | 309.6 μs | 33.1 μs | — | **18.35x** | **1.96x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 16.9 μs | 330.5 μs | 35.3 μs | — | **19.55x** | **2.09x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 16.7 μs | 347.4 μs | 36.9 μs | — | **20.76x** | **2.21x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 16.3 μs | 334.2 μs | 34.7 μs | — | **20.54x** | **2.13x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.0 μs | 329.8 μs | 37.5 μs | — | **19.45x** | **2.21x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 19.3 μs | 542.0 μs | 42.4 μs | — | **28.13x** | **2.20x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 18.3 μs | 528.1 μs | 42.7 μs | — | **28.90x** | **2.34x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **96.4 μs** | 129.7 μs | 3.79 ms | 275.6 μs | — | **39.32x** | **2.86x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 19.7 μs | 528.6 μs | 42.8 μs | — | **26.87x** | **2.18x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **8.15 ms** | 8.15 ms | 11.89 ms | 11.03 ms | — | **1.46x** | **1.35x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **698.5 μs** | 682.3 μs | 88.41 ms | 5.13 ms | — | **129.59x** | **7.52x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 29.6 μs | 642.7 μs | 66.4 μs | — | **21.69x** | **2.24x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 20.3 μs | 394.2 μs | 41.9 μs | — | **19.46x** | **2.07x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 16.4 μs | 332.8 μs | 34.6 μs | — | **20.34x** | **2.12x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 16.1 μs | 272.6 μs | 28.2 μs | — | **16.89x** | **1.75x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.7 μs | 290.4 μs | 30.7 μs | — | **19.71x** | **2.08x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 17.2 μs | 310.5 μs | 30.8 μs | — | **18.01x** | **1.79x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 15.1 μs | 265.9 μs | 27.5 μs | — | **17.59x** | **1.82x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.2 μs | 291.1 μs | 29.8 μs | — | **19.20x** | **1.97x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 16.6 μs | 315.1 μs | 31.3 μs | — | **19.03x** | **1.89x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 15.3 μs | 268.7 μs | 25.8 μs | — | **17.55x** | **1.69x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 14.3 μs | 283.6 μs | 26.8 μs | — | **19.87x** | **1.88x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 16.1 μs | 308.3 μs | 30.5 μs | — | **19.14x** | **1.89x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 15.8 μs | 257.1 μs | 26.3 μs | — | **16.27x** | **1.67x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.8 μs | 289.1 μs | 27.8 μs | — | **18.24x** | **1.75x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 16.8 μs | 298.1 μs | 29.7 μs | — | **17.78x** | **1.77x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 14.9 μs | 266.3 μs | 27.2 μs | — | **17.81x** | **1.82x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 14.7 μs | 285.4 μs | 28.6 μs | — | **19.36x** | **1.94x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 15.2 μs | 312.8 μs | 30.8 μs | — | **20.61x** | **2.03x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.9 μs | 314.8 μs | 32.7 μs | — | **18.66x** | **1.94x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 17.7 μs | 341.0 μs | 34.0 μs | — | **19.29x** | **1.92x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 17.1 μs | 356.7 μs | 35.0 μs | — | **20.91x** | **2.05x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 16.2 μs | 337.0 μs | 33.0 μs | — | **20.76x** | **2.04x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.3 μs | 344.9 μs | 34.0 μs | — | **21.21x** | **2.09x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 17.1 μs | 295.3 μs | 31.5 μs | — | **17.29x** | **1.84x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.9 μs | 297.6 μs | 31.2 μs | — | **17.62x** | **1.85x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.8 μs | 297.4 μs | 30.7 μs | — | **17.68x** | **1.83x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.9 μs | 295.3 μs | 30.3 μs | — | **18.58x** | **1.91x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 19.4 μs | 333.5 μs | 31.5 μs | — | **17.20x** | **1.62x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.7 μs | 278.2 μs | 26.7 μs | — | **17.77x** | **1.71x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 17.2 μs | 289.7 μs | 30.5 μs | — | **16.87x** | **1.78x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 17.2 μs | 315.6 μs | 32.4 μs | — | **18.30x** | **1.88x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.5 μs | 273.3 μs | 27.3 μs | — | **17.63x** | **1.76x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 16.7 μs | 291.2 μs | 29.8 μs | — | **17.41x** | **1.78x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 17.4 μs | 317.4 μs | 33.4 μs | — | **18.20x** | **1.91x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 15.2 μs | 270.6 μs | 25.3 μs | — | **17.85x** | **1.67x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.2 μs | 293.9 μs | 30.3 μs | — | **20.74x** | **2.14x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 17.1 μs | 308.9 μs | 32.3 μs | — | **18.07x** | **1.89x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.4 μs | 267.7 μs | 27.3 μs | — | **17.39x** | **1.77x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 15.1 μs | 291.9 μs | 28.2 μs | — | **19.38x** | **1.87x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 15.6 μs | 314.6 μs | 34.4 μs | — | **20.20x** | **2.21x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.7 μs | 269.5 μs | 27.8 μs | — | **17.17x** | **1.77x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 14.7 μs | 292.6 μs | 29.8 μs | — | **19.91x** | **2.03x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.4 μs | 311.7 μs | 30.3 μs | — | **18.99x** | **1.84x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 16.4 μs | 306.2 μs | 30.3 μs | — | **18.69x** | **1.85x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.0 μs | 334.7 μs | 35.1 μs | — | **20.86x** | **2.19x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 18.0 μs | 349.8 μs | 38.0 μs | — | **19.48x** | **2.12x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.04 ms** | — | 58.35 ms | — | — | **1.21x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.73 ms** | — | 68.78 ms | — | — | **1.82x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **23.63 ms** | 23.69 ms | 45.57 ms | 26.29 ms | — | **1.93x** | **1.11x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **20.18 ms** | 20.25 ms | 72.27 ms | 22.98 ms | — | **3.58x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **45.56 ms** | 45.54 ms | 51.44 ms | 44.34 ms | — | **1.13x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **12.45 ms** | 12.45 ms | 19.04 ms | 7.26 ms | — | **1.53x** | **0.58x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **459.9 μs** | 463.5 μs | 1.11 ms | 736.3 μs | — | **2.41x** | **1.60x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.57 ms** | 2.51 ms | 9.33 ms | 3.09 ms | — | **3.71x** | **1.23x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **796.8 μs** | 1.30 ms | 1.83 ms | 1.16 ms | — | **2.29x** | **1.46x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **5.59 ms** | 5.30 ms | 9.84 ms | 4.93 ms | — | **1.86x** | **0.93x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **257.0 μs** | 349.3 μs | 719.3 μs | 333.0 μs | — | **2.80x** | **1.30x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **914.1 μs** | 931.4 μs | 3.64 ms | 1.19 ms | — | **3.99x** | **1.30x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **530.6 μs** | 431.1 μs | 1.08 ms | 711.4 μs | — | **2.51x** | **1.65x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **1.61 ms** | 2.38 ms | 9.22 ms | 2.91 ms | — | **5.74x** | **1.81x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.23 ms** | 811.0 μs | 1.83 ms | 1.16 ms | — | **2.26x** | **1.43x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **4.73 ms** | 4.53 ms | 10.23 ms | 4.94 ms | — | **2.26x** | **1.09x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **159.8 μs** | 166.0 μs | 624.9 μs | 172.4 μs | — | **3.91x** | **1.08x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **505.7 μs** | 532.7 μs | 1.42 ms | 636.2 μs | — | **2.82x** | **1.26x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.19 ms** | 1.19 ms | 4.01 ms | 1.49 ms | — | **3.38x** | **1.25x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.91 ms** | 3.10 ms | 6.53 ms | 3.49 ms | — | **2.24x** | **1.20x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **48.1 μs** | 77.2 μs | 358.6 μs | 107.3 μs | — | **7.46x** | **2.23x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **492.9 μs** | 504.6 μs | 1.20 ms | 797.3 μs | — | **2.43x** | **1.62x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **679.6 μs** | 866.5 μs | 1.61 ms | 1.12 ms | — | **2.37x** | **1.65x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **87.3 μs** | 123.7 μs | 465.9 μs | 148.5 μs | — | **5.34x** | **1.70x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **863.0 μs** | 844.2 μs | 2.47 ms | 1.22 ms | — | **2.92x** | **1.45x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **1.27 ms** | 1.84 ms | 4.12 ms | 2.16 ms | — | **3.23x** | **1.70x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **36.7 μs** | 63.7 μs | 303.8 μs | 66.1 μs | — | **8.29x** | **1.80x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **288.2 μs** | 333.1 μs | 776.9 μs | 370.8 μs | — | **2.70x** | **1.29x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **426.1 μs** | 490.7 μs | 1.01 ms | 541.2 μs | — | **2.38x** | **1.27x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **83.4 μs** | 78.7 μs | 357.2 μs | 104.6 μs | — | **4.54x** | **1.33x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **471.6 μs** | 550.0 μs | 1.14 ms | 744.2 μs | — | **2.41x** | **1.58x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **730.4 μs** | 690.0 μs | 1.63 ms | 1.15 ms | — | **2.36x** | **1.66x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **120.5 μs** | 123.2 μs | 440.1 μs | 143.7 μs | — | **3.65x** | **1.19x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **849.7 μs** | 947.7 μs | 2.48 ms | 1.23 ms | — | **2.92x** | **1.45x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.00 ms** | 1.84 ms | 4.19 ms | 2.22 ms | — | **2.28x** | **1.21x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **53.1 μs** | 53.1 μs | 383.7 μs | 56.6 μs | — | **7.23x** | **1.07x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **165.3 μs** | 174.5 μs | 656.6 μs | 173.8 μs | — | **3.97x** | **1.05x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **252.0 μs** | 255.3 μs | 842.0 μs | 295.5 μs | — | **3.34x** | **1.17x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **386.7 μs** | 348.0 μs | 1.30 ms | 426.1 μs | — | **3.75x** | **1.22x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **623.7 μs** | 619.4 μs | 1.73 ms | 913.2 μs | — | **2.79x** | **1.47x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **166.9 μs** | 192.1 μs | 849.7 μs | 209.9 μs | — | **5.09x** | **1.26x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **42.6 μs** | 55.7 μs | 559.2 μs | 84.9 μs | — | **13.14x** | **2.00x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **56.0 μs** | 33.1 μs | 586.2 μs | 138.0 μs | — | **17.71x** | **4.17x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **3.43 ms** | 3.67 ms | 5.30 ms | 3.26 ms | — | **1.54x** | **0.95x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **626.6 μs** | 643.8 μs | 1.35 ms | 770.4 μs | — | **2.15x** | **1.23x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **63.0 μs** | 83.6 μs | 461.0 μs | 91.5 μs | — | **7.32x** | **1.45x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **33.0 μs** | 39.7 μs | 275.1 μs | 47.6 μs | — | **8.35x** | **1.44x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **69.1 μs** | 55.3 μs | 357.7 μs | 87.0 μs | — | **6.47x** | **1.57x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **79.6 μs** | 109.2 μs | 440.9 μs | 147.8 μs | — | **5.54x** | **1.86x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **45.8 μs** | 33.9 μs | 285.9 μs | 49.8 μs | — | **8.44x** | **1.47x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **67.2 μs** | 95.6 μs | 396.5 μs | 109.9 μs | — | **5.90x** | **1.63x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **181.7 μs** | 185.1 μs | 604.3 μs | 236.1 μs | — | **3.33x** | **1.30x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **30.0 μs** | 39.8 μs | 257.2 μs | 40.9 μs | — | **8.58x** | **1.36x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **39.2 μs** | 56.8 μs | 306.6 μs | 54.7 μs | — | **7.82x** | **1.40x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **50.3 μs** | 81.9 μs | 374.2 μs | 86.4 μs | — | **7.44x** | **1.72x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **28.0 μs** | 40.8 μs | 266.2 μs | 48.5 μs | — | **9.50x** | **1.73x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **63.8 μs** | 67.6 μs | 347.7 μs | 82.6 μs | — | **5.45x** | **1.29x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **67.9 μs** | 98.6 μs | 444.0 μs | 150.3 μs | — | **6.54x** | **2.21x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **43.0 μs** | 42.9 μs | 278.8 μs | 48.6 μs | — | **6.50x** | **1.13x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **95.9 μs** | 78.8 μs | 396.8 μs | 109.8 μs | — | **5.03x** | **1.39x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **187.4 μs** | 159.3 μs | 603.4 μs | 232.7 μs | — | **3.79x** | **1.46x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **28.4 μs** | 48.4 μs | 361.6 μs | 42.5 μs | — | **12.74x** | **1.50x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **44.5 μs** | 38.4 μs | 391.9 μs | 55.0 μs | — | **10.20x** | **1.43x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **59.6 μs** | 60.8 μs | 426.7 μs | 65.5 μs | — | **7.16x** | **1.10x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **55.2 μs** | 43.2 μs | 380.1 μs | 60.0 μs | — | **8.80x** | **1.39x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **41.9 μs** | 74.0 μs | 432.1 μs | 94.2 μs | — | **10.32x** | **2.25x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **126.9 μs** | 135.4 μs | 562.9 μs | 137.4 μs | — | **4.44x** | **1.08x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **76.2 μs** | 90.3 μs | 562.7 μs | 139.6 μs | — | **7.39x** | **1.83x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **91.1 μs** | 84.8 μs | 562.0 μs | 136.5 μs | — | **6.62x** | **1.61x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **89.8 μs** | 71.3 μs | 553.5 μs | 129.7 μs | — | **7.77x** | **1.82x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **82.6 μs** | 90.1 μs | 559.9 μs | 136.3 μs | — | **6.78x** | **1.65x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **65.0 μs** | 51.7 μs | 438.9 μs | 72.4 μs | — | **8.48x** | **1.40x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **58.6 μs** | 62.9 μs | 471.5 μs | 70.7 μs | — | **8.05x** | **1.21x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **48.2 μs** | 57.1 μs | 496.0 μs | 75.6 μs | — | **10.30x** | **1.57x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **59.7 μs** | 65.5 μs | 456.7 μs | 71.3 μs | — | **7.65x** | **1.19x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **51.2 μs** | 65.3 μs | 488.5 μs | 79.6 μs | — | **9.54x** | **1.55x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **61.2 μs** | 67.2 μs | 508.5 μs | 77.3 μs | — | **8.31x** | **1.26x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **64.2 μs** | 46.6 μs | 424.3 μs | 69.4 μs | — | **9.11x** | **1.49x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **50.4 μs** | 62.8 μs | 463.0 μs | 74.6 μs | — | **9.19x** | **1.48x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **66.0 μs** | 64.2 μs | 500.2 μs | 74.6 μs | — | **7.79x** | **1.16x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **60.2 μs** | 46.6 μs | 440.9 μs | 71.8 μs | — | **9.47x** | **1.54x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **44.1 μs** | 51.6 μs | 482.5 μs | 77.3 μs | — | **10.93x** | **1.75x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **53.7 μs** | 61.6 μs | 509.3 μs | 78.4 μs | — | **9.49x** | **1.46x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **58.9 μs** | 52.5 μs | 455.9 μs | 75.9 μs | — | **8.68x** | **1.44x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **64.2 μs** | 45.7 μs | 470.8 μs | 70.3 μs | — | **10.30x** | **1.54x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **47.6 μs** | 69.0 μs | 496.1 μs | 78.2 μs | — | **10.43x** | **1.64x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **49.4 μs** | 40.8 μs | 611.3 μs | 78.1 μs | — | **14.98x** | **1.91x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **55.9 μs** | 53.6 μs | 638.6 μs | 79.6 μs | — | **11.92x** | **1.49x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **45.0 μs** | 64.6 μs | 659.0 μs | 80.1 μs | — | **14.64x** | **1.78x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.77 ms** | 23.70 ms | 46.38 ms | 26.41 ms | — | **1.96x** | **1.11x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.32 ms** | 20.36 ms | 72.52 ms | 23.06 ms | — | **3.57x** | **1.13x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.70 ms** | 45.72 ms | 51.70 ms | 44.51 ms | — | **1.13x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.65 ms** | 12.63 ms | 33.26 ms | 12.64 ms | — | **2.63x** | **1.00x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **667.0 μs** | 803.9 μs | 1.49 ms | — | — | **2.24x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.91 ms** | 4.03 ms | 13.50 ms | — | — | **3.45x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.28 ms** | 1.23 ms | 2.46 ms | — | — | **2.00x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **8.56 ms** | 7.74 ms | 12.67 ms | — | — | **1.64x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **373.6 μs** | 398.4 μs | 1.04 ms | — | — | **2.78x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.37 ms** | 1.33 ms | 2.58 ms | — | — | **1.93x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **907.4 μs** | 848.8 μs | 1.48 ms | — | — | **1.74x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **4.08 ms** | 3.95 ms | 12.84 ms | — | — | **3.25x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.32 ms** | 1.35 ms | 2.40 ms | — | — | **1.82x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **6.93 ms** | 6.76 ms | 11.45 ms | — | — | **1.69x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **250.4 μs** | 253.6 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **828.3 μs** | 786.8 μs | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.31 ms** | 1.33 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **5.33 ms** | 4.53 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **113.2 μs** | 120.9 μs | 605.2 μs | — | — | **5.35x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **709.9 μs** | 724.4 μs | 1.57 ms | — | — | **2.22x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.03 ms** | 1.02 ms | 2.13 ms | — | — | **2.10x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **174.5 μs** | 174.7 μs | 705.2 μs | — | — | **4.04x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.29 ms** | 1.57 ms | 2.52 ms | — | — | **1.96x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.93 ms** | 2.46 ms | 3.68 ms | — | — | **1.91x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **117.3 μs** | 108.0 μs | 518.2 μs | — | — | **4.80x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **428.1 μs** | 388.4 μs | 1.07 ms | — | — | **2.75x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **587.7 μs** | 560.9 μs | 1.36 ms | — | — | **2.43x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **103.9 μs** | 147.4 μs | 587.3 μs | — | — | **5.65x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **798.5 μs** | 790.0 μs | 1.55 ms | — | — | **1.97x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.29 ms** | 1.21 ms | 2.15 ms | — | — | **1.77x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **212.8 μs** | 192.9 μs | 706.3 μs | — | — | **3.66x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.56 ms** | 1.31 ms | 2.52 ms | — | — | **1.92x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.91 ms** | 1.97 ms | 3.70 ms | — | — | **1.93x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **68.7 μs** | 92.6 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **265.2 μs** | 205.3 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **371.2 μs** | 397.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **419.4 μs** | 414.2 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.18 ms** | 1.18 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **273.6 μs** | 259.8 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **95.7 μs** | 53.3 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **71.9 μs** | 76.7 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **2.43 ms** | 3.79 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **650.3 μs** | 650.9 μs | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **88.2 μs** | 88.2 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **25.2 μs** | 41.4 μs | 285.5 μs | — | — | **11.32x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **61.8 μs** | 45.9 μs | 366.7 μs | — | — | **7.99x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **102.9 μs** | 113.1 μs | 456.6 μs | — | — | **4.44x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **30.9 μs** | 44.4 μs | 285.0 μs | — | — | **9.24x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **86.0 μs** | 74.3 μs | 412.2 μs | — | — | **5.55x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **176.0 μs** | 179.3 μs | 580.3 μs | — | — | **3.30x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **37.8 μs** | 49.5 μs | 275.5 μs | — | — | **7.29x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **55.4 μs** | 78.2 μs | 321.3 μs | — | — | **5.80x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **87.8 μs** | 84.4 μs | 385.4 μs | — | — | **4.57x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **42.3 μs** | 51.5 μs | 279.9 μs | — | — | **6.62x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **88.0 μs** | 75.0 μs | 355.8 μs | — | — | **4.75x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **119.2 μs** | 112.5 μs | 458.2 μs | — | — | **4.07x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **41.8 μs** | 62.5 μs | 315.3 μs | — | — | **7.54x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **102.7 μs** | 74.1 μs | 422.8 μs | — | — | **5.70x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **195.7 μs** | 171.6 μs | 580.9 μs | — | — | **3.38x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **30.2 μs** | 50.7 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **54.3 μs** | 30.1 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **68.7 μs** | 34.5 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **81.6 μs** | 51.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **78.8 μs** | 110.3 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **55.8 μs** | 49.9 μs | 364.3 μs | — | — | **7.30x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **66.7 μs** | 61.2 μs | 354.8 μs | — | — | **5.80x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **36.2 μs** | 66.3 μs | 361.0 μs | — | — | **9.98x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **60.8 μs** | 63.1 μs | 361.4 μs | — | — | **5.95x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **64.0 μs** | 43.7 μs | 355.9 μs | — | — | **8.14x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **27.6 μs** | 27.7 μs | 275.2 μs | — | — | **9.97x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **29.4 μs** | 31.8 μs | 294.8 μs | — | — | **10.02x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **41.9 μs** | 25.1 μs | 300.1 μs | — | — | **11.97x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **22.9 μs** | 39.6 μs | 273.6 μs | — | — | **11.97x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **33.1 μs** | 26.9 μs | 287.0 μs | — | — | **10.67x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **32.5 μs** | 35.4 μs | 306.4 μs | — | — | **9.44x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **31.6 μs** | 32.4 μs | 265.2 μs | — | — | **8.40x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **49.5 μs** | 26.4 μs | 287.2 μs | — | — | **10.90x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **44.1 μs** | 39.0 μs | 291.8 μs | — | — | **7.48x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **47.4 μs** | 36.5 μs | 272.7 μs | — | — | **7.48x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **30.5 μs** | 36.0 μs | 292.6 μs | — | — | **9.61x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **46.5 μs** | 39.4 μs | 302.1 μs | — | — | **7.67x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **33.3 μs** | 29.1 μs | 268.3 μs | — | — | **9.22x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **30.4 μs** | 53.7 μs | 290.1 μs | — | — | **9.55x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **46.6 μs** | 49.2 μs | 306.9 μs | — | — | **6.59x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **46.1 μs** | 36.3 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **51.6 μs** | 29.3 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **46.2 μs** | 29.4 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **382.6 μs** | 379.7 μs | 2.66 ms | 356.9 μs | — | **7.02x** | **0.94x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **370.6 μs** | 371.9 μs | 2.64 ms | 359.8 μs | — | **7.12x** | **0.97x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.02 ms** | 984.1 μs | 7.98 ms | 1.97 ms | — | **8.11x** | **2.00x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.01 ms** | 979.2 μs | 8.00 ms | 1.96 ms | — | **8.17x** | **2.00x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **194.8 μs** | 162.3 μs | 2.58 ms | 513.1 μs | — | **15.87x** | **3.16x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **25.6 μs** | 24.4 μs | 403.5 μs | 66.6 μs | — | **16.57x** | **2.73x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **140.9 μs** | 144.7 μs | 1.51 ms | 160.6 μs | — | **10.72x** | **1.14x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **143.3 μs** | 145.3 μs | 1.50 ms | 162.0 μs | — | **10.43x** | **1.13x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **188.5 μs** | 167.5 μs | 2.15 ms | 333.5 μs | — | **12.85x** | **1.99x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **184.1 μs** | 170.5 μs | 2.16 ms | 321.9 μs | — | **12.65x** | **1.89x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **112.6 μs** | 94.5 μs | 1.93 ms | 191.0 μs | — | **20.45x** | **2.02x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **26.7 μs** | 27.5 μs | 415.4 μs | 63.0 μs | — | **15.53x** | **2.36x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **11.13 ms** | 10.52 ms | 15.06 ms | 20.26 ms | — | **1.43x** | **1.93x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **9.53 ms** | 11.01 ms | 18.18 ms | 17.80 ms | — | **1.91x** | **1.87x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.10 ms** | 11.24 ms | 17.34 ms | 30.87 ms | — | **1.72x** | **3.06x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **30.32 ms** | 30.54 ms | 334.99 ms | 116.95 ms | — | **11.05x** | **3.86x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **290.8 μs** | 286.1 μs | 12.78 ms | 1.57 ms | — | **44.67x** | **5.48x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **38.3 μs** | 30.8 μs | 447.2 μs | 80.5 μs | — | **14.52x** | **2.61x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **2.17 ms** | 2.01 ms | 5.15 ms | 3.71 ms | — | **2.56x** | **1.84x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.83 ms** | 1.87 ms | 4.61 ms | 3.04 ms | — | **2.52x** | **1.66x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.79 ms** | 1.75 ms | 5.13 ms | 5.35 ms | — | **2.92x** | **3.05x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **3.38 ms** | 3.30 ms | 52.55 ms | 16.85 ms | — | **15.93x** | **5.11x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **478.1 μs** | 438.2 μs | 10.02 ms | 2.55 ms | — | **22.86x** | **5.81x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **45.2 μs** | 48.2 μs | 767.5 μs | 134.3 μs | — | **16.99x** | **2.97x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **389.1 μs** | 460.8 μs | 3.31 ms | 622.5 μs | — | **8.52x** | **1.60x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **312.5 μs** | 370.1 μs | 3.24 ms | 554.9 μs | — | **10.36x** | **1.78x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **292.1 μs** | 246.3 μs | 3.28 ms | 782.3 μs | — | **13.30x** | **3.18x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **466.7 μs** | 440.5 μs | 7.84 ms | 1.83 ms | — | **17.80x** | **4.14x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **222.7 μs** | 203.3 μs | 5.12 ms | 532.7 μs | — | **25.18x** | **2.62x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **28.1 μs** | 29.6 μs | 440.7 μs | 76.0 μs | — | **15.71x** | **2.71x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **138.7 μs** | 212.5 μs | 3.12 ms | 309.5 μs | — | **22.52x** | **2.23x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **130.5 μs** | 207.9 μs | 3.11 ms | 299.6 μs | — | **23.82x** | **2.30x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **176.4 μs** | 143.2 μs | 3.13 ms | 322.1 μs | — | **21.83x** | **2.25x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **217.5 μs** | 195.1 μs | 3.73 ms | 442.8 μs | — | **19.10x** | **2.27x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **197.5 μs** | 180.0 μs | 4.60 ms | 332.3 μs | — | **25.57x** | **1.85x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **46.5 μs** | 46.7 μs | 784.1 μs | 123.3 μs | — | **16.85x** | **2.65x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **6.27 ms** | 5.58 ms | 8.48 ms | 14.73 ms | — | **1.52x** | **2.64x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **4.84 ms** | 4.84 ms | 5.05 ms | 11.15 ms | — | **1.04x** | **2.30x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.43 ms** | 4.40 ms | 5.46 ms | 23.54 ms | — | **1.24x** | **5.34x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **6.00 ms** | 5.93 ms | 6.69 ms | 5.85 ms | — | **1.13x** | **0.99x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **159.8 μs** | 148.5 μs | 3.84 ms | 566.0 μs | — | **25.87x** | **3.81x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **28.1 μs** | 29.1 μs | 440.9 μs | 67.5 μs | — | **15.69x** | **2.40x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.33 ms** | 1.17 ms | 3.58 ms | 2.79 ms | — | **3.06x** | **2.38x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **970.3 μs** | 1.01 ms | 2.93 ms | 2.16 ms | — | **3.02x** | **2.22x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **885.6 μs** | 847.5 μs | 2.94 ms | 4.31 ms | — | **3.46x** | **5.09x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **1.22 ms** | 1.13 ms | 3.14 ms | 1.11 ms | — | **2.79x** | **0.99x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **261.4 μs** | 237.1 μs | 3.46 ms | 969.6 μs | — | **14.61x** | **4.09x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **42.7 μs** | 44.7 μs | 745.9 μs | 99.6 μs | — | **17.49x** | **2.34x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **304.9 μs** | 366.6 μs | 2.41 ms | 502.3 μs | — | **7.91x** | **1.65x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **225.8 μs** | 295.4 μs | 2.33 ms | 440.8 μs | — | **10.30x** | **1.95x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **228.5 μs** | 187.8 μs | 2.32 ms | 651.0 μs | — | **12.35x** | **3.47x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **258.9 μs** | 225.2 μs | 2.36 ms | 306.1 μs | — | **10.47x** | **1.36x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **172.0 μs** | 136.1 μs | 3.04 ms | 319.7 μs | — | **22.32x** | **2.35x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **42.2 μs** | 41.0 μs | 733.9 μs | 104.8 μs | — | **17.89x** | **2.55x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **131.4 μs** | 203.1 μs | 2.25 ms | 277.0 μs | — | **17.15x** | **2.11x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **125.0 μs** | 198.6 μs | 2.24 ms | 268.0 μs | — | **17.93x** | **2.14x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **170.7 μs** | 182.6 μs | 3.43 ms | 424.8 μs | — | **20.12x** | **2.49x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **175.1 μs** | 145.8 μs | 2.30 ms | 239.5 μs | — | **15.79x** | **1.64x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **167.1 μs** | 126.7 μs | 3.01 ms | 248.8 μs | — | **23.75x** | **1.96x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **42.4 μs** | 46.2 μs | 762.0 μs | 111.8 μs | — | **17.97x** | **2.64x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **841.1 μs** | 851.5 μs | 1.86 ms | 1.36 ms | — | **2.21x** | **1.61x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **852.0 μs** | 851.1 μs | 1.86 ms | 1.36 ms | — | **2.19x** | **1.60x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.49 ms** | 3.41 ms | 28.96 ms | 13.18 ms | — | **8.50x** | **3.87x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.11 ms** | 5.10 ms | 29.04 ms | 14.09 ms | — | **5.69x** | **2.76x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **620.7 μs** | 596.9 μs | 4.72 ms | 1.91 ms | — | **7.92x** | **3.20x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **30.8 μs** | 29.0 μs | 445.7 μs | 67.9 μs | — | **15.39x** | **2.35x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **178.9 μs** | 213.2 μs | 1.43 ms | 279.7 μs | — | **7.99x** | **1.56x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **176.3 μs** | 222.1 μs | 1.44 ms | 278.9 μs | — | **8.17x** | **1.58x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **454.2 μs** | 423.7 μs | 4.08 ms | 1.46 ms | — | **9.63x** | **3.45x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **616.5 μs** | 597.2 μs | 4.12 ms | 1.59 ms | — | **6.90x** | **2.67x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **160.4 μs** | 137.2 μs | 2.17 ms | 389.2 μs | — | **15.85x** | **2.84x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **28.1 μs** | 26.4 μs | 415.8 μs | 64.5 μs | — | **15.77x** | **2.45x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **870.5 μs** | 732.3 μs | 1.73 ms | 1.23 ms | — | **2.37x** | **1.68x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **695.0 μs** | 724.4 μs | 1.71 ms | 1.24 ms | — | **2.45x** | **1.78x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **72.73 ms** | 10.41 ms | 527.03 ms | 106.65 ms | — | **50.64x** | **10.25x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **74.74 ms** | 10.80 ms | 528.67 ms | 107.37 ms | — | **48.93x** | **9.94x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **6.96 ms** | 1.04 ms | 54.18 ms | 11.73 ms | — | **51.96x** | **11.25x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **28.1 μs** | 27.9 μs | 447.1 μs | 64.0 μs | — | **16.03x** | **2.30x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **165.9 μs** | 216.0 μs | 1.33 ms | 269.7 μs | — | **8.00x** | **1.62x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **155.1 μs** | 211.4 μs | 1.34 ms | 265.6 μs | — | **8.62x** | **1.71x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **6.97 ms** | 1.05 ms | 53.21 ms | 10.73 ms | — | **50.56x** | **10.19x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **6.98 ms** | 1.05 ms | 53.16 ms | 10.76 ms | — | **50.49x** | **10.22x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **808.0 μs** | 184.5 μs | 6.98 ms | 1.36 ms | — | **37.82x** | **7.35x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **28.5 μs** | 26.5 μs | 428.5 μs | 60.1 μs | — | **16.16x** | **2.27x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **70.8 μs** | 129.3 μs | 1.28 ms | 161.7 μs | — | **18.09x** | **2.28x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **72.3 μs** | 129.6 μs | 1.28 ms | 166.5 μs | — | **17.65x** | **2.30x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **793.4 μs** | 195.3 μs | 6.58 ms | 1.25 ms | — | **33.72x** | **6.43x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **801.8 μs** | 193.1 μs | 6.59 ms | 1.24 ms | — | **34.10x** | **6.44x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **197.9 μs** | 102.2 μs | 2.22 ms | 296.3 μs | — | **21.76x** | **2.90x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **25.9 μs** | 25.3 μs | 425.0 μs | 61.1 μs | — | **16.81x** | **2.42x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **3.43 ms** | 3.20 ms | 8.31 ms | 4.17 ms | — | **2.60x** | **1.31x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **3.03 ms** | 3.03 ms | 7.91 ms | 3.84 ms | — | **2.61x** | **1.27x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **4.70 ms** | 4.63 ms | 14.14 ms | 8.61 ms | — | **3.05x** | **1.86x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **31.16 ms** | 31.17 ms | 227.80 ms | 70.80 ms | — | **7.31x** | **2.27x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **2.11 ms** | 1.41 ms | 21.57 ms | 4.98 ms | — | **15.28x** | **3.53x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **38.6 μs** | 37.9 μs | 559.9 μs | 256.0 μs | — | **14.76x** | **6.75x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **789.2 μs** | 868.4 μs | 10.64 ms | 1.27 ms | — | **13.48x** | **1.60x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **693.3 μs** | 746.9 μs | 10.12 ms | 1.20 ms | — | **14.60x** | **1.73x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **429.4 μs** | 403.5 μs | 5.93 ms | 871.7 μs | — | **14.71x** | **2.16x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.48 ms** | 1.48 ms | 17.15 ms | 4.52 ms | — | **11.59x** | **3.05x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **850.6 μs** | 856.6 μs | 18.23 ms | 1.50 ms | — | **21.43x** | **1.76x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **65.1 μs** | 63.3 μs | 943.9 μs | 426.7 μs | — | **14.91x** | **6.74x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **233.3 μs** | 288.1 μs | 9.72 ms | 651.7 μs | — | **41.64x** | **2.79x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **213.2 μs** | 278.8 μs | 9.75 ms | 642.1 μs | — | **45.75x** | **3.01x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **260.8 μs** | 225.7 μs | 9.76 ms | 656.1 μs | — | **43.25x** | **2.91x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **830.7 μs** | 854.6 μs | 12.30 ms | 1.38 ms | — | **14.81x** | **1.67x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **732.1 μs** | 741.6 μs | 16.24 ms | 842.4 μs | — | **22.19x** | **1.15x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **60.4 μs** | 62.5 μs | 942.1 μs | 423.5 μs | — | **15.59x** | **7.01x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **411.7 μs** | 417.2 μs | 2.66 ms | — | — | **6.46x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **402.6 μs** | 409.3 μs | 2.67 ms | — | — | **6.63x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.08 ms** | 1.07 ms | 8.03 ms | — | — | **7.49x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.08 ms** | 1.06 ms | 8.01 ms | — | — | **7.56x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **236.1 μs** | 217.8 μs | 2.57 ms | — | — | **11.78x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **24.5 μs** | 24.8 μs | 407.3 μs | — | — | **16.64x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **177.7 μs** | 189.6 μs | 1.51 ms | — | — | **8.52x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **184.8 μs** | 181.6 μs | 1.51 ms | — | — | **8.33x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **227.2 μs** | 217.4 μs | 2.18 ms | — | — | **10.03x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **233.5 μs** | 212.1 μs | 2.19 ms | — | — | **10.34x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **162.7 μs** | 135.2 μs | 1.94 ms | — | — | **14.38x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **27.7 μs** | 30.7 μs | 486.7 μs | — | — | **17.58x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.40 ms** | 4.95 ms | 11.96 ms | — | — | **2.42x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.15 ms** | 4.18 ms | 8.67 ms | — | — | **2.09x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **7.33 ms** | 7.95 ms | 12.92 ms | — | — | **1.76x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **28.77 ms** | 25.37 ms | 337.72 ms | — | — | **13.31x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **276.5 μs** | 217.8 μs | 9.07 ms | — | — | **41.66x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **33.4 μs** | 31.2 μs | 466.8 μs | — | — | **14.98x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.25 ms** | 1.02 ms | 4.83 ms | — | — | **4.74x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **878.9 μs** | 873.7 μs | 4.27 ms | — | — | **4.89x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **1.31 ms** | 1.21 ms | 4.83 ms | — | — | **3.99x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **2.77 ms** | 2.68 ms | 52.13 ms | — | — | **19.49x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **422.5 μs** | 328.6 μs | 9.68 ms | — | — | **29.47x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **48.8 μs** | 49.2 μs | 794.2 μs | — | — | **16.29x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **313.8 μs** | 349.2 μs | 3.30 ms | — | — | **10.51x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **225.6 μs** | 267.5 μs | 3.24 ms | — | — | **14.37x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **269.3 μs** | 192.0 μs | 3.26 ms | — | — | **16.99x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **410.3 μs** | 316.9 μs | 7.83 ms | — | — | **24.70x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **260.1 μs** | 180.9 μs | 5.07 ms | — | — | **28.05x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **47.9 μs** | 46.5 μs | 759.6 μs | — | — | **16.33x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **142.9 μs** | 195.3 μs | 3.17 ms | — | — | **22.17x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **283.2 μs** | 193.4 μs | 3.13 ms | — | — | **16.17x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **216.5 μs** | 142.5 μs | 3.13 ms | — | — | **21.97x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **258.0 μs** | 179.2 μs | 3.76 ms | — | — | **20.99x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **251.5 μs** | 162.1 μs | 4.63 ms | — | — | **28.55x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **45.8 μs** | 48.1 μs | 785.4 μs | — | — | **17.14x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.07 ms** | 2.40 ms | 8.00 ms | — | — | **3.34x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.61 ms** | 1.60 ms | 4.66 ms | — | — | **2.91x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.01 ms** | 1.97 ms | 5.05 ms | — | — | **2.56x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.13 ms** | 3.05 ms | 6.28 ms | — | — | **2.06x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **172.8 μs** | 118.4 μs | 3.42 ms | — | — | **28.90x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **25.6 μs** | 27.3 μs | 437.4 μs | — | — | **17.07x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **799.7 μs** | 689.2 μs | 3.49 ms | — | — | **5.06x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **460.0 μs** | 464.2 μs | 2.90 ms | — | — | **6.29x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **478.1 μs** | 380.9 μs | 2.89 ms | — | — | **7.58x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **661.5 μs** | 585.9 μs | 3.10 ms | — | — | **5.28x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **255.7 μs** | 170.8 μs | 3.41 ms | — | — | **19.99x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **44.1 μs** | 49.1 μs | 757.8 μs | — | — | **17.18x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **269.4 μs** | 320.4 μs | 2.42 ms | — | — | **8.97x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **184.7 μs** | 235.5 μs | 2.34 ms | — | — | **12.65x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **223.6 μs** | 137.8 μs | 2.34 ms | — | — | **16.96x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **250.7 μs** | 165.3 μs | 2.40 ms | — | — | **14.55x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **210.5 μs** | 128.1 μs | 3.05 ms | — | — | **23.82x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **45.9 μs** | 43.3 μs | 757.7 μs | — | — | **17.51x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **124.0 μs** | 198.7 μs | 2.27 ms | — | — | **18.29x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **127.9 μs** | 192.2 μs | 2.28 ms | — | — | **17.84x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **198.3 μs** | 124.9 μs | 2.28 ms | — | — | **18.24x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **201.4 μs** | 124.0 μs | 2.32 ms | — | — | **18.72x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **205.5 μs** | 131.7 μs | 3.03 ms | — | — | **23.04x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **45.3 μs** | 47.7 μs | 776.3 μs | — | — | **17.14x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **614.7 μs** | 476.2 μs | 1.75 ms | — | — | **3.68x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **442.8 μs** | 465.3 μs | 1.79 ms | — | — | **4.05x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.23 ms** | 1.17 ms | 28.68 ms | — | — | **24.62x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.36 ms** | 1.26 ms | 28.93 ms | — | — | **22.95x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **266.0 μs** | 210.3 μs | 4.48 ms | — | — | **21.31x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **30.3 μs** | 27.3 μs | 442.3 μs | — | — | **16.18x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **154.9 μs** | 189.9 μs | 1.44 ms | — | — | **9.26x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **152.9 μs** | 181.8 μs | 1.43 ms | — | — | **9.36x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **249.2 μs** | 198.7 μs | 4.03 ms | — | — | **20.29x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **257.7 μs** | 202.7 μs | 4.08 ms | — | — | **20.14x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **149.0 μs** | 95.5 μs | 2.17 ms | — | — | **22.73x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **27.9 μs** | 27.8 μs | 432.7 μs | — | — | **15.55x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **578.4 μs** | 428.0 μs | 1.58 ms | — | — | **3.69x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **389.5 μs** | 398.7 μs | 1.60 ms | — | — | **4.10x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **74.18 ms** | 10.94 ms | 530.40 ms | — | — | **48.46x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **75.00 ms** | 10.93 ms | 531.59 ms | — | — | **48.64x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.07 ms** | 1.12 ms | 54.19 ms | — | — | **48.31x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **28.1 μs** | 28.2 μs | 446.6 μs | — | — | **15.91x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **152.2 μs** | 183.4 μs | 1.32 ms | — | — | **8.67x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **149.9 μs** | 175.7 μs | 1.31 ms | — | — | **8.76x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **7.07 ms** | 1.14 ms | 53.80 ms | — | — | **47.28x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **7.05 ms** | 1.15 ms | 54.03 ms | — | — | **47.02x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **866.5 μs** | 284.1 μs | 7.02 ms | — | — | **24.69x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **25.9 μs** | 26.7 μs | 428.1 μs | — | — | **16.56x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **82.1 μs** | 130.6 μs | 1.27 ms | — | — | **15.52x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **79.3 μs** | 126.8 μs | 1.28 ms | — | — | **16.17x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **861.0 μs** | 280.6 μs | 6.62 ms | — | — | **23.59x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **870.5 μs** | 238.9 μs | 6.60 ms | — | — | **27.64x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **262.0 μs** | 148.8 μs | 2.24 ms | — | — | **15.04x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **26.6 μs** | 26.8 μs | 424.9 μs | — | — | **15.96x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.58 ms** | 1.37 ms | 7.36 ms | — | — | **5.38x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.22 ms** | 1.24 ms | 7.06 ms | — | — | **5.81x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.67 ms** | 1.60 ms | 7.58 ms | — | — | **4.73x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **24.64 ms** | 13.77 ms | 128.56 ms | — | — | **9.34x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.03 ms** | 977.6 μs | 20.76 ms | — | — | **21.24x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **36.9 μs** | 36.9 μs | 565.5 μs | — | — | **15.32x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **483.9 μs** | 510.7 μs | 10.02 ms | — | — | **20.70x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **398.8 μs** | 429.5 μs | 9.95 ms | — | — | **24.94x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **451.2 μs** | 370.8 μs | 9.98 ms | — | — | **26.91x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.64 ms** | 1.53 ms | 28.54 ms | — | — | **18.67x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **837.8 μs** | 759.7 μs | 17.93 ms | — | — | **23.60x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **62.7 μs** | 62.1 μs | 973.5 μs | — | — | **15.67x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **197.0 μs** | 252.3 μs | 9.65 ms | — | — | **48.97x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **193.8 μs** | 251.3 μs | 9.69 ms | — | — | **50.00x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **280.8 μs** | 207.2 μs | 9.64 ms | — | — | **46.55x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **832.6 μs** | 745.1 μs | 12.23 ms | — | — | **16.41x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **760.6 μs** | 695.5 μs | 16.13 ms | — | — | **23.19x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **62.6 μs** | 62.6 μs | 985.6 μs | — | — | **15.75x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.70 ms | 293.8 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.56 ms | 309.3 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.72 ms | 293.8 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.54 ms | 309.3 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 1.22 ms | 380.4 | fitsio/fitsio_torch | 1.36× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 6.00 ms | 399.4 | fitsio/fitsio_torch | 1.21× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter] | off | 382.6 μs | 422.3 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter] | off | 379.7 μs | 422.3 | fitsio/fitsio | 1.06× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter_selective] | off | 371.9 μs | 422.3 | fitsio/fitsio | 1.03× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 1.13 ms | 380.4 | fitsio/fitsio | 1.01× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 5.93 ms | 403.5 | fitsio/fitsio | 1.01× |
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
| Linux x86_64 / CUDA | tensor | small_int16_3d [read_full] | off | 156.1 μs | 765.8 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 7.68 ms | 715.5 | fitsio/fitsio_torch | 1.08× |
| Linux x86_64 / CUDA | table | typed_100000 [predicate_filter] | off | 2.09 ms | 738.1 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | table | typed_100000 [predicate_filter] | off | 2.10 ms | 738.1 | fitsio/fitsio | 1.03× |
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260807_013736` | 3057 | 11 | 293.8 | lab + mmap-matrix |
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
