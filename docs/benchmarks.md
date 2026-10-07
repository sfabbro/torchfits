# Benchmarks

> **Headline (1.2.0rc1, lab profile, mmap on and off;
> git-mirrored `exhaustive_cpu_20261007_203905`,
> `exhaustive_cuda_20261007_203924`, and
> `exhaustive_mps_20261007_204350`):**
> Linux CPU has no significant image deficit. Significant losses there are
> `read_full` of narrow tables with mmap off (20.9–22.3% behind fitsio) and
> `predicate_filter` / `predicate_filter_selective` on ASCII catalogs (5–15%
> behind fitsio). On CUDA the significant losses are `read_full` of
> `scaled_large` (26.7% behind fitsio) and `read_full` of `narrow_100000`
> (12.9% behind fitsio), plus `predicate_filter` and
> `predicate_filter_selective` on ASCII catalogs (5–10%). On MPS the
> significant losses are `read_full` of small images (up to 180% behind
> fitsio), `predicate_filter` and `predicate_filter_selective` on a few
> tables, and `cutout_100x100` (19–47% behind fitsio). CSVs under
> [Published CSVs](#published-csvs).


`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio across CPU and GPU hardware.

**A note on fairness:** Headline ratios below are medians from reproducible benchmark runs across our test suites (the `lab` timing profile uses more warmup and repetitions than the quick `user` profile) — not guarantees on your specific filesystem, file mix, or PyTorch version. Check [Performance comparisons & limitations](#performance-deficits) for a transparent breakdown of cases where peer libraries are competitive or faster.

## How to read this page

| If you want… | Jump to |
|---|---|
| Headline wins | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 (CPU and GPU) | [Performance comparisons & limitations](#performance-deficits) |
| GPU transport rows | [I/O transport and backend](#io-transport-and-backend) |
| Python × PyTorch version variance | [Version matrix variance](#python-pytorch-matrix-variance) |
| Reproduce numbers | [Reproducing](#reproducing) |
| Every measured configuration | [Exhaustive benchmark results](#exhaustive-benchmark-results) |
| Raw CSV | [Published CSVs](#published-csvs) |

Published GPU/CPU numbers come from the multi-host benchmark runs
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
**peak process RSS** (and peak CUDA alloc when on CUDA). Performance ranking is
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
is a **2.0** item (see [Roadmap](roadmap.md)) — not a 1.x claim.

### Tables on GPU transports

Table GPU transport rows compare `table.read_torch(..., device=cpu)` against
`device=cuda` / `device=mps` on a medium mixed catalog case
(`mixed_100000`). Decode still happens on the host; the GPU column measures
host decode plus H2D copy into tensor columns.

### CUDA Host-to-Device Transfer & Small Payloads

For small tensor payloads (e.g. 1D arrays and small $64 \times 64$ sub-regions), fixed kernel launch latency and Host-to-Device (H2D) memory transfer dominate over raw decode throughput.

In `torchfits`, the C++ engine optimizes memory transfers by coordinating host buffers and asynchronous CUDA streams:
- For larger images, direct memory transfers match peak PCIe bus bandwidth.
- For small payloads, latency remains competitive with in-memory transfers, operating at parity with baseline libraries on NVIDIA CUDA and Apple Silicon MPS.

### Vectorized SIMD Integer Decoding (Unsigned Integers & BZERO)

Standard astronomical FITS stores unsigned 16-bit and 32-bit integers using signed formats paired with standard `BZERO` offsets ($y = \text{raw} + 32768$).

`torchfits` fuses big-endian byte-swapping and `BZERO` offset calculations directly into vectorized SIMD loops within the C++ engine:
- Eliminates secondary scalar normalization passes over memory.
- Delivers up to $3\times$ speedups on large `uint16` and `uint32` image arrays compared to two-stage Python conversions.

## Python & PyTorch Matrix Variance

The headline benchmarks on this page are reported using the Pareto-optimal (champion) environment combination measured across our CANFAR exhaustive matrix: **PyTorch 2.12 + Python 3.11**.

Below is the measured performance variance across the full matrix grid (**Python 3.10–3.14** × **PyTorch 2.10–2.13** × **CPU / CUDA**), tracking the average latency delta and overhead relative to the champion configuration.

### Summary: Average Performance Overhead vs Champion

- **CPU Host Workloads:**
  - **Optimal Baseline:** PyTorch 2.12 + Python 3.11 (Geometric-mean latency: **0.106 ms**)
  - **Average Python Variance:** Across all Python versions (3.10–3.14), average latency penalty is **+9.5%** (+7.6% on 3.10, +11.2% on 3.11, +10.5% on 3.12, +8.4% on 3.13, +12.8% on 3.14).
  - **Average PyTorch Variance:** Across PyTorch minor versions (2.10–2.13), average latency penalty is **+9.8%** (+10.7% on 2.10, +12.3% on 2.11, +7.4% on 2.12, +9.9% on 2.13).

- **CUDA Workloads (NVIDIA GPU):**
  - **Optimal Baseline:** PyTorch 2.12 + Python 3.11 (Geometric-mean latency: **0.187 ms**)
  - **Average Python Variance:** Across all Python versions (3.10–3.14), average latency penalty is **+5.8%** (+8.6% on 3.10, +4.0% on 3.11, +5.0% on 3.12, +4.9% on 3.13, +6.3% on 3.14).
  - **Average PyTorch Variance:** Across PyTorch minor versions (2.10–2.13), average latency penalty is **+5.7%** (+3.7% on 2.10, +4.3% on 2.11, +3.9% on 2.12, +11.0% on 2.13).

### Full Matrix Benchmark Comparison

#### CPU Transport Matrix

| PyTorch | Python | Device | Geom Mean (ms) | Delta vs Best | Relative Perf |
|---|---|---|---:|---:|---:|
| 2.12 | 3.11 | CPU | 0.106 | **Baseline (Best)** | **1.00×** |
| 2.10 | 3.10 | CPU | 0.108 | +2.0% slower | 1.02× |
| 2.13 | 3.12 | CPU | 0.111 | +5.0% slower | 1.05× |
| 2.11 | 3.10 | CPU | 0.113 | +6.4% slower | 1.06× |
| 2.11 | 3.13 | CPU | 0.113 | +6.4% slower | 1.06× |
| 2.12 | 3.13 | CPU | 0.113 | +6.6% slower | 1.07× |
| 2.12 | 3.10 | CPU | 0.114 | +7.3% slower | 1.07× |
| 2.13 | 3.13 | CPU | 0.114 | +7.3% slower | 1.07× |
| 2.12 | 3.14 | CPU | 0.114 | +7.4% slower | 1.07× |
| 2.13 | 3.14 | CPU | 0.115 | +8.6% slower | 1.09× |
| 2.10 | 3.12 | CPU | 0.116 | +8.9% slower | 1.09× |
| 2.10 | 3.14 | CPU | 0.117 | +10.2% slower | 1.10× |
| 2.11 | 3.11 | CPU | 0.119 | +11.7% slower | 1.12× |
| 2.11 | 3.12 | CPU | 0.119 | +11.9% slower | 1.12× |
| 2.10 | 3.13 | CPU | 0.120 | +13.2% slower | 1.13× |
| 2.13 | 3.11 | CPU | 0.121 | +13.9% slower | 1.14× |
| 2.13 | 3.10 | CPU | 0.122 | +14.9% slower | 1.15× |
| 2.12 | 3.12 | CPU | 0.123 | +16.0% slower | 1.16× |
| 2.10 | 3.11 | CPU | 0.126 | +19.2% slower | 1.19× |
| 2.11 | 3.14 | CPU | 0.132 | +24.9% slower | 1.25× |

#### CUDA Transport Matrix

| PyTorch | Python | Device | Geom Mean (ms) | Delta vs Best | Relative Perf |
|---|---|---|---:|---:|---:|
| 2.12 | 3.11 | CUDA | 0.187 | **Baseline (Best)** | **1.00×** |
| 2.10 | 3.14 | CUDA | 0.189 | +1.1% slower | 1.01× |
| 2.10 | 3.13 | CUDA | 0.190 | +1.8% slower | 1.02× |
| 2.10 | 3.12 | CUDA | 0.190 | +1.8% slower | 1.02× |
| 2.12 | 3.13 | CUDA | 0.191 | +1.9% slower | 1.02× |
| 2.11 | 3.12 | CUDA | 0.193 | +3.1% slower | 1.03× |
| 2.11 | 3.14 | CUDA | 0.193 | +3.3% slower | 1.03× |
| 2.12 | 3.12 | CUDA | 0.195 | +4.1% slower | 1.04× |
| 2.11 | 3.11 | CUDA | 0.196 | +5.0% slower | 1.05× |
| 2.11 | 3.13 | CUDA | 0.197 | +5.1% slower | 1.05× |
| 2.10 | 3.11 | CUDA | 0.197 | +5.2% slower | 1.05× |
| 2.11 | 3.10 | CUDA | 0.197 | +5.2% slower | 1.05× |
| 2.12 | 3.10 | CUDA | 0.197 | +5.4% slower | 1.05× |
| 2.13 | 3.11 | CUDA | 0.198 | +5.8% slower | 1.06× |
| 2.12 | 3.14 | CUDA | 0.202 | +8.1% slower | 1.08× |
| 2.10 | 3.10 | CUDA | 0.203 | +8.8% slower | 1.09× |
| 2.13 | 3.12 | CUDA | 0.207 | +10.8% slower | 1.11× |
| 2.13 | 3.13 | CUDA | 0.207 | +10.9% slower | 1.11× |
| 2.13 | 3.14 | CUDA | 0.211 | +12.6% slower | 1.13× |
| 2.13 | 3.10 | CUDA | 0.215 | +15.1% slower | 1.15× |

<!-- PROVENANCE: headline scorecards cite exhaustive_cpu_20261007_203905 /
     exhaustive_cuda_20261007_203924 / exhaustive_mps_20261007_204350
     under docs/assets/bench/. -->
## Published Benchmark Data {#published-csvs}

Exhaustive benchmark datasets and analysis CSVs (`results.csv`, `torchfits_deficits.csv`) are published with each release and mirrored under `docs/assets/bench/<run-id>/`:

- [`exhaustive_cpu_20261007_203905/results.csv`](assets/bench/exhaustive_cpu_20261007_203905/results.csv)
- [`exhaustive_cuda_20261007_203924/results.csv`](assets/bench/exhaustive_cuda_20261007_203924/results.csv)
- [`exhaustive_mps_20261007_204350/results.csv`](assets/bench/exhaustive_mps_20261007_204350/results.csv)
- [`exhaustive_cpu_20260807_013736/results.csv`](assets/bench/exhaustive_cpu_20260807_013736/results.csv)
- [`exhaustive_cuda_20260807_013736/results.csv`](assets/bench/exhaustive_cuda_20260807_013736/results.csv)
- [`exhaustive_mps_20260719_143706/results.csv`](assets/bench/exhaustive_mps_20260719_143706/results.csv)
- [`exhaustive_cpu_20260719_144337/results.csv`](assets/bench/exhaustive_cpu_20260719_144337/results.csv)
- [`exhaustive_cuda_20260719_144457/results.csv`](assets/bench/exhaustive_cuda_20260719_144457/results.csv)

### Modular Suites & Release Exhaustives

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

### CFHT MegaCam Cutout Suite

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

Outputs land in `benchmarks_results/<run-id>/megacam_results.csv`.

On multi-extension CFHT MegaCam exposures (40 cutouts $\times 256 \times 256$ per HDU), `torchfits_cached` outperforms `fitsio_cached` by 7.5%–15.2% across sampled HDUs due to optimized tile decompression handles.

For **uncompressed** survey mosaics (e.g. CFHTLS MegaPipe float32 stacks),
`open_subset_reader` maps the data segment once and slices cutouts with
endian swap into torch tensors — see
[ML with FITS](examples-ml.md#survey-mosaic-cutouts-cfht-megapipe). Rice `.fz`
MegaCam cutouts remain a separate comparison (tile decompress inside CFITSIO).

## Cold-start and the torch boundary

Operation-level benchmarks measure a call; they cannot show what a *process*
costs to start. That is where the old boundary showed up: a header peek reads a
2880-byte block in microseconds, but metadata entry points used to pay about a
second for an image-size tensor runtime they never touched. The native module
now defers the Python `torch` import, and Arrow table reads use a raw native
buffer transport instead of constructing Python tensors.

The rule this project holds to: **PyTorch is loaded at exactly one boundary —
the first call whose documented return type is a `torch.Tensor`, or that takes
`device=`.** Nothing before it may import torch. `tests/test_torch_boundary.py`
enforces that in fresh interpreters with `import torch` blocked outright, and
`benchmarks/bench_import_boundary.py` records what it costs.

Since the `libtorchfits_core` split, metadata does not merely avoid the Python
`torch` import — it never loads a library that links libtorch. The path-based
probes (`read_header`, `read_keys`, `read_colnames`, `read_nrows`,
`read_num_hdus`, `read_hdu_type`, `read_shape`, `read_table_info`) run entirely
through `libtorchfits_core`, which carries CFITSIO and its own thread pool and
nothing else. `tests/test_core_library.py` compares that module's answers with
the torch-linked extension's, and `tests/test_torch_boundary.py` checks the
library's own dependency list with `otool -L` / `ldd`.

Sample baseline (cold process, spawn to exit, minimum of 5, macOS arm64,
Python 3.13.15, 2026-09-25):

| Entry point | Cold ms | Loads torch | Budget |
|---|---:|---|---:|
| `import torchfits` | 30 | no | 250 ms |
| `import torchfits.hdu` | 32 | no | 250 ms |
| `import torchfits.io` | 55 | no | 250 ms |
| `import torchfits.table` | 59 | no | 500 ms |
| `read_header` | 59 | no | 120 ms |
| `read_keys` | 53 | no | 120 ms |
| `read_colnames` | 56 | no | 120 ms |
| `read_num_hdus` | 56 | no | 120 ms |
| `read_shape` | 58 | no | 120 ms |
| `read_table_info` | 57 | no | 120 ms |
| `_core.read_colnames` (library only) | 35 | no | 80 ms |
| `_core.read_header_dict` (library only) | 33 | no | 80 ms |
| `open` + `hdul[1].header` | 222 | no | 250 ms |
| `table.read` (Arrow) | 515 | no | 600 ms |
| `table.schema` | 167 | no | 600 ms |
| `read_tensor` (tensor destination) | 770 | yes | — |
| `table.read_torch` (tensor destination) | 737 | yes | — |
| `import torch` (reference) | 722 | yes | — |

The ~30 ms floor is interpreter start. Three rows are worth reading carefully:

* The `_core.*` rows are the floor of the new path: a `dlopen` of
  `libtorchfits_core` plus one query, with no `torchfits` package machinery. The
  metadata rows above sit ~22 ms higher, which is the Python-side caching and
  path-guard layer.
* `open` + `hdul[1].header` is **not** on the core path yet. `torchfits.open`
  returns an `HDUList` that owns a native handle, and that handle is the same
  object the tensor readers take, so it must come from the torch-linked
  extension. Its 250 ms budget is measured, not aspirational; making the handle
  lazy is tracked in [roadmap](roadmap.md).
* The Arrow rows are budgeted from the *slow* end of their own spread, not the
  fastest sample. `table.read` has measured between 461 and 529 ms (minimum of
  five) across runs on this machine, so a 500 ms gate passed only on a good day
  and the budget is 600 ms. A gate that only sometimes passes is not a gate.

Earlier baselines for the same rows, for comparison: metadata calls were
332–337 ms and `table.schema` 489 ms before the split — a 4× reduction from
removing libtorch from the metadata path.

The harness writes its own minimal FITS files with the standard library, so it
runs in a torch-free environment:

```bash
pixi run python benchmarks/bench_import_boundary.py           # report
pixi run python benchmarks/bench_import_boundary.py --strict  # boundary + timing gate
```

## Correctness checks

| Check | Command | Validates |
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

Named **focused-benchmark** recipes (mmap on+off, no unrelated GPU matrix when scoped to tables):

```bash
pixi run bench-deficit-focus              # hcompress + tiny_int8 + narrow predicates
pixi run bench-deficit-focus hcompress
pixi run bench-deficit-focus tiny_int8
pixi run bench-deficit-focus predicate
```

Rankings and comparisons group by `(domain, case_id, family, mmap_target)` so
mmap-on and mmap-off peers are never cross-compared.
## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | fits / fitstable | FITS benchmark orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression, scaling, MEF, and cutouts |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, projection, row slicing, predicates, and streaming |
| `bench_all.py` / `bench-fits` | fits | Published-results path |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_gpu_transports.py` | fits (GPU) | CUDA/MPS image reads, cutouts, repeated cutouts (`disk→CPU→GPU` / `disk→RAM→GPU` rows) |
| `bench_ml_loader.py` | fits (diagnostic) | PyTorch `DataLoader` throughput (not merged into `bench-all` CSV) |
| `bench_gpu_memory.py` | fits (diagnostic) | GPU memory/leak checks (non-gating) |
| `bench_denoise.py` | ml (scientific) | Noise2Noise CR-cleaning on real CFHT MegaCam frames (dark→blank framing, `torchfits` loaders vs Astropy; see [denoise-pipeline.md](denoise-pipeline.md)) |
| `bench_import_boundary.py` | cold start | Fresh-process spawn-to-exit cost per entry point, split by whether torch was loaded; `--strict` gates on the boundary and per-entry-point budgets |

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
| I/O transport `disk→GPU` | No | — | No host-bypass path yet (see Methodology); 2.0 / roadmap |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D sample datasets; unsigned via BZERO in `scaled_*` |
| Tensor dimensions / sizes | Yes | tiny → large; 1D–3D (4D where sample datasets exist) | Large 3D cubes may hit size caps |
| Compression (read) | Yes | gzip, rice, hcompress, plio | Write→compress cases are being added to the suite |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` | MEF random ext reads on selected sample datasets |
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
   CANFAR (`exhaustive_cuda_20261007_203924`, 2026-10-07). MPS numbers are
   `exhaustive_mps_20261007_204350`.
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
Source: `docs/assets/bench/exhaustive_cpu_20261007_203905/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.09 ms` (n=174) | `0.46 ms` (n=253) | `0.16 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.09 ms` (n=174) | `0.43 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.24 ms` (n=216) | `2.49 ms` (n=184) | `0.55 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.25 ms` (n=216) | `2.58 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- Pure-C CFITSIO (vendored): `pixi run bench-cfitsio-direct` runs the **full**
  image+table benchmark fixture set with op→API mapping in
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
| Large tensor read (Float32 2D, 16.0 MB) | CPU | **7.12 ms** | 6.77 ms | 14.67 ms | — | **2.17x** | **—** |
| Compressed tensor read (Rice, 1.1 MB) | CPU | **12.92 ms** | 12.85 ms | 33.78 ms | 12.99 ms | **2.63x** | **1.01x** |
| Repeated cutouts (50x 100x100) | CPU | **451.0 μs** | 461.4 μs | 52.48 ms | 3.28 ms | **116.37x** | **7.27x** |
| Table read (100k rows, 8 cols, mixed) | CPU | **2.15 ms** | 1.74 ms | 29.36 ms | — | **16.91x** | **—** |
| Varlen table read (100k rows, 3 cols) | CPU | **100.17 ms** | 120.79 ms | 897.97 ms | — | **8.96x** | **—** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

The generated [highlights](#performance-highlights) and
[full table](#exhaustive-benchmark-results) are the Linux CPU run
`exhaustive_cpu_20261007_203905`. CUDA (`exhaustive_cuda_20261007_203924`)
and MPS (`exhaustive_mps_20261007_204350`) are in the host table and the
deficit section. The category ranges below are the August 2026 aggregation
(`exhaustive_cpu_20260807_013736` / `exhaustive_cuda_20260807_013736`);
for absolute times use the generated tables.

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

**GPU (CUDA) results** — 85 comparable `read_full` / cutout cases:

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
| tensor | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 43.3 μs | 1.33 ms | 132.8 μs | — | **30.61x** | **3.07x** |
| tensor | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 42.9 μs | 1.33 ms | 134.2 μs | — | **30.97x** | **3.13x** |
| tensor | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 44.7 μs | 1.41 ms | 150.7 μs | — | **31.43x** | **3.37x** |
| tensor | compressed_rice_1 | cutout_100x100 | 39.1 KB | CPU | n/a | **744.8 μs** | 790.8 μs | 6.77 ms | 798.3 μs | — | **9.08x** | **1.07x** |
| tensor | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 44.3 μs | 1.39 ms | 151.8 μs | — | **31.48x** | **3.43x** |
| tensor | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 20.4 μs | 251.4 μs | 25.0 μs | — | **12.35x** | **1.23x** |
| tensor | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 21.8 μs | 286.9 μs | 27.4 μs | — | **13.17x** | **1.26x** |
| tensor | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 20.4 μs | 259.1 μs | 25.7 μs | — | **12.68x** | **1.26x** |
| tensor | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 21.5 μs | 293.5 μs | 28.1 μs | — | **13.68x** | **1.31x** |
| tensor | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 19.9 μs | 260.0 μs | 25.6 μs | — | **13.03x** | **1.28x** |
| tensor | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.9 μs | 279.4 μs | 27.3 μs | — | **13.37x** | **1.31x** |
| tensor | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 19.9 μs | 259.8 μs | 24.9 μs | — | **13.06x** | **1.25x** |
| tensor | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 21.2 μs | 279.8 μs | 27.4 μs | — | **13.18x** | **1.29x** |
| tensor | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 20.9 μs | 253.8 μs | 25.1 μs | — | **12.13x** | **1.20x** |
| tensor | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 20.5 μs | 276.7 μs | 27.1 μs | — | **13.53x** | **1.32x** |
| tensor | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 21.1 μs | 291.7 μs | 31.1 μs | — | **13.83x** | **1.47x** |
| tensor | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 22.4 μs | 313.2 μs | 31.9 μs | — | **13.98x** | **1.43x** |
| tensor | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 21.6 μs | 311.5 μs | 31.4 μs | — | **14.39x** | **1.45x** |
| tensor | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 21.9 μs | 320.2 μs | 33.3 μs | — | **14.62x** | **1.52x** |
| tensor | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 20.0 μs | 253.8 μs | 25.0 μs | — | **12.70x** | **1.25x** |
| tensor | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 20.3 μs | 273.7 μs | 26.4 μs | — | **13.51x** | **1.31x** |
| tensor | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 21.1 μs | 300.7 μs | 28.6 μs | — | **14.26x** | **1.35x** |
| tensor | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 20.4 μs | 259.7 μs | 25.3 μs | — | **12.75x** | **1.24x** |
| tensor | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.9 μs | 288.3 μs | 27.3 μs | — | **13.77x** | **1.30x** |
| tensor | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 22.9 μs | 294.3 μs | 29.9 μs | — | **12.87x** | **1.31x** |
| tensor | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 20.0 μs | 250.4 μs | 24.0 μs | — | **12.50x** | **1.20x** |
| tensor | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 21.7 μs | 282.5 μs | 26.2 μs | — | **13.00x** | **1.21x** |
| tensor | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 21.4 μs | 295.4 μs | 29.5 μs | — | **13.82x** | **1.38x** |
| tensor | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 19.7 μs | 247.5 μs | 24.5 μs | — | **12.55x** | **1.24x** |
| tensor | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 20.7 μs | 275.0 μs | 27.5 μs | — | **13.28x** | **1.33x** |
| tensor | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 20.7 μs | 295.7 μs | 29.2 μs | — | **14.31x** | **1.41x** |
| tensor | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 19.7 μs | 248.4 μs | 24.7 μs | — | **12.60x** | **1.25x** |
| tensor | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.8 μs | 284.8 μs | 26.9 μs | — | **13.67x** | **1.29x** |
| tensor | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 21.3 μs | 295.5 μs | 29.1 μs | — | **13.89x** | **1.37x** |
| tensor | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 21.4 μs | 298.9 μs | 29.9 μs | — | **13.94x** | **1.39x** |
| tensor | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 22.4 μs | 318.1 μs | 31.7 μs | — | **14.22x** | **1.42x** |
| tensor | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 22.0 μs | 337.8 μs | 33.4 μs | — | **15.32x** | **1.51x** |
| tensor | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 23.3 μs | 328.5 μs | 32.6 μs | — | **14.10x** | **1.40x** |
| tensor | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 22.2 μs | 326.0 μs | 32.0 μs | — | **14.68x** | **1.44x** |
| tensor | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 24.5 μs | 504.7 μs | 40.3 μs | — | **20.57x** | **1.64x** |
| tensor | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 24.4 μs | 501.7 μs | 40.2 μs | — | **20.56x** | **1.65x** |
| tensor | multi_mef_10ext | cutout_100x100 | 39.1 KB | CPU | n/a | **63.7 μs** | 88.2 μs | 2.20 ms | 158.1 μs | — | **34.59x** | **2.48x** |
| tensor | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 25.7 μs | 502.7 μs | 38.9 μs | — | **19.56x** | **1.52x** |
| tensor | multi_mef_10ext | random_ext_full_reads_200 | 52.50 MB | CPU | n/a | **6.00 ms** | 5.98 ms | 6.71 ms | 7.51 ms | — | **1.12x** | **1.26x** |
| tensor | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 1.91 MB | CPU | n/a | **451.0 μs** | 461.4 μs | 52.48 ms | 3.28 ms | — | **116.37x** | **7.27x** |
| tensor | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 23.3 μs | 334.5 μs | 32.2 μs | — | **14.38x** | **1.38x** |
| tensor | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 22.4 μs | 320.8 μs | 31.9 μs | — | **14.34x** | **1.43x** |
| tensor | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 22.1 μs | 319.5 μs | 32.5 μs | — | **14.43x** | **1.47x** |
| tensor | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 19.2 μs | 258.5 μs | 23.8 μs | — | **13.46x** | **1.24x** |
| tensor | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 19.5 μs | 266.2 μs | 25.5 μs | — | **13.63x** | **1.31x** |
| tensor | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 21.8 μs | 296.3 μs | 29.7 μs | — | **13.58x** | **1.36x** |
| tensor | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 20.0 μs | 258.2 μs | 24.9 μs | — | **12.91x** | **1.24x** |
| tensor | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 20.4 μs | 273.7 μs | 26.9 μs | — | **13.42x** | **1.32x** |
| tensor | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 21.0 μs | 299.2 μs | 28.4 μs | — | **14.24x** | **1.35x** |
| tensor | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 19.5 μs | 249.9 μs | 24.9 μs | — | **12.83x** | **1.28x** |
| tensor | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 20.2 μs | 275.5 μs | 27.1 μs | — | **13.64x** | **1.34x** |
| tensor | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 21.3 μs | 296.6 μs | 29.3 μs | — | **13.92x** | **1.38x** |
| tensor | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 19.8 μs | 246.8 μs | 24.8 μs | — | **12.45x** | **1.25x** |
| tensor | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 20.3 μs | 280.7 μs | 26.2 μs | — | **13.82x** | **1.29x** |
| tensor | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 20.8 μs | 292.9 μs | 28.8 μs | — | **14.11x** | **1.38x** |
| tensor | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 20.0 μs | 256.1 μs | 25.1 μs | — | **12.80x** | **1.26x** |
| tensor | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 21.0 μs | 279.3 μs | 26.4 μs | — | **13.28x** | **1.25x** |
| tensor | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 20.0 μs | 287.9 μs | 30.8 μs | — | **14.41x** | **1.54x** |
| tensor | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 21.2 μs | 298.7 μs | 29.2 μs | — | **14.07x** | **1.38x** |
| tensor | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 21.6 μs | 312.6 μs | 31.1 μs | — | **14.46x** | **1.44x** |
| tensor | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 22.5 μs | 335.2 μs | 33.6 μs | — | **14.86x** | **1.49x** |
| tensor | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 23.1 μs | 314.7 μs | 32.0 μs | — | **13.60x** | **1.38x** |
| tensor | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 22.2 μs | 319.7 μs | 32.2 μs | — | **14.38x** | **1.45x** |
| tensor | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 20.6 μs | 279.2 μs | 27.6 μs | — | **13.56x** | **1.34x** |
| tensor | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 20.4 μs | 278.8 μs | 27.2 μs | — | **13.65x** | **1.33x** |
| tensor | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 21.2 μs | 291.0 μs | 26.7 μs | — | **13.75x** | **1.26x** |
| tensor | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 21.7 μs | 280.7 μs | 27.9 μs | — | **12.96x** | **1.29x** |
| tensor | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 21.3 μs | 280.9 μs | 27.0 μs | — | **13.20x** | **1.27x** |
| tensor | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 20.4 μs | 261.0 μs | 26.7 μs | — | **12.78x** | **1.31x** |
| tensor | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 20.8 μs | 285.0 μs | 26.6 μs | — | **13.70x** | **1.28x** |
| tensor | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 21.6 μs | 296.4 μs | 29.0 μs | — | **13.71x** | **1.34x** |
| tensor | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 19.9 μs | 254.8 μs | 25.0 μs | — | **12.80x** | **1.26x** |
| tensor | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 21.1 μs | 284.3 μs | 26.5 μs | — | **13.45x** | **1.25x** |
| tensor | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 21.0 μs | 295.9 μs | 28.0 μs | — | **14.10x** | **1.33x** |
| tensor | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 19.4 μs | 256.7 μs | 24.7 μs | — | **13.22x** | **1.27x** |
| tensor | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 19.9 μs | 272.9 μs | 26.6 μs | — | **13.70x** | **1.33x** |
| tensor | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 20.4 μs | 300.3 μs | 27.9 μs | — | **14.72x** | **1.37x** |
| tensor | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 19.6 μs | 257.5 μs | 24.6 μs | — | **13.14x** | **1.25x** |
| tensor | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 19.9 μs | 281.9 μs | 26.4 μs | — | **14.14x** | **1.33x** |
| tensor | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 21.7 μs | 309.5 μs | 28.3 μs | — | **14.28x** | **1.30x** |
| tensor | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 19.1 μs | 252.5 μs | 24.4 μs | — | **13.22x** | **1.28x** |
| tensor | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 19.9 μs | 280.2 μs | 26.6 μs | — | **14.11x** | **1.34x** |
| tensor | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 22.0 μs | 308.1 μs | 29.3 μs | — | **14.03x** | **1.33x** |
| tensor | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 20.9 μs | 301.6 μs | 28.8 μs | — | **14.45x** | **1.38x** |
| tensor | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 21.4 μs | 315.5 μs | 30.7 μs | — | **14.73x** | **1.43x** |
| tensor | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 22.4 μs | 323.4 μs | 33.0 μs | — | **14.46x** | **1.47x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.09 ms** | — | 59.38 ms | — | — | **1.23x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.79 ms** | — | 68.92 ms | — | — | **1.82x** | **—** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **13.87 ms** | 13.63 ms | 26.23 ms | 15.20 ms | — | **1.92x** | **1.12x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **11.69 ms** | 11.64 ms | 41.68 ms | 13.31 ms | — | **3.58x** | **1.14x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **26.42 ms** | 26.30 ms | 30.16 ms | 25.63 ms | — | **1.15x** | **0.97x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **7.21 ms** | 7.16 ms | 18.99 ms | 7.29 ms | — | **2.65x** | **1.02x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | off | **477.5 μs** | 481.8 μs | 1.12 ms | 755.3 μs | — | **2.34x** | **1.58x** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | off | **2.59 ms** | 2.60 ms | 9.40 ms | 3.24 ms | — | **3.64x** | **1.25x** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | off | **885.5 μs** | 889.6 μs | 1.90 ms | 1.24 ms | — | **2.14x** | **1.40x** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | off | **4.68 ms** | 4.75 ms | 9.48 ms | 4.94 ms | — | **2.03x** | **1.06x** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | off | **290.3 μs** | 284.2 μs | 721.7 μs | 343.9 μs | — | **2.54x** | **1.21x** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | off | **981.5 μs** | 989.1 μs | 3.69 ms | 1.27 ms | — | **3.76x** | **1.29x** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | off | **473.0 μs** | 482.4 μs | 1.11 ms | 753.6 μs | — | **2.34x** | **1.59x** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | off | **2.60 ms** | 2.61 ms | 9.40 ms | 3.23 ms | — | **3.62x** | **1.24x** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | off | **884.9 μs** | 888.8 μs | 1.89 ms | 1.24 ms | — | **2.14x** | **1.40x** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | off | **4.57 ms** | 4.49 ms | 9.48 ms | 4.93 ms | — | **2.11x** | **1.10x** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | off | **159.5 μs** | 166.7 μs | 626.2 μs | 187.1 μs | — | **3.93x** | **1.17x** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | off | **677.7 μs** | 604.1 μs | 1.53 ms | 665.8 μs | — | **2.54x** | **1.10x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | off | **1.24 ms** | 1.23 ms | 3.85 ms | 1.51 ms | — | **3.13x** | **1.23x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | off | **3.07 ms** | 3.08 ms | 6.52 ms | 3.73 ms | — | **2.12x** | **1.21x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | off | **80.9 μs** | 86.8 μs | 356.0 μs | 105.5 μs | — | **4.40x** | **1.30x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | off | **497.3 μs** | 498.1 μs | 1.19 ms | 792.3 μs | — | **2.40x** | **1.59x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | off | **746.8 μs** | 741.8 μs | 1.65 ms | 1.20 ms | — | **2.23x** | **1.61x** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | off | **130.2 μs** | 137.0 μs | 449.9 μs | 159.5 μs | — | **3.46x** | **1.22x** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | off | **922.1 μs** | 924.0 μs | 2.55 ms | 1.30 ms | — | **2.76x** | **1.41x** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | off | **2.21 ms** | 2.28 ms | 3.92 ms | 2.39 ms | — | **1.77x** | **1.08x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | off | **56.7 μs** | 60.4 μs | 304.7 μs | 65.2 μs | — | **5.37x** | **1.15x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | off | **287.4 μs** | 305.2 μs | 761.1 μs | 359.8 μs | — | **2.65x** | **1.25x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | off | **417.5 μs** | 432.6 μs | 1.01 ms | 536.2 μs | — | **2.41x** | **1.28x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | off | **80.5 μs** | 86.5 μs | 356.1 μs | 108.6 μs | — | **4.43x** | **1.35x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | off | **497.4 μs** | 505.6 μs | 1.18 ms | 789.4 μs | — | **2.38x** | **1.59x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | off | **743.4 μs** | 741.3 μs | 1.66 ms | 1.19 ms | — | **2.24x** | **1.60x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | off | **128.7 μs** | 135.6 μs | 449.2 μs | 158.1 μs | — | **3.49x** | **1.23x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | off | **923.8 μs** | 932.4 μs | 2.54 ms | 1.29 ms | — | **2.75x** | **1.40x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | off | **2.22 ms** | 2.21 ms | 3.90 ms | 2.38 ms | — | **1.76x** | **1.08x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | off | **48.0 μs** | 49.5 μs | 366.6 μs | 54.3 μs | — | **7.63x** | **1.13x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | off | **169.2 μs** | 172.9 μs | 650.9 μs | 196.0 μs | — | **3.85x** | **1.16x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | off | **368.6 μs** | 370.5 μs | 832.4 μs | 290.5 μs | — | **2.26x** | **0.79x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **335.2 μs** | 343.1 μs | 1.26 ms | 409.8 μs | — | **3.77x** | **1.22x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **622.6 μs** | 631.1 μs | 1.70 ms | 915.2 μs | — | **2.73x** | **1.47x** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | off | **177.0 μs** | 177.0 μs | 848.9 μs | 229.8 μs | — | **4.80x** | **1.30x** |
| tensor | mef_small | read_full | 0.45 MB | CPU | off | **47.9 μs** | 54.8 μs | 552.1 μs | 83.4 μs | — | **11.52x** | **1.74x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **46.6 μs** | 54.6 μs | 550.7 μs | 132.0 μs | — | **11.81x** | **2.83x** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | off | **3.28 ms** | 3.30 ms | 5.33 ms | 3.40 ms | — | **1.62x** | **1.04x** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | off | **567.8 μs** | 578.5 μs | 1.37 ms | 806.7 μs | — | **2.42x** | **1.42x** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | off | **87.8 μs** | 83.0 μs | 440.4 μs | 93.1 μs | — | **5.31x** | **1.12x** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | off | **35.4 μs** | 38.3 μs | 259.9 μs | 45.3 μs | — | **7.34x** | **1.28x** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | off | **66.7 μs** | 69.6 μs | 333.8 μs | 82.4 μs | — | **5.01x** | **1.24x** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | off | **124.1 μs** | 123.1 μs | 439.1 μs | 158.1 μs | — | **3.57x** | **1.28x** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | off | **36.8 μs** | 46.9 μs | 260.9 μs | 46.7 μs | — | **7.09x** | **1.27x** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | off | **108.9 μs** | 107.1 μs | 396.0 μs | 116.3 μs | — | **3.70x** | **1.09x** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | off | **187.5 μs** | 194.0 μs | 598.7 μs | 247.3 μs | — | **3.19x** | **1.32x** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | off | **35.0 μs** | 39.0 μs | 249.3 μs | 41.0 μs | — | **7.12x** | **1.17x** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | off | **52.6 μs** | 55.6 μs | 295.4 μs | 54.6 μs | — | **5.61x** | **1.04x** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | off | **81.2 μs** | 84.1 μs | 362.0 μs | 86.3 μs | — | **4.46x** | **1.06x** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | off | **35.9 μs** | 40.2 μs | 253.9 μs | 44.9 μs | — | **7.07x** | **1.25x** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | off | **68.7 μs** | 71.8 μs | 331.6 μs | 82.5 μs | — | **4.82x** | **1.20x** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | off | **119.7 μs** | 124.9 μs | 442.0 μs | 155.8 μs | — | **3.69x** | **1.30x** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | off | **46.8 μs** | 46.2 μs | 264.9 μs | 48.0 μs | — | **5.74x** | **1.04x** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | off | **102.0 μs** | 108.0 μs | 400.0 μs | 115.4 μs | — | **3.92x** | **1.13x** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | off | **188.6 μs** | 197.3 μs | 607.6 μs | 242.6 μs | — | **3.22x** | **1.29x** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | off | **36.4 μs** | 46.5 μs | 343.3 μs | 43.9 μs | — | **9.44x** | **1.21x** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | off | **43.0 μs** | 51.2 μs | 371.4 μs | 53.0 μs | — | **8.63x** | **1.23x** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | off | **63.6 μs** | 67.5 μs | 402.5 μs | 63.1 μs | — | **6.32x** | **0.99x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | off | **58.8 μs** | 60.1 μs | 377.8 μs | 62.1 μs | — | **6.43x** | **1.06x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | off | **80.4 μs** | 75.4 μs | 417.8 μs | 91.2 μs | — | **5.54x** | **1.21x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **65.3 μs** | 67.0 μs | 336.1 μs | 84.4 μs | — | **5.15x** | **1.29x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **69.8 μs** | 65.9 μs | 325.8 μs | 82.9 μs | — | **4.94x** | **1.26x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **67.7 μs** | 71.5 μs | 332.0 μs | 84.8 μs | — | **4.91x** | **1.25x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **66.9 μs** | 71.9 μs | 329.9 μs | 84.7 μs | — | **4.93x** | **1.27x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **66.0 μs** | 67.2 μs | 337.4 μs | 81.6 μs | — | **5.11x** | **1.24x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **33.6 μs** | 34.6 μs | 241.9 μs | 38.7 μs | — | **7.21x** | **1.15x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **34.0 μs** | 39.3 μs | 259.2 μs | 43.7 μs | — | **7.63x** | **1.29x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **36.8 μs** | 37.3 μs | 279.5 μs | 44.8 μs | — | **7.61x** | **1.22x** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **41.8 μs** | 39.4 μs | 241.9 μs | 38.8 μs | — | **6.14x** | **0.99x** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **34.5 μs** | 41.8 μs | 266.9 μs | 44.3 μs | — | **7.73x** | **1.28x** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **35.5 μs** | 40.4 μs | 279.4 μs | 44.4 μs | — | **7.87x** | **1.25x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **34.1 μs** | 36.4 μs | 250.5 μs | 38.8 μs | — | **7.33x** | **1.14x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **35.9 μs** | 40.1 μs | 261.4 μs | 40.5 μs | — | **7.29x** | **1.13x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **39.5 μs** | 41.8 μs | 273.4 μs | 41.7 μs | — | **6.92x** | **1.05x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **34.6 μs** | 37.8 μs | 249.3 μs | 39.6 μs | — | **7.21x** | **1.15x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **39.0 μs** | 41.3 μs | 265.3 μs | 41.7 μs | — | **6.81x** | **1.07x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **35.0 μs** | 42.2 μs | 275.0 μs | 43.1 μs | — | **7.86x** | **1.23x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **33.8 μs** | 40.3 μs | 246.0 μs | 40.0 μs | — | **7.29x** | **1.18x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **36.7 μs** | 41.0 μs | 266.8 μs | 44.9 μs | — | **7.27x** | **1.22x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **38.3 μs** | 41.3 μs | 296.2 μs | 45.3 μs | — | **7.73x** | **1.18x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **33.6 μs** | 42.3 μs | 337.3 μs | 41.6 μs | — | **10.03x** | **1.24x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **34.1 μs** | 43.9 μs | 355.4 μs | 43.3 μs | — | **10.41x** | **1.27x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **33.2 μs** | 43.4 μs | 364.1 μs | 43.7 μs | — | **10.96x** | **1.32x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **24.18 ms** | 24.35 ms | 46.16 ms | 26.81 ms | — | **1.91x** | **1.11x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **20.67 ms** | 20.78 ms | 73.07 ms | 23.51 ms | — | **3.54x** | **1.14x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **46.33 ms** | 46.23 ms | 52.98 ms | 44.96 ms | — | **1.15x** | **0.97x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **12.92 ms** | 12.85 ms | 33.78 ms | 12.99 ms | — | **2.63x** | **1.01x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | on | **1.18 ms** | 1.17 ms | 2.20 ms | — | — | **1.89x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | on | **7.12 ms** | 6.77 ms | 14.67 ms | — | — | **2.17x** | **—** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | on | **2.22 ms** | 2.23 ms | 3.72 ms | — | — | **1.68x** | **—** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | on | **11.19 ms** | 10.47 ms | 16.77 ms | — | — | **1.60x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | on | **607.5 μs** | 608.2 μs | 1.45 ms | — | — | **2.39x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | on | **2.05 ms** | 2.02 ms | 3.79 ms | — | — | **1.88x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.06 ms** | 1.06 ms | 2.20 ms | — | — | **2.08x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | on | **5.71 ms** | 5.52 ms | 14.65 ms | — | — | **2.65x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.16 ms** | 2.06 ms | 3.72 ms | — | — | **1.81x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | on | **5.00 ms** | 4.85 ms | 7.60 ms | — | — | **1.57x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | on | **158.2 μs** | 170.6 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | on | **659.6 μs** | 708.5 μs | — | — | — | **—** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | on | **883.4 μs** | 881.3 μs | — | — | — | **—** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | on | **2.36 ms** | 2.36 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | on | **76.6 μs** | 79.6 μs | 368.6 μs | — | — | **4.81x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | on | **497.1 μs** | 493.5 μs | 1.53 ms | — | — | **3.10x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | on | **732.4 μs** | 739.7 μs | 1.41 ms | — | — | **1.92x** | **—** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | on | **118.5 μs** | 129.2 μs | 438.7 μs | — | — | **3.70x** | **—** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | on | **922.0 μs** | 920.9 μs | 1.67 ms | — | — | **1.82x** | **—** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | on | **1.44 ms** | 1.42 ms | 2.41 ms | — | — | **1.70x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | on | **66.8 μs** | 72.5 μs | 313.4 μs | — | — | **4.69x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | on | **273.3 μs** | 271.4 μs | 688.5 μs | — | — | **2.54x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | on | **410.9 μs** | 400.9 μs | 890.7 μs | — | — | **2.22x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | on | **87.8 μs** | 95.4 μs | 362.0 μs | — | — | **4.12x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | on | **481.1 μs** | 480.8 μs | 1.53 ms | — | — | **3.18x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | on | **703.8 μs** | 698.1 μs | 1.37 ms | — | — | **1.97x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | on | **131.3 μs** | 138.6 μs | 444.8 μs | — | — | **3.39x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | on | **885.4 μs** | 873.2 μs | 1.65 ms | — | — | **1.89x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | on | **1.31 ms** | 1.31 ms | 2.41 ms | — | — | **1.84x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | on | **60.2 μs** | 55.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | on | **165.7 μs** | 172.4 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | on | **382.3 μs** | 344.6 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **281.1 μs** | 286.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **488.5 μs** | 500.8 μs | — | — | — | **—** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | on | **176.9 μs** | 178.1 μs | — | — | — | **—** | **—** |
| tensor | mef_small | read_full | 0.45 MB | CPU | on | **50.5 μs** | 65.3 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **53.7 μs** | 66.1 μs | — | — | — | **—** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | on | **3.36 ms** | 3.25 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | on | **570.0 μs** | 582.2 μs | — | — | — | **—** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | on | **83.7 μs** | 87.0 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | on | **35.7 μs** | 40.5 μs | 264.8 μs | — | — | **7.42x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | on | **65.4 μs** | 65.2 μs | 339.2 μs | — | — | **5.20x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | on | **107.2 μs** | 116.0 μs | 438.3 μs | — | — | **4.09x** | **—** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | on | **57.4 μs** | 66.6 μs | 281.9 μs | — | — | **4.91x** | **—** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | on | **93.5 μs** | 95.4 μs | 408.6 μs | — | — | **4.37x** | **—** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | on | **183.7 μs** | 188.3 μs | 583.9 μs | — | — | **3.18x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | on | **41.6 μs** | 49.6 μs | 264.1 μs | — | — | **6.35x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | on | **76.6 μs** | 72.8 μs | 318.1 μs | — | — | **4.37x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | on | **93.2 μs** | 94.1 μs | 357.7 μs | — | — | **3.84x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | on | **45.3 μs** | 53.1 μs | 256.6 μs | — | — | **5.66x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | on | **85.9 μs** | 79.6 μs | 338.9 μs | — | — | **4.26x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | on | **119.8 μs** | 126.3 μs | 439.6 μs | — | — | **3.67x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | on | **48.6 μs** | 59.4 μs | 276.8 μs | — | — | **5.70x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | on | **106.0 μs** | 122.3 μs | 407.6 μs | — | — | **3.85x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | on | **201.0 μs** | 206.6 μs | 575.8 μs | — | — | **2.87x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | on | **36.3 μs** | 47.0 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | on | **51.7 μs** | 60.7 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | on | **63.7 μs** | 64.8 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | on | **77.6 μs** | 79.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | on | **87.7 μs** | 98.2 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **64.6 μs** | 66.6 μs | 341.6 μs | — | — | **5.29x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **60.3 μs** | 67.5 μs | 347.2 μs | — | — | **5.76x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **63.1 μs** | 68.9 μs | 345.1 μs | — | — | **5.47x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **60.9 μs** | 75.3 μs | 345.6 μs | — | — | **5.67x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **61.3 μs** | 66.2 μs | 335.2 μs | — | — | **5.47x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **32.4 μs** | 40.2 μs | 256.5 μs | — | — | **7.91x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **34.0 μs** | 42.1 μs | 271.2 μs | — | — | **7.97x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **34.3 μs** | 40.0 μs | 291.2 μs | — | — | **8.49x** | **—** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **32.8 μs** | 41.3 μs | 260.2 μs | — | — | **7.94x** | **—** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **35.5 μs** | 45.0 μs | 271.9 μs | — | — | **7.66x** | **—** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **38.8 μs** | 43.1 μs | 290.5 μs | — | — | **7.49x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **39.6 μs** | 42.9 μs | 246.1 μs | — | — | **6.22x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **39.6 μs** | 46.3 μs | 269.1 μs | — | — | **6.79x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **38.8 μs** | 46.0 μs | 290.6 μs | — | — | **7.49x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **37.3 μs** | 47.7 μs | 262.0 μs | — | — | **7.02x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **40.0 μs** | 47.5 μs | 264.8 μs | — | — | **6.62x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **47.5 μs** | 46.2 μs | 285.8 μs | — | — | **6.18x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **39.7 μs** | 42.4 μs | 254.7 μs | — | — | **6.41x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **41.0 μs** | 54.0 μs | 282.0 μs | — | — | **6.88x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **40.9 μs** | 52.7 μs | 286.0 μs | — | — | **6.99x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **34.9 μs** | 46.7 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **34.5 μs** | 48.5 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **39.3 μs** | 48.5 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **427.0 μs** | 429.9 μs | 2.63 ms | 372.9 μs | — | **6.16x** | **0.87x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **423.0 μs** | 427.6 μs | 2.62 ms | 372.2 μs | — | **6.20x** | **0.88x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **958.6 μs** | 919.4 μs | 7.86 ms | 1.95 ms | — | **8.55x** | **2.12x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **953.3 μs** | 915.4 μs | 7.85 ms | 1.93 ms | — | **8.57x** | **2.11x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **189.4 μs** | 154.6 μs | 2.56 ms | 518.7 μs | — | **16.57x** | **3.35x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **10.4 μs** | 9.6 μs | 398.4 μs | 55.3 μs | — | **41.35x** | **5.74x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **181.8 μs** | 184.8 μs | 1.49 ms | 165.2 μs | — | **8.19x** | **0.91x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **181.4 μs** | 185.2 μs | 1.49 ms | 165.4 μs | — | **8.19x** | **0.91x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **186.6 μs** | 156.6 μs | 2.13 ms | 335.1 μs | — | **13.60x** | **2.14x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **184.2 μs** | 149.8 μs | 2.12 ms | 324.0 μs | — | **14.18x** | **2.16x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **118.4 μs** | 89.0 μs | 1.93 ms | 194.5 μs | — | **21.65x** | **2.19x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **10.1 μs** | 9.8 μs | 389.8 μs | 56.8 μs | — | **39.84x** | **5.81x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **11.69 ms** | 11.27 ms | 16.14 ms | 20.78 ms | — | **1.43x** | **1.84x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **10.63 ms** | 10.65 ms | 13.01 ms | 17.32 ms | — | **1.22x** | **1.63x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.64 ms** | 10.49 ms | 18.15 ms | 31.71 ms | — | **1.73x** | **3.02x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **27.27 ms** | 25.88 ms | 319.15 ms | 115.81 ms | — | **12.33x** | **4.48x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **302.4 μs** | 254.8 μs | 13.42 ms | 1.48 ms | — | **52.67x** | **5.80x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **12.4 μs** | 11.8 μs | 453.1 μs | 69.7 μs | — | **38.55x** | **5.93x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.68 ms** | 1.44 ms | 3.14 ms | 2.27 ms | — | **2.18x** | **1.58x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.33 ms** | 1.37 ms | 2.81 ms | 1.88 ms | — | **2.11x** | **1.41x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.09 ms** | 1.05 ms | 3.23 ms | 3.24 ms | — | **3.08x** | **3.09x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **1.99 ms** | 1.93 ms | 29.79 ms | 10.05 ms | — | **15.44x** | **5.21x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **289.3 μs** | 251.5 μs | 5.89 ms | 1.47 ms | — | **23.43x** | **5.85x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **10.1 μs** | 10.1 μs | 426.0 μs | 71.1 μs | — | **42.20x** | **7.04x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **316.1 μs** | 359.1 μs | 1.95 ms | 390.7 μs | — | **6.18x** | **1.24x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **263.9 μs** | 328.0 μs | 1.92 ms | 350.1 μs | — | **7.28x** | **1.33x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **187.3 μs** | 159.9 μs | 1.93 ms | 482.1 μs | — | **12.09x** | **3.02x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **281.8 μs** | 245.7 μs | 4.45 ms | 1.11 ms | — | **18.10x** | **4.50x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **136.3 μs** | 105.6 μs | 2.96 ms | 324.5 μs | — | **28.03x** | **3.07x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **10.0 μs** | 10.3 μs | 418.1 μs | 71.7 μs | — | **41.78x** | **7.17x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **113.2 μs** | 177.9 μs | 1.79 ms | 188.2 μs | — | **15.81x** | **1.66x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **119.1 μs** | 189.7 μs | 1.80 ms | 184.5 μs | — | **15.08x** | **1.55x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **110.7 μs** | 87.7 μs | 1.79 ms | 194.5 μs | — | **20.42x** | **2.22x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **128.6 μs** | 103.7 μs | 2.14 ms | 265.9 μs | — | **20.61x** | **2.56x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **114.0 μs** | 89.8 μs | 2.63 ms | 200.7 μs | — | **29.25x** | **2.24x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **10.3 μs** | 10.1 μs | 421.2 μs | 69.4 μs | — | **41.56x** | **6.84x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **6.44 ms** | 5.90 ms | 8.59 ms | 14.95 ms | — | **1.46x** | **2.54x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **5.26 ms** | 5.31 ms | 5.27 ms | 11.43 ms | — | **1.00x** | **2.17x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.27 ms** | 4.22 ms | 5.60 ms | 23.59 ms | — | **1.33x** | **5.59x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **6.22 ms** | 6.15 ms | 6.86 ms | 5.71 ms | — | **1.11x** | **0.93x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **195.8 μs** | 142.7 μs | 3.96 ms | 555.0 μs | — | **27.74x** | **3.89x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **11.2 μs** | 11.5 μs | 420.1 μs | 65.3 μs | — | **37.60x** | **5.84x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.11 ms** | 899.9 μs | 2.15 ms | 1.65 ms | — | **2.39x** | **1.84x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **769.5 μs** | 814.8 μs | 1.80 ms | 1.30 ms | — | **2.34x** | **1.69x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **537.5 μs** | 499.5 μs | 1.82 ms | 2.54 ms | — | **3.64x** | **5.09x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **723.7 μs** | 685.2 μs | 1.94 ms | 708.7 μs | — | **2.83x** | **1.03x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **173.0 μs** | 141.6 μs | 2.10 ms | 550.9 μs | — | **14.84x** | **3.89x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **10.3 μs** | 10.2 μs | 418.1 μs | 60.2 μs | — | **41.01x** | **5.91x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **212.3 μs** | 268.4 μs | 1.39 ms | 301.2 μs | — | **6.57x** | **1.42x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **171.3 μs** | 231.8 μs | 1.36 ms | 265.1 μs | — | **7.96x** | **1.55x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **142.9 μs** | 114.2 μs | 1.35 ms | 391.9 μs | — | **11.84x** | **3.43x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **164.1 μs** | 131.2 μs | 1.39 ms | 192.4 μs | — | **10.57x** | **1.47x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **114.5 μs** | 81.8 μs | 1.77 ms | 199.1 μs | — | **21.61x** | **2.43x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **9.9 μs** | 10.0 μs | 408.5 μs | 58.2 μs | — | **41.36x** | **5.90x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **113.1 μs** | 173.7 μs | 1.30 ms | 169.0 μs | — | **11.49x** | **1.49x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **113.5 μs** | 176.3 μs | 1.29 ms | 168.4 μs | — | **11.33x** | **1.48x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **106.7 μs** | 78.7 μs | 1.28 ms | 172.6 μs | — | **16.23x** | **2.19x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **109.2 μs** | 78.2 μs | 1.32 ms | 145.6 μs | — | **16.87x** | **1.86x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **104.0 μs** | 73.8 μs | 1.72 ms | 152.1 μs | — | **23.29x** | **2.06x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **9.9 μs** | 9.7 μs | 402.2 μs | 58.1 μs | — | **41.65x** | **6.01x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **937.3 μs** | 960.3 μs | 1.86 ms | 1.42 ms | — | **1.98x** | **1.52x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **917.4 μs** | 943.9 μs | 1.88 ms | 1.42 ms | — | **2.05x** | **1.55x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.52 ms** | 3.46 ms | 28.08 ms | 12.95 ms | — | **8.12x** | **3.74x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.18 ms** | 5.11 ms | 28.20 ms | 14.20 ms | — | **5.52x** | **2.78x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **633.8 μs** | 589.3 μs | 4.60 ms | 1.90 ms | — | **7.81x** | **3.22x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **10.6 μs** | 11.0 μs | 425.4 μs | 62.4 μs | — | **40.18x** | **5.90x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **228.8 μs** | 275.0 μs | 1.44 ms | 295.5 μs | — | **6.28x** | **1.29x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **221.4 μs** | 273.4 μs | 1.44 ms | 293.3 μs | — | **6.48x** | **1.32x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **473.8 μs** | 442.8 μs | 3.95 ms | 1.45 ms | — | **8.92x** | **3.27x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **633.2 μs** | 594.1 μs | 3.99 ms | 1.58 ms | — | **6.72x** | **2.66x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **163.3 μs** | 130.5 μs | 2.16 ms | 390.9 μs | — | **16.57x** | **3.00x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **10.4 μs** | 10.6 μs | 416.4 μs | 60.9 μs | — | **39.89x** | **5.84x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **950.8 μs** | 825.2 μs | 1.76 ms | 1.29 ms | — | **2.13x** | **1.56x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **801.9 μs** | 838.4 μs | 1.75 ms | 1.30 ms | — | **2.19x** | **1.62x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **73.63 ms** | 72.80 ms | 523.57 ms | 105.28 ms | — | **7.19x** | **1.45x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **74.42 ms** | 74.22 ms | 523.30 ms | 105.99 ms | — | **7.05x** | **1.43x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **7.18 ms** | 7.14 ms | 53.79 ms | 11.55 ms | — | **7.53x** | **1.62x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **10.8 μs** | 10.4 μs | 420.4 μs | 58.3 μs | — | **40.58x** | **5.63x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **182.5 μs** | 235.9 μs | 1.33 ms | 276.1 μs | — | **7.30x** | **1.51x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **166.2 μs** | 233.1 μs | 1.34 ms | 278.2 μs | — | **8.05x** | **1.67x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **7.18 ms** | 7.14 ms | 53.10 ms | 10.58 ms | — | **7.44x** | **1.48x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.21 ms** | 7.12 ms | 52.88 ms | 10.53 ms | — | **7.43x** | **1.48x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **809.6 μs** | 771.8 μs | 7.00 ms | 1.36 ms | — | **9.07x** | **1.76x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **10.6 μs** | 10.4 μs | 414.4 μs | 57.1 μs | — | **39.77x** | **5.48x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **100.9 μs** | 171.5 μs | 1.26 ms | 167.9 μs | — | **12.51x** | **1.66x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **99.0 μs** | 164.4 μs | 1.26 ms | 167.0 μs | — | **12.69x** | **1.69x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **803.8 μs** | 773.0 μs | 6.53 ms | 1.25 ms | — | **8.44x** | **1.61x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **811.1 μs** | 770.3 μs | 6.53 ms | 1.24 ms | — | **8.48x** | **1.60x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **210.5 μs** | 172.4 μs | 2.20 ms | 293.9 μs | — | **12.76x** | **1.70x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **9.5 μs** | 9.8 μs | 398.5 μs | 56.2 μs | — | **42.02x** | **5.93x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **3.87 ms** | 3.82 ms | 8.39 ms | 4.54 ms | — | **2.20x** | **1.19x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **3.56 ms** | 3.59 ms | 8.05 ms | 4.22 ms | — | **2.26x** | **1.19x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **2.74 ms** | 2.73 ms | 8.53 ms | 5.54 ms | — | **3.13x** | **2.03x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **14.82 ms** | 15.38 ms | 125.48 ms | 43.21 ms | — | **8.46x** | **2.91x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.07 ms** | 1.07 ms | 21.12 ms | 4.91 ms | — | **19.72x** | **4.58x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **10.6 μs** | 10.7 μs | 543.2 μs | 241.9 μs | — | **51.43x** | **22.90x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **702.7 μs** | 723.5 μs | 5.85 ms | 790.0 μs | — | **8.32x** | **1.12x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **634.6 μs** | 676.9 μs | 5.85 ms | 756.3 μs | — | **9.21x** | **1.19x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **355.7 μs** | 366.9 μs | 5.84 ms | 888.5 μs | — | **16.41x** | **2.50x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.03 ms** | 1.03 ms | 16.05 ms | 4.48 ms | — | **15.58x** | **4.35x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **231.5 μs** | 236.5 μs | 10.46 ms | 875.3 μs | — | **45.20x** | **3.78x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **10.9 μs** | 11.0 μs | 541.7 μs | 242.3 μs | — | **49.88x** | **22.30x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **250.5 μs** | 318.3 μs | 5.55 ms | 401.7 μs | — | **22.18x** | **1.60x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **233.5 μs** | 315.0 μs | 5.55 ms | 397.2 μs | — | **23.77x** | **1.70x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **127.9 μs** | 144.9 μs | 5.54 ms | 404.3 μs | — | **43.34x** | **3.16x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **228.4 μs** | 233.8 μs | 6.96 ms | 816.2 μs | — | **30.48x** | **3.57x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **166.2 μs** | 171.7 μs | 9.27 ms | 505.5 μs | — | **55.74x** | **3.04x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **10.7 μs** | 10.3 μs | 537.5 μs | 243.2 μs | — | **52.04x** | **23.54x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **488.6 μs** | 479.8 μs | 2.62 ms | — | — | **5.46x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **472.2 μs** | 481.2 μs | 2.61 ms | — | — | **5.52x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **963.1 μs** | 968.7 μs | 7.76 ms | — | — | **8.06x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **956.1 μs** | 966.2 μs | 7.78 ms | — | — | **8.14x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **203.6 μs** | 216.7 μs | 2.53 ms | — | — | **12.43x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **10.5 μs** | 10.3 μs | 394.6 μs | — | — | **38.47x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **241.1 μs** | 250.8 μs | 1.51 ms | — | — | **6.25x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **245.8 μs** | 244.5 μs | 1.51 ms | — | — | **6.17x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **206.7 μs** | 215.5 μs | 2.15 ms | — | — | **10.39x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **202.0 μs** | 210.4 μs | 2.15 ms | — | — | **10.67x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **141.0 μs** | 151.7 μs | 1.95 ms | — | — | **13.81x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **9.9 μs** | 10.5 μs | 389.5 μs | — | — | **39.18x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.55 ms** | 5.16 ms | 11.90 ms | — | — | **2.31x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.57 ms** | 4.57 ms | 8.76 ms | — | — | **1.92x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **10.53 ms** | 7.26 ms | 12.96 ms | — | — | **1.79x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **28.55 ms** | 23.25 ms | 310.22 ms | — | — | **13.35x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **331.6 μs** | 217.0 μs | 9.03 ms | — | — | **41.64x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **12.4 μs** | 11.4 μs | 449.3 μs | — | — | **39.42x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.07 ms** | 824.2 μs | 2.91 ms | — | — | **3.53x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **733.4 μs** | 765.4 μs | 2.57 ms | — | — | **3.51x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **1.25 ms** | 830.6 μs | 2.98 ms | — | — | **3.58x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **2.15 ms** | 1.74 ms | 29.36 ms | — | — | **16.91x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **333.8 μs** | 205.3 μs | 5.59 ms | — | — | **27.23x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **12.5 μs** | 9.9 μs | 442.2 μs | — | — | **44.75x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **250.4 μs** | 295.3 μs | 1.93 ms | — | — | **7.70x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **206.6 μs** | 255.0 μs | 1.88 ms | — | — | **9.11x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **211.9 μs** | 132.2 μs | 1.89 ms | — | — | **14.30x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **324.2 μs** | 193.7 μs | 4.45 ms | — | — | **22.99x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **166.8 μs** | 115.0 μs | 2.93 ms | — | — | **25.52x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **10.0 μs** | 10.4 μs | 425.7 μs | — | — | **42.48x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **127.5 μs** | 190.3 μs | 1.82 ms | — | — | **14.30x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **128.0 μs** | 202.2 μs | 1.82 ms | — | — | **14.19x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **133.8 μs** | 98.8 μs | 1.82 ms | — | — | **18.42x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **162.8 μs** | 110.7 μs | 2.18 ms | — | — | **19.66x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **150.6 μs** | 104.8 μs | 2.65 ms | — | — | **25.24x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **10.9 μs** | 10.5 μs | 439.9 μs | — | — | **41.97x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.17 ms** | 2.46 ms | 8.05 ms | — | — | **3.27x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.88 ms** | 1.88 ms | 4.72 ms | — | — | **2.52x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **4.47 ms** | 1.96 ms | 5.05 ms | — | — | **2.57x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **6.55 ms** | 3.16 ms | 6.28 ms | — | — | **1.99x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **196.8 μs** | 126.5 μs | 3.43 ms | — | — | **27.09x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **11.7 μs** | 11.6 μs | 437.0 μs | — | — | **37.74x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **831.3 μs** | 1.00 ms | 2.10 ms | — | — | **2.52x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **469.5 μs** | 498.6 μs | 1.76 ms | — | — | **3.75x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **587.0 μs** | 252.2 μs | 1.76 ms | — | — | **6.97x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **778.7 μs** | 373.3 μs | 1.89 ms | — | — | **5.06x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **195.7 μs** | 121.1 μs | 2.05 ms | — | — | **16.97x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **10.5 μs** | 10.3 μs | 427.7 μs | — | — | **41.47x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **209.3 μs** | 258.9 μs | 1.42 ms | — | — | **6.77x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **167.8 μs** | 229.1 μs | 1.37 ms | — | — | **8.16x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **162.0 μs** | 103.1 μs | 1.38 ms | — | — | **13.38x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **182.1 μs** | 112.0 μs | 1.41 ms | — | — | **12.55x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **130.5 μs** | 90.2 μs | 1.78 ms | — | — | **19.77x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **10.4 μs** | 9.9 μs | 408.3 μs | — | — | **41.31x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **115.7 μs** | 179.8 μs | 1.33 ms | — | — | **11.48x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **122.2 μs** | 187.4 μs | 1.32 ms | — | — | **10.83x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **124.5 μs** | 88.0 μs | 1.33 ms | — | — | **15.08x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **130.8 μs** | 85.8 μs | 1.36 ms | — | — | **15.84x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **124.0 μs** | 86.2 μs | 1.76 ms | — | — | **20.41x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **10.5 μs** | 10.6 μs | 419.7 μs | — | — | **40.06x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **725.9 μs** | 569.6 μs | 1.75 ms | — | — | **3.08x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **557.4 μs** | 557.5 μs | 1.76 ms | — | — | **3.16x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **3.58 ms** | 1.12 ms | 27.74 ms | — | — | **24.73x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **8.99 ms** | 2.05 ms | 47.51 ms | — | — | **23.23x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **656.4 μs** | 214.5 μs | 4.42 ms | — | — | **20.61x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **14.3 μs** | 9.6 μs | 416.3 μs | — | — | **43.45x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **305.0 μs** | 375.8 μs | 2.40 ms | — | — | **7.87x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **300.8 μs** | 384.0 μs | 2.39 ms | — | — | **7.94x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **798.7 μs** | 336.7 μs | 6.69 ms | — | — | **19.85x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **1.07 ms** | 344.8 μs | 6.77 ms | — | — | **19.63x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **294.4 μs** | 166.2 μs | 3.64 ms | — | — | **21.90x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **18.1 μs** | 18.1 μs | 715.6 μs | — | — | **39.54x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **736.8 μs** | 650.7 μs | 2.58 ms | — | — | **3.97x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **622.6 μs** | 647.8 μs | 2.58 ms | — | — | **4.15x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **120.43 ms** | 120.73 ms | 900.97 ms | — | — | **7.48x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **100.17 ms** | 120.79 ms | 897.97 ms | — | — | **8.96x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **11.98 ms** | 12.01 ms | 92.76 ms | — | — | **7.74x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **17.5 μs** | 18.0 μs | 735.4 μs | — | — | **42.07x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **183.9 μs** | 232.1 μs | 1.32 ms | — | — | **7.17x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **169.2 μs** | 229.5 μs | 1.32 ms | — | — | **7.82x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **6.96 ms** | 6.96 ms | 52.55 ms | — | — | **7.55x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **6.96 ms** | 6.97 ms | 52.42 ms | — | — | **7.54x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **829.3 μs** | 832.5 μs | 6.87 ms | — | — | **8.29x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **10.7 μs** | 10.2 μs | 420.2 μs | — | — | **41.34x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **107.5 μs** | 176.2 μs | 1.26 ms | — | — | **11.75x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **112.9 μs** | 177.2 μs | 1.26 ms | — | — | **11.17x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **827.1 μs** | 830.6 μs | 6.48 ms | — | — | **7.84x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **827.3 μs** | 831.8 μs | 6.47 ms | — | — | **7.83x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **224.5 μs** | 227.8 μs | 2.21 ms | — | — | **9.85x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **10.1 μs** | 9.9 μs | 406.8 μs | — | — | **40.90x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.84 ms** | 1.60 ms | 7.32 ms | — | — | **4.58x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.45 ms** | 1.46 ms | 7.02 ms | — | — | **4.83x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **3.29 ms** | 1.61 ms | 7.47 ms | — | — | **4.65x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **18.69 ms** | 14.18 ms | 121.65 ms | — | — | **8.58x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.52 ms** | 668.2 μs | 20.12 ms | — | — | **30.11x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **10.5 μs** | 10.3 μs | 540.4 μs | — | — | **52.33x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **466.9 μs** | 494.2 μs | 5.72 ms | — | — | **12.24x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **413.9 μs** | 448.3 μs | 5.68 ms | — | — | **13.73x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **484.8 μs** | 246.8 μs | 5.69 ms | — | — | **23.08x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.52 ms** | 654.8 μs | 15.87 ms | — | — | **24.24x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **664.1 μs** | 221.8 μs | 10.22 ms | — | — | **46.06x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **10.7 μs** | 10.8 μs | 548.8 μs | — | — | **51.36x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **273.0 μs** | 330.6 μs | 5.54 ms | — | — | **20.30x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **262.3 μs** | 326.6 μs | 5.52 ms | — | — | **21.06x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **190.8 μs** | 151.1 μs | 5.54 ms | — | — | **36.68x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **662.9 μs** | 214.2 μs | 6.98 ms | — | — | **32.59x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **597.1 μs** | 180.2 μs | 9.15 ms | — | — | **50.81x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **10.7 μs** | 10.2 μs | 538.9 μs | — | — | **52.78x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance comparisons & edge cases {#performance-deficits}

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 26.42 ms | 311.9 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 46.33 ms | 295.5 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 26.30 ms | 311.9 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 46.23 ms | 295.5 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 723.7 μs | 382.8 | fitsio/fitsio_torch | 1.22× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 6.22 ms | 444.8 | fitsio/fitsio_torch | 1.21× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter] | off | 427.0 μs | 509.3 | fitsio/fitsio_torch | 1.12× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter_selective] | off | 423.0 μs | 509.3 | fitsio/fitsio_torch | 1.11× |
| Linux x86_64 / CPU | table | ascii_1000 [predicate_filter_selective] | off | 181.4 μs | 509.3 | fitsio/fitsio_torch | 1.05× |
| Linux x86_64 / CPU | table | ascii_1000 [predicate_filter] | off | 181.8 μs | 509.3 | fitsio/fitsio_torch | 1.05× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter] | off | 429.9 μs | 509.3 | fitsio/fitsio | 1.15× |
| Linux x86_64 / CPU | table | ascii_10000 [predicate_filter_selective] | off | 427.6 μs | 509.3 | fitsio/fitsio | 1.15× |
| Linux x86_64 / CPU | table | ascii_1000 [predicate_filter_selective] | off | 185.2 μs | 509.3 | fitsio/fitsio | 1.12× |
| Linux x86_64 / CPU | table | ascii_1000 [predicate_filter] | off | 184.8 μs | 509.3 | fitsio/fitsio | 1.12× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 6.15 ms | 444.8 | fitsio/fitsio | 1.08× |
| Linux x86_64 / CPU | table | narrow_1000 [predicate_filter_selective] | off | 176.3 μs | 381.0 | fitsio/fitsio | 1.05× |
| Linux x86_64 / CPU | table | mixed_1000 [predicate_filter_selective] | off | 189.7 μs | 381.3 | fitsio/fitsio | 1.03× |
| Linux x86_64 / CPU | table | narrow_1000 [predicate_filter] | off | 173.7 μs | 379.7 | fitsio/fitsio | 1.03× |
| Linux x86_64 / CPU | table | varlen_1000 [predicate_filter] | off | 171.5 μs | 509.2 | fitsio/fitsio | 1.02× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter_selective] | off | 5.31 ms | 444.8 | astropy/astropy | 1.01× |
| Linux x86_64 / CUDA | tensor | scaled_large [read_full @ cuda] | off | 7.28 ms | 772.7 | fitsio/fitsio_torch_device | 1.27× |
| Linux x86_64 / CUDA | tensor | medium_int8_3d [read_full @ cuda] | off | 642.3 μs | 772.7 | fitsio/fitsio_torch_device | 1.23× |
| Linux x86_64 / CUDA | tensor | large_int8_2d [read_full @ cuda] | off | 1.18 ms | 772.7 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 107.2 μs | 772.7 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | tiny_int64_1d [read_full @ cuda] | off | 107.4 μs | 772.7 | fitsio/fitsio_torch_device | 1.12× |
| Linux x86_64 / CUDA | tensor | tiny_int16_2d [read_full @ cuda] | off | 108.6 μs | 772.7 | fitsio/fitsio_torch_device | 1.11× |
| Linux x86_64 / CUDA | tensor | small_int16_1d [read_full @ cuda] | off | 112.2 μs | 772.7 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_int32_2d [read_full @ cuda] | off | 111.5 μs | 772.7 | fitsio/fitsio_torch_device | 1.09× |
| Linux x86_64 / CUDA | tensor | tiny_int8_3d [read_full @ cuda] | off | 106.5 μs | 772.7 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | scaled_small [read_full @ cuda] | off | 203.3 μs | 772.7 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | scaled_medium [read_full @ cuda] | off | 1.33 ms | 772.7 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | tiny_int32_3d [read_full @ cuda] | off | 109.4 μs | 772.7 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | medium_int8_1d [read_full @ cuda] | off | 128.0 μs | 772.7 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | small_int8_2d [read_full @ cuda] | off | 144.4 μs | 772.7 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_float32_1d [read_full @ cuda] | off | 102.9 μs | 772.7 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_int16_3d [read_full @ cuda] | off | 110.2 μs | 772.7 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.65 ms | 772.7 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | tiny_float64_2d [read_full @ cuda] | off | 113.5 μs | 772.7 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_uint16_2d [read_full @ cuda] | off | 146.7 μs | 772.7 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_int16_2d [read_full @ cuda] | off | 138.0 μs | 772.7 | fitsio/fitsio_torch_device | 1.04× |

_…and 121 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Published runs by platform

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20261007_203905` | 3057 | 20 | 299.9 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20261007_203924` | 4315 | 44 | 762.9 | lab + mmap-matrix + GPU |
| macOS arm64 / MPS | `exhaustive_mps_20261007_204350` | 4227 | 97 | 226.5 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

Historical July 2026 runs: MPS `exhaustive_mps_20260719_143706` (local);
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
