# Post-1.0 backlog (from deep_review rounds)

Deferred after the 1.0 triage passes. Do not block the 1.0 tag on these.

## Shipped in thin-I/O wave (Pass-3 + skinny meta)

- Skinny `read_*` metadata set + caller wiring (Datasets, image_meta, examples)
- `open_table_reader`, `table.read_torch(where=)`, thin table dispatch
- Pass-3 P1/P2 HTTP cutout byteswap/clone; P6 SigmaClip scalar fill; 8.1 end_row hoist
## Scope cuts (later)

- Merge dual cache subsystems (`cache.py` vs `_io_engine/caches.py`) — document relationship first
- Collapse Dataset class zoo (`FitsImage*` / `FitsCube*` / …) into fewer constructors
- CLI trim (`compress` / `decompress` / `arith` vs fpack/numpy) — Waves 1–2
  shipped (`-j`/`-J`, imarith-class `arith`, batch copy/transform/cutout,
  stats std/median, compress `--algorithm`, header wildcards, setkey
  `--delete`/`@list`). **Wave 3:** thin fitsverify subprocess helper (not
  silent expand of checksum verify); fpack tile/dither/`-i2f`; WCS/catalog
  cutouts; `imexpr`/full STILTS — **not** CFITSIO HTTPS drivers (keep own
  HTTP + SSRF).
- Scorecard / CANFAR re-soak after thin-I/O — **done** (Round-3: MPS/CPU/CUDA
  `exhaustive_*_20260719_14*`; see `docs/benchmarks.md`)
- MegaCam `torchfits_cached` vs `fitsio_cached` (Round-3 CSV on NRC-054711):
  real OK-row medians were **49.3 vs 52.7 MB/s** (~6.5% behind), not the old
  docs table (65.7/55.2 were unrelated μs). Mean already favored torchfits.
  Fresh soak on lab host (2026-07-30) led ~8% — no handle-reuse bug found;
  Rice path is decompress-bound (`materialize` still ~2×). Revisit only with a
  new same-host repro that lags.
- Narrow-table `read_full` ~1.06–1.15× behind fitsio (small; polish later)
- **predicate_filter Round-3 investigation:** fused `where=` used filtered
  gather first; dense ``col > 0`` lagged Astropy numpy ~20–28%. Fixed by
  preferring project+torch-mask in `read_table`/`read_torch` (lab: ~3× vs
  Astropy on narrow 1e6). Selective gather path kept as fallback only.
  Bench still has `predicate_filter` (dense) and `predicate_filter_selective`.

## 1.1 audit deferrals (post-1.1 backlog)

Findings from the pre-1.1 full audit that were triaged as non-blocking.
Ordered roughly by value; each names the site so a future fix can start fast.

### C++ robustness / perf

- GIL held through long IO in `read_full_numpy` (incl. network opens),
  `get_header`, `get_num_hdus` (`fits_bindings.cpp` ~1482–1491, 1556+).
- mmap table paths skip the truncation bounds check the image layer has →
  SIGBUS instead of exception on truncated files (`table_reader.h`
  read_columns_mmap / _filtered / update_rows_mmap; contrast
  `fits_detail.h:513`).
- Strided DLPack numeric payloads silently miswritten in
  `update_rows_mmap` (bool/uint8/string honor strides; numerics index flat).
- Column repeat int32 truncation unguarded for non-string typecodes
  (hostile headers; `table_reader.h:167ff`) + duplicate-TTYPE moved-from UB
  (`:551/:599`).
- `read_full_numpy` float-promotes scaled images (no unsigned convention)
  while tensor paths return uint16/uint32 — decide and document.

### Performance: narrow-table buffered full-read — RESIDUAL QUANTIFIED (2026-08-22 late)

Final CANFAR state after prefetch + corruption fix + fan-out revert:
exactly ONE significant deficit family remains (narrow-table
`read_full`, `mmap=False`, 1.11-1.25x vs fitsio depending on host;
node-normalized via scan_count index the CPU-host number is ~1.08-1.15).
Everything else noise-level or won outright. Next lever unchanged
(single-pass arena decode, 1.2, API-visible). Prefetch now gated to
payloads >= 64 MB: overlap regressed 13 MB warm-cache tables (thread
handoff > warm pread) but holds for large/cold payloads.

### Performance: narrow-table buffered full-read (updated 2026-08-22, evening)

Double-buffered prefetch landed (chunk N+1 pread overlaps chunk N
extract): local single-thread 15.5 -> 10.4 ms on the bench schema; CANFAR
CPU rerun pending. Deeper findings from today's fan-out experiment:

- Per-column CFITSIO reads cost ~8.4 ms EACH locally (1M-row, 13 B rows)
  regardless of threading — `fits_read_col`'s internal row buffering makes
  column-at-a-time strictly worse than our whole-row pread. This is why
  `fitsio`'s apparent win does NOT transfer via per-column strategies and
  why read_column_by_column loses on wide tables.
- A cross-thread fan-out of those same primitives was implemented and
  measured 3-8x WORSE (CFITSIO serialization + per-call overheads);
  reverted. Reader cache now proven warm across worker threads when keyed
  with a slot tag — machinery kept in git history if ever needed.
- Remaining structural lever for the last ~6-13%: single-pass decode into
  caller-visible memory (arena + strided views) — an API-visible change
  (non-contiguous column tensors), deferred to 1.2 design.
- hcompress residual (~1.02-1.03x): three builds of the same CFITSIO
  family span 85.2/89.0/91.9 ms on identical hardware (fitsio-bundled /
  our vendored 4.7.0 / astropy-bundled). Build variance floor, not
  algorithmic.

### Performance: narrow-table buffered projection (measured 2026-08-22)

- `predicate_filter`/`_selective` (mmap=off) remain 26–32% behind
  astropy-numpy on `narrow_1000000`. Profiled: ~10 ms is the C++
  **buffered single-column read** itself — `read_columns_buffered`
  preads whole rows (~20 MB for a 4-col table) then de-interleaves one
  column, versus fitsio/astropy reading only the target column's bytes.
  Python-side mask/gather is already sub-dominant (numpy-vs-torch mask
  saves ~1 ms; verified not the bottleneck). Next lever: a selective-
  projection fast path in the buffered reader (per-column CFITSIO reads
  or strided preads) with explicit TSCAL/TZERO/TNULL semantics parity —
  needs its own bench A/B before landing.
- `narrow_1000000::read_full` mmap-off trails fitsio ~1.11x (5.6 vs
  5.1 ms) — same reader, full-width rows, likely same lever.

### Table semantics polish

- `schema()` reports complex columns (`C`/`M`) as float64 scalars;
  unnamed-column capability checks skip validation; empty-result reads
  silently drop requested unknown columns (`_read_schema.py`).
- `TableHDURef.head(n)` replaces the existing row window instead of
  composing; in-memory `TableHDU.head(-n)` truncates tail rows instead of
  raising; caches keyed on `id(self.header)` are GC-reusable.
- Arrow width-1 chunk route can surface `FixedSizeList<T>[1]` where the
  schema maps repeat==1 to scalar (main decode path verified scalar-only).
- `to_astropy`: TNULL null columns become object dtype; TUNIT not mapped to
  `.unit`. `TableHDURef.to_arrow(columns=...)` kwarg collision.
- `table.write(quantize=)` silently no-ops when no column qualifies.
- Error-type inconsistency across mutation API (KeyError vs ValueError for
  unknown column); broad `except Exception` fallback swallows mask real IO
  errors (`table_api.py`, `_read_scan.py`).

### Remote / data pipeline

- Multiprocess download races on shared `.partial`; resume lacks
  If-Range/ETag; Content-Length-less truncation can be promoted to the
  permanent cache (`data/remote.py`). Staged-cutout cleanup races +
  make_loader double-download of staged remotes.

### CLI / http

- Unhandled non-CliError exceptions exit 1 (= documented diff-discrepancy
  code) in `verify`/`stats`; `stats --json` emits bare NaN/Infinity;
  copy/compress/convert accept OUTPUT == INPUT with lossy card round-trip;
  KeyboardInterrupt returns usage-code 2; `-J` shares one stateful
  transform instance across threads.
- DNS-rebinding TOCTOU contradicts `http_util` docstring (guard resolves
  once; CFITSIO/urllib re-resolve at connect).

### Hygiene / tests

- `[test]` extra cannot run the suite (astropy/fitsio/psutil undeclared);
  declared `performance` marker unused so wall-clock/RSS tests always run;
  GHA release gate omits 4 suites present in local release-gate;
  security.h bracket test manual-only (wire into build); sleep-based race
  in `test_data_datasets.py` prefetch dedupe test; several suites write
  fixtures into process CWD.
- Dead/duplicated code sweep: `read_scaled_cpu_fast`,
  `clear_file_cache(handles=)` unread flag, `_normalize_cpp_chunk` no-op,
  unused header_parser regexes, unreachable inf-guard in `clip.py`,
  worker-split block duplicated ×3 in datasets.py, `_normalize_row_slice`
  ×2, fallback-table double-open per call, negative meta lookups uncached,
  HTTP cutout walks HDU headers twice.

## Round 7 deferrals (safe post-1.0)

- R7-HDU1 — ~~`TensorHDU.to_tensor()` closed-handle guard~~ — fixed
- ~~R7-HDU2 — `TableDataAccessor` squeeze on `(N,1)` (intentional FITS scalar shape)~~ — fixed
- ~~R7-CPP1 — floating-point equality for unsigned TZERO (malformed files only)~~ — #235
- R7-CPP2 — thread-local HDU cache stale after shared-meta invalidation
- R7-CPP3 — `cache.cpp:clear()` retain borrowed handles (ponytail)
- R7-MUT4 — `_normalize_mutation_rows` preprocesses all columns
- B1 — duplicate mmap/torch capability-check helpers in `_table/read.py`

## Pass-3 deferrals (low ROI)

- P3 batch `stack().to().unbind()` VRAM
- P5/P8 SharedReadMeta mutex coalescing / `shared_mutex`
- P10 OrderedDict cache locks
- P11/P12 GPU fallback cache / table device move
- NIT-CPP-2 tl_cache LRU (overlaps R7-CPP2)

## Hygiene / structure

- ~~Header HISTORY/`remove` O(N²) for huge HISTORY lists~~ — #225
- ~~Vestigial `UnifiedCache` shared-handle path~~ — stubs; live state is SharedReadMeta
- ~~`_table/cache.py` no-op close/invalidate stubs after Option A~~ — removed
- Split `_table/read.py` mega-function strategies
- Split `_io_engine/write_api.py` / `_table/mutation.py` coerce vs ops (audit defer)
- Broader `except Exception: pass` audit (soft fallthroughs in strategy probes;
  Round-2 glm notes: batch `read_images_batch` silent fallthrough, NAXIS2→0,
  tnull fill swallow, `update_rows` mmap=auto swallow)
- Install: consider a **2.11+ / 2.13** wheel ABI lane only after scorecard re-soak
  (today: wheels + pixi stay on **PyTorch 2.10**; source builds allow ≥2.10)

## Spectroscopy / continuum (not in torchfits)

Continuum and spectral `FITSTransform`s were **deleted** from torchfits (no
deprecation). Absorb-vs-new design belongs in the sibling astronomy stack repo.
