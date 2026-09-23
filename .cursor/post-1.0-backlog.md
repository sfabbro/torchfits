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

- ~~GIL held through long IO in `read_full_numpy`, `get_header`, `get_num_hdus`~~ — released (`gil_scoped_release`); confirmed at R9.
- ~~mmap table paths skip the truncation bounds check~~ — `ensure_extent_within_file` is on those paths; confirmed at R8.
- ~~Strided DLPack numeric payloads silently miswritten in `update_rows_mmap`~~ — numerics honor strides; re-derived already fixed at R8.
- ~~Column repeat int32 truncation and duplicate-TTYPE moved-from UB~~ — repeat guard and r8a-04 dedup landed.
- ~~`read_full_numpy` skips the unsigned convention~~ — bitwise match of `read_tensor(...).numpy()` since R9. Image BSCALE accumulation stays float32 until 2.0 (see 1.2 audit deferrals).

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
- ~~`TableHDURef.head` replaces the window; `TableHDU.head(-n)` truncates~~ — R6 composes the window and raises `ValueError` for `n < 0`.
- Arrow width-1 chunk route can surface `FixedSizeList<T>[1]` where the
  schema maps repeat==1 to scalar (main decode path verified scalar-only).
- `to_astropy`: TNULL null columns become object dtype; TUNIT not mapped to
  `.unit`. `TableHDURef.to_arrow(columns=...)` kwarg collision.
- ~~`table.write(quantize=)` silently no-ops when no column qualifies~~ — raises `QuantizeError` (R5).
- Error-type inconsistency across mutation API (KeyError vs ValueError for
  unknown column); broad `except Exception` fallback swallows mask real IO
  errors (`table_api.py`, `_read_scan.py`).

### Remote / data pipeline

- Multiprocess download races on shared `.partial`; resume lacks
  If-Range/ETag; Content-Length-less truncation can be promoted to the
  permanent cache (`data/remote.py`). Staged-cutout cleanup races +
  make_loader double-download of staged remotes.

### CLI / http

- ~~CLI exit 1 / bare JSON NaN / same-path copy / Ctrl-C exit 2 / shared `-J` transform~~ — R7: exit 5, JSON `null`, same-path refusal, exit 130, per-file instances.
- ~~DNS-rebinding TOCTOU contradicts `http_util` docstring~~ — Python `http`/`https`/`ftp` fetches are pinned; the CFITSIO residual is documented in `docs/compatibility.md` (R10/R15).

### Hygiene / tests

- ~~`[test]` extra, unused `performance` marker, release-gate list gap, manual bracket test, sleep-based prefetch race~~ — closed in R10/R11. Release-gate file lists match (20 files). A few suites may still write into the process cwd; that residue was not re-swept here.
- Dead/duplicated code sweep: `read_scaled_cpu_fast`,
  `clear_file_cache(handles=)` unread flag, `_normalize_cpp_chunk` no-op,
  unused header_parser regexes, unreachable inf-guard in `clip.py`,
  worker-split block duplicated ×3 in datasets.py, ~~`_normalize_row_slice` ×2~~ (one copy since r6b-02), fallback-table double-open per call, negative meta lookups uncached,
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
- ~~Wheels and pixi stay on PyTorch 2.10~~ — the wheel lane is PyTorch 2.13 (`scripts/torch_lanes.json`). Source builds still allow ≥2.10.

## Spectroscopy / continuum (not in torchfits)

Continuum and spectral `FITSTransform`s were **deleted** from torchfits (no
deprecation). Absorb-vs-new design belongs in the sibling astronomy stack repo.


## 1.2 audit deferrals

Open after the R1–R16 directory pass. Do not re-fix these in silence: each one was recorded with a reason. A later round closed the struck ids below.

### Still open

- r1b-08 — `docs/api-core-io.md` shows `read_extname(path, hdu=1)`; `io.read_extname` defaults `hdu=0`.
- r1b-09 — `io._READ_EXC_TYPES` includes `TypeError`, `AttributeError`, and `MemoryError`, so `read_batch(strict=False)` skips those as failed paths.
- r1c-11 — `docs/api-tables.md` predicate helpers omit `where_identifier_re` (it is in `where.__all__`).
- r1c-13 — `decode_byte_tensor(..., errors="ignore")` drops undecodable bytes. Raising would change string-column reads.
- r2a-12 — `docs/api-transforms.md` import list omits names the package exports (`FITSTransform` and peers). Doc-only.
- r2a-16 — `FITSHeaderScale.forward` runs `check_state` twice (the call plus `view()`). Count is 2→1; no wall-clock evidence.
- r2a-17 / r2b-12 — inverse stats live on the transform instance. Moving them into `Payload` breaks pinned dict-payload tests.
- r3a-16 — `FitsSpectrumDataset` accepts `mmap=` and does not pass it to `read_torch`.
- r3a-17 — `read_images_batch` still has a per-file fallback; table `num_rows` probes swallow errors.
- r3b-09 — `data._eager_table_columns` falls back to pyarrow on any `read_fits_table` exception, including I/O errors.
- r4a-12 — a negative image-meta lookup is not cached (`signature_cached_get` cannot tell a miss from a stored `None`).
- r4a-14 — `image_meta._cache_get` / `_cache_set` and `subset.read_subset` have no callers.
- r4c-14 — `update_rows(mmap="auto")` and the tnull-fill path still swallow non-decode errors.
- r5a-13 — `docs/api-tables.md` backend wording drifts from `_table_engine/read_policy.py`.
- r5a-14 — `stream_table` re-reads the header for its XTENSION probe even when `total_rows=` is set.
- r5a-16 — `fits_schema._iter_tfields_indexed` skips TTYPE-less columns, so every consumer misses them.
- r5b-06 — ASCII string width still truncates to the TFORM width (astropy does too). No typed error.
- r5b-08 — `os.unlink` of the rewrite temp file and `_parse_tform` use `except Exception`.
- r5c-12 — `table.write_csv` / `table.write_ipc` are in `table.__all__` and have no docs section.
- r5c-13 — width-1 `FixedSizeList` can still surface off the main decode path.
- r6a-06 — orphan CONTINUE fusion is fixed (`d424c05`, typed-target gate). What remains: `HDUList` header values stay raw batch-triple strings (`NAXIS` is `"0"`) while `read_header` types them (`NAXIS` is `0`). That split exists on files with no long strings.
- r6a-07 — `read_header_fast`'s slow fallback and `_read_pipeline_fallback.py` still build headers from raw triples.
- r7a-08 — truncated-file HDU scan errors are still the engine's CFITSIO status, not a typed CLI code.
- r7a-09 — `cmds_probe._probe_vos` swallows `handle.close()` failures.
- r7b-12 — table JSON preview stringifies bytes and complex values with `str(value)`.
- r7b-13 — a failed remote `copy` leaves a partial output file (exit 3).
- r7c-15 — `arith -o <same path>` is allowed. The read finishes before the write, so it is in-place rather than corrupt.
- r7c-16 — `compress` / `convert` / `transform` / `arith` leave a partial output if the write fails mid-way.
- r7c-17 — `compress` then `decompress` of a single-image file yields an empty primary plus an image extension.
- r7c-18 — image–image `arith` does not look at `BUNIT`.
- r7c-19 — `BITPIX=64` multiply stays in int64 and can wrap at 2^63.
- r7c-21 — a header dict whose commentary value is a list (`{"HISTORY": ["h1", "h2"]}`) hits `std::bad_cast` in card replay.
- r7c-22 — replaying commentary cards appends `HISTORY`/`COMMENT` instead of replacing them.
- r7c-23 — `write_parquet` / `write_csv` / `write_ipc` reject a column dict that `table.write` accepts.
- r7c-24 — `docs/cli.md` shows `setkey --comment`. `cmds_setkey.py` has no such flag.
- r7c-25 — non-finite pixels cast to an integer output are platform-dependent.
- r8b-08 — table write bindings and `TableReader` construction hold the GIL across CFITSIO calls.
- r8b-09 — after a failed `fits_write_col`, the column loop keeps going with a poisoned status.
- r8b-10 — filtered table reads reject `np.int64` / `np.int32` filter values (`np.float64` works).
- r9b-06 — `open_fits_for_write` does not clean up a partial handle. A 200-iteration fd probe found no leak.
- r11a-07 — `tests/test_bench_suites.py` pins source text with `inspect.getsource`.
- r11a-08 — `TestCaching.test_cache_clearing` calls `get_cache_performance()` and asserts nothing.
- r11a-09 — `test_clear_cache_disk_true_parameter` calls `clear_cache(disk=False)`.
- r11a-10 — `tests/test_api.py` repeats the `mode="invalid"` block.
- r12a-05 — full-image MB/s uses on-disk file bytes, so a compressed image does not report decoded payload throughput. GPU `read_full` throughput stays blank.
- Image BSCALE/BZERO accumulation is float32 for 1.2 (`read_full_scaled_cpu`). The float64 image path is 2.0.
- Arena decode and the `table_reader.h` / `write_api.py` splits stay out of 1.2 (API-visible or megafile work).
- `tests/test_remote_resume.py` `_FakeVosClient.copy` sleeps 0.2s after a `threading.Barrier(2)` so both writers stay inside `copy()`. Not a race.

### Struck (closed after the deferral)

- r1c-12 — blocked-prefix split documented and tested on both layers (R10).
- r1c-14 — CFITSIO residual TOCTOU is stated in `docs/compatibility.md` (R15). Python fetches stay pinned.
- r1b-10 — `to_astropy` accepts `os.PathLike`.
- r4a-15 — `read_batch(..., strict=False)` docs match the warn-and-skip default.
- r4b-13 — LONGSTRN chains reassembled at `HDUList.fromfile` (r6a-01).
- r5c-09 — `TableHDURef.to_arrow(columns=)` raises a typed duplicate-argument error (r6b-01).
- r5a-12 / r5b-07 / r5c-15 — one `_normalize_row_slice` (r6b-02).
- r5b-05 — "preprocess every column" was refuted; not a defect.
