# Deep Review 1 — Faithfulness vs fitsio / astropy (verified findings)

Status: RESOLVED — experiments verified in-source on lane 2.13, python 3.13,
commit ad527c1. Each finding below was reproduced with a standalone probe
(images BSCALE/BZERO, tables TSCAL/TZERO, CONTINUE cards, uint64, TNULL).
Resolution commit: (see git log; wave-5 fixes + tests landed on main together).

## Confirmed defects (reproducer → wrong/divergent result)

### F1. CONTINUE cards parsed as literal 68-char value (read)
- File written by astropy with a 120-char `LONGKEY` (astropy emits `CONTINUE`
  cards; `CONTINUE` present in raw bytes).
- `torchfits.read(return_header=True)["LONGKEY"]` → 68 chars, truncated at the
  first card; astropy reconstructs all 120 chars.
- Source: `src/torchfits/header_parser.py:160-163,188-191` — no continuation
  logic; value ends at the first card.
- Divergence: astropy=120 chars, fitsio=120 chars, torchfits=68.

### F2. uint64 tensor write unsupported (images and tables)
- `torchfits.write(path, uint64_tensor)` → `RuntimeError: ... Unsupported
  tensor dtype` (wrapped; underlying dtype rejection in `_write_helpers.py:56-73`).
- uint16/uint32 have BZERO strategies; uint64 has none. No documented error
  path; users get a bare RuntimeError with no guidance.

### F3. TSCAL/TZERO ignored for FLOAT/DOUBLE table columns (read)
- Table with `TSCALE2=2.0, TZERO2=500.0` on a float64 column: astropy and
  fitsio return physical values `[1000, 1002, ...]`; torchfits
  `table.read_torch` returns raw `[500.0, 502.0, ...]`.
- Source: `src/torchfits/cpp_src/table_reader.h:810-818` — in-memory scaling
  step explicitly skips FLOAT/DOUBLE (and STRING/LOGICAL/VLA); integer-like
  columns ARE scaled correctly (matches astropy on int32 probe).
- Combined effect: any real-world file that scales float columns (rare but
  legal FITS) silently returns wrong values with no warning.

### F4. `where=` on any scaled table column raises (read_torch)
- `table.read_torch(where="val > 60")` on a TSCAL'd table → `RuntimeError:
  Failed to apply where=...`.
- Chain: mask path `_thin_read_table_torch` uses mmap → C++ raises "Scaled
  columns not supported for mmap"; filtered fallback also fails/None; caller
  raises. The mask path would also compare RAW values if it ran (predicate
  applied pre-scaling, `_read_where.py:181-193`), which would be a silent
  wrong-answer risk even for int columns.

## Behavioral gaps (verified, severity low, no wrong answer)

### G1. Write truncates long header values silently
- `header={"LONGKEY": "x"*120}` → written file has 68 chars, NO `CONTINUE`
  cards, no warning logged (fitsio truncates with a warning; astropy writes
  CONTINUE cards). Truncated metadata persists silently.

### G2. mmap read on scaled table column raises instead of falling back
- `read_torch(mmap=True)` on TSCAL'd table → RuntimeError. Documented in
  docs/api.md:158 as intended, but "auto" also inherits the failure for
  `where` paths; no graceful buffered fallback.

## Verified-correct (checked, no action)
- Image BSCALE/BZERO: normal + mmap read match astropy/fitsio exactly.
- int32 table TSCAL/TZERO: matches astropy (fitsio itself overflows here —
  its quirk).
- Float NaN/Inf write/read roundtrip: native float storage, no BLANK needed.
- TNULL read: `read_torch` returns raw sentinel (no masking, no TNULL param on
  read_torch) — parity with fitsio's default; masking exists only in the
  unified `read()` path via apply_fits_nulls.

## Suggested fix plan (for approval — report-first)
1. F1: CONTINUE card assembly in `header_parser.py` (concatenate continuations,
   honor standard 68-char card encoding incl. `END`; add regression tests
   round-tripping astropy-written long strings for HEADER/COMMENT/HISTORY).
2. F2: uint64 write — decide: reject with clear `ValueError` + docs (least
   change) or implement BZERO=2**64 (BITPIX -64 not standard; likely reject).
3. F3: extend `ensure_column_scale`/read scaling to float/double columns in
   `table_reader.h` (match CFITSIO's fits_read_col scaling semantics); keep
   integer fast path.
4. F4: where-mask path must compare PHYSICAL (post-scale) values; and any
   scaled column should fall back from mmap to buffered silently rather than
   raising (plus docs).
5. G1: emit `CONTINUE` on write for values > 68 chars (parity with astropy) or
   warn; decide per fix plan.

## Resolution log (wave-5, all landed on main)

### F1 (resolved)
- `parse_header_string` + `fast_parse_header_cards` both assemble CONTINUE
  chains: LONGSTRN `&'`-terminated segments drop `&` (restored when no
  CONTINUE follows); bare CONTINUE cards append with trailing blanks stripped.
- Verified: torchfits r/w 121-char round-trip, astropy-written 120-char files,
  literal trailing `&`, escaped quotes across CONTINUE.
- Tests: `tests/test_deep_review_wave5.py` (longstr round-trip, astropy file,
  trailing `&`, plain CONTINUE).

### F2 (resolved)
- `_host_tensor_for_fits_write` and `_prepare_unsigned_table_data_for_write`
  raise `ValueError` naming the column/type with conversion guidance
  (int64 < 2**63 or float64); `write_api.py` re-raises ValueError unwrapped
  instead of burying it in RuntimeError. Docs note added to
  `docs/api-core-io.md` (Writes section).
- Tests: uint64 image + uint64 table column both assert the ValueError.

### F3 (resolved)
- `table_reader.h` scaling loop no longer skips FLOAT/DOUBLE; float64 and
  int32 columns with TSCAL=2/TZERO=500 return physical values matching
  astropy on buffered, mmap, and unified `torchfits.read`.
- Tests: parametrized float64/int32 scaled reads (buffered + mmap).

### F4 (resolved, plus a latent bug found)
- `_thin_read_table_torch` retries buffered when mmap raises on scaled
  columns (physical values + working `where=`).
- Latent bug found while testing: the where-mask in `read_table` only masked
  `torch.Tensor` values; numpy-backed columns (the buffered path) silently
  ignored the predicate → `where=` returned unfiltered data. Mask now applied
  to ndarray columns too.
- Contract fix: the buffered fallback returns numpy from C++; read_torch's
  documented contract is torch.Tensor columns, so the fallback converts
  ndarray → tensor (regression: examples/example_quantize_int16.py and
  example_table_interop.py broke when the fallback returned ndarray).
- Tests: scaled where-select (physical comparison), scaled mmap reads.

### G1 (resolved)
- `write_hdu_header_cards` in `fits_bindings.cpp` uses `fits_update_key_longstr`
  for values > 68 chars → standard LONGSTRN CONTINUE cards on disk (verified
  in raw bytes), full-value read-back round-trip.
- Tests: written bytes contain `CONTINUE`, read-back equals original.

### GIL-hold (perf, resolved)
- `read_columns_mmap` returns `unordered_map<string, torch::Tensor>`; mmap
  read bindings release the GIL during the C++ read and acquire only for
  conversion; `read_fits_table_rows_numpy` mmap branch likewise.
- Tests: concurrent mmap full-reads stay consistent across threads.

### Per-batch reopen (perf, resolved)
- New `open_fits_mmap_reader` (persistent TableReader capsule) +
  `read_fits_table_rows_mmap_from_reader`; `_iter_chunks_cpp_table` opens once
  per scan and reads every batch from the reader (was: re-open + re-parse
  header per batch — 13 opens per 13-batch narrow scan).
- Tests: monkeypatched scan asserts exactly 1 open, 8 reader reads, 0 legacy
  opens for 8 batches.

### NIOBUF/MINDIRECT (perf, verified no-op → confirmed working)
- Verified end-to-end: vendored headers carry the `#ifndef` guards (patch
  applied at configure) and configure emits `-- Overriding CFITSIO NIOBUF=80 /
  MINDIRECT=8640` from the pyproject cmake.args. Mechanism confirmed
  functional (no silent no-op remains).
- Test: guards present in `extern/cfitsio/fitsio.{h,2h}`.

### Scale-on-device dead path (perf, resolved)
- `_apply_scale_on_device` / `_apply_unsigned_offset` were defined but never
  called from production code (tests only). Removed; live `scale_on_device`
  path coverage kept in `tests/test_scale_on_device.py` (integration tests).
  `tests/test_signed_byte_device_scale.py` (pure unit tests of the dead
  helpers) deleted.

### Grid deficit context
- Grid (41 legs, commit 9f8f2a1) attributed the top deficits to GIL-hold
  (compressed_hcompress read_full), per-batch reopen (narrow/varlen scans),
  and CUDA small-payload transfer overhead. All source-level items above are
  landed; CUDA small-payload gap is transfer/launch-bound (scale-on-device
  wiring would worsen it) and stays as a benchmark follow-up.

### Wave-1 image/cube profile (2026-08-05)
- Re-ran the flagged >2D image cases on the local host with the existing
  `bench_fits_io.py` schedules, both mmap modes, and a direct same-process
  probe of the C++ entry points. The previous large gaps were not reproducible
  as a stable local result: plain 2D/3D integer reads were generally at or
  ahead of `fitsio_torch`; only compressed HCOMPRESS remained a small gap.
- The arbitrary-scaled `scaled_large` path was profiled because it was the one
  repeatable CPU candidate. An unused raw-read + scale helper was tested, but
  its float32 arithmetic differed from CFITSIO by one ULP on robust-quantized
  images. A double-intermediate implementation preserved parity but regressed
  the representative scaled image, so the speculative dispatch was removed.
- Correctness gate stayed strict: the output-parity suite remained green after
  each experiment (`102 passed` in the focused image/parity run). No Wave-1
  source change is retained until a same-host profile demonstrates a stable
  image/cube deficit and a fix that is both faster and bit-identical.

### Wave-2 cache/batch follow-up (2026-08-05)
- B1 was already resolved by the table-module split: `_table/read.py` now
  re-exports the capability helpers from `_read_schema.py`; the full-read
  predicate implementation is single-sourced there. No duplicate helper patch
  was warranted.
- P10 landed as a Python-side `RLock` around the OrderedDict cache sequences.
  The lock covers read-cache hit/move/evict, image metadata/mmap policy LRUs,
  HDU/header LRUs, fallback cache stores, invalidation, and statistics. A new
  threaded cache test performs 32 concurrent header+image reads and compares
  every output to the reference array.
- P3 was measured but not changed: this host has no CUDA device, and the C++
  batch readers return CPU tensors before `batch_to_device`. A speculative
  stack removal would trade one batched H2D transfer for many launches without
  a measurable VRAM result, so the implementation remains unchanged pending a
  CUDA-host measurement.

### Wave-3 C++ cache internals (2026-08-05)
- `fits_bindings.cpp` thread-local HDU metadata now uses a bounded 4096-entry
  LRU (`std::list` + `unordered_map`) instead of clearing the complete map at
  the threshold. `LocalKey` retains `SharedReadMeta::uid`, so entries from an
  invalidated metadata generation cannot be returned; LRU bounds their stale
  residency.
- `SharedReadMeta::mutex` is now a `std::shared_mutex`. Read-only metadata
  lookups use shared locks; cache population, invalidation, raw-fd creation,
  and current-HDU updates use unique locks. This reduces reader contention
  without sharing mutable CFITSIO handles.
- Focused concurrent-read, cache, image/table parity, and deep-review tests
  passed (`119 passed`).

### Wave-4 item 10: CUDA small-payload measurement (2026-08-05)
- The active host has no CUDA device, so the existing matched CUDA artifacts
  were analyzed instead of using a CPU proxy: 21 CUDA lanes, 3,654 paired
  `read_full`/`read_full_gpu` rows, mmap-on, same case and host.
- Tiny medians: torchfits `0.0656 -> 0.1158 ms` (added `0.0509 ms`) versus
  fitsio `0.0911 -> 0.1094 ms` (added `0.0179 ms`). Small medians: torchfits
  `0.1127 -> 0.1801 ms` (added `0.0732 ms`) versus fitsio `0.1588 -> 0.1857 ms`
  (added `0.0262 ms`). The combined launch/H2D/conversion overhead hypothesis
  is supported for small payloads, but the CSV cannot distinguish launch from
  transfer without Nsight/CUDA events.
- `large_uint16_2d` also has a host decode/dtype gap (`3.0271 -> 3.3141 ms`
  torchfits; fitsio `2.4247 -> 2.2133 ms`), so it is not a small-transfer-only
  issue. No speculative CUDA graph or stream-manager change was landed;
  methodology and disposition are documented in `docs/benchmarks.md`.

### Wave-4 item 9: MegaCam Rice repro (2026-08-05)
- Ran the existing local CFHT sample with two `.fits.fz` files, four image HDUs,
  and 40 256x256 cutouts per HDU. `torchfits_cached` led `fitsio_cached` on
  all four cases by 7.5–15.2% (new run:
  `/scratch/.tmp-sfabbro/opencode/megacam-wave4/20260805_004215`).
- `torchfits_materialize` was faster but used more RSS and is a separate
  full-plane algorithm, not a like-for-like cached cutout path. The required
  lag repro did not appear and no Rice/materialize code change was retained.

### Wave-4 item 8: combined table mutation disposition (2026-08-05)
- Audited fitsio 1.4.2 and `astropy.io.fits`: neither exposes a public
  arbitrary-position `insert_rows` operation comparable to torchfits. fitsio
  exposes lower-level `delete_rows`/`resize`/column writes, while Astropy can
  rebuild a table in memory and write it once; those are not like-for-like
  competitors for the public mutation contract.
- The user's competition gate is therefore not met. No combined mutation
  capsule was added; revisit only if a competitor exposes the same operation
  or a real torchfits caller demonstrates repeated insert+update sessions.

### Wave-4 item 5: GPU fallback/table-device audit (2026-08-05)
- Normal full-image reads return from the generic image fast path before the
  Python `file_cache` check; repeated `device="cuda"` reads therefore reuse
  only C++ metadata/raw-fd state, not decoded device tensors, and pay host
  decode plus H2D each time.
- `table.read_torch(device=...)` uses the thin C++ row reader on CPU and then
  `_move_table_dict` for one `.to(device)` per tensor column. The thin path
  intentionally ignores the legacy cache-capacity knobs; no device data cache
  exists. Non-tensor VLA/list values remain host-side by contract.
- A device data cache would require explicit device-aware keys, VRAM eviction,
  synchronization, and a CUDA benchmark. With no CUDA hardware on this host
  and no measured repeat-read workload, the audit produced no justified patch;
  the current host-first/H2D behavior remains documented in the GPU benchmark
  methodology.

### Wave-4 item 1: narrow-table polish disposition (2026-08-05)
- Fresh buffered `narrow_100000::read_full` repro measured torchfits at
  `1.465 ms` versus fitsio at `1.469 ms`; the specialized torchfits path was
  `1.389 ms`. The earlier 1.06–1.15x lag did not reproduce on this host, so no
  dispatch or table-reader change was justified.

### Wave-5: local benchmark sweep + uint16 mmap gap fix (2026-08-05)
- Full local sweep (fits_io 87 cases × mmap on/off, fitstable 108 cases,
  MegaCam 16 cases) found torchfits ≥ fitsio_torch in 171/178 and 104/108 and
  16/16; the only consistent same-host gap was scaled-16-bit uint16 with mmap
  enabled (`large_uint16_2d` 1.27–1.32x, `medium_uint16_2d` 1.25–1.58x vs the
  in-family tensor peer `fitsio_torch`).
- Methodology note: compressed-image deficits reported earlier in this log
  (0.70–0.91x) were an artifact of comparing torchfits against standalone
  fitsio; against the in-family tensor peer `fitsio_torch` (which pays the same
  torch-conversion cost) torchfits wins gzip 0.89–0.93x, rice 0.93x, hcompress
  0.97x. The compressed cluster is not a real deficit.
- Root cause of the uint16 gap: the mmap fast path applied the BZERO=32768
  unsigned offset with a scalar second pass over the whole buffer after the
  vectorized bswap copy. Folding the offset add into the SIMD loop
  (`paddw`/`vaddq_u16`, which wraps mod 2^16 exactly like the scalar cast)
  removed the second pass; the subset reader shares the same helper.
- Before/after (mmap=on, median of 3 interleaved runs, same host, same seed):
  `large_uint16_2d` 6.180→1.399 ms (tf/ft 1.32→0.57), `medium_uint16_2d`
  1.830→0.408 ms (1.25→0.53), `small_uint16_2d` 0.178→0.124 ms (1.20→0.63).
- Correctness: bitwise equality of the mmap-on path vs the CFITSIO path on
  full-range uint16 data, plus the full parity suite (137 tests incl. 20-seed
  fuzz) and `pixi run ci-local` green.
