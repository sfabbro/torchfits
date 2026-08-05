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
