# Deep Review 1 — Faithfulness vs fitsio / astropy (verified findings)

Status: DRAFT — experiments verified in-source on lane 2.13, python 3.13,
commit ad527c1. Each finding below was reproduced with a standalone probe
(images BSCALE/BZERO, tables TSCAL/TZERO, CONTINUE cards, uint64, TNULL).

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
