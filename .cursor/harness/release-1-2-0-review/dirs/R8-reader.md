# R8 slice A — `cpp_src/table_reader.h` review + fix (prefix `r8a`)

Scope: `src/torchfits/cpp_src/table_reader.h` (exclusive). Tests owned:
`tests/test_strided_update_rows.py` (new), `tests/test_vla_edge_rows.py` (new),
`tests/test_truncated_table_errors.py` (extended). Failing-first evidence was
captured against the current build (no rebuild) before each fix; passing
evidence is pasted per row after the mutex-protocol rebuild.

## Register

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|----|-----|-------|---------------|--------------------------|------------------|--------|
| r8a-01 | BLOCKER | 3 crash/UB | `table_reader.h:update_rows_mmap` (0-dim DLPack) | A 0-d payload carries NULL DLPack strides; the write loop called `tensor.stride(0)` unconditionally → SIGSEGV. Repro: `torchfits._C.update_fits_table_rows_mmap(p, 1, {"A": np.array(42, dtype="<i4")}, 1, 1)` → process killed, rc −11 (captured in test run below). | Hoist `stride0/stride1` guarded by `ndim >= 1/2`; 0-dim = one element (same contract as the buffered `update_rows`, which already handles `ndim == 0`). Test `test_strided_update_rows.py::test_zero_dim_payload_single_row_update` (subprocess: a segfault would take pytest down with it). | fixed |
| r8a-02 | BLOCKER | 1 silent data loss | `table_reader.h:update_rows_mmap` validation loop | The `if (col->scaled) throw` guard was a no-op on a fresh reader (`ensure_column_scale` never ran — scale flags are lazily populated), so a scaled column took the raw-bit mmap write path and stored PHYSICAL payload values as RAW cells on the DEFAULT `mmap="auto"` path. Repro: column TSCAL=2/TZERO=1; `ttable.update_rows(p, {"S": int16 [3,5,7,9,11]}, row_slice=(0,5))` → reads back `[7., 11., 15., 19., 23.]` instead of `[3., 5., 7., 9., 11.]` (captured below). Buffered `fits_write_col` treats payloads as physical (CFITSIO inverse-scales) — the mmap path diverged silently. | Resolve `ensure_column_scale` before the guard → scaled columns raise "Scaled columns not supported for mmap updates" (matching the read-side contract); `mutation.py`'s `mmap="auto"` falls back to the buffered writer (exact physical round-trip) and forced `mmap=True` errors cleanly. Tests `test_scaled_column_update_roundtrips_physical_values`, `test_scaled_column_forced_mmap_update_raises`. | fixed |
| r8a-03 | BLOCKER | 1 silent data loss / 5 convention | `table_reader.h:read_vla_column_flat` | Scaled VLA columns decoded through CFITSIO auto-scaling INTO THE RAW INTEGER DESTINATION: TSCAL=0.5 raw `[1,3,5]` came back `[0,1,2]` (truncated physical values, silent precision loss) and a TNULL sentinel on a scaled VLA was scaled (`-999` → `-1998`) instead of NaN — violating `tnull-read-torch` ("Quantized TNULL is NaN in tensors"). The env-gated heap-pread path read RAW cells, so the two heap paths disagreed as well. Repro captured below (`[0, 1, 2]` and `[4, -1998, 8]`). | Read raw cells (reset CFITSIO auto-scaling with `fits_set_tscale(..., 1.0, 0.0)` before the per-row reads, mirroring the fixed-width path) and apply the shared `apply_scale_and_nulls` helper (extracted from the fixed-width post-process — one convention source): float64 physical values, TNULL→NaN after the map. Tests `test_vla_scaled_values_exact_float`, `test_vla_scaled_tnull_becomes_nan`. | fixed |
| r8a-04 | BLOCKER | 3 moved-from / 1 silent | `table_reader.h:read_columns:ordered`, `read_columns_mmap:ordered`, `read_columns_mmap_filtered:ordered` | Relocated backlog duplicate-TTYPE moved-from UB. Commit 6b25294 keyed the result map by index (name-keyed variant fixed), but the assembly loop still `std::move`s each map entry while the selection lists can contain the SAME index twice (a name requested twice — e.g. `["A","A"]`, or duplicate TTYPE names where every lookup lands on the first index). The second move hands a moved-from `ColumnData` (undefined tensor) to Python → `None` silently. Repro: `cpp.read_fits_table(p, 1, ["I32","I32"], False)["I32"] is None` (captured below; mmap/rows variants and `["I16","I32","I16"]` likewise). | Dedupe selected indices at all three selection sites (a repeated name resolves once). Python dicts collapse repeated names anyway; behavior for unique requests is unchanged. Test `test_duplicate_column_request_returns_data`. | fixed |
| r8a-05 | MAJOR | 5 error contract (A-07) | `table_reader.h:read_columns` (2 `std::cerr` sites) | Diagnostics printed to `std::cerr` immediately before throw while the exception carried only a bare message: "Invalid start row: 99, nrows: 5" and "Column not found: nope. Available columns: I32, I16, " hit the terminal (captured in the failing run) while `str(exc)` was `'Invalid start row'` / `'Column not found: nope'`. | Fold the details into the exception message (`std::runtime_error`, message inside — no stderr writes). Test `test_error_messages_include_details`. | fixed |
| r8a-06 | MAJOR | 5/7 silent wrong-shaped result | `table_reader.h:read_columns`, `read_columns_mmap`, `read_columns_mmap_filtered` | Empty tables returned `{}` BEFORE validating requested names — `read(empty, columns=["nope"])` silently returned `{}` while the same request on a non-empty table raised "Column not found" (repro captured below: DID NOT RAISE). The filtered gather additionally skipped unknown OUTPUT names silently for any row count. | Resolve names first (unknown → `runtime_error` naming the column + available columns for `read_columns`); filtered gather throws on unknown output names. Empty tables with valid names still return `{}` (pinned). Tests `test_empty_table_unknown_column_raises`, `test_empty_table_valid_columns_still_empty`. | fixed |
| r8a-07 | MAJOR | 1/3 duplicate TTYPE consistency | `table_reader.h:update_rows_mmap:column_map` | With duplicate TTYPE cards the update writer resolved a name to the LAST same-named column (`operator[]`) while every read path resolves the FIRST — an update "lands" on a column the corresponding read never returns (silent lost update), or (repro below) dies with `update_rows mmap dtype mismatch for I32` because the map resolved the I32 payload against the renamed I16 column. Repro: rename TTYPE2 to "I32", update `{"I32": int32 vals}`, read `["I32"]` back. | `column_map.emplace` (first match wins) — the same resolution as read-side name lookups. Test `test_duplicate_ttype_update_hits_read_visible_column`. True positional selection of second-and-later same-named columns needs an API-visible redesign → see deferrals. | fixed |
| r8a-08 | MINOR | 2 overflow/truncation | `table_reader.h:analyze_table` (TSTRING width fallbacks), `read_columns` (`requested_bytes`) | Header-derived `width_long` was `(int)`-cast unguarded (ASCII `col.width`, binary width fallback `col.repeat`) and `requested_bytes += col.width * col.repeat` multiplied in `int` (signed overflow UB) — same class as the verify-only repeat guard `:293-297`. No failing-first repro is reachable through the bindings: CFITSIO rejects absurd TFORMs at open first (probe: TFORM `A4294967300` / `0I` → clean `RuntimeError "…: Could not move to HDU"`; pinned as `test_repeat_zero_fixed_column_clean_error`), so the guards are defense-in-depth. | Range guards mirroring the repeat guard (`width_long < 0 ‖ > 0x7fffffffL` → typed error) + `long` math in `requested_bytes`. Code proof below; behavior unreachable, so zero-observable-change hardening. | fixed |
| r8a-09 | — | verify-only anchors | `table_reader.h:293-315`, `:1020/:1281/:1939` (pre-fix numbering), read call sites | TFORM repeat int32 guard present (both the generic `repeat_long > INT_MAX` throw and the string-repeat `0x7fffffffL` guard). `ensure_extent_within_file` called before every mmap: `read_columns_mmap`, `read_columns_mmap_filtered`, `update_rows_mmap` — truncated files raise "…is truncated…" (existing `tests/test_truncated_table_errors.py` + my two extensions green). GIL release around table reads lives in `table_bindings.cpp` (slice B — see `R8-TableOpsBind`'s findings). | No change. | fixed (verified) |
| r8a-10 | — | named candidate re-derived | `table_reader.h:update_rows_mmap` strides | **Strided-DLPack numeric miswrite (plan's BLOCKER candidate): already fixed at HEAD.** `git blame` attributes the stride-honoring element offset `idx = i*stride(0) + j*stride(1)` used by SHORT/INT/LONG/FLOAT/DOUBLE/COMPLEX (BYTE/LOGICAL/BIT/STRING used matching stride-aware offsets) to commits `944f83b`/`405be49`; probes at HEAD round-trip `t[::2]`, F-order slices and negative-stride views exactly (and r8a-01's crash proves strides are NOT normalized away at the cast). | No source change; contract pinned by `tests/test_strided_update_rows.py` (1D all-dtype strided, 2D element-stride variants `(4,1)`/`(1,4)`, negative stride, high-level torch view) — these fail against any flat-indexing regression. | fixed (verified at HEAD, pinned) |
| r8a-11 | — | 7 edge (re-derived) | `table_reader.h:read_vla_column(_flat)` | **VLA zero-length / repeat-0 rows: clean at HEAD.** Zero-length rows anywhere (first/middle/last/all) round-trip exactly on the per-row and the flat values+offsets representations; zero-row windows return empty; Q-descriptor and logical VLAs with empty rows work; heap-pread env path agrees. Fixed repeat-0 columns (`TFORM 0J`) are refused by CFITSIO at open with a clean `RuntimeError` (no garbage rows). | No source change; pinned by `test_vla_edge_rows.py::test_vla_zero_length_rows_roundtrip_exact`, `…all_rows_zero_length`, `…flat_offsets_with_zero_length`, `…zero_row_window_and_margins`, `…raw_tnull_sentinel_stays` (raw TNULL stays a sentinel per `tnull-read-torch`) and `test_repeat_zero_fixed_column_clean_error`. | fixed (verified at HEAD, pinned) |

## Failing-first captures (before the fixes)

`pixi run pytest tests/test_strided_update_rows.py tests/test_vla_edge_rows.py
tests/test_truncated_table_errors.py -q` against the pre-fix build:

```
FAILED tests/test_strided_update_rows.py::test_zero_dim_payload_single_row_update - AssertionError: 0-dim payload crashed: rc=-11
FAILED tests/test_strided_update_rows.py::test_scaled_column_update_roundtrips_physical_values - AssertionError: assert False
 +  where False = <function array_equal>(array([ 7., 11., 15., 19., 23.]), array([ 3.,  5.,  7.,  9., 11.]))
FAILED tests/test_strided_update_rows.py::test_scaled_column_forced_mmap_update_raises - Failed: DID NOT RAISE RuntimeError
FAILED tests/test_strided_update_rows.py::test_duplicate_ttype_update_hits_read_visible_column - RuntimeError: update_rows mmap dtype mismatch for I32
FAILED tests/test_vla_edge_rows.py::test_vla_scaled_values_exact_float - AssertionError: assert dtype('int32') == <class 'numpy.float64'>
 +  where dtype('int32') = array([0, 1, 2], dtype=int32).dtype
FAILED tests/test_vla_edge_rows.py::test_vla_scaled_tnull_becomes_nan - AssertionError: assert dtype('int32') == <class 'numpy.float64'>
 +  where dtype('int32') = array([    4, -1998,    8], dtype=int32).dtype
FAILED tests/test_vla_edge_rows.py::test_duplicate_column_request_returns_data - assert None is not none
FAILED tests/test_vla_edge_rows.py::test_error_messages_include_details - AssertionError: Regex pattern did not match.
  Expected regex: 'Invalid start row.*99'
  Actual message: 'Invalid start row'
----------------------------- Captured stderr call -----------------------------
Invalid start row: 99, nrows: 6
Column not found: nope. Available columns: I32, I16,
FAILED tests/test_vla_edge_rows.py::test_empty_table_unknown_column_raises - Failed: DID NOT RAISE RuntimeError
9 failed, 18 passed
```

(18 passed = the r8a-10 / r8a-11 pins already green at HEAD + the untouched
truncated-table suite.)

## Passing captures (after the fix + rebuild)

Rebuild protocol followed: `pixi run dev` + `pixi run -e test -- pip install -e . --no-build-isolation` (both pixi envs rebuilt clean — the rebuild is the compile proof for the diff above).

`pixi run pytest tests/test_strided_update_rows.py tests/test_vla_edge_rows.py tests/test_truncated_table_errors.py -q`:

```
...........................                                              [100%]
27 passed in 3.64s
```

(27 = the 9 previously failing fixes + 18 pins.)

`pixi run pytest tests/test_output_parity.py -q` (shared read-only oracle):

```
.....................................................................    [100%]
69 passed in 2.18s
```

Table mutation/write regression suites (`test_mutation_errors.py
test_table_file_ops.py test_write_read_identity.py test_writing.py
test_hdu_table_contracts.py test_arrow_table_api.py test_malformed_fits.py`):

```
182 passed, 3 skipped, 5 warnings in 3.43s
```

(warnings are the pre-existing `get_header` fast-path fallback notices in the
malformed-FITS tests, untouched by this change.)

## Code proof for r8a-08 (no feasible Python repro)

CFITSIO refuses the absurd TFORMs before `analyze_table` can see a
`width_long` that overflows `int` (probe evidence above), and the same
`fits_get_coltype` output feeds the existing repeat guard — so the truncation
is unreachable through both path- and handle-based opens today. The guards are
the same shape as the shipped repeat guard (`:293-297` verify-only anchor) and
the `requested_bytes` product is now computed in `long`; signed `int` overflow
there was UB by the standard even where the observable result was not yet
wrong. Sanitizer workflow covers the pushed commit.

## Deferred

- Positional access to 2nd-and-later same-named (duplicate TTYPE) columns: name-keyed selection can only ever resolve the first; selecting duplicates by column index is an API-visible redesign (needs a Python-level contract decision). `update_rows_mmap` now agrees with the reads (first match) so no update is silently lost. Evidence: `test_duplicate_ttype_update_hits_read_visible_column`, `test_duplicate_ttype_read_keeps_all_columns_valid`.
- Unsigned (TZERO-offset) columns on the mmap update path: payloads in the read-side dtype (uint16/uint32) are refused by the dtype check and fall back to the buffered inverse-scaling writer at the `mmap="auto"` level; accepting uint payloads (value − offset) in `update_rows_mmap` is an enhancement. Evidence: dtype-mismatch error is clean; no silent path exists.

## Owed doc notes (docs land at R15)

- Scaled VLA columns now decode to float64 physical values with TNULL→NaN (previously truncated into the raw integer dtype) — the fixed-width table convention (`tnull-read-torch`) now applies to VLA as well.

## Behavioral notes

- A column name repeated in one `column_names` request resolves once (was: a duplicated index produced a moved-from `None` entry). Python dict results are unchanged for unique requests.

## Per-file disposition

| File | Reviewed | Findings | Disposition |
|------|----------|----------|-------------|
| `src/torchfits/cpp_src/table_reader.h` | full 2948 lines (schema/analyze, buffered + mmap decode, mmap filter scan + gather, VLA heap/pread, row update, geometry/extent guards) | r8a-01…r8a-08 fixed; r8a-09 verified; r8a-10/r8a-11 re-derived verified at HEAD + pinned; 2 deferrals | fixed with failing-first evidence (r8a-08 code proof) |
