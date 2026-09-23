# R5 slice B — `_table/write.py`, `_table/mutation.py`, `_table/_mutation_coerce.py`, `_table/utils.py`

Round R5 (slice `r5b`), baseline HEAD `1bb6958` + R1–R4. One-pass review+fix of the
table **write + mutation** slice against the rubric. Evidence: failing-first via
`pixi run pytest tests/<file> -q` before the fix, pasted, then the green run after.
Oracle `tests/test_output_parity.py` run and green (my changes touch write/mutation
contracts, not decode). Owned tests: `tests/test_table.py`, `tests/test_table_file_ops.py`,
`tests/test_bug_table_duplicate_names.py`, `tests/test_ascii_table.py`, new
`tests/test_mutation_errors.py`.

## Findings

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|----|-----|-------|---------------|--------------------------|------------------|--------|
| r5b-01 | BLOCKER | 1 silent-wrong-result / 5 silent-no-op | `_table/write.py:write` | `table.write(..., quantize=)` with **no qualifying (float) column** silently wrote an unpacked file instead of failing. Repro: `table.write(p, {"ID": np.int32([1,2,3])}, quantize="robust")` returned None and wrote native `TFORM=J`; the user asked for packing and nothing qualified. Both `quantize="robust"`/`True`/image-style (blanket over non-float table) and an all-opt-out column map (`{"ID": None}`) hit the silent path via `_prepare_quantized_table_data_for_write` returning `quantized=False`. | Raise typed `QuantizeError` (matches `write_api`'s typed pattern — that file NOT edited) when `quantize is not None and quantize is not False and not quantized`. Test `tests/test_table.py::test_table_write_quantize_no_qualifying_column_raises`. BEFORE: `Failed: DID NOT RAISE QuantizeError`. AFTER: passes. Also pinned: `parse_table_quantize_spec("robust", ["ID","FLUX"])` still selects FLUX and skips ID (unchanged; `test_quantize_int16.py::test_table_quantize_skips_integer_columns_on_blanket_robust` green). | fixed |
| r5b-02 | MAJOR | 5 error-contract | `_table/mutation.py:update_rows`; `_mutation_coerce.py:_normalize_mutation_rows` (used by `append_rows`/`insert_rows`) | Unknown-column errors on `update_rows`/`append_rows`/`insert_rows` were **`ValueError`**, not dict-like **`KeyError`** (`replace_column`/`rename_columns`/`drop_columns` already raised `KeyError`). Repro: `update_rows(p, {"ZZZ": …})` → `ValueError: Unknown columns for table mutation: extra=['ZZZ']`. Inconsistent error type across the mutation API. | Standardize on `KeyError` naming the column(s) everywhere. Changed the two `ValueError` sites to `KeyError`. Test `tests/test_mutation_errors.py::test_unknown_column_raises_keyerror_naming_column` (parametrized over all 6 ops, `match="ZZZ"`). BEFORE: 3 FAILED (`ValueError: … extra=['ZZZ']`) for update/append/insert. AFTER: all 6 pass. Updated the one incidental pin `tests/test_table_file_ops.py::test_table_append_update_rename_drop` `pytest.raises(ValueError)`→`KeyError` (contract change). | fixed |
| r5b-03 | MAJOR | 5 error-contract | `_table/mutation.py:update_rows` (mmap fallback) | `update_rows(mmap="auto")` wrapped `cpp.update_fits_table_rows_mmap` in `except Exception` and silently fell back to the non-mmap writer on **any** error, masking genuine non-decode (e.g. IO) errors. Repro: monkeypatch `update_fits_table_rows_mmap`→`raise OSError("No space left on device")`; `update_rows(..., mmap="auto")` returned success via fallback — the IO error vanished. Also `"truncat" in str(exc)` is `str(e)` matching. | Narrow the broad `except Exception` → **`except RuntimeError`** (the C++ `std::runtime_error` decode/layout type). Non-`RuntimeError` (OSError/ValueError/TypeError — non-decode) now propagate (re-raised). Truncation re-raise preserved. Test `tests/test_mutation_errors.py::test_update_rows_mmap_auto_reraises_non_decode_error` (BEFORE: `DID NOT RAISE OSError`; AFTER: passes) + `…_still_falls_back_on_decode_error` (fallback to non-mmap still works) + existing `tests/test_table_file_ops.py::test_update_rows_mmap_forced_failure_not_swallowed` unchanged/green (a `RuntimeError` mock still falls back in auto; forced still re-raises). Residual: genuine C++ `RuntimeError`s (e.g. "Invalid start row", "Failed to open file for mmap") still fall back and surface via the fallback writer — typed separation needs C++ typed exceptions (see r5b-08). | fixed |
| r5b-04 | BLOCKER | 1 silent data loss / 5 silent-no-op | `_table/mutation.py:insert_column,replace_column,insert_rows,delete_rows` (×4 `int(header_map.get("NAXIS2",0))` sites) | A malformed `NAXIS2` was coerced to `0` rows by `except Exception: … = 0`, turning row mutations into **silent no-ops / wrong results**. Repro (monkeypatch `cpp.read_header`→`NAXIS2="garbage"`): `delete_rows(p, slice(0,2))` returned without deleting anything (silent data loss); `insert_rows(row=0)` proceeded against a phantom zero-row table. | New helper `_naxis2_row_count` (`_table/utils.py`) narrows `except Exception`→`except (TypeError, ValueError)` and **re-raises** as a clear `ValueError` naming the corrupt value instead of defaulting to 0; a genuinely **missing** NAXIS2 still means 0 rows. Wired into all 4 sites. Test `tests/test_mutation_errors.py::test_malformed_naxis2_raises_not_silent_noop` (parametrized delete/insert_rows/insert_column/replace_column). BEFORE: `DID NOT RAISE` for delete_rows & insert_rows. AFTER: all raise. Valid/empty tables unchanged (existing mutation suites green). | fixed |

## Candidates re-derived (no fix landed)

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|----|-----|-------|---------------|--------------------------|------------------|--------|
| r5b-05 | MINOR | 4 perf (candidate — REFUTED) | `_mutation_coerce.py:_normalize_mutation_rows` | Candidate "preprocesses ALL columns → process only requested". Investigated with a count proof + CFITSIO probe: the deferred-default fill for **omitted** columns is **load-bearing**, not waste. Deterministic count: partial `append_rows` (1 of 5 cols) triggers exactly 4 `_default_table_column_values` fills (the 4 omitted cols). CFITSIO probe: a raw `cpp.append_fits_table_rows` with only `ID` fills the omitted `QUAL(TNULL=-999)` column with **0, not -999** — the Python default is what supplies TNULL. "Process only requested" would change `QUAL` from `[7,8,-999]` to `[7,8,0]` and break `tests/test_table_file_ops.py::test_append_rows_partial_payload_respects_tnull`. | No zero-behavior perf fix exists here (the work is required by write semantics). Not landed. Recorded for the record — the candidate was premised on the defaults being removable, which the TNULL/string/VLA partial-append tests refute (`test_append_rows_partial_payload_respects_tnull`, `test_append_rows_partial_payload_string_vla_defaults`). | deferred |
| r5b-06 | MINOR | 1 type-fallthrough (candidate) | `_table/write.py` / `write_api`+C++ inference | Candidate "write-path type fallthroughs (ASCII widths, integer widths)". Compared to astropy ground truth: ASCII string width truncates to the field width — matches astropy (`test_write_fidelity.py::test_ascii_string_width_handling_matches_astropy`); integer widths `int8→B` (negative wraps to uint8) **matches astropy exactly** (`astropy format="B"` also yields `[-1]→255`); `int16/32/64→I/J/K`, `uint16/32→I/J` pseudo-unsigned, `uint64`→rejected loudly (`UInt64WriteError`). No silent deviation from ground truth in the named width categories. Adjacent (recorded, out of named scope / not silent): `float16` is rejected by `table.write` with a misleading message while astropy upconverts to `E`; `2D-uint8` inferred `2B` (numeric) by `table.write` vs `2A` (string) by `_infer_fits_format` (mutation) — ambiguous input; `_infer_fits_scalar_code` (mutation format inference) rejects `int8` (TypeError) while `table.write` writes `int8→B` (inference inconsistency, but reject-`int8` is arguably safer than silent wrap). | No silent-wrong-result vs astropy in the named integer/ASCII widths → no fix landed. The float16-reject + 2D-uint8 + int8-inference divergences are design decisions (upcast-to-float32/E vs reject; string-vs-numeric interpretation; wrap-like-astropy vs reject-safer) and live partly in `write_api`/C++ (outside this slice). Backlogged. | deferred |
| r5b-07 | CLEANUP | 3 hygiene | `_table/utils.py:91 _normalize_row_slice` vs `_hdu/table_hdu_ref.py:158 _normalize_row_slice` | `_normalize_row_slice` is duplicated ×2, but the copies do **not** both live in this slice's files: one is the module function in `_table/utils.py` (mine), the other is a bound method on `TableHDURef` in `_hdu/table_hdu_ref.py` (R6's file). They are semantic duplicates (both normalize `row_slice` → `(start_1indexed, num_rows)`), not byte-identical. | Cross-slice → **not deduped here** (cannot edit `_hdu/table_hdu_ref.py`). Recorded for the owning slice (R6): the duplicate lives at `_hdu/table_hdu_ref.py:158` (`TableHDURef._normalize_row_slice`); dedupe by having the method delegate to `_table.utils._normalize_row_slice`. | deferred |
| r5b-08 | MINOR | 5 error-contract (hygiene) | `_table/write.py:_rewrite_table_hdu_with_schema` (`os.unlink(tmp_path)` `except Exception: pass`); `_table/mutation.py:_warn_numeric_coercion` (`_parse_tform` `except Exception: return`) | Remaining broad catches flagged by rubric class 5 ("no `except Exception: pass` on IO" / narrow broad excepts). Both are **zero-observable-behavior** cleanup/warning-helper paths (a temp-file unlink in `finally`; a best-effort coercion-warning helper). `write.py:_resolve_table_hdu_index_and_columns` `handle.close()` already uses `except Exception as exc: _log.warning(...)` (logs, not `pass`). | Not landed: these are behavior-preserving hygiene narrowings with **no observable wrong behavior**, so no valid failing-before regression test exists (a test would have to monkeypatch `os.unlink`/`parse_tform` to raise a non-`OSError`, which tests implementation, not behavior — against repo test policy). Suggested zero-behavior narrowing for a later pass: `os.unlink` cleanup `except Exception`→`except OSError`; `_warn_numeric_coercion` `_parse_tform` `except Exception`→`except (ValueError, TypeError)`. | deferred |

### Owed doc updates (land at R15)
- `docs/api-tables.md` (table write + mutation): document the new `QuantizeError` contract
  (`table.write(quantize=)` with no qualifying column raises) and the `KeyError` contract for
  unknown mutation columns (dict-like semantics) — the error-type change from `ValueError`.
  Inline `write()` docstring already notes the `QuantizeError`; the external page is R15's.

## Failing-first evidence (pasted)

`pixi run pytest tests/test_mutation_errors.py tests/test_table.py::test_table_write_quantize_no_qualifying_column_raises -q`
against **unfixed** code — **7 failed, 6 passed**:

```
FAILED tests/test_mutation_errors.py::test_unknown_column_raises_keyerror_naming_column[update_rows] - ValueError: Unknown columns for table mutation: extra=['ZZZ']
FAILED tests/test_mutation_errors.py::test_unknown_column_raises_keyerror_naming_column[append_rows] - ValueError: Unknown columns for table mutation: extra=['ZZZ']
FAILED tests/test_mutation_errors.py::test_unknown_column_raises_keyerror_naming_column[insert_rows] - ValueError: Unknown columns for table mutation: extra=['ZZZ']
FAILED tests/test_mutation_errors.py::test_update_rows_mmap_auto_reraises_non_decode_error - Failed: DID NOT RAISE OSError
FAILED tests/test_mutation_errors.py::test_malformed_naxis2_raises_not_silent_noop[delete_rows] - Failed: DID NOT RAISE any of (ValueError, TypeError)
FAILED tests/test_mutation_errors.py::test_malformed_naxis2_raises_not_silent_noop[insert_rows] - Failed: DID NOT RAISE any of (ValueError, TypeError)
FAILED tests/test_table.py::test_table_write_quantize_no_qualifying_column_raises - Failed: DID NOT RAISE QuantizeError
```

After the fixes — owned + consumer + **oracle** green (`tests/test_mutation_errors.py`,
`test_table.py`, `test_table_file_ops.py`, `test_bug_table_duplicate_names.py`,
`test_ascii_table.py`, `test_output_parity.py`, `test_quantize_int16.py`,
`test_write_fidelity.py`, `test_hdu_table_contracts.py`, `test_io_invariants.py`,
`test_pathlike_acceptance.py`, `test_fitsio_upstream_smoke.py`,
`test_astropy_upstream_smoke.py`):

```
313 passed, 3 warnings in 9.87s
```

Oracle `tests/test_output_parity.py` green in the batch (my changes affect write/mutation
error contracts and a raise-only quantize guard, not decode — no parity surface changed).

## Per-file disposition

| file | depth | finding IDs | status |
|------|-------|-------------|--------|
| `src/torchfits/_table/write.py` | deep | r5b-01, r5b-06, r5b-08 | r5b-01 fixed; r5b-06/r5b-08 deferred |
| `src/torchfits/_table/mutation.py` | deep | r5b-02, r5b-03, r5b-04, r5b-08 | r5b-02/03/04 fixed; r5b-08 deferred |
| `src/torchfits/_table/_mutation_coerce.py` | deep | r5b-02, r5b-05 | r5b-02 fixed; r5b-05 deferred (refuted) |
| `src/torchfits/_table/utils.py` | deep | r5b-04, r5b-07 | r5b-04 fixed; r5b-07 deferred (cross-slice, R6 owns dup) |
