# TorchFits Functionality Parity Checklist

Scope: close the remaining FITS functionality gaps (especially vs fitsio-style table workflows) while preserving torch-first and no astropy/fitsio runtime dependency in `torchfits`.

## Current baseline (already in-tree)

- Table read paths: projected columns, row slices, row index selection (`rows=[...]`), chunked streaming, Arrow reader/dataset/scanner adapters.
- Table write paths: binary/ascii table writing via CFITSIO, rich column support (numeric, string, complex, VLA), in-place `append_rows`, `update_rows`, `rename_columns`, `drop_columns`.
- File integrity: checksum write/verify APIs and tests.
- Cache coherency: path cache invalidation after in-place table mutations and external overwrite regression test coverage.
- Compressed image/table file mutations for mixed HDU sets: `insert_hdu(..., compress=...)`, `replace_hdu(..., compress=...)`, `delete_hdu(..., compress=...)`.
- Compressed write supports image tensors, table payloads, and mixed HDU/list/tuple payloads.

## P0 (release-blocking parity items)

### P0-1: In-place row insertion/deletion for table HDUs ✅

Why: fitsio users expect structural row edits, not only append/update.

Deliver:
- Add `torchfits.table.insert_rows(...)`.
- Add `torchfits.table.delete_rows(...)`.
- Add matching C++ bindings in `src/torchfits/cpp_src/table.cpp` and `src/torchfits/cpp_src/bindings.cpp`.
- Invalidate table/file caches before and after mutation (same contract as append/update).

Acceptance tests:
- `tests/test_table_file_ops.py::test_table_insert_rows_mid_table_preserves_order`
- `tests/test_table_file_ops.py::test_table_delete_rows_slice_and_single`
- `tests/test_table_file_ops.py::test_table_insert_delete_with_vla_and_string_columns`

### P0-2: Partial append payloads with deterministic defaults ✅

Why: current `append_rows` requires all columns; this is stricter than typical FITS workflows.

Deliver:
- Allow appending only a subset of columns.
- Fill omitted columns deterministically:
  - numeric/logical: FITS null sentinel when defined, otherwise zero/False.
  - fixed-width string: blank-padded empty string.
  - VLA: empty vector.
- Keep strict validation for unknown/extra columns.

Acceptance tests:
- `tests/test_table_file_ops.py::test_append_rows_partial_payload_numeric_defaults`
- `tests/test_table_file_ops.py::test_append_rows_partial_payload_string_vla_defaults`
- `tests/test_table_file_ops.py::test_append_rows_partial_payload_respects_tnull`

### P0-3: Open-handle table mutation ergonomics (`open()` path) ✅

Why: users using `with torchfits.open(...)` need explicit, safe file-backed mutation methods.

Deliver:
- Add file-backed mutators on `TableHDURef`:
  - `append_rows_file(...)`
  - `update_rows_file(...)`
  - `rename_columns_file(...)`
  - `drop_columns_file(...)`
  - and row ops from P0-1.
- Methods must mutate underlying FITS file and return refreshed `TableHDURef`.

Acceptance tests:
- `tests/test_hdu_file_ops.py::test_tablehduref_file_mutators_roundtrip`
- `tests/test_hdu_file_ops.py::test_tablehduref_mutation_refreshes_schema_and_rowcount`

## P1 (post-parity, high-value)

### P1-1: CFITSIO predicate pushdown for row filtering ✅

Why: current Arrow scanner filters after read; pushdown reduces I/O for large tables.

Deliver:
- Add optional `where=` expression path for table scans/reads.
- Apply row selection at CFITSIO level before materializing arrays.

Acceptance tests:
- `tests/test_arrow_table_api.py::test_scan_where_matches_python_filter`
- `tests/test_arrow_table_api.py::test_scan_where_with_projection`

### P1-2: Column-default-aware insert/replace semantics ✅

Why: parity with mature FITS table editing flows where schema evolution happens in-place.

Deliver:
- Add column insertion with format metadata (`TFORMn`/`TUNITn`/`TDIMn`/`TNULLn`).
- Add safe replace-column helper preserving row count and metadata validation.

Acceptance tests:
- `tests/test_table_file_ops.py::test_insert_column_with_explicit_format_metadata`
- `tests/test_table_file_ops.py::test_replace_column_preserves_metadata_contract`

### P1-3: Table API consistency and docs hardening ✅

Why: avoid user confusion across `read/open/table.*` APIs.

Deliver:
- Align docs/examples to file-backed vs materialized semantics.
- Ensure one canonical recommendation for out-of-core table handling.

Acceptance checks:
- Update `README.md` and `API.md` examples to match final P0 APIs.
- Add docs smoke test that exercises documented table workflow end-to-end.

## Performance guardrails while closing functionality

- No astropy/fitsio imports in runtime package (`tests/test_no_external_fits_backends.py` remains mandatory).
- New functionality must not regress torch-first read benchmarks:
  - especially common image buckets (`BITPIX=16/32/64/-32/-64`, medium/large shapes),
  - and table read hot paths (`torchfits`, `torchfits_hot`, `torchfits_cpp_open_once`).
- Keep new mutations cache-safe (invalidate both Python and C++ handle/reader caches).

## Remaining functionality gaps (before perf-only focus)

1. `spectral.py` helper APIs still have explicit `NotImplementedError` branches for non-core unit conversions (not a FITS I/O parity blocker); decide whether to complete or keep out of release scope.
