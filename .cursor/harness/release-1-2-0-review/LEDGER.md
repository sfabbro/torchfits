# LEDGER — torchfits 1.2.0 directory-by-directory review

Baseline HEAD: `1bb6958d958c5ebdb9a201f95bc5a01e89fe7acb` (2026-02-…, post-R0 upstream sync: already up to date).

Completeness proof: `tracked-files.txt` (468 lines, `git ls-files` at R0). Every tracked file gets one row below (File | Round | Depth | Findings | Status).

| File | Round | Depth | Findings | Status |
|---|---|---|---|---|
| `src/torchfits/__init__.py` | R1-A | full | — | clean |
| `src/torchfits/io.py` | R1-B | full | r1b-05, r1b-06, r1b-09 | fixed (05,06); r1b-09 deferred |
| `src/torchfits/table.py` | R1-B | full | — | clean |
| `src/torchfits/hdu.py` | R1-B | full | — | clean |
| `src/torchfits/cache.py` | R1-A | full | r1a-01..r1a-04 | fixed |
| `src/torchfits/cpp.py` | R1-A | full | — | clean |
| `src/torchfits/_cpp.py` | R1-A | full | — | clean |
| `src/torchfits/interop.py` | R1-B | full | r1b-01, r1b-02, r1b-03, r1b-05 | fixed |
| `src/torchfits/where.py` | R1-C | full | r1c-03, r1c-04 | fixed |
| `src/torchfits/_where.py` | R1-C | full | r1c-01, r1c-02 | fixed |
| `src/torchfits/fits_schema.py` | R1-B | full | r1b-04, r1b-07 | fixed |
| `src/torchfits/header_parser.py` | R1-C | full | r1c-10 | fixed |
| `src/torchfits/http_util.py` | R1-C | full | r1c-06, r1c-08, r1c-13, r1c-14 | fixed (doc note r1c-14 → R15) |
| `src/torchfits/logging.py` | R1-A | full | — | clean |
| `src/torchfits/vos_uri.py` | R1-C | full | — | clean |
| `src/torchfits/_string_decode.py` | R1-C | full | r1c-05, r1c-13 | fixed (errors='ignore' parity → deferred) |
| `src/torchfits/_tensor_buffer.py` | R1-C | full | — | clean |
| `src/torchfits/_C.pyi` | R1-A | full (check-stub) | — | clean (zero stub drift) |
| `src/torchfits/_io_engine/paths.py` | R1-C family | full | r1c-12, r1c-14 | docstring fixed; r1c-12 → R10 |
| `src/torchfits/cli/common.py` | R1-C family | full | — | clean |
| `src/torchfits/cli/cmds_copy.py` | R1-C family | full | r1c-07 | fixed |
| `src/torchfits/data/remote.py` | R1-C family | full | r1c-09 | fixed |
| `src/torchfits/_io_engine/http_subset.py` | R1-C family | full | r1c-08 | fixed (typed 416 → subset fallback) |
| `src/torchfits/transforms/__init__.py` | R2-A | full | r2a-12 | reviewed; doc fixes → R15 |
| `src/torchfits/transforms/base.py` | R2-A | full | r2a-06, r2a-08, r2a-09, r2a-15, r2a-17 | fixed (r2a-17 deferred) |
| `src/torchfits/transforms/state.py` | R2-A | full | r2a-01, r2a-10, r2a-13, r2a-17 | fixed (r2a-17 deferred) |
| `src/torchfits/transforms/fits_meta.py` | R2-A | full | r2a-01..r2a-05, r2a-07, r2a-11, r2a-12, r2a-14, r2a-16, r2a-17 | fixed (r2a-12/16/17 deferred) |
| `src/torchfits/transforms/helpers.py` | R2-B | full | r2b-01, r2b-02, r2b-03, r2b-06, r2b-08, r2b-09, r2b-10, r2b-11 | fixed |
| `src/torchfits/transforms/clip.py` | R2-B | full | r2b-01, r2b-02, r2b-06, r2b-11 | fixed |
| `src/torchfits/transforms/normalize.py` | R2-B | full | r2b-01, r2b-04, r2b-06, r2b-08, r2b-09 | fixed |
| `src/torchfits/transforms/background.py` | R2-B | full | r2b-05, r2b-06, r2b-07, r2b-11 | fixed |
| `src/torchfits/transforms/mask.py` | R2-C | full | r2c-01, r2c-02, r2c-09, r2c-11 | fixed |
| `src/torchfits/transforms/rgb.py` | R2-C | full | r2c-04, r2c-05, r2c-06, r2c-07, r2c-08 | fixed |
| `src/torchfits/transforms/stretch.py` | R2-C | full | r2c-03, r2c-10 | fixed (r2c-10 via base.py cross-slice) |
| `src/torchfits/data/datasets.py` | R3-A | full (12 classes + helpers) | r3a-01..r3a-15, r3b-08 | fixed (r3a-16/17 deferred-recorded) |
| `src/torchfits/data/remote.py` | R3-B | full | r3b-01..r3b-05 | fixed |
| `src/torchfits/data/__init__.py` | R3-B | full | r3b-06, r3b-07, r3a-08 (shared helper) | fixed |
| `src/torchfits/_io_engine/_read_pipeline.py` | R4-A | deep | r4a-01, r4a-04, r4a-05, r4a-09 | fixed |
| `src/torchfits/_io_engine/_read_pipeline_fallback.py` | R4-A; r5a-11 | deep | r4a-05, r4a-06, r4a-11, r5a-11 | fixed (r5a-11 double-open 2→1 via R5 authorization) |
| `src/torchfits/_io_engine/image.py` | R4-A | deep | r4a-01 | fixed |
| `src/torchfits/_io_engine/image_meta.py` | R4-A | deep | r4a-05, r4a-10, r4a-12, r4a-14 | fixed (r4a-12/14 deferred) |
| `src/torchfits/_io_engine/batch.py` | R4-A | full | r4a-01, r4a-05 | fixed |
| `src/torchfits/_io_engine/subset.py` | R4-A | deep | r4a-03, r4a-05, r4a-07, r4a-14 | fixed (r4a-14 deferred) |
| `src/torchfits/_io_engine/http_subset.py` | R4-A | deep | r4a-03, r4a-07, r4a-08 | fixed (R1 family lines untouched) |
| `src/torchfits/_io_engine/write_api.py` | R4-B | deep | r4b-01, r4b-02, r4b-03, r4b-04, r4b-06, r4b-07, r4b-08 | fixed |
| `src/torchfits/_io_engine/_write_helpers.py` | R4-B | deep | r4b-01..r4b-05, r4b-08, r4b-12, r4b-14, r4b-15 | fixed |
| `src/torchfits/_io_engine/_hdu_rewrite.py` | R4-B | deep | r4b-02, r4b-08, r4b-10, r4b-12, r4b-13, r4b-14, r4b-15 | fixed (r4b-13 root → R6; r4c-15 wrapper in mitigation) |
| `src/torchfits/_io_engine/quantize.py` | R4-B | deep | — | clean (`int16-robust-quantize` holds) |
| `src/torchfits/_io_engine/checksum_api.py` | R4-B | deep | r4b-09, r4b-11 | fixed |
| `src/torchfits/_io_engine/table_api.py` | R4-C | full | r4c-10, r4c-11 | fixed |
| `src/torchfits/_io_engine/table_reader_api.py` | R4-C | full | — | clean |
| `src/torchfits/_io_engine/table_streaming.py` | R4-C | full | r4c-07 | fixed |
| `src/torchfits/_io_engine/caches.py` | R4-C | full | r4c-01, r4c-02, r4c-03, r4c-09, r4c-12 | fixed |
| `src/torchfits/_io_engine/device.py` | R4-C | full | r4c-05, r4c-06 | fixed |
| `src/torchfits/_io_engine/options.py` | R4-C | full | r4c-04 | fixed |
| `src/torchfits/_io_engine/paths.py` | R4-C | full | r4c-14 (note) | clean (R1 family lines untouched) |
| `src/torchfits/_io_engine/hdu_api.py` | R4-C | full | r4c-08, r4c-09, r4c-12, r4c-15 | fixed |
| `src/torchfits/_io_engine/__init__.py` | R4-C | full | — | clean |
| `src/torchfits/_hdu/hdu_list.py` | R4-C (authorized close() block only) | block-level | r4c-01 | fixed — R6 reviews whole file; must reference r4c-01 (do not re-fix) and root-fix r4b-13 at `fromfile` |
| `src/torchfits/_table/read.py` | R5-A | deep | r5a-01, r5a-03, r5a-05, r5a-08, r5a-09, r5a-10, r5a-15 | fixed (`__all__` sealed, 8 names confirmed) |
| `src/torchfits/_table/_read_scan.py` | R5-A | deep | r5a-01, r5a-05, r5a-10 | fixed |
| `src/torchfits/_table/_read_schema.py` | R5-A | deep | r5a-02, r5a-03, r5a-04 | fixed |
| `src/torchfits/_table/_read_where.py` | R5-A | deep | r5a-07 | fixed |
| `src/torchfits/_table/engine.py` | R5-A | deep | r5a-06 (A-14) | fixed |
| `src/torchfits/_table_engine/__init__.py` | R5-A | surface | — | clean |
| `src/torchfits/_table_engine/backend_policy.py` | R5-A | surface | — | clean |
| `src/torchfits/_table_engine/read_policy.py` | R5-A | surface | r5a-13 (docs drift) | clean |
| `src/torchfits/_table/write.py` | R5-B | deep | r5b-01, r5b-06 | fixed (r5b-01 QuantizeError contract) |
| `src/torchfits/_table/mutation.py` | R5-B | deep | r5b-02, r5b-03, r5b-04, r5b-08 | fixed |
| `src/torchfits/_table/_mutation_coerce.py` | R5-B | deep | r5b-02, r5b-05 | fixed (r5b-05 candidate refuted with evidence) |
| `src/torchfits/_table/utils.py` | R5-B | deep | r5b-04, r5b-07 | fixed (dedupe r5b-07 → R6-B) |
| `src/torchfits/_table/interop.py` | R5-C | deep | r5c-01, r5c-02, r5c-04, r5c-05, r5c-06, r5c-08 | fixed |
| `src/torchfits/_table/arrow_convert.py` | R5-C | deep | r5c-03, r5c-07, r5c-14 | fixed |
| `src/torchfits/_table/cache.py` | R5-C | deep | r5c-10 | clean (live acquisition seam — residue candidate RE-DERIVED, zero-change) |
| `src/torchfits/_table/__init__.py` | R5-C | surface | r5c-12 (docs) | clean |
| `src/torchfits/_hdu/header.py` | R6-A | deep | r6a-01 (helper wiring), r6a-02 | fixed |
| `src/torchfits/_hdu/card.py` | R6-A | deep | r6a-01, r6a-06 (shared helper + `_is_string_typed`) | fixed |
| `src/torchfits/_hdu/hdu_list.py` | R6-A (whole-file; R4 rows above cover r4c-01 block) | deep | r6a-01, r6a-03, r6a-04, r6a-05 | fixed (r4c-01 block + r4b-13 root resolved) |
| `src/torchfits/_hdu/_repr.py` | R6-A | full | — | clean |
| `src/torchfits/_hdu/table_hdu.py` | R6-B | deep | r6b-06, r6b-07 | fixed |
| `src/torchfits/_hdu/table_hdu_ref.py` | R6-B | deep | r6b-01 (r5c-09), r6b-02 (r5c-15), r6b-05, r6b-13 | fixed |
| `src/torchfits/_hdu/tensor_hdu.py` | R6-B | deep | r6b-04, r6b-08 | fixed |
| `src/torchfits/_hdu/dataview.py` | R6-B | full (verify) | r6b-11 (A-08 verified fixed at HEAD; boundary pins) | clean |
| `src/torchfits/_io_engine/hdu_api.py` | R4-C; r6a-06 | full | r4c-08, r4c-09, r4c-12, r4c-15, r6a-06 | fixed (fusion half at integration; typing-convention residual deferred) |
| `src/torchfits/cli/main.py` | R7-A | full | r7a-01, r7a-02 | fixed |
| `src/torchfits/cli/common.py` | R7-A | full | r7a-03, r7a-07 | fixed |
| `src/torchfits/cli/__init__.py` | R7-A | surface | — | clean |
| `src/torchfits/cli/__main__.py` | R7-A | surface | — | clean |
| `src/torchfits/cli/cmds_header.py` | R7-A | full | — | clean |
| `src/torchfits/cli/cmds_info.py` | R7-A | full | — | clean |
| `src/torchfits/cli/cmds_probe.py` | R7-A | full | r7a-04, r7a-05 | fixed |
| `src/torchfits/cli/cmds_stats.py` | R7-B | deep | r7b-02, r7b-05, r7b-06 | fixed |
| `src/torchfits/cli/cmds_verify.py` | R7-B | deep | r7b-05, r7b-06, r7b-10 | fixed |
| `src/torchfits/cli/cmds_diff.py` | R7-B | deep | r7b-01, r7b-03, r7b-04 | fixed |
| `src/torchfits/cli/cmds_table.py` | R7-B | deep | r7b-05, r7b-07 | fixed |
| `src/torchfits/cli/cmds_copy.py` | R7-B | full (reference) | r7b-13 | clean (same-path refusal reference; r7b-13 deferred) |
| `src/torchfits/cli/cmds_cutout.py` | R7-B | deep | r7b-08 | fixed |
| `src/torchfits/cli/cmds_arith.py` | R7-C | deep | r7c-02, r7c-08, r7c-11, r7c-15 | fixed (r7c-15 deferred: in-place policy) |
| `src/torchfits/cli/cmds_compress.py` | R7-C | deep | r7c-08, r7c-13 | fixed (r7c-13 verified+pinned; r7c-16/17 deferred) |
| `src/torchfits/cli/cmds_convert.py` | R7-C | deep | r7c-01, r7c-09 | fixed |
| `src/torchfits/cli/cmds_setkey.py` | R7-C | deep | r7c-03..r7c-07, r7c-10 | fixed |
| `src/torchfits/cli/cmds_transform.py` | R7-C | deep | r7c-14 | clean (decision-5 verified+pinned) |
| `docs/cli.md` | R15 | full | r7a-06, r7a-10, r7b-09, r15a-04 | fixed |
| `src/torchfits/cpp_src/table_reader.h` | R8-A | deep (2948 ln) | r8a-01..r8a-08, r8a-10, r8a-11 | fixed (r8a-10/11 verified+pinned) |
| `src/torchfits/cpp_src/table_types.h` | R8-B | full | — | clean |
| `src/torchfits/cpp_src/table_ops.h` | R8-B | full | — | clean |
| `src/torchfits/cpp_src/table_ops.cpp` | R8-B | full (1094 ln) | r8b-01, r8b-04, r8b-07, r8b-09 | fixed (r8b-07 pinned; r8b-09 deferred) |
| `src/torchfits/cpp_src/table_bindings.cpp` | R8-B | full (511 ln) | r8b-02, r8b-03, r8b-05, r8b-06, r8b-08 | fixed (r8b-05/06 pinned; r8b-08 deferred) |
| `src/torchfits/cpp_src/fits_bindings.cpp` | R9-A | deep (2627 ln) | r9a-01, r9a-02, r9a-04, r9a-05, r9a-06, r9a-07, r9a-08, r9a-09, r9a-10..r9a-13 | fixed (r9a-03/08/09 re-derived+pinned; r9a-10/11 verify-only; r9a-12 pin) |
| `src/torchfits/cpp_src/fits_file.cpp` | R9-B (rescue) | full (1206 ln) | r9b-02, r9b-03, r9b-04, r9b-05 | fixed |
| `src/torchfits/cpp_src/fits_detail.h` | R9-B (rescue) | full (795 ln) | r9b-01, r9b-04 | fixed |
| `src/torchfits/cpp_src/fits_file.h` | R9-B | full (123 ln) | r9b-03, r9b-04 (internal state only) | fixed (no new public API) |
| `src/torchfits/cpp_src/fits_handle.h` | R9-B | full (50 ln) | — | clean |
| `src/torchfits/cpp_src/fits_rw.h` | R9-B | full (65 ln) | r9b-06 | clean (r9b-06 deferred: defense-in-depth, probe recorded) |
| `src/torchfits/cpp_src/internal_utils.h` | R10 | full (355 ln) | r10a-03, r10a-06 | fixed (byteswap tail UB; no-SIMD fallback verified) |
| `src/torchfits/cpp_src/security.h` | R10 | full (66 ln) | r10a-01 (r1c-12 parity decision), r10a-02 | fixed (complementary-layer split documented + covered on both layers) |
| `src/torchfits/cpp_src/hardware.h` | R10 | full (75 ln) | r10a-06 | clean |
| `src/torchfits/cpp_src/hardware.cpp` | R10 | full (61 ln) | r10a-06 | clean (fallback parity probe) |
| `src/torchfits/cpp_src/torch_compat.h` | R10 | full (111 ln) | r10a-07 | clean (dtype/ABI shims verified) |
| `src/torchfits/cpp_src/torchfits_torch.h` | R10 | full (38 ln) | r10a-07 | clean |
| `src/torchfits/cpp_src/bindings.cpp` | R10 | full (36 ln) | r10a-07 | clean (`cpp-seal-all` holds) |
| `src/torchfits/cpp_src/CMakeLists.txt` | R10 | full | r10a-02, r10a-04, r10a-05 | fixed (-O3 restored; sanitizer propagation; bracket test wired) |
| `tests/conftest.py` | R11 | full | — | clean |
| `tests/cpp/test_bracket_detection.cpp` | R11 | full | — | clean |
| `tests/test_api.py` | R11 | full | r11a-10 | deferred (r11a-10) |
| `tests/test_arrow_table_api.py` | R11 | full | — | clean |
| `tests/test_ascii_table.py` | R11 | full | r11a-01 | fixed (r11a-01) |
| `tests/test_astropy_upstream_smoke.py` | R11 | full | — | clean |
| `tests/test_bench_ranking_mmap.py` | R11 | full | — | clean |
| `tests/test_bench_suites.py` | R11 | full | r11a-03, r11a-04, r11a-07 | fixed (r11a-03, r11a-04); r11a-07 deferred |
| `tests/test_bug_table_duplicate_names.py` | R11 | full | — | clean |
| `tests/test_byteswap.py` | R11 | full | r11a-06 | fixed (r11a-06) |
| `tests/test_bz2.py` | R11 | full | — | clean |
| `tests/test_cache.py` | R11 | full | r11a-08 | deferred (r11a-08) |
| `tests/test_cache_config.py` | R11 | full | — | clean |
| `tests/test_changelog_tooling.py` | R11 | full | — | clean |
| `tests/test_check_torch_extra_pins.py` | R11 | full | — | clean |
| `tests/test_checksum.py` | R11 | full | — | clean |
| `tests/test_clear_all_caches.py` | R11 | full | r11a-09 | deferred (r11a-09) |
| `tests/test_cli.py` | R11 | full | — | clean |
| `tests/test_cli_arith_stats.py` | R11 | full | r11a-05 | fixed (r11a-05) |
| `tests/test_complex_header.py` | R11 | full | r11a-02 | fixed (r11a-02) |
| `tests/test_compressed_nulls.py` | R11 | full | r11a-06 | fixed (r11a-06) |
| `tests/test_compression.py` | R11 | full | r11a-03 | fixed (r11a-03) |
| `tests/test_compression_matrix.py` | R11 | full | — | clean |
| `tests/test_concurrent_same_file_read.py` | R11 | full | — | clean |
| `tests/test_cutout_performance_api.py` | R11 | full | — | clean |
| `tests/test_data.py` | R11 | full | r11b-02 | fixed (r11b-02) |
| `tests/test_data_datasets.py` | R11 | full | r11b-01 | fixed (r11b-01) |
| `tests/test_data_ml.py` | R11 | full | — | clean |
| `tests/test_differential_astropy.py` | R11 | full | — | clean |
| `tests/test_dlpack_roundtrip.py` | R11 | full | — | clean |
| `tests/test_docs_code_snippets.py` | R11 | full | — | clean |
| `tests/test_docs_integrity.py` | R11 | full | — | clean |
| `tests/test_examples_runner.py` | R11 | full | — | clean |
| `tests/test_fits_schema.py` | R11 | full | — | clean |
| `tests/test_fitsio_upstream_smoke.py` | R11 | full | — | clean |
| `tests/test_hdu.py` | R11 | full | — | clean |
| `tests/test_hdu_close_and_overflow.py` | R11 | full | — | clean |
| `tests/test_hdu_file_ops.py` | R11 | full | r11b-03 | fixed (r11b-03) |
| `tests/test_hdu_str.py` | R11 | full | — | clean |
| `tests/test_hdu_table_contracts.py` | R11 | full | — | clean |
| `tests/test_header_ascii_strictness.py` | R11 | full | — | clean |
| `tests/test_header_duplicate_keys.py` | R11 | full | — | clean |
| `tests/test_header_value_typing.py` | R11 | full | — | clean |
| `tests/test_header_versioning.py` | R11 | full | — | clean |
| `tests/test_http_probe_fixture.py` | R11 | full | — | clean |
| `tests/test_integration.py` | R11 | full | r11c-01, r11c-05 | fixed (r11c-01, r11c-05) |
| `tests/test_interop.py` | R11 | full | — | clean |
| `tests/test_interop_import.py` | R11 | full | — | clean |
| `tests/test_io.py` | R11 | full | — | clean |
| `tests/test_io_invariants.py` | R11 | full | — | clean |
| `tests/test_longstr_and_scaled_tables.py` | R11 | full | r11c-03, r11c-04 | fixed (r11c-03, r11c-04) |
| `tests/test_malformed_fits.py` | R11 | full | — | clean |
| `tests/test_mps.py` | R11 | full | — | clean |
| `tests/test_multichunk_buffered_read.py` | R11 | full | — | clean |
| `tests/test_native_stub.py` | R11 | full | — | clean |
| `tests/test_no_external_fits_backends.py` | R11 | full | — | clean |
| `tests/test_open_table_reader.py` | R11 | full | — | clean |
| `tests/test_output_parity.py` | R11 | full | — | clean |
| `tests/test_package_isolation.py` | R11 | full | — | clean |
| `tests/test_patch_bench_docs.py` | R11 | full | r11c-02 | fixed (r11c-02) |
| `tests/test_pathlike_acceptance.py` | R11 | full | — | clean |
| `tests/test_performance.py` | R11 | full | r11d-02 | fixed (r11d-02) |
| `tests/test_public_boundary.py` | R11 | full | — | clean |
| `tests/test_public_where.py` | R11 | full | — | clean |
| `tests/test_quantize_int16.py` | R11 | full | — | clean |
| `tests/test_read_header.py` | R11 | full | r11d-03 | fixed (r11d-03) |
| `tests/test_read_policy.py` | R11 | full | — | clean |
| `tests/test_release_lane.py` | R16 | full | r16a-01, r16a-02 | fixed |
| `tests/test_release_smoke.py` | R11 | full | — | clean |
| `tests/test_remote_http_range.py` | R11 | full | — | clean |
| `tests/test_rgb.py` | R11 | full | — | clean |
| `tests/test_scale_on_device.py` | R11 | full | — | clean |
| `tests/test_security.py` | R11 | full | r11d-02 | fixed (r11d-02) |
| `tests/test_shared_meta_staleness.py` | R11 | full | — | clean |
| `tests/test_skinny_meta.py` | R11 | full | — | clean |
| `tests/test_staged_prefetch.py` | R11 | full | — | clean |
| `tests/test_stream_table_and_cache.py` | R11 | full | — | clean |
| `tests/test_subset_3d.py` | R11 | full | — | clean |
| `tests/test_table.py` | R11 | full | r11d-02 | fixed (r11d-02) |
| `tests/test_table_docs_smoke.py` | R11 | full | — | clean |
| `tests/test_table_file_ops.py` | R11 | full | — | clean |
| `tests/test_table_filter_grammar.py` | R11 | full | — | clean |
| `pixi.toml` | R16 | full | r11d-01, r16a-02 | fixed |
| `.github/workflows/ci.yml` | R16 | full | r11d-01 | fixed (r11d-01); release-gate list matches pixi |
| `tests/test_table_filtering.py` | R11 | full | — | clean |
| `tests/test_table_head.py` | R11 | full | — | clean |
| `tests/test_table_squeeze_and_ref_cache.py` | R11 | full | — | clean |
| `tests/test_tablehdu_schema_cache.py` | R11 | full | — | clean |
| `tests/test_torch_boundary.py` | R11 | full | — | clean |
| `tests/test_transforms.py` | R11 | full | r11e-03 | fixed (r11e-03) |
| `tests/test_transforms_e2e.py` | R11 | full | — | clean |
| `tests/test_transforms_state.py` | R11 | full | r11e-03 | fixed (r11e-03) |
| `tests/test_transforms_typing.py` | R11 | full | — | clean |
| `tests/test_truncated_table_errors.py` | R11 | full | — | clean |
| `tests/test_upstream_parity_inventory.py` | R11 | full | — | clean |
| `tests/test_validation.py` | R11 | full | r11e-01 | fixed (r11e-01) |
| `tests/test_where.py` | R11 | full | — | clean |
| `tests/test_where_and_batch_errors.py` | R11 | full | — | clean |
| `tests/test_write_fidelity.py` | R11 | full | — | clean |
| `tests/test_write_read_identity.py` | R11 | full | — | clean |
| `tests/test_writing.py` | R11 | full | r11e-02 | fixed (r11e-02) |
| `tests/transforms_reference.py` | R11 | full | — | clean |
| `tests/transforms_reference.pyi` | R11 | full | — | clean |
| `tests/test_batch_error_contracts.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_arith_edges.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_convert_edges.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_diff_copy_cutout.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_exit_matrix.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_same_path_refusal.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_setkey_integrity.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_stats_verify.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_cli_transform_parallel.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_deprecated_knobs.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_fitsfile_edges.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_hdu_count_scan.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_hdulist_registry_race.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_http_guard.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_longstr_header_reassembly.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_maskrgb_masking.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_maskrgb_stretch.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_meta_cache_freshness.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_mutation_errors.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_native_infra.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_read_full_numpy_parity.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_read_pipeline_contracts.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_reader_cache_freshness.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_remote_resume.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_strided_update_rows.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_string_decode.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_subset_http_parity.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_table_read_contracts.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_views_dataview_index.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_vla_edge_rows.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_where_matrix.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `tests/test_where_semantics.py` | R11 | scan | — | post-chunk pin (not in the precomputed slices; no wall-clock/RSS assert; see R11.md) |
| `benchmarks/__init__.py` | R12 | full | — | clean |
| `benchmarks/bench_all.py` | R12 | full | — | clean |
| `benchmarks/bench_arrow_tables.py` | R12 | full | r12a-03 | fixed (r12a-03) |
| `benchmarks/bench_cache.py` | R12 | full | — | clean |
| `benchmarks/bench_contract.py` | R12 | full | — | clean |
| `benchmarks/bench_cpp_backend.py` | R12 | full | — | clean |
| `benchmarks/bench_denoise.py` | R12 | full | — | clean |
| `benchmarks/bench_fits_io.py` | R12 | full | r12a-01, r12a-02, r12a-05 | fixed (r12a-01, r12a-02); r12a-05 deferred |
| `benchmarks/bench_fits_write.py` | R12 | full | — | clean |
| `benchmarks/bench_fitstable_io.py` | R12 | full | — | clean |
| `benchmarks/bench_fixtures.py` | R12 | full | — | clean |
| `benchmarks/bench_gpu_memory.py` | R12 | full | — | clean |
| `benchmarks/bench_gpu_transports.py` | R12 | full | r12a-04, r12a-05 | fixed (r12a-04); r12a-05 deferred |
| `benchmarks/bench_http_stream.py` | R12 | full | — | clean |
| `benchmarks/bench_import_boundary.py` | R12 | full | — | clean |
| `benchmarks/bench_median_stack.py` | R12 | full | — | clean |
| `benchmarks/bench_megacam_cutouts.py` | R12 | full | — | clean |
| `benchmarks/bench_metadata.py` | R12 | full | — | clean |
| `benchmarks/bench_ml_loader.py` | R12 | full | — | clean |
| `benchmarks/bench_science_pipeline.py` | R12 | full | — | clean |
| `benchmarks/bench_table.py` | R12 | full | — | clean |
| `benchmarks/bench_timing.py` | R12 | full | — | clean |
| `benchmarks/cfitsio_direct/CMakeLists.txt` | R12 | full | — | clean |
| `benchmarks/cfitsio_direct/bench_cfitsio_direct.c` | R12 | full | — | clean |
| `benchmarks/config.py` | R12 | full | — | clean |
| `benchmarks/mpl_config.py` | R12 | full | — | clean |
| `benchmarks/replays/upstream_sources.json` | R12 | full | — | clean |
| `benchmarks/run_cfitsio_direct_bench.py` | R12 | full | — | clean |
| `benchmarks/suites.py` | R12 | full | — | clean |
| `scripts/aggregate_matrix_bench.py` | R13 | scan | — | clean (scan) |
| `scripts/bench_cfitsio_direct.sh` | R13 | scan | — | clean (scan) |
| `scripts/bench_deficit_focus.sh` | R13 | scan | — | clean (scan) |
| `scripts/bench_exhaustive_local.sh` | R13 | scan | — | clean (scan) |
| `scripts/bench_release_scorecard.sh` | R13 | scan | — | clean (scan) |
| `scripts/bench_suite.sh` | R13 | scan | — | clean (scan) |
| `scripts/build_docs_pages.sh` | R13 | full | — | clean |
| `scripts/build_wheels_local.sh` | R13 | scan | — | clean (scan) |
| `scripts/canfar_denoise_incontainer.sh` | R13 | scan | — | clean (scan) |
| `scripts/canfar_gpu_bench_incontainer.sh` | R13 | scan | — | clean (scan) |
| `scripts/canfar_matrix_bench_incontainer.sh` | R13 | scan | — | clean (scan) |
| `scripts/check_docs_links.py` | R13 | scan | — | clean (scan) |
| `scripts/check_duplicate_cpp.py` | R13 | scan | — | clean (scan) |
| `scripts/check_torch_extra_pins.py` | R13 | scan | — | clean (scan) |
| `scripts/check_wheel_contents.py` | R13 | full | — | clean |
| `scripts/ci_local.sh` | R13 | full | — | clean |
| `scripts/cibuildwheel.sh` | R13 | scan | — | clean (scan) |
| `scripts/cibw_before_build.sh` | R13 | scan | — | clean (scan) |
| `scripts/cibw_test.sh` | R13 | scan | — | clean (scan) |
| `scripts/clean_install_smoke.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_canfar_bench_vos.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_cfht_calib_frames.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_cfht_megacam_sample.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_cfht_megapipe_sample.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_example_samples.sh` | R13 | scan | — | clean (scan) |
| `scripts/fetch_rgb_sky_samples.sh` | R13 | scan | — | clean (scan) |
| `scripts/gen_native_stub.py` | R13 | full | — | clean |
| `scripts/gpu-bootstrap.sh` | R13 | scan | — | clean (scan) |
| `scripts/gpu-env-loader.sh` | R13 | full | — | clean |
| `scripts/import_canfar_bench_artifacts.py` | R13 | scan | — | clean (scan) |
| `scripts/launch_canfar_denoise.sh` | R13 | scan | — | clean (scan) |
| `scripts/launch_canfar_gpu_bench.sh` | R13 | scan | — | clean (scan) |
| `scripts/launch_canfar_matrix_grid.sh` | R13 | scan | — | clean (scan) |
| `scripts/patch_bench_docs.py` | R13 | scan | — | clean (scan) |
| `scripts/patch_canfar_exhaustive_docs.sh` | R13 | scan | — | clean (scan) |
| `scripts/publish_canfar_bench_vos.sh` | R13 | scan | — | clean (scan) |
| `scripts/publish_canfar_wheel_bundle.sh` | R13 | scan | — | clean (scan) |
| `scripts/release_lane.py` | R16 | full | r16a-01 | fixed (r16a-01) |
| `scripts/release_notes.py` | R13 | scan | — | clean (scan) |
| `scripts/render_bench_deficits.py` | R13 | full | — | clean |
| `scripts/render_bench_highlights.py` | R13 | full | r13a-01 | fixed (r13a-01) |
| `scripts/render_bench_iopath_table.py` | R13 | full | — | clean |
| `scripts/render_bench_ml.py` | R13 | scan | — | clean (scan) |
| `scripts/render_bench_quick.py` | R13 | scan | — | clean (scan) |
| `scripts/render_full_benchmarks_table.py` | R13 | full | r13a-02 | fixed (r13a-02) |
| `scripts/run_exhaustive_bench_and_patch_docs.sh` | R13 | scan | — | clean (scan) |
| `scripts/selfcheck_canfar_launcher.sh` | R13 | scan | — | clean (scan) |
| `scripts/sync_docs_examples.sh` | R13 | full | — | clean |
| `scripts/torch_lanes.json` | R13 | full | — | clean |
| `scripts/update_changelog.py` | R13 | full | — | clean |
| `scripts/verify_wheel_cuda_canfar.sh` | R13 | scan | — | clean (scan) |
| `scripts/verify_wheel_cuda_canfar_incontainer.sh` | R13 | scan | — | clean (scan) |
| `scripts/verify_wheel_matrix.sh` | R13 | scan | — | clean (scan) |
| `examples/_plotting.py` | R14 | full | — | clean |
| `examples/_sample_data.py` | R14 | full | — | clean |
| `examples/cli/imstat_imarith.sh` | R14 | full | — | clean |
| `examples/cli/make_rgb_demo.py` | R14 | full | r14a-01 | fixed (r14a-01) |
| `examples/desi_shaped_spectrum.py` | R14 | full | r14a-04 | fixed (r14a-04) |
| `examples/example_ccfits_cookbook.py` | R14 | full | r14a-05 | fixed (r14a-05) |
| `examples/example_cfitsio_cookbook.py` | R14 | full | r14a-02 | fixed (r14a-02) |
| `examples/example_custom_transform.py` | R14 | scan | — | clean |
| `examples/example_cutout_wcs_write.py` | R14 | full | — | clean |
| `examples/example_data_catalogs.py` | R14 | full | — | clean |
| `examples/example_identity_stress.py` | R14 | full | r14a-03 | fixed (r14a-03) |
| `examples/example_image.py` | R14 | full | — | clean |
| `examples/example_image_cube.py` | R14 | full | — | clean |
| `examples/example_image_cutouts.py` | R14 | scan | — | clean |
| `examples/example_image_dataset.py` | R14 | scan | — | clean |
| `examples/example_image_mef.py` | R14 | scan | — | clean |
| `examples/example_lupton_rgb_sdss.py` | R14 | full | — | clean |
| `examples/example_m13_stack.py` | R14 | scan | — | clean |
| `examples/example_make_loader_vs_dataloader.py` | R14 | scan | — | clean |
| `examples/example_manga_logcube.py` | R14 | full | — | clean |
| `examples/example_mef_header.py` | R14 | scan | — | clean |
| `examples/example_megacam_cr_denoise.py` | R14 | full | — | clean |
| `examples/example_megacam_mef_cutouts.py` | R14 | scan | — | clean |
| `examples/example_megapipe_cutout_collage.py` | R14 | full | — | clean |
| `examples/example_ml_galaxyzoo_legacy.py` | R14 | full | — | clean |
| `examples/example_ml_training_loop.py` | R14 | full | r14a-06 | fixed (r14a-06) |
| `examples/example_polars.py` | R14 | scan | — | clean |
| `examples/example_quantize_int16.py` | R14 | full | — | clean |
| `examples/example_rgb_sky.py` | R14 | full | r14a-01 | fixed (r14a-01) |
| `examples/example_staged_cutouts.py` | R14 | scan | — | clean |
| `examples/example_streaming_cubes_spectra.py` | R14 | scan | — | clean |
| `examples/example_table.py` | R14 | full | — | clean |
| `examples/example_table_interop.py` | R14 | scan | — | clean |
| `examples/example_table_recipes.py` | R14 | scan | — | clean |
| `examples/example_time_series.py` | R14 | scan | — | clean |
| `examples/example_transforms.py` | R14 | scan | — | clean |
| `examples/gallery_images.py` | R14 | scan | — | clean |
| `examples/gallery_tables_lc.py` | R14 | scan | — | clean |
| `examples/test_examples.py` | R14 | full | r14a-04 | fixed (r14a-04) |
| `docs/api-core-io.md` | R15 | scan | — | clean |
| `docs/api-data.md` | R15 | scan | — | clean |
| `docs/api-tables.md` | R15 | full | r15a-03, r15a-06 | fixed |
| `docs/api-transforms.md` | R15 | scan | — | clean |
| `docs/api.md` | R15 | full | r15a-07 | fixed |
| `docs/architecture.md` | R15 | full | r15a-02 | fixed |
| `docs/assets/bench/20260719_075555/megacam_results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/summary.md` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/torchfits_deficits.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/summary.md` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/torchfits_deficits.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/summary.md` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/torchfits_deficits.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/summary.md` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/torchfits_deficits.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_mps_20260719_143706/megacam_results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_mps_20260719_143706/ml_results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_mps_20260719_143706/results.csv` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_mps_20260719_143706/summary.md` | R15 | scan | — | clean |
| `docs/assets/bench/exhaustive_mps_20260719_143706/torchfits_deficits.csv` | R15 | scan | — | clean |
| `docs/assets/bench/ml_20260719_145743/ml_results.csv` | R15 | scan | — | clean |
| `docs/assets/gallery/cli_rgb_demo.png` | R15 | scan | — | clean |
| `docs/assets/gallery/image_compose_pipeline.png` | R15 | scan | — | clean |
| `docs/assets/gallery/image_cutout.png` | R15 | scan | — | clean |
| `docs/assets/gallery/lightcurve_asymmetric_sigma_clip.png` | R15 | scan | — | clean |
| `docs/assets/gallery/lightcurve_sigma_clip.png` | R15 | scan | — | clean |
| `docs/assets/gallery/lupton_rgb_sdss.png` | R15 | scan | — | clean |
| `docs/assets/gallery/megapipe_cutout_collage.png` | R15 | scan | — | clean |
| `docs/assets/gallery/ml_gz_class_grid.png` | R15 | scan | — | clean |
| `docs/assets/gallery/rgb_sky_collage.png` | R15 | scan | — | clean |
| `docs/assets/gallery/rgb_vs_lupton_dwarf.png` | R15 | scan | — | clean |
| `docs/assets/gallery/table_fits_scale_columns.png` | R15 | scan | — | clean |
| `docs/assets/katex/contrib/auto-render.min.js` | R15 | scan | — | clean |
| `docs/assets/katex/fonts/KaTeX_AMS-Regular.woff2` | R15 | scan | — | clean |
| `docs/assets/katex/fonts/KaTeX_Main-Regular.woff2` | R15 | scan | — | clean |
| `docs/assets/katex/fonts/KaTeX_Math-Italic.woff2` | R15 | scan | — | clean |
| `docs/assets/katex/fonts/KaTeX_Size1-Regular.woff2` | R15 | scan | — | clean |
| `docs/assets/katex/fonts/KaTeX_Size2-Regular.woff2` | R15 | scan | — | clean |
| `docs/assets/katex/katex.min.css` | R15 | scan | — | clean |
| `docs/assets/katex/katex.min.js` | R15 | scan | — | clean |
| `docs/benchmarks.md` | R15 | scan | — | clean |
| `docs/changelog.md` | R15 | scan | — | clean |
| `docs/cli-recipes.md` | R15 | scan | — | clean |
| `docs/compatibility.md` | R15 | full | r15a-01 | fixed |
| `docs/contributing.md` | R15 | scan | — | clean |
| `docs/denoise-pipeline.md` | R15 | scan | — | clean |
| `docs/examples-ml.md` | R15 | scan | — | clean |
| `docs/examples-transforms.md` | R15 | scan | — | clean |
| `docs/examples.md` | R15 | full | r15a-04 | fixed |
| `docs/index.md` | R15 | scan | — | clean |
| `docs/install.md` | R15 | scan | — | clean |
| `docs/javascripts/katex.js` | R15 | scan | — | clean |
| `docs/logo.svg` | R15 | scan | — | clean |
| `docs/migration_astropy.md` | R15 | scan | — | clean |
| `docs/migration_fitsio.md` | R15 | scan | — | clean |
| `docs/parity.md` | R15 | scan | — | clean |
| `docs/python-workflows.md` | R15 | scan | — | clean |
| `docs/quickstart.md` | R15 | scan | — | clean |
| `docs/release.md` | R15 | full | r15a-05, r15a-08 | fixed |
| `docs/roadmap.md` | R15 | scan | — | clean |
| `docs/stylesheets/extra.css` | R15 | scan | — | clean |
| `docs/torchfits-logo-hero.png` | R15 | scan | — | clean |
| `docs/torchfits-logo-mark.png` | R15 | scan | — | clean |
| `docs/torchfits-logo.png` | R15 | scan | — | clean |
| `.gitattributes` | R16 | scan | — | clean |
| `.github/dependabot.yml` | R16 | scan | — | clean |
| `.github/workflows/bench-report.yml` | R16 | scan | — | clean |
| `.github/workflows/build_wheels.yml` | R16 | full | — | clean |
| `.github/workflows/docs.yml` | R16 | scan | — | clean |
| `.github/workflows/sanitizer.yml` | R16 | full | — | clean |
| `.gitignore` | R16 | scan | — | clean |
| `.pre-commit-config.yaml` | R16 | full | — | clean |
| `AGENTS.md` | R16 | scan | — | clean |
| `CLAUDE.md` | R16 | scan | — | clean |
| `CMakeLists.txt` | R16 | scan | — | clean |
| `LICENSE` | R16 | scan | — | clean |
| `README.md` | R16 | full | — | clean |
| `SDIST-README.txt` | R16 | scan | — | clean |
| `constraints-wheel.txt` | R16 | full | — | clean |
| `extern/VERSIONS.txt` | R16 | full | — | clean |
| `extern/licenses/CFITSIO-LICENSE.txt` | R16 | scan | — | clean |
| `extern/patches/cfitsio-4.7.0-bzip2.patch` | R16 | full | — | clean |
| `extern/patches/cfitsio-4.7.0-plio-cbuf.patch` | R16 | full | — | clean |
| `extern/vendor.sh` | R16 | full | — | clean |
| `overrides/home.html` | R16 | scan | — | clean |
| `overrides/main.html` | R16 | scan | — | clean |
| `overrides/partials/actions.html` | R16 | scan | — | clean |
| `packaging/conda/recipe.yaml` | R16 | full | — | clean |
| `pyproject.toml` | R16 | full | — | clean |
| `zensical.toml` | R16 | scan | — | clean |
