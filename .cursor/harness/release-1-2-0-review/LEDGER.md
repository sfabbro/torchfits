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
| `docs/cli.md` | R7-A (sole editor) | n/a | r7a-06, r7a-10, r7b-09 + decision rows 1/3 | fixed |
