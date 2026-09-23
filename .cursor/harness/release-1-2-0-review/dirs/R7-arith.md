# R7 slice C — `cli/` arith / compress / convert / setkey / transform

Round R7, finding prefix `r7c`. Owner: R7-CliArith. Files (exclusive src):
`src/torchfits/cli/cmds_arith.py`, `cmds_compress.py`, `cmds_convert.py`,
`cmds_setkey.py`, `cmds_transform.py`. Owned tests:
`tests/test_cli_same_path_refusal.py`, `tests/test_cli_transform_parallel.py`,
`tests/test_cli_setkey_integrity.py`, `tests/test_cli_arith_edges.py`,
`tests/test_cli_convert_edges.py` (all new).

## Evidence runs

Failing-first, all five owned test files against unfixed HEAD (`1bb6958`+R1–R6):

```
$ pixi run pytest tests/test_cli_same_path_refusal.py tests/test_cli_transform_parallel.py \
    tests/test_cli_setkey_integrity.py tests/test_cli_arith_edges.py tests/test_cli_convert_edges.py -q
...
FAILED tests/test_cli_arith_edges.py::test_arith_uint32_mul_saturates_instead_of_wrapping - assert [0, 0, 1, 0] == [4294967295, ...1, 4294967295]
FAILED tests/test_cli_arith_edges.py::test_arith_fractional_scalar_on_integer_warns - AssertionError: assert 'fractional' in ''
FAILED tests/test_cli_convert_edges.py::test_convert_png_fails_when_hdu_count_read_fails - assert 0 == 3
FAILED tests/test_cli_convert_edges.py::test_convert_lupton_rejects_auto_recipe_flags - AssertionError: (('--zeropoints', '25,25,25'), '') assert 0 == 2
FAILED tests/test_cli_convert_edges.py::test_convert_auto_rejects_lupton_flags - AssertionError: (('--q', '4.0'), '') assert 0 == 2
20 failed, 6 passed in 43.09s
```

The 20 failures are exactly: `test_convert_table_same_path_refused`,
`test_convert_png_same_path_refused`, `test_convert_png_output_equals_one_input_refused`,
`test_compress_split_hdu_output_must_not_clobber_another_input`,
`test_arith_split_hdu_output_must_not_clobber_another_input`, all 10
`tests/test_cli_setkey_integrity.py` tests except none (10/10 failed), the 2 arith
edge tests above and the 3 convert edge tests above. The 6 passes were the
already-satisfied pins: `test_compress_same_path_refused`,
`test_decompress_same_path_refused`, the 3 `test_transform_parallel_matches_serial`
params, `test_arith_div_by_zero_refused`.

Throwaway probe outputs (pre-fix, `/tmp/r7c_probe.py`, `/tmp/r7c_probe2.py`) captured
the raw broken behavior per finding and are quoted inline below.

After the fixes:

```
$ pixi run pytest tests/test_cli_same_path_refusal.py tests/test_cli_transform_parallel.py \
    tests/test_cli_setkey_integrity.py tests/test_cli_arith_edges.py tests/test_cli_convert_edges.py -q
..........................                                               [100%]
26 passed in 47.46s
```

## Findings

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r7c-01 | BLOCKER | 1/5 | cmds_convert.py:run | `convert` never refused OUTPUT == INPUT (decision 4). `convert img.fits -o img.fits --to png` exited 0 and replaced the FITS source with PNG bytes (probe: `file changed: True`); `convert tab.fits tab.fits --to fits -c a` exited 0 and silently dropped column b from the source; multi-input PNG with `-o` equal to one input clobbers that input mid-read. | Call `common.reject_same_path(src, args.output)` for every INPUT/OUTPUT pair in `run()` — verbatim `cmds_copy.py` contract (UsageError, exit 2, `refusing same-path rewrite in place: {src}`) per R7-CliTableStats' pin. Tests: `test_convert_table_same_path_refused`, `test_convert_png_same_path_refused`, `test_convert_png_output_equals_one_input_refused` (fail before: exit 0 + bytes changed; pass after: exit 2 + bytes intact). | fixed |
| r7c-02 | BLOCKER | 1/2 | cmds_arith.py:_compute | Integer `mul` accumulated in int64 → silent wrap for wide products. `arith a.fits b.fits --op mul` on uint32 images with 4294967290-valued pixels produced `[0, 0, 1, 0]` (true products 1.8e19 wrap 2^63 → negative → clamped to 0) instead of saturating to the uint32 maximum. Scalar `mul --value 1e10` could wrap the same way. This contradicts `_compute`'s own docstring ("without silent integer wraparound"). | `mul` on integer inputs accumulates in float64 (exact for every product surviving saturation into any ≤32-bit FITS dtype; saturates cleanly beyond), except `out_dtype == torch.int64` where the exact int64 path is kept. `_compute` now receives the op name. Test: `test_arith_uint32_mul_saturates_instead_of_wrapping` (fails before: `[0, 0, 1, 0]`; passes after: `[4294967295, 4294967295, 1, 4294967295]` + `saturated` RuntimeWarning). | fixed |
| r7c-03 | BLOCKER | 1 | cmds_setkey.py:_apply_edits | Every edit replayed the FULL header through `_write_header_cards_if_supported`; CFITSIO commentary-card semantics append, so each `setkey` edit duplicated the HDU's HISTORY/COMMENT cards. Probe: 2 HISTORY cards `['h1','h2']` became `['h1','h2','h1','h2']` after `setkey -k NEWKEY --value 1`. Silent header corruption on every use (also bloats repeated edits quadratically in cards written). | Write only the edited cards (CFITSIO `fits_update_key`/`fits_delete_key` per card — the `setkey-no-rewrite` contract); never replay untouched cards. Tests: `test_setkey_edit_preserves_history_and_comment_exactly`, `test_setkey_rename_preserves_history_and_comment_exactly` (fail before: duplicated lists; pass after: exact match). Root cause in the shared replay helper is recorded as r7c-22. | fixed |
| r7c-04 | BLOCKER | 1 | cmds_setkey.py:_apply_edits | `--rename` destroyed data in two ways: (a) `--rename XKEY=XKEY` exited 0 and DELETED `XKEY` (write new card then delete old — same key); (b) `--rename AKEY=BKEY` with `BKEY` present exited 0 and silently replaced `BKEY=2` with `AKEY`'s value (probe: `BKEY=1 AKEY present: False`). | Validate in `_prepare_edits`/phase 1: `--rename OLD=NEW` with identical normalized names raises `UsageError`; renaming onto an existing target raises `UsageError` naming both keys. Tests: `test_setkey_rename_same_name_refused`, `test_setkey_rename_onto_existing_target_refused` (fail before: exit 0 + data destroyed; pass after: exit 2 + file bytes unchanged). | fixed |
| r7c-05 | MAJOR | 5 | cmds_setkey.py:_parse_hdus | Non-integer `--hdu` leaked a raw `ValueError` (`invalid literal for int() with base 10: 'z'`) wrapped as `IoError` → exit 3. The docs exit table maps bad arguments/syntax to 2 (cf. `common.parse_hdu_list` which raises `UsageError`). | `int(part)` wrapped → `UsageError(f"invalid HDU index: {part!r}")` (exit 2), same wording as `common.parse_hdu_list`. Test: `test_setkey_non_integer_hdu_is_usage_error` (fails before: rc 3; passes after: rc 2). `tests/test_cli.py::test_setkey_rejects_negative_hdu_index` (rc 2, `"hdu"` in stderr) still passes. | fixed |
| r7c-06 | MAJOR | 3 | cmds_setkey.py:_apply_edits | Validation was interleaved with mutation: `setkey f -e all --delete AKEY --delete MISSINGKEY` deleted `AKEY` from the file first, then failed on the missing key — probe: `file bytes changed: True; AKEY still present: False` with exit 3. A failed batch left the file partially edited (silent data loss on retry/rollback assumptions). "missing keyword" also exited 3 (I/O) although the exit table maps bad requests to 2. | Two-phase `_apply_edits`: phase 1 validates every edit against all selected HDUs and resolves the exact card ops (order-aware: delete-then-rename composes); phase 2 applies via CFITSIO card ops only after the whole batch is known-good. `missing keyword` is now `UsageError` (exit 2). Test: `test_setkey_failed_batch_leaves_file_untouched` (fails before: bytes changed + rc 3; passes after: bytes identical + rc 2). | fixed |
| r7c-07 | MAJOR | 3 | cmds_setkey.py:_edit_one | `--out`/`--out-dir` destinations resolving to the input through a different string (symlinked out-dir, `./x` vs `x`) hit `shutil.copy2`'s `SameFileError`: probe `setkey hk5.fits --out-dir <symlink-to-dir>` → exit 3 `'/…/hk5.fits' and '/…/link/hk5.fits' are the same file`, no edit applied. Deterministic failure on valid input (the string-equal case already edits in place). | Compare `os.path.realpath(src) != os.path.realpath(dest)` before copying; same file → edit in place (identical final bytes to copy-then-edit). Test: `test_setkey_out_dir_resolving_to_input_edits_in_place` (fails before: rc 3 + no `ZKEY`; passes after: rc 0 + `ZKEY=7`). | fixed |
| r7c-08 | MAJOR | 1 | cmds_compress.py:_rewrite_one_input_split_hdu; cmds_arith.py:_arith_one_file | `--split hdu` generates `{stem}_hduNN.fits` names, which can equal ANOTHER input of the same batch (the natural re-split-in-place workflow: `compress x.fits x_hdu00.fits --split hdu --out-dir .`): the generated output silently overwrote the unrelated input `x_hdu00.fits` (decision 4's OUTPUT == INPUT, across batch members). `ensure_unique_split_stems` only guards output-vs-output collisions. | Before each split write, `reject_same_path(other_input, output_path)` over the batch's inputs (verbatim refusal, exit 2). Tests: `test_compress_split_hdu_output_must_not_clobber_another_input`, `test_arith_split_hdu_output_must_not_clobber_another_input` (fail before: prior input's bytes rewritten / exit 0; pass after: exit 2 + bytes intact). | fixed |
| r7c-09 | MAJOR | 5/1 | cmds_convert.py:_auto_band_indices | `except Exception: n_hdus = 1` swallowed any failure of `torchfits.read_num_hdus` and silently fell back to a GREY render where RGB from HDUs 0,1,2 was the contract (or hid the real IO error behind an exit-0 PNG). Class-5 anti-pattern (`except Exception` on IO). | Swallow removed: a failure reading the source fails the command (exit 3 via `run`'s `IoError` wrapper) instead of rendering a wrong-band PNG. Test: `test_convert_png_fails_when_hdu_count_read_fails` (in-process `main([...])` with `read_num_hdus` raising; fails before: `assert 0 == 3`; passes after: rc 3 + no PNG). | fixed |
| r7c-10 | MINOR | 5 | cmds_setkey.py:run | `--value` without `--key` was silently ignored whenever `--delete`/`--rename` was also present (probe: `setkey f --value 5 --delete AKEY` → exit 0, `AKEY` deleted, `--value` dropped on the floor). Conversely `-k KEY` without `--value` combined with `--rename` failed only per-file AFTER the copy step. | Early checks in `run`: `--value` without `--key` → `UsageError("--value requires --key")`; `--key` without `--value` → `UsageError` unconditionally (same message as before). Tests: `test_setkey_value_requires_key`, `test_setkey_key_requires_value_even_with_rename` (fail before: rc 0 / late rc; pass after: rc 2 + file untouched). | fixed |
| r7c-11 | MINOR | 1/2 | cmds_arith.py:_saturate_to | Integer output with a fractional float operand silently truncated: `arith i16.fits --op add --value 0.5` produced `[100, 101, 102, 103]` with NO diagnostic — every pixel silently lost its `.5` (`div` deliberately promotes to float64 to keep fractions; add/sub/mul truncated in silence). | `_saturate_to` counts fractional losses on the float→int cast and emits a `RuntimeWarning` ("N pixel value(s) lost fractional parts casting to …"). `--dtype auto` help text updated to "warning on saturation or fractional truncation". Test: `test_arith_fractional_scalar_on_integer_warns` (fails before: `'fractional' in ''`; passes after: warning + documented truncation). | fixed |
| r7c-12 | MINOR | 5 | cmds_convert.py:_convert_png | Recipe-scoped flags were silently ignored across recipes: `--recipe lupton` accepted and ignored `--zeropoints`/`--calibrated`/`--brightness`/`--saturation` (an explicitly requested photometric calibration was dropped → silently wrongly-calibrated PNG); `--recipe auto` accepted and ignored `--q`/`--stretch`. Both probed at exit 0. | Explicitly-passed incompatible flags raise `UsageError` naming the flag and its recipe. Numeric defaults moved inside (`default=None` in argparse; documented defaults resolved at use) so explicit use is detectable — no CLI surface change. Tests: `test_convert_lupton_rejects_auto_recipe_flags`, `test_convert_auto_rejects_lupton_flags` (fail before: rc 0 + PNG written; pass after: rc 2 + no output). | fixed |
| r7c-13 | — | 5 | cmds_compress.py:_resolve_file_pairs | Decision 4 re-derivation: `compress`/`decompress` ALREADY refused OUTPUT == INPUT at HEAD via `resolve_batch_io_pairs(..., refuse_same_path=True)` → `common.reject_same_path` (probe: exit 2 `refusing same-path rewrite in place: …`, file intact). No code change needed for the pair path. | Pinned: `test_compress_same_path_refused`, `test_decompress_same_path_refused` (pass before and after; verbatim message + exit 2 + bytes intact). | fixed |
| r7c-14 | — | 4 | cmds_transform.py:run | Decision 5 re-derivation: the shared-instance race is ALREADY fixed at HEAD — with `-J > 1` each file gets a fresh `_build_transform(args.name)` instance inside the worker lambda, and transform `_last_state`/`_last_mask` are per-(instance, thread) `_ThreadedAttr` values whose `forward()` outputs are pure per call. Probe: `-J 4` np-equal to `-J 1` over 4 files for `ZScaleNormalize` (DataState-bearing: `produces = DataState.NORMALIZED`, stores `_last_state`), `RobustNormalize`, `MeshBackgroundSubtract`. The serial path still shares one instance across files but outputs are state-independent (state lives in `Payload` per the R2 contract). | Pinned: `test_transform_parallel_matches_serial` (3 parametrized transforms × 4 files, `np.testing.assert_array_equal` between `-J 1` and `-J 4` outputs; passes before and after). No code change — per-file instance construction is the documented pattern already present at HEAD. | fixed |
| r7c-15 | MINOR | 5 | cmds_arith.py:run | Unlike copy/compress/transform (all refuse), `arith a.fits --op add --value 1 -o a.fits` permits OUTPUT == INPUT. Not corrupting (`_arith_one_file` reads every HDU before writing), so it behaves as an in-place transform, but the in-place policy is inconsistent across rewrite commands and decision 4 did not name `arith`. | None (recorded; behavior change beyond decision 4 would need a policy call). Repro: above command exits 0 and rewrites the source in place. | deferred |
| r7c-16 | MINOR | 3 | cmds_compress.py:_rewrite_file; cmds_convert.py:_convert_table/_convert_png; cmds_transform.py:_transform_one; cmds_arith.py:_arith_one_file | A write failing mid-way (disk full, IO error) leaves a partial OUTPUT file behind. The failure is loud (exit 3) and `overwrite=True` replaces the partial on retry, so no silent wrong result — but the cleanup policy (atomic tmp+rename vs unlink-on-error) is cross-command and matches `cmds_copy._copy_remote`'s existing behavior (also no cleanup). | None (recorded). Sites listed above; ~4 lines per site if adopted uniformly. | deferred |
| r7c-17 | MINOR | 1 | cmds_compress.py:_rewrite_file (via `torchfits.write(..., compress=)`) | `compress`+`decompress` round-trip of a single-image FITS yields primary(empty)+image extension (2 HDUs) instead of the original 1 HDU: the compressed writer always emits its own empty primary (`_is_skippable_empty_primary` convention; probe round-trip err 0.0 but `unfz hdus=2 shapes=[(0,), (4, 4)]`). fpack/funpack drop the empty primary on expand. Data is preserved; structure differs. | None at CLI level (writer-level convention in `_io_engine/_hdu_rewrite.py`, not my set). Repro: probe P1c. | deferred |
| r7c-18 | MINOR | 1 | cmds_arith.py:_apply_op | Image–image ops do not check `BUNIT` (adding counts to mag silently yields nonsense; cf. IRAF imarith which also does not check). Needs a warn-vs-refuse policy call. | None (recorded). | deferred |
| r7c-19 | MINOR | 2 | cmds_arith.py:_compute | Residual of r7c-02: with `out_dtype == torch.int64` (BITPIX=64 images) `mul` keeps exact int64 accumulation and can still wrap at ≥2^63 (float64 accumulation would trade this for low-bit rounding and an unrepresentable int64 clamp bound). No FITS-scaled case produces int64 images in practice. | None (recorded). Repro: int64 image with ~2^32-valued pixels squared. | deferred |
| r7c-20 | MINOR | 1 | cmds_setkey.py:_apply_edits | `--rename` silently dropped the card's comment: `header[new_key] = header[old_key]` copies the mapping VALUE only, so a rename of `BKEY = 2 / keep me` produced `RENAMED = 2` with an empty comment. Metadata loss on a documented operation (CFITSIO `fits_rename_key` would keep the whole card). | The rename op now carries `(value, comment)` from the source card through the per-card write (`{key: (value, comment)}` card form). Test: `test_setkey_rename_preserves_card_comment` (fails before: empty comment; passes after: `keep me` preserved). | fixed |
| r7c-21 | MINOR | 5 | (out of set) _io_engine/write_api.py / C++ `write_hdu_header_cards` | A dict header with a list-valued commentary key (`{"HISTORY": ["h1", "h2"]}`) crashes the C++ card replay with `std::bad_cast`, surfacing as `RuntimeError: Failed to write FITS file …: std::bad_cast` — no key named, untyped error. Multi-HISTORY should use `Header.add_history` but the error contract is poor. | None (out of slice set; recorded for the integration owner / R4). | deferred |
| r7c-22 | MINOR | 1 | (out of set) _io_engine/_write_helpers.py:_write_header_cards_if_supported | Root cause of r7c-03: replaying a full card list onto a file that already contains commentary cards APPENDS HISTORY/COMMENT instead of replacing them (CFITSIO `fits_update_key` semantics on commentary cards). setkey no longer replays full headers, but any other caller passing a full header with HISTORY/COMMENT onto an existing file has the same duplication. | None (out of slice set; recorded for the integration owner / R4). | deferred |
| r7c-23 | MINOR | 5 | (out of set) _table/interop.py:write_parquet/write_csv/write_ipc | Passing a column dict (`{"a": [...]}`) — a first-class input of `table.write` — to `write_parquet` misroutes into `pa.Table.from_batches(list(dict))` and dies with pyarrow's `TypeError: Cannot convert str to pyarrow.lib.RecordBatch` (iterating a dict yields its keys). No dict normalization on the export side. | None (out of slice set; recorded for the integration owner / R5). | deferred |
| r7c-24 | MINOR | 8 | docs/cli.md (owed text — slice A lands) | Doc/flag mismatches found in the review: (a) the `setkey` section documents a `--comment` flag that does not exist; (b) the compress `Supported algorithms` list and `compress --help` disagree with each other and with the accepted set (probe: `RICE`, `RICE_1`, `GZIP_1`, `HCOMPRESS_1`, `BZIP2_1` accepted; `PLIO_1` accepted by the resolver but rejected by CFITSIO for float data — its doc scope is integer images); (c) only `copy` documents the same-path refusal although `compress`/`decompress`/`convert`/`transform`/`cutout` all refuse. Verbatim owed doc text below. | Recorded; `docs/cli.md` is owned by slice A (only authorized doc file). | deferred |
| r7c-25 | MINOR | 1 | cmds_arith.py:_saturate_to | Non-finite intermediates on integer outputs (e.g. image B read through the BLANK→NaN scaled path, `--value nan`) reach `float→int` casts whose result is platform-dependent. Today NaN pixels at least trip the `saturated` warning (`nan != nan`) but the stored value is arbitrary. Needs a NaN policy for integer outputs (refuse vs sentinel). | None (recorded). | deferred |

### Verbatim owed `docs/cli.md` text (r7c-24, for slice A)

1. Delete from `### setkey` (no `--comment` flag exists):

```bash
# Set keyword with comment
torchfits setkey science.fits -k FILTER --value "g" --comment "SDSS g-band filter"
```

2. Replace the `### compress` line
   `` Supported algorithms: `RICE_1` (default), `GZIP_1`, `GZIP_2`, `HCOMPRESS_1`, `PLIO_1`. ``
   with
   `` Supported algorithms: `RICE_1` (default), `RICE`, `GZIP_1`, `GZIP_2`, `HCOMPRESS_1`, `PLIO_1` (integer images only), `BZIP2_1` when the build links it. ``
   and align `compress --help`'s "also RICE, GZIP_1, GZIP_2, HCOMPRESS_1" list to the
   same names (the `--algorithm` string is free-form; CFITSIO rejects unknowns with
   `Unsupported compression algorithm: …`, exit 3).

3. In `### compress`, `### decompress` and `### convert`, add the sentence already used
   in `### copy`:
   `Same-path INPUT OUTPUT is refused.`
   (`transform`/`cutout` refuse too — consider one shared note instead of four.)

### Invariant verification (decision 7 + playbook)

- `cli-j-vs-J`: verified. `-j` maps to `torch.set_num_threads` via `configure_torch_jobs`
  (only when `file_jobs == 1`); `-J` fans out `run_file_jobs` whose `_worker` caps
  `torch.set_num_threads(1)` per worker. Unchanged by this slice.
- `is_remote_path` includes ftp: verified at `common.py:_REMOTE_PREFIXES`
  (`"http://", "https://", "ftp://", "vos://", "vos:", "vault:"`). Unchanged.
- `setkey-no-rewrite`: strengthened — setkey now writes only the edited cards through
  CFITSIO `fits_update_key`/`fits_delete_key` (never an HDUList rewrite); tile-compressed
  HDUs remain covered by `tests/test_cli.py::test_setkey_delete_preserves_tile_compression`.
- `copy-is-binary` refusal style: the exact `reject_same_path` helper is reused everywhere
  (no re-implementation), matching `tests/test_cli_diff_copy_cutout.py`'s pin.

## Per-file disposition

| file | depth | finding IDs | status |
|---|---|---|---|
| src/torchfits/cli/cmds_arith.py | deep (429 LOC, all functions reviewed; probe P2/P2b/P10/P12/P13) | r7c-02, r7c-08, r7c-11, r7c-15, r7c-18, r7c-19, r7c-25 | fixed + deferred noted |
| src/torchfits/cli/cmds_compress.py | deep (174 LOC; probe P1c/P4/P11 + split-clobber scenario) | r7c-08, r7c-13, r7c-16, r7c-17 | fixed + deferred noted |
| src/torchfits/cli/cmds_convert.py | deep (342 LOC; probe P3/P3b/P6/P6b/P7/P9/P9b) | r7c-01, r7c-09, r7c-12, r7c-16, r7c-24 | fixed + deferred noted |
| src/torchfits/cli/cmds_setkey.py | deep (288 LOC; probe P5a–P5g + symlink case) | r7c-03, r7c-04, r7c-05, r7c-06, r7c-07, r7c-10, r7c-20 | fixed |
| src/torchfits/cli/cmds_transform.py | deep (190 LOC; probe P8 across 3 DataState-bearing transforms) | r7c-14 | fixed (verified + pinned; no code change needed) |
