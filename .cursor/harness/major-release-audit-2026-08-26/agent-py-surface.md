# Major-release audit — Python surface (HDU / CLI / data / transforms)

**Date:** 2026-08-26
**Repo:** `/Users/fabbros/src/astroai/torchfits`
**Method:** Read-only. No fixes. Current `src/` plus tests/docs, not git history.
**Scope:** `_hdu/` + `hdu.py`; `cli/`; `data/`; `transforms/`.

Severity:

| Level | Meaning |
|---|---|
| **P0** | Wrong answer, data loss, or a public docs lie users will trust |
| **P1** | Correctness / API trap worth fixing before tag |
| **P2** | Bounded, documented-ish, or freeze implication |
| **P3** | Hygiene / note, not a defect |

Already fixed (do not re-open): Header `del`/`pop` HISTORY orphans (`remove_all=True`, tests in `test_header_versioning.py`); setkey CFITSIO card edit + binary `--out` copy (playbook `setkey-no-rewrite`); CLI `-j` vs `-J` (playbook `cli-j-vs-J`); transform `-J` per-worker instances; HTTP resume `If-Range` + flock; `rgb()` NaN mosaic holes; spectral/continuum transforms deleted and absent from `__all__` / `docs/api-transforms.md`.

**Verdict:** Do not tag as-is. Two P0s (CLI `copy` is not a binary copy; `HDUList.write` can clobber `OUTPUT==INPUT` while the source handle is open) plus several P1 science/API traps. CLI `-j`/`-J` and setkey compression preservation are in good shape.

---

## Issues

### P0-1 — `torchfits copy` is a lossy HDU rewrite, not a binary copy

**Files:** `cli/cmds_copy.py`, `docs/cli.md` (Copy section)
**Evidence:** `copy` does `torchfits.open` + `hdul.write(output, overwrite=True)`. `HDUList.write` calls `_write_hdus_uncompressed`, which `to_tensor()`s every image and materializes every `TableHDURef`. Tile-compressed HDUs land uncompressed; Z\* cards and on-disk encoding are not preserved. Docs say: “Performs an exact, lossless binary copy of a FITS file.”
**Why it matters:** Users will `copy` archive CompImage files and believe they have a bit-identical backup. They do not. Contrast `setkey --out`, which `shutil.copy2`s then edits cards.
**Direction:** Either `shutil.copy2` (and reject/special-case remote), or change the docs and the command name/help. Do not ship the current sentence.

### P0-2 — `copy` / `HDUList.write` with `OUTPUT==INPUT` writes onto an open file

**Files:** `cli/cmds_copy.py`, `_hdu/hdu_list.py` `write()`, `_io_engine/_hdu_rewrite.py` `_write_hdus_uncompressed`
**Evidence:** `resolve_batch_io_pairs` never same-file-checks. `write_api.write()` overwrite uses tempfile + `os.replace`; `HDUList.write` does **not** — it calls `cpp.write_fits_file(path, payload, overwrite)` while `fromfile` still holds `fitsfile*` on that path. `torchfits copy a.fits a.fits` (or `-o` pointing at the input) can truncate/replace the file CFITSIO still has open.
**Also:** `compress`/`convert` go through `write_api.write` (tempfile) so same-path is safer there, but still a silent rewrite of the source. No command rejects `OUTPUT==INPUT`.
**Direction:** Reject same-path unless the command is an intentional in-place editor (`setkey`). Route `HDUList.write` through the tempfile overwrite path.

---

### P1-1 — `DataView` outlives `HDUList.close()`

**Files:** `_hdu/dataview.py`, `_hdu/tensor_hdu.py`, `_hdu/hdu_list.py`
**Evidence:** `close()` → `TensorHDU.mark_closed()` nulls `_file_handle` and `_data_view` on the HDU. A caller who already bound `view = hdul[0].data` keeps a `DataView` whose `_handle` is the now-closed `FITSFile`. `__getitem__` / `shape` / `dtype` have no closed check. Tests (`test_deep_review_wave2.py`) only cover `to_tensor()` after `mark_closed()`, not `DataView`.
**After close, `hdu.data` raises** `ValueError: No file handle available` (not the `RuntimeError: … closed` used by `to_tensor` / `chunks`). In-memory `TensorHDU(data=tensor)` also cannot expose `.data` — DataView is file-handle-only.
**Direction:** `DataView` should check a closed flag (or a generation token) on every I/O; `mark_closed` should poison existing views, not only drop the HDU’s pointer.

### P1-2 — `DataView.__getitem__` is 2-D only and uses the wrong axes on cubes

**Files:** `_hdu/dataview.py`
**Evidence:** `get_shape` returns numpy order `(…, y, x)` (`fits_file.cpp` reverses CFITSIO naxes). Slicing always takes `shape[0]` as y and `shape[1]` as x, then `read_subset(x1,y1,x2,y2)` which is NAXIS1/NAXIS2. For 2-D that matches. For a cube `shape=(N, H, W)`, `data[y, x]` slices **planes × height** and feeds those as spatial x/y. Extra axes are still fully read (C++ `lpixel[i]=naxes[i]` for `i>=2`). Integer index returns a 1-row strip, not a 1-D vector. Unknown BITPIX → silent `float32`.
**Direction:** Require `ndim==2` or slice the last two axes; document 2-D-only if that is the contract. Do not pretend cubes work.

### P1-3 — File-backed `Header`: HISTORY/COMMENT live in `.cards` but not in the dict

**Files:** `_hdu/header.py`
**Evidence:** C++ `read_header` returns `list[tuple[str,str,str]]`. `Header.__init__` for 3-tuples skips mapping insert for `HISTORY`/`COMMENT`. So for `torchfits.open` headers:

- `"HISTORY" in header` is False
- `header["HISTORY"]` is KeyError
- `header.get_history()` returns the cards
- `del header["HISTORY"]` still works (`remove` walks `_cards`)

`add_history()` / `header["HISTORY"] = …` **do** insert the mapping (last value wins). `card("HISTORY")` is first card; mapping is last. `#225` fixed `del`/`pop` orphans when HISTORY *is* in the mapping; it did not unify file-read vs constructed headers.
**Direction:** Either put last HISTORY/COMMENT in the mapping on all construction paths, or reject dict access and document `get_history()` as the only API. Split-brain will generate “HISTORY disappeared” bugs forever.

### P1-4 — `TableHDURef.head(n)` replaces the row window instead of composing

**Files:** `_hdu/table_hdu_ref.py`
**Evidence:** `head` always sets `row_slice=slice(0, n)` of the **full** table. `ref.select(cols).head(10)` is fine; `ref.head(100).head(10)` is still `0:10` of the file, not `0:10` of the current window; a window `slice(100, 200).head(10)` jumps to rows 0–10. `TableHDU.head` is in-memory and raises on `n<0` (backlog “truncates tail” is stale). No test for composition.
**Also:** `to_arrow(**kwargs)` / `scan_arrow` / `reader_arrow` TypeError if the caller passes `columns=` / `row_slice=` / `hdu=` (double kwargs). `filter()` always materializes the whole selection. `select()` does not check that names exist (unlike `TableHDU.select`).
**Direction:** Compose windows (clip `n` against the current slice). Reject colliding kwargs or honor them as overrides explicitly.

### P1-5 — `open()` tables are `TableHDURef`; in-memory API is `TableHDU`

**Files:** `_hdu/hdu_list.py`, `_hdu/table_hdu.py`, `_hdu/table_hdu_ref.py`
**Evidence:** `HDUList.fromfile` always builds `TableHDURef` for ASCII/BINARY tables. `TableHDU.from_fits` is a separate eager path. Mutations: `TableHDU.add_column` / `append_rows` vs `TableHDURef.insert_column_file` / `append_rows_file`. `HDUList.info` / `_repr_html_` label refs as `"TableHDU"`. `HDUList.append` type hint omits `TableHDURef`. `validate()` ignores refs. `TableHDU.select`/`head` drop `source_path`. `num_rows` on `TableHDU` is `cached_property` (stale if `_raw_data` is mutated in place — mutations currently return new objects).
**Direction:** Document the split as the 1.0 contract (lazy vs eager) and make `info` print `TableHDURef`. Do not let `isinstance(..., TableHDU)` be the only type check in user code without a note.

### P1-6 — CLI unhandled exceptions and SIGINT share documented codes

**Files:** `cli/main.py`, `cli/cmds_stats.py`, `cli/cmds_verify.py`, `cli/cmds_diff.py`, `docs/cli.md` Exit Codes
**Evidence:**

| Event | Actual code | Documented meaning |
|---|---|---|
| `KeyboardInterrupt` | 2 (`EXIT_USAGE`) | “Usage error” |
| Uncaught `Exception` (traceback) | 1 | “Difference found” (`diff`) |
| `stats` `read_tensor` failure | 1 (outside the `IoError` try) | should be 3 |
| `verify_checksums` throw | 1 (loop is outside the open try) | should be 3 or 4 |

`copy`/`convert`/`arith`/`setkey` wrap `Exception` → 3. `stats` only wraps `open()`. Unix SIGINT is conventionally 130, not usage-2.
**Direction:** SIGINT → 130 (or a dedicated code). Catch remaining I/O in stats/verify. Never use 1 except `diff`.

### P1-7 — `stats --json` emits bare `NaN` / `Infinity`

**Files:** `cli/cmds_stats.py`, `cli/common.py` `emit_records` / `json_default`
**Evidence:** `json.dumps` default `allow_nan=True` writes `NaN`/`Infinity`, which is not RFC 8259. All-NaN or empty frames produce those tokens. `json_default` only handles `.tolist()` / `str()`. `diff` was taught NaN-aware equality (`_values_equal`); stats JSON was not.
**Direction:** `allow_nan=False` plus `None`, or string sentinels, and a test with an all-NaN image.

### P1-8 — `diff` min/max still call raw `tensor.min()` (uint16/uint32 crash)

**Files:** `cli/cmds_diff.py`
**Evidence:** `stats` upcasts integers before min/max (`test_cli_release_fixes.py` B3). `_image_record` uses `float(tensor.min())` / `.max()` on the raw tensor. PyTorch has no uint16/uint32 reduction kernels → `RuntimeError` wrapped as `IoError` (exit 3) instead of a comparison. Unsigned-convention survey files cannot be `diff`ed.
**Direction:** Same upcast as stats.

### P1-9 — `ArcsinhStretch` / public `safe_arcsinh` restore integer dtype

**Files:** `transforms/helpers.py`, `transforms/stretch.py`, `docs/api-transforms.md`
**Evidence:** `safe_arcsinh` / `safe_log` compute in float then `.to(orig_dtype)`. Integer input → truncated ints (often 0/1), then `ArcsinhStretch` does in-place `div_` on that integer tensor. Docs say normalizers/Arcsinh “require float tensors — integer inputs raise `RuntimeError`”. They do not raise. CLI `transform` casts to float first, so the CLI path is safe; the Python API is not.
`LogStretch.forward` promotes via `clamp_min(..., 0.0)` before `safe_log`, so it accidentally survives. Public `safe_log(int_tensor)` does not.
**Direction:** Keep float output (or raise). Match the docs. Add a test with `torch.int16`.

### P1-10 — `SigmaClip` comment claims non-finite exclusion; mask is NaN-only

**Files:** `transforms/helpers.py` `_get_valid_mask`, `transforms/clip.py`
**Evidence:** `_get_valid_mask` is `~torch.isnan(x) & mask`. Inf is valid. One `+inf` pixel → mean/std inf → the next mask can wipe or fill the frame. `SigmaClip.forward` comment: “excludes … AND non-finite values”. `GlobalScalarNorm` mean/rms already uses `torch.isfinite`; median path does not. All-masked groups clamp count to 1 and fill with 0. Population std (`/N`), not sample.
The old “unreachable inf-guard” is gone; median fill uses `isfinite` → 0, which is reachable for all-NaN groups.
**Direction:** Seed the clip mask with `isfinite`. Decide fill-for-empty-group (NaN vs 0) and test Inf.

### P1-11 — `ZScaleNormalize` is a MAD window named as IRAF zscale

**Files:** `transforms/helpers.py` `zscale_limits`, `transforms/normalize.py`, `docs/api-transforms.md`
**Evidence:** Implementation: `median ± (MAD×1.4826)/contrast`, clipped to min/max. Helper docstring says “fast proxy”. Public class + `api-transforms.md` heading: “IRAF zscale auto-contrast **algorithm**” and then prints the MAD formula. IRAF/astropy `ZScaleInterval` samples, sorts, and fits a line. Limits will not match DS9/IRAF/astropy on images with bright sources.
**Direction:** Rename in docs to “MAD zscale-like display window” (keep the formula), or implement real zscale. Do not call the MAD proxy “the IRAF algorithm.”

### P1-12 — `make_loader` double-fetches staged remotes; Content-Length-less downloads are promoted

**Files:** `data/__init__.py` `make_loader`, `data/datasets.py` `FitsStagedCutoutIterableDataset`, `data/remote.py`
**Evidence:** `make_loader` sees `dataset.files`, `prefetch_urls` into `dataset.cache_dir` or the default remote cache. `FitsStagedCutoutIterableDataset` has no `cache_dir`; `__iter__` downloads again into `staging_dir` / `ephemeral_scratch_dir()`. Two copies; `cleanup` only unlinks the staging path. `FitsCutoutDataset` is skipped (Range cutouts) — good.
`_download_http`: no `Content-Length` and not chunked → `RuntimeWarning` then `tmp.replace(dest)`. A dropped connection is indistinguishable from EOF; truncated bytes become the permanent cache. If-Range resume itself is implemented and tested (`test_remote_http_range.py`).
**Direction:** Skip prefetch for staged datasets (like cutouts), or stage into the same cache key. Refuse promotion without `Content-Length` / chunked.

---

### P2-1 — Dataset class zoo is the 1.0 API

**Files:** `data/datasets.py`, `data/__init__.py`
Eleven public types: `FitsTensor*` + thin `FitsImage*` / `FitsCube*` peers, parallel `FitsSpectrum*` hierarchy, tables in `__init__.py`, `FitsCutoutDataset`, `FitsStagedCutoutIterableDataset`. Image/Cube vs Tensor is `add_channel_dim` + optional `slice_index`. Worker sharding is copy-pasted four times. Collapsing after freeze is a breaking change.
**Note:** Documented in `docs/api-data.md` “Choosing a Dataset”. Fine to ship if that table is the contract.

### P2-2 — `rgb` vs `lupton_rgb` band order

**Files:** `transforms/rgb.py`, `transforms/__init__.py`, `cli/cmds_convert.py`, `docs/api-transforms.md`
**Contract (correct, documented):** `rgb()` shortest-λ first (`rgb(g,r,i)`); `lupton_rgb(r,g,b)` reddest first (Astropy). Convert `--recipe auto|lupton`. Tests cover Lupton↔Astropy and NaN holes for `rgb()`. Residual: easy to swap at the call site; not a code bug.

### P2-3 — Spectral / continuum transforms are gone

**Files:** `transforms/__init__.py`, `docs/changelog.md` 1.0.0rc4, `docs/api-transforms.md`
`spectral.py` / `continuum.py` are deleted. `__all__` has no BandMath / PhaseFold / ALS / AlphaShape. Docs do not resurrect the names. Freeze implication only: do not add them back under the same names without a changelog.

### P2-4 — `FITSHeaderNormalize` unsigned 16/32 is accidental

**Files:** `transforms/fits_meta.py`
BITPIX 16 is marked signed; physical range is `[-32768,32767]*BSCALE+BZERO`, which happens to be `[0,65535]` when BZERO=32768. BITPIX 32 similar. BITPIX 8 has an explicit unsigned branch. Works for the common convention; will confuse anyone reading `_BITPIX_MAP`.

### P2-5 — `HDUList.info` dtype ignores BZERO unsigned convention

**Files:** `_hdu/tensor_hdu.py` `_get_dtype_str`, `_hdu/dataview.py` `dtype`
`info` / `dtype_str` map raw BITPIX (`16` → `int16`). `DataView.dtype` applies BZERO and reports `uint16`. Same HDU, two answers.

### P2-6 — File-read header values are strings

**Files:** `_hdu/header.py` (surface), C++ `get_header` (source)
`test_hdu_file_ops.py` asserts `header["NAXIS"]` is `str`. Astropy gives ints. `int()`/`float()` at use sites mostly cope (`DataView` BSCALE). Migration footgun, known.

### P2-7 — CLI `copy`/`compress` header round-trip via dict sanitizers

Compressed rewrite drops Z\* and structural keys (`_sanitize_header_for_compressed_write` iterates `dict(header)`, so HISTORY already missing from file-read mapping never comes back). Uncompressed `HDUList.write` passes the `Header` object and can keep `.cards`. Inconsistent by path.

### P2-8 — `FitsTableIterableDataset` worker sharding is by scan batch, not row

Documented in the class docstring. Uneven if `batch_size` ≫ rows. OK if frozen with that note.

### P2-9 — `probe` lists `ftp://` as remote then rejects it

**Files:** `cli/common.py` `is_remote_path`, `cli/cmds_probe.py`
`ftp://` is a remote prefix; probe only implements HTTP(S) and vos.

---

### P3 notes (not blocking)

- `TableHDU.col_stats` deprecated, ignored.
- `TableHDU.feat_types` calls integer columns `"categorical"`.
- `TableHDU.filter` goes through Arrow; `TableHDURef.filter` materializes first.
- `HDUList.__del__` → `close()` (interpreter-shutdown noise possible).
- `HDUList.fromfile` `AttributeError` fallback if `open_and_read_headers` missing.
- `cli/cmds_header.py` `_format_card_line` is fitsheader-*ish*, not 80-byte cards.
- `cli/cmds_cutout.py` writes the parent header (CRPIX unchanged) — WCS cutouts are post-1.0.
- `AsymmetricSigmaClip.__repr__` omits `fill`.
- `Compose` does not compose masks returned by children (SigmaClip’s `_last_mask` is unused by Compose).
- Worker-split blocks duplicated in four iterable datasets.
- `_normalize_cpp_chunk` in `data/__init__.py` is identity.

---

## File ledger

Every file in scope. Issue ids refer to the list above.

### `hdu.py` + `_hdu/`

| File | Role | Audit |
|---|---|---|
| `src/torchfits/hdu.py` | Re-export hub (`Header`, `HDUList`, `TensorHDU`, `TableHDU`, `TableHDURef`, `DataView`, `Card`, `TableDataAccessor`) | OK. Public surface matches `__all__`. |
| `src/torchfits/_hdu/card.py` | `Card` NamedTuple + `keyword` alias | OK. Trivial. |
| `src/torchfits/_hdu/header.py` | Dict-like header + `.cards` | **P1-3** HISTORY mapping split; `#225` `del`/`pop`/`remove_all` OK and tested. `setdefault` always bumps `_version`. `_repr_html_` present. |
| `src/torchfits/_hdu/dataview.py` | Lazy image slice + BITPIX dtype | **P1-1** no closed check; **P1-2** 2-D / cube axes; silent float32 fallback; BZERO unsigned dtype **is** correct for 8/16/32. |
| `src/torchfits/_hdu/tensor_hdu.py` | Image HDU, `to_tensor`, `chunks`, `mark_closed` | Closed guard on `to_tensor`/`chunks` OK (private handle per read). `.data` vs `to_tensor` error mismatch. `_get_dtype_str` ignores BZERO (**P2-5**). In-memory HDU has no DataView. |
| `src/torchfits/_hdu/table_hdu.py` | Eager table HDU | **P1-5**. `from_fits` uses public `table.read_torch`. `(N,1)` squeeze on `__getitem__` and accessor. `head(-n)` raises (backlog stale). `select`/`head` drop `source_path`. `num_rows` cached_property. |
| `src/torchfits/_hdu/table_hdu_ref.py` | Lazy file-backed table | **P1-4** `head`; **P1-5** dual API. `_refresh_file_view` invalidates caches then re-reads header. `*_file` mutations OK. `to_arrow(**kwargs)` kwarg collision. Columns cache keyed on `id(header)` without `TableHDU`’s strong-ref guard. |
| `src/torchfits/_hdu/hdu_list.py` | `fromfile` / `close` / `write` / `info` | **P0-2** write; **P1-1** close vs DataView; **P1-5** refs labeled TableHDU. Tables always `TableHDURef`. `close` only `mark_closed`s `TensorHDU` (refs are path-based — OK). `__del__` → close. |

### `cli/`

| File | Role | Audit |
|---|---|---|
| `src/torchfits/cli/__init__.py` | Package marker, empty `__all__` | OK. |
| `src/torchfits/cli/__main__.py` | `python -m torchfits.cli` | OK. |
| `src/torchfits/cli/main.py` | argparse dispatch + exit mapping | **P1-6** SIGINT→2; uncaught→1. `CliError` / `OSError` / `BrokenPipeError` handled. |
| `src/torchfits/cli/common.py` | Exit codes, `-j`/`-J`, batch I/O, JSON emit | **P1-7** `json.dumps` NaN; **P0-2** no same-path check in `resolve_batch_io_pairs`. `-j`/`-J` implementation matches docs/playbook. `run_file_jobs` caps ATen to 1. `is_remote_path` includes ftp (**P2-9**). |
| `src/torchfits/cli/cmds_info.py` | HDU inventory | OK. Uses `iter_file_hdu_pairs` → `IoError`. |
| `src/torchfits/cli/cmds_header.py` | Card dump / keyword table / wildcards | OK. Keyword-table walks `.cards` (HISTORY visible there). Text formatter is not 80-byte FITS. |
| `src/torchfits/cli/cmds_verify.py` | DATASUM/CHECKSUM | **P1-6** `verify_checksums` outside try; exit 4 on fail is correct. |
| `src/torchfits/cli/cmds_diff.py` | Header + image-stat compare | **P1-8** uint min/max; NaN header equality OK (H6). Exit 1 only for real diffs — until uncaught exceptions collide. |
| `src/torchfits/cli/cmds_stats.py` | min/max/mean/std/median | **P1-7** JSON; **P1-6** `read_tensor` unwrapped. Integer upcast for min/max OK (B3). `-j` ignored when `-J>1` (intended). |
| `src/torchfits/cli/cmds_table.py` | Schema + preview | OK. `IoError` on open. Preview via `table.read`. |
| `src/torchfits/cli/cmds_convert.py` | Table export + RGB PNG | **P0-2** allows same path (tempfile write is safer). Recipe auto vs lupton OK (**P2-2**). |
| `src/torchfits/cli/cmds_copy.py` | FITS→FITS | **P0-1**, **P0-2**. Help text does not say “binary.” Docs do. |
| `src/torchfits/cli/cmds_arith.py` | imarith-class ±×÷ | OK for this audit. Integer saturate + div→float64 (B2). `-j`/`-J` wired. Same-path possible via `-o`. |
| `src/torchfits/cli/cmds_cutout.py` | Pixel box / CFITSIO section | OK. Rejects section+`--box`. Header WCS not updated (P3). |
| `src/torchfits/cli/cmds_compress.py` | compress / decompress | Rewrite via `write()` (tempfile). Same-path still overwrites source. `--algorithm` / `--split hdu` OK. Not a binary copy either — expected for this command. |
| `src/torchfits/cli/cmds_transform.py` | Named `transforms` class | **P1-9** mitigated by `.float()`. **`-J` does not share one instance** (rebuilds per worker). Stateful `_last_*` safe under fan-out. |
| `src/torchfits/cli/cmds_probe.py` | Local info or HTTP/vos peek | **P2-9** ftp. SSRF via `http_open`. Header peek is first 2880 bytes only (PRIMARY). |
| `src/torchfits/cli/cmds_setkey.py` | set / rename / delete cards | **OK for 1.0.** CFITSIO delete/update; `--out` is `shutil.copy2` then edit; CompImage Z\* preserved (`test_setkey_delete_preserves_tile_compression`). Rejects HISTORY/COMMENT. Duplicate in-place paths rejected under `-J`. |

### `data/`

| File | Role | Audit |
|---|---|---|
| `src/torchfits/data/__init__.py` | Tables, cutouts, `fits_collate_fn`, `make_loader`, re-exports | **P1-12** prefetch vs staged. Cutout Range skip OK. Collate rejects VLA/strings. `_normalize_cpp_chunk` identity (P3). `FitsTableDataset` loads the full table at init (documented). |
| `src/torchfits/data/datasets.py` | Tensor/Image/Cube/Spectrum + staged cutouts | **P2-1** zoo. **P1-12** staged download+cleanup. Worker-split duplicated ×3. Spectrum table vs IMAGE arms OK. `FitsSpectrumIterableDataset` borrows a dummy `FitsSpectrumDataset(files[:1])` for methods — works because readers take `path`. |
| `src/torchfits/data/remote.py` | HTTP/vos cache, prefetch, flock | **If-Range + flock: done.** **P1-12** Content-Length-less promote. Prefetch errors stored and re-raised. Locks not popped (commented race). vos ignores `.partial` resume (unlink then copy). Windows: no flock, unique `.partial` names. |

### `transforms/`

| File | Role | Audit |
|---|---|---|
| `src/torchfits/transforms/__init__.py` | Public `__all__` | **P2-2**, **P2-3**. No spectral names. `rgb` + `lupton_rgb` both exported. |
| `src/torchfits/transforms/base.py` | `FITSTransform`, `Compose`, `AsModule` | OK. Inverse unwinds reverse. Mask forwarded unchanged (P3: child clip masks unused). |
| `src/torchfits/transforms/helpers.py` | Masked stats, `safe_*`, zscale proxy, MAD | **P1-9** dtype restore; **P1-10** `isnan` vs `isfinite`; **P1-11** zscale proxy. Median via `nanquantile(0.5)` (numpy-like). Integer stats upcast OK. |
| `src/torchfits/transforms/stretch.py` | Arcsinh / Log / Sqrt | **P1-9** Arcsinh. `a<=0` raises. Log/Sqrt clamp negatives (documented). Log inverse exponent clamp present. |
| `src/torchfits/transforms/normalize.py` | ZScale / Robust / BG / percentile / minmax / global | **P1-11** ZScale name. Inverse needs prior forward (documented). `GlobalScalarNorm` mean/rms uses `isfinite`; median does not. Sign-preserving divisor floor OK. |
| `src/torchfits/transforms/clip.py` | `SigmaClip`, `AsymmetricSigmaClip` | **P1-10**. `fill="nan"|"mean"|"median"`. Inverse raises. Inf-guard leftover is gone. Asymmetric uses MAD once (not iterative). `_last_mask` stored, never read by Compose. |
| `src/torchfits/transforms/rgb.py` | `lupton_rgb`, `rgb`, `write_rgb_image` | **P2-2** order. **NaN holes in `rgb()`: fixed** (`isfinite` → NaN through stats, then black). `lupton_rgb` can still emit NaN (Astropy-like). Scarlet mix rows ~normalized. PNG via stdlib zlib. |
| `src/torchfits/transforms/fits_meta.py` | BSCALE / TSCAL / TNULL / header normalize | **P2-4** uint16 range. `FITSHeaderScale` functional (no in-place on aliased float32). `TNullToNan` lossy by design. |

---

## Suggested 1.0 cut (not implemented)

Must-fix before calling the CLI/docs honest:

1. **P0-1** — binary `copy` or stop saying “lossless binary copy”
2. **P0-2** — reject `OUTPUT==INPUT` except `setkey`; tempfile on `HDUList.write`

Should-fix if the tag claims science + HDU handle safety:

3. **P1-1 / P1-2** — closed `DataView`; 2-D-only cube slicing
4. **P1-3 / P1-4 / P1-5** — Header HISTORY mapping; `TableHDURef.head`; label refs in `info`
5. **P1-6 / P1-7 / P1-8** — exit codes, stats JSON, diff uint min
6. **P1-9 / P1-10 / P1-11** — stretch dtypes, SigmaClip Inf, zscale naming
7. **P1-12** — staged `make_loader` prefetch + incomplete HTTP promote

Safe to freeze as-is: `-j`/`-J`, setkey CompImage cards, `rgb` vs `lupton_rgb` docs, spectral-transform removal, Dataset zoo (with the api-data table as the contract).
