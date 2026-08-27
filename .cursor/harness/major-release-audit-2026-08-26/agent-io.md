# Major-release I/O audit

- **Date:** 2026-08-26
- **Scope:** every file under `src/torchfits/_io_engine/` plus `io.py`, `cache.py`, `http_util.py`, `vos_uri.py`, `logging.py`, `fits_schema.py`, `header_parser.py`, `interop.py`, `_tensor_buffer.py`, `_string_decode.py`
- **Method:** from-scratch review of current sources; C++ only as evidence for mmap/CFITSIO/BLANK/checksum behavior. Prior audits ignored except as hypotheses to re-check.
- **Fixes:** none (audit only)
- **Evidence host:** local `pixi run python` probes on 2026-08-26 (darwin)

## Executive summary

The I/O façade is close to a freezeable surface: SSRF guards exist on the Python façades, unsigned BZERO conventions are applied on both mmap and CFITSIO image paths, and `write(..., checksum=True)` stamps every HDU. Three issues should block a major tag until decided:

1. **Quantize NaNs are not NaNs on `torchfits.read`** — write emits `BLANK=-32767`, astropy masks them, torchfits returns a finite physical value near the robust floor.
2. **Default `return_header=True` cache is unsound** — first hit is a `Header`, later hits are a shared `dict`; mutating it poisons subsequent reads.
3. **HDU rewrite (`replace_hdu` / `insert_hdu` / `delete_hdu`) is not CompImage/`setkey`-safe** — full rewrite decompresses, leaves stale `Z*` cards, and drops checksums while `verify_checksums` still reports `ok=True`.

---

## Issues

### IO-001 — Quantize BLANK is not honored on read

- **Severity:** BLOCKER
- **Category:** quantize/scale semantics, mmap vs CFITSIO
- **Files+lines:**
  - `src/torchfits/_io_engine/quantize.py` 26–28, 305–309
  - `src/torchfits/_io_engine/_write_helpers.py` 135–137
  - `src/torchfits/cpp_src/fits_detail.h` 589–597 (nulval only for *compressed* float)
- **Description:** `quantize="robust"` encodes non-finite pixels as `BLANK=-32767` and writes the keyword. `torchfits.read` / `read_tensor` do not substitute those codes with NaN. CFITSIO `fits_read_img` is called with `nullval_ptr=nullptr` for uncompressed scaled images (nulval is set only when `compressed && TFLOAT/TDOUBLE`). The mmap fast path never sees BLANK either (scaled non-unsigned images skip mmap and take this CFITSIO path).
- **Why:** The writer comment says NaN must never masquerade as real data. After read, it does: the sentinel is scaled as `BSCALE * (-32767) + BZERO`, i.e. a finite value near `lo`.
- **Evidence:** `write(randn, quantize="robust")` with `x[0,0]=nan` → header `BLANK=-32767`; `torchfits.read` at `[0,0]` was finite `76.13…`, `isnan=False`. Astropy on the same file yields NaN (`tests/test_release_semantics.py::test_quantize_nan_becomes_blank_not_lo` only asserts astropy).
- **Fix:** On image read (CFITSIO and any remaining integer path), pass `nulval=NaN` whenever `BLANK` is present *or* always for TFLOAT scaled reads; document that mmap of unscaled integer storage returns the sentinel code. Mirror TNULL→NaN on quantized table columns in the thin reader.
- **Compatibility:** Changes values at BLANK/TNULL positions (NaN instead of a finite code). That is the correct FITS contract; callers comparing against the finite masquerade will need updating.
- **Tests:** Round-trip `torchfits.read` (mmap True/False, `read_tensor`, subset) of a quantized image with a NaN; assert `isnan` at that pixel and finite agreement elsewhere. Same for a quantized table column via `read` / `table.read_torch`.

### IO-002 — `return_header=True` cache: Header→dict and in-place poison

- **Severity:** HIGH
- **Category:** correctness, API inconsistency
- **Files+lines:**
  - `src/torchfits/_io_engine/caches.py` 204–321 (`check_read_cache`), 332–360 (`_clone_read_value`), 363–375 (`store_cached_read`)
  - `src/torchfits/_io_engine/_read_pipeline_fallback.py` 236–245, 374–379
- **Description:** Fallback reads (including every `return_header=True`) store `(data, header, sig)` then `_clone_read_value` walks the tuple. `Header` is a `dict` subclass, so it is cloned into a **plain dict** (HISTORY/COMMENT cards dropped). On hit, `cached_header` is returned **without cloning**. Default `cache_capacity=10` enables this.
- **Why:** Public `read(..., return_header=True)` is not referentially transparent: type and mutability change on the second call.
- **Evidence:** First `read(..., return_header=True)` → `Header`; second → `dict`. Setting `hdr3['POISON']=1` made `POISON` appear on the next read.
- **Fix:** Store `(clone(data), Header(list(header.cards)), sig)`; on hit return `Header(list(cards))` plus cloned tensors. Include `mmap`, `raw_scale`, `mode`, `scale_on_device` in the cache key (see IO-003).
- **Compatibility:** Restores Header; any caller that started depending on a plain dict from cache hits will see Header again (the documented type).
- **Tests:** Two `return_header=True` reads; both `isinstance(..., Header)`; mutate first header; second read unchanged. HISTORY cards survive. Different `mmap`/`raw_scale` do not collide.

### IO-003 — `raw_scale=True` is dropped on the fallback path

- **Severity:** HIGH
- **Category:** scale semantics, API inconsistency
- **Files+lines:**
  - `src/torchfits/_io_engine/_read_pipeline.py` 385–409 (fallback call has no `raw_scale`)
  - `src/torchfits/_io_engine/_read_pipeline_fallback.py` 193–221 (`read_fallback_image` always `read_full`)
  - `src/torchfits/io.py` 188–211 (`read_tensor` keeps `raw_scale`)
- **Description:** `read(..., raw_scale=True)` uses `read_full_raw` on the generic fast path. Any fallback (`return_header=True`, named HDU after a failed image probe, table-type cache miss, …) calls `read_full`, which applies BSCALE/BZERO. `read_tensor(..., raw_scale=True)` stays on the image path and is consistent.
- **Why:** Same kwargs, different values depending on whether a fast path ran. Docs say `raw_scale` skips BSCALE/BZERO.
- **Evidence:** uint16 image (`BZERO=32768`): `read_tensor(..., raw_scale=True)` → `int16` storage codes 7232…7247; `read(..., raw_scale=True, return_header=True)` → `uint16` 40000…40015 (logical, scale applied).
- **Fix:** Thread `raw_scale` into `read_fallback_image` and dispatch `read_full_raw` / `read_full_unmapped_raw`. Do not cache raw and logical under the same key.
- **Compatibility:** Fixes silent wrong dtype/values for `read(..., raw_scale=True, return_header=True)` and other fallback cases.
- **Tests:** uint16 and BSCALE≠1 images: `read(raw_scale=True, return_header=True)` matches `read_tensor(raw_scale=True)` dtype and codes.

### IO-004 — HDU rewrite decompresses CompImage and leaves stale Z\* cards

- **Severity:** HIGH
- **Category:** HDU rewrite vs setkey, correctness
- **Files+lines:**
  - `src/torchfits/_io_engine/_hdu_rewrite.py` 27–45 (`_detach_hdus_for_rewrite` materializes tensors), 303–362 (`replace_hdu` preserves header but only drops BSCALE/BZERO/DATASUM/CHECKSUM), 242–267 (full-file rewrite)
  - `src/torchfits/_io_engine/_write_helpers.py` 141–152 (uncompressed image dict does not strip Z\*)
- **Description:** `insert_hdu` / `replace_hdu` / `delete_hdu` always rewrite the file via a temp FITS. Compressed image HDUs are decompressed through `TensorHDU.to_tensor`. `replace_hdu(path, 1, new_tensor)` on a Rice file preserves the old header including `ZIMAGE`/`ZCMPTYPE`/`ZBITPIX` onto an uncompressed `IMAGE` HDU.
- **Why:** Playbook already forbids rewrite for `setkey` because it decompresses CompImage and leaves stale Z\* cards. The public HDU-surgery APIs still do exactly that. Downstream tools that key off `ZIMAGE` will mis-identify the HDU.
- **Evidence:** Rice write: HDU1 `XTENSION=BINTABLE`, `ZIMAGE=True`. After `replace_hdu(..., ones*7)`: HDU1 is `ImageHDU` with `ZIMAGE='T'`, `ZCMPTYPE='RICE_1'`, `ZBITPIX='-32'` (string), pixels mean 7.0 (uncompressed).
- **Fix:** On preserve-header replace, also drop ZIMAGE/ZCMPTYPE/ZBITPIX/ZNAXIS\*/ZTILE\*/ZQUANTIZ/ZBLANK/ZHECKSUM/ZDATASUM (reuse `_sanitize_header_for_compressed_write` skip sets). Prefer CFITSIO in-place HDU insert/delete where possible; if rewrite stays, default `compress=` to the file’s existing algorithm or document that surgery is uncompressed.
- **Compatibility:** Output of `replace_hdu` on CompImage files changes (no lying Z\* cards; possibly still uncompressed unless `compress=` is set).
- **Tests:** Rice file → `replace_hdu` → astropy class is `ImageHDU` (or CompImageHDU if recompressed) and `ZIMAGE` absent unless still compressed; pixel values match; sibling HDUs unchanged.

### IO-005 — HDU mutation drops checksums; `verify` still `ok=True`

- **Severity:** HIGH
- **Category:** checksum
- **Files+lines:**
  - `src/torchfits/_io_engine/_hdu_rewrite.py` 353–355
  - `src/torchfits/_io_engine/checksum_api.py` 44–52
  - `src/torchfits/_io_engine/write_api.py` 92–98, 242–243
  - `src/torchfits/io.py` 494–501
- **Description:** `write(..., checksum=True)` stamps every HDU. `replace_hdu` / rewrite strips DATASUM/CHECKSUM and does not recompute. `verify_checksums` maps “keywords absent” to `ok=True`, `status="no_checksums"`.
- **Why:** Archive ingest that checks `result["ok"]` will accept a file whose stamps were silently removed by HDU surgery. `write_checksums(path)` defaults to `hdu=0` only, so a table-only restamp is easy to get wrong.
- **Evidence:** `write(..., checksum=True)` → `status=ok`. `replace_hdu` → `status=no_checksums`, `ok=True`, `DATASUM` absent. Docs (`docs/changelog.md`, CLI) treat missing keywords as success by design.
- **Fix:** After rewrite, either recompute checksums when the input had them, or add `checksum=` to insert/replace/delete. Separate `ok` from `present` (`ok` false when missing if the caller asked for verification). Keep `status="no_checksums"` for CLI.
- **Compatibility:** Tightening `ok` is breaking for anyone using `ok` as “not corrupt”. Safer: add `present: bool` and deprecate using `ok` for missing.
- **Tests:** checksum-stamped file → replace_hdu → either stamps valid or `ok is False` / `present is False` with a loud status; `write_checksums` on a multi-HDU file documents per-HDU vs all-HDU.

### IO-006 — `read(hdu=[...], mmap=False)` ignores mmap

- **Severity:** HIGH
- **Category:** mmap vs CFITSIO, API inconsistency
- **Files+lines:**
  - `src/torchfits/_io_engine/_read_pipeline.py` 544–551 (`read_hdus_batch(path, list(hdu))` 2-arg; default `use_mmap=true`)
  - `src/torchfits/_io_engine/image.py` 131 (`read_hdus` *does* pass mmap)
  - `src/torchfits/cpp_src/fits_bindings.cpp` 2116–2126
- **Description:** `torchfits.read_hdus(path, hdus, mmap=False)` passes mmap through. `torchfits.read(path, hdu=[0,1], mmap=False)` calls the 2-arg binding first, which succeeds with **mmap=True**. The `TypeError` fallback that would pass mmap never runs.
- **Why:** Two public APIs that look equivalent disagree on mmap. mmap vs CFITSIO can differ on truncated files, BLANK (IO-001), and integer vs float promotion for odd scales.
- **Evidence:** Binding default `use_mmap=true`; Python 2-arg call is first. Float smoke test cannot observe mmap, but the control flow is unconditional.
- **Fix:** Always call `read_hdus_batch(path, hdus, effective_mmap)` with `resolve_image_mmap` (or at least the caller’s bool). Delete the 2-arg try.
- **Compatibility:** `mmap=False` on list-of-HDU `read` starts meaning what it says.
- **Tests:** Patch `read_hdus_batch` to record `use_mmap`; `read(path, hdu=[0], mmap=False)` must pass False. Same for `mmap="auto"` vs compressed.

### IO-007 — Batch path list reads always mmap

- **Severity:** HIGH
- **Category:** mmap vs CFITSIO, API inconsistency
- **Files+lines:**
  - `src/torchfits/_io_engine/_read_pipeline.py` 458–468 (`mmap is not False` → `read_images_batch`)
  - `src/torchfits/_io_engine/batch.py` 37–44
  - `src/torchfits/cpp_src/fits_bindings.cpp` 382–394 (`read_tensor(hdu)` default mmap true); 2104–2114 (no mmap argument)
- **Description:** `read([p1,p2], mmap="auto")` uses `read_images_batch`, which has **no mmap parameter** and always `FITSFile::read_tensor(hdu)` (mmap true). Cold-nommap / compressed-auto policy is skipped. `read_batch()` does not even accept `mmap`. `mmap=False` on `read(list)` correctly skips the batch C++ path.
- **Why:** `mmap="auto"` is documented as a policy, not “always mmap” for list inputs.
- **Evidence:** C++ `read_images_batch` has no `use_mmap`; Python batch wrapper never passes one.
- **Fix:** Add mmap to `read_images_batch` or do not use it unless every path resolves to mmap=True. Give `read_batch` an explicit mmap.
- **Compatibility:** List/`read_batch` may switch some files to CFITSIO (cold nommap / compressed). Values should match; performance may change.
- **Tests:** Compressed + list input must not take the mmap-only batch path; `read([p], mmap=False)` never calls `read_images_batch`.

### IO-008 — `quantize=` silently ignored for compressed HDUList writes

- **Severity:** HIGH
- **Category:** quantize/scale semantics
- **Files+lines:** `src/torchfits/_io_engine/write_api.py` 186–230 vs 295–300
- **Description:** Uncompressed multi-HDU sequences raise if `quantize is not None`. The `compress=` branch applies quantize only for a single Tensor/ndarray; `HDUList` / list / dict-with-`data` skip it with no error. CFITSIO then applies its own lossy default quantization on float Rice.
- **Why:** Callers who pass both `compress` and `quantize="robust"` on an HDUList believe they got robust int16; they got CFITSIO’s compressor quantizer (`ZBITPIX=-32`, no BSCALE from torchfits).
- **Evidence:** `write(HDUList, compress="RICE_1", quantize="robust")` → `ZBITPIX=-32`, `BSCALE=None`. `write(tensor, compress="RICE_1", quantize="robust")` → `ZBITPIX=16` with BSCALE set.
- **Fix:** Either apply `_apply_image_quantize` per image HDU in the compress path, or raise the same ValueError as the uncompressed sequence path.
- **Compatibility:** Raising is breaking for anyone accidentally passing both; applying quantize changes bytes. Raising is the honest freeze.
- **Tests:** HDUList+compress+quantize raises *or* ZBITPIX=16+BSCALE match the tensor path.

### IO-009 — `keep_zero` quantize turns NaN into 0 with no BLANK

- **Severity:** HIGH
- **Category:** quantize/scale semantics
- **Files+lines:** `src/torchfits/_io_engine/quantize.py` 243–265, 307 (`if not keep_zero` for BLANK)
- **Description:** `keep_zero` clips non-finite samples to 0.0 and never sets `blank_code`. Weight/mask intent is documented, but NaNs become valid zeros.
- **Why:** Same masquerade IO-001 was written to prevent, on the keep-zero path.
- **Evidence:** `quantize={..., "keep_zero": True}` on a tensor with a NaN → no `BLANK`; storage code 0; logical read 0.0.
- **Fix:** Still emit BLANK/TNULL for non-finite samples when `keep_zero=True`, or reject non-finite input.
- **Compatibility:** NaN positions become NaN (or an error) instead of 0.
- **Tests:** keep_zero + NaN → BLANK set and read-back NaN (once IO-001 is fixed), or TypeError.

### IO-010 — `cfitsio_base_path` / `has_cfitsio_filter` are not CFITSIO’s rule

- **Severity:** HIGH
- **Category:** path SSRF / path parsing (correctness)
- **Files+lines:**
  - `src/torchfits/_io_engine/paths.py` 48–67
  - used in `hdu_api.py` 139–145, `table_streaming.py` 46–49, CLI cutout/compress
  - C++ `security.h` 16–21 (last path component, must end in `]`)
- **Description:** Python strips at the **first** `[` anywhere. CFITSIO (and C++) only treat a trailing `[...]` on the final component as extended syntax. A directory named `[data]` is a filter to Python and a real path to C++.
- **Why:** `os.path.exists(cfitsio_base_path(path))` can test the parent directory (which exists) instead of the file; `has_cfitsio_filter` makes CLI cutout treat a literal-bracket dir as a section. Existence checks become tautologies; error messages cite the wrong path.
- **Evidence:** `cfitsio_base_path(".../[data]/file.fits")` → `".../"`; `has_cfitsio_filter` True. `tests/test_security.py::test_read_path_with_literal_bracket_in_directory` already requires C++ diskfile routing for this shape.
- **Fix:** Port `has_cfitsio_extended_filename_syntax` to Python; `cfitsio_base_path` should split only when that predicate is true.
- **Compatibility:** Fixes false filter detection; CLI cutout behavior on bracket dirs changes toward “it’s a filename”.
- **Tests:** Reuse the `[data]/file.fits` fixture for `cfitsio_base_path`, `open()`, `stream_table` exists-check, `has_cfitsio_filter`.

### IO-011 — Header cards parser drops Fortran `D` exponents

- **Severity:** HIGH
- **Category:** correctness, API inconsistency
- **Files+lines:**
  - `src/torchfits/header_parser.py` 345–351 (`_parse_card` uses `float(pv)` / `int(pv)`)
  - `src/torchfits/header_parser.py` 461–479 (`_parse_fits_number` handles `D` — used only by the dict parser)
  - `src/torchfits/_io_engine/hdu_api.py` 314–316 (`get_header` / `read_header` uses **cards**)
- **Description:** `fast_parse_header` (dict) accepts `1.5D-3`. `fast_parse_header_cards` (what `read_header` uses) stores `"1.5D-3"` as a **str**. Image-meta Python fallback then `float(BSCALE)` fails and silently uses 1.0 (`image_meta.py` 53–60). C++ `fits_read_key(TDOUBLE)` still parses D, so pixel reads can be correct while the public Header is not.
- **Why:** Users doing `float(header["BSCALE"])` or WCS from `read_header` get a string or a wrong default scale in Python-side policy caches.
- **Evidence:** Card `BSCALE  = 1.5D-3`: dict parser → `0.0015` float; cards parser → `'1.5D-3'` str.
- **Fix:** Call `_parse_fits_number` from `_parse_card`.
- **Compatibility:** `read_header` values that were strings become floats (correct).
- **Tests:** `read_header` on a file with `BSCALE=1.5D-3`; type float, value 0.0015. Same for TSCAL.

### IO-012 — `to_arrow` flattens N-D numeric columns

- **Severity:** HIGH
- **Category:** API inconsistency, correctness
- **Files+lines:**
  - `src/torchfits/_tensor_buffer.py` 31–59 (docstring says 1-D; uses `numel()`)
  - `src/torchfits/interop.py` 162–168
- **Description:** Vector columns `(N, K)` become an Arrow array of length `N*K`. `Table.from_arrays` then reports `N*K` rows with no error if it is the only column.
- **Why:** Table readers keep genuine vector columns as 2-D tensors (`_squeeze_scalar_columns` only squeezes `(N,1)`). Interop silently reshapes science columns.
- **Evidence:** `to_arrow({"x": arange(6).reshape(3,2)})` → 6 rows, values `[0..5]`.
- **Fix:** If `dim()>1`, emit a `list`/`fixed_size_list` array per row, or raise. Do not flatten.
- **Compatibility:** Breaking for anyone who relied on the flatten; correct for FITS vector columns.
- **Tests:** `(N,2)` float column → N rows of pairs; 1-D still zero-copy.

### IO-013 — CFITSIO still resolves public HTTP after the Python SSRF check

- **Severity:** HIGH
- **Category:** path SSRF
- **Files+lines:**
  - `src/torchfits/http_util.py` 66–99, 178–195
  - `src/torchfits/_io_engine/paths.py` 42–45
  - C++ `fits_open_file` on URLs (`fits_detail.h` 340–347)
- **Description:** `guard_cfitsio_remote_path` fail-closes private/loopback via `getaddrinfo` at guard time, then CFITSIO performs its **own** DNS/connect. TTL=0 rebinding (public A record at check, private at open) is not pinned. Python Range (`http_open` + `ValidatingRedirectHandler`) re-checks redirects; the CFITSIO full-open path does not.
- **Why:** The advertised model is “public http(s)/ftp still CFITSIO; private blocked.” Residual SSRF is inherent unless IPs are pinned or HTTP is fetched only in Python.
- **Evidence:** Tests block `127.0.0.1` / `10.0.0.1` / `ftp://192.168.1.1` at the façade (`tests/test_security.py`). No pin/TTL test. On this host, `http://[::ffff:127.0.0.1]/` *is* classified internal (`is_loopback=True`); do not treat mapped-IPv6 as a current bypass here.
- **Fix:** For 1.x, document the residual and keep fail-closed DNS errors. For a freeze: fetch via Python (Range/full) after pinning resolved IPs, or pass a numeric URL to CFITSIO. Do not add a “allow private” env without an explicit name in `src/` and docs.
- **Compatibility:** Pinning can break split-horizon CDNs; document.
- **Tests:** Keep existing private-URL tests. Optional: fake DNS that changes between `getaddrinfo` and connect (if a harness exists).

### IO-014 — Remote URLs never invalidate Python caches

- **Severity:** MEDIUM
- **Category:** correctness
- **Files+lines:**
  - `src/torchfits/_io_engine/caches.py` 128–135 (`path_signature` → `None` on `stat` failure)
  - `src/torchfits/_io_engine/caches.py` 181–190 (stale only if **both** sigs non-None and differ)
  - `src/torchfits/_io_engine/hdu_api.py` 118–121 (`sig is None or cached_sig is None or equal` → hit)
- **Description:** HTTP/FTP/vos paths cannot `stat`. Image-meta, auto-mmap, auto-HDU, header-cards, and data caches then live forever for that URL string.
- **Why:** A replaced remote object under the same URL keeps serving the first header/HDU choice/pixels.
- **Evidence:** Code: `os.stat` except → `None`; hit condition treats `None` as “not stale”.
- **Fix:** Do not cache when `path_signature` is None, or key by URL plus HTTP validators if a fetch layer exists.
- **Compatibility:** More remote I/O; correctness over hit-rate.
- **Tests:** Mock `path_signature` None; second `read_header` must re-enter C++ (spy).

### IO-015 — Silent `except Exception` on table mmap routing

- **Severity:** MEDIUM
- **Category:** silent except Exception, mmap vs CFITSIO
- **Files+lines:**
  - `src/torchfits/_io_engine/table_streaming.py` 74–79, 108–109
  - `src/torchfits/_io_engine/table_api.py` 163–166, 239–288, 377–378
  - `src/torchfits/_io_engine/_read_pipeline_fallback.py` 293–314
- **Description:** ASCII detection, scaled-column detection, filtered reads, and the thin table path swallow **all** exceptions. ASCII/scaled failures then attempt mmap; C++ raises (`ASCII tables are not supported for mmap` / `Scaled columns not supported for mmap`) instead of falling back in the same function. `read_table` then falls through to `read()` or `RuntimeError` with the original cause discarded.
- **Why:** Not always silent corruption (C++ often throws), but it *is* silent loss of the reason and can skip the buffered path that would have worked (`table_api` `except Exception: pass` then unified read).
- **Evidence:** Code; streaming scaled-column probe is the same pattern that Round-N already special-cased for XTENSION=TABLE.
- **Fix:** Catch `(RuntimeError, OSError, ValueError)` only; on ASCII/scale probe failure, force `mmap=False` rather than “assume not ASCII”. Log at debug with `exc_info`.
- **Compatibility:** More errors surface; some tables that currently fail mmap-then-mystery may succeed on buffered retry.
- **Tests:** ASCII + `mmap=True` stream yields data; TSCAL≠1 stream yields physical values; a real I/O error is not swallowed into `data is None`.

### IO-016 — Fallback wraps `ValueError` as `RuntimeError`

- **Severity:** MEDIUM
- **Category:** error messages, API inconsistency
- **Files+lines:** `src/torchfits/_io_engine/_read_pipeline_fallback.py` 107, 179–190
- **Description:** `HDU 'FOO' not found` is `ValueError` internally, then wrapped as `RuntimeError("Failed to read FITS file...")` (and the inner table failure at 179–180 omits `from exc`). Fast paths raise `ValueError` directly.
- **Why:** Callers cannot `except ValueError` uniformly; messages double-wrap.
- **Fix:** Let `ValueError`/`FileNotFoundError`/`HttpBlockedError` propagate; wrap only CFITSIO `RuntimeError`.
- **Compatibility:** Exception type changes (usually toward the documented one).
- **Tests:** Missing EXTNAME via `read()` is `ValueError` not `RuntimeError`.

### IO-017 — `read_batch_info` does not match its docstring

- **Severity:** MEDIUM
- **Category:** API inconsistency
- **Files+lines:**
  - `src/torchfits/io.py` 444–446
  - `src/torchfits/_io_engine/batch.py` 70–89
- **Description:** Docstring: “Inspect shape and dtype consistency across files.” Implementation: `{num_files, existing_files}` via `os.path.exists`, swallowing exists() errors.
- **Why:** Public name/doc promise a planner; they return a count.
- **Fix:** Rename or actually probe shape/dtype (expensive). At freeze, make the docstring match the dict.
- **Compatibility:** Doc-only is safe; adding fields is additive.
- **Tests:** Docs-integrity / a unit test of the returned keys.

### IO-018 — Default HDU index splits image vs table façades

- **Severity:** MEDIUM
- **Category:** API inconsistency
- **Files+lines:**
  - `io.py` `read`/`read_tensor`/`read_header`/`read_keys`/`read_shape` hdu default **0**
  - `io.py` `read_nrows`/`read_colnames`/`read_table_info`/`open_table_reader` default **1**
  - `table_api.py` 195–199 rejects `None`/`"auto"`; `read()` accepts them
- **Description:** Intentional (primary vs first extension) but easy to stamp the wrong HDU with `write_checksums(path)` (IO-005) or `read_keys(path, ["NAXIS2"])` on a table file (reads primary).
- **Why:** Major-release users coming from astropy (`hdul[1]` tables) will mix APIs.
- **Fix:** Docs table of defaults; consider `write_checksums(path, hdu="all")`.
- **Compatibility:** Changing defaults is breaking; docs-only is not.
- **Tests:** Docs-contract already fences some signatures; add a matrix test of defaults on a 2-HDU file.

### IO-019 — MPS `to_device` silently downcasts float64/complex128

- **Severity:** MEDIUM
- **Category:** device/dtype
- **Files+lines:** `src/torchfits/_io_engine/device.py` 9–21, 40–45, 53–59
- **Description:** `validate_device` error text omits `mps:N` (still accepted). `to_device` / `batch_to_device` convert float64→float32 and complex128→complex64 on MPS with no warning.
- **Why:** Science values change precision based on `device=` alone.
- **Fix:** Warn once or require `dtype=` for downcast; fix the error string.
- **Compatibility:** Warning is additive.
- **Tests:** MPS skip-if-missing: float64 image read warns and is float32.

### IO-020 — `TableReaderHandle.close` does not close the native handle

- **Severity:** MEDIUM
- **Category:** correctness
- **Files+lines:** `src/torchfits/_io_engine/table_reader_api.py` 53–55 vs `subset.py` 150–152
- **Description:** Sets `_reader = None` and relies on nanobind/`~TableReader` to `fits_close_file`. `SubsetReader.close()` calls C++ `close()`. Until GC, CFITSIO may keep the file READONLY (blocks rewrite).
- **Why:** Asymmetric with SubsetReader; write-after-read-without-GC is a known CFITSIO pain.
- **Fix:** If the binding exposes `close`, call it; else `del` is fine but document context-manager use.
- **Compatibility:** None if destructor already closes.
- **Tests:** `open_table_reader` → `close()` → `replace_hdu`/`write(overwrite=True)` without `gc.collect()`.

### IO-021 — HTTP Range cutouts skip unsigned/scaled HDUs (good) but disagree on empty-box / 3-D

- **Severity:** MEDIUM
- **Category:** mmap vs CFITSIO, correctness
- **Files+lines:**
  - `src/torchfits/_io_engine/http_subset.py` 123–150, 187–192
  - `src/torchfits/cpp_src/fits_file.cpp` 503–513, 1072–1081
- **Description:** Range path refuses BSCALE/BZERO≠identity and NAXIS≠2 (fallback to full fetch + CFITSIO). Empty boxes: HTTP `empty((max(0,y2-y1), max(0,x2-x1)))`; C++ preserves extra axes and the non-degenerate 2-D side. HTTP does not apply BLANK (IO-001) and does not byteswap BITPIX=8 (correct).
- **Why:** Unsigned 16-bit surveys (BZERO=32768) always materialize the whole file for cutouts. Empty/ND shapes differ by backend.
- **Fix:** Document; optionally allow unsigned convention on the Range path (apply +32768 like mmap). Align empty-box shapes with C++.
- **Compatibility:** Allowing unsigned Range is additive performance; shape alignment is a small break for empty cutouts.
- **Tests:** HTTP unsigned 16-bit cutout equals local SubsetReader; empty width preserves height.

### IO-022 — `fast_parse_header` dict collapses COMMENT/HISTORY

- **Severity:** MEDIUM
- **Category:** correctness, API inconsistency
- **Files+lines:** `src/torchfits/header_parser.py` 179–181, 230–231 vs `fast_parse_header_cards` 547
- **Description:** Dict parser last-COMMENT wins. Cards parser keeps every card. `read_header` uses cards (`Header` mapping still last-wins for COMMENT but `.cards` keeps all). `read_header_fast` in `hdu_api.py` uses the **dict** parser for handle-based headers.
- **Why:** Autodetect compressed-image (`ZIMAGE` from dict) is fine; dumping COMMENT history via the fast handle path is not.
- **Fix:** Prefer cards everywhere, or document that `read_header_fast` is keyword-only.
- **Compatibility:** Dict consumers of handle headers may see more structure if switched to cards.
- **Tests:** Two COMMENT cards: `Header.cards` length 2 via `read_header`; note handle fast path.

### IO-023 — Table schema skips unnamed columns and is TTYPE-case-sensitive

- **Severity:** MEDIUM
- **Category:** correctness
- **Files+lines:** `src/torchfits/fits_schema.py` 105–124, 155–163, 254–271
- **Description:** `_iter_tfields_indexed` `continue`s when `TTYPE{i}` is missing (legal FITS). `selected` / unsigned-dtype maps are exact string match. Streaming mmap “unsigned” heuristic also treats `TZERO=-32768` as unsigned (`table_streaming.py` 100–104), which C++ does **not** (`is_unsigned_short_offset` is +32768 only).
- **Why:** Unnamed columns disappear from schema/VLA/bit maps. `-32768` TZERO may be routed to mmap then rejected by C++, or (if C++ unsigned check is skipped) mis-decoded.
- **Fix:** Yield `TTYPEn` default `Column{n}`; casefold option; align unsigned detection with C++.
- **Compatibility:** Extra columns appear in schema; mmap routing for exotic TZERO may start using the buffered path.
- **Tests:** Header with missing TTYPE2; TZERO=-32768 column reads physical values.

### IO-024 — CacheManager / `configure_cache` are live-looking no-ops

- **Severity:** MEDIUM
- **Category:** API inconsistency
- **Files+lines:** `src/torchfits/cache.py` 223–236, 268–276, 383–398
- **Description:** `CacheManager.clear()` clears C++ `clear_file_cache` but **not** Python `file_cache` unless the caller also hits `clear_cache()`. `configure_cache` / `configure_cpp_cache` warn but still replace the global manager with knobs that do not size any live cache (`max_files` unused by `_io_engine.caches`).
- **Why:** Freeze surface still advertises HPC/cloud memory fractions that do not bind.
- **Fix:** `CacheManager.clear` → `clear_file_cache()`; consider removing `max_files` from the public config or wiring `cache_capacity`.
- **Compatibility:** Deprecation already in place; making `clear` actually clear Python LRUs is a fix.
- **Tests:** Fill `file_cache` via `return_header` reads; `CacheManager().clear()` empties it.

### IO-025 — `stream_table` error text says `batch_size`

- **Severity:** LOW
- **Category:** error messages
- **Files+lines:** `src/torchfits/_io_engine/table_streaming.py` 42–43
- **Description:** Parameter is `chunk_rows`; error says `batch_size must be > 0`.
- **Fix:** Use the real name.
- **Compatibility:** Message-only.
- **Tests:** `chunk_rows=0` match.

### IO-026 — `decode_byte_tensor` copies entire storage and ignores decode errors

- **Severity:** LOW
- **Category:** correctness
- **Files+lines:** `src/torchfits/_string_decode.py` 40–53
- **Description:** `bytes(storage)` materializes the full untyped storage (not just the view). `errors="ignore"` drops invalid bytes. No dtype/ndim guard (float32 2-D “decodes” to garbage).
- **Why:** `to_arrow(..., decode_bytes=True)` is guarded to uint8 2-D; direct helper is not. Large storages with a small view can OOM.
- **Fix:** `tensor.numpy().tobytes()` / `memoryview` on the view; require uint8 2-D; `errors="replace"` or strict.
- **Compatibility:** Stricter dtype is breaking for misuse; view copy is safer.
- **Tests:** Non-contiguous uint8 view; reject float input.

### IO-027 — Broad `except Exception` on cache/handle close and bz2 probe

- **Severity:** LOW
- **Category:** silent except Exception
- **Files+lines:**
  - `paths.py` 24–29 (`has_bz2_support` → False)
  - `caches.py` 130–133, 382–414, 426–431
  - `subset.py` 60–63
  - `image_meta.py` 53–67, 84–103, 189–211
  - `hdu_api.py` 71–76, 105, 188–193, 333
  - `batch.py` 86–87
- **Description:** Close/stat/meta probes swallow `Exception`. Fail-closed bz2 (treat as unsupported) is reasonable. Caching `get_image_meta` **None** after a transient failure is less so (pairs with IO-014).
- **Fix:** Narrow to `OSError`/`RuntimeError`; do not cache None without a short TTL.
- **Compatibility:** More warnings; fewer sticky None metas.
- **Tests:** `get_image_meta` after a failed probe retries.

### IO-028 — Dead / misleading write and read helpers

- **Severity:** CLEANUP
- **Category:** API inconsistency
- **Files+lines:**
  - `_write_helpers.py` 63–72 (second “Pandas” branch unreachable; Polars/Pandas share `columns`+`to_dict`)
  - `_read_pipeline.py` 595–609 (`read_scaled_cpu_fast` unused by `_read_cpu_fast_path`)
  - `_read_pipeline.py` 712–718 (`scale_on_device=False` still `read_full`)
  - `header_parser.py` 24 (`_KEYWORD_PATTERN` unused)
  - `options.py` 22–24 (`handle_cache_capacity` ignored)
  - `table_reader_api.py` close vs C++ destructor (IO-020)
- **Description:** Leftovers from the A2 / Option A refactors. `scale_on_device=False` does not skip scaling; it is a path-selection leftover.
- **Fix:** Delete dead branches or make `scale_on_device=False` an error/no-op documented as such.
- **Compatibility:** Removing ignored kwargs needs a deprecation (already started for handle cache).
- **Tests:** Existing `test_scale_on_device.py` / `test_deep_review_wave5.py`.

### IO-029 — `logging.py` is sound

- **Severity:** CLEANUP (positive)
- **Category:** API
- **Files+lines:** `src/torchfits/logging.py` 1–20
- **Description:** NullHandler on `torchfits`; re-exports stdlib levels. No I/O side effects.
- **Fix:** none.
- **Compatibility:** n/a
- **Tests:** n/a

### IO-030 — `vos_uri.py` leaves `vos://` unmapped and unguarded

- **Severity:** LOW
- **Category:** path SSRF
- **Files+lines:** `src/torchfits/vos_uri.py` 8–39; `http_util.py` 184 (vos/vault untouched)
- **Description:** Short `vos:`/`vault:` map to CADC vault; full `vos://...` is returned unchanged (any host). Guards do not SSRF-check vos (delegated to the vos client).
- **Why:** Freeze should say vos is a separate trust domain.
- **Fix:** Docs only unless vos client is in-process HTTP.
- **Compatibility:** n/a
- **Tests:** `normalize_vos_uri("vos://other/x")` unchanged; `is_vos_path("vos://")` False.

---

## File ledger

| File | Note |
|---|---|
| `_io_engine/__init__.py` | Package marker only. |
| `_io_engine/paths.py` | SSRF delegates to `http_util`; `cfitsio_base_path`/`has_cfitsio_filter` disagree with C++ (IO-010); bz2 probe swallows Exception. |
| `_io_engine/checksum_api.py` | CFITSIO `ffpcks`/`ffvcks` wrapper; missing keywords → `ok=True` (IO-005); hdu must be int ≥0. |
| `_io_engine/quantize.py` | Robust percentile pack + BLANK=-32767; keep_zero skips BLANK (IO-009); large-N percentile is strided (~128k). |
| `_io_engine/device.py` | cpu/cuda/mps; MPS silent f64/c128 downcast (IO-019); error string omits `mps:N`. |
| `_io_engine/options.py` | `ReadOptions`; `handle_cache_capacity` ignored; `scale_on_device`/`raw_scale` flags. |
| `_io_engine/batch.py` | SSRF then `read_images_batch` (no mmap); per-file fallback; `get_batch_info` is exists-count (IO-017). |
| `_io_engine/http_subset.py` | Uncompressed 2-D Range cutout; rejects scaled/compressed/ND; endian swap; no BLANK. |
| `_io_engine/subset.py` | HTTP Range then CFITSIO `SubsetReader`; named HDU resolve; close swallows Exception; HTTP `hdu` property -1 for names. |
| `_io_engine/image.py` | `read_image`/`read_hdus` guard+mmap bool; raw_scale dispatch; header fallback `except Exception`. |
| `_io_engine/image_meta.py` | Skinny then full header; BSCALE parse failures → 1.0; caches None; cold-nommap for bitpix 16/32/-32. |
| `_io_engine/table_reader_api.py` | Persistent `TableReader`; no `where=`; `close()` drops ref only (IO-020). |
| `_io_engine/table_streaming.py` | Chunked rows; ASCII/scale probes swallow Exception (IO-015); mmap default False vs table auto True. |
| `_io_engine/table_api.py` | Thin `read_fits_table_rows`; where= mask; mmap auto→True; broad except then unified read. |
| `_io_engine/hdu_api.py` | Autodetect, skinny keys/shape/type, `get_header` cards cache; EXTNAME exact match; RuntimeError→OSError on dict fallback. |
| `_io_engine/_hdu_rewrite.py` | Full-file atomic rewrite; CompImage/Z\* (IO-004); checksum strip (IO-005). |
| `_io_engine/_write_helpers.py` | Unsigned image/table storage, quantize+BLANK/TNULL, VLA dtype check, header-card replay via `fits_delete_key` sibling. |
| `_io_engine/write_api.py` | `write`/`checksum`/`quantize`/`compress`; HDUList+compress ignores quantize (IO-008); wraps most errors as RuntimeError. |
| `_io_engine/caches.py` | LRU data/meta/header; signature miss for HTTP (IO-014); Header cloned as dict (IO-002); handle cache removed. |
| `_io_engine/_read_pipeline.py` | Unified read; list/HDU-batch mmap bugs (IO-006/007); raw_scale not passed to fallback (IO-003); unused scaled-CPU helper. |
| `_io_engine/_read_pipeline_fallback.py` | Cache + handle read; always scaled `read_full`; ASCII tagged BINARY_TABLE; exception wrapping (IO-016). |
| `io.py` | Public façade; hdu default 0 vs 1 split (IO-018); `read_batch` has no mmap; checksum helpers. |
| `cache.py` | Disk roots + deprecated native knobs (IO-024); `clear_cache` does clear Python LRUs. |
| `http_util.py` | Fail-closed DNS SSRF; credential strip on redirect; CFITSIO schemes http/https/ftp only; residual rebinding (IO-013). |
| `vos_uri.py` | CADC vault map; `vos://` passthrough (IO-030); no whitespace. |
| `logging.py` | NullHandler; clean (IO-029). |
| `fits_schema.py` | TFORM/VLA/bit/unsigned maps; unnamed TTYPE skipped; unsigned = +32768/+2^31 only. |
| `header_parser.py` | LONGSTRN/CONTINUE/HIERARCH; dict vs cards D-exponent (IO-011) and COMMENT collapse (IO-022). |
| `interop.py` | Arrow/pandas/polars/astropy; VLA policy names differ; 2-D flatten (IO-012). |
| `_tensor_buffer.py` | NumPy buffer → Arrow; 1-D docstring vs `numel()` flatten; no uint16. |
| `_string_decode.py` | uint8 (N,width) → list[str]; full-storage copy; `errors="ignore"` (IO-026). |

---

## Hypotheses checked that did not land as defects

- **IPv4-mapped loopback SSRF:** `http://[::ffff:127.0.0.1]/` is blocked on this Python (`is_loopback`/`is_private` True).
- **Unsigned image mmap vs CFITSIO:** both apply +32768 / +2^31; HTTP Range refuses those HDUs and falls back (conservative).
- **`setkey` rewrite:** not in this file set; `_delete_header_key_if_supported` uses `fits_delete_key`. HDU surgery still rewrites (IO-004).
- **`verify_checksums` mixed data/hdu status:** any combo other than (0,0) or (1,1) is `fail`.
- **`guard_fits_path` on listed façades:** present on read/write/subset/batch/checksum/table reader/open; C++ `sh://` / `|` still enforced in `security.h`.
- **`logging.py` attaching a StreamHandler:** it does not.

---

## Suggested freeze order

1. IO-001 (BLANK on read) + IO-009 (keep_zero NaN) — science.
2. IO-002 + IO-003 + IO-006/007 — `read()` contract.
3. IO-004 + IO-005 — HDU mutation / checksums.
4. IO-008 + IO-011 + IO-012 + IO-010 — write/header/interop/paths.
5. IO-013 document-or-pin SSRF residual before calling the HTTP surface “safe”.
