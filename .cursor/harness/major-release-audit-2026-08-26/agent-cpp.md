# C++ / native-boundary audit — 2026-08-26

Scope: every git-tracked file under `src/torchfits/cpp_src/` (`.cpp` `.h` `CMakeLists.txt`), plus `src/torchfits/_cpp.py`, `src/torchfits/cpp.py`, and `tests/cpp/`. Review-only; no fixes.

Method: full reads of all files ≤ ~1100 lines; `table_reader.h` (2762) and `fits_bindings.cpp` (2475) read in sequential chunks covering every function. Patterns grepped: GIL, mmap, `from_blob`, `evict_cached_reader`, `_C` exports, `get_fptr_from_python_object`. No tests were executed. Vendored CFITSIO sources are not in this tree.

Hypotheses from prior notes were treated as claims to verify in current code, not as findings.

---

## Files reviewed

| File | Lines | Note |
|---|---|---|
| `src/torchfits/cpp_src/CMakeLists.txt` | 620 | Torch ABI pin, dummy CUDA targets, vendored CFITSIO patches, ZLIB symlink hack, `_C` visibility |
| `src/torchfits/cpp_src/bindings.cpp` | 36 | ABI import check + `HAS_BZIP2`; tiny |
| `src/torchfits/cpp_src/cache.h` | 54 | No-op cache API + move-only `FitsHandleGuard` |
| `src/torchfits/cpp_src/cache.cpp` | 34 | Stubs; `get_or_open_cached` throws |
| `src/torchfits/cpp_src/security.h` | 66 | Pipe/`sh://` guards; extended-filename `[` detector |
| `src/torchfits/cpp_src/hardware.h` | 75 | `MMapHandle` RAII; bswap re-exports; endian probe |
| `src/torchfits/cpp_src/hardware.cpp` | 61 | mmap ctor; `MAP_PRIVATE` even when writable |
| `src/torchfits/cpp_src/torch_compat.h` | 97 | `THPVariable_Wrap`, DLPack import, numpy bytearray alloc |
| `src/torchfits/cpp_src/torchfits_torch.h` | 38 | `torch/torch.h` fallback aliases |
| `src/torchfits/cpp_src/fits_rw.h` | 65 | Image/table entry decls; `open_fits_for_write` status-104 retry |
| `src/torchfits/cpp_src/fits_file.h` | 118 | `FITSFile` / `SubsetReader` surface |
| `src/torchfits/cpp_src/fits_file.cpp` | 1122 | Open/read/write image, header, subset mmap, compressed write |
| `src/torchfits/cpp_src/fits_detail.h` | 699 | SharedReadMeta, pread/mmap canonical image read, sanitizers |
| `src/torchfits/cpp_src/internal_utils.h` | 302 | SIMD bswap, env helpers, ndarray contig copy |
| `src/torchfits/cpp_src/table_types.h` | 79 | ColumnInfo / TableFilter; buffered-read env |
| `src/torchfits/cpp_src/table_ops.h` | 20 | Mutation decls (global namespace) |
| `src/torchfits/cpp_src/table_ops.cpp` | 1037 | write/append/insert/update/delete/rename/drop |
| `src/torchfits/cpp_src/table_reader.h` | 2762 | Analyze, CFITSIO/buffered/mmap/filter/VLA/update-mmap |
| `src/torchfits/cpp_src/table_bindings.cpp` | 528 | TableReader bind, TLS reader cache, path wrappers |
| `src/torchfits/cpp_src/fits_bindings.cpp` | 2475 | Image binds, write_table_hdu, metadata, cache stubs, `echo_tensor` |
| `src/torchfits/_cpp.py` | 187 | Allowlist + `guard_fits_path`; `__getattr__` leaks `_C` |
| `src/torchfits/cpp.py` | 32 | Deprecated alias; still forwards every `_C` name |
| `tests/cpp/test_bracket_detection.cpp` | 32 | Standalone asserts; not wired into CMake/pytest |

---

## Not fully read / not executed

- Vendored CFITSIO (`extern/cfitsio` is absent in this checkout). Gzip/diskfile driver behavior is inferred from CFITSIO contracts and from comments in `direct_io_ok()`, not from current CFITSIO source.
- SIMD shuffle masks in `internal_utils.h` were spot-checked against the usual 16/32/64-bit bswap patterns, not proven with a unit test on this machine.
- `write_table_hdu` status accumulation after the last `fits_update_key` was not exhaustively simulated for every schema key combination.
- No sanitizer run, no pytest, no `pixi run` rebuild. Findings are static.
- Python I/O façades were grepped only to see whether they mitigate a C++ hole (eviction, gzip, `_C` vs `_cpp`). They are out of scope except as evidence.

---

## Hypotheses checked

| Claim | Result |
|---|---|
| Shared `fitsfile*` LRU still on the hot path | **Disproved.** `cache.cpp` is no-ops; `get_or_open_cached` throws. Live cache is `SharedReadMeta` + TLS `TableReader`. |
| `sh://` only matched case-sensitively | **Disproved.** `security.h` folds ASCII case. |
| Extended-filename check is naive `find('[')` | **Disproved in C++.** `has_cfitsio_extended_filename_syntax` requires trailing `]` and `[` after last `/`. Python `paths.has_cfitsio_filter` is still naive (out of scope). |
| Image NAXIS product can wrap and under-allocate | **Mostly disproved on image paths** that call `checked_nelements_product`. Table `num_rows * repeat` / mmap extent math is not similarly checked. |
| Writable `MMapHandle` uses `MAP_PRIVATE` so table mmap updates never hit disk | **Disproved as a runtime bug.** `update_rows_mmap` maps with `MAP_SHARED` itself. The writable `MMapHandle(filename, true)` ctor is unused. |
| `delete_fits_table_rows` skips `evict_cached_reader` | **Confirmed in C++.** Python `invalidate_path_caches` still evicts around the public mutation façade. |
| `_cpp` allowlist keeps new `_C` symbols private | **Disproved.** `__getattr__` forwards every `_C` name. |

---

## Issues

### CPP-001

- **Severity:** HIGH
- **Category:** accidental public surface
- **Files:** `src/torchfits/_cpp.py` 168–187; `src/torchfits/cpp.py` 22–32; `src/torchfits/cpp_src/table_bindings.cpp` 249, 453–478; `src/torchfits/cpp_src/fits_bindings.cpp` 2466–2474
- **Description:** `_cpp.py` documents that new `_C` symbols stay private until added to `__all__`, then `__getattr__` returns `getattr(_C, name)` for every name. `__dir__` unions `dir(_C)`. `cpp.py` does the same after a deprecation warning.
- **Why it matters:** Anyone can reach `torchfits._cpp.echo_tensor`, `evict_cached_reader`, `open_fits_mmap_reader`, `read_fits_table_rows_mmap_from_reader`, `configure_cache`, `HAS_BZIP2`, `invalidate_file_cache`, etc. Path wrappers in `__all__` are bypassed by going through `__getattr__` or `import torchfits._C`. A major-release freeze cannot claim a closed native surface.
- **Evidence:** `__all__` does not include `echo_tensor`, `evict_cached_reader`, or `open_fits_mmap_reader`, but those are bound on `_C`. `__getattr__` has no allowlist. Package code already imports `_C` directly (`io.py`, `_table/mutation.py`, …), so the allowlist is not the in-tree boundary either.
- **Recommended fix:** Make `__getattr__` raise `AttributeError` for names not in `__all__`. Stop exporting debug/cache stubs (`echo_tensor`, no-op `configure_cache` / `clear_file_cache` / `get_cache_size`) or mark them explicitly private (`_echo_tensor`). Keep `HAS_BZIP2` if Python guards need it, but document it. For 1.0, treat `import torchfits._C` as unsupported and add an isolation test that `dir(torchfits._cpp) == sorted(__all__)` (plus dunders).
- **Compatibility:** Breaking for anyone poking `_C` / undocumented `_cpp` names. Compatible for `__all__` callers.
- **Tests needed:** Isolation test that names bound on `_C` but absent from `__all__` are not reachable via `_cpp` / `cpp`. Inventory of `_C.def` / `class_` vs `__all__`.

### CPP-002

- **Severity:** HIGH
- **Category:** Python/C++ boundary — GIL
- **Files:** `src/torchfits/cpp_src/table_bindings.cpp` 332–345, 341–348, 400–407; `src/torchfits/cpp_src/fits_bindings.cpp` 1364–1366
- **Description:** `read_fits_table_from_handle`, `read_fits_table_rows_from_handle`, and `read_fits_table_rows_numpy_from_handle` do `nb::gil_scoped_release` *before* `get_fptr_from_python_object(file_obj)`. That helper `nanobind::cast<FITSFile&>(obj)` and `get_fptr()` touch a live Python object with the GIL dropped.
- **Why it matters:** `nb::cast` inspects Python types/headers. Concurrent mutation or finalization of `file_obj` is a crash / use-after-free class bug, not a clean exception.
- **Evidence:** Contrast with `TableReader.__init__(file_obj)` (lines 222–225), which casts **with the GIL held**. Contrast with `read_fits_table_rows_mmap_from_reader` (463–476), which reads the capsule pointer before releasing the GIL.
- **Recommended fix:** Extract `fitsfile*` (and a C++ `FITSFile&` if needed) before `gil_scoped_release`. Do not call nanobind/Python APIs in the released region.
- **Compatibility:** Internal calling convention only.
- **Tests needed:** Threaded test: one thread reads from a handle while another closes/deletes the `FITSFile` (under tsan if possible). At minimum, move the cast and add a comment asserting GIL.

### CPP-003

- **Severity:** HIGH
- **Category:** races / stale CFITSIO handle
- **Files:** `src/torchfits/cpp_src/table_bindings.cpp` 305–310 vs 251–303
- **Description:** Every table mutation wrapper except `delete_fits_table_rows` calls `evict_cached_reader` (and invalidate meta/cache). Delete only invalidates the no-op file cache and `SharedReadMeta`.
- **Why it matters:** TLS `TableReader` cache holds a READONLY `fitsfile*`, `nrows_`, `row_width_bytes_`, and a cached pread fd. After a C++-level delete, a later `read_fits_table_rows` on the same thread can reuse a reader whose `nrows_` and file identity still describe the pre-delete table. `file_unchanged()` compares `st_mtime` at **second** resolution (`st_mtime`, not nsec), so a same-second rewrite can also miss.
- **Evidence:** `delete_fits_table_rows` lambda (305–310) has no `evict_cached_reader`. Python `torchfits.table` delete *does* call `invalidate_path_caches` → `cpp.evict_cached_reader` before and after. Direct `_C` / `_cpp.delete_fits_table_rows` does not. `file_unchanged()` is `table_reader.h` 105–116.
- **Recommended fix:** Call `evict_cached_reader` in the delete wrapper, same as the others. Optionally compare `st_mtimespec`/`st_mtim` nsec like `mtime_ns_from_stat`.
- **Compatibility:** Additive; should only change stale-read behavior.
- **Tests needed:** `_C.delete_fits_table_rows` then `_C.read_fits_table_rows` on the same thread without going through Python caches; assert new `NAXIS2`. Same-second rewrite of a table file.

### CPP-004

- **Severity:** HIGH
- **Category:** races — in-flight TLS readers
- **Files:** `src/torchfits/cpp_src/table_bindings.cpp` 84–137, 47–59
- **Description:** `evict()` only destroys **idle** cache entries. `acquire()` takes the reader out of the map; a concurrent writer’s `evict_everywhere` cannot see it. After the write, `release()` puts the pre-mutation reader back (READONLY handle still registered; `nrows_` stale unless `file_unchanged()` trips).
- **Why it matters:** Documented CFITSIO constraint: READWRITE open fails with status 104 while a READONLY handle is registered. The retry in `open_fits_for_write` only helps if eviction already happened. An in-flight reader can (a) read torn/stale rows during the write, (b) re-register a READONLY handle after the writer opened, poisoning later writes.
- **Evidence:** Comments at 117–119 and 51–56 acknowledge idle-only eviction. `release()` (103–115) does not re-check `file_unchanged()` before reinserting.
- **Recommended fix:** On `release()`, drop the reader if `!file_unchanged()`. Consider a generation counter on the path (even a `SharedReadMeta` uid sample) so in-flight readers cannot re-enter. Document that one process must not mutate a file while another thread reads it, if you are not willing to lock the path.
- **Compatibility:** Stricter drop of cached readers; no API change.
- **Tests needed:** Thread A looping `read_fits_table_rows`; thread B `append_fits_table_rows` / `delete_fits_table_rows`; join; subsequent write and read.

### CPP-005

- **Severity:** HIGH
- **Category:** correctness — whole-file compression vs raw fd
- **Files:** `src/torchfits/cpp_src/table_reader.h` 2649–2657, 2386–2461, 890–915; `src/torchfits/cpp_src/fits_detail.h` 339–347, 508–511
- **Description:** `direct_io_ok()` refuses only suffix `.bz2`, URLs, and extended-filename brackets. CFITSIO also transparently decompresses whole-file `.gz` / `.Z` / `.zip` on open. Buffered table reads then `::open` + `pread` the **compressed** bytes at header-derived offsets. Image mmap/pread uses a size check that often fails closed on gzip (compressed file smaller than claimed data) and falls back to CFITSIO; table buffered pread on EOF throws `"Failed to pread table bytes"` instead of falling back to `fits_read_tblbytes`. If the gzip file is *larger* than the uncompressed table extent (tiny tables), pread can succeed and silently decode gzip payload as FITS cells.
- **Why it matters:** `.fits.gz` is a normal astronomy distribution format (examples even cite `manga-*.fits.gz`). Default table path is buffered (`TORCHFITS_TABLE_BUFFERED` default true). This is not a theoretical CFITSIO footnote; the `.bz2` branch exists because the authors already hit this class of bug.
- **Evidence:** `direct_io_ok` 2649–2657; comment 2644–2648 names whole-file compression and only implements `.bz2`. `open_fits_readonly` uses `fits_open_diskfile` for non-URL paths (`fits_detail.h` 340–347), which still selects CFITSIO compression drivers by suffix/magic. Image path 534–535 requires `st_size >= data_offset + nbytes` before mmap. Table buffered path throws on short pread (2456–2460) rather than clearing `data_fd` and using `fits_read_tblbytes`.
- **Recommended fix:** Treat `.gz`, `.z`, `.Z`, `.zip`, `.bz2`, and CFITSIO compressed-driver prefixes as `direct_io_ok() == false`. On pread failure, fall back to `fits_read_tblbytes` instead of throwing. Prefer CFITSIO’s “is this a compressed diskfile?” if a stable API exists.
- **Compatibility:** `.fits.gz` tables that currently throw would start working; any path that accidentally “worked” on gzip garbage would change.
- **Tests needed:** Write a small binary table, gzip it, `read_fits_table` / `read_fits_table_rows` mmap true and false; compare to uncompressed. Same for a tiny 1-row table (gzip size > data extent). Image `.fits.gz` uint16 mmap on/off vs CFITSIO.

### CPP-006

- **Severity:** HIGH
- **Category:** correctness — dtype / endian
- **Files:** `src/torchfits/cpp_src/fits_bindings.cpp` 1540–1551 vs 1487–1493 vs `fits_detail.h` 460–472
- **Description:** `read_full` / `read_full_cached` map BZERO≈32768 / 2^31 integer conventions to `uint16`/`uint32`. `read_full_numpy` treats every scaled image except signed-byte as `float32` + `TFLOAT`. `read_full_numpy_cached` wraps `read_full_cached` then `.numpy()`, so unsigned images come back as numpy uint16/uint32.
- **Why it matters:** Two public `_cpp` entry points advertised as numpy variants of the same read disagree on dtype and values (float-promoted physical vs integer storage+offset). Unsigned FITS still images (very common) would not round-trip between the two numpy helpers.
- **Evidence:** `read_full_numpy` 1540–1551: `if (scaled) { signed-byte → int8; else → float/TFLOAT }`. No `is_unsigned_short_offset` / `is_unsigned_long_offset` branch. `read_tensor_canonical` 460–472 has those branches. `read_full_numpy_cached` 1487–1493.
- **Recommended fix:** Share one dtype/datatype selection function with `read_tensor_canonical`. Make `read_full_numpy` allocate uint16/uint32 (and apply the same mmap/bswap path) for the unsigned conventions.
- **Compatibility:** Numpy unsigned reads that currently return float32 would start returning uint16/uint32 — a breaking dtype change, but aligned with torch and with FITS convention.
- **Tests needed:** BITPIX=16 BZERO=32768 image: `read_full`, `read_full_numpy`, `read_full_numpy_cached` dtypes and values vs astropy/fitsio.

### CPP-007

- **Severity:** HIGH
- **Category:** correctness — overflow / truncated files / SIGBUS
- **Files:** `src/torchfits/cpp_src/table_reader.h` 1748–1766, 801–814, 612, 930; `src/torchfits/cpp_src/fits_detail.h` 381–386
- **Description:** Read paths carefully avoid `start_row + num_rows - 1` overflow (comments at 462–465, 834–835). `update_rows_mmap` still uses `start_row + num_rows - 1 > nrows_` (1764). `ensure_extent_within_file` multiplies `rows_before * row_bytes` in `LONGLONG` without overflow checks. `read_region_via_fd` compares `size_t(offset) + nbytes` which can wrap. `num_rows * col.repeat` (612) and `row_start_offset = (start_row-1) * row_width_bytes_` (930) are unchecked.
- **Why it matters:** A wrapped upper-bound check can skip the range test and `mmap`/`memcpy` past the mapping → SIGBUS or silent corruption. Pathological TFORM repeats / NAXIS2 can wrap `long`/`size_t`.
- **Evidence:** Asymmetric comments in `read_columns` vs `update_rows_mmap`. `ensure_extent_within_file` 801–806. `read_region_via_fd` 382. Image paths already use `checked_nelements_product`; table paths do not.
- **Recommended fix:** Reuse the `nrows_ - start_row + 1` form in mmap updates. Add a checked multiply for table extents (or refuse `row_width * nrows` that exceeds `off_t`/`size_t`). Use unsigned widening before `offset + nbytes`.
- **Compatibility:** Should only reject inputs that already overflowed.
- **Tests needed:** `start_row` near `LONG_MAX` on update_mmap (expect clean error). Header with huge `NAXIS1`/`NAXIS2` on mmap read (no SIGBUS).

### CPP-008

- **Severity:** HIGH
- **Category:** correctness — exception safety / resource leak
- **Files:** `src/torchfits/cpp_src/table_reader.h` 1809–2112
- **Description:** `TableReader::update_rows_mmap` `mmap`s then walks `tensor_dict` with `nb::cast<nb::ndarray<>>` and many mid-loop `throw`s. Error paths inside the `switch` `munmap`+`close`. A throw from `nb::cast` (1834) or from an exception type they do not catch **leaks the mapping and fd**. There is no RAII guard (unlike read mmap, which uses `MMapHandle`).
- **Why it matters:** Leaked `MAP_SHARED` writable maps keep pages dirty and the file locked; repeated failed updates can exhaust maps. Partial writes before the throw are already on disk (`MAP_SHARED`) with no rollback.
- **Evidence:** Read path 921–924 uses `MMapHandle mmap_guard`. Update path 1821–1825 does not. Cast at 1834 is after mmap, before the per-row munmap-on-error pattern.
- **Recommended fix:** `MMapHandle` (or a tiny local guard) with `MAP_SHARED` + `PROT_WRITE`. Consider documenting that mmap update is not atomic across columns.
- **Compatibility:** Internal.
- **Tests needed:** Update mmap with a non-ndarray value / wrong dtype; assert no leaked fd (`lsof` or `/proc/self/fd` count). Two-column dict where the second column has the wrong dtype: first column may already be committed (document or transactionalize).

### CPP-009

- **Severity:** HIGH
- **Category:** correctness — strided writes
- **Files:** `src/torchfits/cpp_src/fits_file.cpp` 393–419, 821–823; `src/torchfits/cpp_src/fits_bindings.cpp` 1229–1247, 1263–1278; `src/torchfits/cpp_src/table_ops.cpp` 575–595
- **Description:** Table **fixed-width** writes go through `ensure_c_contiguous_ndarray`. Image `write_image` / compressed `fits_write_img` pass `tensor.data()` with no contig check. VLA writes in `write_table_hdu` and `populate_rows` use `arr.data()` without that helper. TBIT/TLOGICAL expansion in `write_table_hdu` indexes `src[idx]` as if C-contiguous (loop `idx` 0..nelements-1), ignoring strides.
- **Why it matters:** A Fortran-contiguous or sliced torch/numpy image written via `write_hdus` is silently wrong. A strided VLA row or bool column is silently wrong. This is a science-correctness bug, not a performance nit.
- **Evidence:** `write_image` 419: `fits_write_img(..., tensor.data(), ...)`. `hdu_num`/`bscale`/`bzero` are unused (see CPP-021). `write_table_hdu` 1229 vs 1297 (fixed path *does* call `ensure_c_contiguous_ndarray`). `populate_rows` 575.
- **Recommended fix:** Reuse `ensure_c_contiguous_ndarray` (extend it past ndim=2 for images, or `contiguous()` in Python before the bind). For logical/bit, iterate with `stride()` as the mmap updater already does.
- **Compatibility:** Corrects silent mis-writes; outputs change for strided inputs.
- **Tests needed:** `write` of `tensor.t().contiguous()` vs `tensor.t()` for 2D images; VLA row with negative stride; bool column `arr[::-1]`.

### CPP-010

- **Severity:** HIGH
- **Category:** correctness — filtered mmap gather
- **Files:** `src/torchfits/cpp_src/table_reader.h` 1570–1582, 1598, 1618–1634, 1639–1646
- **Description:** After the predicate scan, gather has three holes: (1) requested `column_names` that do not exist are **silently omitted** (no throw, unlike `read_columns_mmap`); (2) VLA columns are `continue`d silently; (3) `item_size` is 8/4/2/else-1, so `COMPLEX_FLOAT` (8-byte elements but type not FLOAT) and `COMPLEX_DOUBLE` (16-byte `col.width`) are gathered as 1-byte cells without bswap; (4) LOGICAL gather writes `out_bool[k]` only, ignoring `repeat > 1`.
- **Why it matters:** `read_fits_table_filtered` is the C++ fallback for dense `where=` predicates. Wrong/missing columns are science bugs. Vector logical and complex columns are legal FITS.
- **Evidence:** 1574–1581 vs `read_columns_mmap` 853 which throws `"Column not found"`. 1598 skip VLA. 1618–1622 else `item_size = 1`. LOGICAL 1639–1646 vs mmap read 1081–1090 which loops `j < repeat`.
- **Recommended fix:** Throw on unknown names and on VLA/complex if unsupported; set `item_size` from `col.width` / type; index logical as `k * repeat + j`.
- **Compatibility:** Currently-wrong complex/logical filtered reads would start matching CFITSIO or raise.
- **Tests needed:** `read_fits_table_filtered` with a missing column name (must error); `2L` logical column; `1C`/`1M` columns; VLA in `column_names`.

### CPP-011

- **Severity:** HIGH
- **Category:** correctness — TSBYTE tables
- **Files:** `src/torchfits/cpp_src/table_reader.h` 183, 262–284, 1002–1010, 2584–2589, 1902–1916; `src/torchfits/cpp_src/fits_bindings.cpp` 899–904
- **Description:** FITS `S` / CFITSIO `TSBYTE` storage is unsigned bytes with offset 128 (XOR 0x80), same idea as image `SBYTE_IMG`. Image mmap applies `_xor_sign_bit_u8`. Table mmap/buffered BYTE paths `memcpy` raw bytes into `kInt8` / `kUInt8`. Table mmap **write** of TSBYTE copies `src_u8[idx]` with a comment claiming “byte-for-byte identical” signed/unsigned bit patterns. `write_table_hdu` maps Int8 → TFORM `B` / `TBYTE`, not `S`.
- **Why it matters:** Round-trip of signed-byte table columns disagrees with CFITSIO `fits_read_col(TSBYTE)` / astropy, and with the image path. Physical `-1` must be stored as `0x7F`, not `0xFF`.
- **Evidence:** Image: `fits_file.cpp` 1014–1018 XOR. Table mmap read 1002–1010 memcpy. Buffered extract 2584–2589 “No swapping needed for bytes”. Update mmap 1915–1916. `cfitsio_read_datatype` returns `TBYTE` for all `BYTE` columns (table_reader.h 504). `get_tensor_data_ptr` has no `kInt8` case (2530–2540), so buffered TSBYTE **throws** and falls back to `fits_read_col` (inconsistent with mmap).
- **Recommended fix:** On mmap/buffered read of `fits_typecode == TSBYTE`, XOR 0x80 (or read via `TSBYTE`). On mmap write, XOR before store. Keep CFITSIO column path on `TSBYTE`. Decide whether writers emit `S` or `B`+TZERO and stick to it.
- **Compatibility:** Changes TSBYTE mmap values to match CFITSIO.
- **Tests needed:** TFORM `1S` column written by astropy/fitsio; mmap and buffered read vs `fits_read_col`; mmap update then CFITSIO read.

### CPP-012

- **Severity:** HIGH
- **Category:** correctness — column repeat truncation
- **Files:** `src/torchfits/cpp_src/table_reader.h` 183, 262–284
- **Description:** `col.repeat = (int)repeat_long` runs for every column **before** the type switch. A later overflow check `repeat_long > 0x7fffffffL` exists only inside the binary `TSTRING` / `repeat_long > 1` branch, with broken indentation that looks like a mis-applied patch.
- **Why it matters:** An absurd `TFORMn` repeat truncates to 32-bit `int`, then `storage_bytes` and `byte_offset` are wrong → misaligned mmap deinterleave / buffer overruns on the declared row width, or a later “column count mismatch” that does not name the real cause.
- **Evidence:** Line 183 vs 274–280. Numeric columns never hit the throw. `width_long` used as string repeat (281–283) is also unchecked.
- **Recommended fix:** Check `repeat_long` (and `width_long`) once after `fits_get_coltype`, before the `int` cast. Apply to all types.
- **Compatibility:** Pathological headers that previously silently truncated would error.
- **Tests needed:** Header-only fixture with `TFORM1 = '2147483648E'` (or CFITSIO-maximum) must raise, not SIGBUS.

### CPP-013

- **Severity:** HIGH
- **Category:** dead / duplicate API — VLA shape
- **Files:** `src/torchfits/cpp_src/table_bindings.cpp` 312–330 vs 162–198, 354–373
- **Description:** Two `read_fits_table` overloads. The 2-arg version calls `read_columns(..., vla_flat=true)` then builds Python lists from `col_data.vla_data`, which is **empty** in the flat-VLA `ColumnData` constructor. The 4-arg overload uses `table_result_to_python`, which understands `vla_offsets`.
- **Why it matters:** `_C.read_fits_table(path, hdu)` returns empty lists for VLA columns. In-tree Python always passes `column_names` and `mmap`, so it hits the 4-arg overload. The 2-arg form is still a bound public overload.
- **Evidence:** `ColumnData(values, offsets, true)` sets `is_vla=true` and does not fill `vla_data` (table_reader.h 393–394). 2-arg bind 319–324 iterates `vla_data`. 4-arg uses `table_result_to_python`.
- **Recommended fix:** Delete the 2-arg overload, or route it through `table_result_to_python`.
- **Compatibility:** Removing the overload is fine if all callers pass columns/mmap; if someone used 2-arg, VLA results start being correct.
- **Tests needed:** `_C.read_fits_table(path, 1)` on a P/Q column; `_cpp.read_fits_table` same.

### CPP-014

- **Severity:** MEDIUM
- **Category:** correctness — float BSCALE/BZERO ignored
- **Files:** `src/torchfits/cpp_src/fits_detail.h` 175–177, 227–237; `src/torchfits/cpp_src/fits_file.cpp` 227–237; `src/torchfits/cpp_src/fits_bindings.cpp` 194–200, 678–702
- **Description:** `detect_scale_info_fast` returns immediately for `FLOAT_IMG`/`DOUBLE_IMG`. `FITSFile::read_tensor` and `read_full_nocache` skip scale/mmap probes for float-like BITPIX and read raw IEEE bits. Legal FITS still allows BSCALE/BZERO on float images (rare).
- **Why it matters:** Physical values would disagree with astropy/fitsio on those files. Integer scaled images are handled.
- **Evidence:** Early return `fits_detail.h` 177. `read_tensor` 227–237. `read_full_nocache` 678–702 uses `fits_read_img` with `nullval=nullptr` even for compressed floats (canonical path *does* pass NaN nulval for compressed float).
- **Recommended fix:** If you intentionally ignore float BSCALE, document it as a known limitation. Otherwise run the same scale detect for float BITPIX (and pass compressed nulval on the thin float path).
- **Compatibility:** Enabling scale would change rare float+BSCALE reads to match CFITSIO auto-scale.
- **Tests needed:** FLOAT_IMG with BSCALE=2, BZERO=1 vs astropy; CompImage float with undefined pixels vs nulval (see also CPP-015).

### CPP-015

- **Severity:** MEDIUM
- **Category:** correctness — compressed float nulls
- **Files:** `src/torchfits/cpp_src/fits_bindings.cpp` 678–692; `src/torchfits/cpp_src/fits_detail.h` 589–597
- **Description:** `read_tensor_canonical` substitutes NaN for undefined pixels on compressed float/double. `read_full_nocache`’s thin float path calls `fits_read_img` with `nullptr` nulval, so compressed undefined pixels become 0.
- **Why it matters:** Same file, `mmap=False` nocache vs cached/canonical, different nulls. Zero is a plausible science value.
- **Evidence:** Comment at `fits_detail.h` 589–591 vs `read_full_nocache` 689–692 `nullptr`.
- **Recommended fix:** Use the same nulval pointer on the thin path, or always go through `read_tensor_canonical`.
- **Compatibility:** Zeros that were blanked pixels become NaN (correct).
- **Tests needed:** CompImage float with BLANK/undefined tile vs both entry points.

### CPP-016

- **Severity:** MEDIUM
- **Category:** Python/C++ boundary — ownership / DLPack
- **Files:** `src/torchfits/cpp_src/torch_compat.h` 24–97, 38–60
- **Description:** `python_to_tensor` calls `__dlpack__()` with no device/stream args (DLPack 0.8+ / CUDA tensors). Capsule name is `"dltensor"` only (not `"dltensor"` v1 vs DLManagedTensorVersioned). `PyCapsule_SetName(..., "used_dltensor")` failure is `PyErr_Clear()`’d, so a failed rename can double-free if ATen and the capsule both delete. `alloc_numpy_array` multiplies `nelem *= d` without overflow checks, then `PyByteArray_FromStringAndSize(nullptr, (Py_ssize_t)nbytes)`.
- **Why it matters:** GPU / DLPack-v2 inputs fail opaquely. Overflowing shape → undersized bytearray + heap smash. Double-free if capsule rename fails after `fromDLPack`.
- **Evidence:** 73–96. 42–48. `echo_tensor` (fits_bindings 2472–2474) does not even convert — it is a Python identity used by `tests/test_dlpack_roundtrip.py`.
- **Recommended fix:** Prefer `THPVariable_Unpack` when `THPVariable_Check`, else DLPack with versioned capsule. Treat SetName failure as fatal (do not `fromDLPack` then ignore). Check `nbytes` against `PY_SSIZE_T_MAX`.
- **Compatibility:** Stricter errors on exotic DLPack producers.
- **Tests needed:** CPU torch tensor round-trip (already exists). Huge shape alloc must raise. If CUDA wheels exist, GPU `__dlpack__`.

### CPP-017

- **Severity:** MEDIUM
- **Category:** correctness — SharedReadMeta
- **Files:** `src/torchfits/cpp_src/fits_detail.h` 266–328, 350–357, 395–403
- **Description:** `g_shared_meta` is an unbounded `unordered_map` keyed by the **raw path string** (relative vs absolute vs `./x` are different entries). `get_shared_raw_fd` stores a `RawFdHolder` even when `open` returned `-1`, so later callers reuse a cached failure until invalidation. Stat validation is interval-gated (default 1s) and skipped entirely for extended-filename paths.
- **Why it matters:** Long-running processes that touch many files leak meta objects. A transient ENOENT gets sticky. Two spellings of one path do not share invalidation (`invalidate_shared_meta("a.fits")` misses `./a.fits`).
- **Evidence:** 280–293 emplace by `filename`. 354–357 assigns `open_readonly_fd` without checking `fd != -1`. 281–282 skip stat when brackets present.
- **Recommended fix:** Cap the map (LRU). Do not cache `fd == -1`. Canonicalize local paths (realpath, or document that callers must).
- **Compatibility:** Internal cache behavior.
- **Tests needed:** Open missing file then create it without `invalidate`; many unique paths and RSS (optional). Same inode via two path strings after rewrite.

### CPP-018

- **Severity:** MEDIUM
- **Category:** correctness — big-endian table mmap
- **Files:** `src/torchfits/cpp_src/table_reader.h` 987–1001, 1941–1998; `src/torchfits/cpp_src/fits_detail.h` 546–580; `src/torchfits/cpp_src/table_reader.h` 2550–2557
- **Description:** Image mmap bswap is gated on `host_is_little_endian()`. Table mmap read/write **always** bswap. Buffered extract correctly gates on `swap_endian`.
- **Why it matters:** Published wheels are x86_64/aarch64/macOS arm64 (little-endian). A big-endian build would invert table mmap endianness vs CFITSIO. Inconsistent internal paths.
- **Evidence:** `read_tensor_canonical` 546 vs `read_typed_mmap_column` always applying `bswap_fn`. `extract_column_data` 2557.
- **Recommended fix:** Gate table mmap bswap the same way as images, or `memcpy` on big-endian.
- **Compatibility:** None on current wheel targets.
- **Tests needed:** Only if you care about s390x/ppc64; otherwise a compile-time `static_assert` on little-endian for mmap paths.

### CPP-019

- **Severity:** MEDIUM
- **Category:** correctness — `open_fits_file` create vs write
- **Files:** `src/torchfits/cpp_src/fits_file.cpp` 60–68; `src/torchfits/cpp_src/fits_bindings.cpp` 2055–2058
- **Description:** `FITSFile(path, mode)` for `mode != 0` always `fits_create_file`. `open_fits_file(path, "w"|"w+")` maps to mode 1. There is no C++ path to open an existing file READWRITE except `open_fits_for_write` used by mutations. `"w+"` does not mean POSIX r/w existing.
- **Why it matters:** Callers of `_cpp.open_fits_file(path, "w+")` expecting to update an existing file get CFITSIO “file already exists” (unless they prepend `!`). Easy to confuse with mutation APIs.
- **Evidence:** Constructor 63–67. Bind 2055–2057.
- **Recommended fix:** Map `"r"` → READONLY, `"w"` → create (`!` if overwrite policy is elsewhere), `"w+"` / `"rw"` → `open_fits_for_write`. Document create-only if you keep the current mapping.
- **Compatibility:** Changing `"w+"` would be visible to `_cpp` users.
- **Tests needed:** `open_fits_file` on an existing file with `"r"` and `"w+"`.

### CPP-020

- **Severity:** MEDIUM
- **Category:** security / path
- **Files:** `src/torchfits/cpp_src/security.h` 23–64; `src/torchfits/_cpp.py` 79–124
- **Description:** C++ only blocks `|` prefix/suffix and `sh://`. SSRF for private `http(s)`/`ftp` is Python `guard_fits_path`. `_C` and `__getattr__`-leaked names skip that. `has_cfitsio_extended_filename_syntax` only looks at `/`, not `\`. Filter syntax `file.fits[col: ...]` ending in `]` is treated as extended filename (intended).
- **Why it matters:** Playbook requires guards on I/O façades **and** `torchfits.cpp` path APIs. `_cpp` wrappers cover `__all__` path functions, not leaked names or raw `_C`. Windows-style paths with `[dir]` would mis-detect brackets.
- **Evidence:** `security.h` 40–61. `_cpp.py` wraps only listed names. `_C` imported throughout `src/torchfits/`.
- **Recommended fix:** Call the same remote-path policy from C++ for `://` hosts, or refuse to document `_C` and close `__getattr__` (CPP-001). If you support Windows, use `find_last_of("/\\")`.
- **Compatibility:** Stricter URL rejection in C++ would match Python.
- **Tests needed:** `_C.open_fits_file("http://127.0.0.1/...")` vs `_cpp.open_fits_file`. `sh://` mixed case (already conceptually covered).

### CPP-021

- **Severity:** MEDIUM
- **Category:** dead parameters / header fidelity
- **Files:** `src/torchfits/cpp_src/fits_file.cpp` 393–427, 583–671, 629–640; `src/torchfits/cpp_src/fits_bindings.cpp` 1303–1325
- **Description:** `write_image(..., hdu_num, bscale, bzero)` never uses `hdu_num`, `bscale`, or `bzero`; it always `fits_create_img` on the current CHDU and does not write BSCALE/BZERO. `write_table_hdu` applies header cards **without** the structural-key skip list used by `write_hdus` / `write_hdu_header_cards`, so a user header can try to overwrite `TFIELDS`/`NAXIS2`/etc. `sanitize_fits_key("")` becomes `"UNKNOWN"`.
- **Why it matters:** Bindings expose scale arguments that do nothing. Table headers from Python can fight CFITSIO structural cards. Empty keys become a real `UNKNOWN` card.
- **Evidence:** 393–419 vs signature. `write_hdus` skip list 637–640 vs `write_table_hdu` 1303–1324. `sanitize_fits_key` `fits_detail.h` 695.
- **Recommended fix:** Write BSCALE/BZERO when they are not identity, or drop the args. Share one header-writer with the skip list. Reject empty keys.
- **Compatibility:** Dropping unused args is a `_C` signature change; writing scale keys is additive.
- **Tests needed:** `FITSFile.write_image` with bscale≠1; table write with `NAXIS2` in the extra header dict.

### CPP-022

- **Severity:** MEDIUM
- **Category:** build
- **Files:** `src/torchfits/cpp_src/CMakeLists.txt` 168–247, 279–304, 532–540, 50–54
- **Description:** Dummy `CUDA::nvrtc` / `torch::cudart` interface targets exist to satisfy PyTorch CMake when the toolkit is incomplete — fine for a CPU extension, but a dummy `IMPORTED_LOCATION` can still confuse downstream. Configure-time `file(CREATE_LINK)` for ZLIB rewrites the **environment prefix**. `libtorch_python` missing is `WARNING` even though `THPVariable_Wrap` is unconditional. `TORCHFITS_FINITE_MATH_ONLY` is correctly documented as unsafe for NaN sentinels.
- **Why it matters:** A wheel built without `libtorch_python` loads then crashes on first tensor wrap. ZLIB symlinks in conda prefixes are surprising in shared CI machines.
- **Evidence:** 537–539 WARNING not FATAL. 288–294 CREATE_LINK. 50–54 finite-math option.
- **Recommended fix:** `FATAL_ERROR` if `torch_python` is missing. Do not symlink into `$PREFIX` from CMake; fix the imported ZLIB target only.
- **Compatibility:** Fails the build instead of producing a broken extension.
- **Tests needed:** Existing wheel/CI builds. Optional: cmake with a fake prefix without libtorch_python must fail configure.

### CPP-023

- **Severity:** MEDIUM
- **Category:** races / FDs
- **Files:** `src/torchfits/cpp_src/table_reader.h` 900–904, 1145, 1809, 2244, 2397; `src/torchfits/cpp_src/fits_detail.h` 331–337
- **Description:** Several table `open()` calls omit `O_CLOEXEC`. `open_readonly_fd` uses `O_CLOEXEC` then falls back without it. Prefetch `std::thread` in buffered reads (2483–2491) shares `data_fd` with the parent; join-on-exception is handled (2502–2508), which is good.
- **Why it matters:** Fork+exec from the same process (Jupyter, multiprocessing spawn edge cases) can leak FITS fds into children. Less severe than the gzip/GIL issues.
- **Evidence:** `open(filename_.c_str(), O_RDONLY)` at 901 vs `open_readonly_fd` 332–336.
- **Recommended fix:** Always `O_CLOEXEC` / `O_RDWR|O_CLOEXEC`.
- **Compatibility:** Internal.
- **Tests needed:** Optional fd-flag check on Linux.

### CPP-024

- **Severity:** MEDIUM
- **Category:** correctness — empty tables / schema
- **Files:** `src/torchfits/cpp_src/table_reader.h` 452–455, 523–525 vs 1566–1615
- **Description:** `read_columns` / `read_columns_mmap` return `{}` when `nrows_ == 0`, dropping column names and dtypes. Filtered mmap with `num_valid == 0` still allocates empty tensors keyed by schema.
- **Why it matters:** Callers that zip columns against TFORM lists (the rewrite path comment at 441–442) get an empty dict for a 0-row table with defined columns.
- **Evidence:** 452–455 vs 1613–1615.
- **Recommended fix:** Return named empty tensors (and empty VLA lists) for zero-row tables, matching filtered.
- **Compatibility:** 0-row reads gain keys; Python code that treated `{}` as “no table” would change.
- **Tests needed:** Binary table with NAXIS2=0 and two columns; mmap and non-mmap.

### CPP-025

- **Severity:** LOW
- **Category:** correctness — HDUInfo / header parse
- **Files:** `src/torchfits/cpp_src/fits_bindings.cpp` 2137–2146; `src/torchfits/cpp_src/fits_file.cpp` 429–464
- **Description:** `HDUInfo.header` exposes a `dict` of `key → value`, collapsing duplicate `HISTORY`/`COMMENT` and dropping comments. `get_header` already returns triples. `read_keys` parses T/F/`'`/numbers; a keyword whose value is the string `T` becomes boolean.
- **Why it matters:** Round-trip of HISTORY cards through `open_and_read_headers` is lossy if callers use `.header`.
- **Evidence:** 2140–2145 builds `d[key] = value` only.
- **Recommended fix:** Expose the triple list as the canonical property; keep dict as a convenience with a name that says “last wins”.
- **Compatibility:** `.header` dict already exists.
- **Tests needed:** Two HISTORY cards via `open_and_read_headers`.

### CPP-026

- **Severity:** LOW
- **Category:** UB / portability
- **Files:** `src/torchfits/cpp_src/hardware.h` 11–14; `src/torchfits/cpp_src/fits_detail.h` 141–153; `src/torchfits/cpp_src/table_reader.h` 1012, 1640; `src/torchfits/cpp_src/fits_file.h` / `table_reader.h` destructors
- **Description:** `host_is_little_endian()` type-puns `uint16_t` through `uint8_t*` (strict aliasing). `_xor_sign_bit_u8` uses `uint64_t*` over `uint8_t*` after aligning. `data_ptr<bool>()` assumes PyTorch bool storage is `sizeof(bool)==1`. `TableReader` / `FITSFile` define destructors without deleting copy (C++ still generates a copy ctor; copy would double-`fits_close_file`). nanobind in-place construction makes copies unlikely from Python.
- **Why it matters:** Real UB on aggressive compilers; copy of `TableReader` would be catastrophic if it ever happens from C++.
- **Recommended fix:** `std::endian` (C++20) or a `memcpy` endian probe. `= delete` copy on `TableReader`/`FITSFile`. Keep the 64-bit XOR (widely used; alignment is handled).
- **Compatibility:** Internal.
- **Tests needed:** None beyond compiling with `-fstrict-aliasing` and a deleted-copy compile test.

### CPP-027

- **Severity:** LOW
- **Category:** dead code / duplication
- **Files:** `src/torchfits/cpp_src/cache.cpp`; `src/torchfits/cpp_src/fits_bindings.cpp` 2466–2470; `src/torchfits/_cpp.py` 166–170; `src/torchfits/cpp_src/hardware.cpp` 19–41; `src/torchfits/cpp_src/table_bindings.cpp` 312–330; `src/torchfits/cpp_src/fits_bindings.cpp` 2472–2474; `src/torchfits/cpp_src/table_ops.h` (global ns)
- **Description:** Cache configure/clear/size remain bound no-ops (playbook: do not revive `get_or_open_cached` — still true). `_REMOVED_STUBS` is never in `__all__`, so the skip loop is dead. Writable `MMapHandle` ctor unused. 2-arg `read_fits_table` duplicate (CPP-013). `echo_tensor` is a test identity. `kFitsReadWrite` locals unused. `table_ops` functions live in the global namespace while the rest is `torchfits::`.
- **Why it matters:** Inflates the accidental public surface (CPP-001) and confuses a freeze review.
- **Evidence:** `cache.cpp` 16–32. `_cpp.py` 166–170 vs `__all__` 21–77. Grep shows `MMapHandle(filename, writable)` only defined, never called from table/image code.
- **Recommended fix:** Unbind no-op cache APIs from `_C` or prefix `_`. Delete unused ctor or implement `MAP_SHARED`. Namespace `table_ops`.
- **Compatibility:** Removing bound no-ops breaks names that `_cpp.__getattr__` currently leaks.
- **Tests needed:** Isolation inventory after unbinding.

### CPP-028

- **Severity:** LOW
- **Category:** tests / verification
- **Files:** `tests/cpp/test_bracket_detection.cpp`; `src/torchfits/cpp_src/CMakeLists.txt` (no test target)
- **Description:** The only C++ self-check is a manual `clang++` recipe. It is not part of `pixi run` / CI. `has_cfitsio_extended_filename_syntax` regressions would not be caught automatically.
- **Why it matters:** The detector is load-bearing for mmap vs `fits_open_file` and for SSRF-adjacent extended syntax.
- **Evidence:** File header lines 4–6; no CMake/pytest reference (repo grep only hits this file).
- **Recommended fix:** One pytest that subprocesses the binary, or a tiny pybind of the helper, or duplicate the cases in a Python test of mmap-vs-bracket paths.
- **Compatibility:** None.
- **Tests needed:** Wire this file into CI.

### CPP-029

- **Severity:** LOW
- **Category:** correctness — `read_full_unmapped` naxis=0 / static buffer
- **Files:** `src/torchfits/cpp_src/fits_bindings.cpp` 545–570
- **Description:** Unlike other readers, `read_full_unmapped` does not special-case `naxis == 0` before `torch::empty(IntArrayRef(torch_shape, naxis))`. `static LONGLONG firstpixels[9]` is shared but never written after init (safe). Compressed nulval is applied here (good), unlike nocache float (CPP-015).
- **Why it matters:** Zero-axis images may throw or produce a weird rank-0 tensor depending on ATen.
- **Evidence:** 545–550 with no `if (naxis == 0)` (compare `read_full_cached` 234–245).
- **Recommended fix:** Same empty-tensor branch as canonical.
- **Compatibility:** Only empty-image HDUs.
- **Tests needed:** IMAGE HDU with NAXIS=0.

### CPP-030

- **Severity:** CLEANUP
- **Category:** duplication
- **Files:** `fits_file.cpp` `write_hdus` / `write_hdus_compressed_images` header loops; `fits_bindings.cpp` `write_hdu_header_cards`; `table_ops.cpp` append vs `populate_rows` (large parallel VLA/string/BIT blocks)
- **Description:** Header-card writing and table row population are copy-pasted three ways. BIT handling is commented as having been fixed independently in each copy.
- **Why it matters:** The next TFORM edge case will be fixed in one path and missed in another (already true for contig/VLA).
- **Evidence:** Visual duplication of TBIT expansion in `table_ops.cpp` 256–336 vs 714–806 vs `write_table_hdu` 1263–1290.
- **Recommended fix:** One `write_column_payload(fptr, colnum, ...)` used by append/update/create. Not a release blocker.
- **Compatibility:** Internal.
- **Tests needed:** Existing BIT/VLA round-trips should remain the contract.

---

## Accidental `_C` surface (inventory)

Bound on `_C` and **not** in `_cpp.__all__` (reachable via `__getattr__` and `import torchfits._C`):

- `HAS_BZIP2`
- `evict_cached_reader`
- `open_fits_mmap_reader`
- `read_fits_table_rows_mmap_from_reader`
- `configure_cache`, `clear_file_cache`, `invalidate_file_cache`, `get_cache_size`
- `echo_tensor`

Bound and in `__all__` (path-wrapped when accessed through `_cpp`, **unwrapped** on `_C`): all of `_PATH_FIRST` / `_PATH_CTORS` / handle APIs listed in `_cpp.py`.

`cpp.py` deprecates the module but still forwards the full leak.

---

## What looks solid (not issues)

- Move-only `FitsHandleGuard`; constructor failure closes a half-open `fitsfile*`.
- `TableReader` path ctor try/catch closes on analyze failure.
- Image NAXIS overflow checks and Random Groups rejection.
- Unsigned-int tolerance (`1e-5`) shared between image and table scale detect.
- SubsetReader mmap checks file size before mapping; signed-byte XOR on the image cutout path.
- `open_fits_for_write` status-104 retry + `evict_cached_reader` for most writers.
- Thread-local reader cache mutex vs registry lock: no obvious ABBA deadlock (destructor only takes registry lock; `evict` takes per-cache lock while registry is held — can stall thread exit, not deadlock with acquire).
- `ensure_c_contiguous_ndarray` uses signed strides (negative views).
- Compressed-image write sets `quantize_level=0` for lossless-capable codecs; PLIO rejects float.
- ABI check in `NB_MODULE` before binding.

---

## Suggested 1.0 priority

1. Close `_cpp` / `_C` surface (CPP-001) and GIL-before-cast (CPP-002).
2. `direct_io_ok` for all CFITSIO whole-file compressors + pread fallback (CPP-005).
3. Evict on delete + don’t re-cache stale readers (CPP-003, CPP-004).
4. Align numpy/torch unsigned dtypes (CPP-006); contig image/VLA writes (CPP-009); TSBYTE mmap (CPP-011).
5. Filtered-mmap gather and overflow checks (CPP-010, CPP-007, CPP-012).

None of the above requires new dependencies. Several are ABI-visible (`read_full_numpy` dtype, 0-row dict keys, gzip tables that currently error).
