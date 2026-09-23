# R6 slice A — `_hdu/` header layer (header.py, card.py, hdu_list.py, _repr.py)

Baseline: HEAD `44b5bd9` (R1–R5 landed). Routed-in owner: **r4b-13 (BLOCKER)** per `R4-write.md`.
All IDs `r6a-NN`. Evidence = regression test added first, run against unfixed code (failure pasted),
fix, run again (pass pasted). Targeted runs only. `Header._mutation_seq` coordination with slice B:
**not needed** — `Header._version` already bumps on every mutator (`__setitem__/update/clear/pop/
popitem/setdefault/insert/remove`→`__delitem__`, `_append_card`); slice B confirmed and is dropping
the unguarded `id()` cache instead. No cross-slice Header change landed.

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r6a-01 (r4b-13) | BLOCKER | 1 | `_hdu/hdu_list.py:fromfile` (`Header(info.header)` construction); `_hdu/card.py:_reassemble_longstr_cards,_is_string_typed` | LONGSTRN `&`+CONTINUE chains were never reassembled on the `HDUList.fromfile` path: `torchfits.open(p)[0].header["LONGSTR"]` returned 67 chars + a literal `&` with detached CONTINUE cards while `read_header` returned the full 80; every `HDUList.write`/`insert_hdu`/`replace_hdu`/`delete_hdu` rewrite through such a header silently truncated >68-char values AND replayed the raw CONTINUE card as forged `= '        '` value-cards (repro below). Root: cpp `open_and_read_headers`/`read_header` surface raw triples (chain markers kept in values, CONTINUE raw quoted field in the comment slot) and `fromfile` built `Header(info.header)` from them verbatim. | Shared helper `_reassemble_longstr_cards` in `card.py` mirroring `fast_parse_header_cards` (the `read_header` oracle): `&`-marker drop + join + verbatim restore when uncontinued, per-segment `&'` marker detection on the value field only (comments on chain cards do not hide markers), empty-segment handling, bare-CONTINUE append to the last string-typed card (`_is_string_typed` mirrors `_parse_card` typing), orphan CONTINUE normalized to `read_header`'s card shape and never fused into an unrelated keyword, escaped-quote (`''`) segments exact. Wired at `fromfile` construction (`Header(_reassemble_longstr_cards(info.header))`) so DIRECT `HDUList.fromfile` callers are root-fixed; the r4c-15 `open_hdulist` wrapper is untouched and is now a no-op (belt-and-braces) on the fixed output. Tests `tests/test_longstr_header_reassembly.py` (7 new fromfile-direct pins incl. the rewrite-chain byte-identity pin). Before: `6 failed, 6 passed`; after: `12 passed`; owned suites `58 passed`; oracles `tests/test_write_read_identity.py` (incl. the r4b-13 pin `test_rewrite_preserves_long_strings`, verified a NORMAL test — no `xfail`/`strict` remains in `test_writing.py`/`test_write_fidelity.py`/`test_write_read_identity.py`, and it still pins rewrite survival; the fromfile-level pin is now `test_fromfile_longstr_survives_rewrite_chain`) + `test_longstr_and_scaled_tables.py` + `test_differential_astropy.py`: `75 passed`; write/parity oracles `test_output_parity.py test_writing.py test_write_fidelity.py test_hdu_file_ops.py`: `179 passed, 3 skipped`; HDU-layer contracts `test_hdu_close_and_overflow.py test_io_invariants.py test_hdu_table_contracts.py`: `68 passed`. | fixed |
| r6a-02 | MINOR | 1/7 | `_hdu/header.py:Header.__init__` | Unsupported constructor input silently built an EMPTY header: `Header(iter([("K", 1, "")]))`, `Header("K1 = 1")`, `Header(5)` all returned `{}` with no error (every branch missed the input and fell through). Repro test `test_header_rejects_unsupported_cards_input`. Before: `Failed: DID NOT RAISE TypeError`. | Trailing `else` in `__init__` raises `TypeError("cannot build a Header from <type>; expected a Header, dict, or card sequence")`. Test passes after (in `tests/test_header_versioning.py`). | fixed |
| r6a-03 | MINOR | 5 | `_hdu/hdu_list.py:fromfile` (legacy `AttributeError` fallback) | The `except AttributeError` around `cpp.open_and_read_headers` caught AttributeErrors raised INSIDE the batch open and silently rerouted the whole read through the legacy per-HDU fallback path. Repro: monkeypatch `open_and_read_headers` to raise `AttributeError`, `open_fits_file` to raise `AssertionError`; `HDUList.fromfile` ran the fallback. Before: `assert False + where False = isinstance(AssertionError('legacy fallback must not run'), AttributeError)`. | Attribute presence checked up front (`getattr(cpp, "open_and_read_headers", None)`); only a MISSING symbol selects the fallback, an internal AttributeError now surfaces via the documented `RuntimeError("Failed to open FITS file ...")` wrap. Test `test_fromfile_internal_attribute_error_is_not_silently_rerouted` passes after. | fixed |
| r6a-04 | MINOR | 1/4 | `_hdu/hdu_list.py:HDUList.__getitem__` (`_extname_idx`) | The cached EXTNAME→index map went stale as soon as a caller renamed `EXTNAME` on a live header: after `hdul["OLD"]` then `hdul[1].header["EXTNAME"] = "NEW"`, `hdul["NEW"]` raised KeyError and `hdul["OLD"]` silently returned the no-longer-named HDU (stale cache after mutation; silent wrong result). Repro test `test_hdu_list_name_lookup_tracks_header_renames`. Before: `KeyError: "HDU 'NEW' not found"` (trace shows `idx = self._extname_idx.get(key)`). | Dropped the cache; name lookup scans HDUs live (a few `header.get` calls — HDU counts are tiny, metadata walk not a pixel hot path). `_extname_idx` field + `append` reset removed. Test passes after. | fixed |
| r6a-05 | MINOR | 5/7 | `_hdu/hdu_list.py:HDUList.__getitem__` | Integer indexing rejected numpy integers and gave a misleading error for garbage keys: `hdul[np.int64(0)]` fell into name lookup and raised `KeyError: "HDU '0' not found"`; `hdul[1.5]` raised `KeyError: "HDU '1.5' not found"` instead of a type error. House rule (r4b-11) is `operator.index` acceptance at API indices. Repro test `test_hdu_list_index_contract` (captured pre-fix via `git stash` of the fix). Before: `KeyError: "HDU '0' not found"`. | `__index__` acceptance via `operator.index` (negative wrap unchanged); non-index keys raise `TypeError("HDUList indices must be int or EXTNAME str; got <type>")`. Missing EXTNAME name lookup still `KeyError`. Test passes after. | fixed |
| r6a-06 | MAJOR | 1/7 (hostile) | routed: `_io_engine/hdu_api.py:_reassemble_longstr_cards` (r4c-15 wrapper — OFF-LIMITS this round) | Hostile orphan CONTINUE after non-string-typed cards is fused into the preceding keyword by the wrapper's target rule (it lacks the typed-value check): raw file `NAXIS = 0` + `CONTINUE  'orphan segment'` → `torchfits.open(p)[0].header["NAXIS"] == '0orphan segment'` (CONTINUE gone) while `read_header` keeps `NAXIS=0, CONTINUE='orphan segment'` and (post r6a-01) `HDUList.fromfile` matches `read_header`. Pre-existing at HEAD (independent of my fix; the wrapper re-runs after `fromfile`). Probe: `open() NAXIS: '0orphan segment' / fromfile NAXIS: '0' + Card('CONTINUE','orphan segment','') / read_hdr NAXIS: 0 CONTINUE: 'orphan segment'`. | Deferred: fix is one rule in the wrapper's CONTINUE branch (adopt `_is_string_typed` from `card.py` for target validity) — the queue froze `hdu_api.py:open_hdulist`/its helper ("do NOT touch, do NOT re-fix"). Routed to the wrapper's owner (R4-core) / any later round allowed to touch `hdu_api.py`. | deferred |
| r6a-07 | MINOR | 1 | routed: `_io_engine/hdu_api.py:read_header_fast` slow fallback (`cpp.read_header` triples, `fast_header=False` or fast-parse failure), `_io_engine/_read_pipeline_fallback.py` `Header(header_data)` sites | Same un-reassembled-triple root as r6a-01 at the remaining cpp-triple consumers: headers built there keep `&` markers + detached CONTINUE cards for >68-char strings (all six PUBLIC routes verified joined after r6a-01 — `read_header`, `open`, `HDUList.fromfile`, `read_tensor/read_hdus/read return_header`; the residue is error/option paths only). NOTE: the fourth consumer, `_hdu/table_hdu_ref.py:357`/`_refresh_file_view`, was fixed in-wave by slice B (`r6b-13` in `dirs/R6-views.md`, failing-first pin `test_tablehduref_refresh_preserves_long_string_values`) — verified together with this slice's suite: `14 passed`. | Deferred/routed: the `_io_engine` sites are outside this slice; `_hdu/card.py:_reassemble_longstr_cards` is import-ready for their construction sites (`read_header_fast`'s slow fallback and `read_fallback`'s `Header(header_data)`). | deferred |
| r6a-08 | MINOR | 7/8 | `tests/test_header_value_typing.py:test_numpy_scalar_header_values_roundtrip` | A-01 coverage gap (queue: "add if missing"): no round-trip test existed for numpy scalar kinds through `torchfits.write`→`read_header` (the fix itself `_normalize_header_value` at `header.py:16-33` verified intact). | Added pin: `np.int64/np.float32/np.float64/np.bool_` + 0-d int/float arrays round-trip as plain-Python values with plain-Python types; astropy agrees. Passes at and after HEAD (coverage add, no code change). | fixed |
| r6a-09 | MINOR | 7 | `tests/test_header_versioning.py:test_header_delitem_removes_all_comment_cards` | `header-delitem-history` invariant pinned for HISTORY only; `del header["COMMENT"]` orphaning was unpinned (same `remove(remove_all=True)` path today). | Added COMMENT twin of the HISTORY del pin (cards list must end `["SIMPLE"]`, mapping key gone). Passes at HEAD (invariant pin, no code change). | fixed |

## Evidence log (before → after)

r6a-01 failing-first (`pixi run pytest tests/test_longstr_header_reassembly.py -q`, unfixed code):

```
FAILED test_fromfile_direct_joins_longstr_chain - assert 'AAAAAAAAAAAA...AAAAAAAAAAAA&' == 'AAAAAAAAAAAA...AAAAAAAAAAAAA'
FAILED test_fromfile_direct_joins_chain_with_per_card_comments - assert 'This is a lo... that needs &' == 'This is a lo...g.final part.'
FAILED test_fromfile_direct_joins_multi_continue_empty_segment_and_final_amp - assert 'BBBB...&' == 'BBBB...'
FAILED test_fromfile_direct_bare_continue_appends - assert 'abc' == 'abcdef'
FAILED test_fromfile_direct_orphan_continue_stays_verbatim - assert '' == 'orphan segment'
FAILED test_fromfile_longstr_survives_rewrite_chain - assert "CCCCCCCCCCCC...C= '        '" == 'CCCCCCCCCCCC...CCCCCCCCCCCCC'
6 failed, 6 passed in 1.35s
```

(the rewrite-chain failure shows the forged `= '        '` value-cards the replay produced from the
detached raw CONTINUE card — the silent corruption r4b-13 reported.)

After the fix:

```
tests/test_longstr_header_reassembly.py        12 passed in 1.35s
tests/test_header_value_typing.py tests/test_header_duplicate_keys.py tests/test_header_ascii_strictness.py
tests/test_header_versioning.py tests/test_complex_header.py tests/test_longstr_header_reassembly.py
tests/test_read_header.py tests/test_hdu_str.py   58 passed in 2.00s
tests/test_write_read_identity.py tests/test_longstr_and_scaled_tables.py tests/test_differential_astropy.py   75 passed
tests/test_output_parity.py tests/test_writing.py tests/test_write_fidelity.py tests/test_hdu_file_ops.py   179 passed, 3 skipped
tests/test_hdu_close_and_overflow.py tests/test_io_invariants.py tests/test_hdu_table_contracts.py   68 passed
```

r6a-02/03/04 failing-first (`tests/test_header_versioning.py tests/test_complex_header.py` on unfixed code):

```
FAILED test_header_rejects_unsupported_cards_input - Failed: DID NOT RAISE TypeError
FAILED test_fromfile_internal_attribute_error_is_not_silently_rerouted - assert False
 +  where False = isinstance(AssertionError('legacy fallback must not run'), AttributeError)
FAILED test_hdu_list_name_lookup_tracks_header_renames - KeyError: "HDU 'NEW' not found"
3 failed, 18 passed in 1.43s
```

r6a-05 failing-first (fix stashed for the run, then restored):

```
FAILED tests/test_complex_header.py::test_hdu_list_index_contract - KeyError: "HDU '0' not found"
1 failed in 1.31s
```

All five pass post-fix in the `58 passed` run above.

## Verified areas (reviewed, no defect to fix)

- **r4c-01 `close()` block (do-not-re-fix) confirmed sound around it**: `caches._register_open_hdulist` sets `hdulist._registry_key = real` at registration and `_unregister_open_hdulist(None, handle)` has a by-handle full-scan fallback, so `close()` always deregisters even if the key were lost. `__del__`→`close()` double-close is idempotent. The `except Exception: pass` around registration in `fromfile` degrades to "no auto-close-on-mutation" and self-heals via `close()` — left as-is (bookkeeping, not IO).
- **A-01 `_normalize_header_value`** (`header.py:16-33`) intact and now pinned (r6a-08); `docs-api-sync`: no new public symbols; `_normalize_header_value` has no importers outside `header.py`.
- **`header-delitem-history`** holds for HISTORY and COMMENT (r6a-09 + existing `test_header_delitem_removes_all_history_cards`).
- **Duplicate keys** (class 1): first-occurrence mapping rule + full card preservation pinned by `test_header_duplicate_keys.py` (unchanged, all green); `setitem` replaces the first card in place; `remove(remove_all=False)` falls back to the next occurrence; COMMENT/HISTORY mapping tracks the latest line — all pinned at HEAD.
- **Hostile headers** (class 7): duplicate `TTYPE`/absurd `TFORM` cards are recorded faithfully by the header layer (probe: `TTYPE9='DUP'`, `TFORM9='999999999999J'` kept verbatim, no crash) — schema interpretation belongs to `_read_schema`/`table_reader` (R5/R8). Chain hostility fully pinned: nested multi-segment chains, per-card comments, quoted-comment separators inside segments, `''` escaped quotes, empty segments, bare CONTINUE, orphan CONTINUE, broken chains, literal trailing `&` (`test_longstr_header_reassembly.py`).
- **`_repr.py` faithfulness** (class 8): `render_html_table` HTML-escapes every value (`html.escape(str(value))`), style attributes interpolate caller constants only (no data injection in notebook HTML), `aligns`/`cell_extra` length-mismatch raises `ValueError`. Ragged rows render short (callers pass fixed-width rows) — cosmetic, no contract. `HDUList.__repr__`/`info()` summary columns match the documented layout; `Header.__repr__` is the dict-subclass mapping repr — faithful to the documented mapping view (first-occurrence / commentary-latest), with `.cards`/`get_history`/`get_comment` as the documented lossless views and `_repr_html_` showing every card. No change.
- **Overflow** (class 2): no size math in any of the four files (`NAXIS` products, repeats, `size_t` mixes live in `_table`/`cpp_src`). **Concurrency** (class 4): `_HDU_CLASSES` lazy-import dict race is idempotent; `Header` mutation is single-owner by contract (`_version` is a plain int counter — no cross-worker mutation contract exists). **Security** (class 6): `fromfile` guards via `guard_fits_path` before open; no TOCTOU claims in these docstrings. **Perf** (classes 1–6): no perf change landed (the `_extname_idx` removal is a staleness correctness fix with O(#HDUs) recompute, not a perf claim); `_reassemble_longstr_cards` short-circuits with a key scan when no CONTINUE card is present.

## Doc / changelog owed (docs land at R15; changelog via integration commit subjects)

- `Fixed`: LONGSTRN long-string header values (>68 chars) no longer truncated/corrupted through `torchfits.open`, `HDUList.fromfile`, or any `write`/`insert_hdu`/`replace_hdu`/`delete_hdu` rewrite chain (r6a-01/r4b-13).
- `Fixed`: `HDUList` EXTNAME lookup reflects header renames (r6a-04).
- `Changed`: `Header(...)` raises `TypeError` for unsupported constructor input instead of silently building an empty header (r6a-02); `HDUList.__getitem__` raises `TypeError` for non-indexable keys (was `KeyError`) and accepts any `__index__` integer (r6a-05).

Deferred (site + evidence):
- r6a-06 — `_io_engine/hdu_api.py:_reassemble_longstr_cards` fuses a hostile orphan CONTINUE into the preceding keyword (`open(p).header['NAXIS'] == '0orphan segment'` vs `read_header` `'0'` + separate CONTINUE card; raw file `NAXIS = 0` + `CONTINUE  'orphan segment'`); needs the wrapper's typed-value target check — file frozen this round.
- r6a-07 — `read_header_fast` slow fallback (`hdu_api.py`) / `_read_pipeline_fallback.py` `Header(...)` sites build from un-reassembled cpp triples (same root as r6a-01; `table_hdu_ref.py:357` resolved in-wave by slice B's r6b-13); `_hdu/card.py:_reassemble_longstr_cards` is import-ready for those construction sites.

## Per-file disposition

| file | depth | finding IDs | status |
|---|---|---|---|
| `src/torchfits/_hdu/header.py` | deep (full class review: construction, mutators, `_version`, mapping/card views, repr) | r6a-02, r6a-08 (coverage of A-01 area) | fixed |
| `src/torchfits/_hdu/card.py` | deep (Card + new `_reassemble_longstr_cards`/`_is_string_typed`, mirrored against `fast_parse_header_cards`) | r6a-01 | fixed |
| `src/torchfits/_hdu/hdu_list.py` | deep (fromfile, dispatch/fallback, getitem/len/enter/exit/del/close surroundings, write/append/validate, info/_repr_html_/repr) | r6a-01, r6a-03, r6a-04, r6a-05 | fixed |
| `src/torchfits/_hdu/_repr.py` | deep (escaping, style interpolation, alignment validation) | — (verified clean) | clean |
| `tests/test_longstr_header_reassembly.py` | extended (r4c-15 pins kept; 7 fromfile-direct pins added) | r6a-01 | fixed |
| `tests/test_header_value_typing.py` | reviewed + 1 coverage test added | r6a-08 | fixed |
| `tests/test_header_duplicate_keys.py` | reviewed (all pins hold at HEAD) | — | clean |
| `tests/test_header_ascii_strictness.py` | reviewed (write-strict/read-lenient split holds) | — | clean |
| `tests/test_header_versioning.py` | reviewed + 2 tests added | r6a-02, r6a-09 | fixed |
| `tests/test_complex_header.py` | reviewed + 3 tests added | r6a-03, r6a-04, r6a-05 | fixed |
| `tests/test_read_header.py` | reviewed (index/name lookup pinned; fixture-in-CWD left to R11's suite-wide candidate) | — | clean |
| `tests/test_hdu_str.py` | reviewed | — | clean |
