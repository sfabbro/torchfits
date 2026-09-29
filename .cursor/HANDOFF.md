# Handoff — what was left to do

State as of the commit that landed the uncommitted `core`-library work on top of
`origin/main` (`f22c122`). The two bodies of work that were in flight have now
been merged; the items below were **not** done and are the next things to pick up.

## 1. The merge was verified, and it found three real defects

The first version of this note said the suite could not be run. That was true
when the `core` split was first committed and is now obsolete: the extension has
since been built and the full suite run. Keeping the record because *how* it was
verified is the useful part.

The tree arrived with no usable extension — the `_C` in `.pixi/envs/test` was
dated **Sep 7** (predating both bodies of work) and `torchfits._core` had never
been compiled, so every path through it raised `ImportError: cannot import name
'_core'`. After `pixi run dev`, `pytest tests/` is green (2877 passed, 2 skipped
— the skips are the un-fetched CFHT corpus, see item 3), and
`check_core_link` passes, so the core library really does resolve every CFITSIO
symbol without libtorch.

That run caught three defects that **the merge created**, each invisible from
either side alone:

- **The `core` split silently disabled the duplicate-TTYPE fix.** It rerouted
  `table.read` from `read_fits_table` (which guards first-wins) onto
  `read_fits_table_rows_raw` / `reader.read_rows_raw`, whose dict builders had
  no such guard, so a repeated `TTYPE` resolved to the *second* column again.
  Fixed by adding the `seen` guard to `table_result_to_raw_python` and
  `tensor_map_to_raw_python`.
- **`.agents/` was committed while gitignored.** `f22c122` added `.agents/` to
  `.gitignore` as installed tooling; the `core`-split work added tracked files
  under it, which `git add` refuses after a delete/restore cycle. `.agents/` and
  `.codex/` are now untracked and gitignored, and their stale local copies were
  re-synced from the canonical `.cursor/`.
- **A docs/`--comment` collision.** The new `setkey` note named a flag the
  session's gate forbids in that section; reworded rather than relaxing the
  test.

## 2. The one hand-resolved conflict (now covered by a passing test)

`src/torchfits/_table/_read_where.py` had two independent NULL-handling fixes
applied to the same three functions, and they disagreed on the mechanism:

- `origin/main` (commit `43d70b4`) filled unknown to `False` inside each mask
  helper and added `_exclude_float_nan` so an IEEE `NaN` match is treated as
  unknown;
- the uncommitted work instead re-marked null rows as *null* so the unknown
  propagates through a negation and is dropped once at the top of
  `_where_mask_for_table`.

The merge keeps **both**: the null-propagation structure, plus `_exclude_float_nan`
on the negated `IN` / `BETWEEN` paths, plus the `NOT`-over-comparison column walk
that excludes null and NaN rows. `docs/api-tables.md` was updated to state the
NaN rule alongside the `TNULL` rule.

The union holds: `tests/test_where_semantics.py` (56 tests) and
`tests/test_where_matrix.py` both pass, which covers the contract that matters —
`X != v`, `NOT (X == v)`, and `NOT (X IN (...))` agree and leave NaN and null
rows out on **all three** backends (`auto`, `cpp`, `torch`) and for both
`mmap=True` and `mmap=False`. The two mechanisms are composable, so no
follow-up is needed here.

## 3. The real-data corpus is not on this machine

`tests/test_reads_real_data.py`, `tests/test_core_library_real_data.py` and
friends `pytest.skip` unless the CFHT samples are fetched — roughly 5 GB
(`scripts/fetch_cfht_megacam_sample.sh`, `scripts/fetch_cfht_megapipe_sample.sh`).
The skips are deliberate and are not a failure, but it does mean the MegaCam and
MegaPipe paths have had no coverage run against this tree.

## 4. 53 audit deferrals are still open

`.cursor/post-1.0-backlog.md` → "1.2 audit deferrals / Still open" carries 53
unfixed ids across rounds R1–R12 (`r1b-08` … `r7c-24`). Each was recorded with a
reason; they are not regressions, just not-done work.

Minor bookkeeping gap: the LEDGER has R18 rows, but
`.cursor/harness/release-1-2-0-review/dirs/` has no `R18.md` (it goes R17, R19,
R20). Worth a line so the next round knows R18 was a scan-only pass.

## 5. The fork is still ahead of the org

`origin/main` is **110 commits** ahead of `upstream/main` (`astroai/torchfits`).
The org PR for everything from the R15–R20 review rounds and this `core` split is
still owed, and the public-API freeze review should be re-run before the next
SemVer cut because the new `core` library moves code across the import boundary.
