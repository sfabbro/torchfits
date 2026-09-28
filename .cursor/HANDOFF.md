# Handoff — what was left to do

State as of the commit that landed the uncommitted `core`-library work on top of
`origin/main` (`f22c122`). The two bodies of work that were in flight have now
been merged; the items below were **not** done and are the next things to pick up.

## 1. The C++ build and the test suite have not been run against this tree

This is the big one. The merged tree changes `cpp_src/` substantially on both
sides (the new `cpp_src/core/` library, the `table_reader.h` zero-repeat guard,
the first-wins duplicate-TTYPE guard), but no compiled extension existed for it:

- the `_C` extension installed in `.pixi/envs/test` is dated **Sep 7**, which
  predates both bodies of work;
- `torchfits._core`, the new torch-free core library, had **never been compiled**
  at all, so every code path through it raised
  `ImportError: cannot import name '_core' from 'torchfits'`.

So `pixi run preflight-push`, `pixi run ci-local`, and `pytest tests/` could not
be run before this commit. `ruff check .` passes. **Build the extension and run
the suite before tagging a release.**

## 2. One conflict was resolved by hand and needs a real test run

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

This resolution is a judgement call and was **not** verified by a test run
(see item 1). The contract to confirm is the one pinned by
`tests/test_where_matrix.py::test_float_neq_and_not_exclude_nan_on_every_strategy`
and `tests/test_where_semantics.py`: `X != v`, `NOT (X == v)`, and
`NOT (X IN (...))` must agree and must leave NaN and null rows out, on **all
three** backends (`auto`, `cpp`, `torch`), for both `mmap=True` and
`mmap=False`. If that matrix passes, the union is good; if it fails, the two
mechanisms are not composable and one has to give.

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
