# Phase 2 mandate — added by the user after R8 (2026-09-23)

The effort does NOT end at the first directory-by-directory pass. Before any release-ready /
freeze verdict, two more passes run. R18's freeze review happens only after both are green.

User directive (verbatim intent): "do not freeze … we will need to make more passes to ensure
1) there are no more left 2) that you did not introduce new ones."

## Pass 2 (Wave E, R19) — regression audit of OUR diffs

Input: `git diff 1bb6958..HEAD` (the review's own ~16.5k inserted lines). Slices follow the same
ownership map as R1-R8 (each audits the areas it knows). Mandate: find bugs the FIXES introduced.

1. Behavior changes beyond their stated contracts (every fix commit names its contract — verify
   the diff does nothing else observable).
2. Cross-round interaction bugs — the key tool is an interaction matrix of behavior-changing
   commits: e.g. r4c-15's `_io_engine/hdu_api._reassemble_longstr_cards` wrapper + r6a-01's
   `_hdu/card._reassemble_longstr_cards` both process `open_hdulist` headers (double-processing
   contract?); R2's `helpers._ThreadedAttr` vs pickling/copying of transforms; R5's schema dtype
   conventions vs R4's `_write_boundary_header`; r8a-02's buffered fallback vs R5's mutation
   mmap='auto' path; exit-code changes vs every CLI caller/test.
3. Missed callsites of migrated contracts (KeyError/ValueError/QuantizeError sweeps —
   `grep` every consumer of the changed functions).
4. Fixes whose regression tests do not actually assert observable behavior (the repo test bar:
   a plausible bug must fail the test).
5. Re-verify pass-1's "verified correct at HEAD / re-derived false" verdicts (r8a-10, r8b-05/06,
   r5b-05, r5c-10, r7c-13/14, r6b-11/12, …) — trust nothing.

## Pass 3 (Wave F, R20) — second-opinion defect hunt

Fresh agents (no pass-1 findings in context) over EVERY tracked source dir again, same rubric,
shifted lens:

1. Differential/property testing vs astropy & fitsio ground truth: random-schema write→read
   identity; hostile-header fuzzing vs astropy's parser (duplicate TTYPE, absurd TFORM, long
   strings, orphan CONTINUE); where= grammar vs docs/api-tables.md semantics on random predicates;
   table round-trips incl. TNULL/unsigned/scaled columns.
2. Every `deferred` item re-examined: is the deferral justified? (no wrong-result class-1 item
   may remain silently deferred.)
3. Deepest-look re-review of finding-dense modules: `_io_engine/{write_api,_write_helpers,
   _hdu_rewrite,_read_pipeline*}`, `_table/{read,_read_scan,_read_where,mutation}`,
   `transforms/{state,base,fits_meta,helpers}`, `cpp_src/table_reader.h`, `cpp_src/fits_bindings.cpp`.
4. Full oracle suite + fuzz seeds run at the end; any decode/write change re-runs
   `tests/test_output_parity.py`.

## Sequencing

R9, R10 (finish Wave A) → Wave B (R11-R14) → Wave C (R15-R17) → **Wave E (R19)** →
**Wave F (R20)** → R18 (version bump, gates, bench regression, freeze-review verdict, REPORT).
Both new waves use the same Method (failing-first evidence, LEDGER rows, per-round gates) and get
round codes R19/R20 with `dirs/` records like every other round.
