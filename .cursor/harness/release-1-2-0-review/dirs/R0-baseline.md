# R0 — baseline, sync, review-dir skeleton

Date: 2026-02-… | Baseline HEAD: `1bb6958d958c5ebdb9a201f95bc5a01e89fe7acb`

## Steps

1. `git fetch upstream && git merge --ff-only upstream/main` → already up to date (HEAD = `1bb6958`, matching plan baseline). `git push origin main` → "Everything up-to-date".
2. Skeleton created: `baseline/`, `dirs/`, `splits/`; `tracked-files.txt` = 468 lines (`git ls-files`). LEDGER.md header written with HEAD sha.
3. Baseline gates:
   - `pixi run preflight-push` → **PASS** (ruff check, ruff format --check 334 files, mypy 95 source files, compileall, check-lane [torch lane 2.13, torchfits 1.1.3], changelog-check up to date).
   - `pixi run test` → **PASS** (1584 passed, 24 skipped, 21 xfailed, 386 warnings in 299.65s). No baseline failures → no `meta-NN` findings.
4. Bench baseline (same host = this machine for every later delta):
   - `pixi run bench-fits` → run_id `20260922_145241` (87 cases, `scopes=['fits']`, mmap=on) → `baseline/fits-results.csv`
   - `pixi run bench-fitstable` → run_id `20260922_145339` (684 rows, `scopes=['fitstable']`, mmap=on) → `baseline/fitstable-results.csv`

## Findings

(none yet; baseline failures become `meta-NN` assigned to the owning round)
