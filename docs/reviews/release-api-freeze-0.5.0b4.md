# Release API freeze review — 0.5.0b4

**Date:** 2026-06-30  
**Verdict:** **Ship** (feature-freeze + API-freeze for 0.5.0b4)

## Summary

Public API surface is stable and documented. Parity matrix rows are test-backed.
Release gates pass on CPU (37 tests, 3 CUDA skips) and bench-gpu (40 tests).
Bugbot and security review found no blocking issues on the perf/docs diff.

## Blocking

None.

## Should-fix (non-blocking)

| Item | Notes |
|---|---|
| `read_image` deprecation | Documented alias; keep until 0.6+ removal per changelog |
| Bot PRs #171–#173 | Unrelated automated PRs; do not merge for 0.5.0b4 |

## Defer to 0.6.0

- MEF handle pooling, `bench_ml_loader` release gate, `torchfits.data` (per roadmap)
- `read_unified` strategy refactor, `_table/` split

## Evidence

| Gate | Result |
|---|---|
| `pixi run release-gate` | 37 passed, 3 skipped |
| `pixi run -e bench-gpu release-gate` | 40 passed |
| Bugbot | No bugs |
| Security review | No issues |
| Benchmark | `exhaustive_mmap_0.5.0b4_20260630_162835`, 3626 rows, 13 deficits |

## Version triplet

- `pyproject.toml`, `pixi.toml`, `__init__.py`: **0.5.0b4** ✓

## Public API (root `__all__`)

32 symbols + namespaces `table`, `cache`, `cpp`. Quick Paths in `docs/api.md` align.
New in 0.5.0b4: `TABLE_BACKENDS` (table submodule), scale-on-device integer semantics documented.

## API-freeze declaration

From this tag forward until **0.5.0** final: bugfixes and doc corrections only;
no new public symbols or breaking signature changes without a new beta suffix.
