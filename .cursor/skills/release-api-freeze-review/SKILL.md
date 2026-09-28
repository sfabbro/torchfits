---
name: release-api-freeze-review
description: Pre-release feature and public API freeze audit for torchfits. Inventories exports, cross-checks docs/parity/examples against code and tests, verifies release gates and benchmark claims. Use for feature freeze, API freeze, 0.5.0b4, pre-release review, or public API audit before tagging.
disable-model-invocation: true
---

# Release API freeze review (torchfits)

Use before tagging a beta/final when the release claims **feature-freeze** and **API-freeze**.

## Outputs

Write findings to `.cursor/reviews/release-api-freeze-<version>.md`
(gitignored; keep product docs under `docs/` free of freeze audit dumps) with sections:

1. **Verdict** — Ship / Ship with notes / Block
2. **Blocking** — must fix before tag
3. **Should-fix** — fix before tag if time allows
4. **Defer to next minor** — document only
5. **Evidence gaps** — parity rows without tests

## Phase 1 — Public surface inventory

```bash
python .cursor/skills/release-api-freeze-review/scripts/inventory_public_api.py
```

- Compare script output to `docs/api.md` Quick Paths and `src/torchfits/__init__.py` `__all__`
- Flag: undocumented exports, documented-but-missing symbols, deprecated aliases without notice
- Check every lazy namespace in `_NAMESPACES` (`src/torchfits/__init__.py`) —
  this list has grown (`transforms`, `data`, `where`, `hdu` are not in it),
  so read it from the source rather than from this file

## Phase 2 — Docs contract

Read and cross-check:

| Doc | Check |
|---|---|
| `README.md` | No out-of-scope claims; performance claims link to `docs/benchmarks.md` rather than restating numbers |
| `docs/api.md` | Every Quick Path entry resolves; env vars documented |
| `docs/parity.md` | Each **Supported** row has test evidence |
| `docs/examples.md` | Every example path exists and runs |
| `docs/changelog.md` | Unreleased section matches diff |
| `docs/release.md` | Version triplet matches `pyproject.toml`, `pixi.toml`, `__init__.py` |

Run: `pixi run pytest tests/test_docs_integrity.py -q`

## Phase 3 — Behavior & defaults

Audit high-impact defaults (breaking if changed post-freeze):

- `read(..., scale_on_device=True)`, `mmap="auto"`, `device="cpu"`
- `table.read(..., where=)` pushdown policy env vars
- GPU integer paths (signed-byte, unsigned convention)
- Deprecations: `read_image` → `read_tensor`

## Phase 4 — Release gates

```bash
pixi run release-gate
pixi run -e bench-gpu release-gate   # when CUDA available
pixi run pytest tests/test_examples_runner.py -q
```

## Phase 5 — Benchmark claims

- Every run ID cited in `docs/benchmarks.md` has a directory under
  `docs/assets/bench/`, and every directory there belongs to a cited run ID.
  The published CSVs live there, not in `benchmarks_results/` — that is local
  scratch and is gitignored, so a clean clone has none of it.
  (`tests/test_docs_integrity.py::test_benchmark_run_ids_match_published_assets`
  enforces this; run it rather than eyeballing the tables.)
- Deficit count documented honestly

## Phase 6 — Freeze verdict

| Criterion | Pass? |
|---|---|
| No undocumented public exports | |
| Parity Supported rows backed by tests | |
| release-gate green (CPU + GPU if available) | |
| Examples runnable | |
| Version/changelog aligned | |
| No README/docs scope creep (WCS/sphere/HEALPix) | |

**API-freeze rule:** After this review passes, only bugfixes and doc corrections until tag — no new public symbols or signature changes without bumping beta suffix and changelog entry.

## Companion reviews (run separately)

- `/thermo-nuclear-code-quality-review` — maintainability (not API surface)
- `/review-bugbot` — diff bugs
- `/review-security` — security on branch diff
