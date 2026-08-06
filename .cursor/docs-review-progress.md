# Documentation Review Progress

Started after the `1.0.0rc5` release. Review criteria:

- Accessible to astronomers, software developers, and students.
- Public API and CLI examples match the live source.
- Runnable examples are simple, self-consistent, and reproducible.
- Local and external links are verified.
- Redundancy is minimized and navigation is clear.

## Completed review passes

All 23 files under `docs/` have been reviewed against `src/`, `examples/`,
tests, the live CLI parser, and the built Zensical site, and the findings
were fixed and pushed in four commits:

- `3826c11` — review checkpoint (progress tracking).
- `8340d45` — batch 1: quickstart / index / compatibility / install / cli /
  cli-recipes / python-workflows / examples (+ sample-cache prelude and
  scripts fixes: `fetch_example_samples.sh`, `clean_install_smoke.sh`).
- `9cc8da3` — batch 2: api.md / api-core-io.md / api-tables.md / api-data.md
  (return types, WHERE `==` dialect, DuckDB example, read_hdus/read_batch_info
  contracts, flush() removal, cache semantics).
- `b6f7eef` — batch 3: api-transforms.md / benchmarks.md / parity.md /
  examples-transforms.md / migration_astropy.md / migration_fitsio.md /
  changelog.md / roadmap.md / contributing.md / release.md.

## Key fixes by theme

- **Version labels**: all `rc4` / `1.0.0` labels → `1.0.0rc5`; changelog
  dates match git tags; `[1.0.0rc5]` compare link added.
- **WHERE dialect**: single `=` raises `ValueError`; docs now show `==`
  (verified live).
- **Transform contracts**: `PercentileClipNormalize.inverse` approximate
  (forward clamps); zscale output not clamped to `[0,1]`; `lupton_rgb`
  returns float64 `(H,W,3)`; normalizers + `ArcsinhStretch` need float input;
  `LogStretch` / `SqrtStretch` clamp negatives.
- **Benchmarks**: rc5 CPU exhaustive provenance for generated sections;
  category summary cites both sources; repeated-cutouts row synced with the
  full table; published-CSV note for 20260806 runs.
- **Performance snapshots** in migration pages replaced with verifiable rc5
  CPU / Round-3 CUDA medians.
- **Misc**: `flush()` does not exist on `HDUList` (read-oriented `open`);
  `read_batch_info` returns `num_files`/`existing_files`; `cache_dir=` only
  on remote-capable datasets; iterable shuffle is seed-fixed; header tests
  renamed (`test_read_header.py`); HorseHead Compose snippets cast `.float()`;
  PyPI publish is token-based, not trusted publishing.

## Verification

- `pixi run docs-contract` — 23 tests pass.
- `pixi run docs-build` — clean build.
- `pixi run docs-links` — no broken local links; only the two known
  non-fatal external warnings (fonts.gstatic.com, legacysurvey.org).
- Do not commit `dist-local/` or generated `docs/published-examples/`.
