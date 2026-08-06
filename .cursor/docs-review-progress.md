# Documentation Review Progress

Started after the `1.0.0rc5` release. Review criteria:

- Accessible to astronomers, software developers, and students.
- Public API and CLI examples match the live source.
- Runnable examples are simple, self-consistent, and reproducible.
- Local and external links are verified.
- Redundancy is minimized and navigation is clear.

## Completed review passes

The following files were reviewed against `src/`, `examples/`, tests, the live
CLI parser, and the built Zensical site. Findings were captured in the session
and are being converted into small fixes:

- `index.md`, `install.md`, `compatibility.md`, `quickstart.md`
- `cli.md`, `cli-recipes.md`, `python-workflows.md`, `examples.md`
- `api.md`, `api-core-io.md`, `api-tables.md`, `api-data.md`
- `api-transforms.md`, `architecture.md`, `benchmarks.md`, `parity.md`
- `examples-transforms.md`, `examples-ml.md`, `migration_astropy.md`

Remaining review files:

- `migration_fitsio.md`, `changelog.md`, `roadmap.md`, `contributing.md`,
  `release.md`

## Known high-priority fix groups

- Stale `rc4` / `1.0.0` labels and old MPS benchmark claims.
- Quickstart, CLI, table, dataset, transform, migration, and example snippets
  with undefined names or incorrect API contracts.
- Sample-cache path disagreement between shell fetch scripts and Python
  sample resolution.
- Benchmark pages citing ignored/local CSVs without stable published assets or
  family/mmap/run provenance.
- Source-install and compatibility recipes that do not reproduce the current
  PyTorch 2.13 lane.

## Verification

Use `pixi run docs-contract` during edits, `pixi run preflight-push` before
each checkpoint, and `pixi run ci-local` before the final push. Build the site
with `pixi run docs-build` and run `pixi run docs-links` after documentation
changes. Do not commit `dist-local/` or generated `docs/published-examples/`.
