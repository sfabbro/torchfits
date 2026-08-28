# Audit ledger — torchfits @ 7841dec (2026-08-27)

Tracked files: 440. Reviewed: 414. Excluded (data/generated): 26.
Every tracked file in [tracked-files.txt](tracked-files.txt) is accounted for by exactly one row below. Depth legend: **D** = deep read (+ live repro where relevant), **M** = medium read, **T** = triage/provenance check, **E** = excluded (data/generated, not code).

## Coverage groups

| Group | Files | Depth | Notes |
|---|---|---|---|
| `src/torchfits/` root py (17: `__init__, io, table, hdu, cache, cpp, _cpp, interop, where, _where, fits_schema, header_parser, http_util, logging, vos_uri, _string_decode, _tensor_buffer`, `py.typed`) | 17 | D (12) / M (5) | Lazy façade, closed `__all__`, where grammar, guards, tensor buffer, string decode fully read; header_parser/http_util/logging/vos_uri/interop medium. |
| `src/torchfits/_io_engine/` (24 py) | 24 | D (13) / M (11) | Read pipeline, write API+helpers, caches, quantize, paths, device, options, checksum deep; subset, http_subset, table_api, table_reader_api, table_streaming, batch, image, image_meta, hdu_api, _hdu_rewrite, _read_pipeline_fallback, _read_scan-schema-where interfaces medium. |
| `src/torchfits/_table/` + `_table_engine/` (16 py) | 16 | D (9) / M (7) | read, _read_where, engine, read_policy, backend_policy deep; write, mutation, _mutation_coerce, arrow_convert, interop, _read_scan, _read_schema, utils, cache medium (interfaces + spot logic). |
| `src/torchfits/_hdu/` (7) + `hdu.py` | 8 | D | All read fully. |
| `src/torchfits/transforms/` (8) | 8 | M | Interfaces, dispatch, and spot logic; covered by unit + e2e tests. |
| `src/torchfits/data/` (3) | 3 | D (datasets core) / M (remote, __init__) | Sharding/shuffle/prefetch logic read. |
| `src/torchfits/cli/` (18) | 18 | D (2) / M (16) | copy, cutout, common deep; main + remaining cmds_* structure/arg/exit-code review. |
| `src/torchfits/cpp_src/` (20) | 20 | D (14) / M (6) | fits_bindings.cpp full; fits_file.cpp/.h, fits_detail.h, table_reader.h (core paths), table_types, security, cache.h/.cpp, internal_utils, hardware deep; table_ops.cpp, table_bindings.cpp, bindings.cpp, fits_rw.h, torch_compat.h, torchfits_torch.h medium. |
| `tests/` (91: 89 py + tests/cpp + pyi) | 91 | T + targeted D | Full suite executed (1206 pass/23 skip); inventory + deep reads of contract/security/malformed/table/quantize suites; mock-patching pattern flagged (A-19). |
| `benchmarks/` (27) | 27 | T | Methodology headers, suites.py, warmup/run discipline in bench_fits_io; not re-run (time + hardware variance); provenance of docs headline CSVs verified. |
| `docs/` (69: 35 md + assets) | 69 | T (35 md) / E (34 assets) | api*.md vs `__all__` via docs-contract test; benchmarks.md provenance vs `docs/assets/bench/`; parity.md, install, cli reviewed. Assets: gallery/logos/katex/bench CSVs = data. |
| `scripts/` (51) | 51 | T | release_lane, update_changelog, check_torch_extra_pins, cibuildwheel wrappers, bench renderers triaged; several executed via pixi tasks (check-lane, changelog-check). |
| `examples/` (35) | 35 | T | Executed via `tests/test_examples_runner.py` (green); spot-read image/table/dataset examples. |
| CI/build/packaging (18: 5 workflows, 2 CMakeLists, pyproject, pixi.toml, .pre-commit, .gitignore/attributes, LICENSE, constraints-wheel, SDIST-README, packaging/conda/recipe.yaml, extern/2, README, AGENTS.md, zensical.toml) | 18 | D / M | pyproject+pixi+CMake+workflows deep; CI head jobs read fully. |
| `.cursor/`, `.dsh/` (31) | 31 | T | Agent/harness infra incl. prior audit docs; not product code; ledger cross-checked against it. |

## Exclusions (26 files, all data/generated)

- `docs/assets/gallery/*.png` (10), `docs/logo.svg`, `docs/torchfits-logo*.png` (3) — binary image assets.
- `docs/assets/katex/**` (14: fonts/css/js), `docs/javascripts/katex.js` — vendored KaTeX for docs rendering.
- `docs/assets/bench/*/results.csv|summary.md|*_deficits.csv|megacam_results.csv|ml_results.csv` — measured-data provenance; existence-checked against docs citations only (B4 verification).
- `pixi.lock` — generated lockfile; pins validated via `pixi run check-lane`.

Untracked output dirs (`build/`, `site/`, `__pycache__/`, `examples/output/`) are not part of the ledger.
