# Docs audit for 1.1.0 (research-only, 2026-08-25)

Scope: docs/{api,api-core-io,api-data,api-tables,api-transforms,install,compatibility,quickstart,cli}.md + README.md vs src/torchfits/. Verifiers: `pixi run docs-contract` PASS (23 passed), `pixi run docs-links` PASS (warnings only: fonts.gstatic 404, gnuastro network-unreachable x3, legacysurvey 400).

## Blockers

1. **Phantom PyTorch 2.11/2.12 prebuilt wheels** — `docs/install.md:43-51`, `docs/compatibility.md:14,28-29`, `README.md:13`, `docs/quickstart.md:15` claim prebuilt GitHub-Release wheels for torch 2.11.x/2.12.x (`torchfits-1.0.0+torch211/212-*`). Verified via GitHub API: **zero** `+torchNNN` assets exist in any release (all v1.0.0/v1.1.0b1 wheels are plain 2.13-lane builds); all three documented wheel URLs return HTTP 404; `scripts/torch_lanes.json` defines only the 2.13 lane. Either publish those lanes before 1.1.0 or cut the claims to "source build" (as `docs/release.md:16` implies).
   - Sub-nit: `install.md:50` uses a `linux_x86_64` wheel tag while `compatibility.md:29` uses `manylinux_2_28_x86_64` for the same artifact (cibuildwheel produces manylinux_2_28).

## Minor

2. **`write_tensor(checksum=)` undocumented** — `docs/api-core-io.md:446-450` signature block omits `checksum: bool = False` (src/torchfits/io.py:295-303); changelog even advertises it ("write_tensor() accepts checksum= for parity with write()"). `write()` documents it correctly.
3. **Unreleased version named in edge docs** — `docs/benchmarks.md:3` headline says "(v1.1.0, …)" and "lands in 1.2" while 1.1.0 is unreleased (`__init__.py` = 1.1.0b1, changelog under `## Unreleased`). Own runbook (`docs/release.md:55`) forbids naming unreleased versions outside the changelog.
4. **cu128 flavor claim** — `docs/install.md:31`, `docs/compatibility.md:15,36` list cu128 among working CUDA flavors; in-repo scripts note "no cu128 wheels exist for 2.12/2.13" (`scripts/canfar_matrix_bench_incontainer.sh:44-45`). Verify or drop cu128.

## Checked clean

- Root exports/`__all__` (api.md) == `src/torchfits/__init__.py`.
- io.py signatures: read/read_tensor/read_hdus/read_subset/open_subset_reader/open_table_reader/open/read_header/skinny helpers/write/insert|replace|delete_hdu/checksums/read_batch(strict=)/cache utils all match api-core-io.md.
- Table API: read/scan(where= pushdown ✓)/read_torch(simple-dialect where ✓)/scan_torch/reader/schema/mutations/interop/duckdb match api-tables.md; TABLE_BACKENDS exists.
- Transforms: all classes, params, defaults (Arcsinh a=1.0, Log a=1000/eps=1e-9, ZScale contrast=0.25, rgb/lupton_rgb signatures) match api-transforms.md; rgb + lupton_rgb exported.
- Data module: all 13 dataset classes + make_loader/fits_collate_fn defaults match api-data.md (FitsStagedCutoutIterableDataset cutouts_per_file=100 default correct).
- CLI: 15 subcommands, exit codes 0/1/2/3/4, `-e/-f/-o/--out-dir/--stdin/-j/-J`, convert `--recipe lupton --q --stretch --zeropoints --bands -w -c` all match cli.md.
- Env vars: every runtime `TORCHFITS_*` in user-facing audited docs (CACHE_DIR/REMOTE_CACHE/SAMPLE_CACHE/HTTP_AUTHORIZATION/HTTP_TOKEN) exists in src; removed `TORCHFITS_CFITSIO_CACHE_MB/_FILES` appear only in changelog as removal notes ✓.
- Versions: Python 3.10–3.14 everywhere matches pyproject classifiers + published cp310–cp314 assets; torch==2.13.0 pins match pyproject extras; HAS_BZIP2 exists (bindings.cpp:30).
- quickstart code paths (`to_tensor()`, `TableHDURef.read()`, CFITSIO sections) exist.

Verbatim verifier output captured in session log; both gates exited 0.
