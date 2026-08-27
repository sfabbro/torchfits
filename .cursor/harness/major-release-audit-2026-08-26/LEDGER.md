# Audit ledger — torchfits 1.1.0 (2026-08-26)

Tracked files: 429. Reviewed: 404. Excluded: 25.

## Exclusions

| File | Why |
|---|---|
| `docs/assets/gallery/cli_rgb_demo.png` | Binary logo/image asset |
| `docs/assets/gallery/image_compose_pipeline.png` | Binary logo/image asset |
| `docs/assets/gallery/image_cutout.png` | Binary logo/image asset |
| `docs/assets/gallery/lightcurve_asymmetric_sigma_clip.png` | Binary logo/image asset |
| `docs/assets/gallery/lightcurve_sigma_clip.png` | Binary logo/image asset |
| `docs/assets/gallery/lupton_rgb_sdss.png` | Binary logo/image asset |
| `docs/assets/gallery/megapipe_cutout_collage.png` | Binary logo/image asset |
| `docs/assets/gallery/ml_gz_class_grid.png` | Binary logo/image asset |
| `docs/assets/gallery/rgb_sky_collage.png` | Binary logo/image asset |
| `docs/assets/gallery/rgb_vs_lupton_dwarf.png` | Binary logo/image asset |
| `docs/assets/gallery/table_fits_scale_columns.png` | Binary logo/image asset |
| `docs/assets/katex/contrib/auto-render.min.js` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/fonts/KaTeX_AMS-Regular.woff2` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/fonts/KaTeX_Main-Regular.woff2` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/fonts/KaTeX_Math-Italic.woff2` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/fonts/KaTeX_Size1-Regular.woff2` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/fonts/KaTeX_Size2-Regular.woff2` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/katex.min.css` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/assets/katex/katex.min.js` | Vendored KaTeX font/CSS/JS for docs; not product code |
| `docs/javascripts/katex.js` | Vendored KaTeX assets |
| `docs/logo.svg` | Binary logo/image asset |
| `docs/torchfits-logo-hero.png` | Binary logo/image asset |
| `docs/torchfits-logo-mark.png` | Binary logo/image asset |
| `docs/torchfits-logo.png` | Binary logo/image asset |
| `pixi.lock` | Generated lockfile; pins reviewed via pixi.toml + lane scripts |

## Reviewed

| File | Note |
|---|---|
| `.cursor/docs-review-progress.md` | Agent/dev infra (not public product API) |
| `.cursor/harness/.gitignore` | Agent/dev infra (not public product API) |
| `.cursor/harness/config.json` | Agent/dev infra (not public product API) |
| `.cursor/harness/deep-review-1-faithfulness.md` | Agent/dev infra (not public product API) |
| `.cursor/harness/docs-audit-1.1.0.md` | Agent/dev infra (not public product API) |
| `.cursor/harness/matrix-grid-20260804.md` | Agent/dev infra (not public product API) |
| `.cursor/harness/plans/perf-parity-backlog-1-10.md` | Agent/dev infra (not public product API) |
| `.cursor/harness/playbook.md` | Agent/dev infra (not public product API) |
| `.cursor/hooks.json` | Agent/dev infra (not public product API) |
| `.cursor/hooks/harness-session-start.sh` | Agent/dev infra (not public product API) |
| `.cursor/hooks/harness-stop.sh` | Agent/dev infra (not public product API) |
| `.cursor/jules-ledger.md` | Agent/dev infra (not public product API) |
| `.cursor/jules.md` | Agent/dev infra (not public product API) |
| `.cursor/post-1.0-backlog.md` | Agent/dev infra (not public product API) |
| `.cursor/skills/docs-authoring-review/SKILL.md` | Agent/dev infra (not public product API) |
| `.cursor/skills/release-api-freeze-review/SKILL.md` | Agent/dev infra (not public product API) |
| `.cursor/skills/release-api-freeze-review/scripts/inventory_public_api.py` | Agent/dev infra (not public product API) |
| `.dsh/README.md` | Agent/dev infra (not public product API) |
| `.dsh/cordis.patch.yml` | Agent/dev infra (not public product API) |
| `.dsh/skills/science-core/SKILL.md` | Agent/dev infra (not public product API) |
| `.dsh/skills/torchfits-dev/SKILL.md` | Agent/dev infra (not public product API) |
| `.gitattributes` | Repo root / packaging metadata |
| `.github/workflows/bench-report.yml` | GitHub Actions |
| `.github/workflows/build_wheels.yml` | GitHub Actions |
| `.github/workflows/ci.yml` | GitHub Actions |
| `.github/workflows/docs.yml` | GitHub Actions |
| `.github/workflows/sanitizer.yml` | GitHub Actions |
| `.gitignore` | Repo root / packaging metadata |
| `.pre-commit-config.yaml` | Repo root / packaging metadata |
| `AGENTS.md` | Repo root / packaging metadata |
| `CMakeLists.txt` | Repo root / packaging metadata |
| `LICENSE` | Repo root / packaging metadata |
| `README.md` | Repo root / packaging metadata |
| `SDIST-README.txt` | Repo root / packaging metadata |
| `benchmarks/__init__.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_all.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_arrow_tables.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_cache.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_contract.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_cpp_backend.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_denoise.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_fits_io.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_fits_write.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_fitstable_io.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_fixtures.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_gpu_memory.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_gpu_transports.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_http_stream.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_median_stack.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_megacam_cutouts.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_ml_loader.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_science_pipeline.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_table.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/bench_timing.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/cfitsio_direct/CMakeLists.txt` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/cfitsio_direct/bench_cfitsio_direct.c` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/config.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/mpl_config.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/replays/upstream_sources.json` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/run_cfitsio_direct_bench.py` | Benchmark harness / published CSV under docs/assets/bench |
| `benchmarks/suites.py` | Benchmark harness / published CSV under docs/assets/bench |
| `constraints-wheel.txt` | Repo root / packaging metadata |
| `docs/api-core-io.md` | User-facing documentation |
| `docs/api-data.md` | User-facing documentation |
| `docs/api-tables.md` | User-facing documentation |
| `docs/api-transforms.md` | User-facing documentation |
| `docs/api.md` | User-facing documentation |
| `docs/architecture.md` | User-facing documentation |
| `docs/assets/bench/20260719_075555/megacam_results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/summary.md` | User-facing documentation |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/torchfits_deficits.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/summary.md` | User-facing documentation |
| `docs/assets/bench/exhaustive_cpu_20260807_013736/torchfits_deficits.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/summary.md` | User-facing documentation |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/torchfits_deficits.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/summary.md` | User-facing documentation |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/torchfits_deficits.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_mps_20260719_143706/megacam_results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_mps_20260719_143706/ml_results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_mps_20260719_143706/results.csv` | Published scorecard CSV |
| `docs/assets/bench/exhaustive_mps_20260719_143706/summary.md` | User-facing documentation |
| `docs/assets/bench/exhaustive_mps_20260719_143706/torchfits_deficits.csv` | Published scorecard CSV |
| `docs/assets/bench/ml_20260719_145743/ml_results.csv` | Published scorecard CSV |
| `docs/benchmarks.md` | User-facing documentation |
| `docs/changelog.md` | User-facing documentation |
| `docs/cli-recipes.md` | User-facing documentation |
| `docs/cli.md` | User-facing documentation |
| `docs/compatibility.md` | User-facing documentation |
| `docs/contributing.md` | User-facing documentation |
| `docs/denoise-pipeline.md` | User-facing documentation |
| `docs/examples-ml.md` | User-facing documentation |
| `docs/examples-transforms.md` | User-facing documentation |
| `docs/examples.md` | User-facing documentation |
| `docs/index.md` | User-facing documentation |
| `docs/install.md` | User-facing documentation |
| `docs/migration_astropy.md` | User-facing documentation |
| `docs/migration_fitsio.md` | User-facing documentation |
| `docs/parity.md` | User-facing documentation |
| `docs/python-workflows.md` | User-facing documentation |
| `docs/quickstart.md` | User-facing documentation |
| `docs/release.md` | User-facing documentation |
| `docs/roadmap.md` | User-facing documentation |
| `docs/stylesheets/extra.css` | Docs site chrome |
| `examples/_plotting.py` | Runnable examples / gallery helpers |
| `examples/_sample_data.py` | Runnable examples / gallery helpers |
| `examples/cli/imstat_imarith.sh` | Runnable examples / gallery helpers |
| `examples/cli/make_rgb_demo.py` | Runnable examples / gallery helpers |
| `examples/desi_shaped_spectrum.py` | Runnable examples / gallery helpers |
| `examples/example_custom_transform.py` | Runnable examples / gallery helpers |
| `examples/example_cutout_wcs_write.py` | Runnable examples / gallery helpers |
| `examples/example_data_catalogs.py` | Runnable examples / gallery helpers |
| `examples/example_image.py` | Runnable examples / gallery helpers |
| `examples/example_image_cube.py` | Runnable examples / gallery helpers |
| `examples/example_image_cutouts.py` | Runnable examples / gallery helpers |
| `examples/example_image_dataset.py` | Runnable examples / gallery helpers |
| `examples/example_image_mef.py` | Runnable examples / gallery helpers |
| `examples/example_lupton_rgb_sdss.py` | Runnable examples / gallery helpers |
| `examples/example_m13_stack.py` | Runnable examples / gallery helpers |
| `examples/example_make_loader_vs_dataloader.py` | Runnable examples / gallery helpers |
| `examples/example_manga_logcube.py` | Runnable examples / gallery helpers |
| `examples/example_mef_header.py` | Runnable examples / gallery helpers |
| `examples/example_megacam_cr_denoise.py` | Runnable examples / gallery helpers |
| `examples/example_megacam_mef_cutouts.py` | Runnable examples / gallery helpers |
| `examples/example_megapipe_cutout_collage.py` | Runnable examples / gallery helpers |
| `examples/example_ml_galaxyzoo_legacy.py` | Runnable examples / gallery helpers |
| `examples/example_polars.py` | Runnable examples / gallery helpers |
| `examples/example_quantize_int16.py` | Runnable examples / gallery helpers |
| `examples/example_rgb_sky.py` | Runnable examples / gallery helpers |
| `examples/example_staged_cutouts.py` | Runnable examples / gallery helpers |
| `examples/example_streaming_cubes_spectra.py` | Runnable examples / gallery helpers |
| `examples/example_table.py` | Runnable examples / gallery helpers |
| `examples/example_table_interop.py` | Runnable examples / gallery helpers |
| `examples/example_table_recipes.py` | Runnable examples / gallery helpers |
| `examples/example_time_series.py` | Runnable examples / gallery helpers |
| `examples/example_transforms.py` | Runnable examples / gallery helpers |
| `examples/gallery_images.py` | Runnable examples / gallery helpers |
| `examples/gallery_tables_lc.py` | Runnable examples / gallery helpers |
| `examples/test_examples.py` | Runnable examples / gallery helpers |
| `extern/VERSIONS.txt` | CFITSIO vendor pin/patches/license (tarball itself gitignored) |
| `extern/licenses/CFITSIO-LICENSE.txt` | CFITSIO vendor pin/patches/license (tarball itself gitignored) |
| `extern/patches/cfitsio-4.7.0-bzip2.patch` | CFITSIO vendor pin/patches/license (tarball itself gitignored) |
| `extern/patches/cfitsio-4.7.0-plio-cbuf.patch` | CFITSIO vendor pin/patches/license (tarball itself gitignored) |
| `extern/vendor.sh` | CFITSIO vendor pin/patches/license (tarball itself gitignored) |
| `overrides/home.html` | Zensical HTML theme overrides |
| `overrides/main.html` | Zensical HTML theme overrides |
| `overrides/partials/actions.html` | Zensical HTML theme overrides |
| `packaging/conda/recipe.yaml` | Conda recipe |
| `pixi.toml` | Repo root / packaging metadata |
| `pyproject.toml` | Repo root / packaging metadata |
| `scripts/aggregate_matrix_bench.py` | Release/CI/bench/docs scripts |
| `scripts/bench_cfitsio_direct.sh` | Release/CI/bench/docs scripts |
| `scripts/bench_deficit_focus.sh` | Release/CI/bench/docs scripts |
| `scripts/bench_exhaustive_local.sh` | Release/CI/bench/docs scripts |
| `scripts/bench_release_scorecard.sh` | Release/CI/bench/docs scripts |
| `scripts/bench_suite.sh` | Release/CI/bench/docs scripts |
| `scripts/build_docs_pages.sh` | Release/CI/bench/docs scripts |
| `scripts/build_wheels_local.sh` | Release/CI/bench/docs scripts |
| `scripts/canfar_denoise_incontainer.sh` | Release/CI/bench/docs scripts |
| `scripts/canfar_gpu_bench_incontainer.sh` | Release/CI/bench/docs scripts |
| `scripts/canfar_matrix_bench_incontainer.sh` | Release/CI/bench/docs scripts |
| `scripts/check_docs_links.py` | Release/CI/bench/docs scripts |
| `scripts/check_duplicate_cpp.py` | Release/CI/bench/docs scripts |
| `scripts/check_torch_extra_pins.py` | Release/CI/bench/docs scripts |
| `scripts/ci_local.sh` | Release/CI/bench/docs scripts |
| `scripts/cibuildwheel.sh` | Release/CI/bench/docs scripts |
| `scripts/cibw_before_build.sh` | Release/CI/bench/docs scripts |
| `scripts/cibw_test.sh` | Release/CI/bench/docs scripts |
| `scripts/clean_install_smoke.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_canfar_bench_vos.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_cfht_calib_frames.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_cfht_megacam_sample.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_cfht_megapipe_sample.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_example_samples.sh` | Release/CI/bench/docs scripts |
| `scripts/fetch_rgb_sky_samples.sh` | Release/CI/bench/docs scripts |
| `scripts/gpu-bootstrap.sh` | Release/CI/bench/docs scripts |
| `scripts/gpu-env-loader.sh` | Release/CI/bench/docs scripts |
| `scripts/import_canfar_bench_artifacts.py` | Release/CI/bench/docs scripts |
| `scripts/launch_canfar_denoise.sh` | Release/CI/bench/docs scripts |
| `scripts/launch_canfar_gpu_bench.sh` | Release/CI/bench/docs scripts |
| `scripts/launch_canfar_matrix_grid.sh` | Release/CI/bench/docs scripts |
| `scripts/patch_bench_docs.py` | Release/CI/bench/docs scripts |
| `scripts/patch_canfar_exhaustive_docs.sh` | Release/CI/bench/docs scripts |
| `scripts/publish_canfar_bench_vos.sh` | Release/CI/bench/docs scripts |
| `scripts/publish_canfar_wheel_bundle.sh` | Release/CI/bench/docs scripts |
| `scripts/release_lane.py` | Release/CI/bench/docs scripts |
| `scripts/release_notes.py` | Release/CI/bench/docs scripts |
| `scripts/render_bench_deficits.py` | Release/CI/bench/docs scripts |
| `scripts/render_bench_highlights.py` | Release/CI/bench/docs scripts |
| `scripts/render_bench_iopath_table.py` | Release/CI/bench/docs scripts |
| `scripts/render_bench_ml.py` | Release/CI/bench/docs scripts |
| `scripts/render_bench_quick.py` | Release/CI/bench/docs scripts |
| `scripts/render_full_benchmarks_table.py` | Release/CI/bench/docs scripts |
| `scripts/run_exhaustive_bench_and_patch_docs.sh` | Release/CI/bench/docs scripts |
| `scripts/selfcheck_canfar_launcher.sh` | Release/CI/bench/docs scripts |
| `scripts/sync_docs_examples.sh` | Release/CI/bench/docs scripts |
| `scripts/torch_lanes.json` | Release/CI/bench/docs scripts |
| `scripts/update_changelog.py` | Release/CI/bench/docs scripts |
| `scripts/verify_wheel_cuda_canfar.sh` | Release/CI/bench/docs scripts |
| `scripts/verify_wheel_cuda_canfar_incontainer.sh` | Release/CI/bench/docs scripts |
| `scripts/verify_wheel_matrix.sh` | Release/CI/bench/docs scripts |
| `src/torchfits/__init__.py` | Python package source |
| `src/torchfits/_cpp.py` | Python package source |
| `src/torchfits/_hdu/card.py` | Python package source |
| `src/torchfits/_hdu/dataview.py` | Python package source |
| `src/torchfits/_hdu/hdu_list.py` | Python package source |
| `src/torchfits/_hdu/header.py` | Python package source |
| `src/torchfits/_hdu/table_hdu.py` | Python package source |
| `src/torchfits/_hdu/table_hdu_ref.py` | Python package source |
| `src/torchfits/_hdu/tensor_hdu.py` | Python package source |
| `src/torchfits/_io_engine/__init__.py` | Python package source |
| `src/torchfits/_io_engine/_hdu_rewrite.py` | Python package source |
| `src/torchfits/_io_engine/_read_pipeline.py` | Python package source |
| `src/torchfits/_io_engine/_read_pipeline_fallback.py` | Python package source |
| `src/torchfits/_io_engine/_write_helpers.py` | Python package source |
| `src/torchfits/_io_engine/batch.py` | Python package source |
| `src/torchfits/_io_engine/caches.py` | Python package source |
| `src/torchfits/_io_engine/checksum_api.py` | Python package source |
| `src/torchfits/_io_engine/device.py` | Python package source |
| `src/torchfits/_io_engine/hdu_api.py` | Python package source |
| `src/torchfits/_io_engine/http_subset.py` | Python package source |
| `src/torchfits/_io_engine/image.py` | Python package source |
| `src/torchfits/_io_engine/image_meta.py` | Python package source |
| `src/torchfits/_io_engine/options.py` | Python package source |
| `src/torchfits/_io_engine/paths.py` | Python package source |
| `src/torchfits/_io_engine/quantize.py` | Python package source |
| `src/torchfits/_io_engine/subset.py` | Python package source |
| `src/torchfits/_io_engine/table_api.py` | Python package source |
| `src/torchfits/_io_engine/table_reader_api.py` | Python package source |
| `src/torchfits/_io_engine/table_streaming.py` | Python package source |
| `src/torchfits/_io_engine/write_api.py` | Python package source |
| `src/torchfits/_string_decode.py` | Python package source |
| `src/torchfits/_table/__init__.py` | Python package source |
| `src/torchfits/_table/_mutation_coerce.py` | Python package source |
| `src/torchfits/_table/_read_scan.py` | Python package source |
| `src/torchfits/_table/_read_schema.py` | Python package source |
| `src/torchfits/_table/_read_where.py` | Python package source |
| `src/torchfits/_table/arrow_convert.py` | Python package source |
| `src/torchfits/_table/cache.py` | Python package source |
| `src/torchfits/_table/engine.py` | Python package source |
| `src/torchfits/_table/interop.py` | Python package source |
| `src/torchfits/_table/mutation.py` | Python package source |
| `src/torchfits/_table/read.py` | Python package source |
| `src/torchfits/_table/utils.py` | Python package source |
| `src/torchfits/_table/write.py` | Python package source |
| `src/torchfits/_table_engine/__init__.py` | Python package source |
| `src/torchfits/_table_engine/backend_policy.py` | Python package source |
| `src/torchfits/_table_engine/read_policy.py` | Python package source |
| `src/torchfits/_tensor_buffer.py` | Python package source |
| `src/torchfits/_where.py` | Python package source |
| `src/torchfits/cache.py` | Python package source |
| `src/torchfits/cli/__init__.py` | Python package source |
| `src/torchfits/cli/__main__.py` | Python package source |
| `src/torchfits/cli/cmds_arith.py` | Python package source |
| `src/torchfits/cli/cmds_compress.py` | Python package source |
| `src/torchfits/cli/cmds_convert.py` | Python package source |
| `src/torchfits/cli/cmds_copy.py` | Python package source |
| `src/torchfits/cli/cmds_cutout.py` | Python package source |
| `src/torchfits/cli/cmds_diff.py` | Python package source |
| `src/torchfits/cli/cmds_header.py` | Python package source |
| `src/torchfits/cli/cmds_info.py` | Python package source |
| `src/torchfits/cli/cmds_probe.py` | Python package source |
| `src/torchfits/cli/cmds_setkey.py` | Python package source |
| `src/torchfits/cli/cmds_stats.py` | Python package source |
| `src/torchfits/cli/cmds_table.py` | Python package source |
| `src/torchfits/cli/cmds_transform.py` | Python package source |
| `src/torchfits/cli/cmds_verify.py` | Python package source |
| `src/torchfits/cli/common.py` | Python package source |
| `src/torchfits/cli/main.py` | Python package source |
| `src/torchfits/cpp.py` | Python package source |
| `src/torchfits/cpp_src/CMakeLists.txt` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/bindings.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/cache.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/cache.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/fits_bindings.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/fits_detail.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/fits_file.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/fits_file.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/fits_rw.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/hardware.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/hardware.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/internal_utils.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/security.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/table_bindings.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/table_ops.cpp` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/table_ops.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/table_reader.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/table_types.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/torch_compat.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/cpp_src/torchfits_torch.h` | C++ native engine (hot-path + structural; megafiles targeted) |
| `src/torchfits/data/__init__.py` | Python package source |
| `src/torchfits/data/datasets.py` | Python package source |
| `src/torchfits/data/remote.py` | Python package source |
| `src/torchfits/fits_schema.py` | Python package source |
| `src/torchfits/hdu.py` | Python package source |
| `src/torchfits/header_parser.py` | Python package source |
| `src/torchfits/http_util.py` | Python package source |
| `src/torchfits/interop.py` | Python package source |
| `src/torchfits/io.py` | Python package source |
| `src/torchfits/logging.py` | Python package source |
| `src/torchfits/py.typed` | Python package source |
| `src/torchfits/table.py` | Python package source |
| `src/torchfits/transforms/__init__.py` | Python package source |
| `src/torchfits/transforms/base.py` | Python package source |
| `src/torchfits/transforms/clip.py` | Python package source |
| `src/torchfits/transforms/fits_meta.py` | Python package source |
| `src/torchfits/transforms/helpers.py` | Python package source |
| `src/torchfits/transforms/normalize.py` | Python package source |
| `src/torchfits/transforms/rgb.py` | Python package source |
| `src/torchfits/transforms/stretch.py` | Python package source |
| `src/torchfits/vos_uri.py` | Python package source |
| `src/torchfits/where.py` | Python package source |
| `tests/conftest.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/cpp/test_bracket_detection.cpp` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_api.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_arrow_table_api.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_ascii_table.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_astropy_upstream_smoke.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_bench_ranking_mmap.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_bench_suites.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_bug_table_hdu_cache.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_bug_table_ref.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_byteswap.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_bz2.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_cache.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_cache_config.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_changelog_tooling.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_check_torch_extra_pins.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_checksum.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_clear_all_caches.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_cli.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_cli_release_fixes.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_complex_header.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_compressed_nulls.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_compression.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_compression_matrix.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_concurrent_same_file_read.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_cutout_performance_api.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_data.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_data_datasets.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_deep_review_p0.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_deep_review_wave2.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_deep_review_wave4.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_deep_review_wave5.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_dlpack_roundtrip.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_docs_code_snippets.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_docs_integrity.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_examples_runner.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_fits_schema.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_fitsio_upstream_smoke.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_hdu.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_hdu_file_ops.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_hdu_str.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_header_versioning.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_http_probe_fixture.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_integration.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_interop.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_interop_import.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_io.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_malformed_fits.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_mps.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_multichunk_buffered_read.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_no_external_fits_backends.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_open_table_reader.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_output_parity.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_package_isolation.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_patch_bench_docs.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_performance.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_public_boundary.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_public_where.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_quantize_int16.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_read_header.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_read_policy.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_release_lane.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_release_semantics.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_release_smoke.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_remote_http_range.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_rgb.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_scale_on_device.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_security.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_security_eval.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_security_fix.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_skinny_meta.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_staged_prefetch.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_subset_3d.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_table.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_table_docs_smoke.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_table_file_ops.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_table_filtering.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_transforms.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_transforms_e2e.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_transforms_typing.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_truncated_table_errors.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_upstream_parity_inventory.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_validation.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_where.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_write_fidelity.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/test_writing.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/transforms_reference.py` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `tests/transforms_reference.pyi` | Test suite (inventory + targeted deep reads of contract/security/malformed) |
| `zensical.toml` | Repo root / packaging metadata |
