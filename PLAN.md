# torchfits v1.0 TODO (updated 2025-08-13)

A concise, actionable checklist of remaining work for v1.0. Finished items have been removed from this plan.

## Performance

- [x] [P1] Small cutouts (≤32×32): reduce Python-call overhead and slice setup costs; confirm optimal mmap/buffered choice for tiny windows.
	- Implemented: auto small-cutout optimization in `read_multi_cutouts` that reads full image once and slices for tiny uniform windows; fallback to C++ batched path.
		- Bench (256, 10×32²): torchfits full→slice 0.15–0.17 ms; multi-cutout (batched) ~0.27 ms; astropy ~0.64–0.69 ms; fitsio ~0.34–0.36 ms.
		- Micro-bench (CPU, size=512–1024):
			- 500×16²: loop 43.8 ms vs batched 41.5 ms (~1.06×)
			- 1000×16²: loop 85.7 ms vs batched 84.4 ms (~1.02×)
			- 500×8²: loop 41.7 ms vs batched 41.5 ms (~1.01×)
			- 500×32² @1024: ~parity
	- Notes: path currently enabled for sequential, uniform 2D small windows; further parallel reuse is optional follow-up.
- [x] [P2] Sky cutouts (small radii, e.g., 15"): batched WCS + prefer single full read then slice; benchmarked wins at small radii.
	- Implemented: `read_multi_sky_cutouts(path, world_points, radius_arcsec, ...)` batches WCS (wcslib) and for small radii reads once then slices; falls back to per-cutout reads with precomputed WCS.
	- Bench (size=256, radius=15"): torchfits WCS→multi 0.43–0.47 ms; astropy 0.37–0.38 ms; fitsio ~0.38–0.39 ms; torchfits per-cutout ~0.97–1.00 ms.
	- Bench (size=512, radius=30"): torchfits WCS→multi 0.36–0.45 ms; astropy ~0.41–0.42 ms; fitsio ~1.15 ms; torchfits per-cutout ~1.00 ms.
	- Bench (size=1024, radius=60"): torchfits WCS→multi 0.55–0.57 ms; astropy ~0.46 ms; still faster than torchfits per-cutout (~0.92–1.00 ms) and far ahead of fitsio (~4.6 ms). Consider auto threshold tuning later.
	- Added unit test `tests/test_sky_cutouts.py` to validate correctness vs direct slicing using the same WCS transform.
- [x] [P3] Heuristics: auto select mmap/buffered for full-image reads based on size/compression; unit-tested.
	- Implemented: `heuristics.choose_read_mode_for_image` and integration in `read()` when flags are not set and `start/shape` are None.
	- Behavior: compressed → buffered; large uncompressed (≥ TORCHFITS_MMAP_MIN_MB, default 50MB) → mmap; else default path. Backend safely falls back.
	- Tests: `tests/test_heuristics.py` validates chooser flags and smoke for full-image reads.
	- Bench: image full read remains fastest-in-class; 256 ~0.11–0.13 ms, 512 ~0.13 ms, 1024 ~0.25–0.29 ms. Auto flags do not affect cutouts.
- [ ] [P4] Table frameworks gap: profile torchfits table/dataframe vs fitsio/astropy; optimize hot paths (conversion/boxing) to close ~10% gap.
	- In progress:
		- Implemented: DataFrame fast path when `return_metadata=False` via direct tensor-dict→torch-frame; reduced unnecessary casts/copies.
		- Implemented: relax numeric column read parallelism (≥4 scalar numeric cols and ≥100k rows) and raise cap to 8 threads.
	- Current results (rows=200k):
		- Column subset (4 cols): torchfits tensor 14.6–16.1 ms; dataframe 15.6–16.0 ms; fitsio 43–46 ms; astropy 66–71 ms.
		- Frameworks (full table): torchfits table 18.5–21.0 ms; dataframe 19.0–21.5 ms; fitsio 18.0–19.9 ms; astropy.Table 4.3–5.2 ms.
	- Next steps:
		- Profile numeric-only full-table path vs fitsio; experiment with adaptive chunking/threading by row/col shape.
		- Measure metadata extraction cost for DataFrame mode; consider caching header parsing where safe.
		- Maintain parity of DataFrame vs tensor-dict path; add micro-bench in `benchmarks/`.
- [ ] [P5] CI gate for perf targets: wire analyzer to assert median ≥1.2× vs baselines and no case worse by >10% (configurable exceptions).

## Features & Parity

- [ ] [F1] Variable-length arrays: multi-column writer support; expand fuzz to multi-column ragged patterns; verify read/write symmetry.
- [ ] [F2] Strings & nulls: expand edge-case tests (mixed empty/padded, long lengths); confirm TNULL propagation in more schemas.
- [ ] [F3] Parity matrix: complete idioms set, reach ≥95% green; publish artifact in CI and link from docs.
- [ ] [F4] Schema round-trip: include units/meta in FITS ↔ torch-frame ↔ FITS tests; document any intentional lossy fields.
- [ ] [F5] Advanced joins: decide on native vs torch-frame delegated joins; implement or explicitly defer with rationale in docs.
- [ ] [F6] BITPIX-16 floats: add smart conversion from a FITS file with BITPIX=16 to a float arrray, beyond a native pytorch 16bits type, that takes into consideration
	- scaling (BSCALE/BZERO), quantization, and appropriate dtype selection (output float32 by default), with tests documenting precision trade-offs.

## Remote & ML (Smart Cache and Datasets)

- [ ] [R1] Prefetch: formal throughput benchmark and tuning; target ≥1.15× speedup on IO-bound loop; expose knobs (depth, threads) in docs.
- [ ] [R2] Epoch-aware cache management: simple policy (keep last N epochs) with tests; document behavior and overrides.
- [ ] [R3] Fault injection suite: latency spikes, disconnects, partial files; assert bounded retries and integrity recovery.

## Benchmarks & Plots

- [ ] [B1] CI job: run benchmark smoke, append JSONL, generate plots via `benchmarks/plot_full_sweep.py`, upload artifacts.
- [ ] [B2] Docs integration: add a short “Performance” page referencing the latest plots in `artifacts/benchmarks/plots/`.

Notes (2025-08-13 sweep)

- Full sweep re-run and plots regenerated.
	- Results JSONL: `artifacts/benchmarks/full_sweep.jsonl` (appended)
	- Plots: `artifacts/benchmarks/plots/` (4 figures)
	- Highlights: torchfits leads on full image reads and small/medium cutouts; sky-cutout WCS batched path competitive (wins at moderate radii), and table column subset significantly faster than numpy-based stacks; full-table frameworks near fitsio with a ~5–10% gap pending P4.

## Developer Experience & Tooling

- [ ] [T1] Type coverage report (≥95% for public API): implement generator, publish JSON under `artifacts/types/coverage.json`, add CI threshold gate.
- [ ] [T2] Build & distribution: produce wheels (manylinux, macOS universal2, Windows) + sdist; install-test matrix; ABI/import smoke.
- [ ] [T3] Release automation: conventional commits → changelog; RC pipeline attaches benchmark/validation artifacts; publish to TestPyPI→PyPI.
- [ ] [T4] Naming enforcement: keep regex gate in pre-commit/CI; add docs note on naming canon.

## Documentation & Examples

- [ ] [D1] README simplification (<300 lines) with 10-line quickstart; move details to docs site.
- [ ] [D2] Docs site: quickstart, guides (data access, remote/cache, writing/updating, torch-frame), API reference (auto from docstrings).
- [ ] [D3] Example gallery: ≥8 runnable scripts; execute in CI; include minimal ML loop.
- [ ] [D4] Migration guide: astropy/fitsio → torchfits; acceptance notebooks that pass end-to-end.

## AI Guidance Readiness

- [ ] [A1] Stable API manifest for assistant tooling (auto-generated) and kept current in CI.
- [ ] [A2] Prompt seeds in docs for common tasks (image read, table to tensor, remote cache usage, cutouts).

## Release Gates (must be green for v1.0)

- [ ] [G1] Performance targets: median ≥1.2× vs fitsio/astropy; no case worse by >10% (with documented exceptions).
- [ ] [G2] Memory efficiency: peak RSS ≤85% of astropy on large table read.
- [ ] [G3] Cross-platform CI: Linux, macOS (Intel/ARM), Windows; plus one CUDA runner for GPU path.
- [ ] [G4] Remote training resilience: cache hit ≥90% by epoch 2 under injected faults; integrity maintained.

Notes

- Keep benchmark artifacts in `artifacts/benchmarks/*.jsonl` and plots under `artifacts/benchmarks/plots/`.
- Prefer `pixi run ...` tasks in CI to ensure local/CI parity.
