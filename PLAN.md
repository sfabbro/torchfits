# torchfits v1.0 TODO (updated 2025-08-13)

A concise, actionable checklist of remaining work for v1.0. Finished items have been removed from this plan.

## Performance

- [ ] [P1] Small cutouts (≤32×32): reduce Python-call overhead and slice setup costs; confirm optimal mmap/buffered choice for tiny windows.
- [ ] [P2] Sky cutouts (small radii, e.g., 15"): profile remaining overhead after WCS batching; consider lightweight header/WCS reuse where safe.
- [ ] [P3] Heuristics: finalize auto selection (mmap vs buffered) by workload size/compression; add unit tests around the chooser.
- [ ] [P4] Table frameworks gap: profile torchfits table/dataframe vs fitsio/astropy; optimize hot paths (conversion/boxing) to close ~10% gap.
- [ ] [P5] CI gate for perf targets: wire analyzer to assert median ≥1.2× vs baselines and no case worse by >10% (configurable exceptions).

## Features & Parity

- [ ] [F1] Variable-length arrays: multi-column writer support; expand fuzz to multi-column ragged patterns; verify read/write symmetry.
- [ ] [F2] Strings & nulls: expand edge-case tests (mixed empty/padded, long lengths); confirm TNULL propagation in more schemas.
- [ ] [F3] Parity matrix: complete idioms set, reach ≥95% green; publish artifact in CI and link from docs.
- [ ] [F4] Schema round-trip: include units/meta in FITS ↔ torch-frame ↔ FITS tests; document any intentional lossy fields.
- [ ] [F5] Advanced joins: decide on native vs torch-frame delegated joins; implement or explicitly defer with rationale in docs.

## Remote & ML (Smart Cache and Datasets)

- [ ] [R1] Prefetch: formal throughput benchmark and tuning; target ≥1.15× speedup on IO-bound loop; expose knobs (depth, threads) in docs.
- [ ] [R2] Epoch-aware cache management: simple policy (keep last N epochs) with tests; document behavior and overrides.
- [ ] [R3] Fault injection suite: latency spikes, disconnects, partial files; assert bounded retries and integrity recovery.

## Benchmarks & Plots

- [ ] [B1] CI job: run benchmark smoke, append JSONL, generate plots via `benchmarks/plot_full_sweep.py`, upload artifacts.
- [ ] [B2] Docs integration: add a short “Performance” page referencing the latest plots in `artifacts/benchmarks/plots/`.

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
