# Plan for torchfits 1.0

## 🎯 Core Mission

To provide the fast, intuitive, and PyTorch-native library for accessing FITS data, empowering machine learning applications in astronomy, ready for modern very large scale training from astronomy archives.

---

## 🌟 Guiding Principles

1. **Performance First**: Outperform existing libraries like `astropy.io.fits` and `fitsio` in all common FITS use cases, from small images, spectra and tables, to massive data cubes, complex MEFs, and billion-rows tables.
2. **PyTorch Native**: Data should be read directly into `torch.Tensor` objects, ready for GPU acceleration and ML pipelines, without intermediate copies. Includes specialized types of FITSDataset and iterable ones
3. **Familiar and full functional API**: Provide a user-friendly and familiar API for astronomers accustomed to `astropy.io.fits` and `fitsio`, with feature parity
4. **Seamless `pytorch-frame` Integration**: Treat FITS tables as first-class citizens in the PyTorch ecosystem, with deep and meaningful integration with `pytorch-frame`.
5. **Robust Remote Access**: Make working with remote datasets (HTTP, S3, etc.) as simple and efficient as local files, with an intelligent caching system optimized for ML training.

### 🔍 Verification Strategy (Principle → Evidence at v1.0)

| # | Principle | How We Will Verify (Automated Unless Noted) |
|---|-----------|---------------------------------------------|
| 1 | Performance First | Benchmark suite (tables, images, cubes, MEFs, cutouts incl. compressed) shows ≥1.2× median speed vs `fitsio` & `astropy`; no case slower by >10%; peak RSS ≤85% of `astropy` on large table read. CI publishes JSON + plots. |
| 2 | PyTorch Native | Zero (or single unavoidable) host copy before user-visible tensor (instrumentation counters); pinned + async GPU pipeline ≥1.3× faster host→GPU transfer vs naive baseline; ≥95% code paths return tensors without numpy intermediate. |
| 3 | Familiar API | Parity matrix of top 40 astropy/fitsio idioms ≥95% green; migration guide & two acceptance notebooks pass in CI with only import changes. |
| 4 | `pytorch-frame` Integration | Semantic type inference precision & recall ≥0.95 on labeled corpus; round‑trip FITS → DataFrame → FITS preserves schema & values (max abs diff <1e-7). |
| 5 | Robust Remote Access | Cached remote read overhead ≤10%; training sim cache hit ≥90% by epoch 2; fault injection (latency, disconnect, partial) completes epochs with integrity (hash match) & bounded retries. |

Artifacts emitted under `artifacts/validation/` during release CI prior to tagging v1.0.

---

## (Current Status - v0.2)

`torchfits` has successfully established a strong foundation, delivering on key initial promises.

* ✅ **High-Performance Core**: C++ backend leveraging CFITSIO for direct-to-tensor reading, already outperforming competitors in many benchmarks.
* ✅ **Enhanced Table Operations**: The `FitsTable` class provides a powerful, pure-PyTorch, pandas-like interface for table manipulation.
* ✅ **Basic Remote & Cache Support**: Foundational infrastructure for reading from URLs and caching files is in place.
* ✅ **WCS Utilities**: Core WCS transformations are supported.
* ✅ **Initial `torch-frame` Integration**: FITS tables can be converted to `torch_frame.DataFrame` objects.
* ✅ **Familiar API**: `read()` function and `FITS`/`HDU` objects offer an interface that is intuitive for users of existing FITS libraries.

---

## 🚀 Roadmap to v1.0

The path to v1.0 is focused on three pillars: **Performance**, **Complete Feature Set**, and **Intelligent Data Handling**.

### Pillar 1: Performance

**Goal**: Make `torchfits` the undisputed performance leader for FITS I/O in Python.

**Key Initiatives**:

1. **Advanced CFITSIO Integration**:
    * **Memory Mapping**: Implement memory-mapped access to large local files, minimizing memory overhead and I/O. Use low-level CFITSIO routines
    * **Buffered I/O**: Optimize buffer sizes and use CFITSIO's tile-based access for significant speedups on compressed files and cutouts.
    * **Iterator Functions**: Use iterators for highly efficient table column processing, especially for row-wise filtering and transformations.

2. **Parallelization**:
    * **Multi-threaded Column Reading**: Read multiple table columns in parallel using thread-safe CFITSIO calls.
    * **Parallel HDU Processing**: Develop strategies for parallel reading of multiple HDUs from a single MEF file.
    * **Benchmark Harness**: Unified runner in `benchmarks/` producing machine-readable JSON + HTML diff vs baselines.

3. **GPU-Direct Pipeline**:
    * **Pinned Memory**: Read data into pinned (page-locked) memory to enable faster, asynchronous CPU-to-GPU transfers.
    * **CUDA Streams**: Overlap data reading and GPU transfers using CUDA streams to hide latency in ML data loading pipelines.
    * **Latency Profiling**: Automated test measuring disk/remote → GPU tensor latency vs numpy→tensor baseline (assert speedup).

### Pillar 2: Complete Feature Set & API Parity

**Goal**: Achieve functional parity with the most critical features of `astropy.io.fits` and `fitsio`, ensuring users can fully migrate to `torchfits`.

**Key Initiatives**:

1. **Writing and Updating FITS Files**:
    * Implement `torchfits.write()` to save tensors and `FitsTable` objects to new FITS files.
    * Support for updating existing FITS files (in-place modification of data, appending HDUs).
    * Header manipulation: Add, remove, and update header keywords.
    * **Round-Trip Tests**: Write→read retains header semantics (except normalized keywords) & numeric fidelity.

2. **Expanded FITS Standard Support**:
    * **Compressed Images**: Natively handle FITS images compressed with Rice, GZIP, and HCOMPRESS.
    * **Variable-Length Arrays**: Support for reading table columns with variable-length arrays, a common feature in astronomical catalogs.
    * **Random Groups**: Support for this legacy but still-present FITS format.

3. **Enhanced `FitsTable` Functionality**:
    * **String Column Support**: More robust and efficient handling of string columns.
    * **Advanced Joining**: Implement more complex join operations (`outer`, `right`) between `FitsTable` objects.
    * **Missing Data**: More explicit handling of null values (`TNULL`) during read and in `FitsTable` operations.
    * **Parity Matrix Automation**: Script builds Markdown/HTML matrix mapping astropy/fitsio operations to torchfits coverage & test status.

4. **API Familiarity & Migration**:
    * **Idioms Coverage**: Curate top 40 workflows (open, list HDUs, slice, cutouts, compressed read, header edit, WCS, joins, var-length arrays) with side-by-side examples.
    * **Migration Guide**: Document direct translation patterns and caveats.
    * **User Feedback Loop** (stretch): Lightweight usability test with 3–5 external users pre-RC.

### Pillar 3: Intelligent Data Handling (Remote & ML)

**Goal**: Create a best-in-class experience for large-scale, remote datasets, specifically tailored for ML training workflows.

**Key Initiatives**:

1. **ML-Optimized Smart Cache**:
    * **Training-Aware Prefetching**: Implement a `FITSDataset` that intelligently prefetches the next files needed for training based on the dataloader's access pattern.
    * **Epoch-Aware Cache Management**: The cache should understand the concept of training epochs, keeping data for recent epochs and evicting older data.
    * **Cache Resiliency**: Implement checksum verification, automatic cleanup of corrupted files, and robust error handling for network failures.
    * **Fault Injection Suite**: Simulate latency spikes, disconnects, partial downloads; assert recovery, integrity, bounded retries.

2. **Deep `torch-frame` Integration**:
    * **Automatic `stype` Inference**: Automatically map FITS column metadata (units, keywords) to `torch_frame` semantic types (`stype.numerical`, `stype.categorical`, etc.). Develop new astronomy-specific `stypes` if needed (e.g., `stype.celestial_coord`).
    * **Bi-directional Conversion**: Enable seamless conversion from a `torch_frame.DataFrame` back to a `FitsTable` or FITS file.
    * **WCS in DataFrames**: Explore methods to associate WCS information with tables in a `torch-frame` context.
    * **Inference Accuracy Report**: CI job computes precision/recall & confusion matrix.

3. **Production-Ready Usability**:
    * **Comprehensive Documentation**: Create a documentation portal with a user guide, API reference, and a gallery of examples for common astronomy ML tasks.
    * **Actionable Error Messages**: Provide clear, helpful error messages that guide the user to a solution.
    * **Cross-Platform CI**: Rigorous testing on Linux, macOS (Intel & Apple Silicon), and Windows to ensure reliability.
    * **Tutorial Execution Tests**: Example notebooks executed headlessly in CI (smoke + key output checksums).
    * **Error Snapshot Tests**: Selected exceptions captured & asserted to include actionable guidance.

---

### Pillar 4: Developer Experience & Tooling

**Goal**: Make contributing safe, fast, reproducible and stylistically consistent so velocity scales with adoption.

**Key Initiatives**:

1. **pixi Environment & Reproducibility**
    * Single source of dependency truth: `pyproject.toml` + `pixi.lock` updated via scripted task.
    * CI jobs (lint, test, build, benchmarks-smoke) all run via `pixi run ...` ensuring parity with local dev.
    * Daily cron job verifies environment resolution (alerts on upstream breakage / yanked packages).

2. **Build & Distribution Pipeline**
    * Automated wheel + sdist build (manylinux, macOS universal2, Windows) using maturin / setuptools (decide & document rationale).
    * Install test matrix: fresh virtual env installs from (a) source checkout, (b) built wheel, (c) published artifact candidate.
    * ABI sanity: minimal import timing + functionality smoke test post-wheel-build (opens sample FITS, reads HDU to tensor).

3. **Versioning & Release Automation**
    * Conventional commits (subset) → automated changelog generation.
    * Release workflow tags v1.0 RCs, attaches benchmark & validation artifacts, pushes to TestPyPI then PyPI after gate.

4. **Naming Consistency Enforcement**
    * Canon: package name is `torchfits` (lowercase); public class names use ALLCAPS only for FITS standard nouns: `FITS`, `HDU`, `WCS`; otherwise `CamelCase`; functions & modules are `snake_case`.
    * Ban forms: `TorchFits`, `FitsTable` (unless already shipped and deprecated), inconsistent case variants.
    * Tooling: pre-commit hook + ruff custom rule (regex scan) failing on banned tokens; migration script to auto-rename legacy symbols.
    * Deprecation policy: alias old names with `DeprecationWarning` for >=1 minor release.

5. **Static Quality Gates**
    * Linters: ruff (style, complexity), mypy (strict on core), clang-format for C/C++ bindings.
    * Type coverage target ≥95% for Python public API modules.
    * Enforce docstring presence on all public callables (numpydoc style subset) via pydocstyle or ruff rule.

### Test & Example Strategy

**Goal**: High confidence (functional, performance, regression) with approachable, ML-focused examples.

**Test Layers**:

* Unit: pure functions & small classes (fast, <200ms each) — run on every push.
* Integration: reading/writing varied FITS (compressed, var-length, MEF) — parallelized shard.
* Performance: micro (column read, cutout) & macro (end-to-end dataset) — sampled in PR (smoke) + full on nightly.
* Property-based: header round-trips, table joins invariants (hypothesis) for randomized edge cases.
* Fault Injection: network interruptions, partial content, cache evictions.
* GPU Path: pinned memory transfer + async pipeline behavior.

**Coverage Targets**:

* Line ≥90% (excluding generated bindings), branch ≥85%, critical modules (I/O core) ≥95%.
* Golden data fixtures hashed; fixture drift fails CI unless explicitly re-approved.

**Examples & Notebooks**:

* Gallery: minimal runnable scripts: basic read, table ops, WCS transform, cutouts, datacube slice, remote caching, torch-frame integration, ML training loop.
* Each example under 60 lines (excluding comments) with concise narrative.
* Notebooks executed in CI → stored executed copies (strip large outputs) for docs site.

### Documentation Revamp Plan

1. **README Simplification**
    * Target: <300 lines, focused on value prop, quickstart (10 lines), features table, why-not-astropy section, roadmap link.
    * Defer deep content to `/docs` site (mkdocs / sphinx + myst) with clear navigation.

2. **Basic Usage Docs**
    * Quickstart: install, open FITS, read image/table to tensor, minimal training loop.
    * Data Access Guide: images vs tables vs cubes vs MEFs.
    * Remote & Caching Guide.
    * Writing & Updating Guide.
    * torch-frame Integration Guide.
    * Migration Guide (astropy/fitsio → torchfits).

3. **API Reference**
    * Auto-generated from docstrings; ensure stable ordering & summary one-liners for AI parsing.

4. **Changelog & Upgrade Notes**
    * Machine-readable (Keep a Changelog + JSON) to allow automated assistant summarization.

### AI Prompting Reference Appendix

Provide a concise, machine-readable summary to help AI tools generate accurate code:

```text
PACKAGE_NAME: torchfits
PRIMARY_CLASSES: FITS, HDU, FitsTable (may rename to FITSBaseTable?), WCS
DATASET_CLASSES: FITSDataset (map-style), StreamingFITSDataset (iterable)
CORE_FUNCTIONS: read(), write(), open(), cutout(), to_torch_frame(), from_torch_frame()
REMOTE_PROTOCOLS: file, http(s), s3
CACHE_DIR_ENV: TORCHFITS_CACHE
NAMING_CANON: package lowercase, functions snake_case, classes CamelCase except FITS/HDU/WCS uppercase per standard.
PERFORMANCE_TARGETS: median ≥1.2x fitsio & astropy; remote overhead ≤10%; GPU pipeline ≥2x naive throughput.
DOCS_STRUCTURE: README (simplified), docs/quickstart.md, docs/guide/*.md, docs/api/*, examples/*.py
TEST_CATEGORIES: unit, integration, performance, property, fault, gpu
BENCHMARK_OUTPUT_FMT: JSON lines, key fields {"benchmark": str, "case": str, "wall_time_ms": float, "bytes": int}
```

This appendix must stay current; CI check verifies canonical tokens present & unchanged or updated intentionally.


## ✅ v1.0 Success Checklist

### Performance

* [ ] **Tables / Images / Cubes / MEFs**: ≥1.2× median speed vs `fitsio` & `astropy`; no case worse by >10%.
* [ ] **Memory Efficiency**: Peak RSS ≤85% of `astropy` in large table benchmark.
* [ ] **Remote (Cached)**: Overhead ≤10% vs local.
* [ ] **Cutouts & Compressed**: Cutout / compressed decode ≥1.3× speed vs `astropy`.
* [ ] **GPU Workflows**: `FITSDataset` (pinned + async) ≥2× throughput vs naive astropy→numpy baseline.

### Features

* [ ] **Writing**: Images & tables (var-length arrays, strings, compressed) round-trip (diff <1e-7, headers preserved except normalized keywords).
* [ ] **Parity Matrix Green**: ≥95% targeted operations implemented & tested.
* [ ] **Strings & Nulls**: All string & TNULL scenarios covered.
* [ ] **Variable-Length Arrays**: Read/write correctness tests pass.
* [ ] **`torch-frame`**: DataFrame mode ≥95% correct stype inference.
* [ ] **Round-Trip DataFrame**: DataFrame → FITS → DataFrame preserves schema & values.

### Robustness

* [ ] **Remote Training**: Multi-epoch remote job with injected failures completes; cache hit ≥90% by epoch 2; no corruption.
* [ ] **Error Messages**: Snapshot tests include actionable guidance lines.
* [ ] **Documentation**: Tutorials execute & reach expected baseline metrics (e.g., sample model accuracy threshold).
* [ ] **Cross-Platform**: All tests pass on Linux, macOS (Intel & ARM), Windows; GPU tests on at least one CUDA runner.
* [ ] **Security / Integrity**: Checksums validated; tamper test fails fast with clear error.

### Principle Closure Mapping

* [ ] P1 Performance benchmarks published & thresholds met.
* [ ] P2 PyTorch native zero-copy / GPU pipeline tests green.
* [ ] P3 API parity matrix ≥95% & migration guide published.
* [ ] P4 `pytorch-frame` inference & round-trip tests green.
* [ ] P5 Remote cache & fault injection tests green.

### Developer Experience & Tooling

* [ ] **pixi Repro**: All CI tasks invoked via `pixi run`; nightly env resolution check passes.
* [ ] **Build Matrix**: Wheels (Linux manylinux, macOS universal2, Windows) + sdist build & install smoke pass.
* [ ] **Artifact Integrity**: Post-build import + basic read benchmark succeeds for each wheel.
* [ ] **Version Automation**: Tag → changelog + release assets (benchmarks, validation) attach automatically.
* [ ] **Naming Enforcement**: No banned identifiers (`TorchFits`, `FitsDataset`, etc.) in codebase (regex gate green).
* [ ] **Type Coverage**: ≥95% public API modules; report published.

### Documentation & Examples

* [ ] **README Simplified**: <300 lines, quickstart ≤10 lines code.
* [ ] **Docs Site**: Quickstart + Guides + API published; internal link checker passes.
* [ ] **Example Gallery**: ≥8 concise runnable examples executed in CI.
* [ ] **Migration Guide**: Completed & validated by acceptance notebooks.
* [ ] **Changelog JSON**: Machine-readable changelog generated on release.

### AI Guidance Readiness

* [ ] **AI Appendix Present**: Appendix block passes token validation script.
* [ ] **Stable API Summary**: Auto-generated API manifest consumed by assistant tooling.
* [ ] **Prompt Seeds**: Curated prompt examples (in docs) for common tasks (read image, table to tensor, remote cache usage).
* [ ] **No Ambiguous Names**: Lint ensures canonical casing & naming invariants.

This plan outlines a clear path to establishing `torchfits` as the essential, high-performance tool for astronomical data analysis in the modern machine learning era.
