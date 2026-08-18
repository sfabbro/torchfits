---
name: torchfits-dev
description: Develop torchfits (FITS I/O for PyTorch) — tensor/table contracts, C++ CFITSIO engine, verify tiers, docs contract
metadata:
  project: torchfits
  stack: python, c++, pytorch, pixi
---

# torchfits development

FITS I/O for PyTorch: read/write FITS as tensors and tables, cutouts, predicate
filters, and a shell CLI, backed by a C++ engine with vendored CFITSIO. GPU
acceleration (CUDA on Linux, MPS on macOS) when available. See `AGENTS.md`.

## Verify tiers

| When | Command |
|---|---|
| During edits | `pixi run preflight-push` |
| Before push / PR | `pixi run ci-local` |
| Before tag | `pixi run release-gate` |

## Contract rules

- Docs must match the public façade (`docs/api*.md`); env vars must exist in
  `src/`. Run `pixi run docs-contract` and `pixi run docs-links` on docs changes.
- User-facing docs: human-first, copy-pasteable recipes; zero internal jargon.
- Correctness is paramount: parity against the CFITSIO/public API contract.
- Prefer smallest correct diffs; no new dependencies without a clear need.
- Performance claims need before/after timing from an existing `pixi run bench-*`
  case (same host, same `case_id`).

## Surface

- `torchfits.read_tensor("img.fits", device="cuda")` / `write` / `open` (MEF)
- `torchfits.table.read(..., where="MAG < 20")` / `read_torch`
- `FitsImageDataset` + `make_loader(...)`
- CLI: `torchfits info` / `header` / `convert` / `cutout` / …
