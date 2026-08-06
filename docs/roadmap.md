# Roadmap

`torchfits` is FITS I/O for tensors, tables, and the shell. This page is
forward-looking only.

## Current focus (toward 1.0.0)

**`1.0.0rc5`** is on PyPI with manylinux + macOS arm64 wheels
(`pip install "torchfits==1.0.0rc5"`). Collaborator soak is in progress; SemVer
**`1.0.0`** waits for feedback. Scope cuts (cache merge, dataset zoo, CLI trim)
stay deferred (repo `.cursor/post-1.0-backlog.md`) and do not block the tag.

## Parity tiers

| Tier | Meaning |
|------|---------|
| Public contract | Docs and examples match the implemented surface |
| fitsio / Astropy workflows | Common read/write/header/checksum/compression paths interoperate |
| Selected CFITSIO | Documented where we expose CFITSIO-backed semantics (1.x) |
| Non-goals | Full CFITSIO API parity; WCS / sphere / HEALPix / sky simulation |

## Near-term (through 1.0.0)

- Soak `1.0.0rc5` with collaborators; fold feedback before SemVer `1.0.0`
- Keep published exhaustive scorecards honest (tensor vs table, CPU↔GPU deficits,
  CFITSIO-direct rows, MegaCam cutouts) — see [Benchmarks](benchmarks.md)
- Docs / CLI polish that lands with 1.0 (install recipes, `header` dump quality,
  contributor + release checklists)
- Pre-tag public-API freeze review (repo
  `.cursor/skills/release-api-freeze-review/`)

## 1.1 candidates (still on CFITSIO)

- Richer compressed write coverage and public-file replay snapshots
- Broader Astropy table / ASCII / VLA cases where they stay small and tested
- Selected CFITSIO leftovers from the 2026-07 audit only when scorecards show a
  deficit (quantize / HCOMPRESS setters, cold wide-table metadata, flush between
  large multi-HDU writes)

## 2.0 — native engine, no CFITSIO

**Goal:** drop the vendored CFITSIO dependency and ship a torchfits-owned FITS
reader/writer with a **GPU-direct** path (disk/object store → device memory
without a mandatory host bounce).

Expected shape (subject to design spikes):

- Own container / HDU / tile codec stack for the image and table paths we
  already support in 1.x
- GPUDirect Storage (or equivalent) where the platform allows; clean host
  fallback everywhere else
- Stable Python / CLI façade — 2.0 may change engine internals and some
  CFITSIO-shaped edge semantics, not the everyday `read_tensor` /
  `table.read` / CLI inventory story without a migration note

Until 2.0 lands, GPU placement remains “read on host, `.to(device)` inside the
engine” (same as today’s `device=` argument).

## Deferred (cosmetic / low priority)

From the rc audits (not blocking 1.0):

- F1–F5 stylistic / macOS sysctl / CTYPE5 / END HISTORY notes
- C3 fold `insert_rows`+`update_rows` into one CFITSIO open (perf only; moot if
  2.0 replaces the backend)
- M1/M4/M7 / L2/L4 / T* / B* hygiene from prep review — see post-1.0 backlog

## Permanent design choices (not gaps)

- VLA and scaled-column **mmap** updates stay on the buffered path (format / safety).
- PyArrow is the table runtime; Polars/Pandas/DuckDB remain optional.
- Concurrent reads use **private** `fitsfile*` handles while on CFITSIO (R2); do
  not reintroduce a shared-handle LRU across threads.
- No CFITSIO `fits_iterate_data`, `fits_calculator` / `fits_select_rows`,
  histogram binning, hierarchical grouping, table compress, WCS rebin, pixel
  filter, IRAF delete, template exec, or CFITSIO HTTPS/stream drivers (own
  HTTP cache after the `sh://` security fix).
- Spectroscopy / continuum analysis stays out of torchfits (sibling stack).
