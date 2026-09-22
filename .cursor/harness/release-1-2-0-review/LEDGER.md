# LEDGER — torchfits 1.2.0 directory-by-directory review

Baseline HEAD: `1bb6958d958c5ebdb9a201f95bc5a01e89fe7acb` (2026-02-…, post-R0 upstream sync: already up to date).

Completeness proof: `tracked-files.txt` (468 lines, `git ls-files` at R0). Every tracked file gets one row below (File | Round | Depth | Findings | Status).

| File | Round | Depth | Findings | Status |
|---|---|---|---|---|
| `src/torchfits/__init__.py` | R1-A | full | — | clean |
| `src/torchfits/io.py` | R1-B | full | r1b-05, r1b-06, r1b-09 | fixed (05,06); r1b-09 deferred |
| `src/torchfits/table.py` | R1-B | full | — | clean |
| `src/torchfits/hdu.py` | R1-B | full | — | clean |
| `src/torchfits/cache.py` | R1-A | full | r1a-01..r1a-04 | fixed |
| `src/torchfits/cpp.py` | R1-A | full | — | clean |
| `src/torchfits/_cpp.py` | R1-A | full | — | clean |
| `src/torchfits/interop.py` | R1-B | full | r1b-01, r1b-02, r1b-03, r1b-05 | fixed |
| `src/torchfits/where.py` | R1-C | full | r1c-03, r1c-04 | fixed |
| `src/torchfits/_where.py` | R1-C | full | r1c-01, r1c-02 | fixed |
| `src/torchfits/fits_schema.py` | R1-B | full | r1b-04, r1b-07 | fixed |
| `src/torchfits/header_parser.py` | R1-C | full | r1c-10 | fixed |
| `src/torchfits/http_util.py` | R1-C | full | r1c-06, r1c-08, r1c-13, r1c-14 | fixed (doc note r1c-14 → R15) |
| `src/torchfits/logging.py` | R1-A | full | — | clean |
| `src/torchfits/vos_uri.py` | R1-C | full | — | clean |
| `src/torchfits/_string_decode.py` | R1-C | full | r1c-05, r1c-13 | fixed (errors='ignore' parity → deferred) |
| `src/torchfits/_tensor_buffer.py` | R1-C | full | — | clean |
| `src/torchfits/_C.pyi` | R1-A | full (check-stub) | — | clean (zero stub drift) |
| `src/torchfits/_io_engine/paths.py` | R1-C family | full | r1c-12, r1c-14 | docstring fixed; r1c-12 → R10 |
| `src/torchfits/cli/common.py` | R1-C family | full | — | clean |
| `src/torchfits/cli/cmds_copy.py` | R1-C family | full | r1c-07 | fixed |
| `src/torchfits/data/remote.py` | R1-C family | full | r1c-09 | fixed |
| `src/torchfits/_io_engine/http_subset.py` | R1-C family | full | r1c-08 | fixed (typed 416 → subset fallback) |
