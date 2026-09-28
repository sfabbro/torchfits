# `tests/cpp` — standalone C++ self-checks

Five small, dependency-free programs that exercise C++ behaviour the Python
suite cannot reach directly: SIMD byte-swaps, the `parallel_for` pool, CFITSIO's
open-mode conflict contract, and `FitsReader`'s thread safety.

Each is a plain `main()` that **counts** its failures and returns non-zero,
deliberately framework-free so it can be compiled and run against a header or a
library in isolation. They do not use `assert`: under `-DNDEBUG` every `assert`
compiles to nothing, so an assert-based self-check prints "all checks passed"
and exits 0 having verified nothing at all (measured — see the audit ledger).

## Run them

```bash
tests/cpp/run_all.sh
```

The runner locates the built artifacts, writes the FITS fixture the two
file-based checks need, builds all five, runs each **under a timeout**, and
exits non-zero if any fails. It prints `PASS` / `FAIL` / `BUILD FAILED` per
check, and `TIMEOUT` / `FAIL (exit 124)` for a check that never returns.

It also runs a sixth check: the **x86_64 build of `test_bswap_helpers`,
executed under Rosetta**, because the byte-swap helpers have three `#if`
branches and a native build compiles only one. On Apple Silicon that is NEON,
so the SSSE3 branch — which `CMakeLists.txt` compiles into every x86_64 build
via `-mssse3` — would otherwise never be executed here. Set
`TORCHFITS_CPP_X86=0` to skip it; without an x86 runtime it is skipped with a
printed note. The AVX2 branch is not attempted (no shipped build enables it,
and Rosetta raises `SIGILL` on it here): `tests/test_simd_shuffle_masks.py`
checks both x86 mask tables statically, on any platform.

`CXX` selects the compiler (default `clang++`) and may include flags, the way
the two pytest drivers read it (`CXX="c++ -pipe"` works).
`TORCHFITS_CPP_CHECK_TIMEOUT` sets the per-check budget in seconds (default
180). The timeout is load-bearing, not defensive: the regression
`test_parallel_for_nesting` guards *is* a deadlock, so a regression makes that
check hang rather than fail.

## Three of the five also run under pytest

- `test_bracket_detection.cpp` — `tests/test_security.py::test_native_cfitsio_bracket_detector_probe_runs`
  compiles and runs it, and asserts on its "all checks passed" line.
- `test_bswap_helpers.cpp` — `tests/test_byteswap.py::test_native_bswap_helpers_match_a_byte_wise_reference`
  compiles and runs it, and asserts on its exit status and summary line.

- `test_parallel_for_nesting.cpp` — `tests/test_cpp_self_checks.py` compiles
  it (with `core/parallel.cpp`) and runs it at pool sizes 1, 2, 4 and 8, so the
  deadlock regression in `parallel_for` is covered by `pytest tests/ -q` and not
  only by this runner. The same file also guards the runner itself: a check
  cannot be dropped from the case table without failing a test, and the two
  checks below must stay described as manual.

All three read `CXX` with `shlex.split`, so a value carrying flags (`ccache c++`,
`c++ -pipe`) works, and a `CXX` that cannot be run at all **skips with a
reason** instead of erroring. `tests/test_simd_shuffle_masks.py` covers the other side of
the same header: the x86 mask tables, checked statically against
`_mm_shuffle_epi8`'s semantics, so they are verified on any platform. The other three need the torch-free core library and
CFITSIO's archive, which no pytest fixture builds, so they stay behind the
runner. None of the five is in CMake or a workflow.

## Why a runner script rather than a command in each file

Two of the five link the torch-free core library and CFITSIO. Neither is
installed into the pixi environment — they live in the per-build scratch tree
(`.pixi/bld/torchfits/<hash>/bld`), whose hash changes on every rebuild. A
command hardcoding that path rots immediately, which is why the two library
checks used to carry a literal `<build>/libtorchfits_core.dylib` placeholder
that had to be hand-substituted before it could be used at all.

The three header-only checks document a complete, copy-pasteable command
directly in their own file header, and those are verified to work verbatim.

## What each one covers

| File | Covers | Needs a file argument |
|---|---|---|
| `test_bracket_detection.cpp` | `has_cfitsio_extended_filename_syntax`, including the `'['`-in-a-directory false positives | no |
| `test_bswap_helpers.cpp` | the big-endian→host byte-swaps in `internal_utils.h`: 19 sizes × 5 helpers × 4 source/destination alignments against a reference that moves bytes one at a time, plus a host-order pin tying the permutation to the FITS convention. Prints which SIMD branch compiled in, because a silent no-op branch would still "pass". | no |
| `test_parallel_for_nesting.cpp` | `core::parallel_for` nesting (a deadlock regression), that the outer chunks tile the range exactly once, and error propagation from both top-level and nested bodies | no |
| `test_open_for_write_conflict.cpp` | the CFITSIO contract `open_fits_for_write` relies on: a held READONLY handle makes a READWRITE open return `FILE_NOT_OPENED` (104) | yes |
| `test_fitsreader_threads.cpp` | one `FitsReader` shared by 4 threads — CFITSIO keeps one mutable current-HDU cursor per handle, so unlocked queries could interleave into another HDU's answer. The full shape vector is compared, not just its first axis. | yes (≥ 2 usable HDUs) |

The last two are regressions for real defects that were measured, not
hypothesised; each file's header records the measurement.

## Why they stay manual

They cover C++ internals that the Python suite structurally cannot reach — the
`bswap` helpers' unsigned fast paths are gated on `!compressed` while the real
CFHT corpus is Rice-compressed throughout, so no end-to-end read exercises
them. Folding the three library-linked checks into the Python suite would mean
building a C++ test binary from a pytest fixture, which is a larger change than
the value warrants; the runner keeps them one command away instead.
