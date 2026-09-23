# R7 slice A (r7a) — `src/torchfits/cli/` core: dispatch, shared helpers, header/info/probe

Round R7, slice A. Owner files (exclusive src): `main.py, common.py, __init__.py, __main__.py, cmds_header.py, cmds_info.py, cmds_probe.py`.
Cross-file authority: `docs/cli.md` (sole editor for the round), `tests/test_cli_exit_matrix.py` (new), `tests/test_cli.py` (sole owner this round).
Exact decisions landed: (1) unexpected non-`CliError` → traceback to stderr + exit 5 + docs exit-table row; (2) `KeyboardInterrupt` → 130 never 2 (pin + harden); (6) full exit-matrix parametrized test. Also landed B's verbatim decision-3 docs text (and B's `table`/`cutout` doc fixes).

Rows: `| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |`

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r7a-01 | MAJOR | 5 error contract | `cli/main.py:main` | Unexpected non-`CliError` exceptions escaped `main()` uncaught: the process printed a traceback and exited **1**, colliding with `diff`'s documented code 1 ("difference found") and violating the decision-1 contract (exit 5). `build_parser()` also sat outside the `try`, so dispatch-time errors bypassed every handler. Repro (pre-fix, `tests/test_cli_exit_matrix.py::test_matrix_internal_error_exit_5` x15): `src/torchfits/cli/main.py:57: in main / return int(args.func(args)) … E RuntimeError: boom` — 15 failed (`RuntimeError: boom` escaped). | Moved `build_parser()` into the `try`; added `except Exception: traceback.print_exc(); return EXIT_INTERNAL` with `EXIT_INTERNAL = 5` (`common.py`). Same commit adds `docs/cli.md` row `&#124; 5 &#124; Internal error &#124; Unexpected exception; traceback printed to stderr &#124;` (5 = lowest unused code). Test: `test_matrix_internal_error_exit_5[<all 15 commands>]` asserts rc==5 + `Traceback (most recent call last)` + `RuntimeError: boom` on stderr. Before: `15 failed … RuntimeError: boom`. After: `15 passed`. | fixed |
| r7a-02 | MINOR | 5 error contract (pin) | `cli/main.py:main` | Decision 2 "KeyboardInterrupt exits 130, never 2" — re-derived at HEAD on CPython 3.13.15: every reachable interrupt path already gave 130 (`run`-path Ki → `130`; real SIGINT via `signal.SIGINT` → process `130`; uncaught top-level Ki → `130`); no path yielded 2 (argparse only exits 2 for parse errors, which no interrupt path funnels through). Gap found: `build_parser()` before the `try` meant a pre-dispatch Ki escaped `main()` (interpreter path instead of the 130 clause). | Pinned + hardened: `build_parser()` moved into the `try` (r7a-01) so the `except KeyboardInterrupt → EXIT_INTERRUPT` clause covers dispatch too. Pins: `test_matrix_interrupt_exit_130[<all 15 commands>]` (asserts `rc == 130`, `rc != 2`) + `test_real_sigint_exit_130` (real SIGINT through the real handler in a `python -m`-equivalent process → 130). Before/after: `15 passed` (behavior already correct; now pinned). | fixed |
| r7a-03 | MAJOR | 1/7 non-finite floats in JSON | `cli/common.py:_json_safe` | `_json_safe` only recognized Python `float`; numpy-typed non-finite values and arrays fell through to `json.dumps(..., allow_nan=False)` and **crashed** instead of emitting `null` (decision 3 contract: non-finite → `null`, "JSON has no NaN"). Repro: `emit_records([{"v": np.float32("nan")}], format="json")` → `ValueError: Out of range float values are not JSON compliant: nan` (same for `np.float32("inf")`, `np.array([1.0, nan])`; `py float nan`/`np.float64 nan` worked). Test (`tests/test_cli.py::test_emit_records_nulls_nonfinite_values`) covers json+jsonl over py/np64/np32 nan/inf, np ints, arrays and nested containers. Before: `FAILED … ValueError: Out of range float values are not JSON compliant: nan`. After: passed — all non-finite → `null`, `np.int64(7) == 7`, nested `[None, {"deep": None}]`. Shared-machinery fix consumed by `stats`/`table` (slice B confirmed via hub they route through `emit_records` and did not edit `common.py`). | fixed |
| r7a-04 | MINOR | 7 edge inputs | `cli/cmds_probe.py:run` | `--header-bytes` silently clamped to `max(2880, x)` (explicit user input coerced without a word) and `--timeout` accepted `0`, negative, `nan`, `inf` (urllib then failed with confusing wrapped errors). Repro (pre-fix): `torchfits.cli.main.main(["probe", "http://example.com/x.fits", "--header-bytes", "100"])` → rc 0 with the fetch attempted at 2880 bytes. | Usage validation: `--header-bytes` must be `>= 2880` (one FITS block); `--timeout` must be a positive finite number of seconds → `UsageError` (exit 2) before any I/O. Tests: `test_probe_rejects_header_bytes_below_one_block` (`0/100/-1` → 2, no fetch), `test_probe_rejects_nonpositive_timeout` (`0/-2.5/nan/inf` → 2, no fetch). Before: `assert 0 == 2` x2 tests. After: passed. | fixed |
| r7a-05 | MINOR | 5 error contract / side effects | `cli/cmds_probe.py:run` | Mixed local+remote argv ("mixing local paths and remote URLs is not supported") was rejected only **after** the remote fetches had already run — side-effecting network I/O before a usage error. Repro (pre-fix): `main(["probe", <local>, "http://example.com/x.fits"])` with `_probe_http` recorded: fetch list `['http://example.com/x.fits']`, rc 2. | Paths are classified up front; the mixing `UsageError` fires before any probe. Test: `test_probe_mixed_local_remote_rejected_before_fetch` asserts rc==2 AND the fetch recorder stays empty. Before: `assert ['http://example.com/x.fits'] == []`. After: passed. | fixed |
| r7a-06 | MINOR | 8 docs/API faithfulness | `docs/cli.md` Global Options & Flags table | The table claimed all 7 shared flags exist on "the emit-style subcommands (`info`, `header`, `verify`, `stats`, `table`, `probe`)". Reality (from `build_parser()` action dump): `-o/--out`, `--out-dir`, `-j/--jobs`, `-J/--file-jobs` exist on **none** of `info`/`header`/`table`/`probe` and only `-J` on `verify`/`stats`-family subsets; `-o`'s default ("Positional argument or stdout") and `--out-dir`'s ("Current directory") were both wrong (there is no positional `output` alias; `--out-dir` is required for multi-input batches, not defaulted). | Rewrote the intro ("Shared flags (availability varies by subcommand)") and added an `Available on` column with the exact per-flag command sets from the parser dump; corrected both defaults. Verified against `build_parser()` output pasted in this round's evidence. | fixed |
| r7a-07 | CLEANUP | dead code | `cli/common.py:add_out_arg`, `resolve_out_path` | Both helpers had **zero** references across `src/torchfits` and `tests/` (grep evidence: only their own `def` lines); all commands use `resolve_batch_io_pairs`/2-positional paths. `add_out_arg`'s "positional alias of -o/--out" help text also contradicted every real parser. | Zero-behavior-change deletion (~24 lines). Full targeted suite green after deletion (`tests/test_cli_exit_matrix.py` + `tests/test_cli.py`). | fixed |
| r7a-08 | MINOR | 7 truncated files / 5 error contract | engine HDU scan (`cpp_src/fits_file.cpp`/`fits_bindings.cpp` — outside r7a file set); surface `cli/cmds_info.py`, `cli/common.py:iter_file_hdu_pairs` | Truncated/hostile-header files are inventoried silently: `torchfits info truncated.fits -f json` exits **0** listing fewer HDUs with no diagnostic. Repro: 2-HDU image MEF cut at 5760 bytes → `rc=0, records=1`; table with a garbage `NAXIS2`/`TFIELDS` card value → `len(torchfits.open(p)) == 1` (table HDU vanishes), `info` rc=0. astropy agrees on HDU counts for the hostile-card files but **warns** on truncation ("File may have been truncated: actual file length (2880) is smaller than the expected size (5760)") where torchfits is silent. Not the `_info_record` `int(nrows)` path (unreachable: such HDUs are dropped by the engine before records are built). | Not fixable in r7a's file set (locus = engine HDU/extent scan; CLI sees only the reduced HDU count). Suggested fix at the engine round (R8/R9): raise/`IoError` or warn on unparsed trailing header bytes / extent-short files at open. Note: pixel-reading commands (`stats` etc.) DO error via `ensure_extent_within_file` → exit 3; only header-only inventory accepts truncation silently. | deferred |
| r7a-09 | MINOR | 5 error contract | `cli/cmds_probe.py:_probe_vos` | `finally: … except Exception: pass` around `handle.close()` swallows a remote-handle close failure (class-5 "no `except Exception: pass` on IO"). | Deferred: re-raising from `finally` would mask the in-flight primary exception (`IoError` for the fetch); the narrow fix (re-raise only when no exception is active) is not worth the complexity for a close-failure diagnostic. | deferred |
| r7a-10 | MINOR | 8 docs/API faithfulness | `docs/cli.md` (stats/table/cutout sections, exit table) | Cross-slice decision/doc text that slice A must land: decision-1 exit-table row, decision-3 non-finite note (stats section), B's `table` docs example `torchfits table catalog.fits -e 1 -c RA,DEC,FLUX -n 10` documented a nonexistent `-c/--columns` flag (real invocation exits 2 "unrecognized arguments"), B's `cutout --box` validation sentence. Also: `--stdin` docs row omitted that `resolve_paths` reads stdin implicitly when no paths are given and stdin is not a terminal. | Landed verbatim in `docs/cli.md` in this round: exit row `&#124; 5 &#124; Internal error &#124; Unexpected exception; traceback printed to stderr &#124;` (with r7a-01); B's stats note "Non-finite values in JSON output: … reports `null` for `min`, `max`, `mean`, `std`, and `median`."; deleted the two `-c` example lines; added B's `--box` sentence after the two coordinate-format bullets; `--stdin` row now states the implicit-stdin behavior (behavior unchanged, `resolve_paths` keeps the fallback). | fixed |

## Evidence (before → after)

Failing-first runs were executed against the unfixed tree (HEAD + sibling slice work in flight), then re-run after the fix.

1. Decision 1 (exit 5), true failing-first (`pixi run pytest tests/test_cli_exit_matrix.py::test_matrix_internal_error_exit_5 tests/test_cli_exit_matrix.py::test_matrix_interrupt_exit_130 -q`):
```
src/torchfits/cli/main.py:57: in main
    return int(args.func(args))
    ...
    def _boom(_args):
>       raise RuntimeError("boom")
E       RuntimeError: boom
FAILED tests/test_cli_exit_matrix.py::test_matrix_internal_error_exit_5[info] - RuntimeError: boom
    … (all 15 commands)
15 failed, 15 passed in 2.85s
```
(The 15 passes = `test_matrix_interrupt_exit_130[...]` — decision 2 already correct at HEAD.) Note: the first combined run failed these rows on a test-harness bug of mine (3-arg `monkeypatch.setattr` with a string target → `AttributeError: … has no attribute 'run'`); the harness was fixed and the run above is the honest failing-first.

2. Combined failing-first (`pixi run pytest tests/test_cli_exit_matrix.py tests/test_cli.py -q`) — 34 failed (15+15 monkeypatch-harness rows + the 4 below), **120 passed** (all exit-matrix 0/2/3 rows, `diff`→1, `verify`→4, argparse→2, real-SIGINT→130 green at HEAD):
```
FAILED tests/test_cli.py::test_emit_records_nulls_nonfinite_values - ValueError: Out of range float values are not JSON compliant: nan
FAILED tests/test_cli.py::test_probe_rejects_header_bytes_below_one_block - AssertionError: 0
assert 0 == 2
FAILED tests/test_cli.py::test_probe_rejects_nonpositive_timeout - AssertionError: 0
assert 0 == 2
FAILED tests/test_cli.py::test_probe_mixed_local_remote_rejected_before_fetch - AssertionError: assert ['http://example.com/x.fits'] == []
34 failed, 120 passed, 1 warning in 191.55s (0:03:11)
```

3. Post-fix verification (`pixi run pytest tests/test_cli_exit_matrix.py tests/test_cli.py -q`): **153 passed, 1 failed** — the single failure was a bug in the new `test_emit_records_nulls_nonfinite_values` itself (it line-split `emit_records`' pretty-printed multi-line `json` output before `json.loads`, `JSONDecodeError: Expecting value: line 1 column 2`); parsing fixed to load the whole buffer for `json` and line-wise only for `jsonl`, then the test passes (re-run output below):

```
1 failed, 153 passed, 1 warning in 178.69s (0:02:58)
```

Re-run after the test-parse fix (`pixi run pytest tests/test_cli.py::test_emit_records_nulls_nonfinite_values -q`):

```
1 passed in 1.17s
```

Final combined run (`pixi run pytest tests/test_cli_exit_matrix.py tests/test_cli.py -q`) — **all green** (154 tests: 68 exit-matrix rows + 86 pre-existing/added `test_cli.py` tests):

```
154 passed, 1 warning in 170.83s (0:02:50)
```

4. Process-level proof of the exit-5 contract (complementing the in-process matrix rows; `python -c` wrapper raising `RuntimeError` in `run`, then `sys.exit(main(...))` like `__main__`/console script):
```
process exit: 5
traceback on stderr: True
```

5. `_json_safe` probe (pre-fix):
```
py float nan: OK -> [{'v': None}]
np.float64 nan: OK -> [{'v': None}]
np.float32 nan: FAIL ValueError: Out of range float values are not JSON compliant: nan
np array w/ nan: FAIL ValueError: Out of range float values are not JSON compliant: nan
nested nan list: OK -> [{'v': [None]}]
np.float32 inf: FAIL ValueError: Out of range float values are not JSON compliant: inf
```

6. Decision-2 re-derivation (pre-fix probes):
```
--- case: KeyboardInterrupt via monkeypatched run
interrupted
rc = 130
uncaught-KI exit: 130
real-sigint exit: 130
```

## Verify-only invariants (slice A scope)

Perf: no perf changes landed in slice A — dispatch/emit paths are per-process startup-bound with no measured ≥5% defect and no redundant I/O found (single `torchfits.open` per file in `iter_file_hdu_pairs`; one `emit_records` pass per run), so the evidence rule had nothing to certify.

- `cli-j-vs-J`: `common.run_file_jobs` workers cap ATen at 1 (`_worker` calls `torch.set_num_threads(1)` before `fn`) — verified, unchanged.
- `is_remote_path` includes `ftp` (`_REMOTE_PREFIXES` contains `ftp://`) and local-file commands reject ftp remote paths — verified, unchanged (existing `test_cli_rejects_ftp_remote_paths`, `test_cli_arith…` green).
- `iter_file_hdu_pairs` rejects remote paths (`IoError`) and wraps per-path engine failures as typed `IoError` with `from exc` (no `str(e)` matching anywhere in slice A).

## Per-file disposition

| file | depth | finding IDs | status |
|---|---|---|---|
| `src/torchfits/cli/main.py` | full, deep | r7a-01, r7a-02 | fixed |
| `src/torchfits/cli/common.py` | full, deep | r7a-03, r7a-07 (+ `EXIT_INTERNAL` for r7a-01) | fixed |
| `src/torchfits/cli/__init__.py` | full | — (sealed `__all__ = ()`) | clean |
| `src/torchfits/cli/__main__.py` | full | — (`raise SystemExit(main())` correct) | clean |
| `src/torchfits/cli/cmds_header.py` | full, deep | — (card formatting/keyword-table paths conform; all exits 0/2/3) | clean |
| `src/torchfits/cli/cmds_info.py` | full, deep | r7a-08 (engine-locus surface only) | clean (r7a-08 deferred at engine) |
| `src/torchfits/cli/cmds_probe.py` | full, deep | r7a-04, r7a-05, r7a-09 | fixed (r7a-09 deferred) |
| `docs/cli.md` | full (sole editor) | r7a-06, r7a-10 (+ exit row with r7a-01) | fixed |
| `tests/test_cli_exit_matrix.py` | new (decision 6) | r7a-01, r7a-02 pins | fixed |
| `tests/test_cli.py` | extended (sole owner) | r7a-03, r7a-04, r7a-05 tests | fixed |
