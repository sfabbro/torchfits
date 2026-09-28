"""Unit checks for modular bench suites, operation filters, and RSS timing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_contract import write_summary
from benchmarks.bench_timing import (
    _RssPeakSampler,
    time_median,
    time_medians_interleaved,
)
from benchmarks.suites import DEFICIT_FOCUS_SUITES, list_suite_names, resolve_suite


def test_suite_registry_resolves_aliases() -> None:
    s = resolve_suite("hcompress")
    assert s.name == "compressed_hcompress"
    assert s.scope == "fits"
    assert "hcompress" in s.case_filter or "compressed_hcompress" in s.case_filter
    assert s.mmap == "matrix"
    assert resolve_suite("cutouts").mmap == "on"
    assert "release" in list_suite_names()
    assert "compressed_hcompress" in DEFICIT_FOCUS_SUITES


def test_fitstable_predicate_suite_has_operation_filter() -> None:
    s = resolve_suite("fitstable_predicate")
    assert s.scope == "fitstable"
    assert "predicate" in s.operation
    assert s.no_gpu is True


def test_gpu_transports_suite_is_gpu_only() -> None:
    s = resolve_suite("gpu_transports")
    assert s.gpu_only is True


@pytest.mark.performance
def test_time_median_reports_peak_rss() -> None:
    payload = bytearray(2 * 1024 * 1024)

    def _alloc() -> int:
        # Touch the buffer so RSS samples see real residency.
        payload[0] = 1
        payload[-1] = 2
        return len(payload)

    median, peak_rss, _peak_cuda, err = time_median(_alloc, runs=3, warmup=1)
    assert err is None
    assert median is not None and median >= 0.0
    # psutil may be absent in minimal envs; when present RSS must be finite.
    if peak_rss is not None:
        assert peak_rss > 0.0


@pytest.mark.performance
def test_rss_sampler_sees_transient_peak(monkeypatch) -> None:
    import threading

    import benchmarks.bench_timing as timing

    held: list[bytearray] = []
    sampled_while_held = threading.Event()
    real_rss = timing._rss_mb

    def _rss_while_held() -> float | None:
        sample = real_rss()
        if held and sample is not None:
            sampled_while_held.set()
        return sample

    monkeypatch.setattr(timing, "_rss_mb", _rss_while_held)

    def _spike() -> None:
        # Allocate then free so start/end RSS understates peak.
        blob = bytearray(8 * 1024 * 1024)
        blob[0] = 1
        held.append(blob)
        # Sampler thread polls; wait until it observes RSS while the blob is live.
        if timing._PROC is not None:
            assert sampled_while_held.wait(timeout=2.0)
        held.clear()

    with _RssPeakSampler(interval_s=0.001) as sampler:
        _spike()
    # Without a live process RSS hook this may be None; otherwise peak must rise.
    if sampler.peak_mb is not None:
        assert sampler.peak_mb > 0.0


def test_interleaved_warmup_failure_soft_skips() -> None:
    def ok() -> int:
        return 1

    def boom() -> int:
        raise RuntimeError("peer_warmup_fail")

    out = time_medians_interleaved(
        {"ok": ok, "boom": boom},
        runs=2,
        warmup=1,
    )
    assert out["ok"][0] is not None and out["ok"][3] is None
    assert out["boom"][0] is None and out["boom"][3] is not None


def test_scorecard_counts_table_within_floor() -> None:
    rows = [
        {
            "domain": "fitstable",
            "case_id": "narrow::predicate_filter",
            "family": "smart",
            "library": "torchfits",
            "method": "torchfits",
            "comparable": True,
            "status": "OK",
            "time_s": 1.04,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
        {
            "domain": "fitstable",
            "case_id": "narrow::predicate_filter",
            "family": "smart",
            "library": "fitsio",
            "method": "fitsio_torch",
            "comparable": True,
            "status": "OK",
            "time_s": 1.0,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.md"
        write_summary(path, run_id="t", scopes=["fitstable"], rows=rows, deficits=[])
        text = path.read_text(encoding="utf-8")
        assert "1/1" in text


def test_bench_all_exits_nonzero_on_domain_failure(monkeypatch) -> None:
    import benchmarks.bench_all as bench_all

    monkeypatch.setattr(
        bench_all,
        "_parse_args",
        lambda: __import__("argparse").Namespace(
            scope="fits",
            fits_only=False,
            fitstable_only=False,
            suite="",
            output_dir=Path(tempfile.mkdtemp()),
            run_id="fail_test",
            profile="user",
            mmap=False,
            no_mmap=True,
            mmap_matrix=False,
            filter="",
            operation="",
            quick=True,
            keep_temp=False,
            no_gpu=True,
            gpu_only=False,
        ),
    )

    def _boom(**kwargs):
        raise RuntimeError("forced_domain_failure")

    monkeypatch.setattr(bench_all, "run_fits_domain", _boom)
    monkeypatch.setattr(bench_all, "_clear_bench_caches", lambda: None)
    assert bench_all.main() != 0


def test_scorecard_ignores_singleton_torchfits_group() -> None:
    rows = [
        {
            "domain": "fits",
            "case_id": "solo::read_full",
            "family": "smart",
            "library": "torchfits",
            "method": "torchfits",
            "comparable": True,
            "status": "OK",
            "time_s": 0.1,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
        {
            "domain": "fits",
            "case_id": "solo::read_full",
            "family": "smart",
            "library": "torchfits",
            "method": "torchfits_specialized",
            "comparable": True,
            "status": "OK",
            "time_s": 0.2,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.md"
        write_summary(path, run_id="t", scopes=["fits"], rows=rows, deficits=[])
        text = path.read_text(encoding="utf-8")
        assert "0/0" in text or "n/a" in text.lower() or "Scorecard" in text
        assert "1/1" not in text


def test_fitstable_filter_has_dense_and_selective_regimes() -> None:
    """Both keep-rate regimes are first-class ops (not one tuned threshold)."""
    import inspect

    from benchmarks import bench_fitstable_io as m

    schema = [("id", "i4"), ("flux", "f4"), ("err", "f4"), ("flag", "bool")]
    assert m._choose_filter_col(["id", "flux", "err", "flag"], schema) == "flux"
    assert m._dense_predicate("id") == "id > 0"
    assert m._selective_predicate("flux", schema) == "flux > 1.5"
    assert m._selective_predicate("id", schema) == "id > 900000"
    src = inspect.getsource(m._bench_case)
    assert '"predicate_filter"' in src
    assert '"predicate_filter_selective"' in src


def test_fitstable_scan_count_uses_nrows_not_column() -> None:
    """Specialized scan_count must match peer O(1) NAXIS2 / get_nrows contract."""
    import inspect

    from benchmarks import bench_fitstable_io as m

    src_smart = inspect.getsource(m._torchfits_scan_count)
    src_local = inspect.getsource(m._torchfits_scan_count_local)
    assert "read_nrows" in src_smart
    assert "read_nrows" in src_local
    assert "read_header" not in src_smart
    assert "read_header" not in src_local
    assert "read_table" not in src_local
    assert "get_header" not in src_smart
    assert "get_header" not in src_local


def test_summary_does_not_present_process_rss_as_a_library_comparison() -> None:
    """`write_summary` must not print two RSS columns it cannot make differ.

    `_RssPeakSampler` samples `psutil.Process().memory_info().rss` — the whole
    interpreter process, not the library under test. Every method in a ranking
    group runs in that same process, so the two probes measure the same thing.
    Measured over the published CUDA exhaustive run: `peak_rss_mb` is identical
    for every method in **1068 of 1151** comparison groups, and in the two runs
    the benchmarks page cites, **37 of 37** deficit rows have byte-identical
    `torchfits_peak_rss_mb` and `best_peak_rss_mb` (293.8203125 vs
    293.8203125, 766.80859375 vs 766.80859375, …).

    `write_summary` nevertheless emitted them as two adjacent columns, "TF RSS
    (MB)" and "Winner RSS (MB)", in every published `summary.md`. A reader
    scanning that pair concludes the two libraries use the same memory, when
    the truth is that this metric cannot tell them apart at all. The header now
    carries one process-level column and says so.

    (The CUDA counter is *not* in this category: `_cuda_reset()` runs per
    method, so `peak_cuda_alloc_mb` is a real per-call high-water mark. It
    legitimately reports equal values when both libraries allocate an
    identically shaped output, and it stays per-method.)
    """
    rows = [
        {
            "domain": "fits",
            "case_id": "c::read_full",
            "family": "smart",
            "library": lib,
            "method": method,
            "comparable": True,
            "status": "OK",
            "time_s": t,
            "mmap_target": "off",
            "n_points": 1000,
            "peak_rss_mb": 293.8,  # same process, same sampler -> same number
            "metadata": {},
        }
        for lib, method, t in (
            ("torchfits", "torchfits", 0.2),
            ("fitsio", "fitsio_torch", 0.1),
        )
    ]
    # A real deficit row, so the "TorchFits Deficits (Not First)" table is
    # actually rendered -- passing `deficits=[]` would leave that table out of
    # the document and make both assertions below pass for the wrong reason.
    deficits = [
        {
            "domain": "fits",
            "family": "smart",
            "case_id": "c::read_full",
            "case_label": "c [read_full]",
            "operation": "read_full",
            "mmap_target": "off",
            "host": "h",
            "torchfits_method": "torchfits",
            "torchfits_time_s": 0.2,
            "torchfits_peak_rss_mb": 293.8,
            "best_library": "fitsio",
            "best_method": "fitsio_torch",
            "best_time_s": 0.1,
            "best_peak_rss_mb": 293.8,
            "lag_ratio": 2.0,
            "pct_behind": 100.0,
            "significance": "significant",
            "n_points": 1000,
            "perceived_impact": "visible",
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.md"
        write_summary(path, run_id="t", scopes=["fits"], rows=rows, deficits=deficits)
        text = path.read_text(encoding="utf-8")

    assert "TorchFits Deficits (Not First)" in text, (
        "the deficit table was not rendered, so this test would pass vacuously"
    )
    assert "Winner RSS" not in text, (
        "write_summary emitted a 'Winner RSS' column, but the RSS sampler "
        "measures the whole process and cannot distinguish the two libraries:\n"
        + "\n".join(line for line in text.splitlines() if "RSS" in line)
    )
    assert "Process peak RSS (MB)" in text, (
        "the deficit table must label its RSS column as process-level so the "
        "number is not read as a per-library figure"
    )
    assert "not** a per-library memory" in text, (
        "the Notes section must say the RSS column is not a per-library figure"
    )


def test_import_boundary_gate_compares_the_median_not_the_luckiest_spawn() -> None:
    """`--strict` must gate on what users pay, not on the best of N cold starts.

    `bench_import_boundary.check` compares `min_ms` against each entry point's
    budget. The budgets are not arbitrary: the comment above
    `_ARROW_BUDGET_MS` says "Measured 461-515 ms minimum across runs (5
    repeats, macOS arm64), with medians up to 541 ms. Budget is set from the
    slow end of that spread, not the fastest sample: a gate that only passes on
    a good day is not a gate" -- and 600 ms is plainly the median range rounded
    up, not the 515 ms minimum rounded up. Comparing `min` against a
    slow-end budget then throws away roughly half the margin the calibration
    was chosen to provide.

    Measured on this machine (macOS arm64, 5 repeats, 2026-09-27):

    | entry point | min | median | budget | headroom on min | on median |
    |---|---:|---:|---:|---:|---:|
    | `table.read (Arrow)` | 491.6 | 545.9 | 600 | 108 ms (18%) | 54 ms (9%) |
    | `read_num_hdus` | 55.0 | 60.2 | 120 | 65 ms (54%) | 60 ms (50%) |
    | `read_header` | 55.1 | 58.2 | 120 | 65 ms (54%) | 62 ms (51%) |

    The gap is widest exactly where the budget is tightest, and the module
    docstring calls spawn-to-exit "the cost users pay" -- a median, not a
    minimum. `min_ms` is still reported so the spread stays visible.
    """
    from benchmarks.bench_import_boundary import check

    slow_but_within_calibration = {
        "name": "table.read (Arrow)",
        "min_ms": 491.6,  # the lucky spawn
        "median_ms": 660.0,  # what every user actually pays
        "torch_loaded": False,
        "expects_torch": False,
        "budget_ms": 600.0,
    }
    failures = check([slow_but_within_calibration])
    assert failures, (
        "a run whose median cold start (660 ms) is over the 600 ms budget "
        "passed the gate because its fastest of five spawns was 491 ms. The "
        "budget was calibrated from the slow end of the spread, so the "
        "comparison must be against the median."
    )


def test_cfitsio_direct_validation_survives_python_dash_O() -> None:
    """Benchmark result validation must not be an `assert`.

    `benchmarks/run_cfitsio_direct_bench.py` ends by checking that the C
    benchmark produced a usable comparison -- at least 50 OK rows, at least 5
    distinct operations, zero ERROR rows -- and it did that with three bare
    `assert` statements. CPython removes assert statements from the bytecode
    under `-O` / `PYTHONOPTIMIZE=1`; compiling that function at `optimize=2`
    reduces it to a single `RETURN_CONST`. So the one check standing between a
    broken or near-empty `cfitsio_direct.csv` and a published cross-library
    comparison table disappeared under a flag nobody sets on purpose, and
    `main()` returned 0.

    Verified live: the three statements compiled to 3 x
    `COMPARE_OP` / `POP_JUMP_IF_TRUE` / `RAISE_VARARGS` at optimize=0 and to
    nothing at optimize=2. They now raise `SystemExit`, which `-O` cannot
    remove.
    """
    import ast
    import textwrap

    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "run_cfitsio_direct_bench.py"
    ).read_text(encoding="utf-8")
    main_fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    asserts = [n for n in ast.walk(main_fn) if isinstance(n, ast.Assert)]
    assert not asserts, (
        "main() validates its benchmark results with assert, which `python -O` "
        f"strips from the bytecode: {[n.lineno for n in asserts]}. Use an "
        "explicit raise so the check survives."
    )

    # And the replacement is actually reachable at optimize=2.
    probe = textwrap.dedent(
        """
        def validate(stats):
            if stats["error"] != 0:
                raise SystemExit("boom")
            return 0
        """
    )
    for optimize in (0, 2):
        code = compile(probe, "probe.py", "exec", optimize=optimize)
        fn = next(c for c in code.co_consts if getattr(c, "co_name", "") == "validate")
        assert any(
            "RAISE" in i.opname or "CALL" in i.opname
            for i in __import__("dis").get_instructions(fn, show_caches=False)
        ), f"the raise disappeared at optimize={optimize}"


def test_every_runnable_benchmark_script_has_an_entry_point() -> None:
    """A benchmark nobody can launch is not a benchmark; it is a stale file.

    Two scripts had no caller anywhere in the tree -- not a pixi task, not a
    `scripts/` wrapper, not a workflow, not a test, and not even an import from
    another benchmark. `bench_median_stack.py` (641 lines) and
    `bench_metadata.py` (232 lines) both have a real `argparse` entry point and
    a careful module docstring, so nothing about reading them says "dead": the
    only evidence they were unreachable was grepping for their names, which
    found them in dated `.cursor/harness/**/tracked-files.txt` review manifests
    and nowhere else. `bench_metadata.py` is worse than idle -- it is the
    measurement behind `bench_import_boundary._METADATA_BUDGET_MS`, so that
    budget's comment points at numbers only this script produces, while the
    script itself had no way to be run.

    Both now have pixi tasks. This asserts the general rule so the next
    unreferenced benchmark fails here rather than in a code review.

    Two directory classes are excluded from the search, and both exclusions
    were added because the first version of this test passed for the wrong
    reason:

    * `.cursor/harness/**` -- dated review manifests. A name in
      `tracked-files.txt` is a *scope* record ("this existed on 2026-08-26"),
      not a caller.
    * `tests/` -- running `pytest` is not a way to launch a benchmark, and
      without this exclusion the guard matched its own docstring, which names
      both orphans. That is the same substring-gate mistake recorded as CP-012,
      repeated one unit later; the break-it is what caught it.

    Launching by path string counts, which is why this greps the whole tree
    rather than only pixi tasks: `bench_all.py` runs
    `bench_fitstable_io.py` as a subprocess by filename.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "benchmarks"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        pytest.skip(f"not a git checkout: {listed.stderr.strip()}")

    entry_points = [
        rel
        for rel in listed.stdout.split()
        if rel.endswith(".py")
        and 'if __name__ == "__main__":' in (root / rel).read_text(encoding="utf-8")
    ]
    assert entry_points, "no runnable benchmark scripts found; check is vacuous"

    orphans: list[str] = []
    for rel in sorted(entry_points):
        stem = Path(rel).stem
        # -l lists files only; the script's own name appears in its own source.
        found = subprocess.run(
            [
                "git",
                "grep",
                "-l",
                "--",
                stem,
                ":(exclude).cursor/harness",
                ":(exclude)tests",
                f":(exclude){rel}",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )
        if not found.stdout.strip():
            orphans.append(rel)

    assert not orphans, (
        'these benchmark scripts have an `if __name__ == "__main__"` entry '
        "point but nothing in the tree launches, imports or tests them "
        f"(searched all tracked files except .cursor/harness review manifests "
        f"and tests/, neither of which is a launch path): {orphans}"
    )
