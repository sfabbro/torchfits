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
