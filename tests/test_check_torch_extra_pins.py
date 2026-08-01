"""Unit tests for scripts/check_torch_extra_pins.py (no network needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_torch_extra_pins as pins  # noqa: E402


def _pin(extra: str, spec: str) -> tuple[str, str, Version]:
    return (extra, f"torch=={spec}", Version(spec))


def test_wheel_lane_matches_constraints_file() -> None:
    lane = pins.load_wheel_lane()
    assert lane.contains("2.10.0")
    assert not lane.contains("2.11.0")
    assert not lane.contains("2.13.0")


def test_index_for_local_mapping() -> None:
    assert pins.index_for_local("cpu") == "https://download.pytorch.org/whl/cpu"
    assert pins.index_for_local("cu128") == "https://download.pytorch.org/whl/cu128"
    with pytest.raises(SystemExit):
        pins.index_for_local("rocm6")


def test_iter_torch_pins_finds_exact_local_pins() -> None:
    text = (
        "project = { optional-dependencies = { "
        'cpu = ["torch==2.10.0+cpu"], '
        'other = ["numpy>=1.20"] } }'
    )
    pins_found = list(pins.iter_torch_pins(text))
    assert len(pins_found) == 1
    extra, entry, version = pins_found[0]
    assert extra == "cpu"
    assert entry == "torch==2.10.0+cpu"
    assert version.public == "2.10.0"
    assert version.local == "cpu"


def test_iter_torch_pins_respects_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        "project = { optional-dependencies = { "
        "cuda = [\"torch==2.10.0+cu128; sys_platform == 'linux'\"] } }"
    )
    monkeypatch.setattr(sys, "platform", "linux")
    assert len(list(pins.iter_torch_pins(text))) == 1
    monkeypatch.setattr(sys, "platform", "darwin")
    assert list(pins.iter_torch_pins(text)) == []


def test_iter_torch_pins_rejects_non_local_pin() -> None:
    text = 'project = { optional-dependencies = { cpu = ["torch==2.10.0"] } }'
    with pytest.raises(SystemExit):
        list(pins.iter_torch_pins(text))


def test_iter_torch_pins_rejects_range_pin() -> None:
    text = 'project = { optional-dependencies = { cpu = ["torch>=2.10,<2.11"] } }'
    with pytest.raises(SystemExit):
        list(pins.iter_torch_pins(text))


def test_documented_claims_cover_extras_and_lane() -> None:
    indexes, lane_specs = pins.documented_claims()
    assert "https://download.pytorch.org/whl/cpu" in indexes
    assert "https://download.pytorch.org/whl/cu128" in indexes
    lane = pins.load_wheel_lane()
    assert any(SpecifierSet(s) == lane for s in lane_specs)


def test_check_doc_drift_reports_undocumented_index() -> None:
    lane = pins.load_wheel_lane()
    fails = pins.check_doc_drift(lane, [_pin("cuda", "2.10.0+cu999")])
    assert any("cu999" in f for f in fails)


def test_check_doc_drift_passes_for_real_pins() -> None:
    lane = pins.load_wheel_lane()
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    real_pins = list(pins.iter_torch_pins(text))
    if not real_pins:
        # Both flavor pins are Linux-marked; on macOS nothing resolves and the
        # doc-drift contract is vacuous — skip rather than fail macOS CI.
        pytest.skip("no torch flavor pins resolve on this platform")
    assert pins.check_doc_drift(lane, real_pins) == []
