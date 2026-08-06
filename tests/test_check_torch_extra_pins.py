"""Unit tests for scripts/check_torch_extra_pins.py (no network needed)."""

from __future__ import annotations

import json
import subprocess
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
    lanes = json.loads(
        (ROOT / "scripts" / "torch_lanes.json").read_text(encoding="utf-8")
    )
    current = max(lanes)
    major, minor = map(int, current.split("."))
    lane = pins.load_wheel_lane()
    assert lane.contains(f"{current}.0")
    assert not lane.contains(f"{major}.{minor + 1}.0")
    for older in lanes:
        if older != current:
            assert not lane.contains(f"{older}.0")


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
    indexes, lane_specs, exact_pins = pins.documented_claims()
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    real_pins = list(pins.iter_torch_pins(text))
    if not real_pins:
        # Both flavor pins are Linux-marked; on macOS nothing resolves and the
        # doc-drift contract is vacuous — skip rather than fail macOS CI.
        pytest.skip("no torch flavor pins resolve on this platform")
    for _extra, _entry, version in real_pins:
        assert f"https://download.pytorch.org/whl/{version.local}" in indexes
        assert f"torch=={version}" in exact_pins
    lane = pins.load_wheel_lane()
    assert any(SpecifierSet(s) == lane for s in lane_specs)


def test_check_doc_drift_reports_undocumented_index() -> None:
    lane = pins.load_wheel_lane()
    fails = pins.check_doc_drift(lane, [_pin("cuda", "2.10.0+cu999")])
    assert any("cu999" in f for f in fails)


def test_check_doc_drift_reports_undocumented_exact_pin() -> None:
    """Docs may document the index but show a stale exact pin — must fail."""
    lane = pins.load_wheel_lane()
    fails = pins.check_doc_drift(lane, [_pin("cpu", "2.10.1+cpu")])
    assert any("torch==2.10.1+cpu" in f for f in fails)


def test_check_doc_drift_passes_for_real_pins() -> None:
    lane = pins.load_wheel_lane()
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    real_pins = list(pins.iter_torch_pins(text))
    if not real_pins:
        # Both flavor pins are Linux-marked; on macOS nothing resolves and the
        # doc-drift contract is vacuous — skip rather than fail macOS CI.
        pytest.skip("no torch flavor pins resolve on this platform")
    assert pins.check_doc_drift(lane, real_pins) == []


def test_real_extras_marker_skip_across_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real pyproject pins resolve on Linux and skip everywhere else."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    assert len(list(pins.iter_torch_pins(text))) == 2
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        assert list(pins.iter_torch_pins(text)) == []


def test_main_fails_when_pin_off_wheel_lane(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lane guard itself: an off-lane pin must fail the check."""
    monkeypatch.setattr(
        pins, "iter_torch_pins", lambda _text: [_pin("cuda", "2.11.0+cu128")]
    )
    monkeypatch.setattr(pins, "check_doc_drift", lambda _lane, _pins: [])
    assert pins.main() == 1
    out = capsys.readouterr().out
    assert "outside the wheel ABI lane" in out
    assert "2.11.0" in out


def test_lane_for_version_accepts_prerelease_suffix() -> None:
    """rc prerelease states map to the lane of their base version."""
    lanes = json.loads(
        (ROOT / "scripts" / "torch_lanes.json").read_text(encoding="utf-8")
    )
    current = max(lanes)
    base = lanes[current]["torchfits_version"]
    assert pins.lane_for_version(base) == current
    assert pins.lane_for_version(f"{base}rc5") == current


def test_main_fails_when_lane_map_consistency_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A [FAIL] from the lane-map check must propagate to the exit code."""
    monkeypatch.setattr(
        pins, "iter_torch_pins", lambda _text: [_pin("cpu", "2.13.0+cpu")]
    )
    monkeypatch.setattr(
        pins, "check_lane_map_consistency", lambda _lane: ["[FAIL] lane drifted"]
    )
    monkeypatch.setattr(pins, "check_doc_drift", lambda _lane, _pins: [])
    monkeypatch.setattr(
        pins,
        "resolve",
        lambda _spec, _index: subprocess.CompletedProcess([], 0, "", ""),
    )
    assert pins.main() == 1
    assert "lane drifted" in capsys.readouterr().out


def test_main_succeeds_for_real_pins_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end guard (real lane + real pyproject) with a fake pip dry-run."""
    monkeypatch.setattr(
        pins,
        "resolve",
        lambda _spec, _index: subprocess.CompletedProcess([], 0, "", ""),
    )
    assert pins.main() == 0


def test_main_fails_when_pin_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if not list(pins.iter_torch_pins(text)):
        pytest.skip("no torch flavor pins resolve on this platform")
    monkeypatch.setattr(
        pins,
        "resolve",
        lambda _spec, _index: subprocess.CompletedProcess([], 1, "", "boom"),
    )
    assert pins.main() == 1


def test_main_vacuous_pass_when_all_pins_marker_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """macOS-like: every pin is Linux-marked, so the guard passes vacuously."""
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(
        pins,
        "_iter_torch_entries",
        lambda _text: [("cpu", "torch==2.10.0+cpu; sys_platform == 'linux'")],
    )
    assert pins.main() == 0
    out = capsys.readouterr().out
    assert "skipped by platform markers" in out


def test_main_fails_when_no_torch_extras_at_all(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing the extras entirely must fail the guard, not pass silently."""
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(pins, "_iter_torch_entries", lambda _text: [])
    assert pins.main() == 1
    out = capsys.readouterr().out
    assert "no torch flavor extras" in out
