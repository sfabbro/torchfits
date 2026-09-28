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


@pytest.fixture
def linux_pins(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, Version]]:
    """Make the real ``pyproject.toml`` pins resolve, on any host platform.

    Both flavor pins are marked ``sys_platform == 'linux'``, so
    ``iter_torch_pins`` yields nothing on macOS and the three guard tests below
    used to ``pytest.skip`` there. But the contracts they check are
    *platform-independent*: ``documented_claims()`` and ``check_doc_drift()``
    never read ``sys.platform`` -- only ``iter_torch_pins``' marker evaluation
    does, and the module documents that patches are honoured precisely so this
    works. ``test_real_extras_marker_skip_across_platforms`` in this same file
    already monkeypatches ``sys.platform`` to "linux" to get the same two pins.

    So this is not "pretend the host is Linux" for a platform-specific
    behaviour; it is evaluating a *documented-marker* claim on a host that
    cannot otherwise observe it. Running these guards on macOS means doc drift
    is caught on the dev platform too, rather than only on Linux CI.

    The fixture asserts the pins resolved instead of skipping when they do not,
    so a future ``pyproject.toml`` that drops the markers fails loudly here
    rather than quietly turning these three back into no-ops.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    resolved = list(pins.iter_torch_pins(text))
    assert resolved, (
        "no torch flavor pins resolved even with sys.platform='linux'; the real "
        "pyproject.toml no longer carries linux-marked torch pins, so these "
        "guards need real fixtures instead of the project's own extras"
    )
    return resolved


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


def test_documented_claims_cover_extras_and_lane(linux_pins) -> None:
    indexes, lane_specs, exact_pins, prose_minors = pins.documented_claims()
    real_pins = linux_pins
    for _extra, _entry, version in real_pins:
        assert f"https://download.pytorch.org/whl/{version.local}" in indexes
        assert f"torch=={version}" in exact_pins
    lane = pins.load_wheel_lane()
    assert any(SpecifierSet(s) == lane for s in lane_specs)
    # The prose minor is the *other* spelling of the same claim (README's
    # "**PyTorch 2.13.x**", the compatibility matrix's bare "2.13.x" cell).
    # It has no specifier to compare, so it needs its own agreement.
    assert prose_minors == {_lane_minor(lane)}, (
        f"docs name torch minors {sorted(prose_minors)} in prose but the wheel "
        f"lane is 2.{_lane_minor(lane)}.x"
    )


def test_check_doc_drift_rejects_a_stale_prose_lane_minor() -> None:
    """A lane bump that misses the README headline must fail the doc gate.

    `check_doc_drift` compares the *specifier* claims against the lane, and the
    specifier form lives only in docs/install.md -- so before the prose form was
    collected, renaming the README's "PyTorch 2.13.x" to any other minor still
    reported `[ OK ] install one-liners match the extra pins and the wheel
    lane`. The headline sentence is the first thing a reader sees, so it is the
    one claim that must not be able to go stale silently.
    """
    lane = pins.load_wheel_lane()
    minor = _lane_minor(lane)
    stale = str(int(minor) - 1)
    extras = _real_extras()

    readme = pins.ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    try:
        mutated = original.replace(f"2.{minor}.x", f"2.{stale}.x")
        assert mutated != original, f"README.md no longer states 2.{minor}.x"
        readme.write_text(mutated, encoding="utf-8")
        fails = pins.check_doc_drift(lane, extras)
    finally:
        readme.write_text(original, encoding="utf-8")
    assert any(f"2.{stale}.x" in f and "wheel lane" in f for f in fails), (
        f"a stale prose lane minor went unreported; failures were {fails}"
    )
    # ...and the same gate is silent on the real tree, so the assertion above is
    # about the stale claim specifically and not a gate that always fires.
    assert pins.check_doc_drift(lane, extras) == []


def _lane_minor(lane: SpecifierSet) -> str:
    """The lower-bound minor of a `>=x.y,<x.z+1` lane specifier set.

    A SpecifierSet is a frozenset, so `next(iter(...))` is not the lower bound;
    pick the specifier that actually carries one.
    """
    bounds = [spec for spec in lane if spec.operator in {">=", "==", "~="}]
    assert bounds, f"wheel lane {lane} has no lower bound"
    return bounds[0].version.split(".")[1]


def _real_extras() -> list[tuple[str, str, Version]]:
    """The extras as declared, so the prose check runs with no other failures."""
    text = (pins.ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return list(pins.iter_all_torch_pins(text))


def test_check_doc_drift_reports_undocumented_index() -> None:
    lane = pins.load_wheel_lane()
    fails = pins.check_doc_drift(lane, [_pin("cuda", "2.10.0+cu999")])
    assert any("cu999" in f for f in fails)


def test_check_doc_drift_reports_undocumented_exact_pin() -> None:
    """Docs may document the index but show a stale exact pin — must fail."""
    lane = pins.load_wheel_lane()
    fails = pins.check_doc_drift(lane, [_pin("cpu", "2.10.1+cpu")])
    assert any("torch==2.10.1+cpu" in f for f in fails)


def test_check_doc_drift_passes_for_real_pins(linux_pins) -> None:
    lane = pins.load_wheel_lane()
    assert pins.check_doc_drift(lane, linux_pins) == []


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
    for suffix in ("rc5", "b1", "a2", "beta1", "alpha3", "rc1.post1"):
        assert pins.lane_for_version(f"{base}{suffix}") == current, suffix


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
    monkeypatch: pytest.MonkeyPatch, linux_pins
) -> None:
    monkeypatch.setattr(
        pins,
        "resolve",
        lambda _spec, _index: subprocess.CompletedProcess([], 1, "", "boom"),
    )
    assert pins.main() == 1


def test_main_vacuous_pass_when_all_pins_marker_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """macOS-like: every pin is Linux-marked, so *resolution* is vacuous.

    The install-docs guard is platform-independent and must still run, or
    ``pixi run check-torch-pins`` stops guarding ``docs/install.md`` on the dev
    platform. The docs check is stubbed here because the fake pin below is
    deliberately not the one the real docs show.
    """
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(
        pins,
        "_iter_torch_entries",
        lambda _text: [("cpu", "torch==2.10.0+cpu; sys_platform == 'linux'")],
    )
    seen: list[list[tuple[str, str, Version]]] = []
    # The lane-map check reads the real pyproject and has its own tests; the
    # fake 2.10.0 entry below would otherwise trip it.
    monkeypatch.setattr(pins, "check_lane_map_consistency", lambda _lane: [])
    monkeypatch.setattr(
        pins, "check_doc_drift", lambda _lane, plist: seen.append(list(plist)) or []
    )
    assert pins.main() == 0
    out = capsys.readouterr().out
    assert "skipped by platform markers" in out
    assert "install docs still checked" in out
    # The skipped Linux-only pins are what the docs check ran against.
    assert [(_e, _p) for _e, _p, _v in seen[0]] == [
        ("cpu", "torch==2.10.0+cpu; sys_platform == 'linux'")
    ]
    assert [v.local for _e, _p, v in seen[0]] == ["cpu"]


def test_main_fails_on_install_doc_drift_with_every_pin_marker_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A docs regression must fail on macOS, not only on Linux CI."""
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(pins, "check_lane_map_consistency", lambda _lane: [])
    monkeypatch.setattr(
        pins, "check_doc_drift", lambda _lane, _pins: ["[FAIL] docs drifted"]
    )
    monkeypatch.setattr(
        pins,
        "_iter_torch_entries",
        lambda _text: [("cpu", "torch==2.10.0+cpu; sys_platform == 'linux'")],
    )
    assert pins.main() == 1
    assert "docs drifted" in capsys.readouterr().out


def test_main_does_not_discard_a_lane_map_failure_on_the_vacuous_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The marker-skipped path must not ``return 0`` past a pending failure.

    The lane-map consistency check runs before the pin loop; its verdict used
    to be dropped on the way out of the vacuous branch, so a drifted
    ``[cpu]``/``[cuda]`` extra passed as long as the host was not Linux.
    """
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(
        pins, "check_lane_map_consistency", lambda _lane: ["[FAIL] extra drifted"]
    )
    monkeypatch.setattr(pins, "check_doc_drift", lambda _lane, _pins: [])
    monkeypatch.setattr(
        pins,
        "_iter_torch_entries",
        lambda _text: [("cpu", "torch==2.10.0+cpu; sys_platform == 'linux'")],
    )
    assert pins.main() == 1
    assert "extra drifted" in capsys.readouterr().out


def test_iter_all_torch_pins_ignores_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The docs-facing iterator must reach Linux-only pins from any host."""
    monkeypatch.setattr(sys, "platform", "darwin")
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    every = list(pins.iter_all_torch_pins(text))
    assert every, "the real pyproject.toml must carry torch flavor extras"
    assert list(pins.iter_torch_pins(text)) == []


def test_main_fails_when_no_torch_extras_at_all(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing the extras entirely must fail the guard, not pass silently."""
    monkeypatch.setattr(pins, "iter_torch_pins", lambda _text: [])
    monkeypatch.setattr(pins, "_iter_torch_entries", lambda _text: [])
    assert pins.main() == 1
    out = capsys.readouterr().out
    assert "no torch flavor extras" in out
