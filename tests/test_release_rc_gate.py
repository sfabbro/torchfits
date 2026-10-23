"""Unit tests for scripts/check_release_rc.py (no network, no tag creation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_rc as gate  # noqa: E402


def test_split_version_accepts_stable_and_prerelease() -> None:
    assert gate.split_version("1.3.1") == ("1.3.1", None)
    assert gate.split_version("1.3.1rc2") == ("1.3.1", "rc2")
    assert gate.split_version("1.3.1b1") == ("1.3.1", "b1")
    with pytest.raises(ValueError):
        gate.split_version("1.3")


def test_rc_tags_require_the_same_base_version() -> None:
    tags = ["v1.3.1rc1", "v1.3.1rc2", "v1.3.0rc1", "v1.3.1"]
    assert gate.rc_tags("1.3.1", tags) == ["v1.3.1rc1", "v1.3.1rc2"]
    assert gate.rc_tags("1.3.2", tags) == []
    assert gate.rc_tags("1.3.1rc1", tags) == []


def test_gate_passes_for_prerelease_without_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "git_tags", lambda: [])
    assert gate.main(["--version", "1.3.1rc1"]) == 0


def test_gate_fails_for_stable_without_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "git_tags", lambda: ["v1.3.0rc1", "v1.3.0"])
    assert gate.main(["--version", "1.3.1"]) == 1


def test_gate_passes_for_stable_with_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "git_tags", lambda: ["v1.2.0rc1", "v1.2.0"])
    assert gate.main(["--version", "1.2.0"]) == 0
