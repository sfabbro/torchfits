"""Unit tests for scripts/check_release_rc.py (no network, no tag creation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_rc as gate  # noqa: E402
import check_release_tag as release_tag  # noqa: E402


def test_split_version_accepts_stable_and_prerelease() -> None:
    assert gate.split_version("1.3.1") == ("1.3.1", None)
    assert gate.split_version("1.3.1rc2") == ("1.3.1", "rc2")
    with pytest.raises(ValueError):
        gate.split_version("1.3.1b1")
    with pytest.raises(ValueError):
        gate.split_version("1.3.1a1")
    with pytest.raises(ValueError):
        gate.split_version("1.3")


@pytest.mark.parametrize("tag", ["v1.3.1a1", "v1.3.1b1", "v1.3.1rc1"])
def test_manual_release_accepts_only_exact_stable_tag(tag: str) -> None:
    with pytest.raises(ValueError):
        release_tag.release_kind(tag, "workflow_dispatch")
    assert release_tag.release_kind("v1.3.1", "workflow_dispatch") == "stable"


@pytest.mark.parametrize("tag", ["v1.3.1a1", "v1.3.1b1", "v1.3.1dev1"])
def test_push_prerelease_accepts_only_rc(tag: str) -> None:
    with pytest.raises(ValueError):
        release_tag.release_kind(tag, "push")
    assert release_tag.release_kind("v1.3.1rc1", "push") == "rc"


def test_rc_tags_require_the_same_base_version() -> None:
    tags = ["v1.3.1rc10", "v1.3.1rc2", "v1.3.0rc1", "v1.3.1"]
    assert gate.rc_tags("1.3.1", tags) == ["v1.3.1rc2", "v1.3.1rc10"]
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
    monkeypatch.setattr(gate, "validate_source_tree", lambda tag: ["pyproject.toml"])
    assert gate.main(["--version", "1.2.0"]) == 0


def test_gate_selects_latest_numeric_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[str] = []
    monkeypatch.setattr(
        gate,
        "git_tags",
        lambda: ["v1.2.0rc10", "v1.2.0rc9", "v1.2.0rc2"],
    )
    monkeypatch.setattr(
        gate,
        "validate_source_tree",
        lambda tag: selected.append(tag) or [],
    )
    assert gate.main(["--version", "1.2.0", "--print-tag"]) == 0
    assert selected == ["v1.2.0rc10"]


def test_gate_rejects_source_changes_after_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "git_tags", lambda: ["v1.2.0rc1"])

    def reject(tag: str) -> list[str]:
        raise ValueError(f"source changed after {tag}: src/torchfits/io.py")

    monkeypatch.setattr(gate, "validate_source_tree", reject)
    assert gate.main(["--version", "1.2.0"]) == 1


def test_validate_source_tree_allows_only_release_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def run(command, **kwargs):
        if command[1] == "diff":
            return Result(stdout="pyproject.toml\ndocs/changelog.md\n")
        if command[1] == "show":
            version = "1.2.0rc1" if command[2].startswith("v1.2.0rc1:") else "1.2.0"
            return Result(stdout=f'version = "{version}"\n')
        return Result()

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.validate_source_tree("v1.2.0rc1") == [
        "pyproject.toml",
        "docs/changelog.md",
    ]


def test_validate_source_tree_rejects_workflow_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stdout = ".github/workflows/build_wheels.yml\n"

    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match="build_wheels.yml"):
        gate.validate_source_tree("v1.2.0rc1")


def test_validate_source_tree_rejects_non_version_metadata_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def run(command, **kwargs):
        if command[1] == "diff":
            return Result(stdout="pyproject.toml\n")
        if command[1] == "show":
            if command[2].startswith("v1.2.0rc1:"):
                return Result(stdout='version = "1.2.0rc1"\ndependencies = ["torch"]\n')
            return Result(
                stdout='version = "1.2.0"\ndependencies = ["torch", "numpy"]\n'
            )
        return Result()

    monkeypatch.setattr(gate.subprocess, "run", run)
    with pytest.raises(ValueError, match="non-version content changed"):
        gate.validate_source_tree("v1.2.0rc1")
