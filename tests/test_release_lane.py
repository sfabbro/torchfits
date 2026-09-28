"""Unit tests for scripts/release_lane.py (no network, no disk writes)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import release_lane as lane  # noqa: E402


def _rendered_version(rendered: dict[Path, str]) -> str:
    match = re.search(r'^version = "([^"]+)"$', rendered[lane.PYPROJECT], re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_strip_prerelease() -> None:
    assert lane.strip_prerelease("1.0.0") == "1.0.0"
    assert lane.strip_prerelease("1.0.0rc5") == "1.0.0"
    assert lane.strip_prerelease("1.0.0rc5.post1") == "1.0.0"
    assert lane.strip_prerelease("1.0.0b1") == "1.0.0"
    assert lane.strip_prerelease("1.0.0a2") == "1.0.0"
    assert lane.strip_prerelease("1.0.0.dev0+torch212") == "1.0.0"


def _map_version() -> str:
    return lane.load_lanes()["2.13"]["torchfits_version"]


def test_lane_for_version_accepts_prerelease() -> None:
    lanes = lane.load_lanes()
    base = _map_version()
    assert lane.lane_for_version(base, lanes) == "2.13"
    assert lane.lane_for_version(f"{base}rc5", lanes) == "2.13"
    with pytest.raises(SystemExit):
        lane.lane_for_version("2.0.0rc1", lanes)


def test_render_prerelease_applies_suffix() -> None:
    rendered = lane.render("2.13", None, prerelease="rc5")
    expected = f"{_map_version()}rc5"
    assert _rendered_version(rendered) == expected
    for path, text in rendered.items():
        if path.name in ("pyproject.toml", "constraints-wheel.txt"):
            continue
        assert re.search(re.escape(expected), text) is not None


def test_render_committed_prerelease_matches_apply() -> None:
    rendered = lane.render("2.13", None, prerelease="rc5")
    committed = lane.render("2.13", f"{_map_version()}rc5")
    assert rendered == committed


def test_render_plain_apply_is_map_version() -> None:
    base = _map_version()
    assert _rendered_version(lane.render("2.13", None)) == base
    assert _rendered_version(lane.render("2.13", base)) == base


def test_render_rejects_wrong_base_version() -> None:
    with pytest.raises(SystemExit, match=r"not '1\.0\.1rc5'"):
        lane.render("2.13", "1.0.1rc5")
    with pytest.raises(SystemExit, match=r"not '2\.0\.0'"):
        lane.render("2.13", "2.0.0")


def test_render_prerelease_rejects_non_release_lane() -> None:
    with pytest.raises(SystemExit, match="only applies to release lanes"):
        lane.render("2.12", None, prerelease="rc5")


def test_prerelease_help_names_the_lane_base() -> None:
    """--help must show the torch_lanes.json base, not a stale 1.0.0 example."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "release_lane.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    base = _map_version()
    assert f"{base}rc5" in proc.stdout
    assert "1.0.0rc5" not in proc.stdout


def test_pixi_package_uses_vendored_cfitsio_not_conda_4_6() -> None:
    """The conda package compiles vendored CFITSIO; it must not depend on 4.6."""
    text = (ROOT / "pixi.toml").read_text(encoding="utf-8")
    package = text.split("\n[dependencies]\n", 1)[0]
    assert 'cfitsio = "' not in package
    host = package.split("[package.host-dependencies]", 1)[1]
    host = host.split("\n[", 1)[0]
    assert "bzip2" in host


def test_local_lane_recovers_the_experimental_lane() -> None:
    """``render`` writes ``+torch<major><minor>``; the reader must invert it.

    The same recovery lives in ``scripts/verify_wheel_matrix.sh`` (from a wheel
    filename), so the encoding is a repo convention, not a one-off.
    """
    base = _map_version()
    assert lane.local_lane(f"{base}.dev0+torch212") == "2.12"
    assert lane.local_lane(f"{base}.dev0+torch213") == "2.13"
    assert lane.local_lane(base) is None
    assert lane.local_lane(f"{base}rc5") is None
    # Only a dev-local segment names a lane; a prerelease is still the map lane.
    assert lane.lane_for_version(f"{base}rc5", lane.load_lanes()) == "2.13"


def _lane_tree(tmp_path: Path, lane_name: str) -> None:
    """A throwaway tree rendered onto ``lane_name`` by the real renderer."""
    for name in ("pyproject.toml", "constraints-wheel.txt", "pixi.toml"):
        (tmp_path / name).write_text(
            (ROOT / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "recipe.yaml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "recipe.yaml").write_text(
        (ROOT / "packaging/conda/recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "__init__.py").write_text(
        (ROOT / "src/torchfits/__init__.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for path, rendered in lane.render(lane_name, None).items():
        path = tmp_path / path.name
        path.write_text(rendered, encoding="utf-8")


def _use_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(lane, "CONSTRAINTS", tmp_path / "constraints-wheel.txt")
    monkeypatch.setattr(lane, "PIXI", tmp_path / "pixi.toml")
    monkeypatch.setattr(lane, "RECIPE", tmp_path / "recipe.yaml")
    monkeypatch.setattr(lane, "INIT", tmp_path / "__init__.py")
    monkeypatch.setattr(lane, "LANES_FILE", ROOT / "scripts" / "torch_lanes.json")


def test_current_lane_follows_an_experimental_lane_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tree rendered onto 2.12 must not report the 2.13 release lane.

    ``--print-pins`` feeds ``TORCH_PIN`` to seven CI jobs, which install
    ``torch>=<pin>``; reporting the map lane for a dev-lane tree would install a
    torch the tree is not ABI-matched to.
    """
    _lane_tree(tmp_path, "2.12")
    _use_tree(monkeypatch, tmp_path)
    assert lane.committed_torch_spec() == ">=2.12,<2.13"
    assert lane.current_lane() == "2.12"
    assert lane.check() == 0


def test_print_pins_reports_the_committed_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _lane_tree(tmp_path, "2.12")
    _use_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["release_lane.py", "--print-pins"])
    assert lane.main() == 0
    out = capsys.readouterr().out
    assert "lane=2.12" in out
    assert "torch=>=2.12,<2.13" in out


def test_print_pins_fails_loudly_when_the_tree_pin_disagrees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drifted constraints file must not be papered over with a map lookup."""
    _lane_tree(tmp_path, "2.13")
    _use_tree(monkeypatch, tmp_path)
    constraints = tmp_path / "constraints-wheel.txt"
    text, count = re.subn(
        r"^torch>=2\.13,<2\.14$",
        "torch>=2.12,<2.13",
        constraints.read_text(),
        count=0,
        flags=re.M,
    )
    assert count == 1
    constraints.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["release_lane.py", "--print-pins"])
    assert lane.main() == 1
    assert "constraints-wheel.txt" in capsys.readouterr().out


def test_print_pins_reports_the_committed_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prerelease state prints its own version, not the map's base version."""
    _lane_tree(tmp_path, "2.13")
    _use_tree(monkeypatch, tmp_path)
    expected = f"{_map_version()}rc5"
    for name in ("pyproject.toml", "pixi.toml", "recipe.yaml", "__init__.py"):
        path = tmp_path / name
        text, count = re.subn(re.escape(_map_version()), expected, path.read_text())
        assert count >= 1, name
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["release_lane.py", "--print-pins"])
    assert lane.main() == 0
    assert f"version={expected}" in capsys.readouterr().out
