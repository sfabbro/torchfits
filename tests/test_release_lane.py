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
