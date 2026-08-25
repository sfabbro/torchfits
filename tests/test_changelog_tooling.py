"""Unit tests for scripts/update_changelog.py (pure functions, no git)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_changelog as changelog  # noqa: E402


def test_parse_subject_maps_conventional_types() -> None:
    assert changelog.parse_subject("feat(api): add rgb()") == (
        "Added",
        "- api: Add rgb()",
    )
    assert changelog.parse_subject("fix: buffer rotation only while prefetching") == (
        "Fixed",
        "- Buffer rotation only while prefetching",
    )
    assert changelog.parse_subject("perf(cpp): gate prefetch to 64MB") == (
        "Performance",
        "- cpp: Gate prefetch to 64MB",
    )
    assert changelog.parse_subject("docs: note lane pins") == (
        "Docs",
        "- Note lane pins",
    )


def test_parse_subject_skips_noise_and_unknown() -> None:
    assert changelog.parse_subject("chore(release): 1.1.0b1") is None
    assert changelog.parse_subject("ci: tweak workflow") is None
    assert changelog.parse_subject("refactor: split mega function") is None
    assert changelog.parse_subject(
        "refactor: split mega function", include_all=True
    ) == (
        "Changed",
        "- Split mega function",
    )
    assert changelog.parse_subject("not a conventional subject") is None


def test_merge_generated_groups_and_dedupes() -> None:
    text = (
        "# Changelog\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "Intro prose.\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- Buffer rotation only while prefetching.\n"
        "\n"
        "## [1.0.0] — 2026-08-09\n"
    )
    bullets = [
        ("Added", "- Checksum-stamped writes"),
        ("Fixed", "- Buffer rotation only while prefetching"),  # dedupe vs existing
        ("Performance", "- Prefetch gate raised to 64MB"),
        ("Fixed", "- New fix bullet"),
    ]
    merged = changelog.merge_generated(text, bullets)
    unreleased = merged.split("## Unreleased\n", 1)[1].split("\n## [1.0.0]", 1)[0]
    # Existing bullet kept exactly once; new fix appended under the same section.
    assert (
        len(re.findall("rotation only while prefetching", unreleased, flags=re.I)) == 1
    )
    assert unreleased.index("### Fixed") < unreleased.index("New fix bullet")
    # Missing sections are created in canonical order after existing content.
    assert "### Added" in unreleased and "### Performance" in unreleased
    assert unreleased.index("### Added") > unreleased.index("### Fixed")
    assert unreleased.index("### Performance") > unreleased.index("### Added")
    # The released section below is untouched.
    assert merged.endswith("## [1.0.0] — 2026-08-09\n")


def test_stamp_release_renames_and_reopens_unreleased() -> None:
    text = (
        "# Changelog\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- Something.\n"
        "\n"
        "## [1.0.0] — 2026-08-09\n"
        "\n"
        "Old.\n"
        "\n"
        "[Unreleased]: https://github.com/astroai/torchfits/compare/v1.0.0...HEAD\n"
        "[1.0.0]: https://github.com/astroai/torchfits/compare/v1.0.0rc5...v1.0.0\n"
    )
    stamped = changelog.stamp_release(text, "1.1.0", "2026-08-23")
    headings = [line for line in stamped.splitlines() if line.startswith("## ")]
    assert headings[0] == "## Unreleased"
    assert headings[1] == "## [1.1.0] — 2026-08-23"
    assert (
        "[Unreleased]: https://github.com/astroai/torchfits/compare/v1.1.0...HEAD"
        in stamped
    )
    assert (
        "[1.1.0]: https://github.com/astroai/torchfits/compare/v1.0.0...v1.1.0"
        in stamped
    )
    assert stamped.count("[1.1.0]:") == 1


def test_stamp_release_is_idempotent_for_repeat_calls() -> None:
    text = (
        "## Unreleased\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- Something.\n"
        "\n"
        "## [1.0.0] — 2026-08-09\n"
        "\n"
        "[Unreleased]: https://github.com/astroai/torchfits/compare/v1.0.0...HEAD\n"
        "[1.0.0]: https://github.com/astroai/torchfits/compare/v1.0.0...v1.0.0\n"
    )
    once = changelog.stamp_release(text, "1.1.0", "2026-08-23")
    twice = changelog.stamp_release(
        once.replace(
            "## Unreleased\n\n", "## Unreleased\n\n### Fixed\n\n- More.\n\n", 1
        ),
        "1.2.0",
        "2026-09-01",
    )
    assert twice.count("[1.1.0]:") == 1
    assert (
        "[1.2.0]: https://github.com/astroai/torchfits/compare/v1.1.0...v1.2.0" in twice
    )
    assert (
        "[Unreleased]: https://github.com/astroai/torchfits/compare/v1.2.0...HEAD"
        in twice
    )


def test_repo_changelog_tracks_unreleased_without_version() -> None:
    text = (ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    first_version_heading = re.search(r"^## .*$", text, flags=re.M)
    assert first_version_heading is not None
    assert first_version_heading.group(0) == "## Unreleased", (
        "the top changelog section must be a versionless '## Unreleased'"
    )
    assert not re.search(r"^## \[[^\]]+\][^\n]*Unreleased", text, flags=re.M), (
        "no heading may pair a version number with 'Unreleased'"
    )


def test_issue_ids_extraction() -> None:
    assert changelog._issue_ids(
        "fix(cli): Integer-safe arith (B2), uint stats (B3), NaN-aware diff (H6)"
    ) == {"B2", "B3", "H6"}
    assert changelog._issue_ids("docs: rewrite install page") == set()
    # FITS/HCOMPRESS-style words must not match.
    assert changelog._issue_ids("HCOMPRESS tiles for FITS files") == set()


def test_merge_generated_suppresses_id_covered_subjects() -> None:
    """A generated bullet whose tracking IDs are already cited anywhere in
    Unreleased counts as merged, even when the curated wording differs."""
    text = (
        "# Changelog\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- **CompImage null pixels decode as NaN (B1)** with astropy parity.\n"
        "- **arith no longer wraps integer images**; saturating cast (B2).\n"
        "\n"
        "## [1.0.0] — 2026-08-09\n"
    )
    bullets = [
        (
            "Fixed",
            "- cpp: Decode CompImage null pixels as NaN (B1); Random Groups rejection",
        ),
        ("Fixed", "- cli: Integer-safe arith (B2)"),
        ("Fixed", "- tables: brand new fix with no cited ID yet"),
    ]
    merged = changelog.merge_generated(text, [list(b) for b in bullets])
    unreleased = merged.split("## Unreleased\n", 1)[1].split("\n## [1.0.0]", 1)[0]
    assert "cpp: Decode CompImage" not in unreleased  # B1 covered
    assert "cli: Integer-safe arith" not in unreleased  # B2 covered
    assert "- tables: brand new fix with no cited ID yet" in unreleased


def test_issue_id_regex_ignores_plain_words() -> None:
    assert changelog._issue_ids("M16 bench harness") == {"M16"}


def test_release_notes_extracts_stamped_then_unreleased(tmp_path) -> None:
    import release_notes as rn  # noqa: PLC0415  (sys.path set by conftest-ish header)

    text = (
        "# Changelog\n"
        "\n"
        "## Unreleased\n"
        "\n"
        "Fresh work.\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- Something (B1).\n"
        "\n"
        "## [1.1.0] — 2026-08-25\n"
        "\n"
        "Stamped highlights.\n"
        "\n"
        "## [1.0.0] — 2026-08-09\n"
    )
    label, body = rn.extract(text, "1.1.0")
    assert label == "[1.1.0]" and body == "Stamped highlights."

    label, body = rn.extract(text, "1.2.0")  # unstamped -> prerelease fallback
    assert label == "Unreleased" and body.startswith("Fresh work.")

    import pytest  # noqa: PLC0415

    with pytest.raises(SystemExit):
        rn.extract("# Changelog\n", "9.9.9")
