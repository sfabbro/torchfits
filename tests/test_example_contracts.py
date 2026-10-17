"""Example scripts must check what they claim and stay out of the docs tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_rgb_sky_does_not_target_the_docs_tree() -> None:
    import examples.example_rgb_sky as sky

    assert "docs" not in str(getattr(sky, "GALLERY_DIR", ""))
    assert "docs/assets" not in Path(sky.__file__).read_text(encoding="utf-8")


def test_make_rgb_demo_defaults_under_examples_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = ROOT / "examples" / "cli" / "make_rgb_demo.py"
    subprocess.run([sys.executable, str(script)], check=True)
    assert not (tmp_path / "docs").exists()
    assert (ROOT / "examples" / "output" / "cli_rgb_demo.png").is_file()


def test_identity_claims_match_the_checks() -> None:
    text = (ROOT / "examples" / "example_identity_stress.py").read_text(
        encoding="utf-8"
    )
    assert "1.1.1" not in text
    assert "TNULL" not in text
    assert "checksum" not in text.lower()


def test_desi_shaped_spectrum_is_discovered_and_leaves_no_tmp_file() -> None:
    from examples.test_examples import _discover_examples

    assert "desi_shaped_spectrum.py" in _discover_examples()
    leftover = Path("/tmp/torchfits_desi_shaped.fits")
    if leftover.exists():
        leftover.unlink()
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / "desi_shaped_spectrum.py")],
        check=True,
    )
    assert not leftover.exists()
