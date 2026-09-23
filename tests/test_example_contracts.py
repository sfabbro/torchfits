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


def test_cfitsio_ascii_failure_is_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import examples.example_cfitsio_cookbook as cook

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("ascii broke")

    monkeypatch.setattr(cook.tf_table, "read", boom)
    with pytest.raises(RuntimeError, match="ascii broke"):
        cook.test_cfitsio_ascii_table(str(tmp_path))
    assert "ASCII: OK" not in capsys.readouterr().out


def test_cfitsio_checksum_example_rejects_a_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import examples.example_cfitsio_cookbook as cook

    monkeypatch.setattr(
        cook.torchfits,
        "verify_checksums",
        lambda _path, hdu=0: {
            "datastatus": -1,
            "hdustatus": -1,
            "ok": False,
            "present": True,
            "status": "fail",
        },
    )
    with pytest.raises(AssertionError):
        cook.test_cfitsio_checksum(str(tmp_path))


def test_cfitsio_multi_hdu_copy_uses_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import examples.example_cfitsio_cookbook as cook

    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append([str(part) for part in cmd])
        return real_run(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", spy)
    cook.test_cfitsio_multi_hdu_copy(str(tmp_path))
    assert any(
        "torchfits.cli" in part and "copy" in parts for parts in calls for part in parts
    )


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
