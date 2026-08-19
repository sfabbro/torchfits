"""Automated testing for code snippets embedded in documentation markdown files."""

from __future__ import annotations

import ast
import re
import shlex
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.table import Table

from torchfits.cli.main import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"


class CodeSnippet(NamedTuple):
    file_path: Path
    line_number: int
    lang: str
    code: str


def _extract_snippets(lang: str) -> list[CodeSnippet]:
    """Extract all fenced code blocks of a given language from docs/*.md."""
    snippets: list[CodeSnippet] = []
    pattern = re.compile(rf"```{lang}[^\n]*\n(.*?)```", re.DOTALL)

    for md_path in sorted(DOCS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            raw_code = match.group(1)
            code = textwrap.dedent(raw_code)
            line_num = text[: match.start()].count("\n") + 1
            snippets.append(
                CodeSnippet(
                    file_path=md_path,
                    line_number=line_num,
                    lang=lang,
                    code=code,
                )
            )
    return snippets


@pytest.fixture(scope="module")
def mock_fits_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory with standard FITS files for doc snippet execution."""
    ws = tmp_path_factory.mktemp("doc_fits_ws")

    # 1. science.fits (Image with header keywords and EXTNAME=SCI)
    img_data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    prim_hdu = fits.PrimaryHDU(img_data)
    prim_hdu.header["OBJECT"] = "M31"
    prim_hdu.header["EXPTIME"] = 300.0
    prim_hdu.header["FILTER"] = "r"
    prim_hdu.header["OBSERVER"] = "Astronomer"
    prim_hdu.header["EXTNAME"] = "SCI"
    sci_ext = fits.ImageHDU(img_data, name="SCI")
    fits.HDUList([prim_hdu, sci_ext]).writeto(ws / "science.fits", overwrite=True)

    # 2. catalog.fits (Binary table)
    cat_tab = Table(
        {
            "ID": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "RA": np.array([10.1, 10.2, 10.3, 10.4, 10.5], dtype=np.float64),
            "DEC": np.array([-1.0, 0.5, 1.5, -0.2, 2.0], dtype=np.float64),
            "MAG_G": np.array([17.5, 19.2, 21.0, 18.0, 22.5], dtype=np.float32),
            "CLASS_STAR": np.array([0.95, 0.88, 0.12, 0.99, 0.05], dtype=np.float32),
            "FLUX": np.array([10.5, 23.1, 45.0, 12.0, 50.0], dtype=np.float32),
        }
    )
    cat_tab.write(ws / "catalog.fits", format="fits", overwrite=True)

    # 3. giant_mosaic.fits / survey_mosaic.fits / mosaic.fits / horsehead.fits / sdss_*.fits
    mosaic_data = np.ones((512, 512), dtype=np.float32) * 42.0
    fits.writeto(ws / "giant_mosaic.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "survey_mosaic.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "mosaic.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "horsehead.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "sdss_i.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "sdss_r.fits", mosaic_data, overwrite=True)
    fits.writeto(ws / "sdss_g.fits", mosaic_data, overwrite=True)

    # 4. observation.fits / mef_survey.fits (Multi-Extension FITS)
    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header["OBJECT"] = "NGC1234"
    sci_hdu = fits.ImageHDU(img_data, name="SCI")
    cat_hdu = fits.BinTableHDU(cat_tab, name="CATALOG")
    hdul = fits.HDUList([prim_hdu, sci_hdu, cat_hdu])
    hdul.writeto(ws / "observation.fits", overwrite=True)
    hdul.writeto(ws / "mef_survey.fits", overwrite=True)

    # 5. data/survey/*.fits directory structure
    survey_dir = ws / "data" / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        fhdr = fits.Header()
        fhdr["CLASS_ID"] = i % 2
        fhdr["CLASS"] = "GALAXY" if i % 2 == 0 else "STAR"
        fits.writeto(
            survey_dir / f"survey_{i}.fits", img_data, header=fhdr, overwrite=True
        )

    # 6. huge_catalog.fits
    cat_tab.write(ws / "huge_catalog.fits", format="fits", overwrite=True)

    return ws


def test_all_python_snippets_syntax_valid() -> None:
    """Verify that every python code block in docs/*.md is syntactically valid Python."""
    py_snippets = _extract_snippets("python")
    assert len(py_snippets) > 0, "No python snippets found in docs/"

    errors: list[str] = []
    for snippet in py_snippets:
        code = snippet.code.strip()
        try:
            ast.parse(code)
        except SyntaxError as exc:
            # If this is a function call signature like "torchfits.foo(..., *, bar=...)",
            # rewrite as "def foo(..., *, bar=...): pass" to validate function signature syntax
            cleaned = re.sub(r"^torchfits(?:\.\w+)*\.(\w+)\s*\(", r"def \1(", code)
            cleaned = re.sub(r"\)\s*->.*$", "): pass", cleaned, flags=re.DOTALL)
            if not cleaned.endswith(": pass"):
                cleaned += ": pass"
            try:
                ast.parse(cleaned)
            except SyntaxError:
                rel_path = snippet.file_path.relative_to(ROOT)
                errors.append(
                    f"{rel_path}:{snippet.line_number} SyntaxError: {exc}\nCode:\n{code}\n"
                )

    assert not errors, "Syntax errors in python doc snippets:\n" + "\n".join(errors)


def test_all_cli_snippets_valid_commands() -> None:
    """Verify that every `torchfits <subcommand> ...` in bash blocks uses valid subcommands."""
    sh_snippets = _extract_snippets("bash")
    assert len(sh_snippets) > 0, "No bash snippets found in docs/"

    parser = build_parser()
    subcommand_parsers: set[str] = set()
    for action in parser._actions:
        if action.dest == "command" and hasattr(action, "choices"):
            subcommand_parsers = set(action.choices.keys())

    errors: list[str] = []

    for snippet in sh_snippets:
        for line in snippet.code.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("torchfits "):
                try:
                    args = shlex.split(line)[1:]
                except Exception as exc:
                    rel = snippet.file_path.relative_to(ROOT)
                    errors.append(
                        f"{rel}:{snippet.line_number} shlex error on line {line!r}: {exc}"
                    )
                    continue

                if not args:
                    continue

                subcmd = args[0]
                if (
                    subcmd.startswith("$")
                    or subcmd.startswith("<")
                    or subcmd in ("--help", "-h")
                ):
                    continue

                if subcmd not in subcommand_parsers:
                    rel = snippet.file_path.relative_to(ROOT)
                    errors.append(
                        f"{rel}:{snippet.line_number} Unknown subcommand {subcmd!r} in {line!r}"
                    )
                    continue

    assert not errors, "CLI syntax errors in docs:\n" + "\n".join(errors)


def test_runnable_workflow_snippets_execute(
    mock_fits_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute runnable workflow snippets from guides, tutorials, and migration pages."""
    monkeypatch.chdir(mock_fits_workspace)

    target_docs = {
        "index.md",
        "quickstart.md",
        "python-workflows.md",
        "examples-transforms.md",
        "compatibility.md",
    }
    py_snippets = [
        s for s in _extract_snippets("python") if s.file_path.name in target_docs
    ]

    executed = 0
    for snippet in py_snippets:
        code = snippet.code.strip()

        # Skip signature definitions or non-executable fragments
        if code.startswith("torchfits.") and (
            "(" in code.splitlines()[0] and "->" in code
        ):
            continue
        if 'device="cuda"' in code and not torch.cuda.is_available():
            # Replace device="cuda" with device="cpu" if running on non-CUDA host
            code = code.replace('device="cuda"', 'device="cpu"')

        # Inject dummy process_batch / handler functions if referenced in loop snippets
        scope: dict = {"process_batch": lambda b: None, "print": lambda *a, **k: None}

        try:
            exec(code, scope)
            executed += 1
        except Exception as exc:
            rel = snippet.file_path.relative_to(ROOT)
            pytest.fail(
                f"Doc snippet in {rel}:{snippet.line_number} raised {exc.__class__.__name__}: {exc}\nCode:\n{code}"
            )

    assert executed > 0, "No runnable snippets executed"
