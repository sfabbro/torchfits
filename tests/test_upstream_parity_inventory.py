from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_parity_manifest_references_existing_local_paths() -> None:
    manifest_path = ROOT / "benchmarks" / "replays" / "upstream_sources.json"
    benchmark_doc_path = ROOT / "docs" / "benchmarks.md"

    assert manifest_path.exists()
    assert benchmark_doc_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstreams = {entry["upstream"] for entry in manifest}
    assert {"fitsio", "astropy.io.fits"}.issubset(upstreams)

    for entry in manifest:
        for relpath in entry["local"].get("tests", []):
            assert (ROOT / relpath).exists(), (
                f"Missing test path from manifest: {relpath}"
            )
        for relpath in entry["local"].get("replays", []):
            assert (ROOT / relpath).exists(), (
                f"Missing replay path from manifest: {relpath}"
            )


def test_major_format_parity_matrix_has_explicit_statuses() -> None:
    parity = (ROOT / "docs" / "parity.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- major-format-coverage:start -->(.*?)<!-- major-format-coverage:end -->",
        parity,
        flags=re.DOTALL,
    )
    assert match is not None, "docs/parity.md is missing the major-format matrix"

    valid = {"Supported", "Partial", "Unsupported", "Out of Scope"}
    rows = [line for line in match.group(1).splitlines() if line.startswith("|")]
    # Header, separator, then one row per major format family.
    assert len(rows) >= 3
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        assert len(cells) == 3
        status = cells[1].removeprefix("**").removesuffix("**")
        assert status in valid, f"invalid parity status {status!r}: {cells[0]}"
