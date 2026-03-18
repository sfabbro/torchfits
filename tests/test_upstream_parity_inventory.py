from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_parity_manifest_references_existing_local_paths() -> None:
    manifest_path = ROOT / "benchmarks" / "replays" / "upstream_sources.json"
    matrix_path = ROOT / "docs" / "upstream_parity_matrix.md"

    assert manifest_path.exists()
    assert matrix_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        for relpath in entry["local"].get("tests", []):
            assert (ROOT / relpath).exists(), (
                f"Missing test path from manifest: {relpath}"
            )
        for relpath in entry["local"].get("replays", []):
            assert (ROOT / relpath).exists(), (
                f"Missing replay path from manifest: {relpath}"
            )
