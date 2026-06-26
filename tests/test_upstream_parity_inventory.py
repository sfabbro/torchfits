from __future__ import annotations

import json
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
