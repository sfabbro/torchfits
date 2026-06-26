from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_reference_existing_local_files() -> None:
    expected_paths = [
        "docs/api.md",
        "docs/benchmarks.md",
        "docs/changelog.md",
        "docs/examples.md",
        "docs/install.md",
        "docs/parity.md",
        "docs/roadmap.md",
        "docs/release.md",
        "examples/example_image.py",
        "examples/example_image_cube.py",
        "examples/example_image_cutouts.py",
        "examples/example_image_dataset.py",
        "examples/example_image_mef.py",
        "examples/example_polars.py",
        "examples/example_table.py",
        "examples/example_table_interop.py",
        "examples/example_table_recipes.py",
        "benchmarks/bench_all.py",
        "benchmarks/bench_arrow_tables.py",
        "benchmarks/bench_cpp_backend.py",
        "benchmarks/bench_fast.py",
        "benchmarks/bench_fits_io.py",
        "benchmarks/bench_fitstable_io.py",
        "benchmarks/bench_table.py",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing doc-referenced files: {missing}"


def test_public_docs_do_not_claim_torchfits_owns_sky_domain_features() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "MAINTENANCE.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "benchmarks.md",
        ROOT / "docs" / "changelog.md",
        ROOT / "docs" / "contributing.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "parity.md",
        ROOT / "docs" / "release.md",
        ROOT / "docs" / "roadmap.md",
    ]
    forbidden_claims = [
        "covers the same ground",
        "torchfits.get_wcs",
        "torchfits.sphere",
        "healpy-compatible",
        "spherical harmonics",
        "spherical polygons",
        "Sparse HEALPix",
        "ML Integration",
        "torchsky",
    ]

    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for claim in forbidden_claims:
            if claim in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {claim!r}")

    assert not offenders, "\n".join(offenders)
