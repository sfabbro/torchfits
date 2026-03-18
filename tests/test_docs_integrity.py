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
        "docs/release.md",
        "docs/upstream_parity_matrix.md",
        "examples/example_healpix.py",
        "examples/example_image.py",
        "examples/example_image_cube.py",
        "examples/example_image_cutouts.py",
        "examples/example_image_dataset.py",
        "examples/example_image_mef.py",
        "examples/example_ml_pipeline.py",
        "examples/example_polars.py",
        "examples/example_spectral_sht.py",
        "examples/example_table.py",
        "examples/example_table_interop.py",
        "examples/example_table_recipes.py",
        "examples/example_wcs_transform.py",
        "benchmarks/bench_all.py",
        "benchmarks/bench_arrow_tables.py",
        "benchmarks/bench_cpp_backend.py",
        "benchmarks/bench_fast.py",
        "benchmarks/bench_fits_io.py",
        "benchmarks/bench_fitstable_io.py",
        "benchmarks/bench_healpix.py",
        "benchmarks/bench_healpix_advanced.py",
        "benchmarks/bench_ml_loader.py",
        "benchmarks/bench_pipeline_table_sphere.py",
        "benchmarks/bench_sphere_core.py",
        "benchmarks/bench_sphere_geometry.py",
        "benchmarks/bench_sphere_polygons.py",
        "benchmarks/bench_sphere_sparse.py",
        "benchmarks/bench_sphere_suite.py",
        "benchmarks/bench_sphere_spectral.py",
        "benchmarks/bench_spin_matrix.py",
        "benchmarks/bench_table.py",
        "benchmarks/bench_wcs.py",
        "benchmarks/bench_wcs_suite.py",
        "benchmarks/replays/replay_upstream_astropy_healpy.py",
        "benchmarks/replays/replay_upstream_healpy_interp_edges.py",
        "benchmarks/replays/replay_upstream_healpy_spin.py",
        "benchmarks/replays/replay_upstream_spherical_geometry_polygons.py",
        "benchmarks/replays/replay_upstream_test_functions.py",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing doc-referenced files: {missing}"
