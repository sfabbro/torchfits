from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _collect_nav_paths(nav: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in nav:
        if isinstance(item, str):
            paths.append(item)
            continue
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    if not value.startswith("http"):
                        paths.append(value)
                else:
                    paths.extend(_collect_nav_paths(value))
    return paths


def test_docs_reference_existing_local_files() -> None:
    expected_paths = [
        "docs/index.md",
        "zensical.toml",
        "docs/api.md",
        "docs/benchmarks.md",
        "docs/changelog.md",
        "docs/examples.md",
        "docs/install.md",
        "docs/parity.md",
        "docs/roadmap.md",
        "docs/migration_fitsio.md",
        "docs/migration_astropy.md",
        "examples/example_image.py",
        "examples/example_image_cube.py",
        "examples/example_image_cutouts.py",
        "examples/example_image_dataset.py",
        "examples/example_data_catalogs.py",
        "examples/example_transforms.py",
        "examples/example_image_mef.py",
        "examples/example_polars.py",
        "examples/example_table.py",
        "examples/example_table_interop.py",
        "examples/example_table_recipes.py",
        "examples/example_time_series.py",
        "benchmarks/bench_all.py",
        "benchmarks/bench_arrow_tables.py",
        "benchmarks/bench_cpp_backend.py",
        "benchmarks/bench_fits_io.py",
        "benchmarks/bench_fitstable_io.py",
        "benchmarks/bench_gpu_transports.py",
        "benchmarks/bench_table.py",
        "benchmarks/bench_contract.py",
        "scripts/launch_canfar_gpu_bench.sh",
        "scripts/canfar_gpu_bench_incontainer.sh",
        "scripts/canfar_gpu_bench_remote.sh",
        "scripts/selfcheck_canfar_launcher.sh",
        "scripts/gpu-bootstrap.sh",
        "scripts/ci_local.sh",
        "scripts/patch_canfar_exhaustive_docs.sh",
        "scripts/publish_canfar_bench_vos.sh",
        "scripts/fetch_canfar_bench_vos.sh",
        "scripts/import_canfar_bench_artifacts.py",
        "scripts/run_exhaustive_bench_and_patch_docs.sh",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing doc-referenced files: {missing}"


def test_public_docs_do_not_claim_torchfits_owns_sky_domain_features() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "benchmarks.md",
        ROOT / "docs" / "changelog.md",
        ROOT / "docs" / "contributing.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "index.md",
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


def test_public_docs_do_not_reference_missing_root_cache_aliases() -> None:
    """Cache tuning lives on torchfits.cache; root exposes I/O cache helpers only."""
    docs = [
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "install.md",
    ]
    forbidden_root_calls = [
        "torchfits.configure_for_environment(",
        "torchfits.get_cache_stats(",
        "torchfits.clear_cache(",
    ]

    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for call in forbidden_root_calls:
            if call in text:
                offenders.append(f"{path.relative_to(ROOT)} references {call!r}")

    assert not offenders, "\n".join(offenders)


def test_docs_do_not_advertise_unimplemented_worker_handle_env() -> None:
    """TORCHFITS_WORKER_HANDLE is roadmap-only; docs must not present it as settable."""
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "install.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "migration_astropy.md",
        ROOT / "docs" / "migration_fitsio.md",
    ]
    # Allowed: honesty notes that the feature does not exist.
    # Forbidden: documentation that tells users to set / rely on it.
    forbidden = [
        "TORCHFITS_WORKER_HANDLE=1",
        "setting the environment variable `TORCHFITS_WORKER_HANDLE",
        "set the environment variable `TORCHFITS_WORKER_HANDLE",
        "When the environment variable `TORCHFITS_WORKER_HANDLE",
    ]
    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for claim in forbidden:
            if claim in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {claim!r}")
    assert not offenders, "\n".join(offenders)


def test_api_md_env_var_table_matches_source() -> None:
    """Only document TORCHFITS_* env vars that exist in the tree (table rows)."""
    docs_text = "\n".join(
        (ROOT / "docs" / name).read_text(encoding="utf-8")
        for name in ("api.md", "architecture.md", "api-tables.md", "api-core-io.md")
        if (ROOT / "docs" / name).exists()
    )
    # Rows in Environment variables Markdown tables: | `TORCHFITS_...` | ...
    documented = set(re.findall(r"\|\s*`?(TORCHFITS_[A-Z0-9_]+)`?\s*\|", docs_text))
    source_envs: set[str] = set()
    # examples/ is included because docs (e.g. TORCHFITS_EXAMPLE_FAST) may
    # legitimately document example-harness-only vars alongside src/ ones.
    for base in (ROOT / "src", ROOT / "examples"):
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".cpp", ".h", ".hpp", ".cc", ".cu"}:
                continue
            if not path.is_file():
                continue
            source_envs.update(
                re.findall(
                    r"TORCHFITS_[A-Z0-9_]+",
                    path.read_text(encoding="utf-8", errors="ignore"),
                )
            )
    missing = sorted(documented - source_envs)
    assert not missing, f"api.md env table documents missing vars: {missing}"


def test_architecture_md_env_tables_cover_every_source_getenv() -> None:
    """Every TORCHFITS_* var actually read via getenv/os.environ in
    src/torchfits (Python + cpp_src) must appear in docs/architecture.md's
    env tables. Complements test_api_md_env_var_table_matches_source, which
    only checks the reverse (no phantom vars documented)."""
    src_root = ROOT / "src" / "torchfits"
    py_pattern = re.compile(
        r"os\.(?:environ\.get|getenv)\(\s*[\"'](TORCHFITS_[A-Z0-9_]+)[\"']"
    )
    cpp_pattern = re.compile(
        r"(?:std::getenv|env_flag_default_true|env_nonnegative_int)\(\s*\"(TORCHFITS_[A-Z0-9_]+)\""
    )
    found: set[str] = set()
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".py":
            found.update(py_pattern.findall(text))
        elif path.suffix in {".h", ".hpp", ".cc", ".cpp", ".cu"}:
            found.update(cpp_pattern.findall(text))

    # TORCHFITS_TORCH_ABI is a compile-time CMake macro, not a getenv() read.
    found.discard("TORCHFITS_TORCH_ABI")

    arch_text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"\|\s*`(TORCHFITS_[A-Z0-9_]+)`\s*\|", arch_text))

    missing = sorted(found - documented)
    assert not missing, (
        f"docs/architecture.md env tables are missing vars read by src/torchfits: {missing}"
    )


def test_api_md_core_io_signatures_match_live() -> None:
    """Guard against invented parameters on the most-copied Core I/O signatures."""
    import torchfits

    # Persona rewrite splits the hub: signatures live in api-core-io.md.
    api = (ROOT / "docs" / "api-core-io.md").read_text(encoding="utf-8")

    def _section(name: str) -> str:
        # Headings look like: ## `read_tensor()`
        pattern = rf"## `{re.escape(name)}\(\)`\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, api, flags=re.S)
        assert match, f"missing section for {name}"
        return match.group(1)

    def _first_call_sig(section: str) -> str:
        match = re.search(
            r"```python\n(torchfits\.\w+\(.*?)\)\n```", section, flags=re.S
        )
        assert match, "missing python call-signature fence"
        return match.group(1)

    read_tensor_sig = _first_call_sig(_section("read_tensor"))
    assert "scale_on_device" not in read_tensor_sig, (
        "read_tensor must not document scale_on_device (that belongs to read(..., scale_on_device=) / kwargs)"
    )
    assert "mmap" in read_tensor_sig

    subset_section = _section("read_subset")
    subset_sig = _first_call_sig(subset_section)
    assert "device" not in subset_sig
    assert "x1" in subset_sig

    # Table streaming lives in api-tables.md (root stream_table removed).
    tables_api = (ROOT / "docs" / "api-tables.md").read_text(encoding="utf-8")
    scan_match = re.search(
        r"## `table\.scan_torch\(\)`\n(.*?)(?=\n## |\Z)", tables_api, flags=re.S
    )
    assert scan_match, "missing section for table.scan_torch"
    stream_section = scan_match.group(1)
    assert "batch_size" in stream_section
    assert (
        "dict[str, torch.Tensor]" in stream_section
        or "dict[str, Tensor]" in stream_section
        or "Yields" in stream_section
    )

    read_section = _section("read")
    assert "table" in read_section.lower() or "tensor" in read_section.lower()

    # Live sanity: key helpers remain importable
    assert not hasattr(torchfits, "read_table")
    assert callable(torchfits.table.read_torch)
    assert callable(torchfits.read_subset)


def test_docs_index_has_no_i_want_to() -> None:
    text = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "I want to" not in text
    assert "#i-want-to" not in text


def test_docs_examples_reference_existing_scripts() -> None:
    text = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    refs = re.findall(r"\]\(published-examples/([^)#]+)\)", text)
    # Directory index / generated README are not repo examples/
    refs = [name for name in refs if name not in {"README.md", ""}]
    missing_scripts = [name for name in refs if not (ROOT / "examples" / name).exists()]
    assert not missing_scripts, (
        f"docs/examples.md references missing scripts: {missing_scripts}"
    )
    transforms = (ROOT / "docs" / "examples-transforms.md").read_text(encoding="utf-8")
    ml_page = (ROOT / "docs" / "examples-ml.md").read_text(encoding="utf-8")
    for label, page in (
        ("examples-transforms.md", transforms),
        ("examples-ml.md", ml_page),
    ):
        href_refs = re.findall(r"\]\(published-examples/([^)#]+)\)", page)
        missing_href = [
            name for name in href_refs if not (ROOT / "examples" / name).exists()
        ]
        assert not missing_href, (
            f"docs/{label} references missing scripts: {missing_href}"
        )
    gallery_pages = text + "\n" + transforms + "\n" + ml_page
    gallery_refs = re.findall(r"\]\(assets/gallery/([^)]+)\)", gallery_pages)
    missing_png = [
        name
        for name in gallery_refs
        if not (ROOT / "docs" / "assets" / "gallery" / name).exists()
    ]
    assert not missing_png, f"missing gallery assets: {missing_png}"

    # RGB science figures must not be near-black (regression for lupton /peak crush).
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return
    for name in (
        "lupton_rgb_sdss.png",
        "megapipe_cutout_collage.png",
        "ml_gz_class_grid.png",
    ):
        path = ROOT / "docs" / "assets" / "gallery" / name
        if not path.is_file():
            continue
        arr = np.asarray(Image.open(path), dtype=np.float64)
        luma = arr[..., :3].mean(axis=-1) if arr.ndim == 3 else arr
        assert float(luma.mean()) >= 8.0, f"{name} mean too dark: {luma.mean()}"
        assert float(np.percentile(luma, 90)) >= 25.0, (
            f"{name} p90 too dark: {np.percentile(luma, 90)}"
        )


def test_api_tables_documents_ignored_cache_kwargs() -> None:
    """handle_cache_capacity must not look like an active knob without ignored/deprecated."""
    tables_api = (ROOT / "docs" / "api-tables.md").read_text(encoding="utf-8")
    assert "handle_cache_capacity" in tables_api
    # Signature lines may list the kwarg before the prose note; require an
    # ignored/deprecated sentence that names the knob within a short span.
    note = re.search(
        r"handle_cache_capacity.{0,120}?(ignored|deprecated)"
        r"|(ignored|deprecated).{0,120}?handle_cache_capacity",
        tables_api,
        flags=re.S | re.I,
    )
    assert note, "api-tables.md must mark handle_cache_capacity as ignored/deprecated"


def test_api_tables_where_strategy_mentions_mask_or_project() -> None:
    """where= strategy text must not silently regress to C++-only story."""
    tables_api = (ROOT / "docs" / "api-tables.md").read_text(encoding="utf-8")
    where_section = re.search(
        r"## Predicate Pushdown\n(.*?)(?=\n## |\Z)", tables_api, flags=re.S
    )
    assert where_section, "missing Predicate Pushdown section in api-tables.md"
    body = where_section.group(1).lower()
    assert "mask" in body or "project" in body, (
        "api-tables Predicate Pushdown must mention mask/project strategy"
    )
    # read_torch path must not claim C++-only filtering for most sizes.
    assert "filtering happens in c++ for most table sizes" not in body


def test_sync_docs_examples_script_copies_tree() -> None:
    import subprocess

    dest = ROOT / "docs" / "published-examples"
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "sync_docs_examples.sh")],
        check=True,
        cwd=ROOT,
    )
    assert (dest / "gallery_images.py").is_file()
    assert (dest / "cli" / "imstat_imarith.sh").is_file()
    assert (dest / "README.md").is_file()


def test_zensical_config_targets_existing_docs() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    project = config["project"]
    site_url = project["site_url"]
    assert site_url == "https://astroai.github.io/torchfits/"
    assert project["repo_url"] == "https://github.com/astroai/torchfits"

    nav_paths = _collect_nav_paths(project["nav"])
    missing = [path for path in nav_paths if not (ROOT / "docs" / path).exists()]
    assert not missing, f"zensical.toml nav references missing docs: {missing}"

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'Documentation = "{site_url.rstrip("/")}/"' in pyproject, (
        "pyproject.toml Documentation URL must match zensical site_url"
    )
