from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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
        "examples/example_cube.py",
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
        "benchmarks/bench_import_boundary.py",
        "benchmarks/bench_gpu_transports.py",
        "benchmarks/bench_table.py",
        "benchmarks/bench_contract.py",
        "scripts/launch_canfar_gpu_bench.sh",
        "scripts/canfar_gpu_bench_incontainer.sh",
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


def test_public_docs_do_not_claim_native_gpu_decode() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "install.md",
    ]
    forbidden_claims = [
        "automatically enable GPU acceleration",
        "unlocks GPU acceleration automatically",
        "directly onto CUDA GPUs",
        "Native GPU decode",
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


def test_cpp_env_flags_go_through_the_canonical_helpers() -> None:
    """No hand-rolled ``getenv`` parsing of a TORCHFITS_* flag in cpp_src.

    ``internal_utils.h`` provides the canonical readers --
    ``env_flag_default_true`` / ``env_flag_default_false`` for booleans and
    ``env_nonnegative_int`` for numbers. A local parser is easy to add and hard
    to notice, and the vocabularies genuinely disagree. Measured on real
    variables: ``table_types.h`` used to test only the first character of
    ``TORCHFITS_TABLE_BUFFERED``, reading ``off`` as *enabled* while
    ``env_flag_default_true`` reads it as disabled; ``table_reader.h`` did the
    same for ``TORCHFITS_VLA_HEAP_PREAD``, reading ``on`` as *disabled* while
    the opt-in helper reads it as enabled. Opposite answers, no error either
    way.

    A *numeric* knob is exempt -- it legitimately needs its own parse, and
    ``core/parallel.cpp``'s ``TORCHFITS_NUM_THREADS`` (via ``std::stoi``) is the
    one legitimate case. A boolean must go through a helper.
    """
    src_root = ROOT / "src" / "torchfits" / "cpp_src"
    numeric_parse = re.compile(r"\b(?:stoll|stoi|atoll|atoi|strtoll|strtol)\b")
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".h",
            ".hpp",
            ".cc",
            ".cpp",
            ".cu",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            match = re.search(r"std::getenv\(\s*\"(TORCHFITS_[A-Z0-9_]+)\"", line)
            if not match:
                continue
            # Look a few lines ahead: a numeric read parses immediately, a
            # hand-rolled boolean keeps comparing the value against literals.
            window = "\n".join(lines[i : i + 6])
            if numeric_parse.search(window):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{i + 1}: {match.group(1)}")
    assert not offenders, (
        "these C++ flags parse getenv by hand instead of using "
        "internal_utils.h:\n  " + "\n  ".join(offenders)
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
        "rgb_sky_collage.png",
        "rgb_vs_lupton_dwarf.png",
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
        r"## (?:Predicate Pushdown|Row filters \(``where=``\)|Row filters \(`where=`\))\n(.*?)(?=\n## |\Z)",
        tables_api,
        flags=re.S,
    )
    assert where_section, "missing where= section in api-tables.md"
    body = where_section.group(1).lower()
    assert "mask" in body or "project" in body, (
        "api-tables where= section must mention mask/project strategy"
    )
    assert "filtering happens in c++ for most table sizes" not in body
    # read_torch dialect is narrower than table.read — docs must say so.
    assert "simple" in body and ("or" in body or "in" in body), (
        "api-tables must document that read_torch where= is a simple dialect"
    )


def test_api_tables_read_torch_where_not_full_dialect() -> None:
    """Guard against claiming read_torch accepts the full where= dialect."""
    tables_api = (ROOT / "docs" / "api-tables.md").read_text(encoding="utf-8")
    rt = re.search(
        r"## `table\.read_torch\(\)`\n(.*?)(?=\n## |\Z)", tables_api, flags=re.S
    )
    assert rt, "missing table.read_torch section"
    body = rt.group(1).lower()
    assert "same predicate dialect" not in body
    assert "same simple numeric predicate dialect" not in body
    assert "simple" in body or "valueerror" in body


def test_every_published_example_is_linked_from_a_docs_page() -> None:
    """The reverse direction: examples/ -> docs, not just docs -> examples/.

    ``scripts/sync_docs_examples.sh`` copies ``examples/*.py`` and
    ``examples/cli/*`` to ``docs/published-examples/`` at build time, and
    zensical's nav is an explicit list that does not include that directory, so
    a published script is reachable only from a link on some page.
    ``test_docs_examples_reference_existing_scripts`` only checked the other
    direction, which left eight runnable examples published and unlinked --
    including the two cfitsio/CCfits cookbooks and the identity-stress harness,
    the repo's largest correctness demonstrations, at 917 lines between them.
    One of them (``desi_shaped_spectrum.py``) was listed on examples.md under
    "Out of gallery ... not part of the published gallery", which the sync
    script contradicted.

    The published set is derived from the same globs the script uses, so the two
    cannot drift.
    """
    published = {
        path.relative_to(ROOT / "examples").as_posix()
        for pattern in ("*.py", "cli/*")
        for path in (ROOT / "examples").glob(pattern)
        if path.is_file()
    }
    # Underscore-prefixed modules are shared helpers, not standalone examples.
    orphans_expected = {n for n in published if Path(n).name.startswith("_")}

    linked: set[str] = set()
    for page in sorted((ROOT / "docs").glob("*.md")):
        linked |= set(
            re.findall(
                r"\]\(published-examples/([^)#]+)\)", page.read_text(encoding="utf-8")
            )
        )

    orphans = sorted(
        name
        for name in published - linked
        if not Path(name).name.startswith("_") and name != "README.md"
    )
    assert not orphans, (
        "these examples are published to docs/published-examples/ but linked "
        f"from no page: {orphans} (helpers exempt: {sorted(orphans_expected)})"
    )


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


def test_untracked_client_trees_do_not_shadow_tracked_copies() -> None:
    """`.agents/` and `.codex/` may not hold a divergent copy of tracked files.

    `.cursor/` is the canonical, tracked home for the agent skills
    (`AGENTS.md:47`, `docs/release.md`, `docs/changelog.md` and the skills
    themselves all reference it; nothing tracked references `.agents/` or
    `.codex/`). An untracked copy of the same file is worse than no copy at all,
    because which behaviour a client gets then depends on which tree it happens
    to load. Both trees started byte-identical to the tracked originals; the
    review fixes in `.cursor/skills/release-api-freeze-review/…` and
    `.cursor/hooks/harness-stop.sh` left the copies behind, so the 41-of-45
    inventory bug and the state-losing Stop hook are still live under the same
    names.

    This is a forcing function, not a cleanup. It stays red until the duplicates
    are deleted or gitignored -- deliberately, because a skip or a warning is
    the quiet failure this project has been fixing all along. Deleting the
    owner's untracked files is not a review's call, so the test names the exact
    commands that satisfy it instead of running them.
    """
    mirrors = (
        (ROOT / ".agents" / "skills", ROOT / ".cursor" / "skills"),
        (ROOT / ".codex" / "hooks", ROOT / ".cursor" / "hooks"),
    )
    problems: list[str] = []
    for shadow_root, canonical in mirrors:
        if not shadow_root.is_dir():
            continue
        for shadow in sorted(shadow_root.rglob("*")):
            if not shadow.is_file():
                continue
            relative = shadow.relative_to(shadow_root)
            if "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue  # interpreter cache, not content
            tracked = canonical / relative
            label = f"{shadow_root.relative_to(ROOT)}/{relative}"
            if not tracked.is_file():
                problems.append(
                    f"{label} has no tracked counterpart at "
                    f"{tracked.relative_to(ROOT)} -- commit it there or remove it"
                )
            elif shadow.read_bytes() != tracked.read_bytes():
                problems.append(
                    f"{label} differs from the canonical tracked "
                    f"{tracked.relative_to(ROOT)}; the duplicate carries stale "
                    f"behaviour under the same name"
                )
    assert not problems, (
        "untracked client-tree copies diverge from the tracked tree:\n  "
        + "\n  ".join(problems)
        + "\n\nResolve with one of:\n"
        "  rm -rf .agents .codex            # .cursor/ is canonical\n"
        "  printf '\\n.agents/\\n.codex/\\n' >> .gitignore   # keep them local, untracked\n"
        "  cp -a .cursor/skills/. .agents/skills/ && \\\n"
        "    cp -a .cursor/hooks/. .codex/hooks/          # re-sync the copies"
    )


def test_harness_verify_full_tier_type_checks() -> None:
    """No verify tier may be the *only* one that skips the type checker.

    `.cursor/harness/config.json` names three tiers and
    `.cursor/harness/playbook.md` (bullet 2 of 12, so injected into every
    session by the SessionStart hook) described them as
    "verify_fast / verify / verify_full only when human opts in". Measured, the
    three commands were not nested at all:

    | tier | task | mypy | tests |
    |---|---|---|---|
    | verify_fast | preflight-push | yes | none |
    | verify | ci-local | **no** | 20 of 139 files |
    | verify_full | pre-commit | **no** | none |

    So the tier called *full* was the only one that never type-checked and the
    only one that ran no test at all, and the `ci-parity` bullet claimed
    "local ci-local = preflight-push + pixi test" when ci-local calls neither.
    A human opting into `verify_full` got the weakest gate in the repo under the
    most reassuring name.

    `verify_full` is now `ci-local` + `pre-commit`, which makes the three tiers
    genuinely nested. This asserts the nesting holds: every tier's commands must
    reach the type checker, directly or through preflight-push.
    """
    cfg = json.loads(
        (ROOT / ".cursor" / "harness" / "config.json").read_text(encoding="utf-8")
    )
    tasks = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    tiers = {k: v for k, v in cfg.items() if k.startswith("verify")}
    assert set(tiers) == {"verify_fast", "verify", "verify_full"}, (
        f"unexpected verify tiers in config.json: {sorted(tiers)}"
    )

    def type_checks(commands: list[str]) -> bool:
        """True when any command in the tier runs mypy, directly or via a task."""
        for command in commands:
            if "mypy" in command:
                return True
            for task in re.findall(r"pixi run ([a-z0-9-]+)", command):
                if "mypy" in tasks.get(task, ""):
                    return True
        return False

    silent = [tier for tier, cmds in tiers.items() if not type_checks(cmds)]
    assert not silent, (
        f"verify tier(s) {silent} never run mypy, so the tier that a human opts "
        f"into for maximum assurance may be the only one that never "
        f"type-checks. Tiers: {tiers}"
    )

    # Nesting is what makes the names mean anything: a wider tier must run every
    # command a narrower one does.
    def flatten(commands: list[str]) -> set[str]:
        out: set[str] = set()
        for command in commands:
            out.update(re.findall(r"pixi run ([a-z0-9-]+)", command))
        return out

    assert flatten(tiers["verify"]) <= flatten(tiers["verify_full"]), (
        "verify_full must be at least as thorough as verify"
    )
    assert flatten(tiers["verify_fast"]) <= flatten(tiers["verify"]), (
        "verify must be at least as thorough as verify_fast"
    )


def test_harness_task_references_resolve() -> None:
    """Every `pixi run <task>` in a tracked `.cursor/` file must be a real task.

    The injected playbook and the release skill both drive agents by task name,
    and a task that has been renamed or deleted turns into a command that fails
    at the moment an agent needs it. `pixi run` falls back to executing a same-named
    binary from the environment when no task matches, so `pixi run pytest …` works
    by accident -- which is exactly why the reference is worth pinning rather
    than eyeballing.
    """
    tasks = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    listed = subprocess.run(
        ["git", "ls-files", ".cursor"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        pytest.skip(f"not a git checkout: {listed.stderr.strip()}")
    missing: list[str] = []
    for relative in listed.stdout.split():
        path = ROOT / relative
        if path.suffix not in {".md", ".json", ".sh"}:
            continue
        for match in re.finditer(
            r"pixi run (?:-e\s+\S+\s+)?([a-z][a-z0-9-]*(?:-[a-z0-9]+)+)",
            path.read_text(encoding="utf-8", errors="ignore"),
        ):
            name = match.group(1)
            if name not in tasks:
                missing.append(f"{relative}: pixi run {name}")
    assert not missing, "these task references do not exist in pixi.toml: " + "; ".join(
        missing
    )


def test_benchmark_run_ids_match_published_assets() -> None:
    """Every run ID the benchmarks page cites must have published CSVs.

    The API-freeze skill's Phase 5 used to send a reviewer after a
    `BENCH_SNAPSHOT` marker in `docs/benchmarks.md` and a matching
    `benchmarks_results/<run-id>/` directory. The marker exists nowhere in the
    repository, `benchmarks_results/` is gitignored local scratch, and the README
    has no performance table at all -- so the phase could only ever end in "the
    thing you were asked to check is not here". The published CSVs are mirrored
    under `docs/assets/bench/<run-id>/` (docs/benchmarks.md:174), which is
    tracked, so the claim is checkable and is checked here.
    """
    doc = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assets = ROOT / "docs" / "assets" / "bench"
    assert assets.is_dir(), f"published benchmark assets are missing: {assets}"
    cited = set(re.findall(r"(20\d{6}_\d{6})", doc))
    assert cited, "no run IDs found in docs/benchmarks.md; the check is vacuous"
    published = [p.name for p in assets.iterdir() if p.is_dir()]

    missing = sorted(i for i in cited if not any(i in name for name in published))
    assert not missing, (
        f"docs/benchmarks.md cites run IDs with nothing published under "
        f"docs/assets/bench/: {missing}"
    )
    orphan = sorted(
        name for name in published if not any(name.endswith(i) for i in cited)
    )
    assert not orphan, (
        f"docs/assets/bench/ holds run data the benchmarks page never cites: "
        f"{orphan} -- either cite them or drop the directories"
    )


def test_release_skill_inventory_sees_every_public_export() -> None:
    """The API-freeze inventory tool must see every name in `__all__`.

    The tool is what a release review runs to catch an export that never made it
    into `docs/api.md`, and it read `__all__` with a text split that stopped at
    the `)` closing `tuple([` -- before the `*_NAMESPACES` entry. It therefore
    compared 41 of the 45 real exports and still printed "All __all__ symbols
    appear in docs/api.md"; a namespace added to the package was invisible with
    no error anywhere. Nothing ran the tool in CI, so nothing noticed.

    Asserting the tool's export set equals the live package's is the cheap form
    of that guard: any future parse that reads a subset fails here first.
    """
    import importlib.util

    import torchfits

    tool = ROOT / ".cursor" / "skills" / "release-api-freeze-review"
    script = tool / "scripts" / "inventory_public_api.py"
    assert script.is_file(), (
        f"the API-freeze review skill lost its inventory tool: {script}"
    )
    spec = importlib.util.spec_from_file_location("tf_inventory_public_api", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading by path would drop a __pycache__ into the skill tree on every
    # run, which then shows up as a second "divergent copy" in the shadow test
    # above. The tool is loaded read-only here, so nothing needs caching.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous

    seen = set(module.load_all_from_init())
    missing = set(torchfits.__all__) - seen
    assert not missing, (
        f"inventory_public_api.py does not see these public exports: "
        f"{sorted(missing)} -- it would report them as absent from the "
        f"package entirely, so it can never flag them as undocumented"
    )
    # The tool must not invent names either: a phantom export shows up in its
    # "weak/absent in docs" list and sends a reviewer after a symbol that
    # cannot be imported.
    assert not seen - set(torchfits.__all__), (
        f"inventory_public_api.py reports non-exports: "
        f"{sorted(seen - set(torchfits.__all__))}"
    )


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


def test_documented_api_members_exist() -> None:
    """Every symbol named in the API docs must exist on the live package."""
    import torchfits
    import torchfits.cache as cache
    import torchfits.data as data
    import torchfits.table as table
    import torchfits.transforms as transforms
    import torchfits.where as where

    expected = {
        torchfits: [
            "read",
            "write",
            "open",
            "read_header",
            "read_colnames",
            "read_extname",
            "read_hdu_type",
            "read_keys",
            "read_nrows",
            "read_num_hdus",
            "read_shape",
            "read_table_info",
            "read_tensor",
            "read_hdus",
            "read_subset",
            "open_subset_reader",
            "open_table_reader",
            "read_batch",
            "read_batch_info",
            "get_cache_performance",
            "clear_file_cache",
            "verify_checksums",
            "insert_hdu",
            "replace_hdu",
            "delete_hdu",
            "write_checksums",
            "write_tensor",
            "to_pandas",
            "to_arrow",
            "to_polars",
            "to_astropy",
            "Header",
            "Card",
            "HDUList",
            "TensorHDU",
            "TableHDU",
            "TableHDURef",
        ],
        table: [
            "read",
            "read_arrow",
            "read_torch",
            "scan",
            "scan_torch",
            "read_polars",
            "scan_polars",
            "to_polars",
            "to_duckdb",
            "duckdb_query",
            "reader",
            "write",
            "append_rows",
            "insert_rows",
            "update_rows",
            "delete_rows",
            "insert_column",
            "replace_column",
            "rename_columns",
            "drop_columns",
            "schema",
            "dataset",
            "scanner",
            "write_parquet",
            "clear_cache",
            "TABLE_BACKENDS",
        ],
        data: [
            "CutoutSpec",
            "FitsTensorDataset",
            "FitsTensorIterableDataset",
            "FitsImageDataset",
            "FitsImageIterableDataset",
            "FitsCubeDataset",
            "FitsCubeIterableDataset",
            "FitsSpectrumDataset",
            "FitsSpectrumIterableDataset",
            "FitsTableDataset",
            "FitsTableIterableDataset",
            "FitsCutoutDataset",
            "FitsStagedCutoutIterableDataset",
            "make_loader",
            "fits_collate_fn",
        ],
        cache: [
            "configure_for_environment",
            "get_cache_stats",
            "clear_cache",
            "optimize_for_dataset",
        ],
        where: [
            "evaluate_where",
            "parse_where_expression",
            "parse_where_literal",
            "tokenize_where_expression",
            "normalize_where_syntax",
            "where_columns_from_ast",
        ],
        transforms: [
            "ArcsinhStretch",
            "LogStretch",
            "SqrtStretch",
            "ZScaleNormalize",
            "RobustNormalize",
            "BackgroundSubtract",
            "PercentileClipNormalize",
            "MinMaxNormalize",
            "GlobalScalarNorm",
            "SigmaClip",
            "AsymmetricSigmaClip",
            "FITSHeaderScale",
            "FITSScaleColumns",
            "TNullToNan",
            "FITSHeaderNormalize",
            "Compose",
            "FITSTransform",
            "AsModule",
            "as_module",
            "lupton_rgb",
            "rgb",
            "safe_arcsinh",
            "safe_log",
            "estimate_background",
            "zscale_limits",
        ],
    }
    missing = [
        f"{mod.__name__}.{name}"
        for mod, names in expected.items()
        for name in names
        if not hasattr(mod, name)
    ]
    assert not missing, f"docs reference missing API members: {missing}"


def test_cli_docs_subcommands_match_parser() -> None:
    """Commands documented in cli.md / cli-recipes.md must match the parser."""
    import argparse
    import re

    from torchfits.cli.main import build_parser

    parser = build_parser()
    sub_names: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub_names = set(action.choices)
    assert sub_names, "CLI parser exposes no subcommands"

    cli_docs = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    recipes = (ROOT / "docs" / "cli-recipes.md").read_text(encoding="utf-8")
    text = cli_docs + "\n" + recipes
    # Inline code spans `torchfits <cmd> …`; the trailing (?![-a-z0-9_]) stops
    # `torchfits write_checksums(...)` from capturing `write`.
    documented = set(re.findall(r"`torchfits\s+([a-z][a-z0-9-]*)(?![a-z0-9_-])", text))
    # cli.md's subcommand tables list torchfits commands in the first column;
    # compound cells like `compress` / `decompress` carry multiple commands.
    # cli-recipes.md's Familiar-tool map lists *classic* tools first (imstat,
    # imcopy, …), so restrict the row parsing to cli.md. Flag rows (`-e`) and
    # exit-code rows (digits) start with non-letters and are skipped by [a-z].
    for line in cli_docs.splitlines():
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells:
                documented.update(re.findall(r"`([a-z][a-z0-9-]*)`", cells[0]))

    unknown = documented - sub_names
    assert not unknown, f"docs reference unknown CLI subcommands: {sorted(unknown)}"
    undocumented = sub_names - documented
    assert not undocumented, (
        f"CLI subcommands missing from docs: {sorted(undocumented)}"
    )


def test_owed_behavior_notes_match_the_implementation() -> None:
    """Compatibility, architecture, tables, and cutout docs state the 1.2 contracts."""
    compat = (ROOT / "docs" / "compatibility.md").read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    tables = (ROOT / "docs" / "api-tables.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")

    assert "No warning is emitted" not in compat
    assert "undocumented env knobs" not in compat
    assert (
        "MPS does not support float64; downcasting to float32 (precision loss)"
        in compat
    )
    assert (
        "MPS does not support complex128; downcasting to complex64 (precision loss)"
        in compat
    )
    assert "KMP_DUPLICATE_LIB_OK" in compat
    assert "residual TOCTOU" in compat
    assert "float32" in compat and "float64" in compat

    numpy_at = arch.find("`read_full_numpy`")
    assert numpy_at != -1
    assert "bitwise" in arch[numpy_at : numpy_at + 600]

    assert "three-valued" in tables
    assert "NaN" in tables
    assert "QuantizeError" in tables

    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert "TableHDU.head(n)" in api
    assert "n >= 0" in api
    assert "TableHDURef.head(n)" in api

    cutout = cli.split("### `cutout`", 1)[1].split("### `", 1)[0]
    assert "not shifted" in cutout
    setkey = cli.split("### `setkey`", 1)[1].split("### `", 1)[0]
    assert "--comment" not in setkey

    import json

    lanes = json.loads(
        (ROOT / "scripts" / "torch_lanes.json").read_text(encoding="utf-8")
    )
    current = lanes["2.13"]["torchfits_version"]
    release = (ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    assert f"**{current}**" in release
    assert "1.0.0rc5" not in release


def _tracked_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        pytest.skip(f"not a git checkout: {listed.stderr.strip()}")
    return listed.stdout.split()


def test_documented_vendor_commands_pass_the_versions_file() -> None:
    """Every `./extern/vendor.sh` a reader is told to run must actually work.

    Measured 2026-09-27: `bash extern/vendor.sh` with no `--cfitsio-version`
    exits 1 with `Failed to resolve CFITSIO version from: ` — an empty spec and
    no hint about the required argument. That bare form is what three docs and
    a CMake error message told people to run, including the "build from source"
    recipe in docs/install.md and the troubleshooting entry for a failing
    vendor.sh. Every working caller (all five workflows, all eight scripts)
    passes `extern/VERSIONS.txt`; the four that did not were the four that
    could never have worked.
    """
    offenders: list[str] = []
    # Instruction surfaces only: the files a reader copies commands out of.
    # Tests and historical harness reports quote these commands in prose and
    # docstrings, where `--cfitsio-version` need not be on the same line.
    surfaces = (
        ".github/",
        "benchmarks/",
        "docs/",
        "examples/",
        "scripts/",
        "src/",
    )
    for relative in _tracked_files():
        if not relative.startswith(surfaces) and "/" in relative:
            continue
        if relative.startswith(".cursor/"):
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "./extern/vendor.sh" not in line:
                continue
            if "--cfitsio-version" not in line:
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, (
        "these lines run vendor.sh with no --cfitsio-version, which exits 1 "
        "with an empty-spec error: " + "; ".join(offenders)
    )


def test_bench_report_workflow_publishes_the_csvs_it_cites() -> None:
    """The automated benchmark PR must not be born red.

    `bench-report.yml` runs bench-all under a `ci_<timestamp>` run id, feeds it
    to `scripts/patch_bench_docs.py`, which writes that run id into the
    benchmarks page, then opens a PR that stages `docs/benchmarks.md` only --
    the CSVs go to a workflow artifact, which expires. So the page cites a run
    with nothing published, and `test_benchmark_run_ids_match_published_assets`
    (added in the .cursor/ review) fails on the very PR the workflow exists to
    produce, in the `docs-contract` job that runs on every PR. The workflow can
    only ever open an unmergeable PR.

    The fix is the documented policy rather than a test relaxation: commit the
    CSVs under `docs/assets/bench/<run-id>/`, which is what the page already
    says happens ("published with each release and mirrored under
    docs/assets/bench/<run-id>/") and what the seven tracked run directories are.
    """
    workflow = (ROOT / ".github" / "workflows" / "bench-report.yml").read_text(
        encoding="utf-8"
    )
    staged = re.findall(r"git add ([^\n]+)", workflow)
    assert staged, "bench-report.yml stages nothing; the harness below is vacuous"
    publishes = any("docs/assets/bench" in entry for entry in staged)
    assert publishes, (
        "bench-report.yml writes a run id into docs/benchmarks.md but stages no "
        "docs/assets/bench/<run-id>/ CSVs for it, so the PR it opens fails "
        f"test_benchmark_run_ids_match_published_assets. Staged: {staged}"
    )


def test_ci_release_gate_matches_local_release_gate() -> None:
    """The GHA release-gate job and `pixi run release-gate` are one gate.

    docs/release.md tells the maintainer to run `pixi run release-gate` before
    a tag, and docs/changelog.md:2384 claims the CI job mirrors that task. Today
    they do: both name the same 20 test files and both run the docs contract and
    the docs-link check. Nothing holds them there, though -- the lists are
    20 hand-copied strings in two files, and `check-lane`/`preflight-push`/
    `changelog-check` never look at either. Adding a test file to the release
    set means editing two places, and forgetting the second leaves a release
    gated on less than the checklist promises while still reading green.
    """
    tasks = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    pixi_files = set(re.findall(r"tests/\S+\.py", tasks["release-gate"]))
    assert pixi_files, "pixi's release-gate names no test files; check is vacuous"

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for marker in ("\n  release-gate:\n", "name: Release gate tests"):
        assert marker in workflow, f"ci.yml no longer has {marker!r}"
    job = workflow.split("\n  release-gate:\n", 1)[1]
    step = job.split("name: Release gate tests", 1)[1].split("\n      - name:", 1)[0]
    ci_files = set(re.findall(r"tests/\S+\.py", step))

    assert ci_files == pixi_files, (
        "the CI release-gate job and `pixi run release-gate` must gate the same "
        f"test files. Only in CI: {sorted(ci_files - pixi_files)}. "
        f"Only in pixi: {sorted(pixi_files - ci_files)}"
    )
    # mypy is the other half of the claim: preflight-push runs it locally, so a
    # release-gate job that dropped it would be the one gate with no types.
    assert "mypy src/torchfits" in job, "the CI release-gate job no longer runs mypy"
    assert "check_docs_links.py" in job, (
        "the CI release-gate job no longer runs the docs-link check that "
        "`pixi run release-gate` ends with"
    )


def test_docs_contract_gate_is_defined_once() -> None:
    """The docs-contract file list must have exactly one home: `pixi.toml`.

    `test_ci_release_gate_matches_local_release_gate` (above) holds the release
    set together across two files. The docs-contract set had no such guard, and
    it was worse than untested: it was written down **three** times, in
    `pixi.toml`'s `docs-contract`, in the CI job of the same name, and again in
    `scripts/ci_local.sh`, which spelled out the pytest command and the docs
    build separately. All three happened to agree, and nothing was checking.

    That is not hypothetical. Adding `tests/test_docs_snippets.py` to `pixi.toml`
    and to the CI job -- both edits correct, both made in the same minute --
    silently left `scripts/ci_local.sh` gating on the old two files. The
    pre-existing `docs-contract` run stayed green, because the divergence was in
    a file the gate does not read.

    So the fix is two halves, and both are pinned here:

    1. `scripts/ci_local.sh` no longer copies the list; it calls
       `pixi run docs-contract`, which is the same task the CI job mirrors.
    2. This test compares the two remaining copies, so the CI job cannot drift
       from the pixi task the way the local script did.

    `pixi run docs-contract` must keep naming `tests/test_docs_integrity.py` and
    `tests/test_package_isolation.py` alongside whatever else is added: those two
    are the contract's substance, and a task that quietly stopped running them
    would still look like a gate.
    """
    tasks = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    pixi_files = set(re.findall(r"tests/\S+\.py", tasks["docs-contract"]))
    assert pixi_files, "pixi's docs-contract names no test files; check is vacuous"

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for marker in ("\n  docs-contract:\n", "name: Docs integrity"):
        assert marker in workflow, f"ci.yml no longer has {marker!r}"
    job = workflow.split("\n  docs-contract:\n", 1)[1].split("\n  release-gate:", 1)[0]
    ci_files = set(re.findall(r"tests/\S+\.py", job))

    assert ci_files == pixi_files, (
        "the CI docs-contract job and `pixi run docs-contract` must gate the "
        f"same test files. Only in CI: {sorted(ci_files - pixi_files)}. "
        f"Only in pixi: {sorted(pixi_files - ci_files)}"
    )

    # The local mirror must delegate, not restate: a fourth copy would bring the
    # problem straight back.
    ci_local = (ROOT / "scripts" / "ci_local.sh").read_text(encoding="utf-8")
    assert not re.search(r"tests/\S+\.py", ci_local), (
        "scripts/ci_local.sh names test files directly; it must call "
        "`pixi run docs-contract` so the list has one home"
    )
    assert "pixi run docs-contract" in ci_local, (
        "scripts/ci_local.sh no longer runs the docs contract"
    )

    for required in ("tests/test_docs_integrity.py", "tests/test_package_isolation.py"):
        assert required in pixi_files, (
            f"docs-contract stopped running {required}; the task still exists, "
            "so nothing else would notice"
        )


def _cited_bench_runs(doc: str) -> list[str]:
    """Run IDs the benchmarks page's PROVENANCE comment claims to summarise."""
    match = re.search(r"<!--\s*PROVENANCE:(.*?)-->", doc, re.S)
    assert match, "docs/benchmarks.md lost its PROVENANCE comment"
    return sorted(set(re.findall(r"exhaustive_\w+", match.group(1))))


def _headline(doc: str) -> str:
    """The opening blockquote -- the hand-written summary claim block."""
    quoted = [
        line[2:].strip() if line.startswith("> ") else line[1:].strip()
        for line in doc.splitlines()
        if line.startswith(">")
    ]
    assert quoted, "docs/benchmarks.md has no opening blockquote headline"
    return " ".join(quoted)


def test_benchmarks_headline_claims_match_the_cited_runs() -> None:
    """The headline blockquote is prose, so nothing regenerated it. It rotted.

    `scripts/patch_bench_docs.py` rewrites seven marked regions of
    `docs/benchmarks.md` (`BENCH_IOPATH`, `BENCH_HIGHLIGHTS`, `BENCH_FULL_TABLE`,
    `BENCH_DEFICITS`, `BENCH_HOSTS`, `BENCH_QUICK`, `BENCH_ML`). The opening
    blockquote is not one of them: it is hand-written prose, and every run that
    refreshes the numbers below it leaves it untouched. Measured against the two
    runs its own PROVENANCE comment cites:

    1. "The remaining **significant** case family where a peer is ahead is
       **narrow-table full reads**" -- but `exhaustive_cpu_20260807_013736` has
       *three* significant deficits in *two* case families. The unmentioned one
       is `fitstable / specialized / ascii_10000 [predicate_filter] / mmap=off`
       at 1.064x, 6.40% behind, flagged `significance=significant` by
       `compute_deficits` (fitstable floor is 1.05x, so 1.064 clears it). A
       reader auditing "did the 1.2 single-pass fix land?" would check the
       narrow-table rows, see them unchanged, and conclude yes.

    2. "Image HCOMPRESS lags vs fitsio are sub-1.03x noise" -- true on CPU
       (max hcompress lag 1.0219) and **false on CUDA**, where
       `compressed_hcompress_1 [read_full @ cuda]` reaches 1.0367x (3.67% behind
       vs `fitsio_torch_device`, mmap=off). The sentence names no host while the
       blockquote explicitly covers both cited runs.

    The other two clauses check out and are pinned here so they cannot drift
    either: zero significant `fits` deficits in both runs ("100% of significant
    image comparisons"), and smart-family table win rates of 98.9% (CPU) and
    99.5% (CUDA).
    """
    doc = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    runs = _cited_bench_runs(doc)
    assert runs, "the PROVENANCE comment names no runs; this check is vacuous"
    headline = _headline(doc)

    deficits: dict[str, list[dict[str, str]]] = {}
    for run in runs:
        path = ROOT / "docs" / "assets" / "bench" / run / "torchfits_deficits.csv"
        assert path.is_file(), f"PROVENANCE cites {run} but {path} is missing"
        with path.open(encoding="utf-8", newline="") as handle:
            deficits[run] = list(csv.DictReader(handle))
    assert any(deficits.values()), "cited runs contain no deficit rows; vacuous"

    def sig(run: str) -> list[dict[str, str]]:
        return [r for r in deficits[run] if r["significance"] == "significant"]

    # (1) "100% of significant image comparisons".
    image_deficits = {
        run: [r for r in sig(run) if r["domain"] == "fits"] for run in runs
    }
    assert not any(image_deficits.values()), (
        f"the headline claims 100% of significant image comparisons, but "
        f"significant fits deficits exist: {image_deficits}"
    )
    assert "100% of significant image comparisons" in headline

    # (2) every operation that a peer wins significantly must be named.
    # Aliases keep the prose readable ("full reads" for read_full).
    aliases = {"read_full": ("read_full", "full read")}
    unnamed: set[str] = set()
    for run in runs:
        for row in sig(run):
            for form in aliases.get(row["operation"], (row["operation"],)):
                if form in headline:
                    break
            else:
                unnamed.add(f"{run}:{row['operation']} ({row['case_label']})")
    assert not unnamed, (
        "the headline calls the remaining significant case families singular but "
        "the cited runs contain significant deficits in operations it never "
        f"names: {sorted(unnamed)}"
    )

    # (3) A lag bound is attributed to the *nearest* host name in its sentence,
    # and a sentence with no host name is a claim about every cited run. So
    # "sub-1.03x noise on CPU and 1.037x on CUDA" is checkable, and an
    # unqualified "sub-1.03x" has to hold on both -- which is what the CUDA run
    # (max hcompress lag 1.0367) violates today.
    hcompress_max = {
        run: max(
            (
                float(r["lag_ratio"])
                for r in deficits[run]
                if "hcompress" in r["case_label"].lower()
            ),
            default=0.0,
        )
        for run in runs
    }
    host_runs: dict[str, list[str]] = {"CPU": [], "CUDA": [], "MPS": [], "APPLE": []}
    for run in runs:
        for host, slug in (
            ("CPU", "cpu"),
            ("CUDA", "cuda"),
            ("MPS", "mps"),
            ("APPLE", "apple"),
        ):
            if slug in run.lower():
                host_runs[host].append(run)
    assert any(host_runs.values()), (
        f"no cited run id identifies a host, so the lag bounds cannot be "
        f"attributed: {runs}"
    )

    host_re = re.compile(r"\b(CPU|CUDA|MPS|Apple)\b")
    for sentence in re.split(r"(?<=\.)\s+", headline):
        hosts = [(m.start(), m.group(1).upper()) for m in host_re.finditer(sentence)]
        for match in re.finditer(r"(\d+\.\d+)×", sentence):
            bound = float(match.group(1))
            nearest = (
                min(hosts, key=lambda h: abs(h[0] - match.start()))[1] if hosts else ""
            )
            # `Apple Silicon` is the MPS host; the prose may not repeat the name.
            if nearest == "APPLE":
                nearest = "MPS"
            applicable = host_runs[nearest] or runs
            breached = {
                r: hcompress_max[r] for r in applicable if hcompress_max[r] >= bound
            }
            assert not breached, (
                f"headline claims hcompress lags stay under {bound}x"
                f"{f' on {nearest}' if nearest else ' on every cited run'}, but the "
                f"published deficits say otherwise: {breached} (max hcompress lag "
                f"per cited run: {hcompress_max})"
            )


# ──────────────────────────────────────────────────────────────────────────
# Row 18 (docs unit). Every guard below was red before its fix; the
# measurement that made each one a finding is recorded in
# .cursor/reviews/deep-review/20-docs.md.
# ──────────────────────────────────────────────────────────────────────────


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _dependency_floor(name: str) -> str:
    """The installed-version floor pyproject declares for a runtime dependency."""
    for req in _pyproject()["project"]["dependencies"]:
        if req.split(">")[0].split("=")[0].split("[")[0].strip() == name:
            match = re.search(r">=([0-9][0-9.]*)", req)
            assert match, f"{name} dependency has no >= floor: {req!r}"
            return match.group(1)
    raise AssertionError(f"{name} is not a runtime dependency in pyproject.toml")


def test_docs_state_the_real_numpy_and_pyarrow_floors() -> None:
    """docs/compatibility.md must not advertise floors the package cannot install.

    The "Core Libraries" row is the first place a reader looks when asking
    "can I run this on my numpy/pyarrow?", and the wheel-lane guard
    (scripts/check_torch_extra_pins.py) only ever read the *torch* specifier.
    The page claimed NumPy >= 1.20 and PyArrow >= 5.0 while pyproject,
    packaging/conda/recipe.yaml, pixi.toml and scripts/release_lane.py all
    pin numpy>=1.26 / pyarrow>=17.0 -- and docs/changelog.md says so in as many
    words. A reader on numpy 1.22 or pyarrow 12 was told they were supported.
    """
    numpy_floor = _dependency_floor("numpy")
    pyarrow_floor = _dependency_floor("pyarrow")

    # The floors must agree across every packaging surface, or the docs cannot
    # be checked against a single number.
    recipe = (ROOT / "packaging" / "conda" / "recipe.yaml").read_text(encoding="utf-8")
    assert f"numpy >={numpy_floor}" in recipe, (
        f"conda recipe numpy floor drifted from pyproject's >={numpy_floor}"
    )
    assert f"pyarrow >={pyarrow_floor}" in recipe, (
        f"conda recipe pyarrow floor drifted from pyproject's >={pyarrow_floor}"
    )

    text = (ROOT / "docs" / "compatibility.md").read_text(encoding="utf-8")
    row = re.search(r"^\| \*\*Core Libraries\*\* \| (?P<cells>.+?) \|$", text, re.M)
    assert row, "compatibility.md lost its 'Core Libraries' support-matrix row"
    cells = row.group("cells")
    for label, floor in (("NumPy", numpy_floor), ("PyArrow", pyarrow_floor)):
        claimed = re.search(rf"\*\*{label} [≥>]=?\s*([0-9][0-9.]*)\*\*", cells)
        assert claimed, (
            f"compatibility.md's Core Libraries row states no {label} floor; "
            f"pyproject requires >={floor}"
        )
        assert claimed.group(1) == floor, (
            f"docs/compatibility.md claims {label} >= {claimed.group(1)} but the "
            f"real floor is >={floor} (pyproject.toml, packaging/conda/recipe.yaml, "
            f"pixi.toml and scripts/release_lane.py all say >={floor})"
        )


def test_roadmap_deficit_band_matches_the_cited_benchmark_runs() -> None:
    """docs/roadmap.md must not quote a stale "the one remaining" deficit.

    The Current-focus bullet said the remaining significant deficit was
    "narrow-table full reads with mmap=False, ~6-17%", which is the 20260719
    CUDA band. The runs the benchmarks page cites -- and whose CSVs are
    committed under docs/assets/bench/ -- put narrow-table read_full at 20.96%
    and 36.46% on CPU and 8.30% on CUDA, and the CPU run carries a *second*
    significant case (ascii_10000 predicate_filter, 6.40%). The sibling page
    docs/benchmarks.md already states the corrected numbers in its headline.
    """
    text = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    bullets = [b for b in text.split("\n- ") if "mmap=False" in b]
    assert len(bullets) == 1, (
        f"expected exactly one roadmap bullet naming the mmap=False deficit, "
        f"found {len(bullets)}"
    )
    bullet = bullets[0]

    # same PROVENANCE comment the benchmarks headline guard reads
    cited = _cited_bench_runs(
        (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    )
    significant: dict[str, list[float]] = {}
    cases: dict[str, set[str]] = {}
    for run in cited:
        csv_path = ROOT / "docs" / "assets" / "bench" / run / "torchfits_deficits.csv"
        assert csv_path.is_file(), (
            f"PROVENANCE cites {run} but its deficits CSV is absent"
        )
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        sig = [r for r in rows if r.get("significance") == "significant"]
        significant[run] = [float(r["pct_behind"]) for r in sig]
        cases[run] = {r["case_label"] for r in sig}
    assert all(significant[run] for run in cited), (
        f"cited runs have no significant deficits to check: {significant}"
    )

    # Every percentage the bullet quotes must be a significant deficit of a
    # cited run, to the nearest half point.
    quoted = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", bullet)]
    assert quoted, "the roadmap deficit bullet quotes no percentage to check"
    allowed = [v for run in cited for v in significant[run]]
    wrong = [q for q in quoted if not any(abs(q - a) <= 0.5 for a in allowed)]
    assert not wrong, (
        f"docs/roadmap.md quotes deficit percentage(s) {wrong} that no cited run "
        f"reports. Significant pct_behind per cited run: "
        f"{ {run: sorted(v) for run, v in significant.items()} }. The cited runs "
        f"are the 20260807 pair; 6-17% is the 20260719 CUDA band."
    )

    # And it must not claim a single remaining deficit while a cited run has
    # significant deficits in more than one case family.
    multi = {run: sorted(fams) for run, fams in cases.items() if len(fams) > 1}
    if multi:
        singular = re.search(
            r"the (one|single|only) remaining significant", bullet, re.I
        )
        assert singular is None, (
            f"docs/roadmap.md calls it 'the {singular.group(1)} remaining "
            f"significant' deficit, but {multi} report significant deficits in "
            f"more than one case"
        )


def _api_signature_sections(page: str) -> list[tuple[str, str]]:
    """(name, section body) for every ``## `x.y()` `` heading in an api page."""
    text = (ROOT / "docs" / page).read_text(encoding="utf-8")
    out = []
    for match in re.finditer(r"^## `([A-Za-z_][\w.]*)\(\)`\s*$", text, re.M):
        body = text[match.end() :]
        nxt = re.search(r"^## ", body, re.M)
        out.append((match.group(1), body[: nxt.start()] if nxt else body))
    return out


def test_api_parameter_tables_cover_every_documented_parameter() -> None:
    """A `| Parameter | Type | Default |` table must list every parameter its
    own signature fence documents.

    Row 18 found these tables quietly dropping parameters that the section's own
    signature block lists, so the table reads as complete while omitting
    something a reader has to know about: ``table.write()`` dropped ``schema``
    and ``extname`` (2 of 7), and ``read_tensor()`` dropped
    ``fallback_get_header`` with no note that it is internal -- unlike
    ``read()``, whose table omits ``options`` and says so in the admonition
    right below it. ``read_subset()`` folds its window into one
    ``x1, y1, x2, y2`` row, so a comma-joined row counts as coverage.
    """
    import inspect

    import torchfits

    namespaces = {
        "api-core-io.md": torchfits,
        "api-tables.md": torchfits.table,
        "api-data.md": torchfits.data,
        "api-transforms.md": torchfits.transforms,
        "api.md": torchfits,
    }
    # Parameters deliberately left out of the table and covered in prose
    # immediately below it ("Advanced options" admonition on read()).
    prose_covered = {
        ("api-core-io.md", "read", "options"),
        ("api-core-io.md", "read", "kwargs"),
    }
    offenders: list[str] = []
    checked = 0
    for page, ns in namespaces.items():
        for name, section in _api_signature_sections(page):
            fence = re.search(r"```python\n(.*?)```", section, re.S)
            if not fence or "| Parameter | Type | Default" not in section:
                continue
            obj = getattr(ns, name.split(".")[-1], None)
            if obj is None or not callable(obj):
                continue
            try:
                live = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            live_params = {
                p.name
                for p in live.parameters.values()
                if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            }
            table_params: set[str] = set()
            for cells in re.findall(r"^\| `?([A-Za-z_][\w,\s]*?)`? \|", section, re.M):
                for part in cells.split(","):
                    table_params.add(part.strip())
            shown = set(re.findall(r"([A-Za-z_]\w*)\s*=", fence.group(1))) | set(
                re.findall(r"^\s*([A-Za-z_]\w*)[,)]", fence.group(1), re.M)
            )
            shown &= live_params
            if not shown:
                continue
            checked += 1
            # A markdown table that lists the same parameter twice is a defect
            # in its own right: it renders as two rows and reads as if the
            # parameter were listed twice. This bit during row 18 -- a fix pass
            # that raised *after* writing left one row in place and the retry
            # added a second.
            row_names: list[str] = []
            for cells in re.findall(r"^\| `?([A-Za-z_][\w,\s]*?)`? \|", section, re.M):
                row_names.extend(part.strip() for part in cells.split(","))
            dupes = sorted({n for n in row_names if row_names.count(n) > 1})
            assert not dupes, (
                f"docs/{page}: `{name}()` parameter table lists {dupes} more than once"
            )
            missing = sorted(
                p for p in shown - table_params if (page, name, p) not in prose_covered
            )
            if missing:
                offenders.append(
                    f"docs/{page}: `{name}()` parameter table omits {missing} "
                    f"(its own signature fence lists them; live params: "
                    f"{sorted(live_params)})"
                )
    assert checked >= 4, f"only {checked} parameter tables matched the pattern"
    assert not offenders, (
        "parameter tables silently drop documented parameters:\n" + "\n".join(offenders)
    )


def test_docs_only_use_math_alphabets_whose_katex_faces_are_vendored() -> None:
    """No docs expression may use a math alphabet whose woff2 is not shipped.

    docs/assets/katex/katex.min.css references 20 woff2 faces; the tree ships
    5 (AMS-Regular, Main-Regular, Math-Italic, Size1-Regular, Size2-Regular).
    Exactly one docs expression needed a face that is not there --
    ``\\texttt{stat="std"}`` in api-transforms.md -- which rendered in the
    browser's fallback monospace rather than KaTeX's own. This guard keeps
    that true in both directions: a page that starts using a non-vendored
    alphabet fails, and so does a deletion of a face the pages rely on.
    """
    fonts = ROOT / "docs" / "assets" / "katex" / "fonts"
    shipped = {p.name for p in fonts.glob("*.woff2")}
    assert shipped, "no vendored KaTeX woff2 faces found"

    css = (ROOT / "docs" / "assets" / "katex" / "katex.min.css").read_text(
        encoding="utf-8", errors="replace"
    )
    referenced = set(re.findall(r"fonts/(KaTeX_[A-Za-z0-9_\-]+\.woff2)", css))
    assert referenced, "katex.min.css declares no woff2 font faces"
    unknown = {f for f in shipped if f not in referenced}
    assert not unknown, (
        f"vendored KaTeX faces the shipped CSS never asks for: {sorted(unknown)}"
    )

    # command -> the face KaTeX resolves it to
    alphabet_face = {
        "texttt": "KaTeX_Typewriter-Regular.woff2",
        "mathtt": "KaTeX_Typewriter-Regular.woff2",
        "mathsf": "KaTeX_SansSerif-Regular.woff2",
        "mathsfbf": "KaTeX_SansSerif-Bold.woff2",
        "mathbf": "KaTeX_Main-Bold.woff2",
        "textbf": "KaTeX_Main-Bold.woff2",
        "mathcal": "KaTeX_Caligraphic-Regular.woff2",
        "mathbb": "KaTeX_AMS-Regular.woff2",
        "mathfrak": "KaTeX_Fraktur-Regular.woff2",
        "mathscr": "KaTeX_Script-Regular.woff2",
    }
    offenders: list[str] = []
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
        exprs = re.findall(
            r"\$\$(.+?)\$\$|\$(?![\s$])([^$\n]+?)(?<![\s\\])\$", text, re.S
        )
        for pair in exprs:
            body = pair[0] or pair[1]
            for cmd in set(re.findall(r"\\([a-zA-Z]+)", body)):
                face = alphabet_face.get(cmd)
                if face and face not in shipped:
                    offenders.append(
                        f"docs/{page.name}: \\{cmd} needs {face}, which is not "
                        f"vendored (shipped: {len(shipped)} of {len(referenced)} "
                        f"faces the CSS references)"
                    )
    assert not offenders, (
        "docs math uses alphabets whose KaTeX face is not vendored, so the "
        "rendered site silently falls back to a system font:\n"
        + "\n".join(sorted(set(offenders)))
    )


def test_every_gallery_image_is_linked_from_a_docs_page() -> None:
    """docs/assets/gallery/ must not accumulate figures no page displays.

    The mirror of ``test_every_published_example_is_linked_from_a_docs_page``
    (row 19) for the binary assets: that guard derives the published set from
    the two globs ``scripts/sync_docs_examples.sh`` copies, and the gallery
    directory sits one level away with the same blind spot.
    """
    gallery = ROOT / "docs" / "assets" / "gallery"
    figures = sorted(p.name for p in gallery.glob("*.png"))
    assert figures, "no gallery figures found"
    corpus = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md"))
    )
    orphans = [f for f in figures if f not in corpus]
    assert not orphans, (
        f"gallery figures referenced from no docs page: {orphans}. They are "
        f"tracked, copied into site/, and displayed nowhere. Link them or "
        f"delete them."
    )


# A repository path written as inline code in a page: `scripts/foo.sh`. Anchored
# on a top-level directory we own so prose like `src/` or a URL fragment cannot
# match. The trailing class excludes a trailing slash (a directory reference)
# and a colon (a `file.py:42` citation), both of which are not path claims.
_DOC_PATH_IN_CODE = re.compile(
    r"`((?:src|scripts|tests|docs|examples|benchmarks|extern|packaging|"
    r"pixi|pyproject)[A-Za-z0-9_./+-]*/?[A-Za-z0-9_+.-]*)`"
)

# References that are patterns or placeholders rather than one real path.
_NOT_A_CLAIMED_PATH = ("*", "?", "[", "]", "{", "}", "<", ">", "...", " ")


def _doc_pages_citing_paths() -> list[Path]:
    pages = [ROOT / "docs" / page for page in sorted((ROOT / "docs").glob("*.md"))]
    pages += [ROOT / name for name in ("README.md", "AGENTS.md", "CLAUDE.md")]
    return [page for page in pages if page.is_file()]


def test_every_documented_repository_path_is_tracked_or_ignored() -> None:
    """A file the docs cite must be either tracked or explicitly ignored.

    ``test_docs_reference_existing_local_files`` checks a hand-kept list of
    paths against the *filesystem*, which is exactly the wrong question. On the
    day this guard was written, ``docs/architecture.md`` and
    ``scripts/check_wheel_contents.py`` both described
    ``src/torchfits/cpp_src/core/`` and ``torchfits._core.pyi`` as shipped, and
    the local clone had them on disk -- untracked, and not ignored either,
    because the branch had drifted 25 commits behind its own upstream. A fresh
    clone of that commit could not build the core library the documentation
    promised. Filesystem existence said yes; ``git ls-files`` said no; nothing
    in the repository noticed.

    So the rule is the stronger of the two states: a cited path that exists on
    disk must be tracked, or it must be deliberately ignored (generated output
    such as ``docs/published-examples/`` is a legitimate reason). Being neither
    is the incoherence this catches.
    """
    cited: dict[str, set[str]] = {}
    for page in _doc_pages_citing_paths():
        for match in _DOC_PATH_IN_CODE.finditer(page.read_text(encoding="utf-8")):
            raw = match.group(1)
            if any(token in raw for token in _NOT_A_CLAIMED_PATH):
                continue
            path = raw.rstrip(":,.")
            if not (ROOT / path).is_file():  # not a claim about this checkout
                continue
            cited.setdefault(path, set()).add(page.name)

    if not cited:
        pytest.fail(
            "no documented repository paths found; the pattern stopped matching"
        )

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
    )
    untracked = sorted(p for p in cited if p not in tracked)
    # `git check-ignore` exits 0 for an ignored path, 1 for a tracked-or-unlisted
    # one. One batched call: a subprocess per path would dominate the test.
    ignored: set[str] = set()
    if untracked:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=ROOT,
            input="\n".join(untracked) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        ignored = {line for line in result.stdout.splitlines() if line}
    unignored = [path for path in untracked if path not in ignored]
    details = "\n".join(
        f"  {path}  (cited by {', '.join(sorted(cited[path]))})" for path in unignored
    )
    assert not unignored, (
        f"documented files that git does not track and does not ignore: "
        f"{len(unignored)}\n{details}\n\nThese exist in this working tree but "
        f"would be absent from a fresh clone, so the documentation describes a "
        f"repository nobody else has. `git add` them if they are project code, "
        f"or add a .gitignore entry if they are generated."
    )


# A fenced block may be indented (docs/index.md puts its CLI block inside an
# indented "Shell CLI Tool" section), so the fence is matched with a leading
# indent and the body is dedented before it is split into command segments.
_FENCE_BODY = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)


def test_every_documented_torchfits_command_parses() -> None:
    """Every `torchfits ...` command in docs/ must be accepted by the real parser.

    Two existing guards covered the CLI docs and neither could see a bad flag:
    ``test_all_cli_snippets_valid_commands`` (tests/test_docs_code_snippets.py)
    reads only ``args[0]`` and compares it to the subcommand set, and
    ``test_cli_docs_subcommands_match_parser`` harvests subcommand *names* out
    of the same pages. So ``docs/cli.md`` could tell a reader to run
    ``torchfits setkey science.fits -k FILTER --value "g" --comment "..."`` and
    the command exits 2 with "unrecognized arguments", while both gates
    reported OK.

    This walks every command segment in every fenced block of every language
    plus every inline ``\\`torchfits ...\\``` span, splits on shell separators,
    substitutes a dummy path for ellipsis/placeholder positionals, and runs the
    real ``build_parser()`` over what is left. An unrecognised flag or a
    malformed value fails. A command that names only a subcommand
    (``torchfits diff`` used as a noun phrase in prose) is skipped: the parser
    rejects it for a missing positional, which is not a defect.
    """
    import contextlib
    import io
    import shlex
    import textwrap

    from torchfits.cli.main import build_parser

    parser = build_parser()
    sep = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
    span = re.compile(r"`([^`\n]+)`")
    env_assign = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
    placeholder = re.compile(r"^(?:…|\.\.\.|<[^>]+>|\$\{?\w+\}?|FITS_FILES?|PATH)$")

    def segments(text: str) -> list[tuple[int, str]]:
        """(line number, command segment) for every shell command in `text`."""
        out: list[tuple[int, str]] = []
        for match in _FENCE_BODY.finditer(text):
            line = text[: match.start()].count("\n") + 1
            # join `\` continuations, then take one command per line
            body = textwrap.dedent(match.group(1)).replace("\\\n", " ")
            for offset, raw in enumerate(body.splitlines()):
                for part in sep.split(raw):
                    out.append((line + offset, part))
        for match in span.finditer(text):
            line = text[: match.start()].count("\n") + 1
            for part in sep.split(match.group(1)):
                out.append((line, part))
        return out

    def normalise(segment: str) -> list[str] | None:
        segment = segment.strip().lstrip("$ ").strip()
        while env_assign.match(segment):
            segment = segment.split(" ", 1)[1] if " " in segment else ""
        if not segment.startswith("torchfits "):
            return None
        try:
            args = shlex.split(segment)[1:]
        except ValueError as exc:  # an unbalanced quote is itself a defect
            raise AssertionError(f"shell-quoting error: {exc}") from exc
        if not args or args[0] in {"--help", "-h", "--version"}:
            return None
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", args[0]):
            return None
        return ["placeholder.fits" if placeholder.match(a) else a for a in args]

    # docs/changelog.md documents the CLI *as it was* at the release each
    # section describes, so a flag that has since been removed or renamed is
    # correct there: the 0.9.3 entry records `torchfits header --fitsort ...`
    # and the 1.0.0 "Deprecated" section records the rename. Only the released
    # sections are exempt -- `## Unreleased` and every other page are held to
    # the live parser.
    changelog = (ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    first_release = re.search(r"^## \[", changelog, re.M)
    assert first_release, "changelog.md has no released version section"
    history_from = changelog[: first_release.start()].count("\n") + 1

    offenders: list[str] = []
    per_page: dict[str, int] = {}
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        historical = page.name == "changelog.md"
        for line, segment in segments(text):
            try:
                args = normalise(segment)
            except AssertionError as exc:
                offenders.append(f"docs/{page.name}:{line}  {segment.strip()!r}  {exc}")
                continue
            if args is None:
                continue
            if historical and line >= history_from:
                continue
            per_page[page.name] = per_page.get(page.name, 0) + 1
            buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                    _, unknown = parser.parse_known_args(args)
            except SystemExit as exc:
                if exc.code in (0, None):
                    continue
                message = buf.getvalue()
                if "the following arguments are required" in message:
                    continue  # a subcommand mentioned as a noun, not a command
                offenders.append(
                    f"docs/{page.name}:{line}  {segment.strip()!r}\n"
                    "      "
                    + next(
                        (x for x in message.splitlines() if "error" in x), ""
                    ).strip()
                )
                continue
            if unknown:
                offenders.append(
                    f"docs/{page.name}:{line}  {segment.strip()!r}\n"
                    f"      the parser does not accept {unknown}"
                )
    # Per-page floors rather than a total, so a scan that silently stops
    # reading a page fails here instead of quietly passing on the rest.
    # changelog.md is counted from `## Unreleased` only, which is where its
    # current-state commands live.
    for name, floor in (
        ("cli.md", 15),
        ("cli-recipes.md", 8),
        ("quickstart.md", 5),
        ("index.md", 2),
        ("install.md", 1),
        ("changelog.md", 1),
    ):
        assert per_page.get(name, 0) >= floor, (
            f"only {per_page.get(name, 0)} documented commands were parsed out of "
            f"docs/{name} (expected >= {floor}); the scan regressed. "
            f"per-page counts: {per_page}"
        )
    assert not offenders, (
        "documented `torchfits` commands the CLI rejects:\n" + "\n".join(offenders)
    )


def _runner_forced_vars() -> dict[str, list[str]]:
    """Env vars `examples/test_examples.py` force on for a *named* example.

    The runner sets TORCHFITS_EXAMPLE_FAST for every example under CI and,
    separately, unconditionally for the denoise example whose training run
    would otherwise blow the timeout. That second forcing is invisible from
    the environment-variable table, which said the variable was "used by CI,
    examples/test_examples.py" and stopped there -- accurate, and leaving a
    reader to assume the denoise example runs at full scale when the smoke
    gate is what they will ever see.
    """
    import ast

    runner = ROOT / "examples" / "test_examples.py"
    if not runner.is_file():
        pytest.skip("examples/test_examples.py is not present")
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # `if name == "example_x.py":` and nothing else in the test
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "name"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)
        ):
            continue
        example = test.comparators[0].value
        for inner in ast.walk(node):
            if not isinstance(inner, (ast.Assign, ast.AnnAssign)):
                continue
            targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "env"
                    and isinstance(target.slice, ast.Constant)
                ):
                    out.setdefault(str(target.slice.value), []).append(example)
    return out


def test_env_var_table_names_every_forced_example() -> None:
    """docs/architecture.md's env-var table must name every per-example forcing.

    `test_architecture_md_env_tables_cover_every_source_getenv` (above) proves
    every `getenv` in the source has a table row. It says nothing about the row's
    contents, so a row can describe the variable's *purpose* correctly and still
    omit the part a reader needs: which examples the runner silently
    force-enables it for.
    """
    forced = _runner_forced_vars()
    assert forced, "the runner force-sets no per-example env var; the scan is broken"
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    rows: dict[str, str] = {}
    for match in re.finditer(r"^\| `([A-Z0-9_]+)` \| [^|]* \| (.+?) \|$", text, re.M):
        rows.setdefault(match.group(1), match.group(2))
    assert rows, "no env-var rows parsed out of docs/architecture.md"

    offenders: list[str] = []
    for var, examples in sorted(forced.items()):
        description = rows.get(var)
        if description is None:
            offenders.append(
                f"{var} is force-set by examples/test_examples.py for "
                f"{sorted(set(examples))} but has no row in the env-var table"
            )
            continue
        for example in sorted(set(examples)):
            if example not in description:
                offenders.append(
                    f"the `{var}` row does not mention {example}, which "
                    f"examples/test_examples.py force-sets it for"
                )
    assert not offenders, "\n".join(offenders)
