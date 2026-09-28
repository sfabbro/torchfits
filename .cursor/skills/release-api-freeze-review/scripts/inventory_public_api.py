#!/usr/bin/env python3
"""Compare torchfits public exports to docs/api.md quick-path mentions.

Exit code is 1 only when the tool cannot do its job (the static export parse
disagrees with the installed package, or ``__init__.py`` no longer has the shape
this script reads). Report findings -- exports missing from the docs, doc
symbols that are not root exports -- are printed for a human to judge and leave
the exit code at 0, because "the docs mention a name that is not a root export"
is usually a submodule member rather than a defect.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

# Doc symbols that belong to a submodule rather than the root package. Names
# absent from docs/api.md never reach this filter, so most entries here are inert
# today; it is kept because it is the only thing standing between a reader and a
# long list of `torchfits.table.read_torch`-style false positives.
SUBMODULE_ALLOWLIST = frozenset(
    {
        "scan",
        "TABLE_BACKENDS",
        "optimize_for_dataset",
        "configure_for_environment",
        "get_cache_stats",
        "clear_cache",
    }
)


def _unwrap(node: ast.expr) -> ast.expr | None:
    """`x`, `tuple([...])` and `[...]` all name the same literal list."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return node
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return _unwrap(node.args[0])
    return None


def _top_level_literals(text: str) -> dict[str, Any]:
    """Literal values of top-level assignments, with `*name` packs resolved.

    `__all__ = tuple([..., *_NAMESPACES])` and `_NAMESPACES = {...}` are both
    literals here, so the pack is expanded from the sibling definition. A pack
    whose target is *not* a literal in the same module raises rather than being
    skipped: silently dropping it is how four public namespaces went missing
    from this report while it kept printing success.
    """
    out: dict[str, Any] = {}
    packed: list[tuple[str, str]] = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name, value = getattr(node.targets[0], "id", None), node.value
        elif isinstance(node, ast.AnnAssign):
            name, value = getattr(node.target, "id", None), node.value
        else:
            continue
        if name is None or value is None:
            continue
        seq = _unwrap(value)
        if seq is None:
            try:
                out[name] = ast.literal_eval(value)
            except ValueError:
                continue  # computed at runtime, not a literal: nothing to read
            continue
        items: list[Any] = []
        for element in seq.elts:
            if isinstance(element, ast.Starred):
                target = ast.unparse(element.value)
                if not isinstance(out.get(target), (list, dict)):
                    raise SystemExit(
                        f"{name} packs `*{target}`, which is not a literal in "
                        "this module; teach load_all_from_init the new shape "
                        "rather than skipping exports"
                    )
                items.extend(out[target])
                packed.append((name, target))
                continue
            items.append(ast.literal_eval(element))
        out[name] = items
    return out


def load_all_from_init() -> set[str]:
    """Every name in ``__all__``, resolving the ``*_NAMESPACES`` star-unpack.

    This used to be ``text.split("__all__")[1].split(")")[0]``, which stops at
    the ``)`` that closes ``tuple([`` -- i.e. *before* the ``*_NAMESPACES``
    entry. Every lazy namespace was therefore invisible, the script compared 41
    of the 45 real exports, and it printed "All __all__ symbols appear in
    docs/api.md" on that incomplete set. Reading the namespaces from their own
    definition means a namespace added to the package is picked up here with no
    edit to this file.
    """
    init = ROOT / "src" / "torchfits" / "__init__.py"
    literals = _top_level_literals(init.read_text(encoding="utf-8"))

    namespaces = literals.get("_NAMESPACES")
    if not isinstance(namespaces, dict):
        raise SystemExit(
            f"{init}: _NAMESPACES is not a literal dict, so the lazy namespaces "
            "cannot be resolved; update load_all_from_init"
        )
    exported = literals.get("__all__")
    if not isinstance(exported, list):
        raise SystemExit(
            f"{init}: __all__ is not a literal list; update load_all_from_init"
        )
    return {name for name in exported} | set(namespaces)


def unparsed_exports(exports: set[str]) -> set[str]:
    """Exports the installed package has that the static parse did not see.

    The static parse is the only thing between this report and a confident "all
    exports are documented", so it is checked against the package it describes.
    A mismatch means `__init__.py` changed shape and this script is reading a
    subset -- which is how the 41-of-45 bug survived: nothing failed, the report
    just got quieter. Returns an empty set when torchfits cannot be imported, so
    the report still works in an environment without the built extension.
    """
    try:
        import torchfits
    except Exception:  # noqa: BLE001 - any import failure means "not checkable"
        return set()
    return set(torchfits.__all__) - exports


def load_api_doc_symbols() -> set[str]:
    text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    # torchfits.foo( or `foo` in backticks near API sections
    dotted = set(re.findall(r"torchfits\.([a-zA-Z_][a-zA-Z0-9_]*)", text))
    backtick = set(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", text))
    return dotted | backtick


def main() -> int:
    exports = load_all_from_init()
    unseen = unparsed_exports(exports)
    if unseen:
        print("This script's export parse is stale; the report below is partial.")
        print(f"  not seen by the static parse: {sorted(unseen)}")
        return 1

    doc_syms = load_api_doc_symbols()
    # Namespaces referenced in docs but not in __all__
    namespaces = {"table", "cache", "cpp", "transforms", "data", "where", "hdu"}
    doc_api = {s for s in doc_syms if s not in namespaces}

    missing_from_docs = sorted(exports - doc_api - namespaces - {"__version__"})
    extra_in_docs = sorted(
        s for s in doc_api if s not in exports and s not in SUBMODULE_ALLOWLIST
    )

    print(f"__all__ count: {len(exports)}")
    print(f"docs/api.md symbol mentions: {len(doc_syms)}")
    if missing_from_docs:
        print("\nIn __all__ but weak/absent in docs/api.md:")
        for name in missing_from_docs:
            print(f"  - {name}")
    else:
        print("\nAll __all__ symbols appear in docs/api.md (or namespaces).")

    if extra_in_docs:
        print("\nIn docs/api.md but not root __all__ (may be submodule API):")
        for name in extra_in_docs[:30]:
            print(f"  - {name}")
        if len(extra_in_docs) > 30:
            print(f"  ... and {len(extra_in_docs) - 30} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
