#!/usr/bin/env python3
"""Compare torchfits public exports to docs/api.md quick-path mentions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def load_all_from_init() -> set[str]:
    init = ROOT / "src" / "torchfits" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    names: set[str] = set()
    # Static list inside __all__ = tuple([...])
    for match in re.finditer(
        r'"([a-zA-Z_][a-zA-Z0-9_]*)"', text.split("__all__")[1].split(")")[0]
    ):
        names.add(match.group(1))
    names.update({"table", "cache", "cpp"})
    return names


def load_api_doc_symbols() -> set[str]:
    text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    # torchfits.foo( or `foo` in backticks near API sections
    dotted = set(re.findall(r"torchfits\.([a-zA-Z_][a-zA-Z0-9_]*)", text))
    backtick = set(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", text))
    return dotted | backtick


def main() -> int:
    exports = load_all_from_init()
    doc_syms = load_api_doc_symbols()
    # Namespaces referenced in docs but not in __all__
    namespaces = {"table", "cache", "cpp"}
    doc_api = {s for s in doc_syms if s not in namespaces}

    missing_from_docs = sorted(exports - doc_api - namespaces - {"__version__"})
    extra_in_docs = sorted(
        s
        for s in doc_api
        if s not in exports
        and s
        not in {
            "scan",
            "TABLE_BACKENDS",
            "optimize_for_dataset",
            "configure_for_environment",
            "get_cache_stats",
            "clear_cache",
        }
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
