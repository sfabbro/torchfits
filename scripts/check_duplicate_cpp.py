#!/usr/bin/env python3
"""Detect duplicate file-scope function definitions across .cpp files.

Scans all .cpp files under src/torchfits/cpp_src/ and reports any function
name that is defined at file scope in two or more files.  Excludes:

- Functions inside anonymous namespaces (``namespace {``)
- Class/struct member functions (contain ``::`` in the name)
- Control-flow keywords (``if``, ``for``, ``while``, ``switch``, ``catch``)
- Functions inside named namespaces that are intentionally duplicated

Exit 0 on success (no duplicates), exit 1 with a message on failure.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CPP_SRC_DIR = Path("src/torchfits/cpp_src")

# Function names that are *intentionally* duplicated (e.g. ``bind_table`` that
# appears both as a nanobind entry point and in a different .cpp file).
ALLOWLIST: set[str] = set()

# Keywords that should never be treated as function names.
KEYWORDS: set[str] = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "class",
    "struct",
    "namespace",
    "enum",
    "template",
    "typedef",
    "using",
}


# ---------------------------------------------------------------------------
# Stripping helpers
# ---------------------------------------------------------------------------


def strip_comments_and_literals(code: str) -> str:
    """Remove C++ string/char literals, then comments.

    Strings are stripped first so that ``//`` and ``/*`` inside string
    literals don't corrupt comment removal.
    """
    # String literals  "..." (handles escape sequences)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '" "', code)
    # Character literals  '...'
    code = re.sub(r"'(?:\\.|[^'\\])*'", "' '", code)
    # Block comments  /* ... */
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    # Line comments  // ...
    code = re.sub(r"//[^\n]*", " ", code)
    return code


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    [a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)*   # identifier, possibly qualified
    |
    ::                                  # standalone scope operator
    |
    [{}();]                             # structural tokens
    """,
    re.VERBOSE,
)


def _tokens(code: str) -> list[str]:
    return _TOKEN_RE.findall(code)


# ---------------------------------------------------------------------------
# Function-extraction engine
# ---------------------------------------------------------------------------


def extract_file_scope_functions(filepath: Path) -> set[str]:
    """Return the set of file-scope function names defined in *filepath*."""

    try:
        code = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return set()

    code = strip_comments_and_literals(code)
    tokens = _tokens(code)

    functions: set[str] = set()
    depth = 0  # current brace depth (0 = file scope)

    # The brace-depth value at which we entered an anonymous namespace or a
    # class/struct body.  While any of these sets is non-empty we suppress
    # function extraction.
    anon_ns_depths: set[int] = set()
    class_depths: set[int] = set()

    for i, token in enumerate(tokens):
        # ---- entering a block ------------------------------------------------
        if token == "{":
            _detect_block_type(tokens, i, depth, anon_ns_depths, class_depths)
            _try_extract_function(
                tokens, i, depth, anon_ns_depths, class_depths, functions
            )
            depth += 1

        # ---- leaving a block -------------------------------------------------
        elif token == "}":
            depth -= 1
            anon_ns_depths.discard(depth)
            class_depths.discard(depth)

    return functions


def _detect_block_type(
    tokens: list[str],
    brace_idx: int,
    depth: int,
    anon_ns_depths: set[int],
    class_depths: set[int],
) -> None:
    """Record whether the opening brace at *brace_idx* belongs to an anonymous
    namespace or a class/struct definition."""

    # Look at the token immediately before '{'
    prev = tokens[brace_idx - 1] if brace_idx > 0 else ""

    # Anonymous namespace:  `namespace {`
    if prev == "namespace":
        anon_ns_depths.add(depth)

    # Class / struct – look backwards a handful of tokens for 'class'/'struct'
    window = tokens[max(0, brace_idx - 6) : brace_idx]
    if any(t in ("class", "struct") for t in window):
        class_depths.add(depth)


def _try_extract_function(
    tokens: list[str],
    brace_idx: int,
    depth: int,
    anon_ns_depths: set[int],
    class_depths: set[int],
    functions: set[str],
) -> None:
    """If the '{' at *brace_idx* begins a file-scope function body, extract its
    name and add it to *functions*."""

    # Only extract at file-scope (or inside named namespaces, not inside anon
    # namespaces or classes)
    if anon_ns_depths or class_depths:
        return

    if brace_idx < 2:
        return

    # Walk backwards from '{' skipping post-arg qualifiers
    idx = brace_idx - 1
    while idx > 0 and tokens[idx] in ("const", "noexcept", "override", "final"):
        idx -= 1

    if tokens[idx] != ")":
        return  # not a function definition

    # Walk backwards to matching '('
    paren_depth = 1
    idx -= 1
    while idx >= 0 and paren_depth > 0:
        if tokens[idx] == ")":
            paren_depth += 1
        elif tokens[idx] == "(":
            paren_depth -= 1
            if paren_depth == 0:
                break
        idx -= 1

    if paren_depth != 0 or idx <= 0:
        return  # unbalanced parentheses

    func_name = tokens[idx - 1] if idx > 0 else ""

    # Validation ---------------------------------------------------------------
    if not re.match(r"^[a-zA-Z_]\w*$", func_name):
        return
    if func_name in KEYWORDS:
        return  # control-flow keyword, not a function
    if "::" in func_name:
        return  # class member

    functions.add(func_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cpp_files = sorted(CPP_SRC_DIR.glob("*.cpp"))
    if not cpp_files:
        print("❌ No .cpp files found in", str(CPP_SRC_DIR))
        return 1

    func_to_files: dict[str, list[str]] = defaultdict(list)
    for filepath in cpp_files:
        for func in extract_file_scope_functions(filepath):
            if func not in ALLOWLIST:
                func_to_files[func].append(filepath.name)

    duplicates_found = False
    for func in sorted(func_to_files):
        files = func_to_files[func]
        if len(files) >= 2:
            print(f"❌ Duplicate file-scope function:  {func}")
            for f in files:
                print(f"       {f}")
            duplicates_found = True

    if duplicates_found:
        print(
            "\n🚫 CI check failed: duplicate file-scope function definitions detected.\n"
            "   Consolidate or use the ALLOWLIST in scripts/check_duplicate_cpp.py.",
        )
        return 1

    print("✅ No duplicate file-scope functions detected across .cpp files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
