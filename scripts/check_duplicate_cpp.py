#!/usr/bin/env python3
"""Detect duplicate file-scope function definitions across .cpp files.

Scans all .cpp files under src/torchfits/cpp_src/ (recursively, so the
``core/`` library is covered too) and reports any function name that is defined
at file scope in two or more files.  Two definitions of the same name at
namespace scope are a *link* error ("multiple definition of ..."), which is the
failure this guard exists to prevent, and it surfaces far from the edit that
caused it.  Excludes:

- Functions inside anonymous namespaces (internal linkage, never a conflict)
- Class/struct member functions (declared ``Class::method`` or defined inside
  the class body)
- Control-flow keywords (``if``, ``for``, ``while``, ``switch``, ``catch``)
- Macro invocations, lambda bodies, member-initializer lists and
  trailing-return-type parens -- anything whose ``) {`` is not a declarator

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
    # Type-ish tokens that are never a *function name*, but do sit right before
    # the '(' of a trailing return type: ``auto f() -> decltype(g(x)) {``.
    "decltype",
    "typename",
}

# Qualifiers that may sit between a declarator's ')' and its body.
_POST_ARG_QUALIFIERS = frozenset({"const", "noexcept", "override", "final"})

# Tokens that can only precede a *call*, never a declaration.  A file-scope
# definition always has a declaration specifier before its name -- a return
# type, a storage class, ``template``'s ``>``.  When one of these is there
# instead, the '(' belongs to something else: a macro invocation
# (``NB_MODULE(_C, m) {``), an argument's lambda (``with_open(path, [hdu](..) {``),
# a member initializer (``... : owner(owner) {``) or a braced initializer.
_NOT_A_DECLARATOR = frozenset(
    {
        ")",
        ",",
        "=",
        ";",
        "{",
        "}",
        ":",
        "return",
        "?",
    }
)

# Block kinds.  Only a *named namespace* (or a braced initializer, which is not a
# scope at all) may contain an extracted definition; every other enclosing block
# means the '{' is not at file scope.
_NAMED_NAMESPACE = "named-namespace"
_ANON_NAMESPACE = "anon-namespace"
_TYPE_BODY = "type-body"
_FUNCTION_BODY = "function-body"
_INITIALIZER = "initializer"
_OTHER_BLOCK = "other-block"

# Blocks a file-scope definition can appear inside.
_TRANSPARENT = frozenset({_NAMED_NAMESPACE, _INITIALIZER})

_TYPE_KEYWORDS = frozenset({"class", "struct", "union", "enum"})
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


# ---------------------------------------------------------------------------
# Stripping helpers
# ---------------------------------------------------------------------------


# Comments and literals are blanked in ONE left-to-right pass.  Whichever
# construct *starts* first wins, exactly as a C++ lexer sees it: a ``//`` inside
# a string is part of the string, and an apostrophe inside a comment is not a
# character literal.  No raw strings appear in this tree.
_STRIP_RE = re.compile(
    r"""
      //[^\n]*                    # line comment
    | /\*.*?\*/                   # block comment
    | "(?:\\.|[^"\\\n])*"         # string literal
    | '(?:\\.|[^'\\\n])*'         # character literal
    """,
    re.VERBOSE | re.DOTALL,
)


def strip_comments_and_literals(code: str) -> str:
    """Blank out comments and literals, keeping newlines so lines still line up.

    This must not be staged into a literals pass followed by a comments pass.
    Doing so let the character-literal pattern pair an apostrophe inside a
    ``//`` comment with the next apostrophe *anywhere later in the file* and
    delete the real code between them -- measured at 14,516 characters of
    ``fits_file.cpp``, 6,658 of ``core/metadata_api.cpp`` and 2,793 of
    ``core/fits_core.cpp``, unbalanced braces included, so brace bookkeeping
    after that point was reading truncated input.
    """
    return _STRIP_RE.sub(lambda m: "\n" * m.group(0).count("\n") + " ", code)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    [a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)*   # identifier, possibly qualified
    |
    ::                                  # standalone scope operator
    |
    ->                                  # member access / trailing return arrow
    |
    =                                   # assignment (distinguishes `auto x = []() {`)
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


def _declarator_name(tokens: list[str], brace_idx: int) -> str | None:
    """The function name whose body starts at ``brace_idx``, else ``None``.

    Walks back from the ``{`` over trailing qualifiers, then out of the
    parameter list.  Every rejection below is a shape that looks like a
    ``) {`` but declares nothing, and each one used to be reported as a
    file-scope function under a name that was never a function at all:

    ``NB_MODULE(_C, m) {``      a macro invocation      -> ``NB_MODULE``
    ``owner(owner) {``         a member initializer    -> ``owner``
    ``path_(path) {``          a member initializer    -> ``path_``
    ``with_open(p, [h](f) {``  an argument's lambda    -> ``h``
    ``= []() {``               a constexpr initializer -> ``=``
    ``FitsReader::~F() {``     a destructor            -> ``FitsReader``
    ``-> decltype(g(x)) {``    a trailing return type  -> ``decltype``
    """
    idx = brace_idx - 1
    while idx > 0 and tokens[idx] in _POST_ARG_QUALIFIERS:
        idx -= 1
    if tokens[idx] != ")":
        return None  # not a function body
    # A trailing return type sits between the parameter list and the body
    # (``auto f() -> decltype(g(x)) {``); its parens are the ones the '{' sees
    # first, so step back over the whole return type.  The search is bounded by
    # the enclosing statement: an arrow from an *earlier* line belongs to that
    # line's declarator, and stepping back to it would name the wrong function
    # for every '{' in the rest of the file.
    arrow = _trailing_return_arrow(tokens, idx)
    if arrow is not None:
        params = _last_index_before(tokens, ")", arrow)
        if params is None or params == idx:
            return None
        idx = params

    open_idx = _matching_open_paren(tokens, idx)
    if open_idx is None or open_idx < 2:
        return None
    name = tokens[open_idx - 1]
    if not _IDENT_RE.match(name) or name in KEYWORDS:
        return None
    if "::" in name:
        return None  # ``Class::method`` -- a member definition
    if tokens[open_idx - 2] in _NOT_A_DECLARATOR:
        # No declaration specifier before the name: this '(' is a call, an
        # initializer, or a macro argument list, not a declarator.
        return None
    if open_idx >= 3 and tokens[open_idx - 2] == "::":
        return None  # ``void Class::method()`` split around the scope operator
    return name


def _last_index_before(tokens: list[str], token: str, limit: int) -> int | None:
    """Index of the last *token* strictly before *limit*, or ``None``."""
    for i in range(limit - 1, -1, -1):
        if tokens[i] == token:
            return i
    return None


def _trailing_return_arrow(tokens: list[str], close_paren: int) -> int | None:
    """Index of a trailing-return ``->`` inside this declarator, or ``None``.

    Stops at the statement boundary (``;``, ``{``, ``}``) so an arrow belonging
    to the previous line is never mistaken for this one's, and only accepts an
    arrow that directly follows the parameter list's ``)`` -- ``meta->mutex``
    inside the parameter list is not a return type.
    """
    for i in range(close_paren - 1, -1, -1):
        if tokens[i] in (";", "{", "}"):
            return None
        if tokens[i] == "->":
            return i if i > 0 and tokens[i - 1] == ")" else None
    return None


def _matching_open_paren(tokens: list[str], close_idx: int) -> int | None:
    """Index of the ``(`` matching the ``)`` at *close_idx*, or ``None``."""
    depth = 1
    idx = close_idx - 1
    while idx >= 0 and depth > 0:
        if tokens[idx] == ")":
            depth += 1
        elif tokens[idx] == "(":
            depth -= 1
            if depth == 0:
                return idx
        idx -= 1
    return None


def _block_kind(tokens: list[str], brace_idx: int) -> tuple[str, str | None]:
    """Classify the block opened at *brace_idx*, with its declarator name."""
    prev = tokens[brace_idx - 1] if brace_idx > 0 else ""
    if prev == "namespace":
        return _ANON_NAMESPACE, None
    if (
        brace_idx >= 2
        and _IDENT_RE.match(prev)
        and tokens[brace_idx - 2] == "namespace"
    ):
        return _NAMED_NAMESPACE, None
    name = _declarator_name(tokens, brace_idx)
    if name is not None:
        return _FUNCTION_BODY, name
    window = tokens[max(0, brace_idx - 6) : brace_idx]
    if any(t in _TYPE_KEYWORDS for t in window):
        return _TYPE_BODY, None
    if _IDENT_RE.match(prev) and not (
        brace_idx >= 2 and tokens[brace_idx - 2] in _TYPE_KEYWORDS
    ):
        # ``} guard{fptr};`` -- a member initializer, not a scope.  It must not
        # be treated as a block, or the '}' that follows it is credited to it and
        # every later definition in the file looks nested one level too deep.
        return _INITIALIZER, None
    return _OTHER_BLOCK, None


def _matching_brace(tokens: list[str], open_idx: int) -> int | None:
    """Index of the ``}`` matching the ``{`` at *open_idx*, or ``None``."""
    depth = 0
    for j in range(open_idx, len(tokens)):
        if tokens[j] == "{":
            depth += 1
        elif tokens[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return None


def _collect(tokens: list[str], start: int, end: int, at_file_scope: bool) -> set[str]:
    """Names defined at file scope in ``tokens[start:end]``.

    Recursive rather than stack-based: each block's extent comes from counting
    its own braces, so a construct like ``} guard{fptr};`` (a '}' that closes an
    enclosing block and a '{' that opens nothing) cannot shift the pairing for
    the rest of the file.
    """
    found: set[str] = set()
    i = start
    while i < end:
        if tokens[i] != "{":
            i += 1
            continue
        close = _matching_brace(tokens, i)
        if close is None or close > end:
            break
        kind, name = _block_kind(tokens, i)
        if name is not None and at_file_scope:
            found.add(name)
        if kind in _TRANSPARENT:
            found |= _collect(tokens, i + 1, close, at_file_scope)
        i = close + 1
    return found


def extract_file_scope_functions(filepath: Path) -> set[str]:
    """Return the set of file-scope function names defined in *filepath*."""
    try:
        code = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    tokens = _tokens(strip_comments_and_literals(code))
    return _collect(tokens, 0, len(tokens), at_file_scope=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def iter_cpp_files(root: Path | None = None) -> list[Path]:
    """Every .cpp under *root* (default ``CPP_SRC_DIR``), recursively.

    ``cpp_src/core/`` holds the compiled ``torchfits_core`` sources; a
    non-recursive glob hid all of them from this guard.  The default is
    resolved at call time so the tests can point it at a fixture tree.
    """
    return sorted((root if root is not None else CPP_SRC_DIR).rglob("*.cpp"))


def main() -> int:
    cpp_files = iter_cpp_files()
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
        if len(set(files)) >= 2:
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

    print(
        f"✅ No duplicate file-scope functions detected across "
        f"{len(cpp_files)} .cpp files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
