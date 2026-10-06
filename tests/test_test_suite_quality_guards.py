"""Guards against test classes that cannot fail (round-2 unit 9).

Unit 9 audited ``tests/`` (143 files, 45,756 lines) for tests that assert
nothing they could get wrong. Three shapes survived the whole suite; each is
mechanically detectable, so they are pinned here rather than left to review.

**1. Filesystem absence-only guards.** A test whose every assertion is "this
path is not there" is satisfied by an *empty* tree, so it passes when the
thing it inspects has been deleted or renamed. R2-049
(``test_torchfits_contains_only_fits_native_sources``) was exactly this: four
``.exists()`` checks and no positive anchor.

**2. Assertions swallowed by a handler that catches ``AssertionError``.** An
``assert`` inside a ``try`` whose handler ends in ``pass``/``continue`` and
catches ``Exception``/``BaseException``/``AssertionError`` can never fail --
its own failure is discarded. Handlers that catch only specific operational
errors (``except (RuntimeError, OSError, ValueError)``) are **fine** and are
deliberately not flagged: an ``AssertionError`` still propagates through them.

**3. Unreachable mock aliases.** ``with mock.patch(...) as cpp:`` binds ``cpp``
for the duration of the block. If nothing outside that block ever mentions
``cpp``, no assertion in the test can observe what was patched, and the test
verifies only its own bookkeeping. This is the pre-fix shape of
``test_tensor_hdu_concurrent_close_does_not_call_cpp_after_close`` (R2-048):
the mock was built inside the reader thread's ``with`` block and the
``RuntimeError`` encoding the refusal was swallowed, leaving ``assert not
errors`` as the only reachable assertion. The test passed while C++ was being
called after close.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SELF = Path(__file__).name
SCANNED = sorted(
    p
    for p in TESTS_DIR.rglob("*.py")
    if "cpp" not in p.relative_to(TESTS_DIR).parts and p.name != SELF
)

# Filesystem absence only. String/container absence ("x" not in value) is a
# strong assertion and must never match here.
_ABSENCE_ASSERT = re.compile(
    r"not\s+.*\.exists\(\)"
    r"|\.exists\(\)\s*==\s*False"
    r"|not\s+.*\.is_file\(\)"
    r"|not\s+.*\.is_dir\(\)"
)
# An assertion that anchors a test to a non-empty reality.
_POSITIVE_ANCHOR = re.compile(
    r"assert\s+(?!not\b|in\b)[A-Za-z_]\w*"  # `assert files`, not `assert not ...`
    r"|\.is_file\(\)"
    r"|\.is_dir\(\)"
    r"|len\([^()]*\)\s*>?=\s*[1-9]"
    r"|\.rglob\(|\.glob\(|\.iterdir\("
    r"|\.read_text\(|\.read_bytes\("
    r"|pytest\.raises"
)
# Handler exception names that would also catch a failing assertion.
_ASSERTION_EATING = frozenset({"Exception", "BaseException", "AssertionError"})

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


def _parse_first_func(source: str) -> _FuncDef:
    """Parse ``source`` and return its single top-level function definition."""
    first = ast.parse(source).body[0]
    assert isinstance(first, (ast.FunctionDef, ast.AsyncFunctionDef)), (
        f"probe source must define a function, got {type(first).__name__}"
    )
    return first


def _is_patch_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in {"patch", "patch_object"}
    if isinstance(func, ast.Name):
        return func.id in {"patch", "patch_object"}
    return False


def _iter_test_functions(path: Path) -> list[_FuncDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]


def _source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _absence_only_offenders() -> list[str]:
    """Test functions whose every assertion is a filesystem absence check."""
    offenders: list[str] = []
    for path in SCANNED:
        lines = _source_lines(path)
        for func in _iter_test_functions(path):
            body = lines[func.lineno - 1 : func.end_lineno]
            asserts = [ln for ln in body if re.match(r"\s*assert\b", ln)]
            if not asserts:
                continue
            if not all(_ABSENCE_ASSERT.search(a) for a in asserts):
                continue
            if _POSITIVE_ANCHOR.search("\n".join(body)):
                continue
            offenders.append(
                f"{path.relative_to(TESTS_DIR)}:{func.lineno} {func.name}() "
                f"({len(asserts)} absence assertions, no positive anchor)"
            )
    return offenders


def _swallowed_asserts(func: _FuncDef) -> list[tuple[int, str]]:
    """Assertions under a ``try`` whose handler would swallow the failure."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        eats_assertions = False
        for handler in node.handlers:
            statements = handler.body
            if len(statements) != 1 or not isinstance(
                statements[0], (ast.Pass, ast.Continue)
            ):
                continue
            caught: set[str] = set()
            exc = handler.type
            if exc is None:
                eats_assertions = True  # bare except:
            elif isinstance(exc, ast.Name):
                caught = {exc.id}
            elif isinstance(exc, ast.Tuple):
                caught = {e.id for e in exc.elts if isinstance(e, ast.Name)}
            if caught & _ASSERTION_EATING:
                eats_assertions = True
        if not eats_assertions:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assert):
                    hits.append((sub.lineno, ast.unparse(sub)))
    return hits


def _unreachable_patch_aliases(func: _FuncDef) -> list[str]:
    """Patch aliases bound but never referenced outside their ``with`` block."""
    references: dict[str, list[int]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            references.setdefault(node.id, []).append(node.lineno)

    dead: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            if not _is_patch_call(item.context_expr):
                continue
            if item.optional_vars is None:
                continue  # `mock.patch("x")` binds nothing
            if isinstance(item.optional_vars, ast.Name):
                names = [item.optional_vars.id]
            elif isinstance(item.optional_vars, ast.Tuple):
                names = [
                    e.id for e in item.optional_vars.elts if isinstance(e, ast.Name)
                ]
            else:
                continue
            last_line = node.end_lineno or node.lineno
            for name in names:
                outside = [
                    line
                    for line in references.get(name, [])
                    if not (node.lineno <= line <= last_line)
                ]
                if not outside:
                    dead.append(f"{name} (bound at line {node.lineno})")
    return dead


def test_no_guard_asserts_only_filesystem_absence() -> None:
    offenders = _absence_only_offenders()
    assert not offenders, (
        "these tests assert only that paths are absent, so they pass when the "
        "tree they inspect is empty; add an assertion that the container "
        "exists and is non-empty:\n  " + "\n  ".join(offenders)
    )


def test_no_assertion_is_swallowed_by_its_own_handler() -> None:
    offenders: list[str] = []
    for path in SCANNED:
        for func in _iter_test_functions(path):
            for lineno, text in _swallowed_asserts(func):
                offenders.append(
                    f"{path.relative_to(TESTS_DIR)}:{lineno} {func.name}(): {text}"
                )
    assert not offenders, (
        "these assertions sit inside a try whose handler passes and catches "
        "the assertion's own exception, so their failure is discarded and "
        "the test stays green:\n  " + "\n  ".join(offenders)
    )


def test_no_mock_alias_is_unreachable_from_any_assertion() -> None:
    offenders: list[str] = []
    for path in SCANNED:
        for func in _iter_test_functions(path):
            for alias in _unreachable_patch_aliases(func):
                offenders.append(
                    f"{path.relative_to(TESTS_DIR)}:{func.lineno} "
                    f"{func.name}(): patch alias {alias} is never referenced "
                    "outside its own with-block, so no assertion can observe "
                    "what was patched"
                )
    assert not offenders, "\n  ".join(offenders)


def test_scanner_actually_walks_the_suite() -> None:
    """The scanner is only a guard if it is still reading the suite.

    A detector that quietly stops matching reports green while inspecting
    nothing, which is worse than no detector at all.
    """
    assert len(SCANNED) > 100, (
        f"only {len(SCANNED)} test files discovered -- the scanner is not "
        "walking tests/"
    )
    assert SELF not in {p.name for p in SCANNED}

    # Absence detector: matches filesystem absence, rejects string absence.
    assert _ABSENCE_ASSERT.search('assert not (root / "a.cpp").exists()')
    assert _ABSENCE_ASSERT.search("assert not p.is_file()")
    assert not _ABSENCE_ASSERT.search('assert "&" not in value')
    assert not _ABSENCE_ASSERT.search("assert not offenders")
    # A positive anchor rescues an otherwise absence-only test.
    assert _POSITIVE_ANCHOR.search(
        'assert not (root / "a.cpp").exists()\nassert sources'
    )
    assert not _POSITIVE_ANCHOR.search('assert not (root / "a.cpp").exists()')

    # Swallow detector: only handlers that eat AssertionError count.
    def probe(body: str) -> list[tuple[int, str]]:
        return _swallowed_asserts(_parse_first_func(body))

    assert probe(
        "def t():\n"
        "    try:\n"
        "        assert x == 1\n"
        "    except Exception:\n"
        "        pass\n"
    )
    assert not probe(
        "def t():\n"
        "    try:\n"
        "        assert x == 1\n"
        "    except (RuntimeError, OSError, ValueError):\n"
        "        pass\n"
    ), "a handler catching only operational errors must not be flagged"

    # Patch-alias detector: the R2-048 shape, and its fixed counterpart.
    unreachable = _unreachable_patch_aliases(
        _parse_first_func(
            "def t():\n"
            '    with mock.patch("torchfits._C") as cpp:\n'
            "        cpp.read_full.side_effect = f\n"
            "        try:\n"
            "            hdu.to_tensor()\n"
            "        except RuntimeError:\n"
            "            pass\n"
            "    assert not errors\n"
        )
    )
    assert unreachable == ["cpp (bound at line 2)"]
    assert not _unreachable_patch_aliases(
        _parse_first_func(
            "def t():\n"
            '    with mock.patch("torchfits._C") as cpp:\n'
            "        cpp.read_full.side_effect = f\n"
            "    assert cpp.read_full.call_count == 0\n"
        )
    )
    # An unaliased patch binds nothing and is never a finding.
    assert not _unreachable_patch_aliases(
        _parse_first_func(
            'def t():\n    with mock.patch("torchfits._C"):\n        pass\n'
        )
    )


def test_the_r2_049_shape_is_detected() -> None:
    """The pre-fix containment guard, verbatim, must be rejected.

    ``test_torchfits_contains_only_fits_native_sources`` shipped four
    ``.exists()`` assertions and no anchor, so deleting ``cpp_src/`` wholesale
    left it green. Reproduced here so a weakened detector fails loudly.
    """
    pre_fix = _parse_first_func(
        "def test_torchfits_contains_only_fits_native_sources():\n"
        '    native_root = PACKAGE_ROOT / "cpp_src"\n'
        '    assert not (native_root / "wcs.cpp").exists()\n'
        '    assert not (native_root / "healpix.cpp").exists()\n'
        '    assert not (PACKAGE_ROOT / "wcs").exists()\n'
        '    assert not (PACKAGE_ROOT / "sphere").exists()\n'
    )
    lines = pre_fix.body  # four Assert nodes, no positive anchor
    assert all(
        _ABSENCE_ASSERT.search(ast.unparse(a.test))
        for a in lines
        if isinstance(a, ast.Assert)
    )
    assert not _POSITIVE_ANCHOR.search(ast.unparse(pre_fix))

    fixed = _parse_first_func(
        "def fixed():\n"
        '    native_root = PACKAGE_ROOT / "cpp_src"\n'
        "    assert native_root.is_dir()\n"
        "    sources = [p.name for p in native_root.rglob('*')]\n"
        "    assert sources\n"
        '    assert not (native_root / "wcs.cpp").exists()\n'
    )
    assert _POSITIVE_ANCHOR.search(ast.unparse(fixed)), (
        "the fixed shape must be anchored, or the detector rejects the fix too"
    )
