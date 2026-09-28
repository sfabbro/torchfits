"""Unit tests for scripts/check_duplicate_cpp.py (no compiler needed).

Each case is a shape that either must be reported (two file-scope definitions
of one name -- a link error) or must *not* be reported (anything that merely
looks like ``) {``).  The guard runs in CI on every push, so a false positive
blocks a PR and a false negative ships a broken link; both directions are
pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPP_SRC = ROOT / "src" / "torchfits" / "cpp_src"
sys.path.insert(0, str(ROOT / "scripts"))

import check_duplicate_cpp as guard  # noqa: E402


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _names(tmp_path: Path, name: str, body: str) -> set[str]:
    return guard.extract_file_scope_functions(_write(tmp_path, name, body))


# --------------------------------------------------------------------------
# the scan must cover the whole tree, not just its top level
# --------------------------------------------------------------------------


def test_iter_cpp_files_is_recursive() -> None:
    """``cpp_src/core/`` holds compiled sources; a top-level glob missed them."""
    files = guard.iter_cpp_files()
    names = {p.name for p in files}
    assert files, "the real cpp_src tree must not be empty"
    for compiled in ("fits_core.cpp", "metadata_api.cpp", "parallel.cpp"):
        assert compiled in names, f"{compiled} is compiled but invisible to the guard"
    core_dir = {p.parent.name for p in files}
    assert "core" in core_dir
    # No file may be reported twice (dedup by path, not by name).
    assert len(files) == len(set(files))


def test_duplicate_in_a_subdirectory_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(guard, "CPP_SRC_DIR", tmp_path)
    body = "namespace tf { int tf_dup_symbol(int x) { return x; } }\n"
    _write(tmp_path, "a.cpp", body)
    _write(tmp_path / "core", "b.cpp", body.replace("x;", "y;"))
    assert guard.main() == 1
    out = capsys.readouterr().out
    assert "tf_dup_symbol" in out
    assert "a.cpp" in out and "b.cpp" in out


def test_real_tree_has_no_duplicates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(guard, "CPP_SRC_DIR", CPP_SRC)
    assert guard.main() == 0
    assert "No duplicate file-scope functions" in capsys.readouterr().out


# --------------------------------------------------------------------------
# shapes that look like a definition but are not one
# --------------------------------------------------------------------------


def test_macro_invocation_is_not_a_function(tmp_path: Path) -> None:
    got = _names(tmp_path, "m.cpp", 'NB_MODULE(_C, m) {\n    m.def("x", 1);\n}\n')
    assert got == set()


def test_member_initializer_and_destructor_are_not_functions(tmp_path: Path) -> None:
    body = (
        "struct MMapHandle {\n"
        "    MMapHandle(const std::string& f, bool w) : owner(w) {}\n"
        "    MMapHandle() : owner(false) {}\n"
        "    ~MMapHandle() { cleanup(); }\n"
        "    bool owner;\n"
        "};\n"
    )
    got = _names(tmp_path, "h.cpp", body)
    assert got == set(), f"constructor/destructor shapes leaked: {sorted(got)}"


def test_call_argument_parens_and_lambdas_are_not_functions(tmp_path: Path) -> None:
    body = (
        "int use(const std::string& path, int hdu) {\n"
        "    return with_open(path, [hdu](fitsfile* fptr) { return hdu; });\n"
        "}\n"
    )
    got = _names(tmp_path, "l.cpp", body)
    assert got == {"use"}, f"lambda/call shapes leaked: {sorted(got)}"


def test_braced_initializers_are_not_functions(tmp_path: Path) -> None:
    body = (
        "const bool kFlag = []() { return true; }();\n"
        "void g() {\n"
        "    char value[8] = {0};\n"
        "    std::array<long, 9> naxes{};\n"
        "    if (value[0]) { return; }\n"
        "}\n"
    )
    got = _names(tmp_path, "b.cpp", body)
    assert got == {"g"}, f"initializer shapes leaked: {sorted(got)}"


def test_control_flow_blocks_are_not_functions(tmp_path: Path) -> None:
    body = (
        "void f(int x) {\n"
        "    if (x) { x++; } else if (x > 2) { x--; } else { x = 0; }\n"
        "    for (int i = 0; i < x; ++i) { x += i; }\n"
        "    while (x) { x--; }\n"
        "    switch (x) { case 1: break; default: break; }\n"
        "    try { x = 1; } catch (const std::exception& e) { x = 2; }\n"
        "}\n"
    )
    assert _names(tmp_path, "c.cpp", body) == {"f"}


def test_member_definitions_are_not_functions(tmp_path: Path) -> None:
    body = (
        "class Foo {\n"
        "  public:\n"
        "    void in_class() {}\n"
        "    void out_of_class();\n"
        "};\n"
        "void Foo::out_of_class() {}\n"
        "void free_function() {}\n"
    )
    assert _names(tmp_path, "m.cpp", body) == {"free_function"}


def test_trailing_return_type_still_names_the_function(tmp_path: Path) -> None:
    body = (
        "template <typename F>\n"
        "auto with_open(const std::string& path, F&& body) -> decltype(body(nullptr)) {\n"
        "    return body(nullptr);\n"
        "}\n"
    )
    got = _names(tmp_path, "t.cpp", body)
    assert got == {"with_open"}, f"trailing return mis-attributed: {sorted(got)}"


# --------------------------------------------------------------------------
# namespace scope, the one case the guard does report
# --------------------------------------------------------------------------


def test_named_namespace_definitions_are_reported(tmp_path: Path) -> None:
    body = "namespace torchfits {\nnamespace detail {\nvoid helper() {}\n}\n}\n"
    assert _names(tmp_path, "n.cpp", body) == {"helper"}


def test_anonymous_namespace_definitions_are_ignored(tmp_path: Path) -> None:
    """Internal linkage: two translation units may both define these."""
    body = "namespace {\nvoid internal_helper() {}\n}\n"
    assert _names(tmp_path, "a.cpp", body) == set()


def test_duplicate_in_two_named_namespaces_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(guard, "CPP_SRC_DIR", tmp_path)
    _write(tmp_path, "one.cpp", "namespace a { void shared_name() {} }\n")
    _write(tmp_path, "two.cpp", "namespace b { void shared_name() {} }\n")
    assert guard.main() == 1


# --------------------------------------------------------------------------
# the pre-processor must not delete real code
# --------------------------------------------------------------------------


def test_apostrophe_in_a_comment_does_not_swallow_code(tmp_path: Path) -> None:
    """An apostrophe in ``//`` must not pair with the next one in the file.

    The old stripper removed literals in a pass *before* comments, so the
    apostrophe in ``file's`` paired with the one in ``it's`` and deleted
    everything between them -- including the definition below, and its braces.
    The fixture keeps a real definition *between* the two apostrophes, which is
    the shape that truncates a real translation unit.
    """
    body = (
        "// another file's bytes to readers holding old-generation metadata.\n"
        "void survives_the_comment() {\n"
        "    int x = 1;\n"
        "}\n"
        "// and it is fine, so it's still there\n"
    )
    assert _names(tmp_path, "s.cpp", body) == {"survives_the_comment"}


def test_comment_markers_inside_a_string_are_not_comments(tmp_path: Path) -> None:
    body = 'const char* kUrl = "http://example.invalid/a";\nvoid after_the_url() {}\n'
    assert _names(tmp_path, "u.cpp", body) == {"after_the_url"}


def test_brace_in_a_string_does_not_unbalance_the_walk(tmp_path: Path) -> None:
    body = (
        'const char* kFmt = "{not a block}";\n'
        "void after_the_format() {\n"
        "    if (kFmt[0] == '{') { return; }\n"
        "}\n"
    )
    assert _names(tmp_path, "q.cpp", body) == {"after_the_format"}


@pytest.mark.parametrize("source", sorted(p.name for p in CPP_SRC.rglob("*.cpp")))
def test_real_sources_survive_stripping_intact(source: str) -> None:
    """Every compiled source must still balance its braces after stripping.

    This is the standing detector for the truncation defect: the guard reads
    the *stripped* text, so a stripped file that does not balance means real
    code was deleted and every later brace in that file was misread.
    """
    raw = (
        (CPP_SRC / "core" / source)
        if not (CPP_SRC / source).is_file()
        else (CPP_SRC / source)
    )
    text = raw.read_text(encoding="utf-8", errors="replace")
    stripped = guard.strip_comments_and_literals(text)
    depth = 0
    for ch in stripped:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, f"{source}: a '}}' with no '{{' after stripping"
    assert depth == 0, f"{source}: {depth} unclosed brace(s) after stripping"
    # And the two must agree on the number of braces, since nothing outside a
    # comment or literal may contain one in C++ (no raw strings in this tree).
    assert stripped.count("{") == stripped.count("}")
    assert text.count("{") == text.count("}")
