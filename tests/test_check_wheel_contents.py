"""`scripts/check_wheel_contents.py` must accept a real Linux wheel.

The wheel-content contract is the one packaging gate that runs on a plain PR
build instead of waiting for a release tag, and it is the check that exists
because a wheel once shipped without `libtorchfits_core` and passed every
functional test. So it has to be right about the platform it runs on.

Measured 2026-10-04, before this test existed. The classifier separated
extension *modules* from the core *library* by suffix:

    modules   = sorted(n for n in native if n.endswith((".so", ".pyd")))
    libraries = sorted(n for n in native if n not in modules)

That is exactly correct for macOS, where the library is
`libtorchfits_core.dylib`, and exactly wrong on Linux, where CMake writes the
same target as `libtorchfits_core.so` -- an extension module's own suffix. The
classification therefore put all three artifacts in `modules` and left
`libraries` empty, and every one of the three checks failed:

    $ python scripts/check_wheel_contents.py <linux wheel>
    [FAIL] expected the _C and _core extension modules, got
           ['torchfits/_C.cpython-313-x86_64-linux-gnu.so',
            'torchfits/_core.cpython-313-x86_64-linux-gnu.so',
            'torchfits/libtorchfits_core.so']
    [FAIL] expected exactly two extension modules (_C, _core), got [...]
    [FAIL] expected libtorchfits_core as the only non-module library, got [].
           The extension resolves its CFITSIO symbols against it, so _C cannot
           import without it.
    rc=1

The suffix is a platform-dependent proxy for "is this an importable Python
module"; the artifact's stem is not. `_C`/`_core` are the two module names the
build produces on every platform and the two `nanobind_add_module` targets in
`src/torchfits/cpp_src/CMakeLists.txt`, so the stem is the stable identity.

Nothing on macOS could have caught this: the locally built wheel classifies
correctly. The tests below build synthetic wheels for each platform from the
real dist-info shape, so the classification is pinned on every platform from a
macOS checkout.
"""

from __future__ import annotations

import importlib.util
import re
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_wheel_contents.py"

# What `nanobind_add_module` + `install(TARGETS ...)` produce per platform.
# Linux is the one the contract got wrong; Windows is here because `.pyd` is
# the other module suffix the classifier special-cased.
LAYOUTS = {
    "linux-x86_64": (
        "_C.cpython-313-x86_64-linux-gnu.so",
        "_core.cpython-313-x86_64-linux-gnu.so",
        "libtorchfits_core.so",
    ),
    "manylinux_2_28_aarch64": (
        "_C.cpython-313-aarch64-linux-gnu.so",
        "_core.cpython-313-aarch64-linux-gnu.so",
        "libtorchfits_core.so",
    ),
    "macosx_11_0_arm64": (
        "_C.cpython-313-darwin.so",
        "_core.cpython-313-darwin.so",
        "libtorchfits_core.dylib",
    ),
    "win_amd64": (
        "_C.cp313-win_amd64.pyd",
        "_core.cp313-win_amd64.pyd",
        "libtorchfits_core.dll",
    ),
}


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_wheel_contents", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match, "pyproject.toml must carry a literal project version"
    return match.group(1)


def _make_wheel(tmp_path: Path, tag: str, native: tuple[str, ...]) -> Path:
    """A wheel with every entry the contract requires, on a chosen platform.

    The payload is deliberately minimal: these tests are about which native
    entries are classified as modules and which as the library, so the rest of
    the contract is satisfied once and the platform is the only variable.
    """
    version = _pyproject_version()
    dist_info = f"torchfits-{version}.dist-info"
    entries: dict[str, str] = {
        "torchfits/__init__.py": "# torchfits\n",
        "torchfits/py.typed": "",
        # The three symbols STUB_REQUIRED_SYMBOLS names, so a stub that is
        # present and complete does not add a second failure to the report.
        "torchfits/_C.pyi": "class FITSFile: ...\nclass TableReader: ...\n"
        "def read_full(*args, **kwargs): ...\n",
        "torchfits/_core.pyi": "def read_header_dict(*args, **kwargs): ...\n",
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.4\n"
            "Name: torchfits\n"
            f"Version: {version}\n"
            "License-Expression: MIT\n"
        ),
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\n",
        f"{dist_info}/licenses/LICENSE": "MIT\n",
        f"{dist_info}/licenses/extern/licenses/CFITSIO-LICENSE.txt": "CFITSIO\n",
    }
    for name in native:
        # A name carrying a "/" is a full wheel path and is used verbatim, so a
        # test can place an artifact somewhere other than `torchfits/` (or at a
        # second path inside it) without reaching past this helper.
        entries[name if "/" in name else f"torchfits/{name}"] = ""

    wheel = tmp_path / f"torchfits-{version}-cp313-cp313-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return wheel


def _run(wheel: Path) -> tuple[int, str]:
    checker = CHECKER
    problems: list[str] = []
    original_fail = checker._fail

    def capture(found: list[str]) -> int:
        problems.extend(found)
        return original_fail(found)

    checker._fail = capture
    try:
        rc = checker.check(wheel)
    finally:
        checker._fail = original_fail
    return rc, "\n".join(problems)


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_complete_wheel_passes_on_every_platform(tmp_path: Path, tag: str) -> None:
    """The contract must accept the wheel the build actually produces.

    The comment in the checker explains why the classification cannot be done
    by suffix: the core library is `libtorchfits_core.{so,dylib}`, which both
    contains "_core" *and*, on Linux, ends in the extension-module suffix. A
    classifier that gets macOS right is not evidence about Linux.
    """
    wheel = _make_wheel(tmp_path, tag, LAYOUTS[tag])

    rc, report = _run(wheel)

    assert rc == 0, f"{tag}: the contract rejected a complete wheel:\n{report}"


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_wheel_without_the_core_library_is_rejected(tmp_path: Path, tag: str) -> None:
    """Losing `libtorchfits_core` must fail on every platform, not just macOS.

    This is the defect the whole checker exists for: a wheel that shipped
    without the core library imported fine for every functional test, because
    nothing dlopens it until a `fits_*` symbol is called.
    """
    c_module, core_module, _library = LAYOUTS[tag]
    wheel = _make_wheel(tmp_path, tag, (c_module, core_module))

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a wheel missing libtorchfits_core was accepted"
    assert "libtorchfits_core" in report, (
        f"{tag}: the rejection must name the artifact that is missing, "
        f"not just count native entries:\n{report}"
    )


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_native_artifact_outside_the_package_is_rejected(
    tmp_path: Path, tag: str
) -> None:
    """Native artifacts must all land inside `torchfits/`.

    `_C` resolves the core library through an rpath relative to the module
    (`@loader_path` / `$ORIGIN`), so a copy that landed at the wheel root -- an
    auditwheel-vendored library, a stray install rule -- would not be the one
    that gets loaded, and the mismatch would only surface as an undefined
    symbol in a user's environment.
    """
    c_module, core_module, library = LAYOUTS[tag]
    wheel = _make_wheel(
        tmp_path,
        tag,
        (c_module, core_module, library, f"lib/{library}"),
    )

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a native artifact outside torchfits/ was accepted"
    assert "must all live under torchfits/" in report


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_foreign_native_artifact_is_rejected(tmp_path: Path, tag: str) -> None:
    """A native artifact that is neither `_C`, `_core` nor the core library fails.

    It is a stale build output that rode along, and the wheel's only consumer
    of the native tree is `dlopen`, which picks by path -- so the contract has
    to refuse the wheel rather than let a stale copy shadow a fresh module.
    """
    c_module, core_module, library = LAYOUTS[tag]
    stale = c_module.replace("_C.", "_stale.")
    wheel = _make_wheel(tmp_path, tag, (c_module, core_module, library, stale))

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a wheel carrying a foreign native artifact was accepted"
    assert "libtorchfits_core" in report, (
        f"{tag}: the rejection must be about the library set:\n{report}"
    )


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_module_installed_at_two_paths_is_rejected(tmp_path: Path, tag: str) -> None:
    """`len(modules) == 2` is not implied by the stem set; pin it directly.

    With the stem-based classifier, a second copy of `_core` under a different
    directory still yields the stem set `{_C, _core}`, so the set comparison
    alone passes. The count is what catches it, and it is the case a stray
    `LIBRARY_OUTPUT_DIRECTORY` or a duplicated install rule would produce.
    """
    c_module, core_module, library = LAYOUTS[tag]
    wheel = _make_wheel(
        tmp_path,
        tag,
        (c_module, core_module, library, f"nested/{core_module}"),
    )

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a wheel with _core installed twice was accepted"
    assert "exactly two extension modules" in report, (
        f"{tag}: the rejection must be about the module count:\n{report}"
    )


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_two_copies_of_one_module_do_not_satisfy_the_stem_set(
    tmp_path: Path, tag: str
) -> None:
    """The stem-set comparison is not redundant with `len(modules) == 2`.

    Two copies of `_core` and no `_C` is two extension modules, so the count
    alone is satisfied; only the stem set says the wrong module is missing.
    This is the check that stops a *renamed* module from passing, which is
    what a classifier loosened to a substring or prefix match would produce.
    """
    _c_module, core_module, library = LAYOUTS[tag]
    wheel = _make_wheel(
        tmp_path,
        tag,
        (core_module, library, f"nested/{core_module}"),
    )

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a wheel with only _core installed was accepted"
    assert "expected the _C and _core extension modules" in report, (
        f"{tag}: the rejection must name the missing module, not the count:\n{report}"
    )


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_wheel_missing_an_extension_module_is_rejected(
    tmp_path: Path, tag: str
) -> None:
    """Dropping either extension module must still fail.

    This is also what pins the stem set: a wheel without `_C` has exactly one
    extension module, and both the count and the `module_stems` comparison have
    to reject it.
    """
    c_module, core_module, library = LAYOUTS[tag]
    for kept, dropped in ((core_module, c_module), (c_module, core_module)):
        wheel = _make_wheel(tmp_path, tag, (kept, library))
        rc, report = _run(wheel)
        assert rc != 0, f"{tag}: a wheel missing {dropped.split('.')[0]} was accepted"
        assert "extension modules" in report


@pytest.mark.parametrize("tag", sorted(LAYOUTS))
def test_a_wheel_with_a_stale_second_core_library_is_rejected(
    tmp_path: Path, tag: str
) -> None:
    """A leftover library from an earlier build must not ride along.

    The changelog records the shape of this failure: a GPU build left the
    first build's `libtorchfits_core` in place, so the wheel carried two
    copies and dlopen picked one of them.
    """
    c_module, core_module, library = LAYOUTS[tag]
    stale = library.replace("libtorchfits_core", "libtorchfits_core_stale")
    wheel = _make_wheel(tmp_path, tag, (c_module, core_module, library, stale))

    rc, report = _run(wheel)

    assert rc != 0, f"{tag}: a wheel carrying two core libraries was accepted"
    assert "libtorchfits_core" in report
