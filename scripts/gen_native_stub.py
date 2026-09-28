#!/usr/bin/env python3
"""Generate ``src/torchfits/_C.pyi`` — the type stub for the native extension.

The extension is nanobind, so ``nanobind.stubgen`` can derive most of the
surface automatically. Two things it cannot know are corrected here:

1. **Return types of every tensor/array/dict-returning function.** The C++
   bindings return ``nb::object`` wrapping a ``torch::Tensor`` (or a
   ``nb::dict`` of them), which stubgen can only render as ``object``. That is
   truthful but useless: ``mypy`` would reject every ``.to()``/``[key]`` at the
   call site. The mapping below is *measured at runtime* against a real image
   and BINTABLE (see ``tests/test_native_stub.py`` for the same probe under
   assertion), so the stub states what the extension actually returns.

2. **Signatures nanobind emits that are not legal Python.** A binding that
   declares a default on one argument and a required argument after it cannot
   be expressed as a Python signature; such cases are rewritten to the
   most-required form (never a lie: an over-strict stub is safe, an
   over-permissive one is not).

3. **The result is parsed before it is written.** stubgen, the rewrites above and
   the drift gate are all string-level, so a stub can be syntactically broken
   and still be committed faithfully. It is the file an IDE and a type checker
   read, so it is parsed here and the generator refuses to emit one that does
   not parse.

Three renderings also depend on the *environment* rather than the bindings, and
are canonicalized so one committed stub is correct across the whole CI matrix:
module-level constants keep their annotation but lose their value (``HAS_BZIP2``
is a CMake fact and differs by platform), ``types.CapsuleType`` — a name that
only exists from Python 3.13 — is declared once for every version, and the two
capsule-carrying signatures are pinned explicitly.

The result is committed, and ``--check`` fails if the committed stub no longer
matches the extension (same drift-gate pattern as the torch lane and changelog
checks). Regenerate with::

    pixi run python scripts/gen_native_stub.py

Requires the extension to be built and importable (``pixi run dev``).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Return types stubgen cannot infer (torchfits._C). Keys are the bare function name for
# module-level functions and ``Class.method`` for bound methods. Measured
# against a 3x4 float32 image and a 2-column BINTABLE:
# torch.Tensor / numpy.ndarray / dict[str, <that>].
C_RETURN_TYPES: dict[str, str] = {
    # Image reads -> torch.Tensor.
    "read_full": "torch.Tensor",
    "read_full_cached": "torch.Tensor",
    "read_full_nocache": "torch.Tensor",
    "read_full_raw": "torch.Tensor",
    "read_full_scaled_cpu": "torch.Tensor",
    "read_full_unmapped": "torch.Tensor",
    "read_full_unmapped_raw": "torch.Tensor",
    "read_hdus_sequence_last": "torch.Tensor",
    "read_tensor_from_handle": "torch.Tensor",
    "read_hdus_batch": "list[torch.Tensor]",
    "read_images_batch": "list[torch.Tensor]",
    "FITSFile.read_tensor": "torch.Tensor",
    "FITSFile.read_subset": "torch.Tensor",
    "SubsetReader.read": "torch.Tensor",
    "echo_tensor": "torch.Tensor",
    # Image reads -> numpy.ndarray.
    "read_full_numpy": "NDArray[Any]",
    "read_full_numpy_cached": "NDArray[Any]",
    # Table reads -> dict of one column per name.
    "read_fits_table": "dict[str, torch.Tensor]",
    "read_fits_table_from_handle": "dict[str, torch.Tensor]",
    "read_fits_table_rows": "dict[str, torch.Tensor]",
    "read_fits_table_rows_from_handle": "dict[str, torch.Tensor]",
    "read_fits_table_filtered": "dict[str, torch.Tensor]",
    "TableReader.read_rows": "dict[str, torch.Tensor]",
    "read_fits_table_rows_numpy": "dict[str, NDArray[Any]]",
    "read_fits_table_rows_numpy_from_handle": "dict[str, NDArray[Any]]",
    "TableReader.read_rows_numpy": "dict[str, NDArray[Any]]",
    # Header / metadata reads with heterogeneous contents.
    "read_keys": "dict[str, Any]",
    "read_header_dict": "list[tuple[str, Any]]",
    "read_table_info": "dict[str, Any]",
    # Tuple returns whose element order is not pinned by the binding.
    "read_full_raw_with_scale": "tuple[Any, ...]",
    "read_shape": "tuple[Any, ...]",
    "verify_hdu_checksums": "tuple[Any, ...]",
    "open_and_read_headers": "tuple[FITSFile, list[Any]]",
}

# Signatures that nanobind reports but Python cannot express. See module
# docstring: rewritten to the most-required form.
C_SIGNATURE_FIXES: dict[str, str] = {
    # ``nb::arg("column_names") = []`` followed by a required ``filters`` is
    # legal in C++ but has no Python equivalent. Both are in practice always
    # passed by the callers, and ``filters`` may not be empty anyway, so require
    # both rather than advertise a default nobody can use.
    "read_fits_table_filtered": (
        "def read_fits_table_filtered(filename: str, hdu_num: int, "
        "column_names: Sequence[str], filters: list) -> "
        "dict[str, torch.Tensor]: ..."
    ),
    # The mmap handle is an opaque ``nb::capsule``. nanobind spells its type
    # from the running interpreter, so pinning these two signatures keeps the
    # committed stub identical on every supported version. Parameter names and
    # defaults are exactly what the binding declares.
    "open_fits_mmap_reader": (
        "def open_fits_mmap_reader(path: str, hdu_num: int = 1) -> CapsuleType: ..."
    ),
    "read_fits_table_rows_mmap_from_reader": (
        "def read_fits_table_rows_mmap_from_reader(reader: CapsuleType, "
        "column_names: Sequence[str] = [], start_row: int = 1, "
        "num_rows: int = -1) -> object: ..."
    ),
}

# Text substitutions applied to every line, for spellings that differ by
# interpreter version.
_TEXT_FIXES: tuple[tuple[str, str], ...] = (("types.CapsuleType", "CapsuleType"),)

# Canonical import block. stubgen's own imports are dropped and replaced, so
# the block is fixed regardless of which stubs stubgen happens to emit.
C_IMPORTS = """from collections.abc import Sequence
import sys
from typing import Any, overload

import torch
from numpy.typing import NDArray

if sys.version_info >= (3, 13):
    from types import CapsuleType
else:
    class CapsuleType: ...  # opaque native handle; the name only exists on 3.13+

"""

C_HEADER = '''"""Type stubs for the native FITS extension.

Generated by ``scripts/gen_native_stub.py`` — do not edit by hand. Run
``python -m pytest tests/test_native_stub.py`` after changing the C++
bindings; a mismatch between this file and the extension fails the build.

Return types stubgen cannot infer (``torch.Tensor``, ``dict[str,
torch.Tensor]``, dicts of numpy arrays) are substituted from a table measured
against real FITS inputs.

Renderings that depend on the *environment* rather than the bindings are
canonicalized, so this file is byte-identical on every supported interpreter and
platform: module-level constants are typed without their value, and
``CapsuleType`` is declared for every version (the real name only exists on
3.13+).
"""

'''


# ---------------------------------------------------------------------------
# torchfits._core -- the torch-free metadata module.
# ---------------------------------------------------------------------------
# Same nanobind, so the same generator, but a much smaller surface and no torch
# in the return types. The one non-obvious entry is read_shape, whose tuple
# comes back as (bitpix, tuple-of-ints) and is narrowed in _core_api.py.
CORE_MODULE = "torchfits._core"
CORE_TARGET = ROOT / "src" / "torchfits" / "_core.pyi"

CORE_RETURN_TYPES: dict[str, str] = {
    "read_header_dict": "list[tuple[str, str, str]]",
    "read_keys": "dict[str, Any]",
    "read_table_info": "dict[str, Any]",
    "read_shape": "tuple[int, tuple[int, ...]]",
    "Metadata.scale_info": "tuple[bool, bool, float, float]",
    "Metadata.image_info": "tuple[int, int, tuple[int, ...]]",
}

CORE_IMPORTS = """from collections.abc import Sequence
from typing import Any

"""

CORE_HEADER = '''"""Type stubs for the torch-free native metadata module.

Generated by ``scripts/gen_native_stub.py`` — do not edit by hand. Run
``python -m pytest tests/test_native_stub.py`` after changing the C++
bindings; a mismatch between this file and the extension fails the build.

``torchfits._core`` is a nanobind module over ``libtorchfits_core``, which links
CFITSIO and nothing else. Nothing here may ever return a ``torch.Tensor``: the
whole reason this module exists is that the metadata path must not load
libtorch.
"""

'''

# Every stub the generator owns, in the order ``main`` checks them.
STUBS = (
    (
        "_C",
        "torchfits._C",
        ROOT / "src" / "torchfits" / "_C.pyi",
        C_HEADER,
        C_IMPORTS,
        C_RETURN_TYPES,
        C_SIGNATURE_FIXES,
    ),
    (
        "_core",
        CORE_MODULE,
        CORE_TARGET,
        CORE_HEADER,
        CORE_IMPORTS,
        CORE_RETURN_TYPES,
        {},
    ),
)


# The child has to import torch before it imports the extension. A pip-installed
# torch keeps libc10/libtorch under ``torch/lib`` and does not put that directory
# on the dynamic loader's search path; importing torch publishes those libraries
# process-wide, which is exactly why the package's own runtime init imports torch
# before ``torchfits._C``. A conda build carries an rpath and is unaffected, so
# this only bites on a pip install. stubgen runs in a child process, which
# inherits none of the parent's loaded objects.
_DRIVER = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "except Exception:\n"
    "    pass\n"
    "from nanobind.stubgen import main\n"
    "sys.exit(main(sys.argv[1:]))\n"
)


def _loader_var() -> str:
    """The dynamic loader's library-search-path variable for this platform."""
    if sys.platform == "darwin":
        return "DYLD_FALLBACK_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _torch_lib_dir() -> Path | None:
    """``torch/lib`` if torch is importable here, else ``None``."""
    try:
        import torch  # noqa: PLC0415 - deliberate: only needed to locate the libs
    except Exception:
        return None
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    return lib_dir if lib_dir.is_dir() else None


def _child_env() -> dict[str, str]:
    """Environment for the child, with torch's library dir on the loader path.

    Belt and braces alongside the child's own ``import torch``: this is what
    ``scripts/cibw_test.sh`` does for the released wheels, so the same failure
    mode is handled the same way in both places.
    """
    env = os.environ.copy()
    lib_dir = _torch_lib_dir()
    if lib_dir is not None:
        var = _loader_var()
        env[var] = os.pathsep.join(filter(None, (str(lib_dir), env.get(var))))
    return env


def _run_stubgen(out: Path, module: str) -> None:
    cmd = [sys.executable, "-c", _DRIVER, "-m", module, "-o", str(out)]
    env = _child_env()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        var = _loader_var()
        raise SystemExit(
            "nanobind.stubgen failed (is the extension built? try 'pixi run dev'):\n"
            f"{proc.stdout}{proc.stderr}"
            f"\n  interpreter: {sys.executable}"
            f"\n  {var}={env.get(var, '<unset>')}"
        )


# stubgen emits bare ``list``/``dict``/``tuple``/``NDArray`` for arguments whose
# element types it cannot know, which ``mypy --strict`` rejects (type-arg). An
# unknown element type is exactly ``Any``, so parametrize them rather than
# relaxing strictness for this file.
_BARE_GENERICS = (
    ("list", "list[Any]"),
    ("dict", "dict[str, Any]"),
    ("tuple", "tuple[Any, ...]"),
    ("NDArray", "NDArray[Any]"),
)


_CONSTANT_RE = re.compile(r"^([A-Za-z_]\w*): ([^=]+?) = .+$")


def _strip_constant_value(line: str) -> str:
    """Keep a module-level constant's annotation, drop its value.

    A stub describes types, not values, and these values are build facts:
    ``HAS_BZIP2`` reflects whether the vendored CFITSIO found libbz2. Encoding
    it would make the committed stub wrong on whichever platform disagrees.
    """
    match = _CONSTANT_RE.match(line)
    return f"{match.group(1)}: {match.group(2).strip()}" if match else line


def _parametrize_generics(line: str) -> str:
    if not line.lstrip().startswith("def "):
        return line
    for bare, full in _BARE_GENERICS:
        line = re.sub(rf"(?<![\w\[\].]){bare}(?!\[)", full, line)
    return line


def _drop_stubgen_imports(body: str) -> str:
    """Strip stubgen's leading docstring and import lines (replaced by IMPORTS).

    The docstring matters: it is the extension's ``m.doc()`` text, verbatim, and
    the generated stub already carries its own header. Leaving it in would put
    a second module-level string literal in the middle of the file.
    """
    lines = body.splitlines()
    if lines and lines[0].lstrip()[:3] in ('"""', "'''"):
        quote = lines[0].lstrip()[:3]
        if lines[0].strip().endswith(quote) and len(lines[0].strip()) > 3:
            lines.pop(0)
        else:
            lines.pop(0)
            while lines and quote not in lines[-1]:
                lines.pop(0)
            if lines:
                lines.pop(0)
    while lines and (
        not lines[0].strip()
        or lines[0].startswith("import ")
        or lines[0].startswith("from ")
    ):
        lines.pop(0)
    return "\n".join(lines)


def _rewrite(
    body: str,
    return_types: dict[str, str],
    signature_fixes: dict[str, str],
) -> str:
    """Apply return_types / signature_fixes; every entry must match."""
    lines = _drop_stubgen_imports(body).splitlines()
    seen: set[str] = set()
    out: list[str] = []

    for line in lines:
        for old, new in _TEXT_FIXES:
            line = line.replace(old, new)
        if line == line.lstrip():
            line = _strip_constant_value(line)
        line = _parametrize_generics(line)
        stripped = line.strip()
        if not stripped.startswith("def "):
            out.append(line)
            continue

        # ``def name(`` -> bare name; indented -> ``Class.method`` from the
        # most recent class statement.
        name = stripped[4:].split("(", 1)[0]
        key = name
        if line != line.lstrip():
            for prev in reversed(out):
                if prev.startswith("class "):
                    key = f"{prev[len('class ') :].split('(')[0].split(':')[0].strip()}.{name}"
                    break

        if key in signature_fixes:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + _parametrize_generics(signature_fixes[key]))
            seen.add(key)
            continue

        want = return_types.get(key)
        if want is not None and "->" in line:
            head, _, _tail = line.rpartition("->")
            # stubgen closes a def with ``: ...`` when the binding has no
            # docstring, and with a bare ``:`` when it does -- the docstring then
            # follows on indented lines underneath. Re-applying ``: ...`` to the
            # second form gives the def two bodies and the stub stops being
            # valid Python, so keep whichever terminator stubgen emitted. Test
            # the whole line, not the text after ``->``: that still contains the
            # return type stubgen inferred, which is the thing being replaced.
            terminator = ": ..." if line.rstrip().endswith("...") else ":"
            out.append(f"{head}-> {want}{terminator}")
            seen.add(key)
            continue

        out.append(line)

    missing = sorted((set(return_types) | set(signature_fixes)) - seen)
    if missing:
        raise SystemExit(
            "these symbols are no longer in the extension (rename or drop them "
            f"from the mapping in {Path(__file__).name}): {missing}"
        )
    return "\n".join(out) + "\n"


def diff_summary(want: str, got: str, limit: int = 40) -> str:
    """A short unified diff of the first differing lines, for failure messages.

    ``got`` is the committed stub, ``want`` the freshly generated one.
    """
    lines = list(
        difflib.unified_diff(
            got.splitlines(),
            want.splitlines(),
            fromfile="committed",
            tofile="generated",
            lineterm="",
            n=1,
        )
    )
    if not lines:
        return "(no line differences — the files differ only in trailing newlines)"
    if len(lines) > limit:
        lines = [*lines[:limit], f"... ({len(lines) - limit} more diff lines)"]
    return "\n".join(lines)


def build(
    label: str,
    module: str,
    header: str,
    imports: str,
    return_types: dict[str, str],
    signature_fixes: dict[str, str],
) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / f"{label}.pyi"
        _run_stubgen(raw, module)
        body = _rewrite(raw.read_text(encoding="utf-8"), return_types, signature_fixes)
        text = header + imports + body
    # A stub is Python, so it has to parse. Nothing else in this pipeline would
    # notice otherwise: stubgen produces it, the rewrites only compare strings,
    # and the drift gate is happy to commit a file that no type checker and no
    # IDE can read. This is the one place that can say so.
    try:
        ast.parse(text)
    except SyntaxError as exc:
        line = text.splitlines()[exc.lineno - 1] if exc.lineno else ""
        raise SystemExit(
            f"generated stub for {module} is not valid Python: {exc.msg}\n"
            f"  line {exc.lineno}: {line}\n"
            "A binding docstring probably collided with a return-type or "
            "signature rewrite in _rewrite()."
        ) from exc
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed stub differs from a freshly generated one",
    )
    args = parser.parse_args()

    failures = 0
    for label, module, target, header, imports, return_types, signature_fixes in STUBS:
        want = build(label, module, header, imports, return_types, signature_fixes)
        rel = target.relative_to(ROOT)
        if args.check:
            got = target.read_text(encoding="utf-8") if target.is_file() else ""
            if got != want:
                print(
                    f"[FAIL] {rel} is stale — run "
                    "'pixi run python scripts/gen_native_stub.py'",
                    flush=True,
                )
                print(diff_summary(want, got), flush=True)
                failures += 1
            else:
                print(f"[ OK ] {rel} matches the extension", flush=True)
        else:
            target.write_text(want, encoding="utf-8")
            print(f"[ OK ] wrote {rel}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
