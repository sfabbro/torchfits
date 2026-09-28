"""The native extension's type stub must be current and truthful.

``src/torchfits/_C.pyi`` is what makes ``mypy --strict`` see the C++/Python
boundary at all — without it, ``import torchfits._C`` is ``Any`` and every
native call is unchecked. These tests keep it honest in two directions:

1. **Current** — the committed stub must equal a fresh generation from the
   live extension (the same drift-gate pattern as the torch lane and the
   changelog). A C++ binding change that is not reflected in the stub fails
   here.
2. **Truthful** — every return type the generator *declares* must hold at
   runtime. stubgen derives most of the stub from the module, but the
   tensor/array/dict returns come from a hand-written table, and nothing else
   verifies those. This is the test that would catch e.g. a read starting to
   return numpy where the stub promises ``torch.Tensor``.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("nanobind.stubgen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_native_stub as gen  # noqa: E402


@pytest.fixture(scope="module")
def fits_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """A minimal float image and a two-column BINTABLE."""
    import torchfits
    import torchfits.table

    tmp = tmp_path_factory.mktemp("native_stub")
    image = tmp / "img.fits"
    table = tmp / "tbl.fits"

    torchfits.write(
        image, torch.arange(12, dtype=torch.float32).reshape(3, 4), overwrite=True
    )
    torchfits.table.write(
        table,
        {
            "ID": np.arange(5, dtype=np.int32),
            "MAG": np.linspace(0.0, 1.0, 5),
        },
        overwrite=True,
    )
    return str(image), str(table)


def test_stub_matches_live_extension() -> None:
    """Every committed stub is exactly what the generator produces today.

    Two modules are checked: ``_C`` (the torch-linked extension) and ``_core``
    (the metadata module over the torch-free library). A binding added to
    either without a regenerated stub is what this catches.
    """
    for (
        label,
        module,
        target,
        header,
        imports,
        return_types,
        signature_fixes,
    ) in gen.STUBS:
        got = target.read_text(encoding="utf-8") if target.is_file() else ""
        fresh = gen.build(label, module, header, imports, return_types, signature_fixes)
        assert fresh == got, (
            f"{target.relative_to(ROOT)} is stale — run "
            "'pixi run python scripts/gen_native_stub.py'\n"
            + gen.diff_summary(fresh, got)
        )


def test_stubgen_child_can_find_torchs_shared_libraries() -> None:
    """The stubgen child inherits torch's shared libraries.

    Regression guard for a real CI-only failure. stubgen runs in a child
    process, which inherits none of the parent's loaded objects, and a
    pip-installed torch keeps ``libc10``/``libtorch`` under ``torch/lib`` and
    publishes them only once ``import torch`` runs. Without this the child dies
    with ``libc10.so: cannot open shared object file`` on every test cell while a
    conda install passes regardless, because its interpreter carries an rpath.
    """
    lib_dir = gen._torch_lib_dir()
    assert lib_dir is not None, "torch must be importable to exercise this"

    var = gen._loader_var()
    parts = gen._child_env()[var].split(os.pathsep)
    assert parts[0] == str(lib_dir), "torch's libs must be searched first"


def test_stubgen_child_env_keeps_an_existing_search_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing loader path is appended to, not replaced."""
    var = gen._loader_var()
    monkeypatch.setenv(var, "/somewhere/else")

    parts = gen._child_env()[var].split(os.pathsep)
    assert parts[-1] == "/somewhere/else"


def test_declared_return_types_are_accurate(fits_inputs: tuple[str, str]) -> None:
    """Every declared return type holds for a real call."""
    import torchfits._C as cpp

    image, table = fits_inputs
    state: dict[str, object] = {}

    def fits_file() -> object:
        state.setdefault("fitsfile", cpp.FITSFile(image, 0))
        return state["fitsfile"]

    def subset_reader() -> object:
        state.setdefault("subset", cpp.SubsetReader(image, 0))
        return state["subset"]

    def table_reader() -> object:
        state.setdefault("tablereader", cpp.TableReader(table, 1))
        return state["tablereader"]

    # name -> call. Keyed exactly as RETURN_TYPES (bare name, or Class.method).
    calls: dict[str, object] = {
        "read_full": lambda: cpp.read_full(image, 0),
        "read_full_cached": lambda: cpp.read_full_cached(image, 0),
        "read_full_nocache": lambda: cpp.read_full_nocache(image, 0),
        "read_full_raw": lambda: cpp.read_full_raw(image, 0),
        "read_full_scaled_cpu": lambda: cpp.read_full_scaled_cpu(image, 0),
        "read_full_unmapped": lambda: cpp.read_full_unmapped(image, 0),
        "read_full_unmapped_raw": lambda: cpp.read_full_unmapped_raw(image, 0),
        "read_hdus_sequence_last": lambda: cpp.read_hdus_sequence_last(image, [0]),
        "read_tensor_from_handle": lambda: cpp.read_tensor_from_handle(fits_file(), 0),
        "read_hdus_batch": lambda: cpp.read_hdus_batch(image, [0]),
        "read_images_batch": lambda: cpp.read_images_batch([image], 0),
        "FITSFile.read_tensor": lambda: fits_file().read_tensor(0),
        "FITSFile.read_subset": lambda: fits_file().read_subset(0, 0, 2, 2, 0),
        "SubsetReader.read": lambda: subset_reader().read(0, 0, 2, 2),
        "echo_tensor": lambda: cpp.echo_tensor(torch.zeros(2)),
        "read_full_numpy": lambda: cpp.read_full_numpy(image, 0),
        "read_full_numpy_cached": lambda: cpp.read_full_numpy_cached(image, 0),
        "read_fits_table": lambda: cpp.read_fits_table(table, 1),
        "read_fits_table_from_handle": lambda: cpp.read_fits_table_from_handle(
            cpp.open_fits_file(table, "r"), 1
        ),
        "read_fits_table_rows": lambda: cpp.read_fits_table_rows(table, 1),
        "read_fits_table_rows_from_handle": lambda: (
            cpp.read_fits_table_rows_from_handle(cpp.open_fits_file(table, "r"), 1)
        ),
        "read_fits_table_filtered": lambda: cpp.read_fits_table_filtered(
            table, 1, ["ID"], [("ID", ">", 1)]
        ),
        "TableReader.read_rows": lambda: table_reader().read_rows(),
        "read_fits_table_rows_numpy": lambda: cpp.read_fits_table_rows_numpy(table, 1),
        "read_fits_table_rows_numpy_from_handle": lambda: (
            cpp.read_fits_table_rows_numpy_from_handle(
                cpp.open_fits_file(table, "r"), 1
            )
        ),
        "TableReader.read_rows_numpy": lambda: table_reader().read_rows_numpy(),
        "read_keys": lambda: cpp.read_keys(image, 0, ["NAXIS"]),
        "read_header_dict": lambda: cpp.read_header_dict(image, 0),
        "read_table_info": lambda: cpp.read_table_info(table, 1),
        "read_full_raw_with_scale": lambda: cpp.read_full_raw_with_scale(image, 0),
        "read_shape": lambda: cpp.read_shape(image, 0),
        "verify_hdu_checksums": lambda: cpp.verify_hdu_checksums(image, 0),
        "open_and_read_headers": lambda: cpp.open_and_read_headers(image, 0),
    }

    # Declared type -> predicate. A new declaration must add a predicate here,
    # so the table cannot silently grow a type nothing checks.
    def is_tensor_list(v: object) -> bool:
        return isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v)

    def is_tensor_dict(v: object) -> bool:
        return isinstance(v, dict) and all(
            isinstance(x, torch.Tensor) for x in v.values()
        )

    def is_array_dict(v: object) -> bool:
        return isinstance(v, dict) and all(
            isinstance(x, np.ndarray) for x in v.values()
        )

    predicates: dict[str, object] = {
        "torch.Tensor": lambda v: isinstance(v, torch.Tensor),
        "NDArray[Any]": lambda v: isinstance(v, np.ndarray),
        "list[torch.Tensor]": is_tensor_list,
        "dict[str, torch.Tensor]": is_tensor_dict,
        "dict[str, NDArray[Any]]": is_array_dict,
        "dict[str, Any]": lambda v: isinstance(v, dict),
        "list[tuple[str, Any]]": lambda v: (
            isinstance(v, list) and all(isinstance(x, tuple) for x in v)
        ),
        "tuple[Any, ...]": lambda v: isinstance(v, tuple),
        "tuple[FITSFile, list[Any]]": lambda v: isinstance(v, tuple),
    }

    assert set(calls) == set(gen.C_RETURN_TYPES), (
        "every declared return type needs a call in this test"
    )

    wrong: list[str] = []
    for name, declared in sorted(gen.C_RETURN_TYPES.items()):
        check = predicates.get(declared)
        assert check is not None, f"no predicate for declared type {declared!r}"
        value = calls[name]()
        if not check(value):
            wrong.append(
                f"{name}: stub says {declared}, got {type(value).__name__}"
                + (
                    f" of {{{', '.join(sorted({type(x).__name__ for x in value.values()}))}}}"
                    if isinstance(value, dict)
                    else ""
                )
            )

    assert not wrong, "declared return types do not match runtime:\n" + "\n".join(wrong)


def test_core_declared_return_types_are_accurate(fits_inputs: tuple[str, str]) -> None:
    """Every ``_core`` declared return type holds for a real call.

    The metadata module is the one that must never grow a tensor return, so its
    declared types are checked against real values here rather than trusted.
    """
    import torchfits._core as core

    image, table = fits_inputs

    calls: dict[str, object] = {
        "read_header_dict": lambda: core.read_header_dict(table, 1),
        "read_keys": lambda: core.read_keys(image, 0, ["NAXIS"]),
        "read_table_info": lambda: core.read_table_info(table, 1),
        "read_shape": lambda: core.read_shape(image, 0),
        "Metadata.scale_info": lambda: core.Metadata(image).scale_info(0),
        "Metadata.image_info": lambda: core.Metadata(image).image_info(0),
    }

    def is_str_tuple_triples(v: object) -> bool:
        return isinstance(v, list) and all(
            isinstance(x, tuple) and len(x) == 3 and all(isinstance(y, str) for y in x)
            for x in v
        )

    predicates: dict[str, object] = {
        "list[tuple[str, str, str]]": is_str_tuple_triples,
        "dict[str, Any]": lambda v: isinstance(v, dict),
        "tuple[int, tuple[int, ...]]": lambda v: (
            isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], int)
            and isinstance(v[1], tuple)
            and all(isinstance(d, int) for d in v[1])
        ),
        "tuple[bool, bool, float, float]": lambda v: (
            isinstance(v, tuple)
            and len(v) == 4
            and isinstance(v[0], bool)
            and isinstance(v[1], bool)
            and isinstance(v[2], float)
            and isinstance(v[3], float)
        ),
        "tuple[int, int, tuple[int, ...]]": lambda v: (
            isinstance(v, tuple)
            and len(v) == 3
            and isinstance(v[0], int)
            and isinstance(v[1], int)
            and isinstance(v[2], tuple)
            and all(isinstance(d, int) for d in v[2])
        ),
    }

    assert set(calls) == set(gen.CORE_RETURN_TYPES), (
        "every declared core return type needs a call in this test"
    )

    wrong: list[str] = []
    for name, declared in sorted(gen.CORE_RETURN_TYPES.items()):
        check = predicates.get(declared)
        assert check is not None, f"no predicate for declared type {declared!r}"
        if not check(calls[name]()):
            wrong.append(f"_core.{name}: stub says {declared}")

    assert not wrong, "declared core return types do not match runtime:\n" + "\n".join(
        wrong
    )


def test_core_stub_never_declares_a_torch_type() -> None:
    """``_core.pyi`` must not import torch or name a tensor.

    The whole point of the split is that the metadata module does not depend on
    libtorch. A ``torch.Tensor`` in its stub would be the first sign of that
    dependency creeping back in, and it is easier to introduce by accident than
    to notice from a diff. The module docstring is exempt: it *explains* the
    rule and is allowed to spell out what it must not contain.
    """
    core_stub = (ROOT / "src" / "torchfits" / "_core.pyi").read_text(encoding="utf-8")
    quoted = len(core_stub) - len(core_stub.lstrip('"').lstrip("\n"))
    body = core_stub[core_stub.index('"""', 3) + 3 :]
    assert "import torch" not in body
    assert "torch.Tensor" not in body
    assert quoted > 0, "the stub should still open with its generated docstring"


def test_committed_stubs_are_valid_python() -> None:
    """A stub is Python. ``ruff`` is the only thing in the gate that notices.

    Every other step here is string-level: stubgen emits text, ``_rewrite``
    compares and replaces text, and the drift gate is satisfied by two equal
    strings. A syntactically broken stub is therefore perfectly committable --
    and it is the file an IDE and ``mypy`` read, so the damage is that nobody
    gets type checking on the native boundary at all.
    """
    for name in ("_C.pyi", "_core.pyi"):
        path = ROOT / "src" / "torchfits" / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_return_type_rewrite_keeps_a_binding_docstring_attached() -> None:
    """CP-006: a documented def must not gain a second body from the rewrite.

    stubgen closes a def with ``: ...`` when the binding has no docstring, and
    with a bare ``:`` when it does -- the docstring then sits on the indented
    lines underneath. The return-type rewrite re-terminates every def it
    touches, so it has to reproduce whichever form stubgen emitted. It used to
    always append ``: ...``, which turned

        def image_info(self, hdu: int) -> tuple:
            \"\"\"NAXIS order, not row-major.\"\"\"

    into a def ending in ``: ...`` *followed by* that docstring: two bodies,
    and a stub that no longer parses. Nothing caught it, because the drift gate
    only ever compares the broken file against another broken file.

    This went unnoticed for as long as it did because no binding carrying a
    return-type override also had a docstring. Giving the axis-order accessors
    their docstrings is what surfaced it, which is why the case is pinned here
    rather than left to the live extension.
    """
    stubgen_output = '''class Metadata:
    def image_info(self, hdu: int) -> tuple:
        """NAXIS order, not row-major."""

    def num_hdus(self) -> int: ...
'''
    return_types = {"Metadata.image_info": "tuple[int, int, tuple[int, ...]]"}

    out = gen._rewrite(stubgen_output, return_types, {})

    # The documented def keeps the return type override *and* its body.
    assert "-> tuple[int, int, tuple[int, ...]]:" in out
    assert "NAXIS order, not row-major." in out
    # The undocumented def is still rewritten to a bodyless stub.
    assert "def num_hdus(self) -> int: ..." in out
    # And the whole thing is Python.
    ast.parse(out)
