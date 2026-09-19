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
    """The committed stub is exactly what the generator produces today."""
    got = gen.TARGET.read_text(encoding="utf-8") if gen.TARGET.is_file() else ""
    fresh = gen.build()
    assert fresh == got, (
        f"{gen.TARGET.relative_to(ROOT)} is stale — run "
        "'pixi run python scripts/gen_native_stub.py'\n" + gen.diff_summary(fresh, got)
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

    assert set(calls) == set(gen.RETURN_TYPES), (
        "every declared return type needs a call in this test"
    )

    wrong: list[str] = []
    for name, declared in sorted(gen.RETURN_TYPES.items()):
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
