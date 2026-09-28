"""Contract tests for the torch import boundary.

The rule this file enforces: **PyTorch is loaded at exactly one boundary — the
first call whose documented return type is a ``torch.Tensor``, or that takes
``device=``.**  Nothing before it: not ``import torchfits.hdu``, not
``read_header``, not ``torchfits info``.

Two mechanisms, both needed:

* Every assertion runs in a **fresh interpreter**, because an in-process
  ``sys.modules`` check is poisoned by whatever the test session already
  imported.
* Metadata statements additionally run with a **meta-path blocker** that makes
  ``import torch`` raise outright.  Checking ``sys.modules`` alone would pass
  for code that imports torch lazily on some path; the blocker turns that into
  a visible failure and also proves the statement survives torch being
  unimportable at all.

Arrow table transport and metadata CLI statements are part of this contract,
not optional probes. They run in fresh interpreters with an import blocker so
an eager import, lazy import, or tensor-returning native call cannot pass by
merely inspecting ``sys.modules``.

The native metadata half (``libtorchfits_core``, exposed as ``torchfits._core``)
gets its own cases for a second reason: it is the only part of the extension
that can be checked for a *link-level* libtorch dependency. ``_C`` links
libtorch by design; the core is what makes ``read_header`` cheap, and that only
holds if the library has no libtorch in its dependency list.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import torchfits

# Statements are module-level so the phase markers read as a checklist.
PHASE_IMPORT_HYGIENE = (
    "import torchfits.hdu",
    "import torchfits.io",
)

# The torch-free native core. `libtorchfits_core` links CFITSIO and nothing
# else, so these must survive torch being unimportable -- including the read
# paths, not just the import. If any of them ever needs a tensor, the core has
# grown a libtorch dependency and the split is no longer a split.
PHASE_CORE_LIBRARY = (
    "import torchfits._core; torchfits._core.thread_count()",
    "import torchfits._core; torchfits._core.read_num_hdus(__FITS__)",
    "import torchfits._core; torchfits._core.read_hdu_type(__FITS__, 1)",
    "import torchfits._core; torchfits._core.read_nrows(__FITS__, 1)",
    "import torchfits._core; torchfits._core.read_colnames(__FITS__, 1)",
    "import torchfits._core; torchfits._core.read_table_info(__FITS__, 1)",
    "import torchfits._core; torchfits._core.read_keys(__FITS__, 1, ['NAXIS2'])",
    "import torchfits._core; torchfits._core.read_shape(__IMG__, 0)",
    "import torchfits._core; len(torchfits._core.read_header_dict(__FITS__, 1))",
    "import torchfits._core; len(torchfits._core.read_header_string(__FITS__, 1))",
    (
        "import torchfits._core\n"
        "with torchfits._core.Metadata(__FITS__) as md:\n"
        "    assert md.num_hdus() == 2 and md.hdu_type(1) == 'BINARY_TABLE'\n"
        "    assert md.nrows(1) == 3 and md.colnames(1)[0] == 'RA'\n"
        "    assert md.header(1) and md.image_info(0)[0] == 16\n"
        "with torchfits._core.Metadata(__IMG__) as md:\n"
        "    assert md.shape(0) == [4, 4] and md.bitpix(0) == -32\n"
        "    assert md.header_text(0) and md.keywords(0, ['NAXIS1'])['NAXIS1'] == 4\n"
    ),
)

PHASE_CORE_MODULE = (
    "import torchfits; torchfits.read_header(__FITS__, 1)",
    "import torchfits; torchfits.read_keys(__FITS__, ['NAXIS2'], 1)",
    "import torchfits; torchfits.read_colnames(__FITS__, 1)",
    "import torchfits; torchfits.read_extname(__FITS__, 1)",
    "import torchfits; torchfits.read_hdu_type(__FITS__, 1)",
    "import torchfits; torchfits.read_num_hdus(__FITS__)",
    "import torchfits; torchfits.read_nrows(__FITS__, 1)",
    "import torchfits; torchfits.read_shape(__IMG__, 0)",
    "import torchfits; torchfits.read_table_info(__FITS__, 1)",
    "import torchfits; torchfits.verify_checksums(__CHK__, hdu=0)",
    "import torchfits; torchfits.read_batch_info([__FITS__])",
    "import torchfits; hdul = torchfits.open(__FITS__); hdul[1].header",
)

PHASE_TABLE_TRANSPORT = (
    "import torchfits.table; torchfits.table.read(__FITS__, 1)",
    "import torchfits.table; torchfits.table.schema(__FITS__, hdu=1)",
    "import torchfits.table; next(torchfits.table.scan(__FITS__, 1))",
    # A `where=` filter has a torch-accelerated fast path that is tried first
    # and is supposed to return None (letting the Arrow filter take over) when
    # it does not apply. Importing torch unguarded there turned a supported
    # filtered Arrow read into an ImportError in a torch-free environment.
    (
        "import torchfits.table\n"
        "t = torchfits.table.read(__FITS__, 1, where='MAG < 20.0')\n"
        "assert t.num_rows == 1, t.num_rows\n"
        "assert t.column('MAG').to_pylist() == [19.5]\n"
    ),
)

PHASE_TABLE_RICH_TRANSPORT = (
    "import torchfits.table; torchfits.table.read(__RICH__, 1)",
)

# The converse contract: these *must* pay for torch, and must keep working.
TENSOR_STATEMENTS = (
    "import torchfits; torchfits.read_tensor(__IMG__, 0)",
    "import torchfits; torchfits.read(__IMG__)",
    "import torchfits.table; torchfits.table.read_torch(__FITS__, 1)",
    "import torchfits.transforms",
    "import torchfits.data",
)

# command -> (fixture key, argv builder).  Every argv must exit 0 so the
# command's metadata calls are actually reached.
_CLI_METADATA_COMMANDS: dict[str, tuple[str, Callable[[Path], list[str]]]] = {
    "info": ("image", lambda p: ["info", str(p)]),
    "header": ("image", lambda p: ["header", str(p)]),
    "probe": ("image", lambda p: ["probe", str(p)]),
    "verify": ("image", lambda p: ["verify", str(p)]),
    "table": ("table", lambda p: ["table", str(p)]),
    "copy": (
        "table",
        lambda p: ["copy", str(p), "-o", str(p.with_name("copied.fits"))],
    ),
    "setkey": (
        "table",
        lambda p: ["setkey", str(p), "--key", "TESTKEY", "--value", "7"],
    ),
}

_CLI_PIXEL_COMMANDS: dict[str, tuple[str, Callable[[Path], list[str]]]] = {
    "stats": ("image", lambda p: ["stats", str(p)]),
}

_BLOCK_TORCH = """\
import importlib.abc
import sys


class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(
                "torch is blocked by the torch-boundary contract test: " + fullname
            )
        return None


sys.meta_path.insert(0, _BlockTorch())
"""

_PROBE_TEMPLATE = """\
import sys
import time

_t0 = time.perf_counter()
__PRELUDE__
__STATEMENT__
_elapsed_ms = (time.perf_counter() - _t0) * 1000.0
print("PROBE", int("torch" in sys.modules), int(round(_elapsed_ms)))
"""


@dataclass(frozen=True)
class Probe:
    """Outcome of running one statement in a fresh interpreter."""

    torch_loaded: bool
    elapsed_ms: float


def _child_env() -> dict[str, str]:
    """Environment for a probe: torchfits knobs must not leak in from a dev shell."""
    return {k: v for k, v in os.environ.items() if not k.startswith("TORCHFITS_")}


def _run_probe(statement: str, *, block_torch: bool) -> Probe:
    prelude = _BLOCK_TORCH if block_torch else ""
    script = _PROBE_TEMPLATE.replace("__PRELUDE__", prelude).replace(
        "__STATEMENT__", statement
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"probe exited {proc.returncode}: {statement}\n{proc.stderr.strip()}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE ")]
    if not lines:
        pytest.fail(f"probe produced no result: {statement}\n{proc.stdout}")
    _, loaded, elapsed = lines[-1].split()
    return Probe(torch_loaded=loaded == "1", elapsed_ms=float(elapsed))


@pytest.fixture(scope="module")
def fits_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A checksummed image plus a binary table, for the probes to read.

    Both are generated here rather than checked in: ``*.fits`` is gitignored in
    this repository, so a fixture file would exist only for whoever ran a test
    that happened to write it.
    """
    root = tmp_path_factory.mktemp("torch-boundary")
    image = root / "image.fits"
    torchfits.write(
        str(image),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"BITPIX": -32},
        overwrite=True,
        checksum=True,
    )
    table = root / "table.fits"
    torchfits.table.write(
        str(table),
        {
            "RA": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "MAG": torch.tensor([19.5, 20.5, 21.5], dtype=torch.float32),
            "N": torch.tensor([1, 2, 3], dtype=torch.int32),
        },
        overwrite=True,
        extname="MY_TABLE",
    )
    rich = root / "rich-table.fits"
    names = torch.tensor(
        [
            [ord("A"), ord("L"), ord("P"), ord("H"), ord("A"), 0, 0, 0],
            [ord("B"), ord("E"), ord("T"), ord("A"), 0, 0, 0, 0],
            [ord("G"), ord("A"), ord("M"), ord("M"), ord("A"), 0, 0, 0],
        ],
        dtype=torch.uint8,
    )
    torchfits.table.write(
        str(rich),
        {
            "NAME": names,
            "V": [
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([], dtype=torch.int32),
                torch.tensor([3, 4, 5], dtype=torch.int32),
            ],
        },
        overwrite=True,
        extname="RICH_TABLE",
    )
    return {"root": root, "image": image, "table": table, "rich": rich}


def _substitute(statement: str, files: dict[str, Path]) -> str:
    return (
        statement.replace("__FITS__", repr(str(files["table"])))
        .replace("__RICH__", repr(str(files["rich"])))
        .replace("__IMG__", repr(str(files["image"])))
        .replace("__CHK__", repr(str(files["image"])))
    )


def _assert_torch_free(statement: str) -> None:
    probe = _run_probe(statement, block_torch=True)
    assert not probe.torch_loaded, f"torch was imported by: {statement}"


def _label(statement: str) -> str:
    tail = statement.split(";")[-1].strip()
    return tail or "import"


@pytest.mark.parametrize("statement", PHASE_IMPORT_HYGIENE, ids=_label)
def test_import_hygiene_is_torch_free(statement: str) -> None:
    """Phase 1: the HDU/IO import graphs carry no torch."""
    _assert_torch_free(statement)


def test_native_extension_import_is_torch_free() -> None:
    """The native metadata extension must not initialize Python torch."""
    _assert_torch_free("import torchfits._C")


@pytest.mark.parametrize("statement", PHASE_CORE_LIBRARY, ids=_label)
def test_torch_free_core_library_is_usable(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """The metadata core must do real work with torch unimportable.

    Not just "imports cleanly": each case performs the query it is named after,
    so a core that quietly fell back to the torch-linked extension (or grew a
    libtorch dependency of its own) fails here rather than in production.
    """
    _assert_torch_free(_substitute(statement, fits_files))


def test_core_library_does_not_link_libtorch() -> None:
    """``libtorchfits_core`` must not have libtorch in its dependency list.

    ``sys.modules`` proves the *Python* ``torch`` module was not imported, which
    is what the other cases check. It cannot see a libtorch that the dynamic
    loader mapped into the process: linking libtorch costs ~1 s of relocation
    and a few hundred MB of RSS even when nothing calls it. Read the platform's
    own dependency listing, which is the only thing that can see it.

    macOS and Linux are the two platforms torchfits ships wheels for, so this
    is a hard requirement there rather than a best-effort probe.
    """
    if sys.platform == "darwin":
        tool = ["otool", "-L"]
    elif sys.platform.startswith("linux"):
        tool = ["ldd"]
    else:
        pytest.skip(f"no dependency lister for {sys.platform!r}")

    import torchfits._core as core

    core_lib = Path(core.__file__).resolve().parent / _core_library_name()
    assert core_lib.is_file(), f"libtorchfits_core not found next to {core.__file__}"

    proc = subprocess.run(
        [*tool, str(core_lib)], capture_output=True, text=True, check=True
    )
    # `otool -L` prints the inspected path first and the dylib's own install
    # name second; `ldd` prints only dependencies. Drop anything that names the
    # core itself -- its path contains "torchfits", which is not a libtorch
    # dependency -- and match library *names* on what is left.
    lines = [
        ln.strip()
        for ln in proc.stdout.splitlines()[1:]
        if ln.strip() and "libtorchfits_core" not in ln
    ]
    offenders = [
        line for line in lines if re.search(r"lib(torch|c10|c10_cuda|caffe2|omp)", line)
    ]
    assert not offenders, (
        f"{core_lib.name} links libtorch: {offenders}. The metadata core exists so "
        "importing it does not pull libtorch into the process."
    )


def _core_library_name() -> str:
    """Platform-specific file name of ``libtorchfits_core``."""
    if sys.platform == "darwin":
        return "libtorchfits_core.dylib"
    if sys.platform.startswith("linux"):
        return "libtorchfits_core.so"
    if os.name == "nt":  # pragma: no cover - torchfits ships no Windows wheels
        return "torchfits_core.dll"
    raise AssertionError(f"unknown core library name for {sys.platform!r}")


def test_core_build_ids_agree() -> None:
    """The module, the library it loaded, and _C must be one build.

    A half-rebuilt checkout can leave a fresh ``_C``/``_core`` beside a stale
    ``libtorchfits_core``. Every compile-time check accepts that; only the
    runtime comparison catches it, and the consequence of not catching it is a
    struct-layout mismatch inside a C library.
    """
    import torchfits._core as core

    library_id = core.core_library_build_id()
    assert core.__build_id__ == library_id, (
        "torchfits._core and the libtorchfits_core it loaded disagree: "
        f"{core.__build_id__!r} vs {library_id!r}"
    )
    assert core.TORCH_FREE is True


@pytest.mark.parametrize("statement", PHASE_CORE_MODULE, ids=_label)
def test_metadata_calls_are_torch_free(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """Phase 2: metadata calls must not load torch."""
    _assert_torch_free(_substitute(statement, fits_files))


@pytest.mark.parametrize("statement", PHASE_TABLE_TRANSPORT, ids=_label)
def test_arrow_table_reads_are_torch_free(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """Phase 3: Arrow destinations must not load torch."""
    _assert_torch_free(_substitute(statement, fits_files))


@pytest.mark.parametrize("statement", PHASE_TABLE_RICH_TRANSPORT, ids=_label)
def test_raw_string_and_vla_transport_is_torch_free(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """String matrices and VLA offsets use the same torch-free native path."""
    _assert_torch_free(_substitute(statement, fits_files))


@pytest.mark.parametrize("statement", TENSOR_STATEMENTS, ids=_label)
def test_tensor_entry_points_load_torch(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """The converse contract: a tensor destination must load torch."""
    probe = _run_probe(_substitute(statement, files=fits_files), block_torch=False)
    assert probe.torch_loaded, f"torch was not loaded by: {statement}"


def _cli_probe_statement(command: str, target: Path) -> str:
    _, build_argv = {**_CLI_METADATA_COMMANDS, **_CLI_PIXEL_COMMANDS}[command]
    argv = build_argv(target)
    return (
        "from torchfits.cli.main import main\n"
        f"assert main({argv!r}) == 0, 'torchfits {command} did not exit 0'"
    )


@pytest.mark.parametrize(
    "command", list(_CLI_METADATA_COMMANDS), ids=list(_CLI_METADATA_COMMANDS)
)
def test_metadata_cli_commands_are_torch_free(
    command: str, fits_files: dict[str, Path]
) -> None:
    """Phase 2: metadata CLI commands must not pay for torch."""
    fixture, _ = _CLI_METADATA_COMMANDS[command]
    target = fits_files["root"] / f"{command}-target.fits"
    target.write_bytes(fits_files[fixture].read_bytes())
    probe = _run_probe(_cli_probe_statement(command, target), block_torch=True)
    assert not probe.torch_loaded, f"torch was imported by: torchfits {command}"


@pytest.mark.parametrize(
    "command", list(_CLI_PIXEL_COMMANDS), ids=list(_CLI_PIXEL_COMMANDS)
)
def test_pixel_cli_commands_load_torch(
    command: str, fits_files: dict[str, Path]
) -> None:
    """Pixel-math commands legitimately need torch; they must keep working."""
    fixture, _ = _CLI_PIXEL_COMMANDS[command]
    target = fits_files["root"] / f"{command}-pixel.fits"
    target.write_bytes(fits_files[fixture].read_bytes())
    probe = _run_probe(_cli_probe_statement(command, target), block_torch=False)
    assert probe.torch_loaded, f"torch was not loaded by: torchfits {command}"
