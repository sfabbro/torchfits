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

Statements the torch-free-core plan has not delivered yet carry
``xfail(strict=True)`` naming the phase that will deliver them.  ``strict=True``
makes an unexpected pass a *failure*, so finishing a phase turns its markers red
until they are deleted — the marker is the todo entry.  Deleting a marker is
therefore part of implementing its phase, not optional cleanup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import torchfits

TABLE_FITS = Path(__file__).resolve().parent / "table_example.fits"

# Statements are module-level so the phase markers read as a checklist.
PHASE_IMPORT_HYGIENE = (
    "import torchfits.hdu",
    "import torchfits.io",
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
    """A checksummed image plus a binary table, for the probes to read."""
    root = tmp_path_factory.mktemp("torch-boundary")
    image = root / "image.fits"
    torchfits.write(
        str(image),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"BITPIX": -32},
        overwrite=True,
        checksum=True,
    )
    assert TABLE_FITS.is_file(), f"missing fixture table: {TABLE_FITS}"
    return {"root": root, "image": image, "table": TABLE_FITS}


def _substitute(statement: str, files: dict[str, Path]) -> str:
    return (
        statement.replace("__FITS__", repr(str(files["table"])))
        .replace("__IMG__", repr(str(files["image"])))
        .replace("__CHK__", repr(str(files["image"])))
    )


def _assert_torch_free(statement: str) -> None:
    probe = _run_probe(statement, block_torch=True)
    assert not probe.torch_loaded, f"torch was imported by: {statement}"


def _pending(statements: Iterable[str], phase: str) -> list[object]:
    """Mark statements the named phase has not delivered yet."""
    return [
        pytest.param(
            statement,
            marks=pytest.mark.xfail(strict=True, reason=f"delivered by {phase}"),
        )
        for statement in statements
    ]


def _label(statement: str) -> str:
    tail = statement.split(";")[-1].strip()
    return tail or "import"


@pytest.mark.parametrize(
    "statement", _pending(PHASE_IMPORT_HYGIENE, "Phase 1"), ids=_label
)
def test_import_hygiene_is_torch_free(statement: str) -> None:
    """Phase 1: the HDU/IO import graphs carry no torch."""
    _assert_torch_free(statement)


@pytest.mark.parametrize(
    "statement", _pending(PHASE_CORE_MODULE, "Phase 2"), ids=_label
)
def test_metadata_calls_are_torch_free(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """Phase 2: metadata calls must not load torch."""
    _assert_torch_free(_substitute(statement, fits_files))


@pytest.mark.parametrize(
    "statement", _pending(PHASE_TABLE_TRANSPORT, "Phase 3"), ids=_label
)
def test_arrow_table_reads_are_torch_free(
    statement: str, fits_files: dict[str, Path]
) -> None:
    """Phase 3: Arrow destinations must not load torch."""
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
@pytest.mark.xfail(strict=True, reason="delivered by Phase 2")
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
