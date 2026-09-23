"""Exit-matrix: every CLI command's exit paths conform to ``docs/cli.md``.

The docs "Exit Codes" table is the contract being pinned:

===========  =================================================================
0            success
1            difference found (``diff``)
2            usage error (missing arguments, invalid flags, unknown syntax)
3            I/O error (missing file, invalid FITS structure)
4            checksum verification failure (``verify``)
5            internal error (unexpected exception; traceback printed to stderr)
130          interrupted (``KeyboardInterrupt`` / Ctrl-C), never 2
===========  =================================================================

Rows are parametrized over every subcommand of ``torchfits.cli.main``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torchfits
from torchfits.cli.main import main

_COMMANDS = (
    "info",
    "header",
    "verify",
    "diff",
    "stats",
    "table",
    "convert",
    "copy",
    "arith",
    "cutout",
    "compress",
    "decompress",
    "transform",
    "probe",
    "setkey",
)

# ``set_defaults(func=...)`` targets to monkeypatch for the internal-error and
# interrupt rows (one entry point per subcommand).
_RUN_ATTR = {
    "compress": ("torchfits.cli.cmds_compress", "run_compress"),
    "decompress": ("torchfits.cli.cmds_compress", "run_decompress"),
}


def _run_attr(cmd: str) -> tuple[str, str]:
    return _RUN_ATTR.get(cmd, (f"torchfits.cli.cmds_{cmd}", "run"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def matrix_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("exit_matrix")
    img = root / "image.fits"
    torchfits.write(
        str(img),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"BITPIX": -32},
        overwrite=True,
    )
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    tbl = root / "table.fits"
    data = {
        "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
        "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    fits.BinTableHDU(Table(data), name="CAT").writeto(str(tbl), overwrite=True)

    fz = root / "image.fits.fz"
    result = _run_cli("compress", str(img), str(fz))
    assert result.returncode == 0, result.stderr

    return {
        "root": root,
        "img": str(img),
        "tbl": str(tbl),
        "fz": str(fz),
        "missing": str(root / "missing.fits"),
        "missing2": str(root / "missing2.fits"),
    }


@pytest.fixture
def cases(matrix_env):
    """Per-command ``(ok, usage, io)`` argv rows; ``ok`` must exit 0."""
    img = matrix_env["img"]
    tbl = matrix_env["tbl"]
    fz = matrix_env["fz"]
    missing = matrix_env["missing"]
    missing2 = matrix_env["missing2"]
    root = Path(matrix_env["root"])
    return {
        "info": (
            ["info", img],
            ["info", "-e", "notanint", img],
            ["info", missing],
        ),
        "header": (
            ["header", img, "-k", "BITPIX"],
            ["header", img, "--keyword-table"],
            ["header", missing],
        ),
        "verify": (
            ["verify", img],
            ["verify", "-e", "notanint", img],
            ["verify", missing],
        ),
        "diff": (
            ["diff", img, img],
            ["diff", img],
            ["diff", missing, missing2],
        ),
        "stats": (
            ["stats", img, "-e", "0"],
            ["stats", "-e", "notanint", img],
            ["stats", missing],
        ),
        "table": (
            ["table", tbl, "-e", "1"],
            ["table", "-e", "notanint", tbl],
            ["table", missing],
        ),
        "convert": (
            ["convert", tbl, str(root / "t.csv"), "-e", "1"],
            ["convert", tbl, str(root / "t.dat")],
            ["convert", missing, str(root / "t.csv")],
        ),
        "copy": (
            ["copy", img, str(root / "copy.fits")],
            ["copy", img, img],
            ["copy", missing, str(root / "copy.fits")],
        ),
        "arith": (
            ["arith", img, "--op", "add", "--value", "1.0", "-o", str(root / "a.fits")],
            ["arith", img, "--op", "add", "-o", str(root / "a.fits")],
            [
                "arith",
                missing,
                "--op",
                "add",
                "--value",
                "1.0",
                "-o",
                str(root / "a.fits"),
            ],
        ),
        "cutout": (
            ["cutout", img, "-o", str(root / "c.fits"), "-e", "0", "--box", "0,0,2,2"],
            ["cutout", img, "-o", str(root / "c.fits"), "--box", "bad"],
            ["cutout", missing, "-o", str(root / "c.fits"), "--box", "0,0,2,2"],
        ),
        "compress": (
            ["compress", img, str(root / "c.fits.fz")],
            ["compress", img, str(root / "c2.fits.fz"), "--split", "hdu"],
            ["compress", missing, str(root / "c.fits.fz")],
        ),
        "decompress": (
            ["decompress", fz, str(root / "d.fits")],
            ["decompress", fz, str(root / "d2.fits"), "--split", "hdu"],
            ["decompress", missing, str(root / "d.fits")],
        ),
        "transform": (
            ["transform", img, "--name", "ArcsinhStretch", "-o", str(root / "t.fits")],
            ["transform", img, "--name", "NoSuchTransform", "-o", str(root / "t.fits")],
            [
                "transform",
                missing,
                "--name",
                "ArcsinhStretch",
                "-o",
                str(root / "t.fits"),
            ],
        ),
        "probe": (
            ["probe", img],
            ["probe", "-e", "notanint", img],
            ["probe", missing],
        ),
        "setkey": (
            ["setkey", str(root / "setkey_target.fits"), "-k", "FOO", "--value", "1"],
            ["setkey", str(root / "setkey_target.fits")],
            ["setkey", missing, "-k", "FOO", "--value", "1"],
        ),
    }


@pytest.fixture(autouse=True)
def _setkey_target(matrix_env):
    """``setkey`` rewrites its input in place; give each test a private copy."""
    import shutil

    target = Path(matrix_env["root"]) / "setkey_target.fits"
    shutil.copy2(matrix_env["img"], target)
    return target


def test_matrix_covers_every_subcommand(cases):
    """Coverage guard: one row-triple per dispatch target, none added silently."""
    from torchfits.cli.main import _SUBCOMMANDS

    assert set(cases) == set(_COMMANDS)
    assert {name for name, _add, _help in _SUBCOMMANDS} == set(_COMMANDS)


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_matrix_success_exit_0(cmd, cases):
    argv, _usage, _io = cases[cmd]
    result = _run_cli(*argv)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_matrix_usage_error_exit_2(cmd, cases):
    _argv, usage, _io = cases[cmd]
    result = _run_cli(*usage)
    assert result.returncode == 2, result.stderr


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_matrix_io_error_exit_3(cmd, cases):
    _argv, _usage, io_argv = cases[cmd]
    result = _run_cli(*io_argv)
    assert result.returncode == 3, result.stderr


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_matrix_internal_error_exit_5(cmd, cases, monkeypatch, capsys):
    """An unexpected non-CliError exception: traceback to stderr, exit 5."""
    module_name, attr = _run_attr(cmd)

    def _boom(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(f"{module_name}.{attr}", _boom)
    argv, _usage, _io = cases[cmd]
    rc = main(argv)
    assert rc == 5
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "RuntimeError: boom" in err


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_matrix_interrupt_exit_130(cmd, cases, monkeypatch):
    """``KeyboardInterrupt`` exits exactly 130 — never 2 (usage) or any other code."""
    module_name, attr = _run_attr(cmd)

    def _interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(f"{module_name}.{attr}", _interrupt)
    argv, _usage, _io = cases[cmd]
    rc = main(argv)
    assert rc == 130
    assert rc != 2


def test_diff_difference_exit_1(matrix_env, tmp_path):
    other = tmp_path / "other.fits"
    torchfits.write(
        str(other),
        torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1.0,
        header={"BITPIX": -32},
        overwrite=True,
    )
    result = _run_cli("diff", matrix_env["img"], str(other))
    assert result.returncode == 1, result.stderr


def test_verify_checksum_failure_exit_4(tmp_path):
    path = tmp_path / "verify.fits"
    torchfits.write(
        str(path),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"FOO": 1},
        overwrite=True,
    )
    torchfits.write_checksums(str(path), hdu=0)
    with open(path, "r+b") as f:
        raw = f.read()
        idx = raw.find(b"FOO     =")
        assert idx != -1
        card = bytearray(raw[idx : idx + 80])
        one = card.find(b"1")
        assert one != -1
        card[one : one + 1] = b"2"
        f.seek(idx)
        f.write(card)
    result = _run_cli("verify", str(path))
    assert result.returncode == 4, result.stderr


def test_argparse_usage_errors_exit_2(matrix_env):
    result = _run_cli()  # no subcommand
    assert result.returncode == 2, result.stderr
    result = _run_cli("info", "--bogus-flag")  # unknown flag
    assert result.returncode == 2, result.stderr


def test_real_sigint_exit_130(matrix_env):
    """Real SIGINT through the real signal handler: exit 130, never 2."""
    code = (
        "import os, signal, sys\n"
        "import torchfits.cli.cmds_info as ci\n"
        "def _interrupt(_args):\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"
        "    return 0\n"
        "ci.run = _interrupt\n"
        "from torchfits.cli.main import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "info", matrix_env["img"]],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 130, result.stderr
    assert result.returncode != 2
