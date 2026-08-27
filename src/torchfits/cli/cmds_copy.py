"""``torchfits copy`` — byte-identical FITS copy (not an HDU rewrite)."""

from __future__ import annotations

import argparse
import shutil
import urllib.request

from torchfits._io_engine.paths import cfitsio_base_path, guard_fits_path
from torchfits.http_util import HttpBlockedError, http_open

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    is_remote_path,
    reject_same_path,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "copy",
        help="byte-copy FITS file(s)",
        description=(
            "Exact binary copy of FITS file bytes (preserves CompImage tiles). "
            "Multiple inputs need --out-dir; -J fans out across files."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="INPUT [OUTPUT], or multiple INPUTs with --out-dir",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when copying multiple inputs",
    )
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _copy_remote(src: str, output_path: str) -> None:
    lowered = src.lower()
    if lowered.startswith(("vos:", "vault:")):
        raise IoError(f"{src}: vos/vault remote copy is not supported")
    if lowered.startswith(("http://", "https://")):
        with http_open(src) as response, open(output_path, "wb") as dest:
            shutil.copyfileobj(response, dest)
        return
    urllib.request.urlretrieve(src, output_path)


def _copy_one(pair: tuple[str, str]) -> None:
    input_path, output_path = pair
    reject_same_path(input_path, output_path)
    src = cfitsio_base_path(input_path)
    try:
        guard_fits_path(src)
        if is_remote_path(src):
            _copy_remote(src, output_path)
        else:
            shutil.copy2(src, output_path)
    except (UsageError, IoError):
        raise
    except HttpBlockedError as exc:
        raise IoError(str(exc)) from exc
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    pairs = resolve_batch_io_pairs(
        [str(p) for p in args.paths],
        out=args.out,
        out_dir=args.out_dir,
        refuse_same_path=True,
    )
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(pairs))
    run_file_jobs(pairs, _copy_one, file_jobs)
    return EXIT_OK
