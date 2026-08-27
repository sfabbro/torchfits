"""``torchfits cutout`` — read a pixel subset and write FITS."""

from __future__ import annotations

import argparse

import torchfits
from torchfits._io_engine.paths import has_cfitsio_filter

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "cutout",
        help="extract a pixel box to a FITS file",
        description=(
            "Pixel box extraction. Multiple inputs need --out-dir and a shared "
            "--box (or each path may carry its own CFITSIO section); -J fans out."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "INPUT [OUTPUT], or multiple INPUTs with --out-dir; "
            "CFITSIO image section OK on each path"
        ),
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when cutting multiple inputs",
    )
    add_hdu_arg(parser, type=int, default=0, help="source HDU index")
    parser.add_argument(
        "--box",
        default=None,
        help="alt pixel box x1,y1,x2,y2 (0-based half-open); "
        "omit when the path already has a CFITSIO section",
    )
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _parse_box(raw: str) -> tuple[int, int, int, int]:
    parts = [piece.strip() for piece in raw.split(",")]
    if len(parts) != 4:
        raise UsageError("--box requires x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise UsageError("--box values must be integers") from exc
    return x1, y1, x2, y2


def _cutout_one(
    pair: tuple[str, str],
    *,
    hdu: int,
    box: str | None,
) -> None:
    input_path, output_path = pair
    sectioned = has_cfitsio_filter(input_path)
    if sectioned and box is not None:
        raise UsageError(
            "pass either a CFITSIO image section in the path or --box, not both"
        )
    if not sectioned and box is None:
        raise UsageError(
            "cutout needs a CFITSIO section "
            "(e.g. image.fits[10:100,20:200]) or --box x1,y1,x2,y2"
        )
    try:
        if sectioned:
            tensor = torchfits.read_tensor(input_path, hdu=hdu)
        else:
            assert box is not None  # guarded above
            x1, y1, x2, y2 = _parse_box(box)
            tensor = torchfits.read_subset(input_path, hdu, x1, y1, x2, y2)
        header = torchfits.read_header(input_path, hdu)
        torchfits.write_tensor(output_path, tensor, header=header, overwrite=True)
    except UsageError:
        raise
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
    run_file_jobs(
        pairs,
        lambda pair: _cutout_one(pair, hdu=int(args.hdu), box=args.box),
        file_jobs,
    )
    return EXIT_OK
