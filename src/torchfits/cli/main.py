"""``torchfits`` argparse dispatch."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from .cmds_arith import add_parser as add_arith
from .cmds_compress import add_compress_parser as add_compress
from .cmds_compress import add_decompress_parser as add_decompress
from .cmds_convert import add_parser as add_convert
from .cmds_copy import add_parser as add_copy
from .cmds_cutout import add_parser as add_cutout
from .cmds_diff import add_parser as add_diff
from .cmds_header import add_parser as add_header
from .cmds_info import add_parser as add_info
from .cmds_probe import add_parser as add_probe
from .cmds_setkey import add_parser as add_setkey
from .cmds_stats import add_parser as add_stats
from .cmds_table import add_parser as add_table
from .cmds_transform import add_parser as add_transform
from .cmds_verify import add_parser as add_verify
from .common import CliError, EXIT_INTERRUPT, EXIT_IO, EXIT_OK

_SUBCOMMANDS: tuple[tuple[str, Callable[..., None], str], ...] = (
    ("info", add_info, "HDU inventory"),
    ("header", add_header, "dump headers (all HDUs)"),
    ("verify", add_verify, "verify checksum keywords"),
    ("diff", add_diff, "compare FITS files"),
    ("stats", add_stats, "image statistics"),
    ("table", add_table, "table schema/preview"),
    ("convert", add_convert, "convert to parquet/csv/tsv/arrow or PNG"),
    ("copy", add_copy, "byte-copy FITS file(s)"),
    ("arith", add_arith, "image ±×÷ scalar or image"),
    ("cutout", add_cutout, "pixel cutout"),
    ("compress", add_compress, "tile-compress (--out-dir / --split hdu)"),
    ("decompress", add_decompress, "decompress (--out-dir / --split hdu)"),
    ("transform", add_transform, "apply transforms"),
    ("probe", add_probe, "probe files or URLs"),
    ("setkey", add_setkey, "set header keyword"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchfits", description="FITS I/O CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for _name, add_fn, _help in _SUBCOMMANDS:
        add_fn(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except CliError as exc:
        print(exc, file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPT
    except BrokenPipeError:
        return EXIT_OK
    except OSError as exc:
        print(exc, file=sys.stderr)
        return EXIT_IO


if __name__ == "__main__":
    raise SystemExit(main())
