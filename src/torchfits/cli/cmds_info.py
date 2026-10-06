"""``torchfits info`` — HDU inventory."""

from __future__ import annotations

import argparse
from typing import Any

from .common import (
    EXIT_OK,
    add_emit_format_args,
    add_hdu_arg,
    compressed_image_geometry,
    emit_records,
    header_extname,
    hdu_type_name,
    iter_file_hdu_pairs,
    resolve_emit_format,
    resolve_paths,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("info", help="list HDUs in FITS files")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


def _info_record(path: str, index: int, hdu: Any) -> dict[str, Any]:
    header = hdu.header
    record: dict[str, Any] = {
        "file": path,
        "hdu": index,
        "name": header_extname(header, index),
        "type": hdu_type_name(header, hdu),
    }
    if record["type"] == "IMAGE":
        # A compressed image stores its geometry in ZNAXIS*/ZBITPIX; the
        # plain NAXIS*/BITPIX (and therefore shape_str/dtype_str) describe
        # the tile table, so prefer the Z cards when they are present.
        geometry = compressed_image_geometry(header)
        if geometry is not None:
            record["shape"] = str(geometry[0])
            record["dtype"] = geometry[1]
        else:
            shape = getattr(hdu, "shape_str", None)
            dtype = getattr(hdu, "dtype_str", None)
            if shape is not None:
                record["shape"] = shape
            if dtype is not None:
                record["dtype"] = dtype
    elif record["type"] == "TABLE":
        nrows = header.get("NAXIS2")
        if nrows is not None:
            record["nrows"] = int(nrows)
        tfields = header.get("TFIELDS")
        if tfields is not None:
            record["ncols"] = int(tfields)
    return record


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records = [
        _info_record(path, index, hdu)
        for path, index, hdu in iter_file_hdu_pairs(paths, args.hdu)
    ]
    emit_records(records, format=resolve_emit_format(args))
    return EXIT_OK
