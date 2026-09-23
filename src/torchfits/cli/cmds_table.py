"""``torchfits table`` — table schema and preview."""

from __future__ import annotations

import argparse
from typing import Any

from torchfits import table as tf_table

import json

import torchfits

from .common import (
    EXIT_OK,
    CliError,
    IoError,
    _json_safe,
    add_emit_format_args,
    add_hdu_arg,
    emit_records,
    header_extname,
    hdu_type_name,
    json_default,
    resolve_emit_format,
    resolve_paths,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("table", help="table schema and preview rows")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser, help="comma-separated table HDU indices (default: all)")
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=5,
        dest="preview",
        help="preview row count (default: 5)",
    )
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


def _schema_fields(schema: Any) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _preview_rows(path: str, hdu: int, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    arrow_table = tf_table.read(path, hdu=hdu, row_slice=slice(0, limit))
    rows: list[dict[str, Any]] = []
    for row_idx in range(arrow_table.num_rows):
        row: dict[str, Any] = {}
        for name in arrow_table.column_names:
            value = arrow_table.column(name)[row_idx].as_py()
            row[name] = value
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            with torchfits.open(path) as hdul:
                indices = selected_hdu_indices(len(hdul), args.hdu)
                for index in indices:
                    hdu = hdul[index]
                    header = hdu.header
                    if hdu_type_name(header, hdu) != "TABLE":
                        continue
                    schema = tf_table.schema(path, hdu=index)
                    record: dict[str, Any] = {
                        "file": path,
                        "hdu": index,
                        "name": header_extname(header, index),
                        "schema": _schema_fields(schema),
                        "nrows": int(header.get("NAXIS2", 0)),
                    }
                    record["preview"] = _preview_rows(path, index, args.preview)
                    records.append(record)
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
    fmt = resolve_emit_format(args)
    if fmt != "text":
        emit_records(records, format=fmt)
        return EXIT_OK
    for record in records:
        print(
            f"{record['file']}:{record['hdu']} {record['name']} nrows={record['nrows']}"
        )
        for field in record["schema"]:
            print(f"  {field['name']}: {field['type']}")
        if record["preview"]:
            print("preview:")
            # Same JSON contract as -f json: non-finite floats serialize as
            # null so the preview block is valid JSON (no bare NaN/Infinity).
            print(
                json.dumps(
                    _json_safe(record["preview"]),
                    default=json_default,
                    indent=2,
                    allow_nan=False,
                )
            )
    return EXIT_OK
