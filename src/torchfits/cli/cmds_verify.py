"""``torchfits verify`` — checksum verification."""

from __future__ import annotations

import argparse
from typing import Any

import torchfits

from .common import (
    EXIT_OK,
    EXIT_VERIFY_FAIL,
    CliError,
    IoError,
    add_emit_format_args,
    add_file_jobs_arg,
    add_hdu_arg,
    emit_records,
    header_extname,
    resolve_emit_format,
    resolve_file_jobs,
    resolve_paths,
    run_file_jobs,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "verify",
        help="verify FITS checksum keywords",
        description=(
            "Verify DATASUM/CHECKSUM keywords. "
            "-J/--file-jobs fans out across files (Python thread pool)."
        ),
    )
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_emit_format_args(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _verify_one(path: str, hdu: str | None) -> tuple[list[dict[str, Any]], bool]:
    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu)
            # Reuse the headers already read by torchfits.open instead of
            # re-opening the file via read_header for every HDU (1 + 2N opens
            # per call down to 1 + N).
            headers = {index: hdul[index].header for index in indices}
    except CliError:
        raise
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    all_ok = True
    for index in indices:
        try:
            result = torchfits.verify_checksums(path, hdu=index)
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}:{index} {exc}") from exc
        header = headers[index]
        ok = bool(result.get("ok"))
        status_str = str(result.get("status", "fail"))
        all_ok = all_ok and ok
        records.append(
            {
                "file": path,
                "hdu": index,
                "name": header_extname(header, index),
                "ok": ok,
                "status": status_str,
                "datastatus": result.get("datastatus"),
                "hdustatus": result.get("hdustatus"),
            }
        )
    return records, all_ok


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(paths))
    chunks = run_file_jobs(
        paths,
        lambda path: _verify_one(path, args.hdu),
        file_jobs,
    )
    records: list[dict[str, Any]] = []
    all_ok = True
    for chunk_records, chunk_ok in chunks:
        records.extend(chunk_records)
        all_ok = all_ok and chunk_ok

    fmt = resolve_emit_format(args)
    if fmt == "text":
        for record in records:
            status = record["status"]
            if status == "no_checksums":
                label = "OK (no checksum keywords)"
            elif status == "ok":
                label = "OK"
            else:
                label = "FAIL"
            print(
                f"{record['file']}:{record['hdu']} {record['name']} {label} "
                f"data={record['datastatus']} hdu={record['hdustatus']}"
            )
    else:
        emit_records(records, format=fmt)
    return EXIT_OK if all_ok else EXIT_VERIFY_FAIL
