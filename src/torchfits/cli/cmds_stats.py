"""``torchfits stats`` — basic image statistics."""

from __future__ import annotations

import argparse
from typing import Any

import torch

import torchfits

from .common import (
    EXIT_OK,
    IoError,
    add_emit_format_args,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    configure_torch_jobs,
    emit_records,
    header_extname,
    hdu_type_name,
    resolve_emit_format,
    resolve_file_jobs,
    resolve_paths,
    run_file_jobs,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "stats",
        help="image min/max/mean/std/median via read_tensor",
        description=(
            "Image min/max/mean/std/median. "
            "-j = PyTorch intra-op threads; -J = parallel file workers."
        ),
    )
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_emit_format_args(parser)
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _stats_one(path: str, hdu: str | None) -> list[dict[str, Any]]:
    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu)
            headers = {index: hdul[index].header for index in indices}
            types = {
                index: hdu_type_name(headers[index], hdul[index]) for index in indices
            }
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for index in indices:
        if types[index] != "IMAGE":
            continue
        tensor = torchfits.read_tensor(path, hdu=index)
        if not isinstance(tensor, torch.Tensor):
            raise IoError(f"{path}:{index} read_tensor did not return a tensor")
        header = headers[index]
        flat = tensor.float().reshape(-1)
        # min/max must run on an upcast copy: torch ships no reduction kernels
        # for uint16/uint32/uint64 (the reader's unsigned conventions), so the
        # raw tensor would raise RuntimeError on exactly those inputs.
        stats_t = tensor if tensor.dtype.is_floating_point else tensor.float()
        records.append(
            {
                "file": path,
                "hdu": index,
                "name": header_extname(header, index),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "min": float(stats_t.min()),
                "max": float(stats_t.max()),
                "mean": float(flat.mean()),
                "std": float(flat.std(unbiased=False)),
                "median": float(flat.median()),
            }
        )
    return records


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(paths))
    if file_jobs == 1:
        configure_torch_jobs(int(args.jobs))
    chunks = run_file_jobs(
        paths,
        lambda path: _stats_one(path, args.hdu),
        file_jobs,
    )
    records = [record for chunk in chunks for record in chunk]
    emit_records(records, format=resolve_emit_format(args))
    return EXIT_OK
