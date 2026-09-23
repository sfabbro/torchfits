"""``torchfits compress`` / ``decompress`` — tile-compressed FITS I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchfits
from torchfits._io_engine.paths import cfitsio_base_path

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    add_split_arg,
    configure_torch_jobs,
    ensure_unique_basenames,
    _hdu_width,
    ensure_unique_split_stems,
    hdu_type_name,
    reject_same_path,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
    selected_hdu_indices,
)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="+",
        help="INPUT [OUTPUT], or multiple INPUTs with --out-dir",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs (required for multiple inputs or --split hdu)",
    )
    add_split_arg(parser)
    add_hdu_arg(
        parser,
        help="comma-separated HDU indices (default: all; with --split hdu)",
    )
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)


def add_compress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "compress",
        help="write tile-compressed FITS",
        description=(
            "Tile-compress image HDUs (Rice by default via CFITSIO). "
            "-j = PyTorch intra-op threads; -J = parallel file workers. "
            "Use --split hdu --out-dir for one file per image HDU; "
            "--algorithm selects the codec."
        ),
    )
    _add_shared_args(parser)
    parser.add_argument(
        "--algorithm",
        default="RICE_1",
        help=(
            "compression algorithm (default: RICE_1); "
            "also RICE, GZIP_1, GZIP_2, HCOMPRESS_1"
        ),
    )
    parser.set_defaults(func=run_compress)


def add_decompress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "decompress",
        help="write uncompressed FITS",
        description=(
            "Expand tile-compressed image HDUs. "
            "-j = PyTorch intra-op threads; -J = parallel file workers. "
            "Use --split hdu --out-dir for one file per image HDU."
        ),
    )
    _add_shared_args(parser)
    parser.set_defaults(func=run_decompress)


def _resolve_file_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    # Shared resolver in common.py owns the INPUT [OUTPUT] / -o / --out-dir
    # matrix (including basename collision checks and CFITSIO-section naming).
    return resolve_batch_io_pairs(
        [str(p) for p in args.paths],
        out=args.out,
        out_dir=args.out_dir,
        refuse_same_path=True,
    )


def _rewrite_file(
    input_path: str,
    output_path: str,
    *,
    compress: bool | str,
) -> None:
    try:
        with torchfits.open(input_path) as hdul:
            torchfits.write(output_path, hdul, overwrite=True, compress=compress)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc


def _hdu_output_path(
    out_dir: Path, input_path: str, hdu_index: int, *, width: int
) -> str:
    stem = Path(cfitsio_base_path(input_path)).stem
    return str(out_dir / f"{stem}_hdu{hdu_index:0{width}d}.fits")


def _rewrite_one_input_split_hdu(
    input_path: str,
    out_dir: Path,
    hdu: str | None,
    *,
    compress: bool | str,
    batch_inputs: tuple[str, ...] = (),
) -> int:
    """Compress/decompress image HDUs from one input; return count written."""
    try:
        with torchfits.open(input_path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu)
            image_indices = [
                index
                for index in indices
                if hdu_type_name(hdul[index].header, hdul[index]) == "IMAGE"
            ]
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc

    width = _hdu_width(image_indices)
    written = 0
    for index in image_indices:
        try:
            tensor = torchfits.read_tensor(input_path, hdu=index)
            if not isinstance(tensor, torch.Tensor):
                raise IoError(
                    f"{input_path}:{index} read_tensor did not return a tensor"
                )
            header = torchfits.read_header(input_path, index)
            output_path = _hdu_output_path(out_dir, input_path, index, width=width)
            # A generated split name may coincide with another input of the
            # batch (re-splitting into the same directory): refuse instead of
            # silently clobbering an unrelated input.
            for other in batch_inputs:
                reject_same_path(other, output_path)
            torchfits.write(
                output_path,
                tensor,
                header=header,
                overwrite=True,
                compress=compress,
            )
            written += 1
        except UsageError:
            raise
        except IoError:
            raise
        except Exception as exc:
            raise IoError(f"{input_path}:{index}: {exc}") from exc
    return written


def _rewrite_split_hdu(args: argparse.Namespace, *, compress: bool | str) -> None:
    if not args.out_dir:
        raise UsageError("--split hdu requires --out-dir")
    if args.out:
        raise UsageError("--split hdu uses --out-dir (not -o/--out)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [str(p) for p in args.paths]
    ensure_unique_basenames(inputs)
    ensure_unique_split_stems(inputs)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(inputs))
    if file_jobs == 1:
        configure_torch_jobs(int(args.jobs))
    counts = run_file_jobs(
        inputs,
        lambda path: _rewrite_one_input_split_hdu(
            path, out_dir, args.hdu, compress=compress, batch_inputs=tuple(inputs)
        ),
        file_jobs,
    )
    if sum(counts) == 0:
        raise IoError("no image HDUs to process")


def _run_rewrite(args: argparse.Namespace, *, compress: bool | str) -> int:
    split = getattr(args, "split", "file")
    if split == "hdu":
        _rewrite_split_hdu(args, compress=compress)
        return EXIT_OK

    pairs = _resolve_file_pairs(args)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(pairs))
    if file_jobs == 1:
        configure_torch_jobs(int(args.jobs))
    run_file_jobs(
        pairs,
        lambda pair: _rewrite_file(pair[0], pair[1], compress=compress),
        file_jobs,
    )
    return EXIT_OK


def run_compress(args: argparse.Namespace) -> int:
    algo = str(getattr(args, "algorithm", "RICE_1") or "RICE_1").strip() or "RICE_1"
    return _run_rewrite(args, compress=algo)


def run_decompress(args: argparse.Namespace) -> int:
    return _run_rewrite(args, compress=False)
