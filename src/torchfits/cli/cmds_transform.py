"""``torchfits transform`` — apply a named torchfits.transforms class."""

from __future__ import annotations

import argparse
import inspect
from typing import Any

import torch

import torchfits

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    configure_torch_jobs,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
)


def _coerce_transform_kwarg_value(raw: str) -> bool | int | float | str:
    """Coerce a ``key=val`` string value to bool/int/float, else leave as str."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_transform_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``Name`` or ``Name:key=val,key2=val2`` into (name, kwargs)."""
    name, sep, kwargs_str = spec.partition(":")
    if not sep:
        return name, {}
    kwargs: dict[str, Any] = {}
    for piece in kwargs_str.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, eq, value = piece.partition("=")
        key = key.strip()
        if not eq or not key:
            raise UsageError(f"invalid --name kwarg (expected key=val): {piece!r}")
        kwargs[key] = _coerce_transform_kwarg_value(value.strip())
    return name, kwargs


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "transform",
        help="apply a torchfits.transforms class",
        description=(
            "Apply a named transform. Multiple inputs need --out-dir; "
            "-j = torch threads; -J = file workers."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="INPUT [OUTPUT], or multiple INPUTs with --out-dir",
    )
    parser.add_argument(
        "--name",
        required=True,
        help=(
            "transform class name; append :key=val,key2=val2 to pass "
            "constructor kwargs, e.g. ArcsinhStretch:a=2.0"
        ),
    )
    add_hdu_arg(parser, type=int, default=0, help="source HDU index")
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when transforming multiple inputs",
    )
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _build_transform(name_spec: str) -> tuple[str, Any]:
    import torchfits.transforms as tf_transforms

    name, kwargs = _parse_transform_spec(name_spec)
    public = set(tf_transforms.__all__)
    if name not in public:
        raise UsageError(f"unknown transform: {name}")
    cls = getattr(tf_transforms, name)
    if kwargs:
        valid_params = set(inspect.signature(cls).parameters)
        unknown = sorted(set(kwargs) - valid_params)
        if unknown:
            raise UsageError(f"unknown kwarg(s) for {name}: {unknown}")
    return name, cls(**kwargs)


def _transform_one(
    pair: tuple[str, str],
    *,
    name: str,
    transform: Any,
    hdu: int,
) -> None:
    input_path, output_path = pair
    try:
        tensor = torchfits.read_tensor(input_path, hdu=hdu)
        if not isinstance(tensor, torch.Tensor):
            raise IoError(f"{input_path}:{hdu} read_tensor did not return a tensor")
        if not tensor.is_floating_point():
            tensor = tensor.float()
        header = torchfits.read_header(input_path, hdu)
        result = transform(tensor)
        if not isinstance(result, torch.Tensor):
            raise IoError(f"{name} did not return a tensor")
        if result.is_floating_point():
            header = None
        torchfits.write_tensor(output_path, result, header=header, overwrite=True)
    except UsageError:
        raise
    except IoError:
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
    name, transform = _build_transform(args.name)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(pairs))
    if file_jobs == 1:
        configure_torch_jobs(int(args.jobs))

    # Transforms may carry per-call state (_last_state/_last_mask). With
    # multi-file fan-out, give every worker its own instance so concurrent
    # invocations never share mutable state.
    if file_jobs > 1:
        transform = None
    run_file_jobs(
        pairs,
        lambda pair: _transform_one(
            pair,
            name=name,
            transform=transform
            if transform is not None
            else _build_transform(args.name)[1],
            hdu=int(args.hdu),
        ),
        file_jobs,
    )
    return EXIT_OK
