"""``torchfits arith`` — CFITSIO-style image arithmetic (scalar or image–image)."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Callable

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
    ensure_unique_split_stems,
    hdu_type_name,
    resolve_file_jobs,
    run_file_jobs,
    selected_hdu_indices,
)

_OPS: dict[str, Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]] = {
    "add": lambda t, v: t + v,
    "sub": lambda t, v: t - v,
    "mul": lambda t, v: t * v,
    "div": lambda t, v: t / v,
}

_DTYPE_CHOICES = ("auto", "float32", "float64")


def _promote_scalar(value: float) -> float | int:
    """Keep integer scalars exact so integer adds stay in integer arithmetic."""
    try:
        as_int = int(value)
    except (ValueError, OverflowError):
        return value
    return as_int if float(as_int) == value else value


def _saturate_to(t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Cast *t* to *dtype* clamping to the representable range.

    Returns the cast tensor and the number of elements that required clamping
    (i.e. wrapped values a naive cast would have corrupted).
    """
    if not t.dtype.is_floating_point and dtype.is_floating_point:
        return t.to(dtype), 0
    info = torch.iinfo(dtype)
    clipped = torch.clamp(t, info.min, info.max)
    n_clipped = int((clipped != t).sum())
    return clipped.to(dtype), n_clipped


def _compute(
    op_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor | float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run one arithmetic op without silent integer wraparound/truncation.

    Integer inputs are accumulated in int64 (or float64 when the operand is a
    float scalar) and saturating-cast back to ``out_dtype``; out-of-range
    pixels raise a RuntimeWarning instead of wrapping silently.
    """
    if out_dtype.is_floating_point:
        return op_fn(left.to(out_dtype), right)

    if left.dtype.is_floating_point:
        # Already float: native semantics are exact enough for preview math.
        return op_fn(left, right)

    # Integer input: accumulate wide so 65535 + 1 cannot wrap uint16.
    if isinstance(right, float):
        result = op_fn(left.to(torch.float64), float(right))
    else:
        right_t = right if isinstance(right, torch.Tensor) else None
        if right_t is not None and right_t.dtype.is_floating_point:
            result = op_fn(left.to(torch.float64), right_t.to(torch.float64))
        else:
            acc = left.to(torch.int64)
            acc_r = right_t.to(torch.int64) if right_t is not None else right
            result = op_fn(acc, acc_r)
    out, n_clipped = _saturate_to(result, out_dtype)
    if n_clipped:
        warnings.warn(
            f"arith: {n_clipped} pixel value(s) outside {out_dtype} range were "
            "saturated (use --dtype float32/float64 to keep full precision)",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "arith",
        help="image ±×÷ by a scalar or second image",
        description=(
            "CFITSIO-style imarith: image±×÷scalar or image±×÷image. "
            "-j = PyTorch intra-op threads; -J = parallel file workers for multi-A. "
            "Same-shape multi-HDU inputs are stacked for ATen ops."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "operand A path(s); with no --value and exactly two paths plus -o, "
            "the second path is operand B"
        ),
    )
    parser.add_argument(
        "--op", required=True, choices=tuple(_OPS), help="arithmetic op"
    )
    parser.add_argument(
        "--value",
        type=float,
        default=None,
        help="scalar operand B (mutually exclusive with image B)",
    )
    parser.add_argument(
        "--dtype",
        choices=_DTYPE_CHOICES,
        default="auto",
        help=(
            "output/compute dtype: 'auto' keeps the input dtype with "
            "saturation warnings on overflow; float32/float64 compute and "
            "store in that dtype"
        ),
    )
    parser.add_argument(
        "--operand2",
        default=None,
        help="image operand B path (for multi-A + --out-dir, or instead of 2nd positional)",
    )
    add_hdu_arg(
        parser,
        help="HDU indices on A (default: all image HDUs)",
    )
    parser.add_argument(
        "--hdu2",
        default=None,
        help="HDU index on image B (default: same as each A HDU, else 0)",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when processing multiple A inputs",
    )
    add_split_arg(parser)
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _parse_hdu2(hdu2: str | None) -> int | None:
    if hdu2 is None:
        return None
    try:
        return int(hdu2)
    except ValueError as exc:
        raise UsageError(f"invalid --hdu2: {hdu2!r}") from exc


def _resolve_operands(
    args: argparse.Namespace,
) -> tuple[list[str], float | None, str | None]:
    paths = [str(p) for p in args.paths]
    value = args.value
    operand2 = args.operand2
    if value is not None and operand2 is not None:
        raise UsageError("use either --value or --operand2, not both")

    if value is not None:
        return paths, float(value), None

    if operand2 is not None:
        return paths, None, str(operand2)

    # image–image shorthand: arith a.fits b.fits --op mul -o out.fits
    if len(paths) == 2 and args.out and not args.out_dir:
        return [paths[0]], None, paths[1]

    if len(paths) >= 2 and args.out_dir:
        raise UsageError(
            "multi-file A with image B requires --operand2 "
            "(or use --value for a scalar)"
        )

    raise UsageError(
        "operand B required: pass --value FLOAT, --operand2 PATH, "
        "or exactly two paths with -o/--out"
    )


def _image_indices(path: str, hdu: str | None) -> list[int]:
    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu)
            return [
                index
                for index in indices
                if hdu_type_name(hdul[index].header, hdul[index]) == "IMAGE"
            ]
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc


def _read_image(path: str, index: int) -> tuple[torch.Tensor, Any]:
    tensor = torchfits.read_tensor(path, hdu=index)
    if not isinstance(tensor, torch.Tensor):
        raise IoError(f"{path}:{index} read_tensor did not return a tensor")
    header = torchfits.read_header(path, index)
    return tensor, header


def _b_tensor(
    operand2: str,
    a_index: int,
    hdu2: int | None,
) -> torch.Tensor:
    if hdu2 is not None:
        b_index = hdu2
    else:
        try:
            with torchfits.open(operand2) as hdul:
                b_index = a_index if 0 <= a_index < len(hdul) else 0
        except Exception as exc:
            raise IoError(f"{operand2}: {exc}") from exc
    tensor, _ = _read_image(operand2, b_index)
    return tensor


def _hdu_width(indices: list[int]) -> int:
    if not indices:
        return 2
    return max(2, len(str(max(indices))))


def _resolve_out_dtype(spec: str, left: torch.Tensor) -> torch.dtype:
    if spec == "float32":
        return torch.float32
    if spec == "float64":
        return torch.float64
    return left.dtype


def _apply_op(
    op: str,
    left: torch.Tensor,
    right: torch.Tensor | float,
    dtype_spec: str = "auto",
) -> torch.Tensor:
    if op == "div" and not isinstance(right, torch.Tensor) and right == 0:
        raise UsageError("division by zero")
    if op == "div" and isinstance(right, torch.Tensor) and bool((right == 0).any()):
        raise UsageError("division by zero")
    out_dtype = _resolve_out_dtype(dtype_spec, left)
    if op == "div" and dtype_spec == "auto":
        # Fractional quotients must survive: integral inputs compute in
        # float64 rather than truncating through an integer output dtype.
        out_dtype = (
            left.dtype if left.dtype.is_floating_point else torch.float64
        )
    return _compute(_OPS[op], left, right, out_dtype)


def _arith_one_file(
    path_a: str,
    *,
    op: str,
    dtype_spec: str = "auto",
    value: float | None,
    operand2: str | None,
    hdu: str | None,
    hdu2: int | None,
    out_path: str | None,
    out_dir: Path | None,
    split: str,
) -> None:
    indices = _image_indices(path_a, hdu)
    if not indices:
        raise IoError(f"{path_a}: no image HDUs to process")

    tensors: list[torch.Tensor] = []
    headers: list[Any] = []
    for index in indices:
        tensor, header = _read_image(path_a, index)
        tensors.append(tensor)
        headers.append(header)

    rights: list[torch.Tensor | float]
    if value is not None:
        scalar_b: torch.Tensor | float = _promote_scalar(value)
        rights = [scalar_b] * len(tensors)
    else:
        assert operand2 is not None
        rights = [_b_tensor(operand2, index, hdu2) for index in indices]
        for index, left, right in zip(indices, tensors, rights, strict=True):
            if not isinstance(right, torch.Tensor):
                continue
            if left.shape != right.shape:
                raise UsageError(
                    f"shape mismatch at HDU {index}: {tuple(left.shape)} vs "
                    f"{tuple(right.shape)}"
                )

    shapes = {tuple(t.shape) for t in tensors}
    can_stack = len(shapes) == 1 and len(tensors) > 1
    if can_stack and value is not None:
        results = list(_apply_op(op, torch.stack(tensors), scalar_b, dtype_spec).unbind(0))
    elif (
        can_stack
        and value is None
        and all(isinstance(right, torch.Tensor) for right in rights)
    ):
        b_tensors = [right for right in rights if isinstance(right, torch.Tensor)]
        if len({tuple(right.shape) for right in b_tensors}) == 1:
            results = list(
                _apply_op(op, torch.stack(tensors), torch.stack(b_tensors), dtype_spec).unbind(0)
            )
        else:
            results = [
                _apply_op(op, left, right, dtype_spec)
                for left, right in zip(tensors, rights, strict=True)
            ]
    else:
        results = [
            _apply_op(op, left, right, dtype_spec)
            for left, right in zip(tensors, rights, strict=True)
        ]

    if split == "hdu":
        if out_dir is None:
            raise UsageError("--split hdu requires --out-dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        width = _hdu_width(indices)
        stem = Path(cfitsio_base_path(path_a)).stem
        for index, result, header in zip(indices, results, headers, strict=True):
            dest = out_dir / f"{stem}_hdu{index:0{width}d}.fits"
            torchfits.write_tensor(str(dest), result, header=header, overwrite=True)
        return

    if out_path is None:
        raise UsageError("output path required (-o/--out)")
    if len(results) == 1:
        torchfits.write_tensor(out_path, results[0], header=headers[0], overwrite=True)
    else:
        torchfits.write(
            out_path,
            [
                {"data": result, "header": header}
                for result, header in zip(results, headers, strict=True)
            ],
            overwrite=True,
        )


def run(args: argparse.Namespace) -> int:
    a_paths, value, operand2 = _resolve_operands(args)
    if args.out and args.out_dir:
        raise UsageError("use either --out-dir or -o/--out, not both")
    if len(a_paths) > 1 and not args.out_dir:
        raise UsageError("multiple A inputs require --out-dir")
    if len(a_paths) == 1 and not args.out and args.split != "hdu":
        raise UsageError("output path required (-o/--out)")
    if args.split == "hdu" and not args.out_dir:
        raise UsageError("--split hdu requires --out-dir")

    hdu2 = _parse_hdu2(args.hdu2)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(a_paths))
    if file_jobs == 1:
        configure_torch_jobs(int(args.jobs))

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        ensure_unique_basenames(a_paths)
        if args.split == "hdu":
            ensure_unique_split_stems(a_paths)
        out_dir.mkdir(parents=True, exist_ok=True)

    def _one(path_a: str) -> None:
        if out_dir is not None and args.split == "file":
            out_path = str(out_dir / Path(path_a).name)
        else:
            out_path = args.out
        _arith_one_file(
            path_a,
            op=args.op,
            dtype_spec=args.dtype,
            value=value,
            operand2=operand2,
            hdu=args.hdu,
            hdu2=hdu2,
            out_path=out_path,
            out_dir=out_dir,
            split=args.split,
        )

    try:
        run_file_jobs(a_paths, _one, file_jobs)
    except UsageError:
        raise
    except ZeroDivisionError as exc:
        raise UsageError("division by zero") from exc
    except IoError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
