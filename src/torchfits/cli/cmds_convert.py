"""``torchfits convert`` — table export and RGB→PNG."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torchfits
from torchfits import table as tf_table
from torchfits.transforms.rgb import lupton_rgb, rgb as auto_rgb, write_rgb_image

from .common import EXIT_OK, IoError, UsageError, add_hdu_arg

_TABLE_FORMATS = ("parquet", "csv", "tsv", "arrow", "fits")
_ALL_FORMATS = (*_TABLE_FORMATS, "png")
_EXT_TO_FORMAT = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".ipc": "arrow",
    ".fits": "fits",
    ".fit": "fits",
    ".png": "png",
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="convert FITS tables (parquet/csv/tsv/arrow/fits) or RGB→PNG",
    )
    # nargs='+' would swallow a trailing positional output; resolve in run().
    parser.add_argument(
        "paths",
        nargs="+",
        help="input FITS path(s); trailing path is output unless -o/--out",
    )
    parser.add_argument("-o", "--out", default=None, help="output path")
    parser.add_argument(
        "--to",
        choices=_ALL_FORMATS,
        default=None,
        help="output format (default: infer from output extension)",
    )
    add_hdu_arg(parser, type=int, default=1, help="table HDU (default: 1)")
    parser.add_argument(
        "-w",
        "--where",
        help="row filter expression (table convert; same syntax as table.read)",
    )
    parser.add_argument(
        "-c",
        "--columns",
        help="comma-separated column names to keep (table convert)",
    )
    parser.add_argument(
        "--bands",
        help=(
            "comma-separated HDU indices for png "
            "(auto: 1–7 per input file; default: HDUs 0,1,2 of one file "
            "when present, else HDU 0)"
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=("auto", "lupton"),
        default="auto",
        help="png mapping: auto (default, blue→red) or lupton (reddest first)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.15,
        help="auto rgb sky+noise display value (default: 0.15)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=2.0,
        help="auto rgb chroma boost (default: 2; 1 = photometric)",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="read AB zeropoints from MAGZP/PHOTZP/FLUXMAG0/ZP (or use --zeropoints)",
    )
    parser.add_argument(
        "--zeropoints",
        default=None,
        help="comma-separated AB mag of 1 count per band (implies calibrated)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=8.0,
        help="Lupton Q (--recipe lupton)",
    )
    parser.add_argument(
        "--stretch", type=float, default=0.5, help="Lupton stretch (--recipe lupton)"
    )
    parser.set_defaults(func=run)


def _infer_format(output: str, to: str | None) -> str:
    if to is not None:
        return to
    suffix = Path(output).suffix.lower()
    fmt = _EXT_TO_FORMAT.get(suffix)
    if fmt is None:
        raise UsageError(
            "cannot infer convert format from output path; "
            "pass --to parquet|csv|tsv|arrow|fits|png"
        )
    return fmt


def _parse_columns(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [part.strip() for part in raw.split(",") if part.strip()]
    if not cols:
        raise UsageError("--columns requires at least one column name")
    return cols


def _parse_hdu_list(raw: str, *, flag: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise UsageError(f"{flag} must be comma-separated integers") from exc
    if not indices:
        raise UsageError(f"{flag} requires at least one integer")
    return indices


def _lupton_band_indices(raw: str | None, num_inputs: int) -> list[int]:
    if raw is None:
        if num_inputs == 1:
            return [0, 1, 2]
        if num_inputs == 3:
            return [0, 0, 0]
        raise UsageError(
            f"--recipe lupton got {num_inputs} input path(s); need one FITS file "
            "(optionally with --bands 0,1,2) or exactly three band files"
        )
    indices = _parse_hdu_list(raw, flag="--bands")
    if len(indices) != 3:
        raise UsageError("--recipe lupton: --bands requires exactly three HDU indices")
    if num_inputs not in (1, 3):
        raise UsageError("--recipe lupton accepts one FITS or three band FITS files")
    return indices


def _auto_band_indices(
    raw: str | None, num_inputs: int, inputs: list[str] | None = None
) -> list[int]:
    if not 1 <= num_inputs <= 7:
        raise UsageError(f"png convert accepts 1–7 FITS files, got {num_inputs}")
    if raw is None:
        if num_inputs == 1:
            # Keep the historical convenience: a single multi-HDU file
            # defaults to RGB from HDUs 0,1,2; grey otherwise.
            try:
                n_hdus = torchfits.read_num_hdus(inputs[0])
            except Exception:
                n_hdus = 1
            return [0, 1, 2][: min(3, n_hdus)] or [0]
        return [0] * num_inputs
    indices = _parse_hdu_list(raw, flag="--bands")
    if not 1 <= len(indices) <= 7:
        raise UsageError("--bands requires 1–7 HDU indices")
    if num_inputs == 1:
        return indices
    if len(indices) != num_inputs:
        raise UsageError(
            f"--bands length ({len(indices)}) must match input files ({num_inputs})"
        )
    return indices


def _parse_zeropoints(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise UsageError("--zeropoints must be comma-separated numbers") from exc
    if not values:
        raise UsageError("--zeropoints requires at least one value")
    return values


def _ab_zeropoint_from_header(path: str, hdu: int) -> float:
    header = torchfits.read_header(path, hdu)
    for key in ("MAGZP", "PHOTZP", "ZP"):
        value = header[key] if key in header else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    fluxmag0 = header["FLUXMAG0"] if "FLUXMAG0" in header else None
    if fluxmag0 is not None:
        try:
            flux = float(fluxmag0)
        except (TypeError, ValueError) as exc:
            raise UsageError(f"{path}: FLUXMAG0 is not a number") from exc
        if flux <= 0.0:
            raise UsageError(f"{path}: FLUXMAG0 must be positive")
        return 2.5 * math.log10(flux)
    raise UsageError(
        f"{path} HDU {hdu}: --calibrated needs MAGZP, PHOTZP, FLUXMAG0, or ZP "
        "(or pass --zeropoints)"
    )


def _arrow_to_column_dict(table: Any) -> dict[str, Any]:
    """Materialize an Arrow table as a dict of NumPy columns for FITS write."""
    import numpy as np

    out: dict[str, Any] = {}
    for name in table.column_names:
        col = table[name]
        try:
            out[name] = col.to_numpy(zero_copy_only=False)
        except Exception:
            out[name] = np.asarray(col.to_pylist(), dtype=object)
    return out


def _convert_table(args: argparse.Namespace, fmt: str) -> int:
    if len(args.inputs) != 1:
        raise UsageError("table convert accepts one input FITS file")
    path = args.inputs[0]
    hdu = args.hdu
    columns = _parse_columns(args.columns)
    where = args.where
    if where or columns or fmt == "fits":
        # Filter / column-select / FITS out: materialize via table.read.
        arrow = tf_table.read(path, hdu=hdu, columns=columns, where=where)
        if fmt == "parquet":
            tf_table.write_parquet(args.output, arrow)
        elif fmt == "csv":
            tf_table.write_csv(args.output, arrow, delimiter=",")
        elif fmt == "tsv":
            tf_table.write_csv(args.output, arrow, delimiter="\t")
        elif fmt == "arrow":
            tf_table.write_ipc(args.output, arrow)
        elif fmt == "fits":
            tf_table.write(args.output, _arrow_to_column_dict(arrow), overwrite=True)
        else:
            raise UsageError(f"unsupported table format: {fmt}")
        return EXIT_OK

    if fmt == "parquet":
        tf_table.write_parquet(args.output, path, hdu=hdu, stream=True)
    elif fmt == "csv":
        tf_table.write_csv(args.output, path, hdu=hdu, stream=True, delimiter=",")
    elif fmt == "tsv":
        tf_table.write_csv(args.output, path, hdu=hdu, stream=True, delimiter="\t")
    elif fmt == "arrow":
        tf_table.write_ipc(args.output, path, hdu=hdu, stream=True)
    else:
        raise UsageError(f"unsupported table format: {fmt}")
    return EXIT_OK


def _read_band(path: str, hdu: int) -> object:
    return torchfits.read_tensor(path, hdu=hdu).detach().cpu()


def _load_png_bands(inputs: list[str], hdus: list[int]) -> list[Any]:
    if len(inputs) == 1:
        path = inputs[0]
        return [_read_band(path, index) for index in hdus]
    return [_read_band(path, hdu) for path, hdu in zip(inputs, hdus, strict=True)]


def _convert_png(args: argparse.Namespace) -> int:
    if args.where or args.columns:
        raise UsageError("--where / --columns apply only to table convert")
    if args.recipe == "lupton":
        hdus = _lupton_band_indices(args.bands, len(args.inputs))
        bands = _load_png_bands(args.inputs, hdus)
        image = lupton_rgb(*bands, Q=args.q, stretch=args.stretch)
        write_rgb_image(args.output, image)
        return EXIT_OK

    hdus = _auto_band_indices(args.bands, len(args.inputs), args.inputs)
    bands = _load_png_bands(args.inputs, hdus)
    rgb_args: tuple[Any, ...]
    if len(args.inputs) == 1 and len(bands) == 1:
        rgb_args = (bands[0],)
    else:
        rgb_args = tuple(bands)
    zps = _parse_zeropoints(args.zeropoints)
    if args.calibrated and zps is None:
        if len(args.inputs) == 1:
            zps = [_ab_zeropoint_from_header(args.inputs[0], hdu) for hdu in hdus]
        else:
            zps = [
                _ab_zeropoint_from_header(path, hdu)
                for path, hdu in zip(args.inputs, hdus, strict=True)
            ]
    image = auto_rgb(
        *rgb_args,
        brightness=args.brightness,
        saturation=args.saturation,
        zeropoints=zps,
    )
    write_rgb_image(args.output, image)
    return EXIT_OK


def run(args: argparse.Namespace) -> int:
    try:
        if args.out:
            args.inputs = list(args.paths)
            args.output = args.out
        else:
            if len(args.paths) < 2:
                raise UsageError(
                    "output path required (-o/--out or trailing positional)"
                )
            args.inputs = list(args.paths[:-1])
            args.output = args.paths[-1]
        fmt = _infer_format(args.output, args.to)
        if fmt in _TABLE_FORMATS:
            return _convert_table(args, fmt)
        return _convert_png(args)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
