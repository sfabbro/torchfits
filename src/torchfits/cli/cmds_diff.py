"""``torchfits diff`` — compare FITS headers and image metadata."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

import torchfits

from .common import EXIT_DIFF, EXIT_OK, IoError, hdu_type_name


_SKIP_HEADER_KEYS = frozenset({"CHECKSUM", "DATASUM"})


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("diff", help="compare two FITS files")
    parser.add_argument("path_a", help="first FITS path")
    parser.add_argument("path_b", help="second FITS path")
    parser.set_defaults(func=run)


def _header_map(header: Any) -> dict[str, Any]:
    return {
        card.key: card.value
        for card in header.cards
        if card.key not in _SKIP_HEADER_KEYS
    }


def _image_record(path: str, index: int) -> dict[str, Any]:
    tensor = torchfits.read_tensor(path, hdu=index)
    if not isinstance(tensor, torch.Tensor):
        raise IoError(f"{path}:{index} read_tensor did not return a tensor")
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.float().mean()),
    }


def _values_equal(a: Any, b: Any) -> bool:
    """NaN-aware equality: identical files containing NaN must not diff."""
    try:
        both_nan = isinstance(a, float) and isinstance(b, float) and a != a and b != b
    except TypeError:
        return a == b
    if both_nan:
        return True
    return a == b


def _diff_pair(path_a: str, path_b: str) -> list[str]:
    diffs: list[str] = []
    try:
        with torchfits.open(path_a) as hdul_a, torchfits.open(path_b) as hdul_b:
            if len(hdul_a) != len(hdul_b):
                diffs.append(f"HDU count: {len(hdul_a)} vs {len(hdul_b)}")
            for index in range(min(len(hdul_a), len(hdul_b))):
                header_a = hdul_a[index].header
                header_b = hdul_b[index].header
                type_a = hdu_type_name(header_a, hdul_a[index])
                type_b = hdu_type_name(header_b, hdul_b[index])
                if type_a != type_b:
                    diffs.append(f"HDU {index} type: {type_a} vs {type_b}")
                map_a = _header_map(header_a)
                map_b = _header_map(header_b)
                keys = sorted(set(map_a) | set(map_b))
                for key in keys:
                    val_a = map_a.get(key)
                    val_b = map_b.get(key)
                    if not _values_equal(val_a, val_b):
                        diffs.append(f"HDU {index} {key}: {val_a!r} vs {val_b!r}")
                if type_a == "IMAGE":
                    stats_a = _image_record(path_a, index)
                    stats_b = _image_record(path_b, index)
                    for field in ("shape", "min", "max", "mean"):
                        if not _values_equal(stats_a[field], stats_b[field]):
                            diffs.append(
                                f"HDU {index} image {field}: "
                                f"{stats_a[field]!r} vs {stats_b[field]!r}"
                            )
    except IoError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return diffs


def run(args: argparse.Namespace) -> int:
    diffs = _diff_pair(args.path_a, args.path_b)
    if not diffs:
        return EXIT_OK
    for line in diffs:
        print(line, file=sys.stderr)
    return EXIT_DIFF
