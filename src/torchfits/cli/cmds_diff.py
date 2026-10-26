"""``torchfits diff`` — compare FITS headers and image metadata."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torchfits

from .common import EXIT_DIFF, EXIT_OK, IoError, hdu_type_name


_SKIP_HEADER_KEYS = frozenset({"CHECKSUM", "DATASUM"})


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("diff", help="compare two FITS files")
    parser.add_argument("path_a", help="first FITS path")
    parser.add_argument("path_b", help="second FITS path")
    parser.set_defaults(func=run)


def _header_cards(header: Any) -> list[tuple[str, Any, str]]:
    return [
        (card.key, card.value, card.comment)
        for card in header.cards
        if card.key not in _SKIP_HEADER_KEYS
    ]


def _image_record(path: str, index: int) -> dict[str, Any]:
    import torch

    tensor = torchfits.read_tensor(path, hdu=index)
    if not isinstance(tensor, torch.Tensor):
        raise IoError(f"{path}:{index} read_tensor did not return a tensor")
    # min/max/mean must run on an upcast copy (torch has no uint reductions)
    # that KEEPS float64 in float64: tensor.float().mean() rounds float64
    # images to float32 and hides real mean differences. An empty region has
    # no defined stats (NaN, which _values_equal treats as equal to NaN);
    # torch reductions raise on numel() == 0.
    stats_t = tensor if tensor.dtype.is_floating_point else tensor.float()
    empty = stats_t.numel() == 0
    nan = float("nan")
    return {
        "shape": list(tensor.shape),
        "min": nan if empty else float(stats_t.min()),
        "max": nan if empty else float(stats_t.max()),
        "mean": nan if empty else float(stats_t.mean()),
    }


def _values_equal(a: Any, b: Any) -> bool:
    """NaN-aware equality: identical files containing NaN must not diff."""
    try:
        both_nan = isinstance(a, float) and isinstance(b, float) and a != a and b != b
        if both_nan:
            return True
        return bool(a == b)
    except TypeError:
        return bool(a == b)


def _cards_equal(
    card_a: tuple[str, Any, str] | None, card_b: tuple[str, Any, str] | None
) -> bool:
    if card_a is None or card_b is None:
        return card_a is card_b
    return (
        card_a[0] == card_b[0]
        and _values_equal(card_a[1], card_b[1])
        and card_a[2] == card_b[2]
    )


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
                cards_a = _header_cards(header_a)
                cards_b = _header_cards(header_b)
                for position in range(max(len(cards_a), len(cards_b))):
                    card_a = cards_a[position] if position < len(cards_a) else None
                    card_b = cards_b[position] if position < len(cards_b) else None
                    if not _cards_equal(card_a, card_b):
                        diffs.append(
                            f"HDU {index} card {position + 1}: {card_a!r} vs {card_b!r}"
                        )
                if type_a == "IMAGE" and type_b == "IMAGE":
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
