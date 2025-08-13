"""Shared utilities for benchmark scripts.

Includes:
- repeat timing with optional CUDA sync
- simple ASCII table formatting for console output
- helper to check optional deps lazily
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence, Tuple, List
import random

import torch
import numpy as np


def time_repeat(fn: Callable[[], Any], reps: int = 5, sync_cuda: bool = False, warmup: int = 1, use_median: bool = False) -> Tuple[float, float, list[float]]:
    """Time a callable multiple times.

    Returns (mean_or_median_ms, stdev_ms, samples_ms).
    """
    # warmup runs (not recorded)
    for _ in range(max(0, warmup)):
        try:
            fn()
        except Exception:
            break
    samples: list[float] = []
    for _ in range(reps):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    if not samples:
        return float("nan"), float("nan"), samples
    if use_median:
        ms = sorted(samples)
        mid = len(ms) // 2
        median = (ms[mid] if len(ms) % 2 == 1 else 0.5 * (ms[mid-1] + ms[mid]))
        # robust-ish spread: median absolute deviation scaled
        mad = sorted(abs(x - median) for x in ms)
        spread = mad[mid] if mad else 0.0
        return median, spread, samples
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    stdev = math.sqrt(var)
    return mean, stdev, samples


def force_consume(x: Any) -> float:
    """Touch data to force materialization and avoid lazy-memmap artifacts.

    Returns a tiny float so the call can't be optimized away.
    """
    try:
        if isinstance(x, torch.Tensor):
            return float(x.sum().item())
        if isinstance(x, np.ndarray):
            return float(np.sum(x))
    except Exception:
        pass
    return 0.0


def format_table(rows: Sequence[Sequence[Any]], headers: Sequence[str] | None = None) -> str:
    """Render a simple ASCII table without external deps.

    Args:
        rows: list of rows (list/tuple of values)
        headers: optional header row
    Returns: string table
    """
    rows_str = [["" if v is None else str(v) for v in r] for r in rows]
    if headers:
        header_str = [str(h) for h in headers]
        rows_all = [header_str] + rows_str
    else:
        rows_all = rows_str
    # compute widths
    widths = [max(len(r[i]) for r in rows_all) for i in range(len(rows_all[0]))] if rows_all else []

    def fmt_row(r: Sequence[str]) -> str:
        return " | ".join(s.ljust(widths[i]) for i, s in enumerate(r))

    parts: List[str] = []
    if headers:
        parts.append(fmt_row(rows_all[0]))
        parts.append("-+-".join("-" * w for w in widths))
        body = rows_all[1:]
    else:
        body = rows_all
    for r in body:
        parts.append(fmt_row(r))
    return "\n".join(parts)


def try_import(name: str):
    """Try to import a module, return (module|None)."""
    try:
        return __import__(name, fromlist=["__name__"])
    except Exception:
        return None


def numpy_to_torch(a: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor safely.

    - Ensures native endianness (byteswap if needed)
    - Ensures C-contiguous
    - Copies if array is not writeable (torch.from_numpy requires writeable buffer)
    """
    if not isinstance(a, np.ndarray):
        raise TypeError("expected numpy.ndarray")
    # strings/objects unsupported
    if a.dtype.kind in ("U", "S", "O"):  # unicode, bytes, object
        raise TypeError("numpy_to_torch does not support non-numeric dtypes")
    if not a.dtype.isnative:
        # byteswap returns a view; ensure dtype reports native afterwards via numpy API
        a = a.byteswap().view(a.dtype.newbyteorder('='))
    if not a.flags.writeable:
        a = a.copy()
    a = np.ascontiguousarray(a)
    return torch.from_numpy(a)
