#!/usr/bin/env python3
"""Microbenchmark for ``update_rows`` throughput on COMPLEX / BIT / STRING columns.

Compares the in-place mmap writer (``mmap=True``) against the buffered row
path (``mmap=False``) for representative COMPLEX (``1C``/``1M``), BIT
(``8X``), and fixed-width STRING (``12A``) columns.

An **uncompressed** FITS file is used so the measured throughput reflects
column-write cost, not compression. This complements the mmap fast-path
coverage in `tests/test_astropy_upstream_smoke.py` and
`tests/test_fitsio_upstream_smoke.py` and is intended to make the recent
parity shift in `docs/parity.md` and the "Unreleased" section of
`docs/changelog.md` visible at a glance.

Style mirrors `bench_compressed.py`: small, focused, runnable in seconds,
not part of the orchestrated `bench-all` suite.

Run::

    pixi run -e bench python benchmarks/bench_decompressed_complex_bit_string.py
    pixi run -e bench python benchmarks/bench_decompressed_complex_bit_string.py --num-rows 32768
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import torchfits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_uncompressed_table(path: Path, num_rows: int) -> None:
    """Write a small uncompressed FITS BinTable with COMPLEX / BIT / STRING columns.

    Schema (column -> FITS TFORM):
        - ``CFLT``  -> ``1C``  (single complex64 per row)
        - ``CDBL``  -> ``1M``  (single complex128 per row)
        - ``FLAGS`` -> ``8X``  (8 bits packed into one byte per row)
        - ``NAME``  -> ``8A``  (8-byte ASCII string per row)

    ``FLAGS`` is supplied as a 2D ``(num_rows, 8)`` bool array so that
    astropy writes the bits packed (1 byte/row, repeat=1); a 1D uint8
    payload would unpack as 8 bytes/row and trigger an `update_rows mmap
    repeat mismatch` against the FITS column metadata.
    """
    from astropy.io import fits

    bit_init = np.zeros((num_rows, 8), dtype=np.bool_)        # 8X, packed 1 byte/row
    name_init = np.array(
        [f"row_{i:07d}".encode("ascii") for i in range(num_rows)],
        dtype="S8",
    ).astype(np.bytes_)
    cflt_init = np.zeros(num_rows, dtype=np.complex64)
    cdbl_init = np.zeros(num_rows, dtype=np.complex128)

    cols = [
        fits.Column(name="CFLT", format="1C", array=cflt_init),
        fits.Column(name="CDBL", format="1M", array=cdbl_init),
        fits.Column(name="FLAGS", format="8X", array=bit_init),
        fits.Column(name="NAME", format="8A", array=name_init),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, nrows=num_rows)
    hdu.writeto(str(path), overwrite=True)


def _payloads(num_rows: int) -> dict[str, np.ndarray]:
    """New column values to write on every iteration.

    Sized so every column carries exactly ``expected_repeat`` bytes per row,
    which is the path the mmap writer's STRING / BIT / COMPLEX cases
    exercise end-to-end.
    """
    return {
        "CFLT": np.array(
            [complex(float(i), float(i) + 0.5) for i in range(num_rows)],
            dtype=np.complex64,
        ),
        "CDBL": np.array(
            [complex(float(i) + 0.25, float(i)) for i in range(num_rows)],
            dtype=np.complex128,
        ),
        "FLAGS": np.full((num_rows, 8), True, dtype=np.bool_),
        "NAME": np.array(
            [f"upd_{i:07d}".encode("ascii") for i in range(num_rows)],
            dtype="S8",
        ),
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_callable(fn, warmup: int, iterations: int) -> float:
    """Return median wall-clock seconds across ``iterations`` invocations."""
    samples: list[float] = []
    for _ in range(warmup):
        fn()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _update(path: Path, hdu: int, col: str, value: np.ndarray, num_rows: int, mmap: bool) -> None:
    torchfits.table.update_rows(
        str(path),
        {col: value},
        row_slice=slice(0, num_rows),
        hdu=hdu,
        mmap=mmap,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _bench_one_column(
    path: Path, col: str, num_rows: int, warmup: int, iterations: int
) -> tuple[float, float, float]:
    """Return ``(mmap_s, buffered_s, speedup)`` for a single column."""
    payloads = _payloads(num_rows)
    payload = payloads[col]
    mmap_s = _time_callable(
        lambda: _update(path, 1, col, payload, num_rows, True),
        warmup=warmup,
        iterations=iterations,
    )
    buf_s = _time_callable(
        lambda: _update(path, 1, col, payload, num_rows, False),
        warmup=warmup,
        iterations=iterations,
    )
    speedup = mmap_s / buf_s if buf_s > 0 else float("inf")
    return mmap_s, buf_s, speedup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rows", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["CFLT", "CDBL", "FLAGS", "NAME"],
        help="Columns to time one at a time (default: all COMPLEX / BIT / STRING).",
    )
    parser.add_argument(
        "--update-four",
        action="store_true",
        help="Also time a single update_rows call that touches all 4 columns at once.",
    )
    args = parser.parse_args()

    if hasattr(__import__("torch"), "set_num_threads"):
        __import__("torch").set_num_threads(1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decompressed.fits"
        _build_uncompressed_table(path, num_rows=args.num_rows)
        print(
            "[bench_decompressed_complex_bit_string] "
            f"rows={args.num_rows} file={path} "
            f"warmup={args.warmup} iterations={args.iterations}"
        )
        print(
            f"{'column':<8} {'format':<6} {'mmap(s)':>12} {'buffered(s)':>12} {'speedup':>10}"
        )
        format_lookup = {"CFLT": "1C", "CDBL": "1M", "FLAGS": "8X", "NAME": "8A"}
        for col in args.columns:
            mmap_s, buf_s, speedup = _bench_one_column(
                path, col, args.num_rows, args.warmup, args.iterations
            )
            print(
                f"{col:<8} {format_lookup.get(col, '?'):<6} "
                f"{mmap_s:>12.6f} {buf_s:>12.6f} {speedup:>10.3f}x"
            )

        if args.update_four:
            payloads = _payloads(args.num_rows)
            fn_four = lambda: torchfits.table.update_rows(
                str(path),
                payloads,
                row_slice=slice(0, args.num_rows),
                hdu=1,
                mmap=True,
            )
            t_mmap = _time_callable(fn_four, args.warmup, args.iterations)
            fn_four_buf = lambda: torchfits.table.update_rows(
                str(path),
                payloads,
                row_slice=slice(0, args.num_rows),
                hdu=1,
                mmap=False,
            )
            t_buf = _time_callable(fn_four_buf, args.warmup, args.iterations)
            print(
                f"{'ALL':<8} {'mixed':<6} {t_mmap:>12.6f} {t_buf:>12.6f} "
                f"{(t_mmap / t_buf if t_buf > 0 else float('inf')):>10.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
