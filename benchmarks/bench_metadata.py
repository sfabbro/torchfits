"""Frozen latency baseline for the metadata ("skinny") APIs and header reads.

Two dimensions matter here and both were missing from the ad-hoc probing that
first looked at these APIs:

* **Header size.** The skinny APIs exist so a bloated header need not be
  materialized (see ``read_nrows``' own docstring). Measured on a tiny header
  they lose to ``read_header``; measured on a bloated one they win. Reporting a
  single number for this family is meaningless.

* **Cold vs warm.** ``read_shape`` consults ``SharedReadMeta`` and returns
  without opening the file, so a repeated call is ~20x cheaper than the first.
  The other skinny APIs do not, and pay the open+scan floor every time. Cold is
  measured by taking the *first* call on each of N distinct copies, so no cache
  entry can exist.

Usage::

    pixi run python benchmarks/bench_metadata.py
    pixi run python benchmarks/bench_metadata.py --sizes 10 20000 --repeat 30
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

import torchfits  # noqa: E402
import torchfits.table  # noqa: E402

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

ROWS = 50
DEFAULT_SIZES = (10, 1000, 5000, 20000)

# ops whose cost should scale with header size (they must find the table HDU)
TABLE_OPS: dict[str, Callable[[str], Any]] = {
    "read_header(tab,1)": lambda p: torchfits.read_header(p, 1),
    "read_nrows": lambda p: torchfits.read_nrows(p),
    "read_keys(['NAXIS2'])": lambda p: torchfits.read_keys(p, ["NAXIS2"], hdu=1),
    "read_hdu_type(1)": lambda p: torchfits.read_hdu_type(p, 1),
    "read_num_hdus": lambda p: torchfits.read_num_hdus(p),
    "read_colnames(1)": lambda p: torchfits.read_colnames(p, 1),
}

IMAGE_OPS: dict[str, Callable[[str], Any]] = {
    "read_header(img,0)": lambda p: torchfits.read_header(p, 0),
    "read_shape(img,0)": lambda p: torchfits.read_shape(p, 0),
}


def _write_table(path: str, extra_cards: int) -> None:
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="C", format="D", array=np.zeros(ROWS))]
    )
    for i in range(extra_cards):
        hdu.header[f"K{i:06d}"] = i
    hdu.writeto(path, overwrite=True)


def _write_image(path: str) -> None:
    fits.PrimaryHDU(np.zeros((64, 64), dtype=np.float32)).writeto(path, overwrite=True)


def _timed(fn: Callable[[], Any], repeat: int) -> tuple[float, float]:
    """(median microseconds, python peak KiB) for ``fn``, warm by construction.

    Median, not mean: a single scheduling outlier in a short run moves the mean
    by more than the effect being measured.
    """
    fn()  # warm
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)

    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(np.median(samples)), peak / 1024.0


def _touch(path: str) -> None:
    """Fault the file's pages in so a later timed read is not measuring the OS.

    ``shutil.copyfile`` on APFS produces a copy-on-write clone whose first read
    faults pages; without this, 'cold' would measure the filesystem rather than
    the library cache and overstate the cost by more than an order of magnitude.
    """
    with open(path, "rb") as fh:
        while fh.read(1 << 20):
            pass


def _cold_us(fn: Callable[[str], Any], paths: list[str]) -> float:
    """Median microseconds of the first call on each distinct path (no cache)."""
    for path in paths:
        _touch(path)
    samples = []
    for path in paths:
        start = time.perf_counter()
        fn(path)
        samples.append((time.perf_counter() - start) * 1e6)
    return float(np.median(samples))


def run(sizes: tuple[int, ...], repeat: int, cold_copies: int, out: Path) -> list[dict]:
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        image = str(work / "img.fits")
        _write_image(image)
        for name, fn in IMAGE_OPS.items():
            us, peak = _timed(lambda fn=fn: fn(image), repeat)
            rows.append(
                {
                    "section": "image",
                    "cards": 0,
                    "op": name,
                    "us": us,
                    "cold_us": _cold_us(fn, [image]),
                    "py_peak_kib": peak,
                }
            )

        for cards in sizes:
            table = str(work / f"t_{cards}.fits")
            _write_table(table, cards)

            # distinct copies for the cold measurement (fresh cache entries)
            copies_dir = work / f"cold_{cards}"
            copies_dir.mkdir(exist_ok=True)
            copies = []
            for i in range(cold_copies):
                dst = str(copies_dir / f"c{i}.fits")
                shutil.copyfile(table, dst)
                copies.append(dst)

            for name, fn in TABLE_OPS.items():
                us, peak = _timed(lambda fn=fn: fn(table), repeat)
                rows.append(
                    {
                        "section": "table",
                        "cards": cards,
                        "op": name,
                        "us": us,
                        "cold_us": _cold_us(fn, copies),
                        "py_peak_kib": peak,
                    }
                )

            for mmap in (False, True):
                us, peak = _timed(
                    lambda mmap=mmap: torchfits.table.read_torch(table, mmap=mmap),
                    max(5, repeat // 10),
                )
                rows.append(
                    {
                        "section": "table",
                        "cards": cards,
                        "op": f"read_torch(mmap={mmap})",
                        "us": us,
                        "cold_us": float("nan"),
                        "py_peak_kib": peak,
                    }
                )

    print(
        f"{'section':7s} {'cards':>6s} {'op':24s} {'warm us':>10s} {'cold us':>10s} {'py peak KiB':>12s}"
    )
    for r in rows:
        cold = "" if np.isnan(r["cold_us"]) else f"{r['cold_us']:10.1f}"
        print(
            f"{r['section']:7s} {r['cards']:6d} {r['op']:24s} {r['us']:10.1f} {cold:>10s} {r['py_peak_kib']:12.1f}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("section,cards,op,warm_us,cold_us,py_peak_kib\n")
        for r in rows:
            fh.write(
                f"{r['section']},{r['cards']},{r['op']},{r['us']:.3f},"
                f"{r['cold_us']:.3f},{r['py_peak_kib']:.3f}\n"
            )
    print(f"\nwrote {out}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata/skinny-API latency baseline")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="extra header cards per fixture (default: 10 1000 5000 20000)",
    )
    parser.add_argument(
        "--repeat", type=int, default=20, help="warm repetitions (default 20)"
    )
    parser.add_argument(
        "--cold-copies",
        type=int,
        default=10,
        help="distinct files for the cold measurement (default 10)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_DIR / "metadata_results.csv"
    )
    args = parser.parse_args()
    run(tuple(args.sizes), args.repeat, args.cold_copies, args.out)


if __name__ == "__main__":
    main()
