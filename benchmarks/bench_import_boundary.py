"""Cold-start latency of every entry point, split by the torch boundary.

Why this exists: a header peek reads a 2880-byte block and costs microseconds,
yet ``torchfits info`` on a cold process paid ~880 ms for an image-size tensor
runtime it never touched.  Wall-clock numbers for single operations hide that
completely; *process* cold start is the number users feel.

Two properties are measured per entry point, in a **fresh interpreter** each
time:

* whether PyTorch ended up loaded (the boundary), and
* wall-clock from process spawn to exit (the cost users pay).

The harness itself deliberately needs neither torch nor torchfits imported in
its own process: it writes a minimal FITS file with the standard library and
shells out for every measurement.  That means it can run in exactly the
environment the boundary work is supposed to make possible — a torch-free one.

Usage::

    pixi run python benchmarks/bench_import_boundary.py
    pixi run python benchmarks/bench_import_boundary.py --repeat 5
    pixi run python benchmarks/bench_import_boundary.py --strict   # gates
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# Metadata no longer touches the torch-linked extension at all: it goes
# through libtorchfits_core, which has no libtorch in its dependency list. These
# budgets are the real cold-start numbers for that path plus a margin; see
# docs/benchmarks.md for the measured table they are derived from.
#
# ``_core`` itself is the floor: a bare dlopen of the library plus the probe.
# Anything above it is Python-level overhead, which is what these gate.
_METADATA_BUDGET_MS = 120.0
# Measured 461-515 ms minimum across runs (5 repeats, macOS arm64), with
# medians up to 541 ms. Budget is set from the slow end of that spread, not the
# fastest sample: a gate that only passes on a good day is not a gate.
_ARROW_BUDGET_MS = 600.0
# Bare ``import torchfits._core`` plus one query: a dlopen of
# libtorchfits_core with no torch anywhere in the process.
_CORE_BUDGET_MS = 80.0


@dataclass(frozen=True)
class Entry:
    """One cold-start measurement target."""

    name: str
    statement: str
    expects_torch: bool
    budget_ms: float | None = None

    @property
    def expects_torch_free(self) -> bool:
        return not self.expects_torch


def _integer_card(keyword: str, value: int) -> bytes:
    return f"{keyword:<8}= {value:>20}".encode("ascii").ljust(80)


def _logical_card(keyword: str, value: bool) -> bytes:
    return f"{keyword:<8}= {('T' if value else 'F'):>20}".encode("ascii").ljust(80)


def _string_card(keyword: str, value: str) -> bytes:
    return f"{keyword:<8}= '{value:<8}'".encode("ascii").ljust(80)


def write_minimal_image(path: Path) -> None:
    """Write a hand-rolled 4x4 uint8 primary HDU — no torch, no astropy."""
    header = b"".join(
        [
            _logical_card("SIMPLE", True),
            _integer_card("BITPIX", 8),
            _integer_card("NAXIS", 2),
            _integer_card("NAXIS1", 4),
            _integer_card("NAXIS2", 4),
            _logical_card("EXTEND", True),
            b"END".ljust(80),
        ]
    )
    with path.open("wb") as handle:
        handle.write(header.ljust(2880))
        handle.write(bytes(range(16)).ljust(2880))


def write_minimal_table(path: Path) -> None:
    """Write a primary HDU plus a 3-row BINTABLE, by hand.

    The harness has to build its own fixtures: ``*.fits`` is gitignored in this
    repository, so a checked-in sample would exist only for whoever last ran a
    test that wrote one.  Columns are int32 (``J``) and float64 (``D``), which
    covers a scalar integer, a wide float and a table HDU at index 1.
    """
    primary = b"".join(
        [
            _logical_card("SIMPLE", True),
            _integer_card("BITPIX", 8),
            _integer_card("NAXIS", 0),
            _logical_card("EXTEND", True),
            b"END".ljust(80),
        ]
    )
    table_header = b"".join(
        [
            _string_card("XTENSION", "BINTABLE"),
            _integer_card("BITPIX", 8),
            _integer_card("NAXIS", 2),
            _integer_card("NAXIS1", 12),  # row width: int32 + float64
            _integer_card("NAXIS2", 3),
            _integer_card("PCOUNT", 0),
            _integer_card("GCOUNT", 1),
            _integer_card("TFIELDS", 2),
            _string_card("TTYPE1", "N"),
            _string_card("TFORM1", "J"),
            _string_card("TTYPE2", "FLUX"),
            _string_card("TFORM2", "D"),
            _string_card("EXTNAME", "MY_TABLE"),
            b"END".ljust(80),
        ]
    )
    # FITS is big-endian.
    table_data = b"".join(struct.pack(">id", n, n + 0.5) for n in (1, 2, 3))
    with path.open("wb") as handle:
        handle.write(primary.ljust(2880))
        handle.write(table_header.ljust(2880))
        handle.write(table_data.ljust(2880))


def build_entries(image: Path, table: Path) -> list[Entry]:
    img = repr(str(image))
    tab = repr(str(table))
    return [
        # --- the boundary: metadata must not load torch -----------------------
        Entry("import torchfits", "import torchfits", False, 250.0),
        Entry("import torchfits.hdu", "import torchfits.hdu", False, 250.0),
        Entry("import torchfits.io", "import torchfits.io", False, 250.0),
        Entry("import torchfits.table", "import torchfits.table", False, 500.0),
        Entry(
            "read_header",
            f"import torchfits; torchfits.read_header({tab}, 1)",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            "read_keys",
            f"import torchfits; torchfits.read_keys({tab}, ['NAXIS2'], 1)",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            "read_colnames",
            f"import torchfits; torchfits.read_colnames({tab}, 1)",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            "read_num_hdus",
            f"import torchfits; torchfits.read_num_hdus({tab})",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            "read_shape",
            f"import torchfits; torchfits.read_shape({img}, 0)",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            "read_table_info",
            f"import torchfits; torchfits.read_table_info({tab}, 1)",
            False,
            _METADATA_BUDGET_MS,
        ),
        Entry(
            # torchfits.open returns an HDUList that owns a native handle, and
            # that handle is the same object the tensor readers take, so this
            # path cannot avoid the torch-linked extension yet. Budgeted from
            # the measured cost rather than pretending otherwise; routing it
            # through the core means making the handle lazy, which is tracked
            # in docs/roadmap.md.
            "open + header",
            f"import torchfits\nwith torchfits.open({tab}) as h:\n    h[1].header",
            False,
            250.0,
        ),
        # The native core on its own: no torchfits Python package, no cache
        # machinery, just the library and one query. This is the floor the
        # metadata entries above are measured against.
        Entry(
            "_core.read_colnames",
            f"import torchfits._core as c; c.read_colnames({tab}, 1)",
            False,
            _CORE_BUDGET_MS,
        ),
        Entry(
            "_core.read_header_dict",
            f"import torchfits._core as c; c.read_header_dict({tab}, 1)",
            False,
            _CORE_BUDGET_MS,
        ),
        Entry(
            "table.read (Arrow)",
            f"import torchfits.table; torchfits.table.read({tab}, 1)",
            False,
            _ARROW_BUDGET_MS,
        ),
        Entry(
            "table.schema",
            f"import torchfits.table; torchfits.table.schema({tab}, hdu=1)",
            False,
            _ARROW_BUDGET_MS,
        ),
        # --- the converse: tensor destinations must still pay for torch -------
        Entry(
            "read_tensor",
            f"import torchfits; torchfits.read_tensor({img}, 0)",
            True,
        ),
        Entry(
            "table.read_torch",
            f"import torchfits.table; torchfits.table.read_torch({tab}, 1)",
            True,
        ),
        Entry("import torch (reference)", "import torch", True),
    ]


# Printed by the probe so its torch state can be read back without a second run.
_SENTINEL = "__TORCHFITS_BOUNDARY__"


def measure(entry: Entry, repeat: int, timeout: float) -> dict[str, object]:
    """Run *entry* in fresh interpreters; report min/median spawn-to-exit cost."""
    script = (
        f"{entry.statement}\n"
        f"import sys as _sys\n"
        f"print({_SENTINEL!r}, int('torch' in _sys.modules))\n"
    )
    durations: list[float] = []
    torch_loaded = False
    for _ in range(repeat):
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        durations.append((time.perf_counter() - started) * 1000.0)
        if proc.returncode != 0:
            raise SystemExit(
                f"{entry.name}: probe failed with exit {proc.returncode}\n"
                f"{proc.stderr.strip()}"
            )
        for line in proc.stdout.splitlines():
            if line.startswith(_SENTINEL):
                torch_loaded = line.split()[-1] == "1"
    return {
        "name": entry.name,
        "min_ms": min(durations),
        "median_ms": statistics.median(durations),
        "torch_loaded": torch_loaded,
        "expects_torch": entry.expects_torch,
        "budget_ms": entry.budget_ms,
    }


def render(results: list[dict[str, object]]) -> None:
    print()
    print(f"{'entry point':<28}{'min ms':>9}{'median ms':>11}  torch   budget")
    print("-" * 72)
    for row in results:
        budget = row["budget_ms"]
        budget_text = f"{budget:.0f} ms" if isinstance(budget, float) else "-"
        print(
            f"{row['name']:<28}{row['min_ms']:>9.1f}{row['median_ms']:>11.1f}"
            f"  {'yes' if row['torch_loaded'] else 'no ':5}  {budget_text}"
        )
    print()
    print("Budgets are compared against the median column (--strict).")
    print()


def check(results: list[dict[str, object]]) -> list[str]:
    """Return the list of gate failures (empty means the boundary holds).

    Budgets are compared against ``median_ms``, not ``min_ms``. The budgets are
    calibrated from the slow end of the measured spread (see
    ``_ARROW_BUDGET_MS``), so comparing the best of N cold starts would spend
    roughly half the margin that calibration exists to provide: measured here,
    ``table.read (Arrow)`` runs 491.6 ms at its minimum and 545.9 ms at its
    median against a 600 ms budget -- 108 ms of headroom on the minimum, 54 ms
    on the median. Spawn-to-exit latency is what users feel, and a median is
    what they feel; ``min_ms`` is still reported so the spread stays visible.
    """
    failures: list[str] = []
    for row in results:
        name = str(row["name"])
        if row["expects_torch"] and not row["torch_loaded"]:
            failures.append(f"{name}: tensor entry point did not load torch")
        if not row["expects_torch"] and row["torch_loaded"]:
            failures.append(f"{name}: metadata entry point loaded torch")
        budget = row["budget_ms"]
        if isinstance(budget, float) and float(row["median_ms"]) > budget:
            failures.append(
                f"{name}: median {float(row['median_ms']):.0f} ms exceeds the "
                f"{budget:.0f} ms budget (min was {float(row['min_ms']):.0f} ms)"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="cold-start torch-boundary timings")
    parser.add_argument("--repeat", type=int, default=3, help="runs per entry point")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="per-probe seconds"
    )
    parser.add_argument("--json", type=Path, default=None, help="write results as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="gate on both the boundary and the per-entry-point budgets",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "minimal.fits"
        table = Path(tmp) / "minimal_table.fits"
        write_minimal_image(image)
        write_minimal_table(table)
        entries = build_entries(image, table)
        results = [measure(entry, args.repeat, args.timeout) for entry in entries]

    render(results)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    failures = check(results)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        if args.strict:
            return 1
        print("(informational: strict mode gates boundary and timing budgets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
