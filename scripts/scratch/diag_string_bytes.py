"""Diagnostic: dump on-disk bytes for the failing STRING mmap test case.

After landing the routing fix in src/torchfits/table.py, the C++ mmap writer
in src/torchfits/cpp_src/table.cpp is still producing 4-byte stride writes
between ASCII characters (`n\x00\x00\x00e\x00\x00\x00w...` instead of
`new111  `). This script writes a minimal table with the same column layout
the failing test uses, then dumps the raw on-disk bytes for the NAME column
so we can confirm where the corruption is happening.

Run from the activated pixi env:
    pixi run -e bench python scripts/diag_string_bytes.py
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import torchfits


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        ids = np.array([10, 20], dtype=np.int32)
        flux = np.array([1.5, 2.5], dtype=np.float32)
        bits = np.array(
            [
                [True, False, True, False, True, False, True, False],
                [False, True, False, True, False, True, False, True],
            ],
            dtype=np.bool_,
        )
        names8 = np.array(["alpha", "bravo"], dtype="S8")
        z = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

        torchfits.table.write(
            path.as_posix(),
            {
                "ID": ids,
                "FLUX": flux,
                "FLAGS": bits,
                "NAME": names8,
                "Z": z,
            },
            schema={
                "ID": {"format": "J"},
                "FLAGS": {"format": "8X"},
                "NAME": {"format": "8A"},
                "Z": {"format": "C"},
            },
            overwrite=True,
        )

        new_bits = np.array(
            [
                [False, True, False, True, False, True, False, True],
                [True, False, True, False, True, False, True, False],
            ],
            dtype=np.bool_,
        )
        new_complex = np.array([7 + 8j, 9 + 10j], dtype=np.complex64)
        new_names = np.array(["new111", "new222"], dtype="S8")
        new_ids = np.array([30, 40], dtype=np.int32)
        new_flux = np.array([3.5, 4.5], dtype=np.float32)

        torchfits.table.update_rows(
            path.as_posix(),
            {
                "ID": new_ids,
                "FLUX": new_flux,
                "FLAGS": new_bits,
                "NAME": new_names,
                "Z": new_complex,
            },
            row_slice=slice(0, 2),
            hdu=1,
            mmap=True,
        )

        raw = np.fromfile(path, dtype=np.uint8)
        header = torchfits.get_header(path.as_posix(), hdu=1)
        naxis1 = int(header["NAXIS1"])
        naxis2 = int(header["NAXIS2"])
        print(f"NAXIS1 = {naxis1}, NAXIS2 = {naxis2}")
        print(f"file size = {raw.size}")

        # Find the table data start by scanning for the END card.
        end_idx = int(np.where((raw == ord("E")) & (np.roll(raw, -1) == ord("N")) & (np.roll(raw, -2) == ord("D")))[0][0])
        data_start = end_idx + 80
        print(f"data start = {data_start}")

        print("Rows of bytes (each row is %d bytes):" % naxis1)
        for r in range(naxis2):
            row = raw[data_start + r * naxis1 : data_start + (r + 1) * naxis1]
            print(f"  row {r}: " + " ".join(f"{b:02x}" for b in row))

        # Column offsets from the header (TTYPE1..N + TFORM + byte offset not
        # directly in the header — derive from TFORM widths).
        # Layout: J(4) E(4) 8X(1) 8A(8) C(8) = 25 bytes expected?
        # NAXIS1 from header tells us the actual layout.
        print("Note: ID=J(4), FLUX=E(4), FLAGS=8X(1), NAME=8A(8), Z=C(8) -> sum=25")
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
