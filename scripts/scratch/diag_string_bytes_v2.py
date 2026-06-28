#!/usr/bin/env python3
"""Targeted byte-level diagnostic for the failing fitsio BIT/STRING/COMPLEX test.

Replicates the failing test exactly, then dumps the raw bytes of the data section
(bypassing fitsio's decode entirely) so we can see what is actually on disk.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Reuse project src without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torchfits  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="torchfits_diag_") as tmp:
        path = Path(tmp) / "decompressed.fits"

        # Initial table (binary; TFORM=J, E, 8X, 8A, C).
        n = 2
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
            str(path),
            {"ID": ids, "FLUX": flux, "FLAGS": bits, "NAME": names8, "Z": z},
            schema={
                "ID": {"format": "J"},
                "FLAGS": {"format": "8X"},
                "NAME": {"format": "8A"},
                "Z": {"format": "C"},
            },
            overwrite=True,
        )

        new_ids = np.array([30, 40], dtype=np.int32)
        new_flux = np.array([3.5, 4.5], dtype=np.float32)
        new_bits = np.array(
            [
                [False, True, False, True, False, True, False, True],
                [True, False, True, False, True, False, True, False],
            ],
            dtype=np.bool_,
        )
        new_names = np.array(["new111", "new222"], dtype="S8")
        new_z = np.array([7 + 8j, 9 + 10j], dtype=np.complex64)

        torchfits.table.update_rows(
            str(path),
            {
                "ID": new_ids,
                "FLUX": new_flux,
                "FLAGS": new_bits,
                "NAME": new_names,
                "Z": new_z,
            },
            row_slice=slice(0, 2),
            hdu=1,
            mmap=True,
        )

        # Open the file as raw bytes and read the trailer.
        data = path.read_bytes()
        with open(path, "rb") as fh:
            data = fh.read()

        # FITS files pad blocks to 2880 bytes. After 2 header blocks for PHDU + BinTableHDU
        # (NAXIS1=25, NAXIS2=2) the data section lives in the 3rd block (5760+).
        print(f"file size = {len(data)} bytes")
        # Scan for non-zero tail content: data block starts at byte 5760.
        naxis1 = 25
        naxis2 = 2
        tail_start = 5760
        print(f"expected NAXIS1={naxis1}, NAXIS2={naxis2}, row size={naxis1} -> 2 rows = {naxis1 * naxis2} bytes")
        print()
        for i in range(naxis2):
            row_offset = tail_start + i * naxis1
            row_bytes = data[row_offset:row_offset + naxis1]
            hex_str = " ".join(f"{b:02x}" for b in row_bytes)
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "·" for b in row_bytes)
            print(f"row {i} (offset {row_offset}): {hex_str}")
            print(f"           ascii: |{ascii_str}|")

        # Compare against expected layout.
        # Layout: ID[J=4] | FLUX[E=4] | FLAGS[8X=1] | NAME[8A=8] | Z[C=8] -> 25 bytes per row.
        # Row 0 expected:
        #   ID = 30 (big-endian int32) = 00 00 00 1e
        #   FLUX = 3.5 (big-endian float32) = 40 60 00 00
        #   FLAGS = packed bits of [F,T,F,T,F,T,F,T] = 01010101 = 0x55
        #   NAME = "new111  " = 6e 65 77 31 31 31 20 20
        #   Z = 7+8j (big-endian complex64) = 40 e0 00 00 4100 0000 ≠ just bytes; 7.0=40 e0 00 00, 8.0=41 00 00 00
        expected_row0 = bytes([
            0x00, 0x00, 0x00, 0x1e,                # ID=30
            0x40, 0x60, 0x00, 0x00,                # FLUX=3.5
            0x55,                                  # FLAGS bits
            0x6e, 0x65, 0x77, 0x31, 0x31, 0x31, 0x20, 0x20,  # "new111  "
            0x40, 0xe0, 0x00, 0x00,                # Z.re=7.0
            0x41, 0x00, 0x00, 0x00,                # Z.im=8.0
        ])
        actual_row0 = data[tail_start:tail_start + naxis1]
        print()
        print(f"expected row 0 (hex): {' '.join(f'{b:02x}' for b in expected_row0)}")
        print(f"actual   row 0 (hex): {' '.join(f'{b:02x}' for b in actual_row0)}")
        if actual_row0 == expected_row0:
            print("MATCH: on-disk bytes exactly match expected layout.")
        else:
            print("MISMATCH: at offset:", next(
                (k for k, (a, b) in enumerate(zip(actual_row0, expected_row0)) if a != b),
                "no length diff"
            ))
            print(f"actual length={len(actual_row0)}, expected length={len(expected_row0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
