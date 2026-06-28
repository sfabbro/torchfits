"""One-shot patch: route CHAR columns through a (num_rows, width) uint8 ndarray
inside `update_rows` so the mmap fast path sees an ndarray (not a list) and
the C++ STRING case can take over.

Run from the activated pixi env:
    python scripts/apply_routing_fix.py
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/torchfits/table.py")

# NOTE: project's `update_rows` uses 4-space indent at function-body level
# (not the 8-space I tried first). The anchor `    for name, value in
# rows.items():` only appears in `update_rows` -- `_normalize_mutation_rows`
# iterates `for col_name in columns:` instead, so uniqueness is preserved.
ANCHOR_LOOP = re.compile(
    r"    for name, value in rows\.items\(\):\n"
    r"        col_name = str\(name\)\n"
    r"        if col_name in vla_codes:\n"
    r"            values = _coerce_table_vla_values\(\n"
    r"                col_name, value, vla_codes\[col_name\], expected_rows=expected_rows\n"
    r"            \)\n"
    r"            if expected_rows is None:\n"
    r"                expected_rows = len\(values\)\n"
    r"            normalized\[col_name\] = values\n"
    r"        elif col_name in string_widths:\n"
    r"            values = _coerce_table_string_values\(\n"
    r"                col_name, value, expected_rows=expected_rows\n"
    r"            \)\n"
    r"            if expected_rows is None:\n"
    r"                expected_rows = len\(values\)\n"
    r"            normalized\[col_name\] = values\n"
)

REPLACEMENT = (
    "    for name, value in rows.items():\n"
    "        col_name = str(name)\n"
    "        if col_name in vla_codes:\n"
    "            values = _coerce_table_vla_values(\n"
    "                col_name, value, vla_codes[col_name], expected_rows=expected_rows\n"
    "            )\n"
    "            if expected_rows is None:\n"
    "                expected_rows = len(values)\n"
    "            normalized[col_name] = values\n"
    "        elif col_name in string_widths:\n"
    "            values = _coerce_table_string_values(\n"
    "                col_name, value, expected_rows=expected_rows\n"
    "            )\n"
    "            if expected_rows is None:\n"
    "                expected_rows = len(values)\n"
    "            # Materialise fixed-width CHAR columns as a (num_rows, width)\n"
    "            # uint8 ndarray so the mmap fast path\n"
    "            # (cpp.update_fits_table_rows_mmap) routes through the new\n"
    "            # STRING case in the C++ writer rather than the buffered\n"
    "            # fallback (which accepts list[str]). The C++ writer copies\n"
    "            # bytes left-to-right per row, so short user payloads are\n"
    "            # right-padded with ASCII spaces (0x20) before they hit\n"
    "            # disk. _coerce_table_string_values already truncates at\n"
    "            # the column width when user payloads are wider, so we\n"
    "            # only need to handle the short-payload case here.\n"
    "            import numpy as _np\n"
    "            width = string_widths[col_name]\n"
    "            arr = _np.full((expected_rows, width), 0x20, dtype=_np.uint8)\n"
    "            for i, s in enumerate(values):\n"
    "                if isinstance(s, (bytes, bytearray)):\n"
    "                    encoded = bytes(s)\n"
    "                elif isinstance(s, str):\n"
    "                    encoded = s.encode(\"ascii\", \"ignore\")\n"
    "                else:\n"
    "                    encoded = str(s).encode(\"ascii\", \"ignore\")\n"
    "                length = min(len(encoded), width)\n"
    "                if length > 0:\n"
    "                    arr[i, :length] = _np.frombuffer(\n"
    "                        encoded[:length], dtype=_np.uint8\n"
    "                    )\n"
    "            normalized[col_name] = arr\n"
)


def main() -> None:
    src = TARGET.read_text()
    match = ANCHOR_LOOP.search(src)
    if not match:
        raise SystemExit(
            f"ANCHOR_LOOP not found in {TARGET}; "
            "either already patched or file drifted."
        )
    if len(ANCHOR_LOOP.findall(src)) != 1:
        raise SystemExit(
            "ANCHOR_LOOP matched more than once; cannot safely apply patch."
        )
    new_src = src[: match.start()] + REPLACEMENT + src[match.end() :]
    TARGET.write_text(new_src)
    print(
        f"OK: {TARGET} patched (length {len(src)} -> {len(new_src)} chars); "
        f"ANCHOR_LOOP matched exactly once."
    )


if __name__ == "__main__":
    main()
