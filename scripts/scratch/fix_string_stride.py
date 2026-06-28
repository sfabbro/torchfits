"""One-shot patch: make update_rows_mmap's STRING and BIT cases
stride-aware so nanobind DLPack views (which may expose a (N, W) uint8
ndarray as a strided view of NumPy / UCS-4 memory) are read correctly.

Run from the activated pixi env:
    pixi run -e bench python scripts/fix_string_stride.py
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/torchfits/cpp_src/table.cpp")

PATCHES = [
    # --- STRING CASE: use tensor.stride() to handle DLPack strided views.
    (
        re.compile(
            r"                        case FITSColumnType::STRING: \{\n"
            r"                            if \!\(dt\.code == \(uint8_t\)nb::dlpack::dtype_code::UInt && dt\.bits == 8\) \{\n"
            r"                                munmap\(map_ptr, sb\.st_size\);\n"
            r"                                close\(fd\);\n"
            r"                                throw std::runtime_error\(\"update_rows mmap dtype mismatch for \" \+ name\);\n"
            r"                            \}\n"
            r"                            // STRING: dest stride is 1 byte\. Pad trailing bytes with\n"
            r"                            // ASCII spaces when the user-provided width is shorter\n"
            r"                            // than the FITS column width\.\n"
            r"                            if \(j < user_repeat\) \{\n"
            r"                                dest\[0\] = src_u8\[i \* user_repeat \+ j\];\n"
            r"                            \} else \{\n"
            r"                                dest\[0\] = 0x20; // ASCII space; matches FITS CHAR convention\n"
            r"                            \}\n"
            r"                            break;\n"
            r"                        \}"
        ),
        (
            "                        case FITSColumnType::STRING: {\n"
            "                            if (!(dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8)) {\n"
            "                                munmap(map_ptr, sb.st_size);\n"
            "                                close(fd);\n"
            "                                throw std::runtime_error(\"update_rows mmap dtype mismatch for \" + name);\n"
            "                            }\n"
            "                            // STRING: dest stride is 1 byte. Pad trailing bytes with\n"
            "                            // ASCII spaces when the user-provided width is shorter\n"
            "                            // than the FITS column width. Read src via tensor.stride()\n"
            "                            // so nanobind DLPack strided views (e.g. S8 over UCS-4)\n"
            "                            // land the right byte on disk.\n"
            "                            if (j < user_repeat) {\n"
            "                                long byte_offset = (ndim == 2)\n"
            "                                    ? i * tensor.stride(0) + j * tensor.stride(1)\n"
            "                                    : i * tensor.stride(0) + j;\n"
            "                                dest[0] = src_u8[byte_offset];\n"
            "                            } else {\n"
            "                                dest[0] = 0x20; // ASCII space; matches FITS CHAR convention\n"
            "                            }\n"
            "                            break;\n"
            "                        }"
        ),
    ),
    # --- BIT CASE: the same strided-view risk applies because src_u8 is used
    # to source individual bits from a (rows, repeat) uint8 tensor.
    (
        re.compile(
            r"                        case FITSColumnType::BIT: \{\n"
            r"                            // Extract a bool from a packed BIT column \(MSB-first\)\.\n"
            r"                            bool val = false;\n"
            r"                            if \(dt\.code == \(uint8_t\)nb::dlpack::dtype_code::Bool && dt\.bits == 8\) \{\n"
            r"                                val = src_bool\[idx\];\n"
            r"                            \} else if \(dt\.code == \(uint8_t\)nb::dlpack::dtype_code::UInt && dt\.bits == 8\) \{\n"
            r"                                val = src_u8\[idx\] != 0;\n"
            r"                            \} else \{\n"
            r"                                munmap\(map_ptr, sb\.st_size\);\n"
            r"                                close\(fd\);\n"
            r"                                throw std::runtime_error\(\"update_rows mmap dtype mismatch for \" \+ name\);\n"
            r"                            \}\n"
            r"                            // FITS BIT columns are MSB-first within each byte\.\n"
            r"                            // The default dest stride \(j \* col->width\) is meaningless\n"
            r"                            // for BIT \(col->width == 1 but storage is bit-packed\), so\n"
            r"                            // we operate directly on dest_row instead\.\n"
            r"                            uint8_t\* target_byte = dest_row \+ \(j / 8\);\n"
            r"                            uint8_t bit_position = static_cast<uint8_t>\(7 - \(j % 8\)\);\n"
            r"                            if \(j % 8 == 0\) \{\n"
            r"                                \*target_byte = 0;\n"
            r"                            \}\n"
            r"                            if \(val\) \{\n"
            r"                                \*target_byte \|= static_cast<uint8_t>\(1U << bit_position\);\n"
            r"                            \}\n"
            r"                            break;\n"
            r"                        \}"
        ),
        (
            "                        case FITSColumnType::BIT: {\n"
            "                            // Extract a bool from a packed BIT column (MSB-first).\n"
            "                            long byte_offset = (ndim == 2)\n"
            "                                ? i * tensor.stride(0) + j * tensor.stride(1)\n"
            "                                : i * tensor.stride(0) + j;\n"
            "                            bool val = false;\n"
            "                            if (dt.code == (uint8_t)nb::dlpack::dtype_code::Bool && dt.bits == 8) {\n"
            "                                if (ndim == 2) {\n"
            "                                    val = *(reinterpret_cast<const bool*>(\n"
            "                                        static_cast<const uint8_t*>(tensor.data()) + byte_offset\n"
            "                                    ));\n"
            "                                } else {\n"
            "                                    val = src_bool[byte_offset];\n"
            "                                }\n"
            "                            } else if (dt.code == (uint8_t)nb::dlpack::dtype_code::UInt && dt.bits == 8) {\n"
            "                                val = src_u8[byte_offset] != 0;\n"
            "                            } else {\n"
            "                                munmap(map_ptr, sb.st_size);\n"
            "                                close(fd);\n"
            "                                throw std::runtime_error(\"update_rows mmap dtype mismatch for \" + name);\n"
            "                            }\n"
            "                            // FITS BIT columns are MSB-first within each byte.\n"
            "                            // The default dest stride (j * col->width) is meaningless\n"
            "                            // for BIT (col->width == 1 but storage is bit-packed), so\n"
            "                            // we operate directly on dest_row instead.\n"
            "                            uint8_t* target_byte = dest_row + (j / 8);\n"
            "                            uint8_t bit_position = static_cast<uint8_t>(7 - (j % 8));\n"
            "                            if (j % 8 == 0) {\n"
            "                                *target_byte = 0;\n"
            "                            }\n"
            "                            if (val) {\n"
            "                                *target_byte |= static_cast<uint8_t>(1U << bit_position);\n"
            "                            }\n"
            "                            break;\n"
            "                        }"
        ),
    ),
]


def main() -> None:
    src = TARGET.read_text()
    for pattern, replacement in PATCHES:
        if pattern.search(src) is None:
            raise SystemExit(
                f"ANCHOR pattern not found in {TARGET}; "
                "either already patched, signalling syntax differs from "
                "expected, or file drifted."
            )
        if len(pattern.findall(src)) != 1:
            raise SystemExit(
                f"ANCHOR pattern matched multiple times; cannot safely patch."
            )
    for pattern, replacement in PATCHES:
        src = pattern.sub(replacement, src, count=1)
    TARGET.write_text(src)
    print(f"OK: {TARGET} patched ({len(PATCHES)} anchors found uniquely)")


if __name__ == "__main__":
    main()
