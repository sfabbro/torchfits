"""The x86 byte-swap shuffle tables, verified without x86 hardware.

``internal_utils.h``'s byte-swap helpers have three branches selected by
``#if``: NEON, AVX2 and SSSE3. The SSSE3 branch is compiled into every x86_64
build -- ``CMakeLists.txt`` adds ``-mssse3`` for ``x86_64`` -- so a wrong byte
in those mask tables would silently produce byte-reversed pixels for every x86
user of the mmap cutout path, which is exactly the failure the byte-wise
reference in ``tests/cpp/test_bswap_helpers.cpp`` exists to catch. That
reference can only catch it on a machine that *runs* the branch, so on arm64
(and in this repository's CI) the SSSE3 masks had never been executed or
checked by anything. The AVX2 branch is not enabled by any shipped build
(``__AVX2__`` is never defined; the build deliberately avoids ``-march=native``)
and is only reachable if a user passes ``-mavx2``, so it had no coverage at all.

So the tables are checked here, statically, against the instruction's own
semantics, which needs no particular CPU:

    _mm_shuffle_epi8(a, b):  dst[i] = 0  if b[i] & 0x80
                             dst[i] = a[b[i] & 0x0F]   otherwise

with indices relative to each 128-bit lane, and ``_mm*_set_epi8`` taking its
arguments from the highest byte to the lowest (so byte ``i`` is
``args[len - 1 - i]``). The permutation every branch must implement is a byte
reversal inside each ``w``-byte element, i.e. for byte ``i`` in a lane::

    element, offset = divmod(i, w)
    want            = element * w + (w - 1 - offset)

The loop bounds are checked in the same pass, because an iteration that does not
cover a whole number of elements would read or write a partial element.

Nothing is transcribed by hand: the masks, the bounds and the increments are
parsed out of the header, so this fails if the constants change, not only if
they are wrong. Load-bearing in seven distinct directions -- a single wrong
mask byte, a mask byte with the high bit set (which makes the instruction emit
a *zero* byte), a lane-crossing index, a short vector loop, a dropped BZERO
add, a mask transposed to the wrong element width, and a mask one byte short
of its register are each caught.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_HEADER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "torchfits"
    / "cpp_src"
    / "internal_utils.h"
)

# helper -> element width in bytes
_HELPERS = {
    "bswap16_copy": 2,
    "bswap16_copy_u16_offset": 2,
    "bswap32_copy": 4,
    "bswap32_copy_u32_offset": 4,
    "bswap64_copy": 8,
}
# the BZERO variants must add after shuffling; without the add every unsigned
# FITS value would come back wrong by 32768 / 2^31
_OFFSET_VARIANTS = ("bswap16_copy_u16_offset", "bswap32_copy_u32_offset")
_X86 = ("ssse3", "avx2")

_SET_EPI8 = re.compile(r"_mm(256)?_set_epi8\(([^)]*)\)", re.S)
_LOOP = re.compile(r"n\s*-\s*i\s*>=\s*(\d+)\s*;\s*i\s*\+=\s*(\d+)")
_IFDEF = re.compile(r"defined\(__(\w+)\)")


def _function_body(name: str) -> str:
    """The text between the braces of ``inline void <name>(...)``."""
    text = _HEADER.read_text()
    start = text.index(f"inline void {name}(")
    depth = 0
    i = text.index("{", start)
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
    raise AssertionError(f"{name}: unbalanced braces")


def _isa_blocks(body: str) -> dict[str, str]:
    """Split a body into its ``#if`` / ``#elif`` SIMD branches."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#if", "#elif")):
            found = _IFDEF.search(stripped)
            assert found, f"unrecognised preprocessor line: {stripped!r}"
            current = found.group(1).strip("_").lower()
            out.setdefault(current, [])
            continue
        if stripped.startswith(("#else", "#endif")):
            current = None
            continue
        if current is not None:
            out[current].append(line)
    return {key: "\n".join(value) for key, value in out.items()}


def _loop_bounds(isa: str, name: str) -> tuple[int, int]:
    block = _isa_blocks(_function_body(name))[isa]
    found = _LOOP.search(block)
    assert found, f"{name}/{isa}: no `n - i >= K; i += K` vector loop"
    return int(found.group(1)), int(found.group(2))


@pytest.mark.parametrize("name,width", sorted(_HELPERS.items()))
@pytest.mark.parametrize("isa", _X86)
def test_every_helper_keeps_both_x86_branches(name: str, width: int, isa: str) -> None:
    """A helper must not lose its x86 branch to an #if edit.

    Without this, a branch that disappeared would make the tests below skip
    silently rather than fail.
    """
    block = _isa_blocks(_function_body(name)).get(isa, "")
    assert "_shuffle_epi8" in block, f"{name}: the {isa.upper()} branch is gone"


@pytest.mark.parametrize("name,width", sorted(_HELPERS.items()))
@pytest.mark.parametrize("isa", _X86)
def test_x86_mask_implements_the_element_reversal(
    name: str, width: int, isa: str
) -> None:
    """Every mask byte must select the byte the reversal calls for."""
    block = _isa_blocks(_function_body(name))[isa]
    found = _SET_EPI8.search(block)
    assert found, f"{name}/{isa}: no _mm*_set_epi8 mask"
    wide = bool(found.group(1))
    lane_bytes = 32 if wide else 16
    args = [
        int(part)
        for part in found.group(2).replace("\n", " ").split(",")
        if part.strip()
    ]

    assert len(args) == lane_bytes, (
        f"{name}/{isa}: mask has {len(args)} bytes, the register holds {lane_bytes}"
    )
    mask = list(reversed(args))

    zeroed = [i for i, byte in enumerate(mask) if byte & 0x80]
    assert not zeroed, (
        f"{name}/{isa}: mask byte(s) {zeroed} have the high bit set, which makes "
        f"_mm_shuffle_epi8 write a ZERO byte there"
    )

    wrong = []
    for i, byte in enumerate(mask):
        in_lane = i % 16  # the instruction is per 128-bit lane
        element, offset = divmod(in_lane, width)
        want = element * width + (width - 1 - offset)
        if (byte & 0x0F) != want:
            wrong.append(f"byte {i}: selects src[{byte & 0x0F}], want src[{want}]")
    assert not wrong, f"{name}/{isa}: wrong permutation -- " + "; ".join(wrong)


@pytest.mark.parametrize("name,width", sorted(_HELPERS.items()))
@pytest.mark.parametrize("isa", _X86)
def test_x86_vector_loop_covers_whole_elements(name: str, width: int, isa: str) -> None:
    """A vector iteration must be a whole number of elements, and step by that."""
    need, step = _loop_bounds(isa, name)
    elems = (32 if isa == "avx2" else 16) // width
    assert need == step, f"{name}/{isa}: runs while n - i >= {need} but steps {step}"
    assert need == elems, (
        f"{name}/{isa}: {need} elements per iteration, but the vector holds "
        f"{elems} of {width} bytes"
    )


@pytest.mark.parametrize("name", _OFFSET_VARIANTS)
@pytest.mark.parametrize("isa", _X86)
def test_bzero_variants_add_after_the_shuffle(name: str, isa: str) -> None:
    """The unsigned FITS convention is an add, and it must survive the shuffle."""
    block = _isa_blocks(_function_body(name))[isa]
    bits = "16" if "16" in name else "32"
    assert re.search(rf"_mm(256)?_add_epi{bits}", block), (
        f"{name}/{isa}: no _mm*_add_epi{bits} after the shuffle "
        f"-- the BZERO offset would be dropped"
    )
