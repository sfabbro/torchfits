"""Fixed-width byte-column decode: correctness anchor for windowed views."""

import torch

from torchfits._string_decode import decode_byte_tensor


def test_decode_byte_tensor_full_column():
    t = torch.tensor(
        [[ord(c) for c in "STAR_A  "], [ord(c) for c in "QSO-B   "]],
        dtype=torch.uint8,
    )
    assert decode_byte_tensor(t) == ["STAR_A", "QSO-B"]


def test_decode_byte_tensor_offset_window():
    """A small row window of a large column decodes only its own rows.

    The window is contiguous with a large storage behind it; decoding must
    return exactly the window's rows (and must not copy the whole storage).
    """
    big = torch.zeros((1000, 20), dtype=torch.uint8)
    for row in (500, 501, 502):
        text = f"ROW{row}".ljust(20)
        big[row] = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
    window = big[500:503]
    assert window.is_contiguous()
    assert decode_byte_tensor(window) == ["ROW500", "ROW501", "ROW502"]


def test_decode_byte_tensor_strided_view_and_width_zero():
    t = torch.zeros((4, 6), dtype=torch.uint8)
    t[0, 0:3] = ord("A")
    t[2, 0:3] = ord("B")
    assert decode_byte_tensor(t[::2]) == ["AAA", "BBB"]
    assert decode_byte_tensor(torch.zeros((3, 0), dtype=torch.uint8)) == ["", "", ""]
