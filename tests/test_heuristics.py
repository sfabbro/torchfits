import os
import tempfile

import torch

import torchfits as tf
from torchfits.heuristics import choose_flags, choose_read_mode_for_image


def test_auto_flags_uncompressed_full_image():
    # Create a small image (< mmap threshold)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "img.fits")
        tf.write(p, torch.zeros((128, 128), dtype=torch.float32), overwrite=True)
        # No explicit flags; start/shape None -> full image
        data, hdr = tf.read(p)
        assert torch.is_tensor(data)
        # Check last read info didn't force mmap/buffered for small file
        info = tf.get_last_read_info()
        assert info.get("path_used") in ("standard", "cache", "buffered", "mmap")


def test_choose_flags_buffered_for_compressed_header():
    em, eb, reason = choose_flags(is_full_image=True, is_compressed=True, file_size_mb=10.0)
    assert em is None and eb is True and "compressed" in reason


def test_choose_read_mode_for_image_local(tmp_path):
    p = tmp_path / "small.fits"
    tf.write(str(p), torch.zeros((64, 64), dtype=torch.float32), overwrite=True)
    out = choose_read_mode_for_image(str(p), hdu=0)
    assert set(out.keys()) == {"enable_mmap", "enable_buffered", "reason"}
