import os
import tempfile

import torch

import torchfits as tf


def test_update_header_idempotent_and_object_method():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "hdr_update.fits")
        img = torch.zeros((10, 10), dtype=torch.float32)
        tf.write(path, img, {"OBJECT": "Init"}, overwrite=True)

        # Path-level update
        updates = {"OBJECT": "Updated", "EXPTIME": "42", "OBSERVER": "Ada"}
        tf.update_header(path, updates, hdu=1)  # writer API expects 1-based
        hdr1 = tf.get_header(path, hdu=0)  # reader/header API uses 0-based
        assert hdr1.get("OBJECT") == "Updated"
        assert str(hdr1.get("EXPTIME")) == "42"
        assert hdr1.get("OBSERVER") == "Ada"

        # Idempotent second update with same values
        tf.update_header(path, updates, hdu=1)
        hdr2 = tf.get_header(path, hdu=0)
        assert hdr2.get("OBJECT") == "Updated"
        assert str(hdr2.get("EXPTIME")) == "42"
        assert hdr2.get("OBSERVER") == "Ada"

        # Object method update on HDU by name resolves to same HDU
        with tf.FITS(path) as f:
            h = f[0]
            h.update_header({"EXPTIME": "43"})
        hdr3 = tf.get_header(path, hdu=0)
        assert str(hdr3.get("EXPTIME")) == "43"
