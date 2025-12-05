import os

import torch
from torch_frame import stype

from torchfits.frame import read_tensor_frame, to_tensor_frame, write_tensor_frame


def test_frame_integration():
    filename = "test_frame.fits"

    # Create sample data
    num_rows = 100
    data = {
        "col_num": torch.randn(num_rows),
        "col_cat": torch.randint(0, 5, (num_rows,), dtype=torch.int64),
    }

    # Convert to TensorFrame
    tf = to_tensor_frame(data)
    assert stype.numerical in tf.feat_dict
    assert stype.categorical in tf.feat_dict
    assert tf.feat_dict[stype.numerical].shape == (num_rows, 1)
    assert tf.feat_dict[stype.categorical].shape == (num_rows, 1)

    # Write to FITS
    write_tensor_frame(filename, tf, overwrite=True)
    assert os.path.exists(filename)
    # Read back
    tf_read = read_tensor_frame(filename)

    # Verify content
    # Note: FITS might change column order or names case (though torchfits tries to preserve)
    # torchfits write uses original names.

    # Check numerical
    assert stype.numerical in tf_read.feat_dict
    val_orig = tf.feat_dict[stype.numerical]
    val_read = tf_read.feat_dict[stype.numerical]
    # FITS float32 precision might cause small diffs if original was float64?
    # torch.randn is float32 by default in PyTorch? No, float32.
    assert torch.allclose(val_orig, val_read, atol=1e-5)

    # Check categorical
    assert stype.categorical in tf_read.feat_dict
    cat_orig = tf.feat_dict[stype.categorical]
    cat_read = tf_read.feat_dict[stype.categorical]
    assert torch.equal(cat_orig, cat_read)

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
