import torch
import torchfits as tf


def test_varlen_multi_column_roundtrip(tmp_path):
    # Two ragged columns with equal row count but different per-row lengths
    col_a = [torch.arange(i, dtype=torch.float32) for i in range(5)]
    col_b = [torch.arange(2 * i, dtype=torch.float32) for i in range(5)]
    path = tmp_path / "vla_multi.fits"

    # Write via dict-of-list-of-tensors detection path
    tf.write(
        str(path),
        {"A": col_a, "B": col_b},
        {"EXTNAME": "VLA"},
        overwrite=True,
    )

    # Read back and verify shapes/types
    data, hdr = tf.read(str(path), hdu=1, format="tensor")
    assert isinstance(data, dict)
    assert "A" in data and "B" in data
    a = data["A"]; b = data["B"]
    assert isinstance(a, list) and isinstance(b, list)
    assert len(a) == len(col_a) == 5
    assert len(b) == len(col_b) == 5
    # Dtypes should be float64 per writer's base type
    assert all(t.dtype == torch.float64 for t in a)
    assert all(t.dtype == torch.float64 for t in b)
    # Content parity
    for i in range(5):
        assert torch.allclose(a[i].float(), col_a[i].float())
        assert torch.allclose(b[i].float(), col_b[i].float())
