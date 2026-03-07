import pytest
import torch
import unittest.mock
import sys
sys.modules['torchfits.cpp'] = unittest.mock.MagicMock()

try:
    from torch_frame import stype
    from torch_frame.data import MultiNestedTensor
    from torch_frame import TensorFrame
except Exception:
    stype = None

from torchfits.frame import write_tensor_frame


def test_write_tensor_frame_stypes():
    if stype is None:
        pytest.skip("torch_frame not installed")

    feat_dict = {}
    col_names_dict = {}

    # 1. 2D types
    feat_dict[stype.numerical] = torch.tensor([[1.0], [2.0], [3.0]])
    col_names_dict[stype.numerical] = ["num"]

    feat_dict[stype.categorical] = torch.tensor([[0], [1], [2]])
    col_names_dict[stype.categorical] = ["cat"]

    if hasattr(stype, "timestamp"):
        feat_dict[stype.timestamp] = torch.tensor([[100], [200], [300]])
        col_names_dict[stype.timestamp] = ["time"]

    # 2. 3D types
    if hasattr(stype, "embedding"):
        feat_dict[stype.embedding] = torch.rand(3, 1, 5)
        col_names_dict[stype.embedding] = ["emb"]

    if hasattr(stype, "text_embedded"):
        feat_dict[stype.text_embedded] = torch.rand(3, 1, 8)
        col_names_dict[stype.text_embedded] = ["text_emb"]

    if hasattr(stype, "image_embedded"):
        feat_dict[stype.image_embedded] = torch.rand(3, 1, 16)
        col_names_dict[stype.image_embedded] = ["img_emb"]

    # 3. MultiNestedTensor types
    if hasattr(stype, "multicategorical"):
        feat_dict[stype.multicategorical] = MultiNestedTensor(
            num_rows=3,
            num_cols=1,
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            offset=torch.tensor([0, 1, 3, 6])
        )
        col_names_dict[stype.multicategorical] = ["multicat"]

    if hasattr(stype, "sequence_numerical"):
        feat_dict[stype.sequence_numerical] = MultiNestedTensor(
            num_rows=3,
            num_cols=1,
            values=torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
            offset=torch.tensor([0, 2, 4, 6])
        )
        col_names_dict[stype.sequence_numerical] = ["seq"]

    tf = TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)

    with unittest.mock.patch('torchfits.write') as mock_write:
        write_tensor_frame("dummy.fits", tf, overwrite=True)

        # Verify the data dict passed to torchfits.write
        args, kwargs = mock_write.call_args
        assert args[0] == "dummy.fits"
        data = args[1]

        # Check shapes
        assert data["num"].shape == (3,)
        assert data["cat"].shape == (3,)
        if hasattr(stype, "timestamp"):
            assert data["time"].shape == (3,)
        if hasattr(stype, "embedding"):
            assert data["emb"].shape == (3, 5)
        if hasattr(stype, "text_embedded"):
            assert data["text_emb"].shape == (3, 8)
        if hasattr(stype, "image_embedded"):
            assert data["img_emb"].shape == (3, 16)

        if hasattr(stype, "multicategorical"):
            assert isinstance(data["multicat"], list)
            assert len(data["multicat"]) == 3
            assert torch.equal(data["multicat"][0], torch.tensor([1]))
            assert torch.equal(data["multicat"][1], torch.tensor([2, 3]))
            assert torch.equal(data["multicat"][2], torch.tensor([4, 5, 6]))

        if hasattr(stype, "sequence_numerical"):
            assert isinstance(data["seq"], list)
            assert len(data["seq"]) == 3
            assert torch.allclose(data["seq"][0], torch.tensor([1.1, 2.2]))
            assert torch.allclose(data["seq"][1], torch.tensor([3.3, 4.4]))
            assert torch.allclose(data["seq"][2], torch.tensor([5.5, 6.6]))
