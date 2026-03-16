import torch
from torch_frame import TensorFrame, stype
from torch_frame.data import MultiNestedTensor

feat_dict = {}
col_names_dict = {}

feat_dict[stype.sequence_numerical] = MultiNestedTensor(
    num_rows=10,
    num_cols=1,
    values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    offset=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
)
col_names_dict[stype.sequence_numerical] = ["seq"]

try:
    tf = TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)
    print("TF sequence_numerical working")
except Exception as e:
    print(e)
